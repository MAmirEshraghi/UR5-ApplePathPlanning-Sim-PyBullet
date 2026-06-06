#!/usr/bin/env python3
"""
Interactive viewer for agent-data HDF5 files produced by
generate_agent_data_from_waypoints.py.

Layout
------
Row 0 — RGB + mask overlay  |  Next-obs RGB  |  Optical flow  |  Scalar readout
Row 1 — Reward curve (full trajectory)     |  EE-distance-to-goal curve
Row 2 — Action components curve            |  Joint angles curve
Row 3 — Action bar (current frame)         |  Trajectory summary

Keyboard
--------
← / → (or A / D)  : prev / next frame
P / N              : prev / next trajectory
Home / End         : first / last frame
Space              : toggle auto-play (advances frames automatically)
[ / ]              : slower / faster playback speed
L                  : toggle loop (loop trajectory vs. auto-advance to next)
Q / Escape         : quit

Buttons (bottom bar)
--------------------
▶ Play / ⏸ Pause   ½× Slower   2× Faster   ↺ Loop
◀ Traj   Traj ▶   ◀ Frame   Frame ▶

Usage
-----
python playground/visualize_agent_data.py
python playground/visualize_agent_data.py --file output/agent_data/20260504_203032.hdf5
python playground/visualize_agent_data.py --file output/agent_data/20260504_203032.hdf5 --traj 3
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Image / flow utilities
# ─────────────────────────────────────────────────────────────────────────────

def _chw_to_hwc(arr: np.ndarray) -> np.ndarray:
    """(C, H, W) uint8 or float → (H, W, C) uint8."""
    img = arr.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        if img.max() > 1.0:
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    (2, H, W) optical flow → (H, W, 3) uint8 HSV colour-wheel image.
    Hue = direction, Value = magnitude (brightness = speed).
    """
    from matplotlib.colors import hsv_to_rgb as _h2r
    u = flow[0].astype(np.float32)
    v = flow[1].astype(np.float32)
    mag = np.sqrt(u ** 2 + v ** 2)
    max_mag = mag.max()
    if max_mag < 1e-6:
        # All-zero flow — return grey image
        return np.full((*mag.shape, 3), 128, dtype=np.uint8)
    ang = (np.arctan2(v, u) + np.pi) / (2.0 * np.pi)   # hue [0, 1]
    hsv = np.stack([ang, np.ones_like(ang), mag / max_mag], axis=-1)
    return (_h2r(hsv) * 255).astype(np.uint8)


def _mask_overlay(rgb: np.ndarray, mask: np.ndarray,
                  color=(220, 60, 60), alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary mask on an RGB uint8 image."""
    out = rgb.astype(np.float32)
    m = (mask > 0.5)
    for c, col in enumerate(color):
        out[..., c] = np.where(m, out[..., c] * (1.0 - alpha) + col * alpha, out[..., c])
    return out.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sort_key(k: str):
    try:
        return (0, float(k))
    except ValueError:
        return (1, k)


def _load_traj(f: h5py.File, key: str) -> dict:
    """Load one group from an open HDF5 file into plain numpy arrays."""
    grp = f[key]
    return {
        "key":      key,
        "actions":  grp["actions"][:],          # (N, 6)
        "rewards":  grp["rewards"][:],           # (N,)
        "dones":    grp["dones"][:],             # (N,)
        "obs":      {k: grp["observations"][k][:] for k in grp["observations"]},
        "nobs":     {k: grp["next_observations"][k][:] for k in grp["next_observations"]},
        "attrs":    dict(grp.attrs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Viewer
# ─────────────────────────────────────────────────────────────────────────────

class AgentDataViewer:
    _ACTION_LABELS = ["vx", "vy", "vz", "wx", "wy", "wz"]
    _ACTION_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

    def __init__(self, hdf5_path: Path, start_traj: int = 0):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.widgets import Button, Slider

        self.path = hdf5_path
        self._plt = plt
        self._cache: dict[str, dict] = {}

        with h5py.File(hdf5_path, "r") as f:
            self.keys = sorted(f.keys(), key=_sort_key)

        if not self.keys:
            print("ERROR: No trajectories found in file.")
            sys.exit(1)

        print(f"Loaded {len(self.keys)} trajectories from {hdf5_path.name}")

        self.traj_idx = -1   # force load
        self.frame_idx = 0

        # Auto-play state
        self._playing = False
        self._play_interval = 200   # ms per frame  (≈ 5 fps default)
        self._loop = True           # loop at end vs. advance to next trajectory
        # Slowest → fastest: 0.5, 1, 2, 5, 10, 20, 40 fps
        _SPEEDS = [2000, 1000, 500, 200, 100, 50, 25]
        self._speeds = _SPEEDS
        self._speed_idx = _SPEEDS.index(self._play_interval)

        self._build_figure()
        self._load_traj(start_traj)
        self._redraw()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        from matplotlib.gridspec import GridSpec
        from matplotlib.widgets import Button, Slider

        fig = self._plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor("#1a1a2e")
        self.fig = fig

        gs = GridSpec(
            4, 4,
            figure=fig,
            top=0.93, bottom=0.20,
            hspace=0.55, wspace=0.35,
        )

        # Row 0 — images
        self.ax_rgb   = fig.add_subplot(gs[0, 0])
        self.ax_nrgb  = fig.add_subplot(gs[0, 1])
        self.ax_flow  = fig.add_subplot(gs[0, 2])
        self.ax_scalars = fig.add_subplot(gs[0, 3])

        # Row 1 — reward + ee-dist
        self.ax_reward = fig.add_subplot(gs[1, :2])
        self.ax_eedist = fig.add_subplot(gs[1, 2:])

        # Row 2 — action curves + joints
        self.ax_actcurve = fig.add_subplot(gs[2, :2])
        self.ax_joints   = fig.add_subplot(gs[2, 2:])

        # Row 3 — action bar + traj summary
        self.ax_actbar  = fig.add_subplot(gs[3, :2])
        self.ax_summary = fig.add_subplot(gs[3, 2:])

        for ax in (self.ax_rgb, self.ax_nrgb, self.ax_flow,
                   self.ax_scalars, self.ax_summary):
            ax.axis("off")

        _style_axes(fig,
                    self.ax_reward, self.ax_eedist,
                    self.ax_actcurve, self.ax_joints, self.ax_actbar)

        # ── frame slider ──────────────────────────────────────────────────────
        ax_sl = fig.add_axes([0.10, 0.155, 0.55, 0.025])
        ax_sl.set_facecolor("#2d2d44")
        self.sl_frame = Slider(ax_sl, "Frame", 0, 1, valinit=0, valstep=1,
                               color="#4a90d9", track_color="#2d2d44")
        self.sl_frame.label.set_color("white")
        self.sl_frame.valtext.set_color("white")
        self.sl_frame.on_changed(self._on_slider)

        # ── playback controls row ─────────────────────────────────────────────
        _BH = 0.038   # button height
        _BY = 0.108   # button y (control row)

        ax_play = fig.add_axes([0.10, _BY, 0.075, _BH])
        self.btn_play = Button(ax_play, "▶  Play", color="#1e6b3c", hovercolor="#2ecc71")
        self.btn_play.label.set_color("white"); self.btn_play.label.set_fontsize(9)
        self.btn_play.on_clicked(self._toggle_play)

        ax_slow = fig.add_axes([0.182, _BY, 0.055, _BH])
        btn_slow = Button(ax_slow, "½× Slow", color="#2d2d44", hovercolor="#4a90d9")
        btn_slow.label.set_color("white"); btn_slow.label.set_fontsize(8)
        btn_slow.on_clicked(self._slower)

        ax_fast = fig.add_axes([0.242, _BY, 0.055, _BH])
        btn_fast = Button(ax_fast, "2× Fast", color="#2d2d44", hovercolor="#4a90d9")
        btn_fast.label.set_color("white"); btn_fast.label.set_fontsize(8)
        btn_fast.on_clicked(self._faster)

        ax_loop = fig.add_axes([0.304, _BY, 0.075, _BH])
        self.btn_loop = Button(ax_loop, "↺ Loop: ON", color="#2d2d44", hovercolor="#4a90d9")
        self.btn_loop.label.set_color("#2ecc71"); self.btn_loop.label.set_fontsize(8)
        self.btn_loop.on_clicked(self._toggle_loop)

        self._speed_text = fig.text(
            0.39, _BY + _BH / 2, f"5.0 fps",
            color="#aaaaaa", fontsize=8, va="center",
        )

        # ── navigation buttons row (right side, same y) ───────────────────────
        nav_defs = [
            (0.67, "◀ Traj",  lambda _: (self._load_traj(self.traj_idx - 1), self._redraw())),
            (0.74, "Traj ▶",  lambda _: (self._load_traj(self.traj_idx + 1), self._redraw())),
            (0.82, "◀ Frame", lambda _: self._step_frame(-1)),
            (0.89, "Frame ▶", lambda _: self._step_frame(+1)),
        ]
        self._buttons = [self.btn_play, btn_slow, btn_fast, self.btn_loop]
        for x, label, cb in nav_defs:
            ax_b = fig.add_axes([x, _BY, 0.065, _BH])
            btn = Button(ax_b, label, color="#2d2d44", hovercolor="#4a90d9")
            btn.label.set_color("white"); btn.label.set_fontsize(8)
            btn.on_clicked(cb)
            self._buttons.append(btn)

        fig.text(
            0.10, 0.072,
            "Keys:  Space = play/pause   [ ] = speed   L = loop   "
            "← → / A D = frame   P N = traj   Home/End   Q = quit",
            fontsize=8, color="#666666",
        )

        fig.canvas.mpl_connect("key_press_event", self._on_key)

        # ── playback timer ────────────────────────────────────────────────────
        self._timer = fig.canvas.new_timer(interval=self._play_interval)
        self._timer.add_callback(self._tick)

    # ── data loading ──────────────────────────────────────────────────────────

    def _get_traj(self, idx: int) -> dict:
        key = self.keys[idx % len(self.keys)]
        if key not in self._cache:
            with h5py.File(self.path, "r") as f:
                self._cache[key] = _load_traj(f, key)
        return self._cache[key]

    def _load_traj(self, idx: int):
        idx = int(idx) % len(self.keys)
        if idx == self.traj_idx:
            return
        self.traj_idx = idx
        self.traj = self._get_traj(idx)
        n = len(self.traj["actions"])
        self.n_frames = n

        # Pre-compute derived time-series
        rd = self.traj["obs"]["relative_distance"]          # (N, 3)
        self.ee_dists    = np.linalg.norm(rd, axis=1)       # (N,)
        self.action_norms = np.linalg.norm(self.traj["actions"], axis=1)

        self.frame_idx = 0
        self.sl_frame.valmax = n - 1
        self.sl_frame.ax.set_xlim(0, n - 1)
        self.sl_frame.set_val(0)

    # ── navigation ────────────────────────────────────────────────────────────

    def _step_frame(self, delta: int):
        new = int(np.clip(self.frame_idx + delta, 0, self.n_frames - 1))
        if new != self.frame_idx:
            self.frame_idx = new
            self.sl_frame.set_val(new)   # triggers _on_slider → _redraw

    def _on_slider(self, val):
        self.frame_idx = int(val)
        self._redraw()

    def _on_key(self, event):
        k = event.key
        if k in ("right", "d"):       self._step_frame(+1)
        elif k in ("left", "a"):      self._step_frame(-1)
        elif k == "n":                self._load_traj(self.traj_idx + 1); self._redraw()
        elif k == "p":                self._load_traj(self.traj_idx - 1); self._redraw()
        elif k == "home":             self._step_frame(-self.n_frames)
        elif k == "end":              self._step_frame(+self.n_frames)
        elif k == " ":                self._toggle_play()
        elif k == "[":                self._slower()
        elif k == "]":                self._faster()
        elif k == "l":                self._toggle_loop()
        elif k in ("q", "escape"):    self._plt.close("all"); sys.exit(0)

    # ── auto-play ─────────────────────────────────────────────────────────────

    def _tick(self):
        """Called by the timer every `_play_interval` ms while playing."""
        if not self._playing:
            return
        if self.frame_idx >= self.n_frames - 1:
            if self._loop:
                # Restart from the beginning of this trajectory
                self.frame_idx = 0
                self.sl_frame.set_val(0)
            else:
                # Auto-advance to the next trajectory and keep playing
                next_idx = (self.traj_idx + 1) % len(self.keys)
                self._load_traj(next_idx)
                self._redraw()
        else:
            self._step_frame(+1)

    def _toggle_play(self, _=None):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self._playing = True
        self.btn_play.label.set_text("⏸  Pause")
        self.btn_play.ax.set_facecolor("#5b2020")
        self._timer.start()
        self.fig.canvas.draw_idle()

    def _stop_play(self):
        self._playing = False
        self.btn_play.label.set_text("▶  Play")
        self.btn_play.ax.set_facecolor("#1e6b3c")
        self._timer.stop()
        self.fig.canvas.draw_idle()

    def _slower(self, _=None):
        self._speed_idx = min(self._speed_idx + 1, len(self._speeds) - 1)
        self._apply_speed()

    def _faster(self, _=None):
        self._speed_idx = max(self._speed_idx - 1, 0)
        self._apply_speed()

    def _apply_speed(self):
        interval = self._speeds[self._speed_idx]
        self._play_interval = interval
        fps = 1000.0 / interval
        was_playing = self._playing
        if was_playing:
            self._timer.stop()
        self._timer.interval = interval
        if was_playing:
            self._timer.start()
        self._speed_text.set_text(f"{fps:.1f} fps")
        self.fig.canvas.draw_idle()
        print(f"Playback: {fps:.1f} fps  ({interval} ms/frame)")

    def _toggle_loop(self, _=None):
        self._loop = not self._loop
        if self._loop:
            self.btn_loop.label.set_text("↺ Loop: ON")
            self.btn_loop.label.set_color("#2ecc71")
        else:
            self.btn_loop.label.set_text("↺ Loop: OFF")
            self.btn_loop.label.set_color("#e67e22")
        self.fig.canvas.draw_idle()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        t  = self.traj
        fi = self.frame_idx
        xs = np.arange(self.n_frames)

        # ── Row 0: images ─────────────────────────────────────────────────────
        rgb_raw  = _chw_to_hwc(t["obs"]["rgb"][fi])          # (H, W, 3)
        mask     = t["obs"]["point_mask"][fi]                 # (H, W)
        nrgb_raw = _chw_to_hwc(t["nobs"]["rgb"][fi])         # (H, W, 3)
        flow     = t["obs"]["optical_flow"][fi]               # (2, 224, 224)

        rgb_overlaid = _mask_overlay(rgb_raw, mask)
        flow_img     = _flow_to_rgb(flow)

        for ax, img, title in [
            (self.ax_rgb,  rgb_overlaid, f"obs_rgb + apple mask  [frame {fi+1}/{self.n_frames}]"),
            (self.ax_nrgb, nrgb_raw,     "next_obs_rgb  (after action)"),
            (self.ax_flow, flow_img,     "optical flow  (HSV: dir=hue, speed=brightness)"),
        ]:
            ax.clear(); ax.axis("off")
            ax.imshow(img, interpolation="nearest")
            ax.set_title(title, fontsize=8, color="white", pad=3)

        # ── Row 0: scalars text ───────────────────────────────────────────────
        self.ax_scalars.clear(); self.ax_scalars.axis("off")
        ag  = t["obs"]["achieved_goal"][fi]
        dg  = t["obs"]["desired_goal"][fi]
        rd  = t["obs"]["relative_distance"][fi]
        act = t["actions"][fi]
        rew = float(t["rewards"][fi])
        done= bool(t["dones"][fi])

        scalar_text = "\n".join([
            f"{'Traj':>10}: {self.traj_idx+1} / {len(self.keys)}",
            f"{'Frame':>10}: {fi+1} / {self.n_frames}",
            "",
            f"{'Reward':>10}: {rew:+.5f}",
            f"{'Done':>10}: {done}",
            f"{'EE dist':>10}: {self.ee_dists[fi]:.4f} m",
            "",
            "EE pos (local):",
            f"  [{ag[0]:+.4f}, {ag[1]:+.4f}, {ag[2]:+.4f}]",
            "Goal pos (local):",
            f"  [{dg[0]:+.4f}, {dg[1]:+.4f}, {dg[2]:+.4f}]",
            "Rel dist (EE→goal):",
            f"  [{rd[0]:+.4f}, {rd[1]:+.4f}, {rd[2]:+.4f}]",
            "",
            "Action EE velocity:",
            f"  lin [{act[0]:+.4f}, {act[1]:+.4f}, {act[2]:+.4f}]",
            f"  ang [{act[3]:+.4f}, {act[4]:+.4f}, {act[5]:+.4f}]",
            f"  |a| {np.linalg.norm(act):.5f}",
        ])
        self.ax_scalars.text(
            0.05, 0.97, scalar_text,
            transform=self.ax_scalars.transAxes,
            va="top", ha="left", fontsize=8,
            fontfamily="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2d2d44", alpha=0.9),
        )
        self.ax_scalars.set_title("Step scalars", fontsize=8, color="white", pad=3)

        # ── Row 1: reward curve ───────────────────────────────────────────────
        ax = self.ax_reward
        ax.clear()
        ax.plot(xs, t["rewards"], color="#4a90d9", lw=1.3, label="reward")
        ax.fill_between(xs, t["rewards"], 0,
                        where=t["rewards"] >= 0, color="#2ecc71", alpha=0.15)
        ax.fill_between(xs, t["rewards"], 0,
                        where=t["rewards"] < 0,  color="#e74c3c", alpha=0.15)
        ax.axhline(0, color="#555555", lw=0.7, ls="--")
        ax.axvline(fi, color="#ff6b6b", lw=1.8, alpha=0.85, label=f"frame {fi+1}")
        done_steps = np.where(t["dones"])[0]
        for d in done_steps:
            ax.axvline(d, color="#2ecc71", lw=1.5, alpha=0.6, ls=":")
        ax.set_title("Reward per step", fontsize=9, color="white")
        ax.set_xlabel("step", color="#aaa"); ax.set_ylabel("reward", color="#aaa")
        ax.legend(fontsize=7, loc="upper left", facecolor="#2d2d44",
                  labelcolor="white", edgecolor="#555")
        ax.grid(True, alpha=0.2)

        # ── Row 1: EE-distance curve ──────────────────────────────────────────
        ax = self.ax_eedist
        ax.clear()
        ax.plot(xs, self.ee_dists, color="#2ecc71", lw=1.3, label="||EE − goal||")
        ax.fill_between(xs, self.ee_dists, alpha=0.1, color="#2ecc71")
        ax.axhline(0.05, color="#f39c12", lw=1.0, ls="--", alpha=0.7, label="goal tol 0.05m")
        ax.axvline(fi, color="#ff6b6b", lw=1.8, alpha=0.85)
        ax.set_title("EE distance to goal (m)", fontsize=9, color="white")
        ax.set_xlabel("step", color="#aaa"); ax.set_ylabel("metres", color="#aaa")
        ax.legend(fontsize=7, facecolor="#2d2d44",
                  labelcolor="white", edgecolor="#555")
        ax.grid(True, alpha=0.2)

        # ── Row 2: action component curves ───────────────────────────────────
        ax = self.ax_actcurve
        ax.clear()
        acts_all = t["actions"]   # (N, 6)
        for i, (lbl, col) in enumerate(zip(self._ACTION_LABELS, self._ACTION_COLORS)):
            ax.plot(xs, acts_all[:, i], color=col, lw=0.9, alpha=0.8, label=lbl)
        ax.plot(xs, self.action_norms, color="white", lw=1.4, alpha=0.6,
                ls="--", label="|a|")
        ax.axhline(0, color="#555555", lw=0.5)
        ax.axvline(fi, color="#ff6b6b", lw=1.8, alpha=0.85)
        ax.set_title("Action components over trajectory", fontsize=9, color="white")
        ax.set_xlabel("step", color="#aaa"); ax.set_ylabel("value", color="#aaa")
        ax.legend(fontsize=6, loc="upper right", facecolor="#2d2d44",
                  labelcolor="white", edgecolor="#555", ncol=4)
        ax.grid(True, alpha=0.2)

        # ── Row 2: joint angles ───────────────────────────────────────────────
        ax = self.ax_joints
        ax.clear()
        joints = t["obs"]["joint_angles"]    # (N, 12)
        n_j = joints.shape[1]
        cmap = self._plt.cm.tab20(np.linspace(0, 1, n_j))
        for j in range(n_j):
            ax.plot(xs, joints[:, j], color=cmap[j], lw=0.8, alpha=0.75, label=f"j{j}")
        ax.axvline(fi, color="#ff6b6b", lw=1.8, alpha=0.85)
        ax.set_title("Joint angles (normalised)", fontsize=9, color="white")
        ax.set_xlabel("step", color="#aaa"); ax.set_ylabel("value", color="#aaa")
        ax.legend(fontsize=5, loc="upper right", facecolor="#2d2d44",
                  labelcolor="white", edgecolor="#555", ncol=6)
        ax.grid(True, alpha=0.2)

        # ── Row 3: action bar (current frame) ─────────────────────────────────
        ax = self.ax_actbar
        ax.clear(); ax.axis("on")
        vals = t["actions"][fi]
        bar_cols = [c if v >= 0 else "#c0392b"
                    for v, c in zip(vals, self._ACTION_COLORS)]
        bars = ax.bar(self._ACTION_LABELS, vals, color=bar_cols,
                      edgecolor="#555555", linewidth=0.6)
        ax.axhline(0, color="#aaaaaa", lw=0.8)
        for bar, val in zip(bars, vals):
            offset = abs(val) * 0.04 + 0.0002
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (offset if val >= 0 else -offset),
                    f"{val:+.4f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7, color="white")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(f"Action at frame {fi+1}  (|a|={np.linalg.norm(vals):.5f})",
                     fontsize=9, color="white")
        ax.tick_params(colors="#aaa"); ax.set_ylabel("EE velocity", color="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.grid(True, alpha=0.2, axis="y")

        # ── Row 3: trajectory summary ─────────────────────────────────────────
        self.ax_summary.clear(); self.ax_summary.axis("off")
        attrs = t["attrs"]
        rews  = t["rewards"]
        cumret = np.cumsum(rews)

        summary = "\n".join([
            "─── Trajectory Summary ───────────",
            f"  Index       : {self.traj_idx+1} / {len(self.keys)}",
            f"  Transitions : {self.n_frames}",
            f"  Success     : {attrs.get('success', '?')}",
            f"  Goal achiev : {attrs.get('goal_achieved_count', '?')}×",
            f"  In-frame    : {attrs.get('count_in_frame', '?')} / {self.n_frames}",
            f"  Ctrl mode   : {str(attrs.get('controller_mode','?'))[:28]}",
            "",
            "─── Reward stats ─────────────────",
            f"  Total return: {rews.sum():.3f}",
            f"  Cumul@frame : {cumret[fi]:.3f}",
            f"  min / mean  : {rews.min():.4f} / {rews.mean():.4f}",
            f"  max         : {rews.max():.4f}",
            f"  pos-steps   : {(rews > 0).mean():.2f}",
            "",
            "─── Geometry ─────────────────────",
            f"  EE dist now : {self.ee_dists[fi]:.4f} m",
            f"  EE dist min : {self.ee_dists.min():.4f} m",
            f"  EE dist max : {self.ee_dists.max():.4f} m",
        ])
        self.ax_summary.text(
            0.04, 0.97, summary,
            transform=self.ax_summary.transAxes,
            va="top", ha="left", fontsize=8,
            fontfamily="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2d2d44", alpha=0.9),
        )
        self.ax_summary.set_title("Trajectory summary", fontsize=8, color="white", pad=3)

        # ── figure title ──────────────────────────────────────────────────────
        key_short = t["key"][:22]
        self.fig.suptitle(
            f"Agent Data Viewer  ·  {self.path.name}"
            f"  ·  traj {self.traj_idx+1}/{len(self.keys)}"
            f"  ·  key={key_short}  ·  frame {fi+1}/{self.n_frames}",
            fontsize=10, fontweight="bold", color="white",
        )
        self.fig.canvas.draw_idle()

    def show(self):
        self._plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Style helper
# ─────────────────────────────────────────────────────────────────────────────

def _style_axes(fig, *axes):
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--file", "-f", default=None,
        help="Path to agent-data HDF5 file. "
             "Defaults to the most-recently-modified file in output/agent_data/.",
    )
    p.add_argument(
        "--traj", "-t", type=int, default=0,
        help="Start at this trajectory index (0-based, default 0).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.file is None:
        agent_dir = _REPO_ROOT / "output" / "agent_data"
        files = sorted(agent_dir.glob("*.hdf5"), key=lambda p: p.stat().st_mtime)
        if not files:
            print(f"No HDF5 files found in {agent_dir}")
            sys.exit(1)
        hdf5_path = files[-1]
        print(f"Auto-selected: {hdf5_path}")
    else:
        hdf5_path = Path(args.file).resolve()

    if not hdf5_path.exists():
        print(f"File not found: {hdf5_path}")
        sys.exit(1)

    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            pass   # fall back to whatever is available

    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

    viewer = AgentDataViewer(hdf5_path, start_traj=args.traj)
    viewer.show()


if __name__ == "__main__":
    main()
