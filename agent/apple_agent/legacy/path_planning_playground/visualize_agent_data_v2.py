#!/usr/bin/env python3
"""
visualize_agent_data_v2.py

Same interactive HDF5 viewer as ``visualize_agent_data.py``, plus a live PyBullet
GUI. Default mode matches the HDF5 joint pose (**kinematic**). Optional
``--pb-mode physics`` replays recorded actions via ``env.step`` /
``step_joint_delta``, like ``replay_agent_data.py`` (much slower).

Arm pose in PyBullet (default ``--pb-mode kinematic``)
    Matches the HDF5 observation at frame ``fi`` by decoding
    ``obs/joint_angles[fi]`` and moving the simulated arm — **instant**, safe for
    large scrubs / long trajectories (avoids multi-second physics bursts that can
    wedge the GPU / reset the stack).

Arm pose via dynamics (optional ``--pb-mode physics``)
    Applies stored ``actions`` with ``env.step`` / ``step_joint_delta``, like
    ``replay_agent_data.py``. Steps are executed in **small chunks** on a separate
    timer so the GPU / window server keep breathing (avoids freezes that have
    been seen when hundreds of physics steps ran in one UI callback).

Playback: matplotlib presets extend to ``--max-ms`` (default 3 ms/frame). Actual
playback may still be limited by PyBullet unless you use ``kinematic`` mode.

Use ``[`` / ``]`` or the buttons to adjust.

Keyboard / buttons — same as v1 (see ``visualize_agent_data.py``).

Usage (from repo root, with ``PYTHONPATH`` including the repo):
────────────────────────────────────────────────────────────────
    PYTHONPATH=. python feature_apple_path_planning/playground/visualize_agent_data_v2.py

    PYTHONPATH=. python feature_apple_path_planning/playground/visualize_agent_data_v2.py \\
        --file output/agent_data/20260504_203032.hdf5 --traj 3

Optional:
    --slow-ms MS   Slow ``[`` preset (default 4000 ms per plot frame).
    --max-ms MS    Fast ``]`` preset floor (default 3 ms … 1 ms; higher = smoother UI).
    --pb-mode kinematic | physics    (default kinematic — fast / safe scrub).
    --pb-chunk N   Physics replay: env.steps per GUI tick.

Close the matplotlib window to exit (PyBullet disconnects on figure close).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

_FP_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _FP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec_dg = importlib.util.spec_from_file_location(
    "apple_data_generator_72_vizpb",
    _FP_ROOT / "run_apple_data_generator.py",
)
dg = importlib.util.module_from_spec(_spec_dg)
assert _spec_dg.loader is not None
_spec_dg.loader.exec_module(dg)

from apple_picking_env import ApplePickingEnv  # noqa: E402

# Load sibling viewer module without requiring package imports.
_spec_viz = importlib.util.spec_from_file_location(
    "visualize_agent_data_mod",
    Path(__file__).resolve().parent / "visualize_agent_data.py",
)
viz_mod = importlib.util.module_from_spec(_spec_viz)
assert _spec_viz.loader is not None
_spec_viz.loader.exec_module(viz_mod)


def decode_joint_angles(encoded: np.ndarray) -> np.ndarray:
    """Sin/cos packed joints → radians (same as replay_agent_data.py)."""
    n = encoded.shape[-1] // 2
    return np.arctan2(encoded[:n], encoded[n:]).astype(np.float64)


def _remove_body(pb_client, body_id: int | None) -> None:
    if body_id is None or body_id < 0:
        return
    try:
        pb_client.removeBody(body_id)
    except Exception:
        pass


def _place_markers(pb_client, apple_pos, goal_pos, marker_ids: dict) -> None:
    _remove_body(pb_client, marker_ids.get("apple"))
    _remove_body(pb_client, marker_ids.get("goal"))
    marker_ids["apple"] = dg.draw_debug_sphere(
        pb_client, list(apple_pos), 0.05, [0.9, 0.08, 0.08, 0.9]
    )
    marker_ids["goal"] = dg.draw_debug_sphere(
        pb_client, list(goal_pos), 0.025, [0.08, 0.92, 0.12, 0.92]
    )


class AgentDataViewerV2(viz_mod.AgentDataViewer):
    """
    Extends AgentDataViewer with PyBullet replay synced to the scrubbed frame index.
    """

    def __init__(
        self,
        hdf5_path: Path,
        start_traj: int = 0,
        *,
        slow_ms: float = 4000.0,
        fast_ms: float = 3.0,
        pb_mode: str = "kinematic",
        pb_chunk_size: int = 24,
    ):
        mode = pb_mode.strip().lower()
        if mode not in ("kinematic", "physics"):
            raise ValueError("pb_mode must be 'kinematic' or 'physics'")

        self._pb_marker_ids: dict = {}
        self.env: ApplePickingEnv | None = None
        self.pb_client = None
        self._pb_sim_frame: int = 0

        self._pb_mode = mode
        self._pb_chunk_size = max(1, int(pb_chunk_size))

        self._pb_sync_goal: int | None = None
        self._pb_catchup_timer = None
        self._pb_catchup_active = False

        super().__init__(hdf5_path, start_traj=start_traj)

        # Parent __init__ overwrites _speeds — rebuild wider range then pick a faster default.
        slow_ms = max(float(slow_ms), 50.0)
        fast_ms = max(min(float(fast_ms), slow_ms), 0.5)
        ratio = (slow_ms / fast_ms) ** (1.0 / 14.0)
        speeds = [
            max(int(round(slow_ms / (ratio ** i))), int(round(fast_ms)))
            for i in range(15)
        ]
        speeds[-1] = max(1, int(round(fast_ms)))
        self._speeds = sorted(set(max(1, int(x)) for x in speeds))

        # Default plot refresh: kinematic poses are cheap so we bias toward faster UI.
        default_interval = 6 if self._pb_mode == "kinematic" else 35
        nearest = min(
            range(len(self._speeds)),
            key=lambda i: abs(self._speeds[i] - default_interval),
        )
        self._speed_idx = min(nearest, len(self._speeds) - 1)
        self._apply_speed()

        self._pb_catchup_timer = self.fig.canvas.new_timer(interval=12)
        self._pb_catchup_timer.add_callback(self._pb_catchup_tick)
        # One shot per arm: reschedule manually so we don't stack callbacks during catch-up.
        try:
            self._pb_catchup_timer.single_shot = True
        except AttributeError:
            pass

        self.fig.canvas.mpl_connect("close_event", self._on_figure_close)

    # ── PyBullet lifecycle ──────────────────────────────────────────────────

    def _ensure_pb_env(self) -> None:
        if self.env is not None:
            return
        os.chdir(str(_REPO_ROOT))
        config = {
            **dg.CONFIG,
            "simulation_setup": {**dg.CONFIG["simulation_setup"], "renders": True},
        }
        self.env = ApplePickingEnv(config=config)
        self.pb_client = self.env.pb_client

    def _on_figure_close(self, _event=None) -> None:
        try:
            if self._pb_catchup_timer is not None:
                self._pb_catchup_timer.stop()
        except Exception:
            pass
        self._pb_catchup_active = False
        self._pb_sync_goal = None
        self._pb_marker_ids.clear()
        if self.pb_client is not None:
            try:
                self.pb_client.disconnect()
            except Exception:
                pass
        self.env = None
        self.pb_client = None

    def _pb_meta_vec(self, key: str, default, dtype=np.float64) -> np.ndarray:
        raw = self.traj["attrs"].get(key, default)
        return np.asarray(raw, dtype=dtype).reshape(3)

    def _pb_action_format(self) -> str:
        return str(self.traj["attrs"].get("action_format", "ee_velocity_local_scaled"))

    def _pb_step_recorded_action(self, step_k: int) -> None:
        assert self.env is not None
        action = self.traj["actions"][step_k].astype(np.float32)
        fmt = self._pb_action_format()
        if fmt == "joint_delta_rad":
            self.env.step_joint_delta(action.astype(np.float64))
        else:
            self.env.step(action)

    def _pb_teleport_to_frame(self, fi: int) -> None:
        """Move the robot to pose stored at HDF5 observation index ``fi`` (no dynamics)."""
        assert self.env is not None
        fi = int(np.clip(fi, 0, self.n_frames - 1))
        q = decode_joint_angles(self.traj["obs"]["joint_angles"][fi])
        self.env.robot.set_joint_angles_no_collision(q.tolist())
        # Do not call stepSimulation here — replay bootstrap avoids extra steps so
        # tree contact impulses cannot throw the arm before rendering.
        self.env._get_obs()
        self._pb_sim_frame = fi

    def _pb_bootstrap_current_trajectory(self) -> None:
        """Match replay_agent_data._load_trajectory bootstrap for this traj."""
        self._ensure_pb_env()
        assert self.env is not None
        gd = self.traj
        apple_pos = self._pb_meta_vec("apple_center", [0, 0, 0])
        goal_pos = self._pb_meta_vec("goal_pos", [0, 0, 0])

        self.env.reset(seed=None, options={"target_apple": apple_pos.astype(np.float32)})
        self.env.reward_goal = goal_pos.astype(np.float32)

        q0 = decode_joint_angles(gd["obs"]["joint_angles"][0])
        self.env.robot.set_joint_angles_no_collision(q0.tolist())

        self.env.step_counter = 0
        self.env.is_goal_state = False
        self.env.sum_reward = 0.0
        self.env.init_pos_ee, self.env.init_or_ee = self.env.robot.get_current_pose(
            self.env.robot.tool0_link_idx
        )
        self.env._get_obs()

        _place_markers(self.pb_client, apple_pos, goal_pos, self._pb_marker_ids)

        try:
            if self._pb_catchup_timer is not None:
                self._pb_catchup_timer.stop()
        except Exception:
            pass
        self._pb_catchup_active = False

        if self._pb_mode == "kinematic":
            self._pb_teleport_to_frame(0)
        else:
            self._pb_sim_frame = 0

    def _pb_clear_markers(self) -> None:
        if self.pb_client is None:
            return
        _remove_body(self.pb_client, self._pb_marker_ids.get("apple"))
        _remove_body(self.pb_client, self._pb_marker_ids.get("goal"))
        self._pb_marker_ids.clear()

    def _pb_sync_to_frame_request(self, fi: int) -> None:
        """Update PyBullet to match displayed frame ``fi`` (called from redraw)."""
        if self.env is None:
            return
        fi = int(np.clip(fi, 0, self.n_frames - 1))

        if self._pb_mode == "kinematic":
            if fi != self._pb_sim_frame:
                self._pb_teleport_to_frame(fi)
            return

        try:
            if self._pb_catchup_timer is not None:
                self._pb_catchup_timer.stop()
        except Exception:
            pass
        self._pb_catchup_active = False

        if fi < self._pb_sim_frame:
            self._pb_bootstrap_current_trajectory()

        self._pb_sync_goal = fi

        # Already aligned and no physics burst needed.
        if fi == self._pb_sim_frame:
            return

        self._pb_catchup_pulse()

    def _pb_catchup_tick(self):
        """Timer callback — continue physics catch-up toward ``_pb_sync_goal``."""
        if self.env is None or self._pb_sync_goal is None:
            self._pb_catchup_active = False
            return

        fi = int(np.clip(self._pb_sync_goal, 0, self.n_frames - 1))

        # User jumped backward while chasing an old goal
        if fi < self._pb_sim_frame:
            self._pb_bootstrap_current_trajectory()

        if self._pb_sim_frame >= fi:
            try:
                if self._pb_catchup_timer is not None:
                    self._pb_catchup_timer.stop()
            except Exception:
                pass
            self._pb_catchup_active = False
            return

        self._pb_catchup_pulse()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _pb_catchup_pulse(self):
        """Run up to `_pb_chunk_size` physics steps, then reschedule if needed."""
        if self.env is None or self._pb_sync_goal is None:
            return

        fi = int(np.clip(self._pb_sync_goal, 0, self.n_frames - 1))

        if fi < self._pb_sim_frame:
            self._pb_bootstrap_current_trajectory()

        n_total = fi - self._pb_sim_frame
        if n_total <= 0:
            try:
                if self._pb_catchup_timer is not None:
                    self._pb_catchup_timer.stop()
            except Exception:
                pass
            self._pb_catchup_active = False
            return

        n_do = min(n_total, self._pb_chunk_size)
        deadline = time.perf_counter() + 0.012  # also cap wall time slice per pulse

        for _ in range(n_do):
            if time.perf_counter() >= deadline:
                break
            self._pb_step_recorded_action(self._pb_sim_frame)
            self._pb_sim_frame += 1
            if self._pb_sim_frame >= fi:
                break

            ctr = int(getattr(self.env, "step_counter", 0))
            mx = int(getattr(self.env, "max_steps", 10_000))
            if ctr >= mx:
                break

        stalled = int(getattr(self.env, "step_counter", 0)) >= int(
            getattr(self.env, "max_steps", 10_000)
        )

        if self._pb_sim_frame < fi and not stalled:
            self._pb_catchup_active = True
            try:
                if self._pb_catchup_timer is not None:
                    self._pb_catchup_timer.start()
            except Exception:
                pass
        else:
            try:
                if self._pb_catchup_timer is not None:
                    self._pb_catchup_timer.stop()
            except Exception:
                pass
            self._pb_catchup_active = False

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    # ── Overrides ────────────────────────────────────────────────────────────

    def _load_traj(self, idx: int):
        prev = self.traj_idx
        super()._load_traj(idx)
        if prev != self.traj_idx:
            self._ensure_pb_env()
            self._pb_clear_markers()
            self._pb_sync_goal = None
            try:
                if self._pb_catchup_timer is not None:
                    self._pb_catchup_timer.stop()
            except Exception:
                pass
            self._pb_catchup_active = False
            self._pb_bootstrap_current_trajectory()

    def _redraw(self):
        super()._redraw()
        try:
            self._pb_sync_to_frame_request(self.frame_idx)
        except Exception as exc:
            print(f"[visualize_agent_data_v2] PyBullet sync warning: {exc}")

    def _build_figure(self):
        super()._build_figure()
        mode_note = (
            "kinematic (HDF5 joint pose)" if self._pb_mode == "kinematic" else "physics (chunked env.step)"
        )
        self.fig.text(
            0.10,
            0.055,
            f"PyBullet: {mode_note}. Use --pb-mode physics for full dynamics replay (slower).",
            fontsize=7,
            color="#888888",
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--file", "-f", default=None,
        help="HDF5 path (default: newest under output/agent_data/).",
    )
    p.add_argument(
        "--traj", "-t", type=int, default=0,
        help="Starting trajectory index (0-based).",
    )
    p.add_argument(
        "--slow-ms", type=float, default=4000.0,
        help="Slowest matplotlib preset ([ key), milliseconds per frame.",
    )
    p.add_argument(
        "--max-ms", type=float, default=3.0,
        help="Fastest matplotlib preset (] key): ms per plotted frame floor (≥1 ms).",
    )
    p.add_argument(
        "--pb-mode",
        choices=("kinematic", "physics"),
        default="kinematic",
        help=(
            "kinematic: set arm from HDF5 joints (fast, safe scrub). "
            "physics: replay actions via env.step in small slices (slow, stresses GPU)."
        ),
    )
    p.add_argument(
        "--pb-chunk",
        type=int,
        default=24,
        help="Physics replay: env.step batch size per GUI timer tick (~16 ms).",
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
            pass

    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    os.chdir(_REPO_ROOT)
    viewer = AgentDataViewerV2(
        hdf5_path,
        start_traj=args.traj,
        slow_ms=args.slow_ms,
        fast_ms=args.max_ms,
        pb_mode=args.pb_mode,
        pb_chunk_size=args.pb_chunk,
    )
    viewer.show()


if __name__ == "__main__":
    main()
