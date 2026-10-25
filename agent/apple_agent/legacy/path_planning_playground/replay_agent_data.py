#!/usr/bin/env python3
"""
replay_agent_data.py

Visualize recorded agent-data trajectories from output/agent_data/<run_id>.hdf5
in a PyBullet GUI.

REPLAY MODE (default): executes the stored 6-D EE-velocity *actions* through
env.step() so you see exactly what the arm would do during RL training — a
quality check for the generated data.  The arm is teleported to the recorded
start position, then each recorded action is applied through the full physics
pipeline (Jacobian DLS → joint velocities → PyBullet step).

ALL PyBullet calls run on the main thread (via tkinter root.after) to avoid
OpenGL/C-extension thread-safety issues that cause corrupted arm poses.

Usage (from repo root, venv activated):
──────────────────────────────────────
List groups in the file:

    PYTHONPATH=. python feature_apple_path_planning/playground/replay_agent_data.py \\
        --run-id 20260504_203032 --list-groups

Auto-play all trajectories (action-based):

    PYTHONPATH=. python feature_apple_path_planning/playground/replay_agent_data.py \\
        --run-id 20260504_203032

Replay a specific trajectory by index:

    PYTHONPATH=. python feature_apple_path_planning/playground/replay_agent_data.py \\
        --run-id 20260504_203032 --group-index 2

Step manually (press A or → in the control window):

    PYTHONPATH=. python feature_apple_path_planning/playground/replay_agent_data.py \\
        --run-id 20260504_203032 --manual-step

Show reward + action plots alongside each trajectory:

    PYTHONPATH=. python feature_apple_path_planning/playground/replay_agent_data.py \\
        --run-id 20260504_203032 --plot

Control window keys:
  ← / →     — previous / next trajectory (loads start pose)
  Enter     — run current trajectory (auto-play; or one step if --manual-step)
  P         — pause / resume playback
  N / B     — next / back one step (while paused)
  Q         — quit
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import importlib.util
import io
import logging
import os
import re
import sys
import threading
import tkinter as tk
import warnings
from pathlib import Path

import h5py
import numpy as np
from zenlog import log

# ── path setup ────────────────────────────────────────────────────────────────
_FP_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _FP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── load shared generator helpers (CONFIG, draw_debug_sphere, …) ──────────────
_spec = importlib.util.spec_from_file_location(
    "apple_data_generator_72",
    _FP_ROOT / "run_apple_data_generator.py",
)
dg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dg)

from feature_apple_path_planning.apple_picking_env import ApplePickingEnv


def _patch_tree_setup_from_metadata(config: dict, metadata: dict) -> None:
    """Align CONFIG tree_setup with recorded trajectory metadata (matches data generator)."""
    ts = config["tree_setup"]
    if metadata.get("tree_id") is not None:
        ts["tree_id"] = int(np.asarray(metadata["tree_id"]).reshape(-1)[0])
    if metadata.get("tree_scale") is not None:
        ts["scale"] = float(np.asarray(metadata["tree_scale"]).reshape(-1)[0])
    if metadata.get("tree_pos") is not None:
        ts["position"] = np.asarray(metadata["tree_pos"], dtype=np.float64).reshape(3)
    if metadata.get("tree_orientation") is not None:
        ts["orientation"] = np.asarray(metadata["tree_orientation"], dtype=np.float64).reshape(4)
    log.info(
        "Replay tree_setup: tree_id=%s scale=%.4f pos=%s",
        ts["tree_id"],
        ts["scale"],
        np.round(np.asarray(ts["position"]), 4).tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logging — OS-level pipe tee (captures PyBullet C-level output too)
# ─────────────────────────────────────────────────────────────────────────────

_ANSI_ESC_RE = re.compile(rb"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(data: bytes) -> bytes:
    return _ANSI_ESC_RE.sub(b"", data)


def _tee_reader(read_fd: int, term_fd: int, log_fd: int) -> None:
    """Drain read_fd; write raw bytes to terminal, ANSI-stripped bytes to log."""
    try:
        while True:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            try:
                os.write(term_fd, chunk)
            except OSError:
                pass
            try:
                os.write(log_fd, _strip_ansi(chunk))
            except OSError:
                pass
    except OSError:
        pass
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass


def setup_logging(debug: bool = False) -> Path:
    """
    Redirect stdout + stderr through an OS-level pipe so that every byte
    (including PyBullet b3Printf / OpenGL lines written at the C level) is
    captured in a timestamped log file under logs/data_generator_path_planner/.

    Identical mechanism to generate_agent_data_from_waypoints.py.
    """
    base = _REPO_ROOT / "logs" / dg.LOG_SESSION_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_replay_agent_data.log").resolve()

    # flush before touching fds
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

    orig_out_fd = os.dup(1)
    orig_err_fd = os.dup(2)

    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()

    os.dup2(out_w, 1)
    os.dup2(err_w, 2)
    os.close(out_w)
    os.close(err_w)

    out_thread = threading.Thread(
        target=_tee_reader, args=(out_r, orig_out_fd, log_fd),
        daemon=True, name="tee-stdout",
    )
    err_thread = threading.Thread(
        target=_tee_reader, args=(err_r, orig_err_fd, log_fd),
        daemon=True, name="tee-stderr",
    )
    out_thread.start()
    err_thread.start()

    sys.stdout = io.TextIOWrapper(
        io.FileIO(os.dup(1), mode="w", closefd=True),
        encoding="utf-8", errors="replace", line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        io.FileIO(os.dup(2), mode="w", closefd=True),
        encoding="utf-8", errors="replace", line_buffering=True,
    )
    warnings.simplefilter("always")

    def _restore(
        ofd=orig_out_fd, efd=orig_err_fd, lfd=log_fd,
        ot=out_thread, et=err_thread,
    ) -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(ofd, 1)
            os.dup2(efd, 2)
        except Exception:
            pass
        ot.join(timeout=0.5)
        et.join(timeout=0.5)
        for fd in (ofd, efd, lfd):
            try:
                os.close(fd)
            except OSError:
                pass

    atexit.register(_restore)

    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not debug:
        for h in zen_logger.handlers[:]:
            h.setLevel(logging.INFO)

    log.info("Replay log: %s  (debug=%s)", log_path, debug)
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (no PyBullet, safe to call anywhere)
# ─────────────────────────────────────────────────────────────────────────────

def _sort_keys(keys: list[str]) -> list[str]:
    def _k(s: str):
        try:
            return (0, float(s))
        except ValueError:
            return (1, s)
    return sorted(keys, key=_k)


def decode_joint_angles(encoded: np.ndarray) -> np.ndarray:
    """Recover joint angles from sin/cos encoding.

    env encodes as hstack([sin(q0..q_{n-1}), cos(q0..q_{n-1})]).
    """
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


def _hdf5_scalar_to_str(val) -> str:
    """Normalize HDF5 attribute values for console output."""
    if val is None:
        return "-"
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.shape == ():
            return _hdf5_scalar_to_str(val.item())
        return str(val)[:40]
    return str(val)


def list_groups(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        keys = _sort_keys(list(f.keys()))
        print(
            f"\n{'#':<4}  {'HDF5 group':<24}  {'Source wp key':<18}  "
            f"{'Steps':>5}  {'Success':>7}  Apple center"
        )
        print("-" * 96)
        for idx, k in enumerate(keys):
            g = f[k]
            steps = int(g.attrs.get("path_length", g["actions"].shape[0]))
            success = bool(g.attrs.get("success", False))
            apple = np.asarray(g.attrs.get("apple_center", [0, 0, 0]), dtype=float)
            src = _hdf5_scalar_to_str(g.attrs.get("source_group_key", ""))
            if len(src) > 16:
                src = src[:14] + "…"
            print(
                f"{idx:<4}  {k:<24}  {src:<18}  {steps:>5}  {str(success):>7}  "
                f"[{apple[0]:.3f}, {apple[1]:.3f}, {apple[2]:.3f}]"
            )
    print()


def _plot_trajectory(
    group_key: str, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray
) -> None:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plot.")
        return
    steps = np.arange(len(rewards))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Agent Data  |  group: {group_key}", fontsize=11)
    ax0 = axes[0]
    ax0.plot(steps, rewards, color="steelblue", linewidth=1.5, label="reward")
    for ds in np.where(dones)[0]:
        ax0.axvline(ds, color="red", linestyle="--", linewidth=1.0, alpha=0.8, label="done")
    ax0.set_ylabel("Reward")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax1 = axes[1]
    labels = ["vx", "vy", "vz", "wx", "wy", "wz"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    for j in range(actions.shape[1]):
        ax1.plot(steps, actions[:, j], linewidth=1.0, label=labels[j],
                 color=colors[j], alpha=0.85)
    ax1.set_ylabel("Action value")
    ax1.set_xlabel("Step")
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI arg parsing & HDF5 resolution
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay agent-data trajectories in PyBullet GUI."
    )
    p.add_argument("--run-id", default=None,
                   help="Run-id stem for output/agent_data/<run_id>.hdf5.")
    p.add_argument("--hdf5-path", default=None,
                   help="Direct path to an agent-data HDF5 file (overrides --run-id).")
    p.add_argument("--group-index", type=int, default=None,
                   help="0-based index of the trajectory to replay. All groups if omitted.")
    p.add_argument("--group-key", default=None,
                   help="Exact HDF5 group key (overrides --group-index).")
    p.add_argument("--list-groups", action="store_true",
                   help="Print group summary and exit (no GUI).")
    p.add_argument("--delay", type=float, default=0.05,
                   help="Seconds between steps in auto-play mode (default 0.05).")
    p.add_argument("--manual-step", action="store_true",
                   help="Press A / → / Enter in control window to advance one step.")
    p.add_argument("--plot", action="store_true",
                   help="Show reward + action plots for each replayed trajectory.")
    p.add_argument("--loop", action="store_true",
                   help="Loop back to the first trajectory after the last one.")
    p.add_argument("--debug-log", action="store_true",
                   help="Set log level to DEBUG (verbose per-step env output).")
    return p.parse_args()


def _resolve_hdf5(args: argparse.Namespace) -> Path:
    if args.hdf5_path is not None:
        p = Path(args.hdf5_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"HDF5 file not found: {p}")
        return p
    agent_data_dir = _REPO_ROOT / "output" / "agent_data"
    if not agent_data_dir.exists():
        raise FileNotFoundError(f"Agent data directory not found: {agent_data_dir}")
    if args.run_id is not None:
        p = (agent_data_dir / f"{args.run_id}.hdf5").resolve()
        if not p.exists():
            raise FileNotFoundError(f"No agent-data file for run_id={args.run_id!r}: {p}")
        return p
    candidates = sorted(agent_data_dir.glob("*.hdf5"), key=lambda x: x.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .hdf5 files in {agent_data_dir}")
    p = candidates[-1]
    log.info("No --run-id given; using latest file: %s", p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main — everything runs on the main thread via tkinter root.after
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.chdir(_REPO_ROOT)

    args = parse_args()

    # ── logging: set up before anything else so PyBullet output is captured ──
    if not args.list_groups:
        log_path = setup_logging(debug=args.debug_log)

    h5_path = _resolve_hdf5(args)
    log.info("Agent data file: %s", h5_path)

    if args.list_groups:
        list_groups(h5_path)
        return

    # ── load trajectories into memory ────────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        all_keys = _sort_keys(list(f.keys()))
        if args.group_key is not None:
            if args.group_key not in f:
                raise KeyError(f"Group key {args.group_key!r} not found in {h5_path}")
            selected_keys = [args.group_key]
        elif args.group_index is not None:
            if args.group_index < 0 or args.group_index >= len(all_keys):
                raise IndexError(
                    f"--group-index {args.group_index} out of range "
                    f"(file has {len(all_keys)} groups)."
                )
            selected_keys = [all_keys[args.group_index]]
        else:
            selected_keys = all_keys

        groups_data: list[dict] = []
        for k in selected_keys:
            g = f[k]
            groups_data.append({
                "key": k,
                "joint_angles": g["observations/joint_angles"][:],
                "final_joint_angles": g["next_observations/joint_angles"][-1],
                "actions": g["actions"][:],
                "rewards": g["rewards"][:],
                "dones": g["dones"][:],
                "metadata": {attr: np.asarray(v) for attr, v in g.attrs.items()},
            })

    log.info("Loaded %d trajectory group(s) from %s", len(groups_data), h5_path.name)

    # ── print summary ─────────────────────────────────────────────────────────
    planner_start = np.asarray(dg.CONFIG["robot_setup"]["start_joint_angles"])
    print(f"\nPlanner start (CONFIG): {np.round(planner_start, 4).tolist()}\n")
    print(
        f"{'#':<4}  {'HDF5 group':<22}  {'Src waypoint':<16}  "
        f"{'Steps':>5}  {'Success':>7}  start_err   Final joints (rad)"
    )
    print("-" * 108)
    for idx, gd in enumerate(groups_data):
        q0 = decode_joint_angles(gd["joint_angles"][0])
        qf = decode_joint_angles(gd["final_joint_angles"])
        err = float(np.max(np.abs(q0 - planner_start)))
        success = bool(gd["metadata"].get("success", False))
        src = _hdf5_scalar_to_str(gd["metadata"].get("source_group_key", ""))
        if len(src) > 14:
            src = src[:12] + "…"
        print(
            f"{idx:<4}  {str(gd['key']):<22}  {src:<16}  {len(gd['rewards']):>5}  {str(success):>7}  "
            f"{err:.5f}     [" + ", ".join(f"{v:+.3f}" for v in qf) + "]"
        )
    print()

    # ── build PyBullet env (GUI) ──────────────────────────────────────────────
    config = {**dg.CONFIG, "simulation_setup": {**dg.CONFIG["simulation_setup"], "renders": True}}
    _patch_tree_setup_from_metadata(config, groups_data[0]["metadata"])
    env = ApplePickingEnv(config=config)
    pb = env.pb_client
    log.info("PyBullet GUI environment ready.")

    delay_ms = max(10, int(args.delay * 1000))

    # ── tkinter control window ────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Replay Agent Data — Control")
    root.resizable(False, False)

    pad = {"padx": 14, "pady": 5}
    key_help = (
        "← / →  — prev / next trajectory\n"
        "Enter  — run trajectory (or one step with --manual-step)\n"
        "P      — pause / resume\n"
        "N / B  — next / back step (while paused)\n"
        "Q      — quit"
    )
    tk.Label(root, text=key_help, justify=tk.LEFT, font=("Courier", 11), **pad).pack()

    status_var = tk.StringVar(value="Initialising …")
    tk.Label(root, textvariable=status_var, justify=tk.LEFT,
             font=("Courier", 10), fg="navy", **pad).pack()
    step_var = tk.StringVar(value="")
    tk.Label(root, textvariable=step_var, justify=tk.LEFT,
             font=("Courier", 10), fg="#444", **pad).pack()

    # ── shared mutable state (all touched only on main thread) ───────────────
    state = {
        "traj_idx": 0,
        "step_idx": 0,
        "playing": False,
        "paused": True,
        "busy": False,
        "marker_ids": {},
    }

    # ── helpers that run on main thread ──────────────────────────────────────

    def _update_status() -> None:
        gd = groups_data[state["traj_idx"]]
        n = state["traj_idx"]
        success = bool(gd["metadata"].get("success", False))
        status_var.set(
            f"Trajectory {n + 1} / {len(groups_data)}   key: {gd['key']}\n"
            f"steps: {len(gd['rewards'])}   success: {success}"
        )

    def _update_step(si: int, n_steps: int, reward: float, done: bool) -> None:
        step_var.set(
            f"step {si + 1:>3} / {n_steps}   env_reward {reward:+.4f}   done={done}"
        )

    def _load_trajectory(traj_idx: int) -> None:
        """Reset env + teleport arm to recorded start. Must run on main thread.

        Mirrors bootstrap_rollout() in generate_agent_data_from_waypoints.py so
        the physics state (EE reference, step counter, etc.) is identical to what
        the data generator had at step 0.
        """
        gd = groups_data[traj_idx]
        apple_pos = np.asarray(gd["metadata"].get("apple_center", [0, 0, 0]), dtype=np.float64)
        goal_pos  = np.asarray(gd["metadata"].get("goal_pos",    [0, 0, 0]), dtype=np.float64)
        n_steps   = len(gd["rewards"])
        success   = bool(gd["metadata"].get("success", False))

        log.info(
            "─── traj %d/%d  key=%s  steps=%d  success=%s  action_format=%s ───",
            traj_idx + 1,
            len(groups_data),
            gd["key"],
            n_steps,
            success,
            str(gd["metadata"].get("action_format", "ee_velocity_local_scaled")),
        )

        # ── full bootstrap (mirrors generate_agent_data_from_waypoints.py) ──
        env.reset(seed=None, options={"target_apple": apple_pos.astype(np.float32)})
        env.reward_goal = goal_pos.astype(np.float32)
        env.tree.build_collision_kdtree()

        # teleport to the recorded start (same as refined_path[0] = planner_start)
        q0 = decode_joint_angles(gd["joint_angles"][0])
        env.robot.set_joint_angles_no_collision(q0.tolist())

        # sync env internals so reward / termination logic starts fresh
        env.step_counter = 0
        env.is_goal_state = False
        env.sum_reward    = 0.0
        env.init_pos_ee, env.init_or_ee = env.robot.get_current_pose(
            env.robot.tool0_link_idx
        )
        # NOTE: do NOT call pb.stepSimulation() here.
        # bootstrap_rollout() in the generator never steps after teleporting, so
        # an extra step here lets tree-collision impulses fling the arm to the
        # wrong position before the first action is applied (dist jumps from
        # ~0.13 m to ~0.54 m).  env.reset() already ran one internal step.

        # Sync observation_info['achieved_pos'] to the correct EE position at q0.
        # env.reset() internally runs stepSimulation() which may push the arm
        # (collision impulse) and stores that wrong position in observation_info.
        # Without this call, the FIRST env.step()'s reward uses the wrong prev_pos
        # (producing a fake +40-unit spike for traj 1).  Matches bootstrap_rollout().
        env._get_obs()

        start_err = float(np.max(np.abs(q0 - planner_start)))
        ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        dist0 = float(np.linalg.norm(ee_pos - goal_pos))
        log.info(
            "start_joints=%s  planner_start=%s  max_err=%.6f  dist_to_goal=%.4f m",
            np.round(q0, 4).tolist(), np.round(planner_start, 4).tolist(), start_err, dist0,
        )

        # markers
        _place_markers(pb, apple_pos, goal_pos, state["marker_ids"])

        # reset step counter for this trajectory
        state["step_idx"] = 0
        _update_status()
        _update_step(-1, n_steps, 0.0, False)

        if args.plot:
            _plot_trajectory(gd["key"], gd["actions"], gd["rewards"], gd["dones"])

    def _do_one_step() -> None:
        """Apply the recorded action for the current step via env.step().

        This executes the stored EE-velocity action through the full physics
        pipeline (Jacobian DLS → joint velocities → PyBullet), mirroring what
        the RL agent would do during training.  Logs both the env's live reward
        and the reward stored in the HDF5 file for comparison.
        Must run on main thread.
        """
        traj_idx = state["traj_idx"]
        si = state["step_idx"]
        gd = groups_data[traj_idx]
        n_steps = len(gd["rewards"])

        if si < n_steps:
            action = gd["actions"][si].astype(np.float32)
            meta = gd["metadata"]
            action_format = str(meta.get("action_format", "ee_velocity_local_scaled"))
            if action_format == "joint_delta_rad":
                obs, env_reward, terminated, truncated, info = env.step_joint_delta(
                    action.astype(np.float64)
                )
            else:
                obs, env_reward, terminated, truncated, info = env.step(action)

            stored_reward = float(gd["rewards"][si])
            stored_done   = bool(gd["dones"][si])

            ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
            dist = float(np.linalg.norm(ee_pos - env.reward_goal))

            _update_step(si, n_steps, env_reward, terminated or truncated)
            if args.debug_log:
                rd = float(abs(env_reward - stored_reward))
                if rd > 1e-6:
                    log.debug(
                        "step %s  |env_rew - hdf5_rew|=%.6g  (expected ~0; float noise / physics micro-drift)",
                        si + 1, rd,
                    )
            log.info(
                "step %3d/%3d  env_rew=%+.4f  hdf5_rew=%+.4f  "
                "dist=%.4f m  terminated=%s  truncated=%s  hdf5_done=%s  "
                "action=%s",
                si + 1, n_steps, env_reward, stored_reward,
                dist, terminated, truncated, stored_done,
                np.round(action, 3).tolist(),
            )
            state["step_idx"] += 1

            if terminated or truncated:
                log.info("physics terminated/truncated at step %d — holding final pose", si + 1)
                root.after(1000, _show_final_and_advance)
                return

        else:
            _show_final_and_advance()
            return

        if state["playing"] and not state["paused"]:
            root.after(delay_ms, _auto_tick)
        else:
            state["busy"] = False

    def _show_final_and_advance() -> None:
        """Arm reached end of recorded actions; log final EE state then advance.

        In action-replay mode the arm is already at its physics-final position —
        no teleportation needed.  Log distance to goal so you can see how close
        the actions brought the arm.
        """
        traj_idx = state["traj_idx"]
        gd = groups_data[traj_idx]
        n_steps = len(gd["rewards"])
        _update_step(n_steps, n_steps, float(gd["rewards"][-1]), True)

        ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        dist = float(np.linalg.norm(ee_pos - env.reward_goal))
        log.info(
            "trajectory_end  ee_pos=%s  dist_to_goal=%.4f m  sum_reward=%.4f",
            np.round(ee_pos, 4).tolist(), dist, env.sum_reward,
        )

        if state["playing"] and not args.manual_step:
            root.after(max(delay_ms, 500), _advance_trajectory)
        else:
            state["step_idx"] = n_steps + 1
            state["busy"] = False

    def _advance_trajectory() -> None:
        """Clean up current traj and load next. Must run on main thread."""
        _remove_body(pb, state["marker_ids"].get("apple"))
        _remove_body(pb, state["marker_ids"].get("goal"))
        state["marker_ids"].clear()

        next_idx = state["traj_idx"] + 1
        if next_idx >= len(groups_data):
            if args.loop:
                log.info("All trajectories done — looping.")
                next_idx = 0
            else:
                log.info("All trajectories replayed. Close the control window to exit.")
                step_var.set("All trajectories done. Close window to exit.")
                state["busy"] = False
                return

        state["traj_idx"] = next_idx
        _load_trajectory(next_idx)
        state["busy"] = False
        if state["playing"] and not state["paused"]:
            root.after(delay_ms, _auto_tick)

    # ── auto-play tick ────────────────────────────────────────────────────────

    def _auto_tick() -> None:
        """Called repeatedly by root.after in auto-play mode."""
        if state["paused"]:
            root.after(50, _auto_tick)
            return
        if state["busy"]:
            root.after(10, _auto_tick)
            return
        _advance_once()

    # ── advance (shared by manual keypress and auto-play) ─────────────────────

    def _advance_once(_event=None) -> None:
        """Schedule one step of work on the main thread."""
        if state["busy"]:
            return
        state["busy"] = True

        def work() -> None:
            try:
                si = state["step_idx"]
                gd = groups_data[state["traj_idx"]]
                n_steps = len(gd["rewards"])
                if si > n_steps:
                    _advance_trajectory()
                    return
                _do_one_step()
                root.update_idletasks()
            finally:
                si_after = state["step_idx"]
                gd = groups_data[state["traj_idx"]]
                n_steps = len(gd["rewards"])
                if si_after <= n_steps and not (state["playing"] and not state["paused"]):
                    state["busy"] = False

        root.after(0, work)

    def _goto_traj(idx: int) -> None:
        idx = max(0, min(len(groups_data) - 1, idx))
        if idx == state["traj_idx"] and state["step_idx"] == 0 and not state["playing"]:
            return
        state["playing"] = False
        state["paused"] = True
        state["traj_idx"] = idx
        _load_trajectory(idx)
        state["busy"] = False

    def _step_back(_event=None) -> None:
        if state["busy"] or not state["paused"]:
            return
        si = state["step_idx"]
        if si <= 0:
            return
        state["busy"] = True
        try:
            state["step_idx"] = si - 1
            gd = groups_data[state["traj_idx"]]
            q = decode_joint_angles(gd["joint_angles"][si - 1])
            env.robot.set_joint_angles_no_collision(q.tolist())
            env.step_counter = max(0, si - 1)
            env.is_goal_state = False
            env._get_obs()
            n_steps = len(gd["rewards"])
            r = float(gd["rewards"][si - 2]) if si - 1 > 0 else 0.0
            _update_step(si - 2, n_steps, r, False)
        finally:
            state["busy"] = False

    def _on_enter(_event=None) -> None:
        if args.manual_step:
            _advance_once()
            return
        if state["playing"] and state["paused"]:
            state["paused"] = False
            if not state["busy"]:
                _auto_tick()
            return
        gd = groups_data[state["traj_idx"]]
        if state["step_idx"] > len(gd["rewards"]):
            _load_trajectory(state["traj_idx"])
        state["playing"] = True
        state["paused"] = False
        if not state["busy"]:
            _auto_tick()

    def _on_pause(_event=None) -> None:
        state["paused"] = not state["paused"]
        log.info("replay  %s", "PAUSED" if state["paused"] else "RESUMED")
        if not state["paused"] and state["playing"] and not state["busy"]:
            _auto_tick()

    def _on_prev_traj(_event=None) -> None:
        if state["traj_idx"] > 0:
            _goto_traj(state["traj_idx"] - 1)

    def _on_next_traj(_event=None) -> None:
        if state["traj_idx"] + 1 < len(groups_data):
            _goto_traj(state["traj_idx"] + 1)
        elif args.loop:
            _goto_traj(0)

    def _on_quit(_event=None) -> None:
        root.destroy()

    root.bind("<Left>", _on_prev_traj)
    root.bind("<Right>", _on_next_traj)
    root.bind("<Return>", _on_enter)
    root.bind("n", _advance_once)
    root.bind("N", _advance_once)
    root.bind("b", _step_back)
    root.bind("B", _step_back)
    root.bind("p", _on_pause)
    root.bind("P", _on_pause)
    root.bind("q", _on_quit)
    root.bind("Q", _on_quit)
    root.protocol("WM_DELETE_WINDOW", _on_quit)
    root.focus_force()

    # ── bootstrap: load first trajectory on main thread ───────────────────────
    def _bootstrap() -> None:
        _load_trajectory(0)
        state["busy"] = False
        step_var.set("← / → browse   Enter run   P pause   N/B step when paused")

    root.after(0, _bootstrap)
    root.mainloop()
    log.info("Done.")


if __name__ == "__main__":
    main()
