#!/usr/bin/env python3
"""
generate_agent_data_from_waypoints.py

Batch-generate RL agent-data transitions from a pre-planned waypoints HDF5 file.

For every SUCCESS group the script:
  1. Refines the joint-space path (task-space re-sampling, optional smoothing).
  2. Anchors refined_path[0] to the planner's fixed start joint angles — the arm
     always begins at the collision-free start that the path was planned from.
  3. Resets the env (apple target + grasp goal) and teleports the arm there.
  4. **Kinematic-step** execution per waypoint segment:
       a. Teleport arm to planned q_start (always exact — zero drift).
       b. Get obs_before from this clean position.
       c. Compute one Jacobian EE-velocity action (with dq-cap + EMA smoothing).
       d. Call env.step(action) once for the reward/done signal.
       e. Teleport arm to planned q_tgt (kinematic correction — locks next position).
       f. Get obs_after from this clean position.
       g. Record (obs_before, action, reward, done, obs_after).
  5. Optionally runs a short cartesian goal-correction pass at the end
     (pure physics, arm near goal so drift is minimal).
  6. Validates and saves the transition buffer to output/agent_data/<run_id>.hdf5.

Why kinematic-step instead of pure physics tracking?
  The Jacobian → env.step() controller was diverging because PyBullet's physics
  step duration (step_time) is larger than the control period, causing overshoot
  that compounds across substeps. Kinematic-step gives perfectly coherent
  (obs_before, action, obs_after) pairs where both observations are at planned
  joint positions, while still using env.step() for realistic reward/done signals.

Run from repository root (venv activated):

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints.py \\
        --run-id 20260504_203032

Process only one group:

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints.py \\
        --run-id 20260504_203032 --group-key <key>

Check what was saved:

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints.py \\
        --run-id 20260504_203032 --verify-hdf5

Log is written to:
    logs/data_generator_path_planner/<timestamp>_agent_data_gen.log
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
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
from zenlog import log

# ── resolve roots and insert into sys.path ────────────────────────────────────
_FP_ROOT = Path(__file__).resolve().parents[1]   # feature_apple_path_planning/
_REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root
for _p in (_REPO_ROOT, _FP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── load generator module (CONFIG, planning helpers, save fn, …) ─────────────
_dg_spec = importlib.util.spec_from_file_location(
    "apple_data_generator_72",
    _FP_ROOT / "run_apple_data_generator.py",
)
dg = importlib.util.module_from_spec(_dg_spec)
assert _dg_spec.loader is not None
_dg_spec.loader.exec_module(dg)

from apple_picking_env import ApplePickingEnv  # noqa: E402
from utils_conversions import convert_global_action_to_local  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


# Matches all ANSI/VT100 escape sequences (colors, cursor moves, etc.)
_ANSI_ESC_RE = re.compile(rb"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(data: bytes) -> bytes:
    return _ANSI_ESC_RE.sub(b"", data)


def _tee_reader(read_fd: int, term_fd: int, log_fd: int) -> None:
    """
    Background thread: drain read_fd, write raw bytes to the terminal (term_fd)
    and ANSI-stripped bytes to the log file (log_fd).

    Writing raw to terminal preserves colors in the live session.
    Stripping ANSI for the log file keeps it clean and grep-able.
    """
    try:
        while True:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            try:
                os.write(term_fd, chunk)          # terminal: keep colors
            except OSError:
                pass
            try:
                os.write(log_fd, _strip_ansi(chunk))   # log file: clean text
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
    Comprehensive stdout/stderr capture via OS-level pipe tee.

    Why OS-level and not just sys.stdout replacement?
    PyBullet (and other C extensions) write directly to file descriptor 1/2 at
    the OS level, completely bypassing Python's sys.stdout/sys.stderr.  The
    only way to capture those is to redirect the actual file descriptors using
    os.dup2(), which intercepts ALL writes regardless of origin.

    Steps:
      1. Open the log file at the OS level.
      2. Save the original terminal file descriptors (copies of fd 1 and fd 2).
      3. Create two pipes (one for stdout, one for stderr).
      4. Redirect fd 1 → stdout-pipe-write-end  (so every C-level write to
         stdout lands in the pipe instead of the terminal directly).
         Same for fd 2.
      5. Start two daemon threads that drain the pipes and mirror every byte to
         both the original terminal fd AND the log file fd.
      6. Rebuild Python sys.stdout / sys.stderr from the new fd 1/2 so that
         print() and logging stream handlers also flow through the pipe.
      7. Attach a structured FileHandler to the zenlog logger for the
         timestamped INFO/DEBUG lines (appended after the raw tee).

    The result: the log file contains EVERYTHING visible in the terminal,
    including PyBullet b3Printf lines, GL init messages, and library prints.

    Default log level: INFO (suppresses per-step env DEBUG spam).
    Use --debug-log for full DEBUG output.
    """
    base = _REPO_ROOT / "logs" / dg.LOG_SESSION_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_agent_data_gen.log").resolve()

    # ── 1. flush Python buffers before touching file descriptors ─────────────
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # ── 2. open log file at OS level ─────────────────────────────────────────
    log_fd = os.open(
        str(log_path),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o644,
    )

    # ── 3. save original terminal fds ─────────────────────────────────────────
    orig_out_fd = os.dup(1)   # copy of fd 1 → real terminal stdout
    orig_err_fd = os.dup(2)   # copy of fd 2 → real terminal stderr

    # ── 4. create pipes and redirect fd 1/2 to pipe write ends ───────────────
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()

    os.dup2(out_w, 1)   # fd 1 now points to stdout-pipe-write-end
    os.dup2(err_w, 2)   # fd 2 now points to stderr-pipe-write-end
    os.close(out_w)     # close the extra reference (fd 1 is the surviving copy)
    os.close(err_w)

    # ── 5. start tee threads (pipe → terminal + log file) ────────────────────
    out_thread = threading.Thread(
        target=_tee_reader,
        args=(out_r, orig_out_fd, log_fd),
        daemon=True, name="tee-stdout",
    )
    err_thread = threading.Thread(
        target=_tee_reader,
        args=(err_r, orig_err_fd, log_fd),
        daemon=True, name="tee-stderr",
    )
    out_thread.start()
    err_thread.start()

    # ── 6. rebuild Python sys.stdout/stderr from new fd 1/2 ──────────────────
    sys.stdout = io.TextIOWrapper(
        io.FileIO(os.dup(1), mode="w", closefd=True),
        encoding="utf-8", errors="replace", line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        io.FileIO(os.dup(2), mode="w", closefd=True),
        encoding="utf-8", errors="replace", line_buffering=True,
    )
    # Route Python warnings through stderr so they land in the tee
    warnings.simplefilter("always")

    # Restore on exit (flush buffers, restore fds, let tee threads drain)
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
        # Give tee threads ~0.5 s to drain remaining bytes
        ot.join(timeout=0.5)
        et.join(timeout=0.5)
        for fd in (ofd, efd, lfd):
            try:
                os.close(fd)
            except OSError:
                pass

    atexit.register(_restore)

    # ── 7. logger level — no extra FileHandler needed ────────────────────────
    # The tee already captures ALL stdout/stderr (including every log.info line)
    # into the log file. Adding a FileHandler would duplicate every structured
    # log message. We only set the logger level here.
    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Suppress DEBUG noise from the env (e.g. 'Goal is ON-SCREEN') unless
    # --debug-log is requested by raising the root logger's effective level.
    if not debug:
        # The zenlog stream handler writes at DEBUG but we want INFO+ in the log.
        # Replace the existing stream handler with an INFO-filtered one.
        for h in zen_logger.handlers[:]:
            h.setLevel(logging.INFO)

    log.info("Agent-data generation log: %s  (debug_log=%s)", log_path, debug)
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 / buffer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sort_key(k: str):
    try:
        return (0, float(k))
    except ValueError:
        return (1, str(k))


def sort_group_keys(keys: list[str]) -> list[str]:
    return sorted(keys, key=_sort_key)


def attrs_to_metadata(grp: h5py.Group) -> dict:
    return {k: np.asarray(v) for k, v in grp.attrs.items()}


def _is_finite_array(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr)))


def _hdf5_safe_attr_value(value):
    """Convert metadata values into h5py-friendly attr types."""
    if isinstance(value, (str, bytes)):
        return str(value)
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return value.item() if hasattr(value, "item") else value
    arr = np.asarray(value)
    if np.isscalar(arr):
        if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("U", "S", "O"):
            return str(arr.item())
        return arr.item() if hasattr(arr, "item") else value
    if arr.dtype.kind in ("i", "u", "f", "b"):
        return arr
    if arr.dtype.kind in ("U", "S", "O"):
        return [str(x) for x in arr.reshape(-1).tolist()]
    return str(value)


def validate_transition_buffers(buffers: dict, env: ApplePickingEnv) -> tuple[bool, str]:
    obs = buffers["observations"]
    nxt = buffers["next_observations"]
    acts = buffers["actions"]
    rews = buffers["rewards"]
    dns = buffers["dones"]
    if not obs:
        return False, "observations buffer is empty"
    n = len(obs)
    if not (len(nxt) == len(acts) == len(rews) == len(dns) == n):
        return False, "buffer length mismatch"
    obs_keys = list(obs[0].keys())
    act_dim = int(env.action_space.shape[0])
    for i in range(n):
        if set(obs[i].keys()) != set(obs_keys) or set(nxt[i].keys()) != set(obs_keys):
            return False, f"inconsistent observation keys at step {i}"
        for k in obs_keys:
            if not (
                _is_finite_array(np.asarray(obs[i][k]))
                and _is_finite_array(np.asarray(nxt[i][k]))
            ):
                return False, f"non-finite observation at step {i}, key={k}"
        act = np.asarray(acts[i], dtype=np.float64).reshape(-1)
        if act.shape[0] != act_dim:
            return False, f"action dim mismatch at step {i}: got {act.shape[0]}, expected {act_dim}"
        if not _is_finite_array(act):
            return False, f"non-finite action at step {i}"
        if not np.isfinite(float(rews[i])):
            return False, f"non-finite reward at step {i}"
    return True, "ok"


def verify_agent_hdf5(path: Path) -> None:
    if not path.exists():
        log.warning("[verify] agent-data file not found: %s", path)
        return
    with h5py.File(path, "r") as f:
        keys = sort_group_keys(list(f.keys()))
        log.info("[verify] file=%s  trajectories=%s", path, len(keys))
        if not keys:
            return
        first = f[keys[0]]
        obs_shapes = {k: tuple(first["observations"][k].shape) for k in first["observations"].keys()}
        log.info(
            "[verify] first_group=%s  actions=%s  rewards=%s  dones=%s",
            keys[0],
            tuple(first["actions"].shape),
            tuple(first["rewards"].shape),
            tuple(first["dones"].shape),
        )
        log.info("[verify] observation_shapes=%s", obs_shapes)


# ─────────────────────────────────────────────────────────────────────────────
# Visual markers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_marker(pb_client, body_id: int | None) -> None:
    if body_id is None or body_id < 0:
        return
    try:
        pb_client.removeBody(body_id)
    except Exception:
        pass


def refresh_markers(env: ApplePickingEnv, metadata: dict, marker_ids: dict) -> None:
    """Draw a red sphere at the active apple center (goal dot removed)."""
    apple_pos = np.asarray(metadata["apple_center"], dtype=np.float64).reshape(3).tolist()
    pb = env.pb_client
    _remove_marker(pb, marker_ids.get("apple"))
    marker_ids["apple"] = dg.draw_debug_sphere(pb, apple_pos, 0.05, [0.9, 0.08, 0.08, 0.9])


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_rollout(env: ApplePickingEnv, metadata: dict, refined_path: list) -> dict:
    """
    Reset the env for a new group and teleport the arm to refined_path[0].

    refined_path[0] is already anchored to the planner's start_joint_angles
    (collision-free, consistent with the planned path) by the caller.
    """
    apple = np.asarray(metadata["apple_center"], dtype=np.float32).reshape(3,)
    goal_p = np.asarray(metadata["goal_pos"], dtype=np.float32).reshape(3,)
    env.reset(seed=None, options={"target_apple": apple})
    env.reward_goal = goal_p.astype(np.float32, copy=False)
    env.robot.set_joint_angles_no_collision(refined_path[0])
    env.step_counter = 0
    env.is_goal_state = False
    env.sum_reward = 0.0
    env.init_pos_ee, env.init_or_ee = env.robot.get_current_pose(env.robot.tool0_link_idx)
    return env._get_obs()


# ─────────────────────────────────────────────────────────────────────────────
# Action computation helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_action(
    env: ApplePickingEnv,
    q_start: np.ndarray,
    q_tgt: np.ndarray,
    control_freq: float,
    max_ee_vel: float,
    dq_cap: float,
    step_scale: float,
    prev_action: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """
    Compute a single Jacobian EE-velocity action from q_start toward q_tgt.

    Assumes the arm is already kinematically positioned at q_start by the caller.
    dq_cap clamps per-joint delta before Jacobian conversion (prevents velocity
    spikes from large waypoint gaps). EMA smoothing (alpha) reduces jitter between
    consecutive actions.
    """
    dq = q_tgt - q_start
    if dq_cap > 0.0:
        dq = np.clip(dq, -dq_cap, dq_cap)

    # Jacobian query at q_start — arm is already there, just a kinematic snapshot
    joint_vel = dq * control_freq
    jac = env.robot.calculate_jacobian()
    global_ee_vel = np.matmul(jac, joint_vel)
    local_vel = np.asarray(
        convert_global_action_to_local(env.robot, global_ee_vel), dtype=np.float64
    )
    local_vel = np.clip(local_vel, -max_ee_vel, max_ee_vel)
    action_raw = local_vel * step_scale

    if prev_action is None:
        return action_raw
    return (1.0 - alpha) * action_raw + alpha * prev_action


# ─────────────────────────────────────────────────────────────────────────────
# Quality reporting
# ─────────────────────────────────────────────────────────────────────────────

def log_quality_report(
    *,
    group_key: str,
    rewards: np.ndarray,
    actions: list,
    ee_dists: list[float],
    ee_step_displacements: list[float],
    start_ee_dist: float,
    num_segments: int,
    count_in_frame: int,
    success: bool,
) -> None:
    """
    Compute and log a structured quality report for one group's trajectory.

    Metrics
    -------
    total_return       Undiscounted sum of all rewards. Positive = net approach.
    pos_step_frac      Fraction of steps with reward > 0. < 0.3 warns of a
                       mostly-negative trajectory (large detour or poor path).
    detour_ratio       max(ee_dists) / ee_dists[0]. > 1.0 means the path
                       temporarily moved the EE away from the goal (normal for
                       collision avoidance). > 2.0 is a red flag.
    final_slope        Linear slope (m/step) of the last min(10, N) EE distances.
                       Should be negative — EE converging to the goal at the end.
    backward_frac      Fraction of steps where EE moved AWAY from the goal.
                       Near 0 = smooth forward motion. > 0.2 indicates shaking.
    mean_step_m        Mean EE Cartesian displacement per step (metres).
                       Gauge of how far the arm moves each step along the path.
    action_cos_sim     Mean cosine similarity between consecutive recorded actions.
                       Near 1.0 = consistent direction. < 0.5 = frequent reversals.
    max_action_delta   L2 norm of the largest action change between consecutive
                       steps. Large values indicate jittery / non-smooth actions.
    mean_action_norm   Average L2 norm of all actions. Gauge of overall velocity.
    coverage_frac      Fraction of planned segments executed before termination.
                       < 0.5 means goal was reached very early.
    visibility_frac    Fraction of transitions where the apple is in frame.
    """
    n = len(rewards)
    if n == 0:
        log.warning("quality_report  key=%s  NO transitions — skipping report", group_key)
        return

    action_arr = np.asarray(actions, dtype=np.float32)

    # ── metrics ───────────────────────────────────────────────────────────────
    total_return = float(rewards.sum())
    pos_step_frac = float(np.mean(rewards > 0))

    dists = np.asarray(ee_dists, dtype=np.float64)
    detour_ratio = float(dists.max() / dists[0]) if dists[0] > 1e-6 else float("nan")

    tail = max(1, min(10, len(dists)))
    if len(dists) >= 2:
        xs = np.arange(tail, dtype=np.float64)
        final_slope = float(np.polyfit(xs, dists[-tail:], 1)[0])
    else:
        final_slope = float("nan")

    if len(action_arr) >= 2:
        deltas = np.linalg.norm(np.diff(action_arr, axis=0), axis=1)
        max_action_delta = float(deltas.max())
        # cosine similarity between consecutive actions (motion direction consistency)
        norms = np.linalg.norm(action_arr, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-9, 1.0, norms)
        unit_actions = action_arr / safe_norms
        cos_sims = np.sum(unit_actions[:-1] * unit_actions[1:], axis=1)
        action_cos_sim = float(np.mean(cos_sims))
    else:
        max_action_delta = float("nan")
        action_cos_sim = float("nan")

    mean_action_norm = float(np.linalg.norm(action_arr, axis=1).mean()) if n > 0 else 0.0
    coverage_frac = n / num_segments if num_segments > 0 else float("nan")
    visibility_frac = count_in_frame / n if n > 0 else float("nan")

    # ── motion smoothness (shaking indicators) ────────────────────────────────
    # backward_frac: fraction of steps where EE moved AWAY from goal
    if len(dists) >= 2:
        backward_steps = int(sum(dists[i] > dists[i - 1] for i in range(1, len(dists))))
        backward_frac = backward_steps / (len(dists) - 1)
    else:
        backward_steps = 0
        backward_frac = float("nan")

    steps_arr = np.asarray(ee_step_displacements, dtype=np.float64)
    mean_step_m = float(steps_arr.mean()) if len(steps_arr) > 0 else float("nan")

    # ── warnings ──────────────────────────────────────────────────────────────
    flags: list[str] = []
    if pos_step_frac < 0.30:
        flags.append("LOW_POS_FRAC")
    if detour_ratio > 2.0:
        flags.append("HIGH_DETOUR")
    if final_slope is not None and not np.isnan(final_slope) and final_slope > 0:
        flags.append("NOT_CONVERGING")
    if not np.isnan(backward_frac) and backward_frac > 0.40:
        flags.append("HIGH_BACKWARD_FRAC")   # >40% steps moved away — may indicate bad path
    if float(rewards.min()) < -3.0:
        flags.append("LARGE_NEG_REWARD")     # single step drove EE far from goal
    if not np.isnan(max_action_delta) and max_action_delta > 0.15:
        flags.append("JITTERY_ACTIONS")
    if not np.isnan(action_cos_sim) and action_cos_sim < 0.50:
        flags.append("LOW_DIR_CONSISTENCY")  # frequent action direction reversals
    if visibility_frac < 0.90:
        flags.append("LOW_VISIBILITY")
    if not success:
        flags.append("NOT_SUCCESS")

    flag_str = "  [" + " ".join(flags) + "]" if flags else ""

    min_reward = float(rewards.min())
    log.info(
        "QUALITY  key=%s  n=%d  start_dist=%.3fm  return=%.3f  min_r=%.3f  pos_frac=%.2f  "
        "detour=%.2fx  final_slope=%.4f m/step  "
        "backward_frac=%.2f  mean_step=%.4fm  action_cos=%.3f  "
        "max_Δaction=%.4f  mean_action=%.4f  "
        "coverage=%.2f  visibility=%.2f%s",
        group_key, n,
        start_ee_dist, total_return, min_reward, pos_step_frac,
        detour_ratio, final_slope,
        backward_frac, mean_step_m, action_cos_sim,
        max_action_delta, mean_action_norm,
        coverage_frac, visibility_frac,
        flag_str,
    )

    if flags:
        log.warning("QUALITY FLAGS  key=%s  %s", group_key, " ".join(flags))


# ─────────────────────────────────────────────────────────────────────────────
# Per-group execution — kinematic-step approach
# ─────────────────────────────────────────────────────────────────────────────

def run_group(
    *,
    env: ApplePickingEnv,
    group_key: str,
    metadata: dict,
    refined_path: list,
    optical_flow_model,
    agent_data_path: Path,
    args: argparse.Namespace,
    control_freq: float,
    max_ee_vel: float,
) -> str:
    """
    Execute one waypoint group with the kinematic-step controller and save.

    Each segment:
        1. Teleport arm to planned q_start  (zero drift, always exact)
        2. obs_before = env._get_obs()      (camera from planned position)
        3. Compute Jacobian action
        4. env.step(action)                 (physics: reward + done signal)
        5. Teleport arm to planned q_tgt    (kinematic correction)
        6. obs_after = env._get_obs()       (camera from planned next position)
        7. Record (obs_before, action, reward, done, obs_after)

    Returns "saved" | "skipped:<reason>" | "error:<msg>"
    """
    num_segments = len(refined_path) - 1
    log.info(
        "GROUP START  key=%s  refined_len=%s  segments=%s",
        group_key, len(refined_path), num_segments,
    )

    buffers: dict = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "goal_achieved_count": 0,
        "truncated_flag": False,
    }

    # Bootstrap: reset env, set apple target, teleport arm to path start
    bootstrap_rollout(env, metadata, refined_path)

    marker_ids: dict = {"apple": -1}
    refresh_markers(env, metadata, marker_ids)

    alpha = float(np.clip(args.action_smoothing_alpha, 0.0, 1.0))
    dq_cap = max(0.0, float(args.record_joint_dq_cap))
    prev_action: np.ndarray | None = None
    terminated = False
    truncated = False

    # Starting EE-to-goal distance (before any movement) for quality reporting
    goal_pos_arr_start = np.asarray(metadata["goal_pos"], dtype=np.float64)
    ee_pos_start, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
    start_ee_dist = float(np.linalg.norm(np.asarray(ee_pos_start) - goal_pos_arr_start))

    # Per-step motion quality tracking (for quality report and per-segment log)
    ee_dists: list[float] = []          # EE-to-goal distance at each q_tgt
    ee_step_displacements: list[float] = []  # EE Cartesian distance moved each step
    goal_pos_arr = np.asarray(metadata["goal_pos"], dtype=np.float64)

    # ── kinematic-step segment loop ───────────────────────────────────────────
    #
    # Segment order (eliminates visual shaking):
    #   1. Arm is already at q_start from bootstrap / previous segment's lock.
    #      Guard-teleport ensures correctness for si==0 and after any anomaly.
    #   2. obs_before  at q_start  → sets prev_obs_info[achieved_pos] = EE@q_start
    #   3. Jacobian action at q_start
    #   4. Teleport arm FORWARD to q_tgt  ← smooth visual advance along path
    #   5. env.step(zeros) — arm stays at q_tgt, physics computes reward ✓
    #      Zero EE velocity: no overshoot, so step-6 re-lock is a near-no-op.
    #   6. Re-lock arm to q_tgt           ← correct near-zero physics drift
    #   7. obs_after at q_tgt
    #
    # Reward correctness: inside env.step(), prev_pos = EE@q_start (from step 2),
    # curr_pos ≈ EE@q_tgt (arm is there at step 5), so
    # distance_reward = (||EE_q_start - goal|| - ||EE_q_tgt - goal||) × 100  ✓
    # Recorded action is the Jacobian motion command (not the zero sent to env). ✓
    # ─────────────────────────────────────────────────────────────────────────
    for si in range(num_segments):
        if terminated or truncated:
            break

        q_start = np.asarray(refined_path[si], dtype=np.float64)
        q_tgt = np.asarray(refined_path[si + 1], dtype=np.float64)
        dq_true = float(np.linalg.norm(q_tgt - q_start))

        # Step 1 — guard-teleport to q_start (no-op for si>0; safety for si=0)
        env.robot.set_joint_angles_no_collision(q_start.tolist())
        # Capture EE Cartesian position at q_start for step-displacement tracking
        _ee_pos_q_start, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        ee_pos_q_start = np.asarray(_ee_pos_q_start, dtype=np.float64)

        # Early-exit if arm is already within goal tolerance at this q_start.
        # The last few planned waypoints often enter the apple's collision radius —
        # executing env.step() from inside the apple fires a large collision penalty
        # even though the arm is approaching the goal. Stop here and mark success.
        dist_at_q_start = float(np.linalg.norm(ee_pos_q_start - goal_pos_arr))
        if dist_at_q_start <= float(args.final_goal_pos_tol) and si > 0:
            log.info(
                "segment %s/%s: arm within goal tol at q_start"
                " (dist=%.4fm ≤ %.4fm) — stopping path, marking prev done=True",
                si + 1, num_segments, dist_at_q_start, args.final_goal_pos_tol,
            )
            buffers["dones"][-1] = True
            break

        # Step 2 — observation at q_start (sets env.prev_observation_info = EE@q_start)
        obs_before = env._get_obs()
        obs_before = dg.ensure_optical_flow(obs_before, optical_flow_model)

        # Step 3 — Jacobian action (arm is at q_start; no redundant teleport)
        action_applied = compute_action(
            env, q_start, q_tgt,
            control_freq, max_ee_vel, dq_cap,
            float(args.step_scale), prev_action, alpha,
        )

        # Step 4 — advance arm to q_tgt BEFORE the physics step.
        # Visually the arm moves forward along the path (not backward from drift).
        env.robot.set_joint_angles_no_collision(q_tgt.tolist())

        # Step 5 — physics step with ZERO EE velocity.
        # Sending zeros keeps the arm at q_tgt during physics (no overshoot beyond
        # q_tgt, so the step-6 re-lock is essentially a no-op — no visible snap-back).
        # Reward is still correct: env uses prev_pos=EE@q_start (set in step 2) and
        # curr_pos≈EE@q_tgt (arm is there), giving dist(q_start→goal) improvement. ✓
        # The RECORDED action (action_applied) is the true Jacobian motion command. ✓
        _obs_physics, reward, terminated, truncated, info = env.step(
            np.zeros(6, dtype=np.float64)
        )

        # Step 6 — re-lock arm to exact q_tgt (corrects near-zero physics drift)
        env.robot.set_joint_angles_no_collision(q_tgt.tolist())

        # Step 7 — observation at clean planned next position
        obs_after = env._get_obs()
        obs_after = dg.ensure_optical_flow(obs_after, optical_flow_model)

        done_flag = bool(terminated or truncated)

        # Record transition
        buffers["observations"].append(obs_before)
        buffers["next_observations"].append(obs_after)
        buffers["actions"].append(np.asarray(action_applied, dtype=np.float32))
        buffers["rewards"].append(float(reward))
        buffers["dones"].append(done_flag)
        if bool(info.get("goal_achieved", False)):
            buffers["goal_achieved_count"] += 1
        if truncated:
            buffers["truncated_flag"] = True
        if args.max_transitions > 0 and len(buffers["actions"]) >= int(args.max_transitions):
            terminated = True

        prev_action = np.asarray(action_applied, dtype=np.float64)

        # ── motion quality diagnostics ─────────────────────────────────────────
        # EE position and distances at q_tgt (arm is re-locked there in step 6)
        _ee_pos_tgt, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        ee_pos_tgt = np.asarray(_ee_pos_tgt, dtype=np.float64)

        ee_dist_to_goal = float(np.linalg.norm(ee_pos_tgt - goal_pos_arr))
        ee_step_m = float(np.linalg.norm(ee_pos_tgt - ee_pos_q_start))  # physical EE displacement
        prev_dist = ee_dists[-1] if ee_dists else start_ee_dist
        delta_dist = prev_dist - ee_dist_to_goal   # positive = approaching goal
        toward_goal = delta_dist >= 0.0

        ee_dists.append(ee_dist_to_goal)
        ee_step_displacements.append(ee_step_m)

        log.debug(
            "si=%s/%s  dq=%.5f  reward=%.5f  ee_dist=%.4f  Δdist=%+.4f  "
            "step=%.4fm  →goal=%s  terminated=%s  action_norm=%.5f",
            si + 1, num_segments,
            dq_true, reward, ee_dist_to_goal, delta_dist,
            ee_step_m, toward_goal, terminated,
            float(np.linalg.norm(action_applied)),
        )

        log.info(
            "segment %s/%s  reward=%+.5f  ee_dist=%.4fm  Δdist=%+.5fm  "
            "step=%.4fm  →goal=%s  terminated=%s",
            si + 1, num_segments, reward,
            ee_dist_to_goal, delta_dist, ee_step_m, toward_goal, terminated,
        )
        if reward < -2.0:
            log.warning(
                "LARGE_NEG_REWARD  si=%s/%s  reward=%.5f  ee_dist=%.4fm"
                "  Δdist=%+.5fm  step=%.4fm  — likely collision with apple/tree",
                si + 1, num_segments, reward, ee_dist_to_goal, delta_dist, ee_step_m,
            )

    # ── final goal-correction pass (pure physics, arm is near goal) ───────────
    max_corr = max(0, int(args.final_correction_steps))
    if not terminated and not truncated and max_corr > 0:
        goal_pos = np.asarray(metadata["goal_pos"], dtype=np.float64)
        control_time = float(dg.CONFIG["simulation_setup"]["control_time"])
        log.info(
            "final_correction  max_steps=%s  goal_tol=%.4f",
            max_corr, args.final_goal_pos_tol,
        )

        for corr_idx in range(1, max_corr + 1):
            ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
            ee_pos = np.asarray(ee_pos, dtype=np.float64)
            err_norm = float(np.linalg.norm(goal_pos - ee_pos))

            # If the arm is already within tolerance, mark the previous segment's
            # transition as done=True (arm reached goal) and exit without calling
            # env.step() — sending an approach action into the apple at this range
            # causes a collision penalty which would corrupt the last transition.
            if err_norm <= float(args.final_goal_pos_tol):
                if buffers["dones"]:
                    buffers["dones"][-1] = True
                log.info(
                    "final_correction arm at goal  step=%s  err=%.5f"
                    "  — marking last transition done=True  (no env.step)",
                    corr_idx, err_norm,
                )
                break

            obs_before_corr = env._get_obs()
            obs_before_corr = dg.ensure_optical_flow(obs_before_corr, optical_flow_model)

            ee_vel_global = np.zeros(6, dtype=np.float64)
            ee_vel_global[:3] = (goal_pos - ee_pos) / max(control_time, 1e-6)
            local_vel = np.asarray(
                convert_global_action_to_local(env.robot, ee_vel_global), dtype=np.float64
            )
            local_vel = np.clip(local_vel, -max_ee_vel, max_ee_vel)
            action_raw = local_vel * float(args.step_scale)
            if prev_action is None:
                action_corr = action_raw
            else:
                action_corr = (1.0 - alpha) * action_raw + alpha * prev_action

            obs_after_corr, reward, terminated, truncated, info = env.step(
                action_corr.astype(np.float64)
            )
            obs_after_corr = dg.ensure_optical_flow(obs_after_corr, optical_flow_model)
            done_flag = bool(terminated or truncated)

            buffers["observations"].append(obs_before_corr)
            buffers["next_observations"].append(obs_after_corr)
            buffers["actions"].append(np.asarray(action_corr, dtype=np.float32))
            buffers["rewards"].append(float(reward))
            buffers["dones"].append(done_flag)
            if bool(info.get("goal_achieved", False)):
                buffers["goal_achieved_count"] += 1
            if truncated:
                buffers["truncated_flag"] = True
            prev_action = np.asarray(action_corr, dtype=np.float64)

            log.info(
                "final_correction %s/%s  err=%.5f  reward=%.5f  terminated=%s",
                corr_idx, max_corr, err_norm, reward, terminated,
            )
            if done_flag:
                log.info("final_correction done  step=%s  err=%.5f", corr_idx, err_norm)
                break

    # ── quality gates ─────────────────────────────────────────────────────────
    success = bool(buffers["dones"] and buffers["dones"][-1])

    if buffers["truncated_flag"]:
        log.warning("skip save  key=%s  reason=truncated", group_key)
        return "skipped:truncated"

    valid, reason = validate_transition_buffers(buffers, env)
    if not valid:
        log.error("skip save  key=%s  invalid_buffers(%s)", group_key, reason)
        return f"skipped:invalid_buffers({reason})"

    if args.save_only_success and not success:
        log.info("skip save  key=%s  reason=not_success (--save-only-success)", group_key)
        return "skipped:not_success"

    n_transitions = len(buffers["actions"])
    count_in_frame = int(
        sum(
            1 for o in buffers["next_observations"]
            if "point_mask" in o and np.any(o["point_mask"] > 0)
        )
    )
    visibility_frac = count_in_frame / n_transitions if n_transitions > 0 else 0.0
    min_vis = float(args.min_visibility)
    if min_vis > 0.0 and visibility_frac < min_vis:
        log.warning(
            "skip save  key=%s  reason=low_visibility  "
            "visibility=%.2f  threshold=%.2f  (use --min-visibility 0 to keep)",
            group_key, visibility_frac, min_vis,
        )
        return f"skipped:low_visibility({visibility_frac:.2f}<{min_vis:.2f})"

    save_info = {
        "count_in_frame": count_in_frame,
        "path_length": int(len(buffers["actions"])),
        "source_group_key": str(group_key),
        "controller_mode": "kinematic_step_ee_env_step",
        "goal_achieved_count": int(buffers["goal_achieved_count"]),
        "truncated_flag": bool(buffers["truncated_flag"]),
    }
    tree_info = {k: _hdf5_safe_attr_value(v) for k, v in metadata.items()}

    try:
        dg.save_agent_data_to_hdf5(
            buffers["observations"],
            buffers["actions"],
            buffers["rewards"],
            buffers["dones"],
            buffers["next_observations"],
            save_info,
            success,
            tree_info,
            np.asarray(dg.CONFIG["tree_setup"]["position"]),
            np.asarray(dg.CONFIG["tree_setup"]["orientation"]),
            str(agent_data_path),
        )
    except Exception as exc:
        log.error("save error  key=%s  err=%s", group_key, exc, exc_info=True)
        return f"error:{exc}"

    reward_np = np.asarray(buffers["rewards"], dtype=np.float64)
    log_quality_report(
        group_key=group_key,
        rewards=reward_np,
        actions=buffers["actions"],
        ee_dists=ee_dists,
        ee_step_displacements=ee_step_displacements,
        start_ee_dist=start_ee_dist,
        num_segments=num_segments,
        count_in_frame=count_in_frame,
        success=success,
    )
    log.info(
        "SAVED  key=%s  transitions=%s  success=%s  "
        "reward[min/mean/max]=[%.5f / %.5f / %.5f]  "
        "goal_achieved_count=%s  count_in_frame=%s",
        group_key,
        len(buffers["actions"]),
        success,
        float(np.min(reward_np)),
        float(np.mean(reward_np)),
        float(np.max(reward_np)),
        int(buffers["goal_achieved_count"]),
        count_in_frame,
    )
    return "saved"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch-generate RL agent-data from a pre-planned waypoints HDF5 file "
            "using the kinematic-step approach (teleport → obs → step → teleport → obs)."
        )
    )

    # ── input / output ────────────────────────────────────────────────────────
    p.add_argument(
        "--run-id", default=None,
        help="Waypoints HDF5 stem under output/waypoints/. Latest file used if omitted.",
    )
    p.add_argument(
        "--group-key", default=None,
        help="Process only this exact HDF5 group key (must be SUCCESS).",
    )
    p.add_argument(
        "--max-groups", type=int, default=0,
        help="Stop after processing N SUCCESS groups (0 = all).",
    )
    p.add_argument(
        "--save-only-success", action="store_true",
        help="Save only trajectories where the episode terminates with done=True.",
    )
    p.add_argument(
        "--min-visibility", type=float, default=0.0,
        help=(
            "Skip saving groups where the fraction of transitions with the apple "
            "visible in frame is below this threshold [0, 1]. "
            "E.g. 0.5 skips trajectories where the apple is off-screen >50%% of the time. "
            "Default 0.0 (keep all)."
        ),
    )
    p.add_argument(
        "--optical-flow", action="store_true",
        help="Compute RAFT optical flow each step (default: off → zero-flow arrays).",
    )
    p.add_argument(
        "--verify-hdf5", action="store_true",
        help="Print agent-data HDF5 summary and exit (no data generation).",
    )

    # ── controller knobs ──────────────────────────────────────────────────────
    p.add_argument(
        "--step-scale", type=float, default=0.15,
        help="EE-velocity action scale applied after Jacobian conversion (default 0.15).",
    )
    p.add_argument(
        "--action-smoothing-alpha", type=float, default=0.35,
        help="EMA weight on previous action [0,1]. 0=no smoothing (default 0.35).",
    )
    p.add_argument(
        "--record-joint-dq-cap", type=float, default=0.04,
        help="Per-joint dq clamp before Jacobian to prevent velocity spikes (default 0.04 rad).",
    )
    p.add_argument(
        "--final-correction-steps", type=int, default=8,
        help="Cartesian goal-correction steps after path end, pure physics (default 8).",
    )
    p.add_argument(
        "--final-goal-pos-tol", type=float, default=0.05,
        help="Stop correction once EE-to-goal distance <= this (metres, default 0.05).",
    )
    p.add_argument(
        "--max-transitions", type=int, default=0,
        help="Safety cap on transitions per group (0 = no cap).",
    )

    # ── logging ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--debug-log", action="store_true",
        help=(
            "Write DEBUG-level messages to the log file (very verbose — includes "
            "per-step env messages like 'Goal is ON-SCREEN'). Default: INFO only."
        ),
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.chdir(_REPO_ROOT)
    setup_logging(debug=args.debug_log)

    waypoints_dir, agent_data_dir, _ = dg._ensure_output_dirs()
    run_id, waypoints_path = dg._resolve_run_id(args.run_id, waypoints_dir, should_exist=True)
    agent_data_path = (agent_data_dir / f"{run_id}.hdf5").resolve()

    log.info(
        "run_id=%s  waypoints=%s  agent_data=%s",
        run_id, waypoints_path, agent_data_path,
    )

    if args.verify_hdf5:
        verify_agent_hdf5(agent_data_path)
        return

    # ── collect SUCCESS group keys ────────────────────────────────────────────
    with h5py.File(waypoints_path, "r") as f:
        all_keys = sort_group_keys(list(f.keys()))
        success_keys = [
            k for k in all_keys
            if int(f[k].attrs.get("fail_mode", -1)) == dg.ResultMode.SUCCESS.value
        ]

    if not success_keys:
        log.error("No SUCCESS groups found in %s — nothing to do.", waypoints_path)
        return

    if args.group_key is not None:
        if args.group_key not in success_keys:
            log.error(
                "group-key %r not found in SUCCESS groups. Available: %s",
                args.group_key, success_keys,
            )
            return
        success_keys = [args.group_key]

    if args.max_groups > 0:
        success_keys = success_keys[: args.max_groups]

    log.info(
        "SUCCESS groups to process: %s  (total in file: %s)",
        len(success_keys), len(all_keys),
    )
    log.info(
        "Controller: step_scale=%.3f  alpha=%.3f  dq_cap=%.4f  "
        "final_correction_steps=%s  save_only_success=%s  min_visibility=%.2f",
        args.step_scale, args.action_smoothing_alpha, args.record_joint_dq_cap,
        args.final_correction_steps, args.save_only_success, args.min_visibility,
    )

    # ── optical flow ──────────────────────────────────────────────────────────
    optical_flow_model = None
    if args.optical_flow:
        try:
            optical_flow_model = dg.OpticalFlow(size=dg.DEFAULT_OPTICAL_FLOW_SIZE)
            log.info("Optical flow model loaded.")
        except Exception as exc:
            log.error("Optical flow init failed: %s — using zeros.", exc)

    # ── build env once ────────────────────────────────────────────────────────
    env = ApplePickingEnv(config=dg.CONFIG)
    control_freq = 1.0 / dg.CONFIG["simulation_setup"]["control_time"]
    max_ee_vel = dg.CONFIG["planning"]["max_ee_velocity"]
    planner_start = np.asarray(
        dg.CONFIG["robot_setup"]["start_joint_angles"], dtype=np.float64
    )

    log.info(
        "Env ready.  control_freq=%.1f Hz  max_ee_vel=%.3f  planner_start=%s",
        control_freq, max_ee_vel, np.round(planner_start, 4).tolist(),
    )

    # Draw a red sphere at every apple centroid so the tree looks populated
    for center in env.apple_centroids:
        dg.draw_debug_sphere(env.pb_client, list(center), 0.05, [0.9, 0.08, 0.08, 0.9])

    # ── build planning functions once (needed for path refinement) ────────────
    _, _, extend_fn, collision_fn = dg.setup_planning_functions(
        env.robot,
        env.tree.pyb_id,
        dg.CONFIG["planning"],
        dg.CONFIG["visualization"],
        env.pb_client,
        env.tree,
    )

    # ── main group loop ───────────────────────────────────────────────────────
    stats: dict[str, int] = {"saved": 0, "skipped": 0, "error": 0}
    t0_total = time.time()

    for g_idx, group_key in enumerate(success_keys):
        log.info(
            "PROCESSING group %s/%s  key=%s",
            g_idx + 1, len(success_keys), group_key,
        )
        t0_group = time.time()

        with h5py.File(waypoints_path, "r") as f:
            grp = f[group_key]
            waypoints = grp["waypoints"][:]
            metadata = attrs_to_metadata(grp)

        if len(waypoints) < 2:
            log.warning("Group %r has fewer than 2 waypoints — skipping.", group_key)
            stats["skipped"] += 1
            continue

        # refine path
        try:
            refined_path = dg.shortcut_and_refine_path(
                env.robot,
                waypoints.tolist(),
                extend_fn,
                collision_fn,
                dg.CONFIG["planning"]["task_space_refinement_threshold"],
                enable_smoothing=dg.CONFIG["planning"].get("enable_smoothing", True),
            )
        except Exception as exc:
            log.error("Path refinement failed for %r: %s", group_key, exc, exc_info=True)
            stats["error"] += 1
            continue

        if len(refined_path) < 2:
            log.warning("Refined path for %r too short — skipping.", group_key)
            stats["skipped"] += 1
            continue

        # KEY FIX: anchor the start to the collision-free planner start position
        refined_path[0] = planner_start.tolist()

        log.info(
            "refined path: %s waypoints → %s segments  start=%s",
            len(refined_path), len(refined_path) - 1,
            np.round(refined_path[0], 4),
        )

        try:
            status = run_group(
                env=env,
                group_key=group_key,
                metadata=metadata,
                refined_path=refined_path,
                optical_flow_model=optical_flow_model,
                agent_data_path=agent_data_path,
                args=args,
                control_freq=control_freq,
                max_ee_vel=max_ee_vel,
            )
        except Exception as exc:
            log.error("run_group crashed for %r: %s", group_key, exc, exc_info=True)
            status = f"error:{exc}"

        elapsed = time.time() - t0_group
        log.info("group %r  status=%s  elapsed=%.1fs", group_key, status, elapsed)
        print("=" * 64, flush=True)  # separator after each group; captured by tee

        if status == "saved":
            stats["saved"] += 1
        elif status.startswith("error"):
            stats["error"] += 1
        else:
            stats["skipped"] += 1

    # ── final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0_total
    log.info(
        "DONE  groups=%s  saved=%s  skipped=%s  errors=%s  total_time=%.1fs",
        len(success_keys), stats["saved"], stats["skipped"], stats["error"], total_elapsed,
    )
    verify_agent_hdf5(agent_data_path)


if __name__ == "__main__":
    main()
