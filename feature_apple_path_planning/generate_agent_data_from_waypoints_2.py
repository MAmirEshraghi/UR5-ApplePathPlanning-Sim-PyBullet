#!/usr/bin/env python3
"""
generate_agent_data_from_waypoints_2.py  —  PHYSICS-STEP version

Batch-generate RL agent-data transitions from a pre-planned waypoints HDF5 file.

WHAT WAS WRONG IN v1 (generate_agent_data_from_waypoints.py)
─────────────────────────────────────────────────────────────
v1 used a "kinematic-step" loop:
  a. Teleport arm to q_start.
  b. obs_before = _get_obs().
  c. Compute Jacobian action (action_applied).
  d. Teleport arm to q_tgt.
  e. env.step(np.zeros(6))   ← ZEROS, not action_applied!
  f. Re-lock arm to q_tgt.
  g. obs_after = _get_obs().

The stored transition was (obs_before@q_start, action_applied, obs_after@q_tgt)
but the action that was ACTUALLY executed by the physics engine was ZERO.
When replayed through env.step(action_applied) the arm received a 13× scaled
velocity, immediately self-collided, and moved away from the goal.
The training data was useless for RL because the (obs, action, next_obs) triples
were physically inconsistent.

WHAT v2 DOES INSTEAD  —  continuous physics rollout
────────────────────────────────────────────────────
For every SUCCESS group the script:
  1. Refines the joint-space path (task-space re-sampling, optional smoothing).
  2. Anchors refined_path[0] to the planner's fixed start joint angles.
  3. Resets the env and teleports the arm ONCE to refined_path[0].
  4. **Continuous physics rollout** — NO teleportation between segments:
       a. si==0 only: teleport to refined_path[0].  For si>0: arm is at
          the actual physics position from the previous step.
       b. q_start  = env.robot.get_joint_angles()  (actual current position).
       c. obs_before = env._get_obs()
       d. Compute Jacobian action from actual q_start toward next planned q_tgt.
       e. env.step(action)  ← REAL action, arm moves continuously through physics.
       f. obs_after = observation returned by env.step().
       g. Record (obs_before, action, reward, done, obs_after).
  5. Runs a short cartesian goal-correction pass at the end (pure physics).
  6. Validates and saves to output/agent_data/<run_id>.hdf5.

Why no teleportation between segments?
  Because the RL agent never teleports at inference time.  By generating data
  the same way the agent runs, the stored (obs, action, next_obs) triples chain
  together exactly — replaying them through env.step() reproduces the identical
  trajectory with matching rewards and no self-collision drift.

Run from repository root (venv activated):

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_2.py \\
        --run-id 20260504_203032

Process only one group:

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_2.py \\
        --run-id 20260504_203032 --group-key <key>

Tip — defaults match the smoother “config 1” baseline (step_scale=0.3, alpha=0.55, dq_cap=0.03).

Experimental longer / stricter correction (optional):

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_2.py \\
        --run-id 20260504_203032 \\
        --final-correction-steps 45 \\
        --final-correction-divergence-min-corr-idx 2

Check what was saved:

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_2.py \\
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
from scipy.spatial.transform import Rotation
from zenlog import log

# ── resolve roots and insert into sys.path ────────────────────────────────────
_FP_ROOT = Path(__file__).resolve().parent       # feature_apple_path_planning/
_REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root
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

from feature_apple_path_planning.apple_picking_env import ApplePickingEnv
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


def print_terminal_run_report(rows: list[dict]) -> None:
    """Brief one-screen summary for tee’d logs; complements zenlog file output."""
    if not rows:
        return
    print("\n" + "=" * 96, flush=True)
    print(
        f"{'#':>3}  {'HDF5 group':<22}  {'status':<18}  {'succ':>5}  "
        f"{'tr':>4}  {'r_min':>7}  {'r_mean':>7}  {'goals':>5}  {'vis':>5}  correction",
        flush=True,
    )
    print("-" * 96, flush=True)
    for i, r in enumerate(rows):
        gk = str(r.get("group_key", ""))[:20]
        st = str(r.get("status", ""))[:16]
        succ = r.get("success")
        succ_s = "yes" if succ is True else ("no" if succ is False else "—")
        nt = r.get("transitions")
        nt_s = str(nt) if nt is not None else "—"
        rmin = r.get("r_min")
        rmn = r.get("r_mean")
        rmin_s = f"{rmin:+.2f}" if isinstance(rmin, (int, float)) and np.isfinite(rmin) else "—"
        rmn_s = f"{rmn:+.2f}" if isinstance(rmn, (int, float)) and np.isfinite(rmn) else "—"
        gc = r.get("goal_achieved_count")
        gc_s = str(gc) if gc is not None else "—"
        vf = r.get("visibility_frac")
        vf_s = f"{100.0 * float(vf):.0f}%" if vf is not None else "—"
        cor = str(r.get("correction_outcome", "—"))[:28]
        print(
            f"{i:3d}  {gk:<22}  {st:<18}  {succ_s:>5}  "
            f"{nt_s:>4}  {rmin_s:>7}  {rmn_s:>7}  {gc_s:>5}  {vf_s:>5}  {cor}",
            flush=True,
        )
    print("=" * 96 + "\n", flush=True)


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
) -> tuple[str, dict]:
    """
    Execute one waypoint group with the continuous physics controller and save.

    The arm is teleported ONCE to refined_path[0], then runs continuously:
        si==0  : teleport to refined_path[0] (bootstrap anchor)
        si > 0 : q_start = actual physics position (NO teleport)

    Each segment:
        1. q_start = actual current joint angles  (or refined_path[0] for si==0)
        2. obs_before = env._get_obs()
        3. Compute Jacobian action from actual q_start toward next planned q_tgt
        4. obs_after, reward, done = env.step(action)  ← REAL action, real physics
        5. Record (obs_before, action, reward, done, obs_after)

    No teleportation between segments means the transitions chain continuously,
    exactly as the RL agent will experience at inference time.

    Returns (status, report_row) where status is
    "saved" | "skipped:<reason>" | "error:<msg>" and report_row is a dict for the
    terminal summary table.
    """
    def _summary(status: str, **extra: object) -> tuple[str, dict]:
        return status, {"group_key": group_key, "status": status, **extra}

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
    final_correction_note = "not_run"

    # Bootstrap: reset env, set apple target, teleport arm to path start
    bootstrap_rollout(env, metadata, refined_path)


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

    # ── continuous physics-step segment loop ──────────────────────────────────
    #
    # Segment order:
    #   1. For si==0 only: teleport to refined_path[0] (bootstrap anchor).
    #      For si>0: arm is at the physics position from the previous step —
    #      NO teleport.  This is the key fix: data is generated the same way
    #      the RL agent will run at inference, so replay matches generation.
    #   2. q_start = actual current joint angles (physics position, not planned).
    #   3. obs_before = env._get_obs()   (sets prev_obs_info = current EE pos)
    #   4. Jacobian action from actual q_start toward next planned q_tgt.
    #   5. env.step(action) — arm moves continuously through physics.
    #   6. obs_after = observation from env.step() (actual physics result).
    #   7. Record (obs_before, action, reward, done, obs_after).
    #
    # Because the arm never teleports between steps, chaining all stored
    # (obs, action, next_obs) triples in replay reproduces the exact same
    # trajectory — no self-collision drift, matching rewards.
    # ─────────────────────────────────────────────────────────────────────────
    for si in range(num_segments):
        if terminated or truncated:
            break

        q_tgt = np.asarray(refined_path[si + 1], dtype=np.float64)

        # Step 1 — only teleport on the very first segment (bootstrap anchor).
        # For si > 0 the arm is already at the physics-result of the previous step.
        if si == 0:
            q_start = np.asarray(refined_path[0], dtype=np.float64)
            env.robot.set_joint_angles_no_collision(q_start.tolist())
        else:
            # Actual current joint angles — the arm has drifted slightly from the
            # planned path, and the action is computed from where it actually is.
            q_start = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)

        dq_true = float(np.linalg.norm(q_tgt - q_start))

        # EE position at actual current state (for goal-tolerance check and tracking)
        _ee_pos_q_start, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        ee_pos_q_start = np.asarray(_ee_pos_q_start, dtype=np.float64)

        # Early-exit if EE is already within goal tolerance.
        dist_at_q_start = float(np.linalg.norm(ee_pos_q_start - goal_pos_arr))
        if dist_at_q_start <= float(args.final_goal_pos_tol) and si > 0:
            log.info(
                "segment %s/%s: arm within goal tol at current position"
                " (dist=%.4fm ≤ %.4fm) — stopping, marking prev done=True",
                si + 1, num_segments, dist_at_q_start, args.final_goal_pos_tol,
            )
            buffers["dones"][-1] = True
            break

        # Step 2 — observation at current state
        # (sets env.prev_observation_info[achieved_pos] = current EE pos)
        obs_before = env._get_obs()
        obs_before = dg.ensure_optical_flow(obs_before, optical_flow_model)

        # Step 3 — Jacobian action from actual current position toward next waypoint.
        # compute_action calls env.robot.calculate_jacobian() which uses the live
        # robot state, so the Jacobian is correct for the actual (drifted) position.
        action_applied = compute_action(
            env, q_start, q_tgt,
            control_freq, max_ee_vel, dq_cap,
            float(args.step_scale), prev_action, alpha,
        )

        # Step 4 — continuous physics step with the REAL action.
        # No teleportation. The arm moves from its current position through the
        # full DLS Jacobian → joint-velocity → PyBullet physics pipeline.
        # Replaying this action at inference will produce the same transition.
        obs_after_raw, reward, terminated, truncated, info = env.step(
            action_applied.astype(np.float64)
        )

        # Step 5 — observation from the physics result (actual position, no reset)
        obs_after = dg.ensure_optical_flow(obs_after_raw, optical_flow_model)

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
        # EE position after physics step (actual position, not planned q_tgt)
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
        log.debug(
            "si=%s/%s  ee_pos=%s  joint_angles=%s",
            si + 1, num_segments,
            np.round(ee_pos_tgt, 4).tolist(),
            np.round(env.robot.get_joint_angles(), 4).tolist(),
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
    div_min_corr_idx = max(1, int(args.final_correction_divergence_min_corr_idx))

    if not terminated and not truncated and max_corr > 0:
        goal_pos = np.asarray(metadata["goal_pos"], dtype=np.float64)
        control_time = float(dg.CONFIG["simulation_setup"]["control_time"])

        # Starting error for the correction pass
        _ee0, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        corr_start_err = float(np.linalg.norm(goal_pos - np.asarray(_ee0)))

        env_goal_tol = float(dg.CONFIG['planning'].get('fk_pos_tolerance', 0.05))
        max_corr_radius = max(env_goal_tol, float(args.final_goal_pos_tol)) * 5.0

        if corr_start_err > max_corr_radius:
            final_correction_note = (
                f"skipped_far(start_err>{max_corr_radius:.2f}m)"
            )
            log.info(
                "final_correction skipped summary  key=%s  note=%s  start_err=%.4fm  max_radius=%.4fm",
                group_key, final_correction_note, corr_start_err, max_corr_radius,
            )
        else:
            log.info(
                "final_correction  max_steps=%s  goal_tol=%.4f  start_err=%.4f"
                "  diverge_min_idx=%s",
                max_corr, args.final_goal_pos_tol, corr_start_err, div_min_corr_idx,
            )

            corr_transitions_added = 0
            final_correction_note = "in_progress"
            corr_break = "exhausted"
            for corr_idx in range(1, max_corr + 1):
                ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
                ee_pos = np.asarray(ee_pos, dtype=np.float64)
                err_before_step = float(np.linalg.norm(goal_pos - ee_pos))

                if err_before_step <= float(args.final_goal_pos_tol):
                    if buffers["dones"]:
                        buffers["dones"][-1] = True
                    final_correction_note = (
                        f"at_goal(idx={corr_idx},err={err_before_step:.4f}m,no_step)"
                    )
                    corr_break = "at_goal"
                    log.info(
                        "final_correction arm at goal  step=%s  err=%.5f"
                        "  — marking last transition done=True  (no env.step)",
                        corr_idx, err_before_step,
                    )
                    break

                # Divergence vs. the error at the *start* of the whole correction pass.
                # Require corr_idx >= div_min_corr_idx so one noisy Euler step cannot
                # abort when error briefly rises before the next step improves (see logs).
                if (
                    corr_idx >= div_min_corr_idx
                    and err_before_step > corr_start_err * 1.10
                ):
                    final_correction_note = (
                        f"diverged(idx={corr_idx},discarded={corr_transitions_added})"
                    )
                    corr_break = "diverged"
                    log.warning(
                        "final_correction diverging at step %s  err=%.5f > start=%.5f*1.10=%.5f"
                        " — discarding all %s correction transition(s) added so far",
                        corr_idx, err_before_step, corr_start_err, corr_start_err * 1.10,
                        corr_transitions_added,
                    )
                    for _ in range(corr_transitions_added):
                        for buf in ("observations", "next_observations", "actions",
                                    "rewards", "dones"):
                            if buffers[buf]:
                                buffers[buf].pop()
                    break

                obs_before_corr = env._get_obs()
                obs_before_corr = dg.ensure_optical_flow(obs_before_corr, optical_flow_model)

                ee_vel_global = np.zeros(6, dtype=np.float64)
                error_vec = goal_pos - ee_pos
                error_norm = np.linalg.norm(error_vec)
                if error_norm > 1e-6:
                    approach_vel = min(error_norm / max(control_time, 1e-6), max_ee_vel)
                    ee_vel_global[:3] = (error_vec / error_norm) * approach_vel
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
                corr_transitions_added += 1
                if bool(info.get("goal_achieved", False)):
                    buffers["goal_achieved_count"] += 1
                if truncated:
                    buffers["truncated_flag"] = True
                prev_action = np.asarray(action_corr, dtype=np.float64)

                log.info(
                    "final_correction %s/%s  err=%.5f  reward=%.5f  terminated=%s",
                    corr_idx, max_corr, err_before_step, reward, terminated,
                )
                log.debug(
                    "final_correction %s/%s  ee_pos=%s  joint_angles=%s",
                    corr_idx, max_corr,
                    np.round(ee_pos, 4).tolist(),
                    np.round(env.robot.get_joint_angles(), 4).tolist(),
                )
                if done_flag:
                    final_correction_note = (
                        f"env_done(idx={corr_idx},err_pre_step={err_before_step:.4f}m)"
                    )
                    corr_break = "env_done"
                    log.info(
                        "final_correction done  step=%s  err_pre_step=%.5f",
                        corr_idx, err_before_step,
                    )
                    break
            if final_correction_note == "in_progress":
                final_correction_note = f"exhausted_added={corr_transitions_added}"
            log.info(
                "final_correction summary  key=%s  outcome=%s  break=%s  start_err=%.4fm",
                group_key, final_correction_note, corr_break, corr_start_err,
            )
    elif max_corr <= 0:
        final_correction_note = "disabled(steps=0)"
    elif terminated or truncated:
        final_correction_note = "skipped(episode_terminated_or_truncated)"

    # ── quality gates ─────────────────────────────────────────────────────────
    success = bool(buffers["dones"] and buffers["dones"][-1])
    n_tr = len(buffers["actions"])
    reward_np_gate = (
        np.asarray(buffers["rewards"], dtype=np.float64) if n_tr else np.array([])
    )
    count_vis_gate = int(
        sum(
            1 for o in buffers["next_observations"]
            if "point_mask" in o and np.any(o["point_mask"] > 0)
        )
    ) if n_tr else 0
    vis_frac_gate = count_vis_gate / n_tr if n_tr else None

    if buffers["truncated_flag"]:
        log.warning("skip save  key=%s  reason=truncated", group_key)
        return _summary(
            "skipped:truncated",
            correction_outcome=final_correction_note,
            success=success,
            transitions=n_tr,
            r_min=float(np.min(reward_np_gate)) if n_tr else None,
            r_mean=float(np.mean(reward_np_gate)) if n_tr else None,
            goal_achieved_count=int(buffers["goal_achieved_count"]),
            visibility_frac=vis_frac_gate,
        )

    valid, reason = validate_transition_buffers(buffers, env)
    if not valid:
        log.error("skip save  key=%s  invalid_buffers(%s)", group_key, reason)
        return _summary(
            f"skipped:invalid_buffers({reason})",
            correction_outcome=final_correction_note,
            success=success,
            transitions=n_tr,
            r_min=float(np.min(reward_np_gate)) if n_tr else None,
            r_mean=float(np.mean(reward_np_gate)) if n_tr else None,
            goal_achieved_count=int(buffers["goal_achieved_count"]),
            visibility_frac=vis_frac_gate,
        )

    if args.save_only_success and not success:
        log.info("skip save  key=%s  reason=not_success (--save-only-success)", group_key)
        return _summary(
            "skipped:not_success",
            correction_outcome=final_correction_note,
            success=False,
            transitions=n_tr,
            r_min=float(np.min(reward_np_gate)) if n_tr else None,
            r_mean=float(np.mean(reward_np_gate)) if n_tr else None,
            goal_achieved_count=int(buffers["goal_achieved_count"]),
            visibility_frac=vis_frac_gate,
        )

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
        return _summary(
            f"skipped:low_visibility({visibility_frac:.2f}<{min_vis:.2f})",
            correction_outcome=final_correction_note,
            success=success,
            transitions=n_transitions,
            r_min=float(np.min(reward_np_gate)) if n_transitions > 0 else None,
            r_mean=float(np.mean(reward_np_gate)) if n_transitions > 0 else None,
            goal_achieved_count=int(buffers["goal_achieved_count"]),
            visibility_frac=visibility_frac,
        )

    save_info = {
        "count_in_frame": count_in_frame,
        "path_length": int(len(buffers["actions"])),
        "source_group_key": str(group_key),
        "controller_mode": "continuous_physics_rollout",
        "goal_achieved_count": int(buffers["goal_achieved_count"]),
        "truncated_flag": bool(buffers["truncated_flag"]),
    }
    enc = np.asarray(buffers["observations"][0]["joint_angles"], dtype=np.float64).reshape(-1)
    half = len(enc) // 2
    save_info["initial_joint_angles"] = np.arctan2(enc[:half], enc[half:]).astype(np.float64)

    tree_info = {k: _hdf5_safe_attr_value(v) for k, v in metadata.items()}

    robot_conf = dg.CONFIG["robot_setup"]
    robot_pos_save = np.asarray(robot_conf["start_position"], dtype=np.float64)
    robot_or_save = Rotation.from_euler(
        "xyz", robot_conf["start_orientation_euler_deg"], degrees=True
    ).as_quat()

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
            robot_pos_save,
            robot_or_save,
            str(agent_data_path),
        )
    except Exception as exc:
        log.error("save error  key=%s  err=%s", group_key, exc, exc_info=True)
        return _summary(
            f"error:{exc}",
            correction_outcome=final_correction_note,
            success=success,
            transitions=n_transitions,
            r_min=float(np.min(reward_np_gate)) if n_transitions > 0 else None,
            r_mean=float(np.mean(reward_np_gate)) if n_transitions > 0 else None,
            goal_achieved_count=int(buffers["goal_achieved_count"]),
            visibility_frac=visibility_frac if n_transitions > 0 else None,
        )

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
    return _summary(
        "saved",
        correction_outcome=final_correction_note,
        success=success,
        transitions=len(buffers["actions"]),
        r_min=float(np.min(reward_np)),
        r_mean=float(np.mean(reward_np)),
        goal_achieved_count=int(buffers["goal_achieved_count"]),
        visibility_frac=visibility_frac,
    )


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
        "--step-scale", type=float, default=0.3,
        help="EE-velocity action scale after Jacobian conversion (default 0.3).",
    )
    p.add_argument(
        "--action-smoothing-alpha", type=float, default=0.55,
        help="EMA weight on previous action [0,1]. 0=no smoothing (default 0.55, config-1 baseline).",
    )
    p.add_argument(
        "--record-joint-dq-cap", type=float, default=0.03,
        help="Per-joint dq clamp before Jacobian in rad (default 0.03, config-1 baseline).",
    )
    p.add_argument(
        "--final-correction-steps", type=int, default=30,
        help="Cartesian goal-correction steps after path end, pure physics (default 30).",
    )
    p.add_argument(
        "--final-correction-divergence-min-corr-idx",
        type=int,
        default=3,
        help=(
            "Apply the 10%% divergence abort only when correction iteration index "
            ">= this (default 3). Use 2 for stricter / legacy-like early abort."
        ),
    )
    p.add_argument(
        "--final-goal-pos-tol",
        type=float,
        default=float(dg.CONFIG["planning"].get("fk_pos_tolerance", 0.05)),
        help=(
            "Stop correction once EE-to-goal distance <= this (metres). "
            "Default matches CONFIG planning fk_pos_tolerance so env goal threshold aligns."
        ),
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

    try:
        dg.apply_tree_setup_from_waypoints_hdf5(
            waypoints_path, ref_group_key=args.group_key
        )
    except (ValueError, KeyError, OSError) as exc:
        log.error("Cannot auto-apply tree_setup from waypoints HDF5: %s", exc)
        return

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
        "final_correction_steps=%s  final_corr_div_min_idx=%s  "
        "save_only_success=%s  min_visibility=%.2f",
        args.step_scale, args.action_smoothing_alpha, args.record_joint_dq_cap,
        args.final_correction_steps, args.final_correction_divergence_min_corr_idx,
        args.save_only_success, args.min_visibility,
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
    if bool(env.robot.robot_conf.get("disable_self_collisions", False)):
        log.warning(
            "Sanity: robot.yaml has disable_self_collisions=true — PyBullet often "
            "suppresses robot–robot contact points, so env reward/termination may "
            "NOT flag self-collision even if links visually intersect. "
            "Environmental (tree) collisions still use the tree body checks."
        )
    control_freq = 1.0 / dg.CONFIG["simulation_setup"]["control_time"]
    max_ee_vel = dg.CONFIG["planning"]["max_ee_velocity"]
    planner_start = np.asarray(
        dg.CONFIG["robot_setup"]["start_joint_angles"], dtype=np.float64
    )

    log.info(
        "Env ready.  control_freq=%.1f Hz  max_ee_vel=%.3f  planner_start=%s",
        control_freq, max_ee_vel, np.round(planner_start, 4).tolist(),
    )

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
    report_rows: list[dict] = []
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
            already_refined = bool(grp.attrs.get("waypoints_saved_refined", False))

        if len(waypoints) < 2:
            log.warning("Group %r has fewer than 2 waypoints — skipping.", group_key)
            stats["skipped"] += 1
            continue

        # refine path (skip second smooth_path if planner already saved refined waypoints)
        try:
            _do_smooth = dg.CONFIG["planning"].get("enable_smoothing", True)
            if already_refined:
                _do_smooth = False
            refined_path = dg.shortcut_and_refine_path(
                env.robot,
                waypoints.tolist(),
                extend_fn,
                collision_fn,
                dg.CONFIG["planning"]["task_space_refinement_threshold"],
                enable_smoothing=_do_smooth,
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
            status, row = run_group(
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
            row = {
                "group_key": group_key,
                "status": status,
                "correction_outcome": "—",
            }

        elapsed = time.time() - t0_group
        row["elapsed_s"] = round(elapsed, 1)
        report_rows.append(row)
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
    print_terminal_run_report(report_rows)


if __name__ == "__main__":
    main()
