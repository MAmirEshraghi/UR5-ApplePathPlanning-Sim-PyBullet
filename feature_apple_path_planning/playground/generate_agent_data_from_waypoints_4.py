#!/usr/bin/env python3
"""
generate_agent_data_from_waypoints_4.py — joint-space waypoint execution (v4)

Batch-generate RL agent-data from a pre-planned waypoints HDF5.

Default ``--execution-mode joint`` subdivides each planner edge into steps of at
most ``--max-joint-step`` radians and calls ``ApplePickingEnv.step_joint_delta``
each substep (obs/action/reward/done recorded every substep).

Legacy ``--execution-mode cartesian`` keeps the v2 Jacobian / one-env.step-per-edge
behavior.  With ``--joint-final-goal-correction``, optional post-path closing uses
``step_joint_delta``.  By default this uses **PyBullet IK** toward ``goal_pos`` with
current EE orientation held; use ``--final-correction-method jacobian`` for the
incremental position-only Jacobian path.  The same closing logic runs after
``--execution-mode cartesian`` segments.

Logs: logs/data_generator_path_planner/<timestamp>_agent_data_gen_v4.log

Each saved trajectory uses an HDF5 group name equal to the waypoints ``group``
key (``source_group_key``), so logs and replay lists match the planner file.
Appending twice to the same agent-data file adds ``__dup1``, ``__dup2``, … if
needed.

By default only **successful** episodes are written; pass ``--no-save-only-success``
to keep failed runs too.

Examples::

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_4.py \\
        --run-id 20260504_203032

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_4.py \\
        --run-id 20260504_203032 --max-joint-step 0.02 --validate-each-step --debug-log

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_4.py \\
        --run-id 20260504_203032 --log-transition-diagnostics

    PYTHONPATH=. python feature_apple_path_planning/playground/generate_agent_data_from_waypoints_4.py \\
        --run-id 20260504_203032 --execution-mode cartesian
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
    log_path = (base / f"{ts}_agent_data_gen_v4.log").resolve()

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

    log.info("Agent-data v4 log: %s  (debug_log=%s)", log_path, debug)
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


def validate_transition_buffers(
    buffers: dict,
    env: ApplePickingEnv,
    action_dim: int | None = None,
) -> tuple[bool, str]:
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
    adim = int(env.action_space.shape[0]) if action_dim is None else int(action_dim)
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
        if act.shape[0] != adim:
            return False, f"action dim mismatch at step {i}: got {act.shape[0]}, expected {adim}"
        if not _is_finite_array(act):
            return False, f"non-finite action at step {i}"
        if not np.isfinite(float(rews[i])):
            return False, f"non-finite reward at step {i}"
    return True, "ok"


def subdivide_joint_segment(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    max_joint_step: float,
) -> list[np.ndarray]:
    """Return ordered joint-space samples on the line segment q_start → q_goal."""
    q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    delta = q_goal - q_start
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-12:
        return [q_start.copy(), q_goal.copy()]
    n = max(1, int(np.ceil(dist / max(float(max_joint_step), 1e-9))))
    out: list[np.ndarray] = []
    for i in range(n + 1):
        t = i / n
        out.append(q_start + t * delta)
    return out


def clip_joint_delta_vector(delta: np.ndarray, max_norm: float) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(d))
    cap = max(float(max_norm), 1e-12)
    if n <= cap:
        return d
    return d * (cap / n)


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


def compute_joint_delta_toward_goal_world(
    env: ApplePickingEnv,
    goal_pos: np.ndarray,
    *,
    control_time: float,
    max_ee_vel: float,
    step_scale: float,
    max_joint_step: float,
    dls_damping: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Map world-frame linear motion toward ``goal_pos`` into a joint Δq for one
    ``control_time`` horizon using **position-only** damped least squares on the
    top three rows of the spatial Jacobian (linear velocity block). Desired
    tool linear velocity points from current EE position toward ``goal_pos`` in
    world frame and is capped; orientation is not constrained.

    ``max_joint_step`` clips the L2 norm of Δq (same as joint waypoint substeps).
    """
    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    error_vec = goal_pos - ee_pos
    error_norm = float(np.linalg.norm(error_vec))

    dt = max(float(control_time), 1e-9)
    vel_cap = float(max_ee_vel) * max(0.2, min(1.0, float(step_scale) * 1.5))
    v_des = np.zeros(3, dtype=np.float64)
    if error_norm > 1e-9:
        u = error_vec / error_norm
        approach_speed = min(error_norm / dt, vel_cap)
        v_des = u * approach_speed

    jacobian = np.asarray(env.robot.calculate_jacobian(), dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] < 3:
        raise ValueError(f"expected spatial Jacobian with ≥3 rows, got {jacobian.shape}")
    j_lin = jacobian[:3, :]
    lam = float(dls_damping)
    m = np.asarray(j_lin @ j_lin.T + (lam ** 2) * np.eye(3, dtype=np.float64), dtype=np.float64)
    joint_vel = j_lin.T @ np.linalg.solve(m, v_des.astype(np.float64))
    joint_vel = np.asarray(joint_vel, dtype=np.float64).reshape(-1)

    delta_raw = joint_vel * dt
    delta = clip_joint_delta_vector(delta_raw, float(max_joint_step))

    dbg = {
        "approach_speed": float(np.linalg.norm(v_des)),
        "vel_cap": float(vel_cap),
        "error_norm": error_norm,
        "delta_raw_norm": float(np.linalg.norm(delta_raw)),
        "delta_norm": float(np.linalg.norm(delta)),
    }
    return delta, dbg


def compute_joint_delta_toward_goal_ik(
    env: ApplePickingEnv,
    goal_pos: np.ndarray,
    *,
    max_joint_step: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    One increment toward ``goal_pos`` using PyBullet inverse kinematics for tool0.
    Target orientation is the **current** EE quaternion so only position is
    driven.  ``q_ik - q`` is L2-clipped by ``max_joint_step`` (same as waypoints).
    """
    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    ee_pos, ee_orient = env.robot.get_current_pose(env.robot.tool0_link_idx)
    err_norm = float(np.linalg.norm(goal_pos - np.asarray(ee_pos, dtype=np.float64)))

    q_ik_t = env.robot.calculate_ik(
        goal_pos.tolist(),
        ee_orient,
    )
    q_ik = np.asarray(q_ik_t, dtype=np.float64).reshape(-1)
    q_curr = np.asarray(env.robot.get_joint_angles(), dtype=np.float64).reshape(-1)
    if q_ik.size != q_curr.size:
        raise ValueError(
            f"IK returned length {q_ik.size}, expected {q_curr.size} control joints"
        )
    delta_raw = q_ik - q_curr
    delta = clip_joint_delta_vector(delta_raw, float(max_joint_step))
    dbg = {
        "approach_speed": float("nan"),
        "vel_cap": float("nan"),
        "error_norm": err_norm,
        "delta_raw_norm": float(np.linalg.norm(delta_raw)),
        "delta_norm": float(np.linalg.norm(delta)),
    }
    return delta, dbg


def _run_final_goal_correction_joint_delta(
    *,
    env: ApplePickingEnv,
    goal_pos: np.ndarray,
    buffers: dict,
    ee_dists: list,
    ee_step_displacements: list,
    optical_flow_model,
    args: argparse.Namespace,
    max_corr: int,
    validate_each_step: bool,
    state_invalid_fn,
    global_sub_mut: list[int] | None,
    style: str,
    correction_method: str,
) -> tuple[bool, bool]:
    """
    Shared post-path goal correction → Δq → ``step_joint_delta``.

    ``correction_method``: ``"ik"`` (PyBullet IK toward position, fixed orient) or
    ``"jacobian"`` (position-only DLS increment).

    ``style`` is ``"joint"`` or ``"cartesian"`` (log prefixes only).
    ``global_sub_mut`` is ``[global_sub]`` for joint rollouts so substeps stay
    consistent when correction transitions are rolled back; ``None`` for cartesian.
    Returns updated ``(terminated, truncated)``.
    """
    assert style in ("joint", "cartesian")
    assert correction_method in ("ik", "jacobian")
    joint_like = style == "joint"
    hdr = "joint: final_correction" if joint_like else "final_correction"
    pre = "joint final_correction" if joint_like else "final_correction"

    control_time = float(dg.CONFIG["simulation_setup"]["control_time"])
    max_ee_vel = float(dg.CONFIG["planning"]["max_ee_velocity"])
    dls_damping = float(dg.CONFIG["planning"].get("jacobian_dls_damping", 0.05))

    goal_pos = np.asarray(goal_pos, dtype=np.float64)
    _ee0, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
    corr_start_err = float(np.linalg.norm(goal_pos - np.asarray(_ee0)))

    env_goal_tol = float(dg.CONFIG["planning"].get("fk_pos_tolerance", 0.05))
    max_corr_radius = max(env_goal_tol, float(args.final_goal_pos_tol)) * 5.0

    terminated = False
    truncated = False

    if corr_start_err > max_corr_radius:
        log.info(
            "%s skipped: start_err=%.4fm > max_radius=%.4fm",
            hdr,
            corr_start_err,
            max_corr_radius,
        )
        return terminated, truncated

    diverge_lim_hdr = corr_start_err * 1.10
    method_tag = (
        "PyBullet IK (hold orient) → clip Δq"
        if correction_method == "ik"
        else "position-only J_lin→Δq"
    )
    log.info(
        "%s (%s / step_joint_delta)  method=%s  max_steps=%s  "
        "start_err=%.5fm  diverge_lim=start_err*1.10=%.5fm  goal_tol=%.5fm  "
        "step_scale=%.4f  control_time=%.4f  max_ee_vel=%.4f  max_joint_step=%.5f",
        hdr,
        method_tag,
        correction_method,
        max_corr,
        corr_start_err,
        diverge_lim_hdr,
        float(args.final_goal_pos_tol),
        float(args.step_scale),
        control_time,
        max_ee_vel,
        float(args.max_joint_step),
    )

    q_before_final_corr = np.asarray(env.robot.get_joint_angles(), dtype=np.float64).copy()
    step_counter_before_corr = int(env.step_counter)
    corr_transitions_added = 0

    for corr_idx in range(1, max_corr + 1):
        ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        err_norm = float(np.linalg.norm(goal_pos - ee_pos))
        diverge_lim = corr_start_err * 1.10
        log.info(
            "%s pre-step  iter=%s/%s  err=%.5fm  start_err=%.5fm  "
            "diverge_if_err>=%.5fm  goal_tol=%.5fm",
            pre,
            corr_idx,
            max_corr,
            err_norm,
            corr_start_err,
            diverge_lim,
            float(args.final_goal_pos_tol),
        )

        if err_norm <= float(args.final_goal_pos_tol):
            if buffers["dones"]:
                buffers["dones"][-1] = True
            if joint_like:
                log.info("%s at goal  step=%s  err=%.5f", pre, corr_idx, err_norm)
            else:
                log.info(
                    "%s arm at goal  step=%s  err=%.5f  — marking last transition done=True  (no step)",
                    pre,
                    corr_idx,
                    err_norm,
                )
            terminated = True
            break

        if err_norm > diverge_lim:
            log.warning(
                "%s diverging at iter=%s  err=%.5fm > diverge_lim=%.5fm "
                "(start_err=%.5fm * 1.10) — discarding %s correction transition(s); "
                "restoring pre-correction joints",
                pre,
                corr_idx,
                err_norm,
                diverge_lim,
                corr_start_err,
                corr_transitions_added,
            )
            for _ in range(corr_transitions_added):
                for buf in ("observations", "next_observations", "actions", "rewards", "dones"):
                    if buffers[buf]:
                        buffers[buf].pop()
            for _ in range(corr_transitions_added):
                if ee_dists:
                    ee_dists.pop()
                if ee_step_displacements:
                    ee_step_displacements.pop()
            if global_sub_mut is not None:
                global_sub_mut[0] -= int(corr_transitions_added)
            env.robot.set_joint_angles_no_collision(np.asarray(q_before_final_corr, dtype=np.float64).tolist())
            env.step_counter = step_counter_before_corr
            env._get_obs()
            terminated = False
            truncated = False
            break

        if correction_method == "ik":
            delta, dbg = compute_joint_delta_toward_goal_ik(
                env,
                goal_pos,
                max_joint_step=float(args.max_joint_step),
            )
        else:
            delta, dbg = compute_joint_delta_toward_goal_world(
                env,
                goal_pos,
                control_time=control_time,
                max_ee_vel=max_ee_vel,
                step_scale=float(args.step_scale),
                max_joint_step=float(args.max_joint_step),
                dls_damping=dls_damping,
            )
        if float(dbg["delta_norm"]) < 1e-10:
            log.info("%s: negligible Δq at iter=%s — stopping", pre, corr_idx)
            break

        if (
            validate_each_step
            and state_invalid_fn is not None
            and state_invalid_fn(tuple(np.asarray(env.robot.get_joint_angles(), dtype=np.float64).tolist()))
        ):
            log.warning("%s: planner-invalid pose before substep iter=%s — rollback", pre, corr_idx)
            for _ in range(corr_transitions_added):
                for buf in ("observations", "next_observations", "actions", "rewards", "dones"):
                    if buffers[buf]:
                        buffers[buf].pop()
            for _ in range(corr_transitions_added):
                if ee_dists:
                    ee_dists.pop()
                if ee_step_displacements:
                    ee_step_displacements.pop()
            if global_sub_mut is not None:
                global_sub_mut[0] -= int(corr_transitions_added)
            env.robot.set_joint_angles_no_collision(np.asarray(q_before_final_corr, dtype=np.float64).tolist())
            env.step_counter = step_counter_before_corr
            env._get_obs()
            terminated = False
            truncated = False
            break

        if correction_method == "ik":
            log.info(
                "%s command  iter=%s  method=ik ||q_ik-q||=%.5f  ||dq||=%.5f  err=%.5fm",
                pre,
                corr_idx,
                dbg["delta_raw_norm"],
                dbg["delta_norm"],
                dbg["error_norm"],
            )
        else:
            log.info(
                "%s command  iter=%s  method=jacobian  approach_speed=%.5f  vel_cap=%.5f  "
                "||dq||=%.5f  ||dq_raw||=%.5f",
                pre,
                corr_idx,
                dbg["approach_speed"],
                dbg["vel_cap"],
                dbg["delta_norm"],
                dbg["delta_raw_norm"],
            )

        _ee_pos_q_start = np.asarray(
            env.robot.get_current_pose(env.robot.tool0_link_idx)[0],
            dtype=np.float64,
        )
        obs_before_corr = env._get_obs()
        obs_before_corr = dg.ensure_optical_flow(obs_before_corr, optical_flow_model)

        obs_after_corr, reward, term_step, trunc_step, info = env.step_joint_delta(delta.astype(np.float64))
        obs_after_corr = dg.ensure_optical_flow(obs_after_corr, optical_flow_model)
        done_flag = bool(term_step or trunc_step)
        terminated = term_step
        truncated = trunc_step

        if validate_each_step and state_invalid_fn is not None:
            q_after = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
            if state_invalid_fn(tuple(q_after.tolist())):
                log.warning("%s: planner-invalid after substep iter=%s — rollback", pre, corr_idx)
                for _ in range(corr_transitions_added):
                    for buf in ("observations", "next_observations", "actions", "rewards", "dones"):
                        if buffers[buf]:
                            buffers[buf].pop()
                for _ in range(corr_transitions_added):
                    if ee_dists:
                        ee_dists.pop()
                    if ee_step_displacements:
                        ee_step_displacements.pop()
                if global_sub_mut is not None:
                    global_sub_mut[0] -= int(corr_transitions_added)
                env.robot.set_joint_angles_no_collision(np.asarray(q_before_final_corr, dtype=np.float64).tolist())
                env.step_counter = step_counter_before_corr
                env._get_obs()
                terminated = False
                truncated = False
                break

        buffers["observations"].append(obs_before_corr)
        buffers["next_observations"].append(obs_after_corr)
        buffers["actions"].append(np.asarray(delta, dtype=np.float32))
        buffers["rewards"].append(float(reward))
        buffers["dones"].append(done_flag)
        corr_transitions_added += 1
        if global_sub_mut is not None:
            global_sub_mut[0] += 1
        if bool(info.get("goal_achieved", False)):
            buffers["goal_achieved_count"] += 1
        if truncated:
            buffers["truncated_flag"] = True

        _ee2, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        err_after = float(np.linalg.norm(goal_pos - np.asarray(_ee2, dtype=np.float64)))
        ee_step_m = float(np.linalg.norm(np.asarray(_ee2, dtype=np.float64) - _ee_pos_q_start))
        ee_dists.append(err_after)
        ee_step_displacements.append(ee_step_m)

        log.info(
            "%s post-step  iter=%s  err_pre=%.5fm  err_after=%.5fm  Δerr=%+.5fm  "
            "reward=%.5f  terminated=%s  truncated=%s  goal_achieved=%s  env_step_counter=%s",
            pre,
            corr_idx,
            err_norm,
            err_after,
            err_after - err_norm,
            float(reward),
            terminated,
            truncated,
            bool(info.get("goal_achieved", False)),
            env.step_counter,
        )
        log.debug(
            "%s %s/%s  ee_pos_pre=%s  ee_pos_post=%s  joint_angles=%s",
            pre,
            corr_idx,
            max_corr,
            np.round(ee_pos, 4).tolist(),
            np.round(np.asarray(_ee2, dtype=np.float64), 4).tolist(),
            np.round(np.asarray(env.robot.get_joint_angles(), dtype=np.float64), 4).tolist(),
        )

        if args.max_transitions > 0 and len(buffers["actions"]) >= int(args.max_transitions):
            truncated = True
            buffers["truncated_flag"] = True
            log.warning("%s: max_transitions cap reached  cap=%s", pre, int(args.max_transitions))
            break

        if done_flag:
            log.info("%s done  iter=%s  err_after=%.5fm", pre, corr_idx, err_after)
            break

    return terminated, truncated


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
# Per-group execution
# ─────────────────────────────────────────────────────────────────────────────

def run_group_joint(
    *,
    env: ApplePickingEnv,
    group_key: str,
    metadata: dict,
    refined_path: list,
    optical_flow_model,
    agent_data_path: Path,
    args: argparse.Namespace,
    state_invalid_fn,
) -> str:
    """
    Execute planner waypoints in joint space: subdivide each edge, record every substep.

    Recorded ``actions`` are ``joint_delta_rad`` (6,) applied to ``step_joint_delta``.
    """
    num_wp_segments = len(refined_path) - 1
    n_j = len(env.robot.control_joints)
    log.info(
        "GROUP START (JOINT)  key=%s  refined_len=%s  wp_segments=%s  max_joint_step=%.5f  validate_each_step=%s",
        group_key,
        len(refined_path),
        num_wp_segments,
        float(args.max_joint_step),
        bool(args.validate_each_step),
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

    bootstrap_rollout(env, metadata, refined_path)
    marker_ids: dict = {"apple": -1}
    refresh_markers(env, metadata, marker_ids)

    terminated = False
    truncated = False
    rollout_failure: str | None = None
    fail_segment: int | None = None
    fail_substep: int | None = None

    goal_pos_arr_start = np.asarray(metadata["goal_pos"], dtype=np.float64)
    ee_pos_start, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
    start_ee_dist = float(np.linalg.norm(np.asarray(ee_pos_start) - goal_pos_arr_start))
    ee_dists: list[float] = []
    ee_step_displacements: list[float] = []
    goal_pos_arr = np.asarray(metadata["goal_pos"], dtype=np.float64)

    global_sub = 0
    dist_to_goal_tol = float(args.final_goal_pos_tol)

    for si in range(num_wp_segments):
        if terminated or truncated or rollout_failure:
            break

        q_lane_a = np.asarray(refined_path[si], dtype=np.float64)
        q_lane_b = np.asarray(refined_path[si + 1], dtype=np.float64)
        subs = subdivide_joint_segment(q_lane_a, q_lane_b, float(args.max_joint_step))
        targets = subs[1:]

        if si == 0:
            env.robot.set_joint_angles_no_collision(q_lane_a.tolist())

        for k, target in enumerate(targets):
            if terminated or truncated or rollout_failure:
                break

            q_phys = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
            delta = clip_joint_delta_vector(target - q_phys, float(args.max_joint_step))

            _ee_pos_q_start, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
            ee_pos_q_start = np.asarray(_ee_pos_q_start, dtype=np.float64)
            dist_at_q_start = float(np.linalg.norm(ee_pos_q_start - goal_pos_arr))
            if dist_at_q_start <= dist_to_goal_tol and global_sub > 0:
                log.info(
                    "joint rollout: within goal tol (dist=%.4fm) — marking last done=True",
                     dist_at_q_start,
                )
                if buffers["dones"]:
                    buffers["dones"][-1] = True
                terminated = True
                break

            if float(np.linalg.norm(delta)) < 1e-10:
                continue

            if (
                args.validate_each_step
                and state_invalid_fn is not None
                and state_invalid_fn(tuple(q_phys.tolist()))
            ):
                rollout_failure = "planner_invalid_before_substep"
                fail_segment, fail_substep = si, k
                log.error(
                    "JOINT ROLLOUT ABORT  key=%s  reason=%s  segment=%s/%s  sub=%s  global_sub=%s  q=%s",
                    group_key,
                    rollout_failure,
                    si + 1,
                    num_wp_segments,
                    k,
                    global_sub,
                    np.round(q_phys, 4).tolist(),
                )
                break

            obs_before = env._get_obs()
            obs_before = dg.ensure_optical_flow(obs_before, optical_flow_model)

            obs_after_raw, reward, terminated, truncated, info = env.step_joint_delta(
                delta.astype(np.float64)
            )
            obs_after = dg.ensure_optical_flow(obs_after_raw, optical_flow_model)
            done_flag = bool(terminated or truncated)

            if args.validate_each_step and state_invalid_fn is not None:
                q_after = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
                if state_invalid_fn(tuple(q_after.tolist())):
                    rollout_failure = "planner_invalid_after_substep"
                    fail_segment, fail_substep = si, k
                    log.error(
                        "JOINT ROLLOUT ABORT  key=%s  reason=%s  segment=%s/%s  sub=%s  global_sub=%s  q_after=%s",
                        group_key,
                        rollout_failure,
                        si + 1,
                        num_wp_segments,
                        k,
                        global_sub,
                        np.round(q_after, 4).tolist(),
                    )
                    unstoppable, cdetails = env._check_collisions_with_planning_config()
                    if unstoppable:
                        log.error(
                            "collision detail  label=%s  self=%s",
                            cdetails.get("collided_obstacle_label"),
                            cdetails.get("is_self_collision_unacceptable"),
                        )
                    break

            action_record = np.asarray(delta, dtype=np.float32)

            buffers["observations"].append(obs_before)
            buffers["next_observations"].append(obs_after)
            buffers["actions"].append(action_record)
            buffers["rewards"].append(float(reward))
            buffers["dones"].append(done_flag)
            if bool(info.get("goal_achieved", False)):
                buffers["goal_achieved_count"] += 1
            if truncated:
                buffers["truncated_flag"] = True
                log.warning(
                    "joint rollout: env hit max_steps (truncated)  step_counter=%s/%s  "
                    "transitions=%s  seg=%s/%s  (raise simulation_setup.max_steps or "
                    "use shorter paths / larger --max-joint-step)",
                    env.step_counter,
                    env.max_steps,
                    len(buffers["actions"]),
                    si + 1,
                    num_wp_segments,
                )

            global_sub += 1

            _ee_pos_tgt, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
            ee_pos_tgt = np.asarray(_ee_pos_tgt, dtype=np.float64)
            ee_dist_to_goal = float(np.linalg.norm(ee_pos_tgt - goal_pos_arr))
            ee_step_m = float(np.linalg.norm(ee_pos_tgt - ee_pos_q_start))
            ee_dists.append(ee_dist_to_goal)
            ee_step_displacements.append(ee_step_m)

            if global_sub <= 3 or global_sub % 20 == 0:
                log.info(
                    "joint substep  seg=%s/%s  k=%s  gsub=%s  ||Δq||=%.5f  ee_dist=%.4f  rew=%+.4f  term=%s",
                    si + 1,
                    num_wp_segments,
                    k,
                    global_sub,
                    float(np.linalg.norm(delta)),
                    ee_dist_to_goal,
                    reward,
                    terminated or truncated,
                )

            if args.max_transitions > 0 and len(buffers["actions"]) >= int(args.max_transitions):
                truncated = True
                buffers["truncated_flag"] = True
                log.warning(
                    "joint rollout: max_transitions cap reached  cap=%s  env_step=%s/%s  "
                    "transitions=%s  (use --max-transitions 0 to disable)",
                    int(args.max_transitions),
                    env.step_counter,
                    env.max_steps,
                    len(buffers["actions"]),
                )
                break

            if rollout_failure:
                break

        if rollout_failure:
            break

    # Optional post-path goal correction (IK or Jacobian -> Delta q -> step_joint_delta).
    max_corr = max(0, int(args.final_correction_steps))
    if (
        rollout_failure is None
        and (not terminated)
        and (not truncated)
        and max_corr > 0
        and bool(args.joint_final_goal_correction)
    ):
        _g_sub_box = [global_sub]
        terminated, truncated = _run_final_goal_correction_joint_delta(
            env=env,
            goal_pos=np.asarray(metadata["goal_pos"], dtype=np.float64),
            buffers=buffers,
            ee_dists=ee_dists,
            ee_step_displacements=ee_step_displacements,
            optical_flow_model=optical_flow_model,
            args=args,
            max_corr=max_corr,
            validate_each_step=bool(args.validate_each_step),
            state_invalid_fn=state_invalid_fn,
            global_sub_mut=_g_sub_box,
            style="joint",
            correction_method=str(args.final_correction_method),
        )
        global_sub = _g_sub_box[0]

    _ee_fin, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
    fin_dist = float(np.linalg.norm(np.asarray(_ee_fin) - goal_pos_arr))

    success = bool(
        rollout_failure is None
        and (not buffers["truncated_flag"])
        and fin_dist <= float(env.distance_threshold)
    )

    if rollout_failure:
        log.error(
            "JOINT ROLLOUT FAILURE  key=%s  reason=%s  seg=%s  sub=%s  transitions=%s  final_dist=%.4fm",
            group_key,
            rollout_failure,
            fail_segment,
            fail_substep,
            len(buffers["actions"]),
            fin_dist,
        )
        success = False

    if getattr(args, "log_transition_diagnostics", False):
        log.info(
            "TRANSITION_DIAG  key=%s  mode=joint  transitions=%s  joint_substeps=%s  "
            "env.step_counter=%s  env.max_steps=%s  truncated_flag=%s  rollout_failure=%r  "
            "final_dist=%.4fm  success=%s",
            group_key,
            len(buffers["actions"]),
            int(global_sub),
            env.step_counter,
            env.max_steps,
            buffers["truncated_flag"],
            rollout_failure,
            fin_dist,
            success,
        )

    if buffers["truncated_flag"]:
        log.warning("skip save  key=%s  reason=truncated", group_key)
        return "skipped:truncated"

    valid, reason = validate_transition_buffers(
        buffers,
        env,
        action_dim=n_j,
    )
    if not valid:
        log.error("skip save  key=%s  invalid_buffers(%s)", group_key, reason)
        return f"skipped:invalid_buffers({reason})"

    if args.save_only_success and not success:
        log.info(
            "skip save  key=%s  reason=not_success (default: save-only-success; "
            "pass --no-save-only-success to keep failing episodes)",
            group_key,
        )
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
            "skip save  key=%s  reason=low_visibility  visibility=%.2f  threshold=%.2f",
            group_key,
            visibility_frac,
            min_vis,
        )
        return f"skipped:low_visibility({visibility_frac:.2f}<{min_vis:.2f})"

    save_info = {
        "count_in_frame": count_in_frame,
        "path_length": int(len(buffers["actions"])),
        "source_group_key": str(group_key),
        "controller_mode": "joint_space_substeps",
        "action_format": "joint_delta_rad",
        "execution_mode": "joint",
        "max_joint_step_rad": float(args.max_joint_step),
        "validate_each_step": bool(args.validate_each_step),
        "joint_final_goal_correction": bool(args.joint_final_goal_correction),
        "final_correction_method": str(args.final_correction_method),
        "goal_achieved_count": int(buffers["goal_achieved_count"]),
        "truncated_flag": bool(buffers["truncated_flag"]),
        "rollout_failure": rollout_failure or "",
        "failure_segment": int(-1 if fail_segment is None else fail_segment),
        "failure_substep": int(-1 if fail_substep is None else fail_substep),
        "final_ee_dist_to_goal_m": fin_dist,
        "num_waypoint_edges": int(num_wp_segments),
        "joint_substeps_total": int(global_sub),
    }
    tree_info = {k: _hdf5_safe_attr_value(v) for k, v in metadata.items()}

    try:
        hdf5_grp = dg.save_agent_data_to_hdf5(
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
        log.info(
            "HDF5_MAPPING  waypoint_source_key=%s  hdf5_group_key=%r",
            group_key,
            hdf5_grp,
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
        num_segments=max(1, global_sub),
        count_in_frame=count_in_frame,
        success=success,
    )
    log.info(
        "SAVED  key=%s  mode=joint  transitions=%s  success=%s  joint_substeps=%s  final_dist=%.4fm  "
        "reward[min/mean/max]=[%.5f / %.5f / %.5f]",
        group_key,
        len(buffers["actions"]),
        success,
        global_sub,
        fin_dist,
        float(np.min(reward_np)) if len(reward_np) else 0.0,
        float(np.mean(reward_np)) if len(reward_np) else 0.0,
        float(np.max(reward_np)) if len(reward_np) else 0.0,
    )
    return "saved"


def run_group_cartesian_legacy(
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
            log.warning(
                "cartesian rollout: env hit max_steps  step_counter=%s/%s  "
                "segment=%s/%s  transitions=%s",
                env.step_counter,
                env.max_steps,
                si + 1,
                num_segments,
                len(buffers["actions"]),
            )
        if args.max_transitions > 0 and len(buffers["actions"]) >= int(args.max_transitions):
            truncated = True
            buffers["truncated_flag"] = True
            log.warning(
                "cartesian rollout: max_transitions cap reached  cap=%s  env_step=%s/%s",
                int(args.max_transitions),
                env.step_counter,
                env.max_steps,
            )
            break

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

        if terminated or truncated:
            break

    # Optional post-path goal correction (same IK/Jacobian options as joint mode).
    max_corr = max(0, int(args.final_correction_steps))
    if not terminated and not truncated and max_corr > 0:
        terminated, truncated = _run_final_goal_correction_joint_delta(
            env=env,
            goal_pos=np.asarray(metadata["goal_pos"], dtype=np.float64),
            buffers=buffers,
            ee_dists=ee_dists,
            ee_step_displacements=ee_step_displacements,
            optical_flow_model=optical_flow_model,
            args=args,
            max_corr=max_corr,
            validate_each_step=False,
            state_invalid_fn=None,
            global_sub_mut=None,
            style="cartesian",
            correction_method=str(args.final_correction_method),
        )

    # ── quality gates ─────────────────────────────────────────────────────────
    success = bool(buffers["dones"] and buffers["dones"][-1])

    if getattr(args, "log_transition_diagnostics", False):
        log.info(
            "TRANSITION_DIAG  key=%s  mode=cartesian  transitions=%s  "
            "env.step_counter=%s  env.max_steps=%s  truncated_flag=%s  success=%s",
            group_key,
            len(buffers["actions"]),
            env.step_counter,
            env.max_steps,
            buffers["truncated_flag"],
            success,
        )

    if buffers["truncated_flag"]:
        log.warning("skip save  key=%s  reason=truncated", group_key)
        return "skipped:truncated"

    valid, reason = validate_transition_buffers(buffers, env)
    if not valid:
        log.error("skip save  key=%s  invalid_buffers(%s)", group_key, reason)
        return f"skipped:invalid_buffers({reason})"

    if args.save_only_success and not success:
        log.info(
            "skip save  key=%s  reason=not_success (use --no-save-only-success to keep)",
            group_key,
        )
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
        "controller_mode": "continuous_physics_rollout",
        "execution_mode": "cartesian",
        "action_format": (
            "ee_velocity_local_scaled_segments_plus_joint_delta_final_correction"
        ),
        "goal_achieved_count": int(buffers["goal_achieved_count"]),
        "truncated_flag": bool(buffers["truncated_flag"]),
        "final_correction_method": str(args.final_correction_method),
    }
    tree_info = {k: _hdf5_safe_attr_value(v) for k, v in metadata.items()}

    try:
        hdf5_grp = dg.save_agent_data_to_hdf5(
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
        log.info(
            "HDF5_MAPPING  waypoint_source_key=%s  hdf5_group_key=%r",
            group_key,
            hdf5_grp,
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
    state_invalid_fn,
):
    """Dispatch: joint-space waypoint tracking (default) or legacy Cartesian rollout."""
    if args.execution_mode == "cartesian":
        return run_group_cartesian_legacy(
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
    return run_group_joint(
        env=env,
        group_key=group_key,
        metadata=metadata,
        refined_path=refined_path,
        optical_flow_model=optical_flow_model,
        agent_data_path=agent_data_path,
        args=args,
        state_invalid_fn=state_invalid_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "v4: Generate RL agent-data from waypoints HDF5. Default: joint-space "
            "execution (step_joint_delta). Use --execution-mode cartesian for legacy v2."
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
        "--save-only-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only save successful trajectories (default: true). "
            "Use --no-save-only-success to also store episodes that do not reach the goal."
        ),
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
    p.add_argument(
        "--execution-mode",
        choices=["joint", "cartesian"],
        default="joint",
        help="joint: step_joint_delta along planner waypoints (default). "
        "cartesian: legacy one env.step per edge (v2 Jacobian).",
    )
    p.add_argument(
        "--max-joint-step",
        type=float,
        default=0.02,
        help="Joint mode: max L2 norm of Δq per substep (radians). Default 0.02.",
    )
    p.add_argument(
        "--validate-each-step",
        action="store_true",
        help="Joint mode: run planner is_state_valid_fn on joint angles each substep.",
    )
    p.add_argument(
        "--joint-final-goal-correction",
        action="store_true",
        default=False,
        help=(
            "Joint mode: after waypoints, run the final closing phase toward the goal "
            "(see --final-correction-method; default ik). Records joint Delta q via "
            "step_joint_delta like the rest of the rollout."
        ),
    )
    p.add_argument(
        "--final-correction-method",
        choices=["ik", "jacobian"],
        default="ik",
        help=(
            "Post-path closing: 'ik' = PyBullet IK to goal position (holds current EE "
            "orientation), L2-clipped by --max-joint-step (default). "
            "'jacobian' = incremental position-only damped Jacobian."
        ),
    )
    p.add_argument(
        "--step-scale", type=float, default=0.15,
        help=(
            "Scale for Jacobian EE path and for capping approach speed in final correction "
            "(default 0.15)."
        ),
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
        "--final-correction-steps", type=int, default=30,
        help=(
            "Maximum position-only goal-closing substeps after the path ends (default 30); "
            "each substep uses step_joint_delta (joint mode if --joint-final-goal-correction; "
            "cartesian mode always runs this phase after segments)."
        ),
    )
    p.add_argument(
        "--final-goal-pos-tol",
        type=float,
        default=float(dg.CONFIG["planning"].get("fk_pos_tolerance", 0.06)),
        help=(
            "Stop correction once EE-to-goal distance <= this (metres). "
            "Default matches CONFIG planning fk_pos_tolerance so env goal threshold aligns."
        ),
    )
    p.add_argument(
        "--max-transitions", type=int, default=0,
        help="Safety cap on transitions per group (0 = no cap).",
    )
    p.add_argument(
        "--log-transition-diagnostics",
        action="store_true",
        help=(
            "Log one summary line per group (step count, max_steps, truncation) and "
            "reward mismatch hints when replay-debugging."
        ),
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
        "Generator v4  execution_mode=%s  max_joint_step=%.4f  validate_each_step=%s  "
        "joint_final_goal_correction=%s  final_correction_method=%s  save_only_success=%s  "
        "min_visibility=%.2f  log_transition_diagnostics=%s",
        args.execution_mode,
        float(args.max_joint_step),
        bool(args.validate_each_step),
        bool(args.joint_final_goal_correction),
        str(args.final_correction_method),
        args.save_only_success,
        args.min_visibility,
        bool(args.log_transition_diagnostics),
    )
    if args.execution_mode == "cartesian":
        log.info(
            "Cartesian legacy knobs: step_scale=%.3f  alpha=%.3f  dq_cap=%.4f  final_correction_steps=%s",
            args.step_scale,
            args.action_smoothing_alpha,
            args.record_joint_dq_cap,
            args.final_correction_steps,
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
                state_invalid_fn=collision_fn,
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
