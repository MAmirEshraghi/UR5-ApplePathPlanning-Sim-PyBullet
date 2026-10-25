#!/usr/bin/env python3
"""
Generator step-debug playground: load one SUCCESS waypoint group from a run HDF5,
refine the joint path like the generator stage, then advance one Jacobian-derived
env.step per keypress (Right or Enter).

Run from repository root (venv activated):

    PYTHONPATH=. python feature_apple_path_planning/playground/debug_generator_stepper.py \\
        --run-id 20260504_203032 --success-index 0

List groups in the waypoint file:

    PYTHONPATH=. python feature_apple_path_planning/playground/debug_generator_stepper.py \\
        --run-id 20260504_203032 --list-groups

Pin an exact HDF5 group:

    PYTHONPATH=. python feature_apple_path_planning/playground/debug_generator_stepper.py \\
        --run-id 20260504_203032 --group-key <exact_group_name>

While the GUI is open: **Up** switches to the next SUCCESS group in the file, **Down**
to the previous (sorted key order). Right / Enter still advance one trajectory segment.

Logging mirrors zenlog to ``logs/data_generator_path_planner/<ts>_generator_debugging.log``.
Optical flow is off by default (fast); pass ``--optical-flow`` for RAFT parity with the generator.

Pass ``--kinematic-step`` to teleport to the next refined waypoint (no ``env.step``), matching
``scrub_waypoints_hdf5`` for A/B checks. Default remains Jacobian + ``env.step`` per keypress.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import logging
import os
import sys
import tkinter as tk
from pathlib import Path

import cv2
import h5py
import numpy as np
from zenlog import log

_FP_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _FP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = importlib.util.spec_from_file_location(
    "apple_data_generator_72",
    _FP_ROOT / "run_apple_data_generator.py",
)
dg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dg)

from apple_picking_env import ApplePickingEnv
from utils_conversions import convert_global_action_to_local


def setup_generator_debug_file_logging() -> Path:
    """Mirror zenlog to logs/data_generator_path_planner/<timestamp>_generator_debugging.log."""
    base = _REPO_ROOT / "logs" / dg.LOG_SESSION_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_generator_debugging.log").resolve()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.addHandler(fh)
    zen_logger.setLevel(logging.DEBUG)
    log.info("Debug session log file: %s", log_path)
    return log_path


def sort_group_keys(keys: list[str]) -> list[str]:
    def sort_key(k: str):
        try:
            return (0, float(k))
        except ValueError:
            return (1, str(k))

    return sorted(keys, key=sort_key)


def list_waypoint_groups(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        for key in sort_group_keys(list(f.keys())):
            grp = f[key]
            fm = grp.attrs.get("fail_mode", None)
            nw = len(grp["waypoints"]) if "waypoints" in grp else 0
            print(f"{key!r}\tfail_mode={fm}\twaypoint_count={nw}")


def select_success_group(f: h5py.File, success_index: int, group_key: str | None) -> tuple[str, h5py.Group]:
    keys = sort_group_keys([k for k in f.keys()])
    success_keys = [k for k in keys if int(f[k].attrs.get("fail_mode", -1)) == dg.ResultMode.SUCCESS.value]
    if not success_keys:
        raise RuntimeError("No SUCCESS groups found in this HDF5 file.")

    if group_key is not None:
        if group_key not in f:
            raise KeyError(f"group-key {group_key!r} not found in HDF5.")
        g = f[group_key]
        if int(g.attrs.get("fail_mode", -1)) != dg.ResultMode.SUCCESS.value:
            log.warning(
                "Group %r fail_mode is not SUCCESS (%s); proceeding anyway.",
                group_key,
                g.attrs.get("fail_mode"),
            )
        return group_key, g

    if success_index < 0 or success_index >= len(success_keys):
        raise IndexError(
            f"success-index {success_index} out of range for {len(success_keys)} SUCCESS group(s)."
        )
    k = success_keys[success_index]
    return k, f[k]


def sorted_success_group_keys(h5_path: Path) -> list[str]:
    """SUCCESS groups only, same sorted order as ``select_success_group``."""
    with h5py.File(h5_path, "r") as f:
        keys = sort_group_keys(list(f.keys()))
        sk = [k for k in keys if int(f[k].attrs.get("fail_mode", -1)) == dg.ResultMode.SUCCESS.value]
    if not sk:
        raise RuntimeError("No SUCCESS groups found in this HDF5 file.")
    return sk


def attrs_to_metadata(grp: h5py.Group) -> dict:
    return {k: np.asarray(v) for k, v in grp.attrs.items()}


def bootstrap_rollout(
    env: ApplePickingEnv, metadata: dict, refined_path: list, *, settle_steps: int = 0
) -> dict:
    apple = np.asarray(metadata["apple_center"], dtype=np.float32).reshape(3,)
    goal_p = np.asarray(metadata["goal_pos"], dtype=np.float32).reshape(3,)
    env.reset(seed=None, options={"target_apple": apple})
    env.reward_goal = goal_p.astype(np.float32, copy=False)
    env.robot.set_joint_angles_no_collision(refined_path[0])
    for _ in range(int(settle_steps)):
        env.pb_client.stepSimulation()
    env.step_counter = 0
    env.is_goal_state = False
    env.sum_reward = 0.0
    env.init_pos_ee, env.init_or_ee = env.robot.get_current_pose(env.robot.tool0_link_idx)
    return env._get_obs()


def _remove_pb_marker(pb_client, body_id: int | None) -> None:
    if body_id is None or body_id < 0:
        return
    try:
        pb_client.removeBody(body_id)
    except Exception:
        pass


def refresh_apple_goal_markers(env: ApplePickingEnv, metadata: dict, marker_ids: dict) -> None:
    """Replace PyBullet markers: red sphere at apple, small green sphere at grasp goal."""
    apple_pos = np.asarray(metadata["apple_center"], dtype=np.float64).reshape(3).tolist()
    goal_pos = np.asarray(metadata["goal_pos"], dtype=np.float64).reshape(3).tolist()
    pb = env.pb_client
    _remove_pb_marker(pb, marker_ids.get("apple"))
    _remove_pb_marker(pb, marker_ids.get("goal"))
    marker_ids["apple"] = dg.draw_debug_sphere(pb, apple_pos, 0.05, [0.9, 0.08, 0.08, 0.9])
    marker_ids["goal"] = dg.draw_debug_sphere(pb, goal_pos, 0.02, [0.08, 0.92, 0.12, 0.92])


def compute_segment_action(env: ApplePickingEnv, path: list, segment_idx: int, control_freq: float, max_ee_vel: float):
    target_joint_angles = path[segment_idx + 1]
    current_joint_angles = env.robot.get_joint_angles()
    dq = np.asarray(target_joint_angles, dtype=np.float64) - np.asarray(current_joint_angles, dtype=np.float64)
    joint_velocities = dq * control_freq
    env.robot.set_joint_angles_no_collision(current_joint_angles)
    jacobian = env.robot.calculate_jacobian()
    global_ee_vel = np.matmul(jacobian, joint_velocities)
    local_raw = convert_global_action_to_local(env.robot, global_ee_vel)
    local_raw = np.asarray(local_raw, dtype=np.float64)
    clipped = np.array(local_raw, copy=True)
    velocity_clipped = False
    if np.any(np.abs(clipped) > max_ee_vel):
        clipped = np.clip(clipped, -max_ee_vel, max_ee_vel)
        velocity_clipped = True
    return {
        "q_curr": np.asarray(current_joint_angles, dtype=np.float64),
        "q_tgt": np.asarray(target_joint_angles, dtype=np.float64),
        "dq": dq,
        "dq_norm": float(np.linalg.norm(dq)),
        "joint_velocities": joint_velocities,
        "global_ee_vel": global_ee_vel,
        "local_ee_vel_raw": local_raw,
        "local_ee_vel_clipped": clipped,
        "velocity_clipped": velocity_clipped,
    }


def project_world_point_ndc(env: ApplePickingEnv, world_xyz: np.ndarray):
    """Same projection pipeline as ApplePickingEnv._compute_deprojected_point_mask (logging only)."""
    view_matrix = env.robot.get_view_mat_at_curr_pose(env.camera_sensor)
    view_matrix_np = np.array(view_matrix).reshape(4, 4, order="F")
    proj_matrix_np = np.array(env.camera_sensor.depth_proj_mat).reshape(4, 4, order="F")
    goal_pos_world = np.array([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0])
    clip_space_pos = proj_matrix_np @ view_matrix_np @ goal_pos_world
    if clip_space_pos[3] == 0:
        return None, False
    ndc = clip_space_pos[:3] / clip_space_pos[3]
    on_screen = bool(-1 <= ndc[0] <= 1 and -1 <= ndc[1] <= 1)
    return ndc.astype(np.float64), on_screen


def log_step_telemetry(
    *,
    run_id: str,
    group_key: str,
    metadata: dict,
    segment_idx: int,
    total_segments: int,
    seg: dict,
    env: ApplePickingEnv,
    action_scale: float,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    new_obs: dict,
    step_scale: float,
    action_applied: np.ndarray | None = None,
    substep_idx: int = 1,
    substeps_total: int = 1,
    dq_norm_after: float | None = None,
    effective_step_scale: float | None = None,
    clipped_streak: int = 0,
    early_stop_reason: str | None = None,
):
    fail_mode = metadata.get("fail_mode", None)
    apple_c = np.asarray(metadata.get("apple_center"), dtype=np.float64).reshape(3,)
    goal_p = np.asarray(metadata.get("goal_pos"), dtype=np.float64).reshape(3,)
    ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    dist_reward_goal = float(np.linalg.norm(ee_pos - np.asarray(env.reward_goal, dtype=np.float64)))
    dist_desired = float(np.linalg.norm(ee_pos - np.asarray(env.desired_goal, dtype=np.float64)))

    mask = new_obs.get("point_mask")
    mask_sum = float(mask.sum()) if mask is not None else float("nan")
    mask_pos = int((mask > 0).sum()) if mask is not None else -1

    ndc, on_screen = project_world_point_ndc(env, env.desired_goal)

    log.info(
        "step_debug run_id=%s group=%s segment=%s/%s fail_mode=%s apple_center_norm=%.4f goal_pos_norm=%.4f",
        run_id,
        group_key,
        segment_idx + 1,
        total_segments,
        fail_mode,
        float(np.linalg.norm(apple_c)),
        float(np.linalg.norm(goal_p)),
    )
    log.info(
        "step_debug joints dq_norm=%.6f velocity_clipped=%s q_curr=%s q_tgt=%s",
        seg["dq_norm"],
        seg["velocity_clipped"],
        np.round(seg["q_curr"], 4).tolist(),
        np.round(seg["q_tgt"], 4).tolist(),
    )
    log.debug("step_debug dq=%s joint_vel=%s", seg["dq"].tolist(), np.round(seg["joint_velocities"], 5).tolist())
    log.info(
        "step_debug ee_vel global=%s local_raw=%s local_clipped=%s action_scale=%s",
        np.round(seg["global_ee_vel"], 5).tolist(),
        np.round(seg["local_ee_vel_raw"], 5).tolist(),
        np.round(seg["local_ee_vel_clipped"], 5).tolist(),
        action_scale,
    )
    log.info(
        "step_debug action step_scale=%s effective_step_scale=%s action_applied=%s",
        step_scale,
        effective_step_scale,
        None if action_applied is None else np.round(action_applied, 5).tolist(),
    )
    log.info(
        "step_debug substep=%s/%s dq_norm_before=%.6f dq_norm_after=%s clipped_streak=%s early_stop_reason=%s",
        substep_idx,
        substeps_total,
        seg["dq_norm"],
        None if dq_norm_after is None else round(float(dq_norm_after), 6),
        clipped_streak,
        early_stop_reason,
    )
    log.info(
        "step_debug post_step reward=%.6f terminated=%s truncated=%s dist_reward_goal=%.5f dist_desired_goal=%.5f",
        reward,
        terminated,
        truncated,
        dist_reward_goal,
        dist_desired,
    )
    log.info("step_debug reward_info=%s", {k: info[k] for k in info if k != "success"})
    log.info(
        "step_debug visibility mask_sum=%.6f mask_px_gt0=%s ndc_xy=%s ndc_z=%s on_screen_ndc=%s",
        mask_sum,
        mask_pos,
        None if ndc is None else np.round(ndc[:2], 5).tolist(),
        None if ndc is None else float(ndc[2]),
        on_screen,
    )


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
        return False, "buffer lengths mismatch"

    obs_keys = list(obs[0].keys())
    act_dim = int(env.action_space.shape[0])
    for i in range(n):
        if set(obs[i].keys()) != set(obs_keys) or set(nxt[i].keys()) != set(obs_keys):
            return False, f"inconsistent observation keys at step {i}"
        for k in obs_keys:
            a = np.asarray(obs[i][k])
            b = np.asarray(nxt[i][k])
            if not (_is_finite_array(a) and _is_finite_array(b)):
                return False, f"non-finite observation detected at step {i}, key={k}"
        act = np.asarray(acts[i], dtype=np.float64).reshape(-1)
        if act.shape[0] != act_dim:
            return False, f"action dim mismatch at step {i}: got {act.shape[0]}, expected {act_dim}"
        if not _is_finite_array(act):
            return False, f"non-finite action at step {i}"
        if not np.isfinite(float(rews[i])):
            return False, f"non-finite reward at step {i}"
    return True, "ok"


def verify_agent_hdf5_file(path: Path) -> None:
    if not path.exists():
        print(f"[verify-hdf5] file not found: {path}")
        return
    with h5py.File(path, "r") as f:
        keys = sort_group_keys(list(f.keys()))
        print(f"[verify-hdf5] file={path}")
        print(f"[verify-hdf5] trajectories={len(keys)}")
        if not keys:
            return
        first = f[keys[0]]
        obs_shapes = {k: tuple(first["observations"][k].shape) for k in first["observations"].keys()}
        next_obs_shapes = {k: tuple(first["next_observations"][k].shape) for k in first["next_observations"].keys()}
        print(f"[verify-hdf5] first_group={keys[0]}")
        print(f"[verify-hdf5] actions_shape={tuple(first['actions'].shape)}")
        print(f"[verify-hdf5] rewards_shape={tuple(first['rewards'].shape)} dones_shape={tuple(first['dones'].shape)}")
        print(f"[verify-hdf5] observations_shapes={obs_shapes}")
        print(f"[verify-hdf5] next_observations_shapes={next_obs_shapes}")


def parse_args():
    p = argparse.ArgumentParser(description="Step-debug generator rollout from waypoint HDF5.")
    p.add_argument(
        "--run-id",
        default=None,
        help="Waypoint HDF5 stem under output/waypoints/. Latest file used if omitted.",
    )
    p.add_argument("--success-index", type=int, default=0, help="Nth SUCCESS group (sorted keys). Default 0.")
    p.add_argument("--group-key", default=None, help="Exact HDF5 group name (overrides success-index).")
    p.add_argument("--list-groups", action="store_true", help="Print (key, fail_mode, waypoint_count) and exit.")
    p.add_argument(
        "--optical-flow",
        action="store_true",
        help="Compute RAFT optical flow each step (default: off, equivalent to --no-optical-flow).",
    )
    p.add_argument(
        "--kinematic-step",
        action="store_true",
        help="Right/Enter: teleport to next waypoint via set_joint (no env.step); compare to scrubber.",
    )
    p.add_argument(
        "--step-scale",
        type=float,
        default=0.15,
        help="Scale applied clipped action for env.step (default 0.15). Ignored with --kinematic-step.",
    )
    p.add_argument(
        "--substeps-per-segment",
        type=int,
        default=8,
        help="Max env.step micro-steps per waypoint segment in dynamic mode.",
    )
    p.add_argument(
        "--dq-epsilon",
        type=float,
        default=0.03,
        help="Early stop threshold for ||q_tgt - q_curr|| during substeps.",
    )
    p.add_argument(
        "--bootstrap-settle-steps",
        type=int,
        default=0,
        help="Number of stepSimulation() calls after teleporting to refined_path[0] (default 0).",
    )
    p.add_argument(
        "--start-sync-tol",
        type=float,
        default=0.02,
        help="Max allowed ||q_curr - path[segment_idx]|| before auto-resync (default 0.02 rad).",
    )
    p.add_argument(
        "--record-agent-data",
        action="store_true",
        help="Record transitions and append them to output/agent_data/{run_id}.hdf5.",
    )
    p.add_argument(
        "--auto-run-group",
        action="store_true",
        help="Automatically execute all segments for the selected group (no keypresses).",
    )
    p.add_argument(
        "--save-only-success",
        action="store_true",
        help="When recording, save only trajectories with final done=True.",
    )
    p.add_argument(
        "--max-transitions",
        type=int,
        default=0,
        help="Safety cap for recorded transitions per group (0 disables cap).",
    )
    p.add_argument(
        "--verify-hdf5",
        action="store_true",
        help="Print quick shape/count summary for output/agent_data/{run_id}.hdf5 and exit.",
    )
    p.add_argument(
        "--record-joint-gain",
        type=float,
        default=0.2,
        help="Record-mode gain multiplier for joint-error tracking (default 0.2).",
    )
    p.add_argument(
        "--record-divergence-ratio",
        type=float,
        default=2.5,
        help="Record-mode divergence guard: stop segment if dq_after > ratio*dq_before (default 2.5).",
    )
    p.add_argument(
        "--max-resync-to-save",
        type=int,
        default=0,
        help="Skip saving recorded group if resync count exceeds this value (default 0).",
    )
    p.add_argument(
        "--max-dqnorm-to-save",
        type=float,
        default=0.2,
        help="Skip saving recorded group if max observed dq_norm_after exceeds this value (default 0.2).",
    )
    p.add_argument(
        "--record-soft-sync-tol",
        type=float,
        default=0.07,
        help=(
            "Record-mode soft start-sync tolerance. If start drift is <= this value, do not teleport/resync-count "
            "(default 0.07 rad). Values > --start-sync-tol reduce harmless resync spam while keeping hard guard."
        ),
    )
    p.add_argument(
        "--record-joint-dq-cap",
        type=float,
        default=0.04,
        help="Clamp per-joint dq in record mode before Jacobian conversion (default 0.04 rad).",
    )
    p.add_argument(
        "--action-smoothing-alpha",
        type=float,
        default=0.35,
        help=(
            "EMA smoothing for record-mode actions in [0,1]. 0=no smoothing, higher=more damping "
            "(default 0.35)."
        ),
    )
    p.add_argument(
        "--final-correction-steps",
        type=int,
        default=8,
        help="Extra cartesian correction steps after path end when not terminated (default 8).",
    )
    p.add_argument(
        "--final-goal-pos-tol",
        type=float,
        default=0.05,
        help="Stop extra correction once EE-to-goal distance <= this (meters, default 0.05).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    setup_generator_debug_file_logging()

    if args.record_agent_data and args.kinematic_step:
        raise ValueError("--record-agent-data requires dynamic stepping (do not use --kinematic-step).")

    waypoints_dir, agent_data_dir, _ = dg._ensure_output_dirs()
    run_id, waypoints_path = dg._resolve_run_id(args.run_id, waypoints_dir, should_exist=True)
    agent_data_path = (agent_data_dir / f"{run_id}.hdf5").resolve()

    if args.verify_hdf5:
        verify_agent_hdf5_file(agent_data_path)
        return

    if args.list_groups:
        print(f"run_id={run_id}\tpath={waypoints_path}")
        list_waypoint_groups(Path(waypoints_path))
        return

    optical_flow_model = None
    if args.optical_flow:
        try:
            optical_flow_model = dg.OpticalFlow(size=dg.DEFAULT_OPTICAL_FLOW_SIZE)
        except Exception as exc:
            log.error("Optical flow init failed: %s — using zeros.", exc, exc_info=True)

    wp_path = Path(waypoints_path)
    success_keys = sorted_success_group_keys(wp_path)

    if args.group_key is not None:
        if args.group_key not in success_keys:
            raise ValueError(
                f"group-key {args.group_key!r} is not among SUCCESS groups; use --list-groups."
            )
        start_success_idx = success_keys.index(args.group_key)
    else:
        if args.success_index < 0 or args.success_index >= len(success_keys):
            raise IndexError(
                f"success-index {args.success_index} out of range for {len(success_keys)} SUCCESS group(s)."
            )
        start_success_idx = args.success_index

    success_idx_holder = [start_success_idx]
    group_key = success_keys[start_success_idx]
    with h5py.File(wp_path, "r") as f:
        grp = f[group_key]
        waypoints = grp["waypoints"][:]
        metadata = attrs_to_metadata(grp)

    if len(waypoints) < 2:
        raise ValueError(f"Group {group_key!r} has fewer than 2 waypoints.")

    waypoints_list = waypoints.tolist()

    env = ApplePickingEnv(config=dg.CONFIG)
    robot = env.robot
    _, _, extend_fn, collision_fn = dg.setup_planning_functions(
        robot,
        env.tree.pyb_id,
        dg.CONFIG["planning"],
        dg.CONFIG["visualization"],
        env.pb_client,
        env.tree,
    )

    refined_path = dg.shortcut_and_refine_path(
        robot,
        waypoints_list,
        extend_fn,
        collision_fn,
        dg.CONFIG["planning"]["task_space_refinement_threshold"],
        enable_smoothing=dg.CONFIG["planning"].get("enable_smoothing", True),
    )

    # Anchor start of refined path to planner start joint configuration
    planner_start = np.asarray(dg.CONFIG["robot_setup"]["start_joint_angles"], dtype=np.float64)
    if len(refined_path) >= 1:
        refined_path[0] = planner_start.tolist()

    if len(refined_path) > dg.CONFIG["simulation_setup"]["max_steps"]:
        log.warning(
            "Refined path length %s exceeds max_steps %s — stepping anyway for debug.",
            len(refined_path),
            dg.CONFIG["simulation_setup"]["max_steps"],
        )

    rollout_ctx = {"group_key": group_key, "metadata": metadata, "refined_path": refined_path}

    marker_ids: dict = {"apple": -1, "goal": -1}
    obs = bootstrap_rollout(
        env,
        rollout_ctx["metadata"],
        rollout_ctx["refined_path"],
        settle_steps=args.bootstrap_settle_steps,
    )
    refresh_apple_goal_markers(env, rollout_ctx["metadata"], marker_ids)
    obs = dg.ensure_optical_flow(obs, optical_flow_model)

    control_freq = 1.0 / dg.CONFIG["simulation_setup"]["control_time"]
    max_ee_vel = dg.CONFIG["planning"]["max_ee_velocity"]
    action_scale = env.action_scale

    log.info(
        "Bootstrap complete run_id=%s group=%s SUCCESS_group %s/%s refined_len=%s segments=%s kinematic_step=%s step_scale=%s substeps_per_segment=%s dq_epsilon=%s renders=%s",
        run_id,
        rollout_ctx["group_key"],
        success_idx_holder[0] + 1,
        len(success_keys),
        len(rollout_ctx["refined_path"]),
        len(rollout_ctx["refined_path"]) - 1,
        args.kinematic_step,
        args.step_scale,
        args.substeps_per_segment,
        args.dq_epsilon,
        dg.CONFIG["simulation_setup"]["renders"],
    )

    episode_buffers = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "goal_achieved_count": 0,
        "resync_count": 0,
        "truncated_flag": False,
        "max_dq_norm_after": 0.0,
    }

    state = {
        "segment_idx": 0,
        "episode_done": False,
        "obs": obs,
        "busy": False,
        "prev_action_applied": None,
    }

    def reset_episode_buffers():
        episode_buffers["observations"].clear()
        episode_buffers["next_observations"].clear()
        episode_buffers["actions"].clear()
        episode_buffers["rewards"].clear()
        episode_buffers["dones"].clear()
        episode_buffers["goal_achieved_count"] = 0
        episode_buffers["resync_count"] = 0
        episode_buffers["truncated_flag"] = False
        episode_buffers["max_dq_norm_after"] = 0.0

    def maybe_save_recorded_group():
        if not args.record_agent_data:
            return
        try:
            valid, reason = validate_transition_buffers(episode_buffers, env)
            if not valid:
                log.error("Skipping save for group=%s: invalid transition buffers (%s).", rollout_ctx["group_key"], reason)
                return
            observations = episode_buffers["observations"]
            new_observations = episode_buffers["next_observations"]
            actions = episode_buffers["actions"]
            rewards = episode_buffers["rewards"]
            dones = episode_buffers["dones"]
            success = bool(dones[-1])
            if episode_buffers["truncated_flag"]:
                log.warning("Skipping save for group=%s: rollout truncated.", rollout_ctx["group_key"])
                return
            if int(episode_buffers["resync_count"]) > int(args.max_resync_to_save):
                log.warning(
                    "Skipping save for group=%s: resync_count=%s > max_resync_to_save=%s.",
                    rollout_ctx["group_key"],
                    episode_buffers["resync_count"],
                    int(args.max_resync_to_save),
                )
                return
            if float(episode_buffers["max_dq_norm_after"]) > float(args.max_dqnorm_to_save):
                log.warning(
                    "Skipping save for group=%s: max_dq_norm_after=%.6f > max_dqnorm_to_save=%.6f.",
                    rollout_ctx["group_key"],
                    float(episode_buffers["max_dq_norm_after"]),
                    float(args.max_dqnorm_to_save),
                )
                return
            if args.save_only_success and not success:
                log.info("Skipping save for group=%s due to --save-only-success.", rollout_ctx["group_key"])
                return
            count_in_frame = int(sum(1 for o in new_observations if "point_mask" in o and np.any(o["point_mask"] > 0)))
            info = {
                "count_in_frame": count_in_frame,
                "path_length": int(len(actions)),
                "source_group_key": str(rollout_ctx["group_key"]),
                "controller_mode": "joint_interp_to_ee_env_step",
                "goal_achieved_count": int(episode_buffers["goal_achieved_count"]),
                "resync_count": int(episode_buffers["resync_count"]),
                "truncated_flag": bool(episode_buffers["truncated_flag"]),
                "max_dq_norm_after": float(episode_buffers["max_dq_norm_after"]),
            }
            tree_info = {k: _hdf5_safe_attr_value(v) for k, v in rollout_ctx["metadata"].items()}
            log.info(
                "save_attempt group=%s transitions=%s success=%s path=%s",
                rollout_ctx["group_key"],
                len(actions),
                success,
                str(agent_data_path),
            )
            dg.save_agent_data_to_hdf5(
                observations,
                actions,
                rewards,
                dones,
                new_observations,
                info,
                success,
                tree_info,
                np.asarray(dg.CONFIG["tree_setup"]["position"]),
                np.asarray(dg.CONFIG["tree_setup"]["orientation"]),
                str(agent_data_path),
            )
            log.info(
                "save_success group=%s transitions=%s path=%s",
                rollout_ctx["group_key"],
                len(actions),
                str(agent_data_path),
            )
            action_np = np.asarray(actions, dtype=np.float64)
            reward_np = np.asarray(rewards, dtype=np.float64)
            done_np = np.asarray(dones, dtype=np.bool_)
            terminal_idx = int(np.argmax(done_np)) if bool(np.any(done_np)) else -1
            log.info(
                "record_summary group=%s transitions=%s success=%s reward[min/mean/max]=[%.6f, %.6f, %.6f] "
                "action[min/max]=[%s, %s] done_count=%s terminal_idx=%s resync_count=%s max_dq_norm_after=%.6f",
                rollout_ctx["group_key"],
                len(actions),
                success,
                float(np.min(reward_np)),
                float(np.mean(reward_np)),
                float(np.max(reward_np)),
                np.round(np.min(action_np, axis=0), 5).tolist(),
                np.round(np.max(action_np, axis=0), 5).tolist(),
                int(np.sum(done_np)),
                terminal_idx,
                int(episode_buffers["resync_count"]),
                float(episode_buffers["max_dq_norm_after"]),
            )
        except Exception as exc:
            log.error("save_failure group=%s path=%s error=%s", rollout_ctx["group_key"], str(agent_data_path), exc)
            log.exception("save_failure_trace")

    def apply_final_goal_correction() -> tuple[bool, bool]:
        if not args.record_agent_data:
            return False, False
        max_steps = max(0, int(args.final_correction_steps))
        if max_steps == 0:
            return False, False
        goal_pos = np.asarray(rollout_ctx["metadata"]["goal_pos"], dtype=np.float64)
        control_time = float(dg.CONFIG["simulation_setup"]["control_time"])
        max_ee_vel_arr = np.asarray(max_ee_vel, dtype=np.float64)
        alpha = float(np.clip(args.action_smoothing_alpha, 0.0, 1.0))
        terminated = False
        truncated = False
        for step_idx in range(1, max_steps + 1):
            ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
            ee_pos = np.asarray(ee_pos, dtype=np.float64)
            err = goal_pos - ee_pos
            err_norm = float(np.linalg.norm(err))
            if err_norm <= float(args.final_goal_pos_tol):
                log.info(
                    "final_correction stop: within goal tolerance step=%s err=%.6f tol=%.6f",
                    step_idx,
                    err_norm,
                    float(args.final_goal_pos_tol),
                )
                break
            ee_vel_global = np.zeros(6, dtype=np.float64)
            ee_vel_global[:3] = err / max(control_time, 1e-6)
            local_vel = np.asarray(convert_global_action_to_local(env.robot, ee_vel_global), dtype=np.float64)
            local_vel = np.clip(local_vel, -max_ee_vel_arr, max_ee_vel_arr)
            action_raw = local_vel * float(args.step_scale)
            prev = state.get("prev_action_applied")
            if prev is None:
                action_applied = action_raw
            else:
                action_applied = (1.0 - alpha) * action_raw + alpha * np.asarray(prev, dtype=np.float64)
            obs_before = state["obs"]
            new_obs, reward, terminated, truncated, info = env.step(action_applied.astype(np.float64))
            new_obs = dg.ensure_optical_flow(new_obs, optical_flow_model)
            done_flag = bool(terminated or truncated)
            episode_buffers["observations"].append(obs_before)
            episode_buffers["next_observations"].append(new_obs)
            episode_buffers["actions"].append(np.asarray(action_applied, dtype=np.float32))
            episode_buffers["rewards"].append(float(reward))
            episode_buffers["dones"].append(done_flag)
            if bool(info.get("goal_achieved", False)):
                episode_buffers["goal_achieved_count"] += 1
            if truncated:
                episode_buffers["truncated_flag"] = True
            state["obs"] = new_obs
            state["prev_action_applied"] = np.asarray(action_applied, dtype=np.float64)
            log.info(
                "final_correction step=%s/%s err_norm=%.6f action=%s reward=%.6f terminated=%s truncated=%s",
                step_idx,
                max_steps,
                err_norm,
                np.round(action_applied, 5).tolist(),
                float(reward),
                terminated,
                truncated,
            )
            if done_flag:
                break
        return bool(terminated), bool(truncated)

    root = tk.Tk()
    root.title("Generator step-debug (↑↓ group · → Enter step)")
    tk.Label(
        root,
        text=(
            "↑ Next SUCCESS group    ↓ Previous SUCCESS group\n"
            "→ or Enter — one segment per press"
            + (" (kinematic teleport, no env.step)" if args.kinematic_step else " (Jacobian + env.step)")
            + "\nClose window to exit."
        ),
        justify=tk.LEFT,
        padx=12,
        pady=12,
    ).pack()

    def change_success_group(delta: int, _event=None):
        if state["busy"]:
            return
        state["busy"] = True

        def work():
            try:
                idx = success_idx_holder[0]
                new_idx = int(np.clip(idx + delta, 0, len(success_keys) - 1))
                if new_idx == idx:
                    log.debug("SUCCESS group list boundary at index %s", idx)
                    return

                key = success_keys[new_idx]
                with h5py.File(wp_path, "r") as f:
                    grp = f[key]
                    w = grp["waypoints"][:]
                    md = attrs_to_metadata(grp)

                if len(w) < 2:
                    log.warning("Group %r has fewer than 2 waypoints; staying on current group.", key)
                    return

                wl = w.tolist()
                rp = dg.shortcut_and_refine_path(
                    robot,
                    wl,
                    extend_fn,
                    collision_fn,
                    dg.CONFIG["planning"]["task_space_refinement_threshold"],
                    enable_smoothing=dg.CONFIG["planning"].get("enable_smoothing", True),
                )

                # Re-anchor start of refined path for new group
                if len(rp) >= 1:
                    rp[0] = planner_start.tolist()

                success_idx_holder[0] = new_idx
                rollout_ctx["group_key"] = key
                rollout_ctx["metadata"] = md
                rollout_ctx["refined_path"] = rp

                obs2 = bootstrap_rollout(
                    env,
                    md,
                    rp,
                    settle_steps=args.bootstrap_settle_steps,
                )
                refresh_apple_goal_markers(env, md, marker_ids)
                obs2 = dg.ensure_optical_flow(obs2, optical_flow_model)
                state["obs"] = obs2
                state["segment_idx"] = 0
                state["episode_done"] = False
                state["prev_action_applied"] = None
                reset_episode_buffers()

                if dg.CONFIG["simulation_setup"]["renders"]:
                    dg.visualize_actions_camera(obs2)
                    try:
                        cv2.waitKey(1)
                    except Exception:
                        pass

                log.info(
                    "Switched SUCCESS group %s/%s key=%r refined_len=%s",
                    new_idx + 1,
                    len(success_keys),
                    key,
                    len(rp),
                )

                root.update_idletasks()
            finally:
                state["busy"] = False

        root.after(0, work)

    def run_one_segment_step():
        if state["episode_done"]:
            log.warning("Episode ended — ignoring advance.")
            return
        si = state["segment_idx"]
        path = rollout_ctx["refined_path"]
        num_segments = len(path) - 1
        if si >= num_segments:
            log.info("All %s segment(s) executed.", num_segments)
            state["episode_done"] = True
            maybe_save_recorded_group()
            return

        # Ensure simulator start state matches planned start for this segment
        q_curr = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
        q_start = np.asarray(path[si], dtype=np.float64)
        dq_start = q_start - q_curr
        dq_start_norm = float(np.linalg.norm(dq_start))
        if dq_start_norm > float(args.start_sync_tol):
            # In record mode, allow moderate simulator drift without teleporting.
            # This avoids resync-count inflation while preserving a hard resync guard.
            if args.record_agent_data and dq_start_norm <= float(args.record_soft_sync_tol):
                log.info(
                    "Record-mode soft-sync accepted for segment %s: ||q_curr - q_start||=%.6f "
                    "(start_sync_tol=%.6f, record_soft_sync_tol=%.6f).",
                    si,
                    dq_start_norm,
                    float(args.start_sync_tol),
                    float(args.record_soft_sync_tol),
                )
            else:
                log.warning(
                    "Resyncing start of segment %s: ||q_curr - q_start||=%.6f > sync threshold. "
                    "Teleporting back to path[si].",
                    si,
                    dq_start_norm,
                )
                env.robot.set_joint_angles_no_collision(path[si])
                q_curr = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
                dq_start = q_start - q_curr
                dq_start_norm = float(np.linalg.norm(dq_start))
                log.info(
                    "Post-resync start error for segment %s: ||q_curr - q_start||=%.6f",
                    si,
                    dq_start_norm,
                )
                episode_buffers["resync_count"] += 1

        # Precompute segment info once: joint-space delta and diagnostics
        seg = compute_segment_action(env, path, si, control_freq, max_ee_vel)
        action_applied = None

        if args.kinematic_step:
            # One-shot teleport to next waypoint (same as scrubber, no intermediate points)
            env.robot.set_joint_angles_no_collision(path[si + 1])
            new_obs = env._get_obs()
            new_obs = dg.ensure_optical_flow(new_obs, optical_flow_model)
            reward, terminated, truncated, info = 0.0, False, False, {}
            dq_norm_after = 0.0
            log_step_telemetry(
                run_id=run_id,
                group_key=rollout_ctx["group_key"],
                metadata=rollout_ctx["metadata"],
                segment_idx=si,
                total_segments=num_segments,
                seg=seg,
                env=env,
                action_scale=action_scale,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                new_obs=new_obs,
                step_scale=float(args.step_scale),
                action_applied=action_applied,
                substep_idx=1,
                substeps_total=1,
                dq_norm_after=dq_norm_after,
                effective_step_scale=0.0,
                clipped_streak=0,
                early_stop_reason="kinematic_step",
            )
        else:
            # Option 1 base: joint-space interpolation path.
            # In record mode, convert each interpolation target into EE command and run env.step.
            q_start = np.asarray(path[si], dtype=np.float64)
            q_tgt = np.asarray(path[si + 1], dtype=np.float64)
            dq = q_tgt - q_start
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            new_obs = state["obs"]
            dq_norm_after = float(np.linalg.norm(q_tgt - q_start))
            early_stop_reason = "max_substeps"

            num_substeps = max(1, int(args.substeps_per_segment))
            adaptive_step_scale = float(args.step_scale)
            worsening_streak = 0
            for sub_idx in range(1, num_substeps + 1):
                frac = float(sub_idx) / float(num_substeps)
                q_interp = q_start + frac * dq

                if args.record_agent_data:
                    obs_before = state["obs"] if sub_idx == 1 else new_obs
                    q_now = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
                    dq_sub = q_tgt - q_now
                    dq_cap = max(0.0, float(args.record_joint_dq_cap))
                    if dq_cap > 0.0:
                        dq_sub = np.clip(dq_sub, -dq_cap, dq_cap)
                    dq_norm_before = float(np.linalg.norm(dq_sub))
                    joint_vel = dq_sub * control_freq * float(args.record_joint_gain)
                    env.robot.set_joint_angles_no_collision(q_now)
                    jac = env.robot.calculate_jacobian()
                    global_ee_vel = np.matmul(jac, joint_vel)
                    local_vel = np.asarray(convert_global_action_to_local(env.robot, global_ee_vel), dtype=np.float64)
                    local_vel = np.clip(local_vel, -max_ee_vel, max_ee_vel)
                    action_raw = local_vel * adaptive_step_scale
                    alpha = float(np.clip(args.action_smoothing_alpha, 0.0, 1.0))
                    prev = state.get("prev_action_applied")
                    if prev is None:
                        action_applied = action_raw
                    else:
                        action_applied = (1.0 - alpha) * action_raw + alpha * np.asarray(prev, dtype=np.float64)
                    new_obs, reward, terminated, truncated, info = env.step(action_applied.astype(np.float64))
                    new_obs = dg.ensure_optical_flow(new_obs, optical_flow_model)
                    done_flag = bool(terminated or truncated)
                    episode_buffers["observations"].append(obs_before)
                    episode_buffers["next_observations"].append(new_obs)
                    episode_buffers["actions"].append(np.asarray(action_applied, dtype=np.float32))
                    episode_buffers["rewards"].append(float(reward))
                    episode_buffers["dones"].append(done_flag)
                    if bool(info.get("goal_achieved", False)):
                        episode_buffers["goal_achieved_count"] += 1
                    if truncated:
                        episode_buffers["truncated_flag"] = True
                    if args.max_transitions > 0 and len(episode_buffers["actions"]) >= int(args.max_transitions):
                        terminated = True
                        early_stop_reason = "max_transitions_reached"
                    state["prev_action_applied"] = np.asarray(action_applied, dtype=np.float64)
                else:
                    env.robot.set_joint_angles_no_collision(q_interp)
                    new_obs = env._get_obs()
                    new_obs = dg.ensure_optical_flow(new_obs, optical_flow_model)
                    reward, terminated, truncated, info = 0.0, False, False, {}
                    action_applied = None

                q_after = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
                dq_norm_after = float(np.linalg.norm(q_tgt - q_after))
                episode_buffers["max_dq_norm_after"] = max(
                    float(episode_buffers["max_dq_norm_after"]), float(dq_norm_after)
                )

                if dq_norm_after < float(args.dq_epsilon):
                    early_stop_reason = "epsilon_reached"
                elif bool(terminated or truncated):
                    early_stop_reason = "terminated_or_truncated"
                elif args.record_agent_data and dq_norm_after > dq_norm_before * float(args.record_divergence_ratio):
                    early_stop_reason = "divergence_guard"
                elif args.record_agent_data and dq_norm_after > dq_norm_before:
                    worsening_streak += 1
                    adaptive_step_scale = max(0.02, adaptive_step_scale * 0.7)
                    if worsening_streak >= 2:
                        early_stop_reason = "worsening_guard"
                elif early_stop_reason != "max_transitions_reached":
                    early_stop_reason = "max_substeps"
                    worsening_streak = 0

                log_step_telemetry(
                    run_id=run_id,
                    group_key=rollout_ctx["group_key"],
                    metadata=rollout_ctx["metadata"],
                    segment_idx=si,
                    total_segments=num_segments,
                    seg=seg,
                    env=env,
                    action_scale=action_scale,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    new_obs=new_obs,
                    step_scale=float(args.step_scale),
                    action_applied=action_applied,
                    substep_idx=sub_idx,
                    substeps_total=num_substeps,
                    dq_norm_after=dq_norm_after,
                    effective_step_scale=adaptive_step_scale,
                    clipped_streak=0,
                    early_stop_reason=early_stop_reason,
                )

                if early_stop_reason in {
                    "epsilon_reached",
                    "terminated_or_truncated",
                    "max_transitions_reached",
                    "worsening_guard",
                    "divergence_guard",
                }:
                    break

        if dg.CONFIG["simulation_setup"]["renders"]:
            dg.visualize_actions_camera(new_obs)

        state["obs"] = new_obs
        state["segment_idx"] = si + 1

        if state["segment_idx"] >= num_segments or terminated or truncated:
            if not terminated and not truncated:
                c_term, c_trunc = apply_final_goal_correction()
                terminated = bool(terminated or c_term)
                truncated = bool(truncated or c_trunc)
            state["episode_done"] = True
            log.info("Episode finished terminated=%s truncated=%s", terminated, truncated)
            maybe_save_recorded_group()

    def advance_once(_event=None):
        if state["busy"]:
            return
        state["busy"] = True

        def work():
            try:
                run_one_segment_step()
                root.update_idletasks()
                try:
                    cv2.waitKey(1)
                except Exception:
                    pass
            finally:
                state["busy"] = False

        root.after(0, work)

    root.bind("<Up>", lambda e: change_success_group(+1))
    root.bind("<Down>", lambda e: change_success_group(-1))
    root.bind("<Right>", advance_once)
    root.bind("<Return>", advance_once)

    if args.auto_run_group:
        log.info("Auto-run enabled for group=%s record_agent_data=%s", rollout_ctx["group_key"], args.record_agent_data)
        while not state["episode_done"]:
            run_one_segment_step()
            if dg.CONFIG["simulation_setup"]["renders"]:
                try:
                    cv2.waitKey(1)
                except Exception:
                    pass
        verify_agent_hdf5_file(agent_data_path) if args.record_agent_data else None
        return

    root.mainloop()


if __name__ == "__main__":
    main()
