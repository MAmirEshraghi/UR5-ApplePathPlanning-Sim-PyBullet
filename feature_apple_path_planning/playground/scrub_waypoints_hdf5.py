#!/usr/bin/env python3
"""
Waypoint HDF5 visual scrubber: load a SUCCESS waypoint group, bootstrap the env
(apple + reward goal + start pose), then scrub joint poses with Arrow Left/Right.
Does not call env.step — only teleports joints for inspection.

Run from repository root:

    PYTHONPATH=. python3 feature_apple_path_planning/playground/scrub_waypoints_hdf5.py \\
        --run-id 20260504_203032

Use raw HDF5 waypoints by default; add ``--refine`` for the same refinement as the generator.

    PYTHONPATH=. python3 feature_apple_path_planning/playground/scrub_waypoints_hdf5.py \\
        --run-id 20260504_203032 --refine

Keys: Left / Right (prev/next waypoint), Home / End (first/last).
Up / Down: next / previous SUCCESS group in the same HDF5 (sorted key order).

PyBullet: red debug sphere at ``apple_center``, small green sphere at ``goal_pos`` (per group),
same idea as the generator stage visualization.

By default playback is **kinematic** (joint teleport only), matching ``run_apple_data_generator.py``
``--stage visualize``. Pass ``--physics-settle`` to call ``stepSimulation`` after each teleport (legacy;
poses can drift away from the saved plan).

Logging: ``logs/data_generator_path_planner/<ts>_waypoint_scrubbing.log`` on logger ``pythonConfig``.
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


def setup_scrub_session_logging() -> Path:
    base = _REPO_ROOT / "logs" / dg.LOG_SESSION_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_waypoint_scrubbing.log").resolve()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.addHandler(fh)
    zen_logger.setLevel(logging.DEBUG)
    log.info("Waypoint scrub session log: %s", log_path)
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


def _maybe_step_sim_after_teleport(env: ApplePickingEnv, physics_settle: bool, steps: int = 2) -> None:
    if not physics_settle:
        return
    for _ in range(steps):
        env.pb_client.stepSimulation()


def bootstrap_rollout(
    env: ApplePickingEnv,
    metadata: dict,
    path: list,
    *,
    physics_settle: bool = False,
) -> dict:
    apple = np.asarray(metadata["apple_center"], dtype=np.float32).reshape(3,)
    goal_p = np.asarray(metadata["goal_pos"], dtype=np.float32).reshape(3,)
    env.reset(seed=None, options={"target_apple": apple})
    env.reward_goal = goal_p.astype(np.float32, copy=False)
    env.robot.set_joint_angles_no_collision(path[0])
    _maybe_step_sim_after_teleport(env, physics_settle)
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


def apply_waypoint_index(
    env: ApplePickingEnv,
    path: list,
    idx: int,
    optical_flow_model,
    *,
    physics_settle: bool = False,
) -> dict:
    """Teleport to path[idx], optionally step sim, refresh obs without changing init_pos_ee."""
    env.robot.set_joint_angles_no_collision(path[idx])
    _maybe_step_sim_after_teleport(env, physics_settle)
    obs = env._get_obs()
    obs = dg.ensure_optical_flow(obs, optical_flow_model)
    if dg.CONFIG["simulation_setup"]["renders"]:
        dg.visualize_actions_camera(obs)
        try:
            cv2.waitKey(1)
        except Exception:
            pass
    return obs


def log_scrub_telemetry(
    *,
    run_id: str,
    group_key: str,
    idx: int,
    path_len: int,
    refined: bool,
    env: ApplePickingEnv,
    obs: dict,
):
    ee_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    q = np.asarray(env.robot.get_joint_angles(), dtype=np.float64)
    mask = obs.get("point_mask")
    mask_px = int((mask > 0).sum()) if mask is not None else -1
    log.info(
        "scrub run_id=%s group=%s idx=%s/%s refined=%s ee_pos=%s ||q||=%.5f mask_px_gt0=%s",
        run_id,
        group_key,
        idx,
        path_len - 1,
        refined,
        np.round(ee_pos, 4).tolist(),
        float(np.linalg.norm(q)),
        mask_px,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Scrub waypoint HDF5 poses with Arrow Left/Right.")
    p.add_argument(
        "--run-id",
        default=None,
        help="Waypoint HDF5 stem under output/waypoints/. Latest file used if omitted.",
    )
    p.add_argument("--success-index", type=int, default=0, help="Nth SUCCESS group (sorted keys). Default 0.")
    p.add_argument("--group-key", default=None, help="Exact HDF5 group name (overrides success-index).")
    p.add_argument("--list-groups", action="store_true", help="Print groups then exit.")
    p.add_argument(
        "--refine",
        action="store_true",
        help="Apply shortcut_and_refine_path like the generator (default: raw HDF5 waypoints).",
    )
    p.add_argument("--optical-flow", action="store_true", help="Compute RAFT optical flow each update (default off).")
    p.add_argument(
        "--physics-settle",
        action="store_true",
        help="Step simulation after each joint teleport (legacy). Default is kinematic, like --stage visualize.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    setup_scrub_session_logging()

    waypoints_dir, _, _ = dg._ensure_output_dirs()
    run_id, waypoints_path = dg._resolve_run_id(args.run_id, waypoints_dir, should_exist=True)

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

    if len(waypoints) < 1:
        raise ValueError(f"Group {group_key!r} has no waypoints.")

    waypoints_list = waypoints.tolist()

    env = ApplePickingEnv(config=dg.CONFIG)
    robot = env.robot

    extend_fn = None
    collision_fn = None
    if args.refine:
        _, _, extend_fn, collision_fn = dg.setup_planning_functions(
            robot,
            env.tree.pyb_id,
            dg.CONFIG["planning"],
            dg.CONFIG["visualization"],
            env.pb_client,
            env.tree,
        )

    def path_from_raw_waypoints(wl: list) -> list:
        if not args.refine:
            return wl
        if len(wl) < 2:
            log.warning("Refine requested but fewer than 2 waypoints; using raw path.")
            return wl
        assert extend_fn is not None and collision_fn is not None
        return dg.shortcut_and_refine_path(
            robot,
            wl,
            extend_fn,
            collision_fn,
            dg.CONFIG["planning"]["task_space_refinement_threshold"],
            enable_smoothing=dg.CONFIG["planning"].get("enable_smoothing", True),
        )

    path = path_from_raw_waypoints(waypoints_list)
    scrub_ctx = {"group_key": group_key, "metadata": metadata, "path": path}
    marker_ids: dict = {"apple": -1, "goal": -1}

    obs = bootstrap_rollout(
        env, scrub_ctx["metadata"], scrub_ctx["path"], physics_settle=args.physics_settle
    )
    refresh_apple_goal_markers(env, scrub_ctx["metadata"], marker_ids)
    obs = dg.ensure_optical_flow(obs, optical_flow_model)
    if dg.CONFIG["simulation_setup"]["renders"]:
        dg.visualize_actions_camera(obs)
        try:
            cv2.waitKey(1)
        except Exception:
            pass

    path_len = len(scrub_ctx["path"])
    log.info(
        "Scrubber ready run_id=%s group=%s SUCCESS_group %s/%s path_len=%s refined=%s physics_settle=%s renders=%s",
        run_id,
        scrub_ctx["group_key"],
        success_idx_holder[0] + 1,
        len(success_keys),
        path_len,
        args.refine,
        args.physics_settle,
        dg.CONFIG["simulation_setup"]["renders"],
    )
    log_scrub_telemetry(
        run_id=run_id,
        group_key=scrub_ctx["group_key"],
        idx=0,
        path_len=path_len,
        refined=args.refine,
        env=env,
        obs=obs,
    )

    state = {"idx": 0, "busy": False}

    root = tk.Tk()
    root.title("Waypoint scrubber (↑↓ group · ← → waypoint)")
    tk.Label(
        root,
        text=(
            "↑ Next SUCCESS group    ↓ Previous SUCCESS group\n"
            "← Previous waypoint   → Next waypoint\n"
            "Home / End — first / last waypoint\n"
            "Close window to exit."
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

                if len(w) < 1:
                    log.warning("Group %r has no waypoints; staying on current group.", key)
                    return

                new_path = path_from_raw_waypoints(w.tolist())
                success_idx_holder[0] = new_idx
                scrub_ctx["group_key"] = key
                scrub_ctx["metadata"] = md
                scrub_ctx["path"] = new_path

                obs2 = bootstrap_rollout(env, md, new_path, physics_settle=args.physics_settle)
                refresh_apple_goal_markers(env, md, marker_ids)
                obs2 = dg.ensure_optical_flow(obs2, optical_flow_model)
                state["idx"] = 0

                if dg.CONFIG["simulation_setup"]["renders"]:
                    dg.visualize_actions_camera(obs2)
                    try:
                        cv2.waitKey(1)
                    except Exception:
                        pass

                log.info(
                    "Switched SUCCESS group %s/%s key=%r path_len=%s",
                    new_idx + 1,
                    len(success_keys),
                    key,
                    len(new_path),
                )
                log_scrub_telemetry(
                    run_id=run_id,
                    group_key=key,
                    idx=0,
                    path_len=len(new_path),
                    refined=args.refine,
                    env=env,
                    obs=obs2,
                )
                root.update_idletasks()
            finally:
                state["busy"] = False

        root.after(0, work)

    def goto_idx(new_idx: int, reason: str):
        if state["busy"]:
            return
        pl = len(scrub_ctx["path"])
        new_idx = int(np.clip(new_idx, 0, max(pl - 1, 0)))
        if new_idx == state["idx"] and reason != "init":
            log.debug("scrub clamp/boundary idx=%s (%s)", new_idx, reason)
            return

        state["busy"] = True

        def work():
            try:
                state["idx"] = new_idx
                obs2 = apply_waypoint_index(
                    env,
                    scrub_ctx["path"],
                    state["idx"],
                    optical_flow_model,
                    physics_settle=args.physics_settle,
                )
                log_scrub_telemetry(
                    run_id=run_id,
                    group_key=scrub_ctx["group_key"],
                    idx=state["idx"],
                    path_len=len(scrub_ctx["path"]),
                    refined=args.refine,
                    env=env,
                    obs=obs2,
                )
                root.update_idletasks()
            finally:
                state["busy"] = False

        root.after(0, work)

    def on_left(_e=None):
        goto_idx(state["idx"] - 1, "left")

    def on_right(_e=None):
        goto_idx(state["idx"] + 1, "right")

    def on_home(_e=None):
        goto_idx(0, "home")

    def on_end(_e=None):
        goto_idx(len(scrub_ctx["path"]) - 1, "end")

    root.bind("<Up>", lambda e: change_success_group(+1))
    root.bind("<Down>", lambda e: change_success_group(-1))
    root.bind("<Left>", on_left)
    root.bind("<Right>", on_right)
    root.bind("<Home>", on_home)
    root.bind("<End>", on_end)

    root.mainloop()


if __name__ == "__main__":
    main()
