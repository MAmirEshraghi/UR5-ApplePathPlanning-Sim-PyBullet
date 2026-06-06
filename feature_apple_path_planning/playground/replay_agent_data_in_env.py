#!/usr/bin/env python3
"""
Replay generated agent-data trajectories in ApplePickingEnv automatically.

This script is non-interactive by default: it loads one trajectory from
output/agent_data/<run_id>.hdf5, resets the environment, and replays all saved
actions with env.step().

Run from repository root:

    PYTHONPATH=. python3 feature_apple_path_planning/playground/replay_agent_data_in_env.py \
        --run-id 20260504_203032

Choose a specific trajectory:

    PYTHONPATH=. python3 feature_apple_path_planning/playground/replay_agent_data_in_env.py \
        --run-id 20260504_203032 --group-index 0
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import logging
import re
import sys
import time
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


def setup_logging() -> Path:
    base = _REPO_ROOT / "logs" / dg.LOG_SESSION_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_agent_data_replay.log").resolve()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.addHandler(fh)
    zen_logger.setLevel(logging.DEBUG)
    log.info("Agent-data replay log file: %s", log_path)
    return log_path


def sort_group_keys(keys: list[str]) -> list[str]:
    def _sort_key(k: str):
        try:
            return (0, float(k))
        except ValueError:
            return (1, str(k))

    return sorted(keys, key=_sort_key)


def resolve_hdf5_path(run_id: str | None, hdf5_path: str | None) -> Path:
    if hdf5_path:
        p = Path(hdf5_path).expanduser().resolve()
    elif run_id:
        p = (_REPO_ROOT / "output" / "agent_data" / f"{run_id}.hdf5").resolve()
    else:
        raise ValueError("Provide either --run-id or --hdf5-path.")
    if not p.exists():
        raise FileNotFoundError(f"Agent data HDF5 not found: {p}")
    return p


def choose_group(group_keys: list[str], group_index: int | None, group_key: str | None) -> tuple[int, str]:
    if not group_keys:
        raise RuntimeError("No trajectory groups found in HDF5.")
    if group_key is not None:
        if group_key not in group_keys:
            raise KeyError(f"group-key {group_key!r} not found.")
        idx = group_keys.index(group_key)
        return idx, group_key
    idx = int(group_index or 0)
    idx = max(0, min(idx, len(group_keys) - 1))
    return idx, group_keys[idx]


def as_vec3(v, default=None) -> np.ndarray:
    if v is None:
        if default is None:
            raise ValueError("Missing required vec3 value.")
        return np.asarray(default, dtype=np.float32).reshape(3,)
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str):
        # Support stringified arrays/lists in attrs, e.g. "[0.1 0.2 0.3]".
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        if len(nums) >= 3:
            arr = np.asarray([float(nums[0]), float(nums[1]), float(nums[2])], dtype=np.float32).reshape(3,)
            return arr
        if default is None:
            raise ValueError(f"Could not parse vec3 from string: {v!r}")
        return np.asarray(default, dtype=np.float32).reshape(3,)

    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"Expected vec3-like value, got shape {arr.shape}")
    return arr[:3].reshape(3,)


def _get_start_joint_angles(obs_grp: h5py.Group, use_obs_joint_start: bool) -> np.ndarray:
    if use_obs_joint_start and "joint_angles" in obs_grp:
        q_obs = np.asarray(obs_grp["joint_angles"][0], dtype=np.float64).reshape(-1)
        if q_obs.size >= 6:
            return q_obs[:6]
    return np.asarray(dg.CONFIG["robot_setup"]["start_joint_angles"], dtype=np.float64).reshape(6,)


def bootstrap_env_from_trajectory(
    env: ApplePickingEnv, attrs: dict, obs_grp: h5py.Group, *, use_obs_joint_start: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    apple_center = None
    goal_pos = None
    if "apple_center" in attrs:
        apple_center = as_vec3(attrs["apple_center"])
    if "goal_pos" in attrs:
        goal_pos = as_vec3(attrs["goal_pos"])

    if apple_center is None:
        # Fallback from desired_goal in first obs.
        apple_center = as_vec3(obs_grp["desired_goal"][0], default=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if goal_pos is None:
        goal_pos = as_vec3(obs_grp["desired_goal"][0], default=np.array([0.0, 0.0, 1.0], dtype=np.float32))

    env.reset(seed=None, options={"target_apple": apple_center})
    env.desired_goal = apple_center.astype(np.float32, copy=False)
    env.reward_goal = goal_pos.astype(np.float32, copy=False)

    q_arm = _get_start_joint_angles(obs_grp, use_obs_joint_start)
    env.robot.set_joint_angles_no_collision(q_arm.tolist())
    env.step_counter = 0
    env.is_goal_state = False
    env.sum_reward = 0.0
    env.init_pos_ee, env.init_or_ee = env.robot.get_current_pose(env.robot.tool0_link_idx)
    obs0 = env._get_obs()
    rel0 = np.asarray(obs0.get("relative_distance", np.zeros(3, dtype=np.float32)), dtype=np.float64).reshape(-1)
    return q_arm, apple_center, rel0[:3] if rel0.size >= 3 else np.zeros(3, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay saved agent-data actions in ApplePickingEnv.")
    ap.add_argument("--run-id", type=str, default=None, help="Run ID for output/agent_data/<run_id>.hdf5")
    ap.add_argument("--hdf5-path", type=str, default=None, help="Explicit path to agent-data HDF5")
    ap.add_argument("--group-index", type=int, default=0, help="Sorted group index to replay")
    ap.add_argument("--group-key", type=str, default=None, help="Exact group key to replay")
    ap.add_argument("--loop", action="store_true", help="Loop trajectory replay until interrupted")
    ap.add_argument("--realtime-scale", type=float, default=1.0, help="Sleep multiplier over control_time")
    ap.add_argument("--max-steps", type=int, default=-1, help="Limit replay to first N actions (-1 = full)")
    ap.add_argument("--no-camera-visualize", action="store_true", help="Disable camera frame visualization")
    ap.add_argument(
        "--use-obs-joint-start",
        action="store_true",
        help="Initialize from observations/joint_angles[0][:6] instead of CONFIG start joints.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    h5_path = resolve_hdf5_path(args.run_id, args.hdf5_path)
    log.info("Opening agent-data file: %s", h5_path)

    env = ApplePickingEnv(config=dg.CONFIG)
    control_time = float(dg.CONFIG["simulation_setup"]["control_time"])

    try:
        with h5py.File(h5_path, "r") as f:
            group_keys = sort_group_keys(list(f.keys()))
            group_idx, group_key = choose_group(group_keys, args.group_index, args.group_key)
            grp = f[group_key]

            if "actions" not in grp or "observations" not in grp or "next_observations" not in grp:
                raise RuntimeError(f"Group {group_key!r} missing required datasets.")

            actions = np.asarray(grp["actions"], dtype=np.float64)
            rewards_saved = np.asarray(grp["rewards"], dtype=np.float64) if "rewards" in grp else None
            dones_saved = np.asarray(grp["dones"], dtype=np.bool_) if "dones" in grp else None
            obs_grp = grp["observations"]
            next_obs_grp = grp["next_observations"]
            attrs = dict(grp.attrs.items())

            if actions.ndim != 2 or actions.shape[1] != 6:
                raise ValueError(f"Expected actions shape (T,6), got {actions.shape}")
            total = int(actions.shape[0])
            if total <= 0:
                raise RuntimeError("No actions to replay.")

            steps = total if args.max_steps < 0 else min(total, int(args.max_steps))
            replay_id = 0
            while True:
                replay_id += 1
                q_start, apple_center, rel0 = bootstrap_env_from_trajectory(
                    env, attrs, obs_grp, use_obs_joint_start=args.use_obs_joint_start
                )
                saved_rel0 = None
                if "relative_distance" in obs_grp:
                    saved_rel0_raw = np.asarray(obs_grp["relative_distance"][0], dtype=np.float64).reshape(-1)
                    if saved_rel0_raw.size >= 3:
                        saved_rel0 = saved_rel0_raw[:3]
                init_rel_l1_err = float(np.linalg.norm((rel0 - saved_rel0), ord=1)) if saved_rel0 is not None else float("nan")
                log.info(
                    "Replay start group=%s index=%d/%d replay_id=%d steps=%d q_start=%s apple_center=%s init_rel_l1_err=%s",
                    group_key,
                    group_idx + 1,
                    len(group_keys),
                    replay_id,
                    steps,
                    np.array2string(q_start, precision=5),
                    np.array2string(apple_center, precision=5),
                    "n/a" if saved_rel0 is None else f"{init_rel_l1_err:.8f}",
                )

                reward_diff_abs_sum = 0.0
                rel_dist_diff_abs_sum = 0.0
                done_mismatch_count = 0

                for i in range(steps):
                    action = actions[i]
                    obs_next, reward, terminated, truncated, info = env.step(action)

                    if not args.no_camera_visualize:
                        dg.visualize_actions_camera(obs_next)
                        cv2.waitKey(1)

                    if rewards_saved is not None:
                        reward_diff_abs_sum += abs(float(reward) - float(rewards_saved[i]))

                    if "relative_distance" in next_obs_grp and "relative_distance" in obs_next:
                        saved_rd = np.asarray(next_obs_grp["relative_distance"][i], dtype=np.float64).reshape(-1)
                        sim_rd = np.asarray(obs_next["relative_distance"], dtype=np.float64).reshape(-1)
                        if saved_rd.size >= 3 and sim_rd.size >= 3:
                            rel_dist_diff_abs_sum += float(np.linalg.norm(saved_rd[:3] - sim_rd[:3], ord=1))

                    done_now = bool(terminated or truncated)
                    if dones_saved is not None and done_now != bool(dones_saved[i]):
                        done_mismatch_count += 1

                    log.info(
                        "replay_step=%d/%d reward=%.6f saved_reward=%s done=%s saved_done=%s rel_dist=%s",
                        i + 1,
                        steps,
                        float(reward),
                        "n/a" if rewards_saved is None else f"{float(rewards_saved[i]):.6f}",
                        done_now,
                        "n/a" if dones_saved is None else bool(dones_saved[i]),
                        np.array2string(np.asarray(obs_next.get("relative_distance", [])), precision=5),
                    )

                    if args.realtime_scale > 0:
                        time.sleep(control_time * float(args.realtime_scale))

                    if done_now:
                        log.info("Replay terminated at step %d.", i + 1)
                        break

                replayed_steps = i + 1 if steps > 0 else 0
                mean_reward_abs_err = reward_diff_abs_sum / max(1, replayed_steps)
                mean_rel_dist_l1 = rel_dist_diff_abs_sum / max(1, replayed_steps)
                log.info(
                    "Replay summary: steps=%d mean_abs_reward_err=%.8f mean_rel_dist_l1=%.8f done_mismatch_count=%d",
                    replayed_steps,
                    mean_reward_abs_err,
                    mean_rel_dist_l1,
                    done_mismatch_count,
                )

                if not args.loop:
                    break

    finally:
        env.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
