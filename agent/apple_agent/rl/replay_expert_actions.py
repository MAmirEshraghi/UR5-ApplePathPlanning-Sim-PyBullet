"""
Open-loop replay of expert actions from HDF5 in ApplePickingEnv (no policy).

Use this before policy eval to check whether the sim/env matches data collection:
if expert replay fails here, the problem is env mismatch — not RL training.
"""
import argparse
import datetime
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np


from apple_agent.envs.apple_picking_env import ApplePickingEnv
from apple_agent.config import CONFIG as APPLE_PICKING_CONFIG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from apple_agent.rl.eval_apple_picking_expert import (
    _classify_episode,
    _load_expert_metadata,
    _per_tree_summary,
    _reconfigure_to_trajectory,
    _summarize_episodes,
    _tree_label,
)
from apple_agent.rl.train_apple_picking_expert import _apply_tree_setup_for_training, _log_train_data


def run_expert_replay_episode(
    env,
    hdf5_path: Path,
    traj_key: str,
    metadata: dict,
    *,
    max_steps: int,
) -> Tuple[dict, float, int]:
    """Replay stored expert actions open-loop; return (last_info, ep_reward, n_actions)."""
    _reconfigure_to_trajectory(env, metadata)
    ep_reward = 0.0
    last_info: dict = {}

    with h5py.File(hdf5_path, "r") as f:
        actions = np.asarray(f[traj_key]["actions"][:], dtype=np.float32)

    n_actions = len(actions)
    for step_i, action in enumerate(actions):
        if step_i >= max_steps:
            break
        action_batch = np.asarray(action, dtype=np.float32).reshape(1, -1)
        _, reward, done, info = env.step(action_batch)
        ep_reward += float(reward[0])
        last_info = dict(info[0])
        if bool(done[0]):
            break
    else:
        last_info.setdefault("episode_length", min(n_actions, max_steps))

    return last_info, ep_reward, n_actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay expert HDF5 actions open-loop in sim (diagnostic, no checkpoint)."
    )
    parser.add_argument("--run_id", required=True, help="Run ID for output/agent_data/{run_id}.hdf5")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of trajectories to replay")
    parser.add_argument(
        "--eval_all",
        action="store_true",
        help="Replay every trajectory in the HDF5 (overrides --n_episodes)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for trajectory sampling")
    parser.add_argument(
        "--renders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable PyBullet GUI (default: off)",
    )
    parser.add_argument(
        "--output",
        default="logs/eval/expert_replay_seed0.json",
        help="JSON path for results",
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        help="Skip writing JSON results file",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Session log path (default: logs/replay_expert_actions/<ts>_<run_id>.log)",
    )
    args = parser.parse_args()

    log_path: Optional[Path] = None
    if args.log_file:
        log_path = Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = (Path("logs") / "replay_expert_actions" / f"{ts}_{args.run_id}.log").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout

    class _Tee:
        def write(self, data):
            orig_stdout.write(data)
            log_fp.write(data)

        def flush(self):
            orig_stdout.flush()
            log_fp.flush()

    sys.stdout = _Tee()
    print(f"Session log: {log_path}", flush=True)

    agent_data_path = Path("output") / "agent_data" / f"{args.run_id}.hdf5"
    if not agent_data_path.is_file():
        raise FileNotFoundError(f"Agent data not found: {agent_data_path}")

    training_config = {
        **APPLE_PICKING_CONFIG,
        "simulation_setup": {
            **APPLE_PICKING_CONFIG["simulation_setup"],
            "renders": args.renders,
            "draw_apple_centroid_markers": True,
            "terminate_on_self_collision": True,
        },
    }
    _apply_tree_setup_for_training(training_config, args.run_id, agent_data_path)
    _log_train_data(agent_data_path, training_config)

    trajectory_keys, metadata_cache = _load_expert_metadata(str(agent_data_path.resolve()))

    if args.eval_all:
        episode_keys = list(trajectory_keys)
    else:
        rng = random.Random(args.seed)
        n = min(args.n_episodes, len(trajectory_keys))
        episode_keys = rng.sample(trajectory_keys, n)

    print(
        f"[replay:config] mode=open_loop_expert hdf5={agent_data_path.resolve()} "
        f"n_episodes={len(episode_keys)} seed={args.seed}",
        flush=True,
    )
    print(f"[replay:config] episode_keys={episode_keys}", flush=True)

    print("INFO: Building env (PyBullet load can take 1–3+ minutes)…", flush=True)
    env = make_vec_env(
        ApplePickingEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"config": training_config},
    )
    print("INFO: Environment ready.", flush=True)

    max_steps = int(training_config["simulation_setup"].get("max_steps", 500))
    episodes: List[dict] = []

    for i, traj_key in enumerate(episode_keys):
        meta = metadata_cache[traj_key]
        info, ep_rew, n_actions = run_expert_replay_episode(
            env,
            agent_data_path,
            traj_key,
            meta,
            max_steps=max_steps,
        )
        reason = _classify_episode(info)
        ep_len = int(info.get("episode_length", 0))
        final_dist = float(info.get("dist_to_goal", float("nan")))
        start_dist = float(info.get("start_dist_to_goal", float("nan")))
        tree = _tree_label(meta)

        record = {
            "traj": traj_key,
            "tree": tree,
            "reason": reason,
            "ep_len": ep_len,
            "ep_rew": ep_rew,
            "final_ee_dist": final_dist,
            "start_ee_dist": start_dist,
            "n_expert_actions": n_actions,
        }
        episodes.append(record)

        print(
            f"[replay:episode] {i + 1}/{len(episode_keys)} traj={traj_key} tree={tree} "
            f"reason={reason} len={ep_len} n_actions={n_actions} rew={ep_rew:.2f} "
            f"ee_dist={final_dist:.4f} start_dist={start_dist:.4f}",
            flush=True,
        )

    summary = _summarize_episodes(episodes)
    per_tree = _per_tree_summary(episodes)

    print(
        f"[replay:summary] n={summary['n_episodes']} goals={summary['n_goals']} "
        f"success_rate={summary['success_rate']:.3f} "
        f"collision_rate={summary['collision_rate']:.3f} "
        f"timeout_rate={summary['timeout_rate']:.3f} "
        f"mean_final_ee_dist={summary['mean_final_ee_dist']:.4f} "
        f"mean_ep_len={summary['mean_ep_len']:.1f} "
        f"goal_ep_len_mean={summary['goal_ep_len_mean']}",
        flush=True,
    )

    if len(per_tree) > 1:
        for tree, stats in per_tree.items():
            print(
                f"[replay:tree] {tree} n={stats['n_episodes']} "
                f"success_rate={stats['success_rate']:.3f} "
                f"collision_rate={stats['collision_rate']:.3f}",
                flush=True,
            )

    if not args.no_json:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "open_loop_expert",
            "run_id": args.run_id,
            "seed": args.seed,
            "episode_keys": episode_keys,
            "summary": summary,
            "per_tree": per_tree,
            "episodes": episodes,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[replay:output] saved {out_path}", flush=True)

    env.close()
    sys.stdout = orig_stdout
    log_fp.close()


if __name__ == "__main__":
    main()
