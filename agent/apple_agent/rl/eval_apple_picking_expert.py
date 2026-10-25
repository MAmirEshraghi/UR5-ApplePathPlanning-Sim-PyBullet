"""
Pure evaluation for apple-picking checkpoints — sim rollouts only, no training updates.
"""
import argparse
import datetime
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


from apple_agent.envs.apple_picking_env import ApplePickingEnv
from apple_agent.config import CONFIG as APPLE_PICKING_CONFIG
from apple_agent.data import agent_data
from apple_agent.apple_picking_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAEWithExpert
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import json_to_data
from stable_baselines3.common.vec_env import DummyVecEnv

from apple_agent.rl.train_apple_picking_expert import _apply_tree_setup_for_training, _log_train_data


def _load_expert_metadata(hdf5_path: str) -> Tuple[List[str], Dict[str, dict]]:
    with h5py.File(hdf5_path, "r") as f:
        trajectory_keys = list(f.keys())
        metadata_cache = {key: agent_data.trajectory_metadata(f[key]) for key in trajectory_keys}
    return trajectory_keys, metadata_cache


def checkpoint_image_channels(checkpoint_path: Path) -> tuple:
    """Image channels the checkpoint's policy was built with (read without loading weights)."""
    with zipfile.ZipFile(checkpoint_path) as archive:
        data = json_to_data(archive.read("data").decode())
    policy_kwargs = data.get("policy_kwargs", {})
    if "image_channels" not in policy_kwargs:
        raise ValueError(f"{checkpoint_path} predates image_channels / agent-data schema v2; retrain it.")
    return tuple(policy_kwargs["image_channels"])


def _vec_obs_from_env(env) -> dict:
    obs_list = env.env_method("_get_obs")
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _reconfigure_to_trajectory(env, metadata: dict) -> None:
    env.env_method("reconfigure_scene", metadata=metadata)
    env.env_method("reset_env_variables")
    env.env_method("snapshot_episode_start_dist")
    # SB3 Monitor requires reset before step; reconfigure_scene bypasses reset().
    for sub_env in env.envs:
        if hasattr(sub_env, "needs_reset"):
            sub_env.needs_reset = False


def _classify_episode(info: dict) -> str:
    if info.get("goal_achieved", False):
        return "GOAL"
    if info.get("collision_terminated", False):
        return "COLLISION"
    return "TIMEOUT"


def _tree_label(metadata: dict) -> str:
    if metadata.get("tree_id_str") is not None:
        return str(metadata["tree_id_str"])
    if metadata.get("tree_id") is not None:
        return f"tree_id={int(np.asarray(metadata['tree_id']).reshape(-1)[0])}"
    return "unknown"


def run_episode(
    model,
    env,
    metadata: dict,
    *,
    deterministic: bool = True,
    max_steps: int = 500,
) -> Tuple[dict, float]:
    _reconfigure_to_trajectory(env, metadata)
    obs = _vec_obs_from_env(env)
    lstm_states = None
    episode_start = np.array([True])
    ep_reward = 0.0
    last_info: dict = {}

    for step_i in range(max_steps):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        episode_start = np.array([False])
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward[0])
        last_info = dict(info[0])
        if bool(done[0]):
            break
    else:
        last_info.setdefault("episode_length", step_i + 1)

    return last_info, ep_reward


def _summarize_episodes(episodes: List[dict]) -> dict:
    n = len(episodes)
    if n == 0:
        return {
            "n_episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
        }

    goals = [e["reason"] == "GOAL" for e in episodes]
    collisions = [e["reason"] == "COLLISION" for e in episodes]
    timeouts = [e["reason"] == "TIMEOUT" for e in episodes]
    goal_lens = [e["ep_len"] for e in episodes if e["reason"] == "GOAL"]

    return {
        "n_episodes": n,
        "success_rate": float(np.mean(goals)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_final_ee_dist": float(np.mean([e["final_ee_dist"] for e in episodes])),
        "mean_ep_len": float(np.mean([e["ep_len"] for e in episodes])),
        "mean_ep_rew": float(np.mean([e["ep_rew"] for e in episodes])),
        "goal_ep_len_mean": float(np.mean(goal_lens)) if goal_lens else None,
        "n_goals": int(np.sum(goals)),
    }


def _per_tree_summary(episodes: List[dict]) -> Dict[str, dict]:
    by_tree: Dict[str, List[dict]] = {}
    for ep in episodes:
        by_tree.setdefault(ep["tree"], []).append(ep)
    return {tree: _summarize_episodes(eps) for tree, eps in sorted(by_tree.items())}


DEFAULT_EVAL_OUTPUT_DIR = Path("logs/eval")
DEFAULT_EVAL_OUTPUT_JSON = DEFAULT_EVAL_OUTPUT_DIR / "final_model_all.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate apple-picking checkpoint in sim (no training).")
    parser.add_argument("--run_id", required=True, help="Run ID for output/agent_data/{run_id}.hdf5")
    parser.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of expert trajectories to eval")
    parser.add_argument(
        "--eval_all",
        action="store_true",
        help="Evaluate every trajectory in the expert HDF5 (overrides --n_episodes)",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions (default: on)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for trajectory sampling")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Episode step cap (default: CONFIG simulation_setup.max_steps = 350). "
             "~1.3x the longest expert trajectory is enough to judge reaching.",
    )
    parser.add_argument(
        "--renders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable PyBullet GUI (default: off)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EVAL_OUTPUT_JSON),
        help=f"JSON path for results (default: {DEFAULT_EVAL_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        help="Skip writing JSON results file",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Append stdout to this file (default: logs/eval_apple_picking_expert/<ts>_<run_id>.log)",
    )
    args = parser.parse_args()

    DEFAULT_EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path: Optional[Path] = None
    if args.log_file:
        log_path = Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = (Path("logs") / "eval_apple_picking_expert" / f"{ts}_{args.run_id}.log").resolve()
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

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_channels = checkpoint_image_channels(checkpoint_path)
    training_config = {
        **APPLE_PICKING_CONFIG,
        "simulation_setup": {
            **APPLE_PICKING_CONFIG["simulation_setup"],
            "image_channels": image_channels,
            "renders": args.renders,
            **({"max_steps": args.max_steps} if args.max_steps else {}),
            "draw_apple_centroid_markers": not args.renders,
            "draw_trajectory_target_markers": args.renders,
            "terminate_on_self_collision": True,
        },
    }
    _apply_tree_setup_for_training(training_config, args.run_id, agent_data_path)
    _log_train_data(agent_data_path, training_config)

    expert_hdf5 = str(agent_data_path.resolve())
    trajectory_keys, metadata_cache = _load_expert_metadata(expert_hdf5)

    if args.eval_all:
        episode_keys = list(trajectory_keys)
    else:
        rng = random.Random(args.seed)
        n = min(args.n_episodes, len(trajectory_keys))
        episode_keys = rng.sample(trajectory_keys, n)

    print(
        f"[eval:config] checkpoint={checkpoint_path} n_episodes={len(episode_keys)} "
        f"deterministic={args.deterministic} seed={args.seed} "
        f"image_channels={image_channels}",
        flush=True,
    )

    print("INFO: Building env (PyBullet load can take 1–3+ minutes)…", flush=True)
    env = make_vec_env(
        ApplePickingEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"config": training_config},
    )
    print("INFO: Environment ready.", flush=True)

    print(f"INFO: Loading checkpoint {checkpoint_path}…", flush=True)
    model = RecurrentPPOAEWithExpert.load(
        str(checkpoint_path),
        env=env,
        use_online_data=False,
        use_offline_data=False,
        use_ppo_offline=False,
        use_online_bc=True,
        use_awac=False,
        path_trajectories=expert_hdf5,
        algo_size=(224, 224),
        bc_only=False,
        device="cuda",
    )
    model.policy.set_training_mode(False)

    max_steps = int(training_config["simulation_setup"].get("max_steps", 500))
    episodes: List[dict] = []

    for i, traj_key in enumerate(episode_keys):
        meta = metadata_cache[traj_key]
        info, ep_rew = run_episode(
            model,
            env,
            meta,
            deterministic=args.deterministic,
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
        }
        episodes.append(record)

        print(
            f"[eval:episode] {i + 1}/{len(episode_keys)} traj={traj_key} tree={tree} "
            f"reason={reason} len={ep_len} rew={ep_rew:.2f} "
            f"ee_dist={final_dist:.4f} start_dist={start_dist:.4f}",
            flush=True,
        )

    summary = _summarize_episodes(episodes)
    per_tree = _per_tree_summary(episodes)

    print(
        f"[eval:summary] n={summary['n_episodes']} goals={summary['n_goals']} "
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
                f"[eval:tree] {tree} n={stats['n_episodes']} "
                f"success_rate={stats['success_rate']:.3f} "
                f"collision_rate={stats['collision_rate']:.3f}",
                flush=True,
            )

    if not args.no_json:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "run_id": args.run_id,
            "deterministic": args.deterministic,
            "seed": args.seed,
            "summary": summary,
            "per_tree": per_tree,
            "episodes": episodes,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval:output] saved {out_path}", flush=True)

    env.close()
    sys.stdout = orig_stdout
    log_fp.close()


if __name__ == "__main__":
    main()
