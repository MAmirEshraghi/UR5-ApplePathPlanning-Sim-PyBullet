"""Compare expert vs policy actions at t=0 on sim obs."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch as th

from debug._common import (
    action_l2,
    load_policy,
    run_with_session_log,
    sample_episode_keys,
    setup_env,
    sim_obs_step0,
)


def _interpret(mean_l2: float) -> str:
    if mean_l2 < 0.1:
        return "GOOD — policy matches expert at t=0 on sim obs"
    if mean_l2 < 0.3:
        return "MARGINAL — small t=0 mismatch on sim obs"
    return "BAD — large t=0 mismatch on sim obs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-0 expert vs policy on sim observations.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_cached_optical_flow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="False=live RAFT on sim rgb (default, matches tree08 v2 training).",
    )
    args = parser.parse_args()

    def _run() -> dict:
        episode_keys, metadata_cache = sample_episode_keys(args.run_id, args.n_episodes, args.seed)
        env, _, hdf5_path = setup_env(args.run_id)
        print(
            f"[diag:sim_step0] checkpoint={Path(args.checkpoint).resolve()} "
            f"n={len(episode_keys)} seed={args.seed} "
            f"use_cached_optical_flow={args.use_cached_optical_flow}",
            flush=True,
        )
        model = load_policy(
            args.checkpoint,
            env,
            hdf5_path,
            use_cached_optical_flow=args.use_cached_optical_flow,
        )
        model.policy.use_cached_optical_flow = args.use_cached_optical_flow
        model.policy.set_training_mode(False)
        proj = getattr(model.policy, "actor_feature_proj", None)
        if isinstance(proj, th.nn.Linear):
            print(
                f"[diag:sim_step0] actor_feature_proj=Linear in={proj.in_features} out={proj.out_features} "
                f"weight_norm={float(proj.weight.norm().item()):.4f}",
                flush=True,
            )

        records = []
        try:
            with h5py.File(hdf5_path, "r") as f:
                for traj_key in episode_keys:
                    sim_obs = sim_obs_step0(env, metadata_cache[traj_key])
                    obs = {k: np.stack([v]) for k, v in sim_obs.items()}
                    expert_action = np.asarray(f[traj_key]["actions"][0], dtype=np.float32)
                    policy_action, _ = model.predict(
                        obs, state=None, episode_start=np.array([True]), deterministic=True
                    )
                    policy_action = np.asarray(policy_action[0], dtype=np.float32)
                    mse, l2 = action_l2(policy_action, expert_action)
                    records.append(
                        {
                            "traj": traj_key,
                            "expert_action": expert_action.tolist(),
                            "policy_action": policy_action.tolist(),
                            "mse": mse,
                            "l2": l2,
                        }
                    )
                    print(
                        f"[diag:sim_step0] traj={traj_key} mse={mse:.5f} l2={l2:.4f} "
                        f"expert={np.round(expert_action, 3).tolist()} "
                        f"policy={np.round(policy_action, 3).tolist()}",
                        flush=True,
                    )
        finally:
            env.close()

        mean_l2 = float(np.mean([r["l2"] for r in records]))
        mean_mse = float(np.mean([r["mse"] for r in records]))
        verdict = _interpret(mean_l2)
        summary = {"mean_mse": mean_mse, "mean_l2": mean_l2, "verdict": verdict}
        print(f"[diag:sim_step0:summary] {summary}", flush=True)
        return {
            "script": "diag_policy_step0",
            "run_id": args.run_id,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "seed": args.seed,
            "summary": summary,
            "episodes": records,
        }

    run_with_session_log("diag_policy_step0", args.run_id, _run)


if __name__ == "__main__":
    main()
