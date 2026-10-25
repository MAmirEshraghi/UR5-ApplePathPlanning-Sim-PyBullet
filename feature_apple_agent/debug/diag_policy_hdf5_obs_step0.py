"""Policy predict on HDF5 t=0 obs vs expert action (diagnostic #2)."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch as th

from debug._common import (
    action_l2,
    hdf5_obs_for_policy,
    hdf5_obs_step0,
    load_policy,
    predict_policy,
    run_with_session_log,
    sample_episode_keys,
    setup_env,
)


def _interpret(mean_l2: float) -> str:
    if mean_l2 < 0.1:
        return "GOOD — policy matches expert on HDF5 obs (sim/vision mismatch likely)"
    if mean_l2 < 0.3:
        return "MARGINAL — partial match on HDF5 obs"
    return "BAD — policy does not match expert even on HDF5 obs (weights/training)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy vs expert at t=0 using HDF5 observations.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def _run() -> dict:
        episode_keys, _ = sample_episode_keys(args.run_id, args.n_episodes, args.seed)
        env, _, hdf5_path = setup_env(args.run_id, checkpoint=args.checkpoint)
        print(
            f"[diag:hdf5_obs] checkpoint={Path(args.checkpoint).resolve()} "
            f"n={len(episode_keys)} seed={args.seed}",
            flush=True,
        )
        model = load_policy(args.checkpoint, env, hdf5_path)
        model.policy.set_training_mode(False)
        proj = getattr(model.policy, "actor_feature_proj", None)
        if isinstance(proj, th.nn.Linear):
            print(
                f"[diag:hdf5_obs] actor_feature_proj=Linear in={proj.in_features} out={proj.out_features}",
                flush=True,
            )

        records = []
        try:
            with h5py.File(hdf5_path, "r") as f:
                for traj_key in episode_keys:
                    obs = hdf5_obs_for_policy(model, hdf5_obs_step0(f, traj_key))
                    expert_action = np.asarray(f[traj_key]["actions"][0], dtype=np.float32)
                    policy_action, _ = predict_policy(
                        model, obs, state=None, episode_start=np.array([True]), deterministic=True
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
                        f"[diag:hdf5_obs] traj={traj_key} mse={mse:.5f} l2={l2:.4f} "
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
        print(f"[diag:hdf5_obs:summary] {summary}", flush=True)
        return {
            "script": "diag_policy_hdf5_obs_step0",
            "run_id": args.run_id,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "seed": args.seed,
            "summary": summary,
            "episodes": records,
        }

    run_with_session_log("diag_policy_hdf5_obs_step0", args.run_id, _run)


if __name__ == "__main__":
    main()
