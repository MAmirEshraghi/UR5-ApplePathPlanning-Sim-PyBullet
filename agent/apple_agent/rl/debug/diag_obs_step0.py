"""Compare HDF5 expert obs vs sim obs at t=0 (diagnostic #1)."""
import argparse
from pathlib import Path

import h5py

from debug._common import (
    compare_obs_pair,
    hdf5_obs_step0,
    run_with_session_log,
    sample_episode_keys,
    setup_env,
    sim_obs_step0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare expert HDF5 vs sim observations at t=0.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def _run() -> dict:
        episode_keys, metadata_cache = sample_episode_keys(args.run_id, args.n_episodes, args.seed)
        env, _, hdf5_path = setup_env(args.run_id)
        print(
            f"[diag:obs] run_id={args.run_id} n={len(episode_keys)} seed={args.seed}",
            flush=True,
        )
        records = []
        try:
            with h5py.File(hdf5_path, "r") as f:
                for traj_key in episode_keys:
                    expert_obs = hdf5_obs_step0(f, traj_key)
                    sim_obs = sim_obs_step0(env, metadata_cache[traj_key])
                    fields, summary = compare_obs_pair(expert_obs, sim_obs)
                    rec = {"traj": traj_key, "summary": summary, "fields": fields}
                    records.append(rec)
                    print(
                        f"[diag:obs] traj={traj_key} "
                        f"tabular_mad={summary['mean_tabular_mean_abs_diff']:.6f} "
                        f"rgb_mad={summary['mean_rgb_mean_abs_diff']:.2f} "
                        f"expert_flow={summary['expert_has_optical_flow']} sim_flow={summary['sim_has_optical_flow']}",
                        flush=True,
                    )
                    for row in fields:
                        if row.get("key") in (
                            "joint_angles",
                            "relative_distance",
                            "desired_goal",
                            "achieved_goal",
                        ) and row.get("shape_match"):
                            print(
                                f"  {row['key']}: mad={row['mean_abs_diff']:.6f} l2={row.get('l2_diff', 0):.6f}",
                                flush=True,
                            )
        finally:
            env.close()

        tab_mads = [r["summary"]["mean_tabular_mean_abs_diff"] for r in records]
        rgb_mads = [r["summary"]["mean_rgb_mean_abs_diff"] for r in records]
        overall = {
            "mean_tabular_mean_abs_diff": float(sum(tab_mads) / len(tab_mads)),
            "mean_rgb_mean_abs_diff": float(sum(rgb_mads) / len(rgb_mads)),
        }
        if overall["mean_tabular_mean_abs_diff"] < 0.01 and overall["mean_rgb_mean_abs_diff"] < 5:
            verdict = "GOOD — sim obs matches HDF5 at t=0"
        elif overall["mean_tabular_mean_abs_diff"] < 0.05:
            verdict = "MARGINAL — tabular close; check rgb/flow"
        else:
            verdict = "BAD — sim obs differs from HDF5; fix reconfigure/obs pipeline"
        overall["verdict"] = verdict
        print(f"[diag:obs:summary] {overall} verdict={verdict}", flush=True)
        return {
            "script": "diag_obs_step0",
            "run_id": args.run_id,
            "seed": args.seed,
            "summary": overall,
            "episodes": records,
        }

    run_with_session_log("diag_obs_step0", args.run_id, _run)


if __name__ == "__main__":
    main()
