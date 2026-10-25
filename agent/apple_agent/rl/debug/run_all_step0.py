#!/usr/bin/env python3
"""Run all step-0 diagnostics in order (#1 obs, #2 hdf5 policy, #3 sim policy)."""
import argparse
import subprocess
import sys



def main() -> None:
    parser = argparse.ArgumentParser(description="Run all step-0 debug diagnostics.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scripts = [
        "diag_obs_step0",
        "diag_policy_hdf5_obs_step0",
        "diag_policy_step0",
    ]

    for name in scripts:
        cmd = [
            sys.executable,
            "-m",
            f"apple_agent.rl.debug.{name}",
            "--run_id",
            args.run_id,
            "--n_episodes",
            str(args.n_episodes),
            "--seed",
            str(args.seed),
        ]
        if "policy" in name:
            cmd += ["--checkpoint", args.checkpoint]
        print(f"\n=== running {name} ===", flush=True)
        subprocess.run(cmd, check=True)  # same workspace (cwd) and environment as this launcher
    print("\n[debug:run_all] done — see logs/debug/ and logs/debug/json/", flush=True)


if __name__ == "__main__":
    main()
