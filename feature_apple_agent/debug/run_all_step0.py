#!/usr/bin/env python3
"""Run all step-0 diagnostics in order (#1 obs, #2 hdf5 policy, #3 sim policy)."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _AGENT_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all step-0 debug diagnostics.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scripts = [
        "diag_obs_step0.py",
        "diag_policy_hdf5_obs_step0.py",
        "diag_policy_step0.py",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{_AGENT_DIR}"

    for name in scripts:
        script = _AGENT_DIR / "debug" / name
        cmd = [
            sys.executable,
            str(script),
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
        subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, check=True)
    print("\n[debug:run_all] done — see logs/debug/ and logs/debug/json/", flush=True)


if __name__ == "__main__":
    main()
