#!/usr/bin/env python3
"""
Closed-loop policy rollout: action L2 vs expert and ee_dist at each step.

Launcher only — imports PyTorch and initializes CUDA before any PyBullet import.
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _AGENT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _probe_cuda() -> tuple[bool, str | None]:
    """Return (ok, error_message). Must run before any PyBullet import."""
    import torch as th

    if not th.cuda.is_available():
        return False, "torch.cuda.is_available()=False (driver not loaded or GPU busy)"
    try:
        _ = th.zeros(1, device="cuda")
        th.cuda.synchronize()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _bootstrap_device(request: str) -> str:
    """Create CUDA context before PyBullet is imported (PyBullet breaks late CUDA init)."""
    req = (request or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req not in ("auto", "cuda"):
        raise ValueError(f"device must be auto, cuda, or cpu — got {request!r}")

    ok, err = _probe_cuda()
    if ok:
        print("[debug:device] CUDA bootstrap OK (before PyBullet import)", flush=True)
        return "cuda"

    hint = (
        "Check: nvidia-smi (driver up?), kill stale python/training using the GPU, "
        "log out/in if you see 'CUDA unknown error', or run with --device cpu."
    )
    if req == "cuda":
        print(
            f"[debug:device] WARNING: --device cuda requested but CUDA unavailable ({err}). "
            f"Falling back to CPU for this diagnostic. {hint}",
            flush=True,
        )
        return "cpu"

    print(f"[debug:device] CUDA unavailable ({err}) — using CPU. {hint}", flush=True)
    return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop policy vs expert action L2 at each step (sim obs, live RAFT)."
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--eval_all", action="store_true", help="Run all trajectories in HDF5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_log_steps",
        type=int,
        default=50,
        help="Max steps to log per trajectory (default 50)",
    )
    parser.add_argument(
        "--l2_threshold",
        type=float,
        default=0.25,
        help="L2 threshold for 'spike' / early divergence (default 0.25)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=5,
        help="Print every N steps unless spike/done (default 5)",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministic policy actions (default: on)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Torch device for policy/RAFT. auto=cuda if free else cpu; "
        "cuda falls back to cpu with a warning if the GPU/driver is unavailable (default: auto)",
    )
    parser.add_argument(
        "--use_cached_optical_flow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="False=live RAFT on sim rgb (default, matches tree08 v2 training).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    resolved_device = _bootstrap_device(args.device)

    # PyBullet import chain only after CUDA bootstrap.
    from debug.diag_policy_multistep_l2_impl import run_diagnostic

    run_diagnostic(args, resolved_device)


if __name__ == "__main__":
    main()
