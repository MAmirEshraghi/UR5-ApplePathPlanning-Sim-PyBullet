#!/usr/bin/env python3
"""
merge_agent_data.py

Merge several per-tree expert-trajectory HDF5 files (output/agent_data/<run_id>.hdf5)
into a single combined file for multi-tree training.

Each source group is copied verbatim (datasets + attrs) into the output file under a
flat prefixed key (``<run_id>__<group_key>``) so timestamp-based keys from different
runs never collide while staying a top-level trajectory group (the training callback
iterates top-level groups). Per-group attrs already carry tree_id / tree_scale / tree_pos /
tree_orientation, so the merged file is fully self-describing and the training
callback (random trajectory per episode) naturally spans every tree.

Run from repository root (venv activated):

    python -m apple_agent.data.merge_agent_data \\
        --inputs tree00 tree01 tree02 tree03 tree04 tree05 tree06 tree07 tree08 tree09 \\
        --output trees00to09

Or merge a contiguous range with a shared prefix:

    python -m apple_agent.data.merge_agent_data \\
        --range 0-9 --prefix tree --output trees00to09
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

AGENT_DATA_DIR = Path("output") / "agent_data"  # relative to the workspace (current directory)


def parse_range(value: str) -> list[int]:
    parts = value.split("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(f"expected START-END, got {value!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected START-END, got {value!r}") from exc
    if start > end:
        raise argparse.ArgumentTypeError(f"start must be <= end ({start} > {end})")
    return list(range(start, end + 1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", nargs="+", metavar="RUN_ID", help="Source run_ids under output/agent_data/")
    src.add_argument("--range", type=parse_range, metavar="START-END", help="Contiguous tree indices, e.g. 0-9")
    p.add_argument("--prefix", default="tree", help="Run-id prefix used with --range (default: tree)")
    p.add_argument("--pad", type=int, default=2, help="Zero-pad width for --range ids (default: 2 → tree00)")
    p.add_argument("--output", required=True, metavar="RUN_ID", help="Output run_id (writes output/agent_data/<id>.hdf5)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
    return p


def resolve_input_run_ids(args) -> list[str]:
    if args.inputs is not None:
        return list(args.inputs)
    return [f"{args.prefix}{i:0{args.pad}d}" for i in args.range]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_ids = resolve_input_run_ids(args)
    src_paths = [AGENT_DATA_DIR / f"{rid}.hdf5" for rid in run_ids]

    missing = [str(p) for p in src_paths if not p.is_file()]
    if missing:
        print("ERROR: missing input files:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    out_path = AGENT_DATA_DIR / f"{args.output}.hdf5"
    if out_path.exists() and not args.overwrite:
        print(f"ERROR: output exists: {out_path} (use --overwrite)", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_groups = 0
    per_tree: Counter[str] = Counter()
    with h5py.File(out_path, "w") as out:
        for rid, src in zip(run_ids, src_paths):
            with h5py.File(src, "r") as f:
                keys = list(f.keys())
                for key in keys:
                    dst_key = f"{rid}__{key}"
                    f.copy(f[key], out, name=dst_key)
                    tid = f[key].attrs.get("tree_id_str")
                    if tid is None:
                        raw = f[key].attrs.get("tree_id")
                        tid = f"tree_id={int(np.asarray(raw).reshape(-1)[0])}" if raw is not None else rid
                    per_tree[str(tid)] += 1
                total_groups += len(keys)
                print(f"  + {rid}: {len(keys)} groups  ({src.name})", flush=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nMerged {len(run_ids)} files → {out_path}", flush=True)
    print(f"  trajectories={total_groups}  size_mb={size_mb:.2f}", flush=True)
    print("  per-tree group counts:", flush=True)
    for tid, n in sorted(per_tree.items()):
        print(f"    {tid}: {n}", flush=True)
    print(f"\nTrain with:\n  python -m apple_agent.rl.train_apple_picking_expert --run_id {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
