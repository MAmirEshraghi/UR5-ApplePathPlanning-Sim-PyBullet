#!/usr/bin/env python3
"""
Minimal PyBullet viewer: raw LPY **PLY** only (NOT the same as the sim).

This does NOT load pybullet_tree_sim/meshes/.../unlabeled/obj/LPy_envy_*.obj.
For sim-identical loading (URDF + exported unlabeled OBJ), use run_sim_tree_viewer.py.

Uses lpy_treesim/dataset/lpy/tree_{id}.ply (PlantGL ASCII). Converts once to a
cached OBJ under logs/tmp/ply_cache/, then loads grey visual meshes.

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_ply_tree_viewer.py --hold-gui

Compare tree 8 and 10:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_ply_tree_viewer.py --compare --hold-gui

Single tree:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_ply_tree_viewer.py --tree-id 8 --hold-gui
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

_REPO_ROOT = Path(__file__).resolve().parents[3]
LPY_PLY_DIR = _REPO_ROOT / "lpy_treesim" / "dataset" / "lpy"
PLY_CACHE_DIR = _REPO_ROOT / "logs" / "tmp" / "ply_cache"

DEFAULT_COMPARE_IDS = (8, 10)
DEFAULT_SPACING_M = 3.0
DEFAULT_RGBA = [0.45, 0.45, 0.48, 1.0]


def ply_path_for_tree(tree_id: int) -> Path:
    return LPY_PLY_DIR / f"tree_{tree_id}.ply"


def cached_obj_path(tree_id: int) -> Path:
    return PLY_CACHE_DIR / f"tree_{tree_id}_visual.obj"


def stream_ply_to_obj(ply_path: Path, obj_path: Path) -> tuple[int, int]:
    """Convert PlantGL ASCII PLY → minimal OBJ (v + triangular f)."""
    n_verts = n_faces = 0
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    with ply_path.open(encoding="utf-8", errors="replace") as fin:
        lines = fin.readlines()

    i = 0
    while i < len(lines) and lines[i].strip() != "end_header":
        row = lines[i].strip()
        if row.startswith("element vertex"):
            n_verts = int(row.split()[-1])
        elif row.startswith("element face"):
            n_faces = int(row.split()[-1])
        i += 1
    i += 1  # skip end_header

    with obj_path.open("w", encoding="utf-8") as fout:
        fout.write(f"# from {ply_path.name}\n")
        fout.write(f"o tree_{ply_path.stem}\n")

        for _ in range(n_verts):
            parts = lines[i].split()
            i += 1
            fout.write(f"v {parts[0]} {parts[1]} {parts[2]}\n")

        for _ in range(n_faces):
            parts = lines[i].split()
            i += 1
            n = int(parts[0])
            idx = [int(x) + 1 for x in parts[1 : 1 + n]]
            if n == 3:
                fout.write(f"f {idx[0]} {idx[1]} {idx[2]}\n")
            elif n == 4:
                fout.write(f"f {idx[0]} {idx[1]} {idx[2]}\n")
                fout.write(f"f {idx[0]} {idx[2]} {idx[3]}\n")
            elif n > 4:
                for j in range(1, n - 1):
                    fout.write(f"f {idx[0]} {idx[j]} {idx[j + 1]}\n")

    return n_verts, n_faces


def ensure_obj_cache(tree_id: int, *, force: bool = False) -> Path:
    ply = ply_path_for_tree(tree_id)
    if not ply.is_file():
        raise FileNotFoundError(f"PLY not found: {ply}")

    obj = cached_obj_path(tree_id)
    if not force and obj.is_file() and obj.stat().st_mtime >= ply.stat().st_mtime:
        print(f"[ply] cache hit: {obj}")
        return obj

    print(f"[ply] converting {ply.name} → {obj.name} (one-time, may take a minute)...")
    t0 = time.perf_counter()
    n_v, n_f = stream_ply_to_obj(ply, obj)
    print(f"[ply] done in {time.perf_counter() - t0:.1f}s  verts={n_v}  faces={n_f}")
    return obj


def tree_positions(tree_ids: list[int], spacing: float) -> dict[int, list[float]]:
    n = len(tree_ids)
    if n == 1:
        return {tree_ids[0]: [0.0, 0.0, 0.0]}
    half = (n - 1) / 2.0
    return {
        tid: [0.0, (i - half) * spacing, 0.0]
        for i, tid in enumerate(tree_ids)
    }


def load_tree_visual(
    obj_path: Path,
    position: list[float],
    *,
    rgba: list[float],
    mesh_scale: float,
) -> int:
    scale = [mesh_scale] * 3
    visual = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(obj_path),
        meshScale=scale,
        rgbaColor=rgba,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def parse_tree_ids(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_tree_ids(args: argparse.Namespace) -> list[int]:
    if args.compare:
        return args.tree_ids if args.tree_ids is not None else list(DEFAULT_COMPARE_IDS)
    if args.tree_id is not None:
        return [args.tree_id]
    if args.tree_ids is not None:
        return args.tree_ids
    return list(DEFAULT_COMPARE_IDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal PLY tree viewer (PyBullet).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tree-id", type=int, help="Single tree id (PLY tree_{id}.ply)")
    mode.add_argument("--compare", action="store_true", help=f"Load trees {DEFAULT_COMPARE_IDS}")
    parser.add_argument("--tree-ids", type=parse_tree_ids, metavar="IDS", help="e.g. 8,10")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="PyBullet mesh scale")
    parser.add_argument(
        "--rgba",
        type=float,
        nargs=4,
        default=DEFAULT_RGBA,
        metavar=("R", "G", "B", "A"),
        help="Mesh color (default grey)",
    )
    parser.add_argument("--direct", action="store_true", help="No GUI window")
    parser.add_argument("--hold-gui", action="store_true", help="Keep GUI until Ctrl+C")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force PLY→OBJ conversion")
    args = parser.parse_args()

    tree_ids = resolve_tree_ids(args)
    for tid in tree_ids:
        if not ply_path_for_tree(tid).is_file():
            print(f"ERROR: missing {ply_path_for_tree(tid)}", file=sys.stderr)
            sys.exit(1)

    print(f"[ply] trees={tree_ids}  spacing={args.spacing}m  scale={args.mesh_scale}")

    p.connect(p.DIRECT if args.direct else p.GUI)
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    if not args.direct:
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=55,
            cameraPitch=-25,
            cameraTargetPosition=[0.0, 0.0, 1.2],
        )

    positions = tree_positions(tree_ids, args.spacing)
    bodies: list[tuple[int, int]] = []
    for tid in tree_ids:
        obj = ensure_obj_cache(tid, force=args.rebuild_cache)
        pos = positions[tid]
        bid = load_tree_visual(
            obj,
            pos,
            rgba=list(args.rgba),
            mesh_scale=args.mesh_scale,
        )
        bodies.append((tid, bid))
        print(f"[ply] tree {tid}  body_id={bid}  pos={pos}")

    print("PLY VIEWER OK")
    if args.direct:
        return

    if args.hold_gui:
        print("[ply] holding GUI; Ctrl+C to exit")
        try:
            while True:
                p.stepSimulation()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    else:
        print("[ply] use --hold-gui to keep the window open")


if __name__ == "__main__":
    main()
