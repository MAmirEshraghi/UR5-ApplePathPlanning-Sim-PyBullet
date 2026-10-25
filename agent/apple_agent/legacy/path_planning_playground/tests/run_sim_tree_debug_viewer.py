#!/usr/bin/env python3
"""Debug grid: one tree id, many load modes at separate positions (PyBullet GUI).

Row Y=0 (overview): sim URDF, URDF+collision visible, unlabeled OBJ, labeled OBJ, collision-only.
Row Y=spacing (parts): each mat_*.obj from parts/LPy_envy_XXXXX/ alone.

Example:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_sim_tree_debug_viewer.py \\
        --tree-id 17 --texture --preserve-export-mtl --spacing 2.5 --hold-gui

Export assets first if missing:

    python lpy_treesim/export_pybullet_tree_v2.py --tree-id 17 --lpy-index 17 --regenerate-urdf
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pybullet_tree_sim import MESHES_PATH

LABELED_DIR = Path(MESHES_PATH) / "trees" / "envy" / "labeled" / "obj"
UNLABELED_DIR = Path(MESHES_PATH) / "trees" / "envy" / "unlabeled" / "obj"
GENERATED_URDF_DIR = _REPO_ROOT / "pybullet_tree_sim" / "urdf" / "trees" / "envy" / "generated"
DEFAULT_SCALE = 0.6
DEFAULT_SPACING = 2.5
ROW_OVERVIEW_Y = 0.0
ROW_PARTS_Y_FACTOR = 1.0  # parts row at Y = spacing * this


@dataclass
class DebugSlot:
    label: str
    position: list[float]
    description: str
    path: str


def id_str(tree_id: int, namespace: str = "LPy", tree_type: str = "envy") -> str:
    return f"{namespace}_{tree_type}_{tree_id:05d}"


def textured_mtl_path(tree_id_or_str: int | str) -> Path:
    sid = tree_id_or_str if isinstance(tree_id_or_str, str) else id_str(tree_id_or_str)
    return UNLABELED_DIR / f"{sid}_textured.mtl"


def _import_sim_deps():
    import pybullet as p

    from pybullet_tree_sim.tree import Tree
    from run_sim_tree_viewer import (
        configure_gui_lighting,
        ensure_sim_assets,
        make_tree_shell,
    )

    return p, Tree, configure_gui_lighting, ensure_sim_assets, make_tree_shell


def parts_dir(tree_id: int) -> Path:
    return UNLABELED_DIR / "parts" / id_str(tree_id)


def list_part_meshes(tree_id: int) -> list[Path]:
    d = parts_dir(tree_id)
    if not d.is_dir():
        return []
    return sorted(d.glob("mat_*.obj"))


def build_layout(
    tree_id: int,
    spacing: float,
    *,
    include_urdf: bool = True,
    include_parts: bool = True,
) -> list[DebugSlot]:
    sid = id_str(tree_id)
    slots: list[DebugSlot] = []
    x = 0.0

    if include_urdf:
        slots.extend(
            [
                DebugSlot(
                    "A_sim_urdf",
                    [x, ROW_OVERVIEW_Y, 0.0],
                    "URDF multi-link (sim): parts textured, collision visual hidden",
                    str(GENERATED_URDF_DIR / f"{sid}.urdf"),
                ),
                DebugSlot(
                    "B_urdf_collision_visible",
                    [x + spacing, ROW_OVERVIEW_Y, 0.0],
                    "Same URDF; full unlabeled OBJ visual on base link (not hidden)",
                    str(GENERATED_URDF_DIR / f"{sid}.urdf"),
                ),
            ]
        )
        x += 2 * spacing

    slots.append(
        DebugSlot(
            "C_unlabeled_obj",
            [x, ROW_OVERVIEW_Y, 0.0],
            "Unlabeled OBJ visual only (full tree, *_textured.mtl)",
            str(UNLABELED_DIR / f"{sid}.obj"),
        )
    )
    x += spacing

    labeled = LABELED_DIR / f"{sid}_labeled.obj"
    slots.append(
        DebugSlot(
            "D_labeled_obj",
            [x, ROW_OVERVIEW_Y, 0.0],
            "Labeled OBJ visual only (vertex RGB; may look flat in PyBullet)",
            str(labeled),
        )
    )
    x += spacing

    slots.append(
        DebugSlot(
            "E_collision_only",
            [x, ROW_OVERVIEW_Y, 0.0],
            "Collision shape from full unlabeled OBJ (green tint visual)",
            str(UNLABELED_DIR / f"{sid}.obj"),
        )
    )

    if include_parts:
        row_y = spacing * ROW_PARTS_Y_FACTOR
        for i, part_path in enumerate(list_part_meshes(tree_id)):
            slots.append(
                DebugSlot(
                    f"part_{part_path.stem}",
                    [i * spacing, row_y, 0.0],
                    f"Single part mesh (URDF visual link source)",
                    str(part_path),
                )
            )

    return slots


def print_layout_report(
    tree_id: int,
    slots: list[DebugSlot],
    *,
    spacing: float,
    scale: float,
) -> None:
    sid = id_str(tree_id)
    print()
    print("=" * 72)
    print(f"DEBUG LAYOUT  tree_id={tree_id}  id={sid}  spacing={spacing}m  scale={scale}")
    print("=" * 72)
    print(f"{'LABEL':<28} {'X':>6} {'Y':>6} {'Z':>6}  DESCRIPTION")
    print("-" * 72)
    for s in slots:
        print(
            f"{s.label:<28} {s.position[0]:6.2f} {s.position[1]:6.2f} {s.position[2]:6.2f}  {s.description}"
        )
        print(f"{'':28} {'':6} {'':6} {'':6}  → {s.path}")
    print("-" * 72)
    print("Row Y=0: overview modes.  Row Y=spacing: one material part per column.")
    print("Camera target suggestion: center of grid.")
    if slots:
        xs = [s.position[0] for s in slots]
        ys = [s.position[1] for s in slots]
        print(f"  center ≈ [{sum(xs)/len(xs):.2f}, {sum(ys)/len(ys):.2f}, 1.0]")
    print("=" * 72)
    print()


def _mesh_scale(scale: float) -> list[float]:
    return [scale, scale, scale]


def _visual_flags(pb) -> int:
    flags = 0
    if hasattr(pb, "VISUAL_SHAPE_DOUBLE_SIDED"):
        flags |= pb.VISUAL_SHAPE_DOUBLE_SIDED
    return flags


def load_obj_visual(
    pb,
    mesh_path: Path,
    position: list[float],
    *,
    scale: float,
    rgba: list[float] | None = None,
) -> int:
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    kwargs: dict = dict(
        shapeType=pb.GEOM_MESH,
        fileName=str(mesh_path.resolve()),
        meshScale=_mesh_scale(scale),
        flags=_visual_flags(pb),
    )
    if rgba is not None:
        kwargs["rgbaColor"] = rgba
    vis = pb.createVisualShape(**kwargs)
    return pb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=position,
    )


def load_collision_body(
    pb,
    mesh_path: Path,
    position: list[float],
    *,
    scale: float,
) -> int:
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    col = pb.createCollisionShape(
        shapeType=pb.GEOM_MESH,
        fileName=str(mesh_path.resolve()),
        meshScale=_mesh_scale(scale),
    )
    vis = pb.createVisualShape(
        shapeType=pb.GEOM_MESH,
        fileName=str(mesh_path.resolve()),
        meshScale=_mesh_scale(scale),
        rgbaColor=[0.15, 0.75, 0.2, 0.35],
    )
    return pb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
    )


def load_urdf_slot(
    pb,
    tree_id: int,
    position: list[float],
    *,
    scale: float,
    apply_texture: bool,
    preserve_export_mtl: bool,
    hide_collision_visual: bool,
    make_tree_shell,
    ensure_sim_assets,
) -> int:
    tree = make_tree_shell(tree_id, position, scale=scale)
    ensure_sim_assets(tree, preserve_export_mtl=preserve_export_mtl)
    tree.log_pybullet_mesh_preflight()
    return tree.load_pybullet_body(
        pb,
        apply_bark_texture=apply_texture,
        use_obj_mtl=True,
        preserve_exported_mtl=preserve_export_mtl and apply_texture,
        hide_collision_visual=hide_collision_visual,
    )


def spawn_debug_scene(
    pb,
    tree_id: int,
    slots: list[DebugSlot],
    *,
    scale: float,
    apply_texture: bool,
    preserve_export_mtl: bool,
    make_tree_shell,
    ensure_sim_assets,
) -> dict[str, int]:
    bodies: dict[str, int] = {}

    for slot in slots:
        print(f"[debug] loading {slot.label} at {slot.position}")
        try:
            if slot.label == "A_sim_urdf":
                bid = load_urdf_slot(
                    pb,
                    tree_id,
                    slot.position,
                    scale=scale,
                    apply_texture=apply_texture,
                    preserve_export_mtl=preserve_export_mtl,
                    hide_collision_visual=True,
                    make_tree_shell=make_tree_shell,
                    ensure_sim_assets=ensure_sim_assets,
                )
            elif slot.label == "B_urdf_collision_visible":
                bid = load_urdf_slot(
                    pb,
                    tree_id,
                    slot.position,
                    scale=scale,
                    apply_texture=apply_texture,
                    preserve_export_mtl=preserve_export_mtl,
                    hide_collision_visual=False,
                    make_tree_shell=make_tree_shell,
                    ensure_sim_assets=ensure_sim_assets,
                )
            elif slot.label == "E_collision_only":
                bid = load_collision_body(
                    pb,
                    Path(slot.path),
                    slot.position,
                    scale=scale,
                )
            elif slot.label.startswith("part_") or slot.label in (
                "C_unlabeled_obj",
                "D_labeled_obj",
            ):
                bid = load_obj_visual(
                    pb,
                    Path(slot.path),
                    slot.position,
                    scale=scale,
                )
            else:
                bid = load_obj_visual(
                    pb,
                    Path(slot.path),
                    slot.position,
                    scale=scale,
                )
            bodies[slot.label] = bid
            print(f"[debug]   {slot.label} → body_id={bid}")
        except FileNotFoundError as exc:
            print(f"[debug]   SKIP {slot.label}: {exc}", file=sys.stderr)
            bodies[slot.label] = -1

    return bodies


def camera_target(slots: list[DebugSlot]) -> list[float]:
    if not slots:
        return [0.5, 0.0, 1.0]
    xs = [s.position[0] for s in slots]
    ys = [s.position[1] for s in slots]
    return [sum(xs) / len(xs), sum(ys) / len(ys), 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-id", type=int, required=True)
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--preserve-export-mtl", action="store_true")
    parser.add_argument("--grey", action="store_true")
    parser.add_argument("--skip-urdf", action="store_true", help="Skip A/B URDF slots")
    parser.add_argument("--skip-parts", action="store_true", help="Skip part_* row")
    parser.add_argument("--report-only", action="store_true", help="Print layout table only")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--hold-gui", action="store_true")
    args = parser.parse_args()

    apply_texture = bool(args.texture) and not args.grey
    preserve_mtl = bool(args.preserve_export_mtl)

    if apply_texture and preserve_mtl and not textured_mtl_path(id_str(args.tree_id)).is_file():
        print(
            f"ERROR: missing {textured_mtl_path(id_str(args.tree_id))}\n"
            f"Run: python lpy_treesim/export_pybullet_tree_v2.py --tree-id {args.tree_id}",
            file=sys.stderr,
        )
        sys.exit(1)

    slots = build_layout(
        args.tree_id,
        args.spacing,
        include_urdf=not args.skip_urdf,
        include_parts=not args.skip_parts,
    )
    print_layout_report(args.tree_id, slots, spacing=args.spacing, scale=args.scale)

    if args.report_only:
        return

    p, Tree, configure_gui_lighting, ensure_sim_assets, make_tree_shell = _import_sim_deps()
    pb = p
    pb.connect(pb.DIRECT if args.direct else pb.GUI)
    pb.resetSimulation()
    pb.setGravity(0, 0, 0)
    if not args.direct:
        configure_gui_lighting(pb)
        tgt = camera_target(slots)
        pb.resetDebugVisualizerCamera(
            cameraDistance=max(8.0, args.spacing * 3),
            cameraYaw=55,
            cameraPitch=-22,
            cameraTargetPosition=tgt,
        )
    else:
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)

    spawn_debug_scene(
        pb,
        args.tree_id,
        slots,
        scale=args.scale,
        apply_texture=apply_texture,
        preserve_export_mtl=preserve_mtl,
        make_tree_shell=make_tree_shell,
        ensure_sim_assets=ensure_sim_assets,
    )

    print("DEBUG TREE VIEWER OK")
    if args.direct:
        return
    if args.hold_gui:
        print("[debug] holding GUI; Ctrl+C to exit")
        try:
            while True:
                pb.stepSimulation()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    else:
        print("[debug] use --hold-gui to keep the window open")


if __name__ == "__main__":
    main()
