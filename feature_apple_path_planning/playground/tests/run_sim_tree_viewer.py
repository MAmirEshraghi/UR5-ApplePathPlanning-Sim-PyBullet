#!/usr/bin/env python3
"""
Debug viewer using the **same load path as the PyBullet sim** (not raw PLY).

Sim loads:
  URDF  → pybullet_tree_sim/urdf/trees/envy/generated/LPy_envy_{id:05d}.urdf
  Mesh  → pybullet_tree_sim/meshes/trees/envy/unlabeled/obj/LPy_envy_{id:05d}.obj
  MTL   → .../LPy_envy_{id:05d}_labeled.mtl  (grey or bark via Tree.ensure_obj_mtl)

Those unlabeled OBJs come from export_pybullet_tree (Blender textured_obj), not from PLY
at runtime. PLY is only the LPY source: lpy_treesim/dataset/lpy/tree_{id}.ply.

Tree-only by default (Tree shell, no PKL at init). Optional --with-robot loads the
same UR5 stack as run_scene_test.py / ApplePickingEnv.

Debug tree visuals (A/B — compare GUI; logs should all show PARTS_CHECK PASS):

    # A) baseline: no robot, no room
    PYTHONPATH=. python .../run_sim_tree_viewer.py --tree-id 17 --texture --apple-markers --hold-gui

    # B) tree first, then robot (no room) — tree visible before arm occludes
    PYTHONPATH=. python .../run_sim_tree_viewer.py --tree-id 17 --texture --with-robot --no-room --hold-gui

    # C) full planner scene (robot + room + gravity -9.81)
    PYTHONPATH=. python .../run_sim_tree_viewer.py --tree-id 17 --texture --with-robot --hold-gui

    # D) robot + room but gravity 0 (like tree-only)
    PYTHONPATH=. python .../run_sim_tree_viewer.py --tree-id 17 --texture --with-robot --zero-gravity --hold-gui

Session log: logs/tests/<timestamp>_tree_viewer.log (grep VIZ_VALIDATE VIEWER_SCENE PARTS_CHECK)

Compare tree 8 vs 12 (grey, default):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_sim_tree_viewer.py --compare --hold-gui

Single tree with v2 multi-texture (apple / leaf / wood):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_sim_tree_viewer.py \\
        --tree-id 13 --texture --preserve-export-mtl --hold-gui

Apple centroid markers (PKL cache; log line VIZ_VALIDATE viewer_apple_markers):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_sim_tree_viewer.py \\
        --tree-id 17 --texture --preserve-export-mtl --apple-markers --scale 0.8 --hold-gui

Single tree with v1 single-bark texture:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_sim_tree_viewer.py --tree-id 12 --texture --hold-gui

Compare with texture:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_sim_tree_viewer.py --compare --texture --hold-gui

For raw LPY geometry (PLY only), use run_ply_tree_viewer.py instead.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pybullet_tree_sim import MESHES_PATH, PKL_PATH, URDF_PATH
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.pyb_utils import create_room_backdrop
from feature_apple_path_planning.apple_picking_env import (
    frame_debug_camera_on_tree,
    resolve_preserve_export_mtl,
)
from feature_apple_path_planning.viz_debug import (
    apple_marker_surface_position,
    draw_debug_sphere,
    gui_refresh,
    log_apple_marker_batch,
    log_scene_physics_summary,
    log_viewer_scene_mode,
)
from zenlog import log

LOG_SUBDIR = "tests"

DEFAULT_COMPARE_IDS = (8, 12)
DEFAULT_SPACING_M = 3.0
# Match run_apple_data_generator CONFIG tree_setup.scale
DEFAULT_SCALE = 0.8
DEFAULT_APPLE_MARKER_RADIUS = 0.035
DEFAULT_APPLE_MARKER_OFFSET_M = 0.0
# Offset markers toward robot side (planner robot at [0, 1.3, 0])
MARKER_OFFSET_REF_POS = np.array([0.0, 1.3, 0.0], dtype=np.float64)
ROBOT_START_POSITION = [0.0, 1.3, 0.0]
ROBOT_START_ORIENTATION_EULER_DEG = [0, 0, 180]
ROBOT_START_JOINT_ANGLES = [-1.978, -1.51, 2.622, -1.896, 0.579, 0.848]
UNLABELED_DIR = Path(MESHES_PATH) / "trees" / "envy" / "unlabeled" / "obj"
GENERATED_URDF_DIR = _REPO_ROOT / "pybullet_tree_sim" / "urdf" / "trees" / "envy" / "generated"


def load_robot_like_scene_test(pb) -> Robot:
    """Same robot load as run_scene_test.py / planner CONFIG."""
    cached = os.path.join(URDF_PATH, "tmp", "robot.urdf")
    orient = Rotation.from_euler(
        "xyz", ROBOT_START_ORIENTATION_EULER_DEG, degrees=True
    ).as_quat()
    print("[sim] Loading robot (cached URDF if present)...")
    robot = Robot(
        pbclient=pb,
        position=ROBOT_START_POSITION,
        orientation=orient,
        regenerate_urdf=not os.path.isfile(cached),
    )
    robot.set_joint_angles_no_collision(ROBOT_START_JOINT_ANGLES)
    print(
        f"[sim] robot body_id={robot.robot} tool0_link={robot.tool0_link_idx} "
        f"urdf={robot.robot_urdf_path}"
    )
    return robot


def setup_viewer_file_logging() -> Path:
    """Mirror zenlog to logs/tests/<timestamp>_tree_viewer.log."""
    base = _REPO_ROOT / "logs" / LOG_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_tree_viewer.log").resolve()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.addHandler(fh)
    zen_logger.setLevel(logging.DEBUG)
    log.info("Session log: %s", log_path)
    return log_path


def configure_gui_lighting(pb) -> None:
    """Match PyBUtils scene-test lighting so bark map_Kd is visible in GUI."""
    pb.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_SHADOWS"):
        pb.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    pb.configureDebugVisualizer(lightPosition=[2.0, 3.0, 4.0])
    pb.configureDebugVisualizer(shadowMapResolution=2048)


def textured_mtl_path(id_str: str) -> Path:
    return UNLABELED_DIR / f"{id_str}_textured.mtl"


def describe_visual_mode(apply_texture: bool, *, preserve_export_mtl: bool) -> str:
    if not apply_texture:
        return "grey (flat, no bark map_Kd)"
    if preserve_export_mtl:
        return "multi-material (v2 export *_textured.mtl, preserve_export_mtl)"
    tex = Tree.resolve_bark_texture_path()
    if tex:
        return f"single bark (ensure_obj_mtl map_Kd) → {tex}"
    return "bark requested but no texture file found under pybullet_tree_sim/textures/"


def make_tree_shell(
    tree_id: int,
    position: list[float],
    *,
    scale: float = DEFAULT_SCALE,
    namespace: str = "LPy",
    tree_type: str = "envy",
) -> Tree:
    """Tree instance without __init__ (no PKL / labeled OBJ load)."""
    t = Tree.__new__(Tree)
    t.tree_id = tree_id
    t.tree_type = tree_type
    t.tree_namespace = namespace
    t.id_str = f"{namespace}_{tree_type}_{tree_id:05d}"
    t.urdf_path = str(GENERATED_URDF_DIR / f"{t.id_str}.urdf")
    t.mesh_path = str(UNLABELED_DIR / f"{t.id_str}.obj")
    t.pos = np.array(position, dtype=np.float64)
    t.orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    t.scale = scale
    t.pyb_id = None
    return t


def ensure_sim_assets(tree: Tree, *, preserve_export_mtl: bool = False) -> None:
    if not os.path.isfile(tree.mesh_path):
        raise FileNotFoundError(
            f"Unlabeled OBJ missing for {tree.id_str}:\n  {tree.mesh_path}\n"
            f"Run: python lpy_treesim/export_pybullet_tree_v2.py --tree-id {tree.tree_id} "
            f"--lpy-index <N>"
        )
    textured = textured_mtl_path(tree.id_str)
    has_v2 = Tree.has_textured_export_mtl(
        tree.tree_id, tree.tree_type, tree.tree_namespace
    )
    need_textured_urdf = preserve_export_mtl and textured.is_file()
    if not os.path.isfile(tree.urdf_path):
        print(f"[sim] generating URDF for {tree.id_str}")
        if need_textured_urdf or has_v2:
            Tree.regenerate_textured_urdf(
                tree_id=tree.tree_id,
                tree_type=tree.tree_type,
                namespace=tree.tree_namespace,
            )
        else:
            Tree.regenerate_urdf_from_xacro(
                tree_id=tree.tree_id,
                tree_type=tree.tree_type,
                namespace=tree.tree_namespace,
                use_visual_material=True,
            )
    elif need_textured_urdf:
        print(
            f"[sim] regenerating multi-material URDF for {tree.id_str} "
            f"(one link per usemtl + OBJ/MTL textures)"
        )
        Tree.regenerate_textured_urdf(
            tree_id=tree.tree_id,
            tree_type=tree.tree_type,
            namespace=tree.tree_namespace,
        )
    elif has_v2:
        print(
            f"[sim] ensuring multi-material URDF for {tree.id_str} "
            f"(v2 export; grey or single-bark overlay)"
        )
        Tree.regenerate_textured_urdf(
            tree_id=tree.tree_id,
            tree_type=tree.tree_type,
            namespace=tree.tree_namespace,
        )
    elif not preserve_export_mtl:
        print(f"[sim] regenerating single-mesh URDF for {tree.id_str} (xacro, no v2 export)")
        Tree.regenerate_urdf_from_xacro(
            tree_id=tree.tree_id,
            tree_type=tree.tree_type,
            namespace=tree.tree_namespace,
            use_visual_material=True,
        )


def load_tree_like_sim(
    pb,
    tree: Tree,
    *,
    apply_texture: bool,
    preserve_export_mtl: bool = False,
    hide_collision_visual: bool = True,
) -> int:
    ensure_sim_assets(tree, preserve_export_mtl=preserve_export_mtl)
    textured = textured_mtl_path(tree.id_str)
    if preserve_export_mtl and not textured.is_file():
        print(
            f"[sim] WARNING: --preserve-export-mtl but missing {textured}\n"
            f"         Run: python lpy_treesim/export_pybullet_tree_v2.py --tree-id {tree.tree_id}",
            file=sys.stderr,
        )
    tree.log_pybullet_mesh_preflight()
    pyb_id = tree.load_pybullet_body(
        pb,
        apply_bark_texture=apply_texture,
        use_obj_mtl=True,
        preserve_exported_mtl=preserve_export_mtl and apply_texture,
        hide_collision_visual=hide_collision_visual,
    )
    tree.log_pybullet_visual_summary(pb, phase="loaded")
    if apply_texture:
        mtl = textured if preserve_export_mtl else tree.obj_mtl_path()
        print(f"[sim] {tree.id_str} MTL: {mtl}")
    return pyb_id


def draw_apple_centroid_markers(
    pb,
    tree: Tree,
    *,
    radius: float,
    surface_offset_m: float,
    gui: bool,
) -> None:
    """Red spheres offset from PKL apple centroids (needs pybullet_tree_sim/pkl/<id>_points.pkl)."""
    pkl_path = os.path.join(PKL_PATH, f"{tree.id_str}_points.pkl")
    if not tree._try_load_full_pkl_cache(pkl_path):
        print(
            f"[sim] apple markers: PKL miss for {tree.id_str} ({pkl_path}); "
            "match --scale/--tree-pos to cache or run planner prewarm",
            file=sys.stderr,
        )
        return
    centroids = tree.get_apple_centroids()
    if not centroids:
        print(f"[sim] apple markers: no APPLE_* labels in PKL for {tree.id_str}")
        return
    rgba = [0.9, 0.08, 0.08, 0.9]
    body_ids = []
    positions = []
    for c in centroids:
        centroid = np.asarray(c, dtype=np.float64).reshape(3)
        marker_pos = apple_marker_surface_position(
            centroid, MARKER_OFFSET_REF_POS, surface_offset_m
        )
        positions.append(marker_pos)
        body_ids.append(draw_debug_sphere(pb, marker_pos, radius, rgba))
    log_apple_marker_batch(
        "viewer_apple_markers",
        body_ids,
        centroids,
        positions,
        radius,
        surface_offset_m=surface_offset_m,
    )
    print(f"[sim] apple markers: {len(centroids)} spheres radius={radius} m for {tree.id_str}")
    gui_refresh(pb, gui, steps=1)


def tree_positions(tree_ids: list[int], spacing: float) -> dict[int, list[float]]:
    n = len(tree_ids)
    if n == 1:
        return {tree_ids[0]: [0.5, 0.0, 0.0]}
    half = (n - 1) / 2.0
    return {
        tid: [0.5, (i - half) * spacing, 0.0]
        for i, tid in enumerate(tree_ids)
    }


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


def print_pipeline_hint(tree_id: int) -> None:
    ply = _REPO_ROOT / "lpy_treesim" / "dataset" / "lpy" / f"tree_{tree_id}.ply"
    print(f"[sim] {tree_id}: PLY source (not loaded here)     → {ply}")
    print(f"[sim] {tree_id}: sim unlabeled OBJ (loadURDF)    → {UNLABELED_DIR / f'LPy_envy_{tree_id:05d}.obj'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load trees exactly like the sim (URDF + unlabeled OBJ), no robot/PKL."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tree-id", type=int, help="Single sim tree id")
    mode.add_argument("--compare", action="store_true", help=f"Default trees {DEFAULT_COMPARE_IDS}")
    parser.add_argument("--tree-ids", type=parse_tree_ids, metavar="IDS")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"globalScaling (planner default {DEFAULT_SCALE})",
    )
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument(
        "--texture",
        action="store_true",
        help="Textured: single bark (v1 export) unless --preserve-export-mtl (v2)",
    )
    parser.add_argument(
        "--preserve-export-mtl",
        action="store_true",
        help="Force v2 *_textured.mtl (default: auto if file exists, like planner)",
    )
    parser.add_argument(
        "--no-preserve-export-mtl",
        action="store_true",
        help="Disable auto v2 MTL even when *_textured.mtl exists",
    )
    parser.add_argument(
        "--show-collision-shell",
        action="store_true",
        help="Keep base-link full-mesh visual (default: hidden like planner)",
    )
    vis.add_argument(
        "--grey",
        action="store_true",
        help="Flat grey mesh (default; same as run_scene_test.py --no-texture)",
    )
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--hold-gui", action="store_true")
    parser.add_argument("--show-paths", action="store_true", help="Print PLY vs sim OBJ paths per tree")
    parser.add_argument(
        "--apple-markers",
        action="store_true",
        help="Draw red spheres at apple centroids from PKL (grep VIZ_VALIDATE viewer_apple_markers)",
    )
    parser.add_argument(
        "--apple-marker-radius",
        type=float,
        default=DEFAULT_APPLE_MARKER_RADIUS,
        help=f"Sphere radius in meters (default {DEFAULT_APPLE_MARKER_RADIUS})",
    )
    parser.add_argument(
        "--apple-marker-offset",
        type=float,
        default=DEFAULT_APPLE_MARKER_OFFSET_M,
        help=f"Push markers outside mesh toward robot (default {DEFAULT_APPLE_MARKER_OFFSET_M} m)",
    )
    parser.add_argument(
        "--with-robot",
        action="store_true",
        help="Load UR5 + room backdrop like run_scene_test.py (planner layout)",
    )
    parser.add_argument(
        "--no-room",
        action="store_true",
        help="With --with-robot: skip leafy room backdrop (robot + tree only; debug visuals)",
    )
    parser.add_argument(
        "--zero-gravity",
        action="store_true",
        help="With --with-robot: gravity=0 like tree-only (default with robot: -9.81)",
    )
    args = parser.parse_args()

    if args.no_room and not args.with_robot:
        print("[sim] NOTE: --no-room only applies with --with-robot (ignored for tree-only)", file=sys.stderr)

    tree_ids = resolve_tree_ids(args)
    apply_texture = bool(args.texture)
    hide_collision_visual = not args.show_collision_shell
    first_tid = tree_ids[0]
    tree_conf = {
        "tree_id": first_tid,
        "tree_type": "envy",
        "tree_namespace": "LPy",
        "preserve_export_mtl": False if args.no_preserve_export_mtl else None,
    }
    if args.preserve_export_mtl:
        tree_conf["preserve_export_mtl"] = True
    preserve_mtl = resolve_preserve_export_mtl(tree_conf) if apply_texture else False
    mode_label = describe_visual_mode(apply_texture, preserve_export_mtl=preserve_mtl)
    room_backdrop = bool(args.with_robot and not args.no_room)
    gravity_z = 0.0 if (not args.with_robot or args.zero_gravity) else -9.81

    log_path = setup_viewer_file_logging()
    print(f"[sim] log file: {log_path}")

    print(f"[sim] tree ids={tree_ids}  scale={args.scale}")
    print(f"[sim] visual mode: {mode_label}")
    log_viewer_scene_mode(
        with_robot=args.with_robot,
        room_backdrop=room_backdrop,
        gravity_z=gravity_z,
        hide_collision_visual=hide_collision_visual,
        preserve_mtl=preserve_mtl,
        scale=args.scale,
    )
    print(
        f"VIZ_VALIDATE viewer_config: scale={args.scale} preserve_mtl={preserve_mtl} "
        f"hide_collision_visual={hide_collision_visual} with_robot={args.with_robot} "
        f"room_backdrop={room_backdrop} gravity_z={gravity_z} "
        f"apple_markers={'ON' if args.apple_markers else 'OFF'} "
        f"marker_radius={args.apple_marker_radius} offset_m={args.apple_marker_offset}"
    )
    if args.with_robot and args.no_room:
        print("[sim] DEBUG: --no-room — robot only, no leaves-dead.png walls/floor")
    if args.with_robot and args.zero_gravity:
        print("[sim] DEBUG: --zero-gravity — matches tree-only physics")
    if apply_texture and not preserve_mtl and Tree.resolve_bark_texture_path() is None:
        print("[sim] ERROR: --texture requires bark JPG under pybullet_tree_sim/textures/", file=sys.stderr)
        sys.exit(1)
    if args.show_paths:
        for tid in tree_ids:
            print_pipeline_hint(tid)

    pb = p
    p.connect(p.DIRECT if args.direct else p.GUI)
    p.resetSimulation()
    p.setGravity(0, 0, gravity_z)

    robot = None
    if room_backdrop:
        create_room_backdrop(pb, collision=False)

    if not args.direct:
        configure_gui_lighting(pb)
    else:
        pb.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    positions = tree_positions(tree_ids, args.spacing)
    if not args.direct and tree_ids:
        pos0 = positions[tree_ids[0]]
        frame_debug_camera_on_tree(
            pb,
            pos0,
            {
                "frame_tree_camera": True,
                "debug_camera_distance": 4.5,
                "debug_camera_yaw": 55,
                "debug_camera_pitch": -22,
                "debug_camera_target_z_offset": 1.0,
            },
        )

    loaded_trees: list[Tree] = []
    for tid in tree_ids:
        tree = make_tree_shell(tid, positions[tid], scale=args.scale)
        tid_conf = {
            "tree_id": tid,
            "tree_type": "envy",
            "tree_namespace": "LPy",
            "preserve_export_mtl": tree_conf.get("preserve_export_mtl"),
        }
        tid_preserve = (
            resolve_preserve_export_mtl(tid_conf) if apply_texture else False
        )
        print(f"[sim] loading {tree.id_str} at {positions[tid]} preserve_mtl={tid_preserve}")
        try:
            bid = load_tree_like_sim(
                pb,
                tree,
                apply_texture=apply_texture,
                preserve_export_mtl=tid_preserve,
                hide_collision_visual=hide_collision_visual,
            )
            loaded_trees.append(tree)
            print(f"[sim] {tree.id_str}  pyb_id={bid}  mesh={tree.mesh_path}")
            tree.log_tree_parts_integrity(
                pb,
                context="viewer_after_load",
                require_hidden_base=hide_collision_visual,
            )
            if args.apple_markers:
                draw_apple_centroid_markers(
                    pb,
                    tree,
                    radius=args.apple_marker_radius,
                    surface_offset_m=args.apple_marker_offset,
                    gui=not args.direct,
                )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    if loaded_trees and args.with_robot:
        primary = loaded_trees[0]
        log_scene_physics_summary(
            pb,
            context="viewer_after_tree_before_robot",
            tree_body_id=primary.pyb_id,
            robot_body_id=None,
        )
        gui_refresh(pb, not args.direct, steps=2)

    if args.with_robot:
        print("[sim] loading robot after tree (GUI shows tree first)")
        robot = load_robot_like_scene_test(pb)
        log_scene_physics_summary(
            pb,
            context="viewer_after_robot",
            tree_body_id=loaded_trees[0].pyb_id if loaded_trees else None,
            robot_body_id=robot.robot,
        )

    if loaded_trees:
        primary = loaded_trees[0]
        log_scene_physics_summary(
            pb,
            context="viewer_final",
            tree_body_id=primary.pyb_id,
            robot_body_id=robot.robot if robot is not None else None,
        )
        parts_ok = primary.log_tree_parts_integrity(
            pb,
            context="viewer_final",
            require_hidden_base=hide_collision_visual,
        )
        print(
            f"VIZ_VALIDATE viewer_parts_check: {'PASS' if parts_ok else 'FAIL'} "
            f"(grep PARTS_CHECK SCENE_BODIES in logs)"
        )

    print("SIM TREE VIEWER OK")
    if args.direct:
        return
    if args.hold_gui:
        print("[sim] holding GUI; Ctrl+C to exit")
        try:
            while True:
                pb.stepSimulation()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    else:
        print("[sim] use --hold-gui to keep the window open")


if __name__ == "__main__":
    main()
