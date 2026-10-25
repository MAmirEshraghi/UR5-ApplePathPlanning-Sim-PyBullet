#!/usr/bin/env python3
"""
Scene-only PyBullet test: load Robot + Tree (same sequence as ApplePickingEnv._setup_scene),
log to logs/tests/, and print a structured terminal summary for debugging.

No Gym env, planner, or HDF5 dependencies.

First run for a new tree id can be slow while Tree builds/caches point data under PKL_PATH;
that is expected, not a failure.

Run from repository root (venv activated):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test.py

Headless (no GUI, exits when done):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test.py --direct

Keep GUI open after the check (Ctrl+C to quit):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test.py --hold-gui

Override tree id:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test.py --tree-id 2 --direct

Grey tree (no bark / MTL textures):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test.py --no-texture --hold-gui

Semantic debug markers (red apple centroids, green leaf samples):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test.py --show-markers --hold-gui
"""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from zenlog import log

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pybullet_tree_sim import URDF_PATH
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.pyb_utils import PyBUtils

from feature_apple_path_planning.apple_picking_env import (
    ensure_tree_sim_assets,
    frame_debug_camera_on_tree,
    load_tree_pybullet_body,
    resolve_preserve_export_mtl,
)
from feature_apple_path_planning.viz_debug import (
    apple_marker_surface_position,
    log_apple_marker_batch,
    log_scene_physics_summary,
)

LOG_SUBDIR = "tests"

# Matches planner visualization / CONFIG
APPLE_MARKER_RADIUS = 0.08
APPLE_MARKER_OFFSET_M = 0.12
APPLE_MARKER_RGBA = [0.9, 0.08, 0.08, 0.9]
LEAF_MARKER_RADIUS = 0.025
LEAF_MARKER_RGBA = [0.2, 0.75, 0.25, 0.65]
DEFAULT_MAX_LEAF_MARKERS = 200

# Mirrors run_apple_data_generator.CONFIG simulation_setup / robot_setup / tree_setup
TEST_CONFIG = {
    "simulation_setup": {
        "renders": True,
        "gravity": -9.81,
        "settle_steps": 100,
    },
    "robot_setup": {
        "start_position": [0, 1.3, 0],
        "start_orientation_euler_deg": [0, 0, 180],
        "start_joint_angles": [-1.978, -1.51, 2.622, -1.896, 0.579, 0.848],
    },
    "tree_setup": {
        "tree_id": 9,
        "tree_type": "envy",
        "tree_namespace": "LPy",
        "scale": 0.8,
        "position": np.array([0.5, 0, 0]),
        "orientation": np.array([0, 0, 0, 1]),
        "apply_texture": True,
        "preserve_export_mtl": None,
        "hide_collision_visual": True,
    },
}


def setup_test_file_logging() -> Path:
    """Mirror zenlog to logs/tests/<timestamp>_scene_test.log."""
    base = _REPO_ROOT / "logs" / LOG_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_scene_test.log").resolve()
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


def build_scene(
    pbutils: PyBUtils,
    config: dict,
    *,
    apply_tree_texture: bool = True,
) -> tuple[Robot, Tree]:
    sim_setup = config["simulation_setup"]
    pbutils.reset_simulation_scene(
        gravity=sim_setup["gravity"],
        room_collision=bool(sim_setup.get("room_collision", False)),
    )

    log.info("Loading Robot...")
    robot_conf = config["robot_setup"]
    robot_start_orientation_quat = Rotation.from_euler(
        "xyz", robot_conf["start_orientation_euler_deg"], degrees=True
    ).as_quat()

    cached_robot_urdf = os.path.join(URDF_PATH, "tmp", "robot.urdf")
    robot = Robot(
        pbclient=pbutils.pbclient,
        position=robot_conf["start_position"],
        orientation=robot_start_orientation_quat,
        regenerate_urdf=not os.path.isfile(cached_robot_urdf),
    )
    if robot_conf.get("start_joint_angles"):
        robot.set_joint_angles_no_collision(robot_conf["start_joint_angles"])

    log.info("Loading Tree...")
    tree_conf = config["tree_setup"]
    tree_conf = {**tree_conf, "apply_texture": apply_tree_texture}
    tree = Tree(
        pbutils=pbutils,
        tree_id=tree_conf["tree_id"],
        tree_type=tree_conf["tree_type"],
        namespace=tree_conf["tree_namespace"],
        scale=tree_conf["scale"],
        position=tree_conf["position"],
        orientation=tree_conf["orientation"],
    )
    ensure_tree_sim_assets(tree, tree_conf)
    load_tree_pybullet_body(pbutils.pbclient, tree, tree_conf)
    preserve = resolve_preserve_export_mtl(tree_conf) if apply_tree_texture else False
    if pbutils.renders:
        frame_debug_camera_on_tree(
            pbutils.pbclient,
            tree_conf["position"],
            {**sim_setup, "frame_tree_camera": True},
        )
    visual_mode = (
        "v2 multi-material"
        if apply_tree_texture and preserve
        else ("grey" if not apply_tree_texture else "bark")
    )
    log.info(
        "VIZ_VALIDATE scene_config: scale=%s preserve_mtl=%s hide_collision_visual=%s",
        tree.scale,
        preserve,
        tree_conf.get("hide_collision_visual"),
    )
    log.info("Tree visual mode: %s", visual_mode)
    print(f"[scene] tree visual: {visual_mode}")
    return robot, tree


def log_scene_summary(pbutils: PyBUtils, robot: Robot, tree: Tree, config: dict) -> int:
    """Log full scene summary; return apple centroid count."""
    pb = pbutils.pbclient
    sim = config["simulation_setup"]
    mode = "GUI" if sim.get("renders") else "DIRECT"
    log.info("=== Scene test summary ===")
    log.info("Connection mode: %s (renders=%s)", mode, sim.get("renders"))
    log.info("Gravity (z): %s  step_time: %s", sim["gravity"], pbutils.step_time)

    log.info("--- Robot ---")
    log.info("robot_urdf_path: %s", robot.robot_urdf_path)
    log.info("body_unique_id: %s  num_joints: %s", robot.robot, pb.getNumJoints(robot.robot))
    log.info("tool0_link_idx: %s", robot.tool0_link_idx)

    log.info("--- Tree ---")
    log.info("id_str: %s", tree.id_str)
    log.info("urdf_path: %s", tree.urdf_path)
    log.info("mesh_path (unlabeled): %s", tree.mesh_path)
    log.info("labeled_mesh_path: %s", tree.labeled_mesh_path)
    log.info("pyb_id: %s  scale: %s", tree.pyb_id, tree.scale)
    n_apples = len(tree.get_apple_centroids())
    log.info("apple_centroids count: %s", n_apples)

    log.info("--- Bodies in client ---")
    for i in range(pb.getNumBodies()):
        info = pb.getBodyInfo(i)
        name = info[1].decode("utf-8", errors="replace") if info[1] else "?"
        log.info("  body_index=%s  name=%s", i, name)

    print(f"[scene] mode={mode}  tree={tree.id_str}  robot_body_id={robot.robot}  apples={n_apples}")
    return n_apples


def create_debug_sphere(pb, position, radius: float, rgba) -> int:
    """Visual-only sphere (no collision), same pattern as ApplePickingEnv."""
    try:
        vid = pb.createVisualShape(
            shapeType=pb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
        )
        return pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vid,
            basePosition=list(np.asarray(position, dtype=np.float64).reshape(-1)[:3]),
        )
    except Exception as exc:
        log.warning("Debug sphere failed at %s: %s", position, exc)
        return -1


def sample_leaf_positions(tree: Tree, max_markers: int, seed: int = 0) -> list[np.ndarray]:
    """Subsample LEAF-labeled vertices for GUI markers (full leaf sets are huge)."""
    verts = getattr(tree, "transformed_vertices", None) or []
    leaf_pts = [
        np.asarray(coords, dtype=np.float64).reshape(-1)[:3]
        for coords, label in verts
        if label == "LEAF"
    ]
    if not leaf_pts:
        return []
    arr = np.stack(leaf_pts, axis=0)
    if len(arr) <= max_markers:
        return [arr[i] for i in range(len(arr))]
    idx = np.random.default_rng(seed).choice(len(arr), size=max_markers, replace=False)
    return [arr[i] for i in idx]


def spawn_scene_markers(
    pbutils: PyBUtils,
    tree: Tree,
    config: dict,
    *,
    max_leaf_markers: int = DEFAULT_MAX_LEAF_MARKERS,
) -> tuple[int, int]:
    """Draw red spheres at apple centroids and green spheres at sampled leaf vertices."""
    pb = pbutils.pbclient
    robot_pos = np.asarray(config["robot_setup"]["start_position"], dtype=np.float64)
    centroids = tree.get_apple_centroids()
    body_ids = []
    marker_positions = []
    apple_count = 0
    for center in centroids:
        centroid = np.asarray(center, dtype=np.float64).reshape(3)
        marker_pos = apple_marker_surface_position(
            centroid, robot_pos, APPLE_MARKER_OFFSET_M
        )
        marker_positions.append(marker_pos)
        bid = create_debug_sphere(pb, marker_pos, APPLE_MARKER_RADIUS, APPLE_MARKER_RGBA)
        body_ids.append(bid)
        if bid >= 0:
            apple_count += 1
    log_apple_marker_batch(
        "scene_test_apple_markers",
        body_ids,
        centroids,
        marker_positions,
        APPLE_MARKER_RADIUS,
        APPLE_MARKER_OFFSET_M,
    )

    leaf_positions = sample_leaf_positions(tree, max_leaf_markers)
    leaf_count = 0
    for pos in leaf_positions:
        bid = create_debug_sphere(pb, pos, LEAF_MARKER_RADIUS, LEAF_MARKER_RGBA)
        if bid >= 0:
            leaf_count += 1
    if not leaf_positions:
        log.warning(
            "No LEAF-labeled vertices on tree %s (labeled mesh may omit LEAF paint); "
            "green leaf markers skipped.",
            tree.id_str,
        )

    log.info(
        "Scene markers: %s apples (r=%s), %s/%s leaf samples (r=%s)",
        apple_count,
        APPLE_MARKER_RADIUS,
        leaf_count,
        len(leaf_positions),
        LEAF_MARKER_RADIUS,
    )
    print(
        f"[markers] apples={apple_count} r={APPLE_MARKER_RADIUS}  "
        f"leaves={leaf_count}/{len(leaf_positions)} r={LEAF_MARKER_RADIUS}"
    )
    return apple_count, leaf_count


def probe_scene(robot: Robot, tree: Tree, pbutils: PyBUtils) -> None:
    """Post-load probes: EE pose, joints, apples, optional camera RGB."""
    ee_pos, ee_quat = robot.get_current_pose(robot.tool0_link_idx)
    joint_angles = robot.get_joint_angles()
    apples = tree.get_apple_centroids()

    log.info("--- Scene probe ---")
    log.info("EE position: %s", np.round(ee_pos, 4).tolist())
    log.info("EE orientation (quat): %s", np.round(ee_quat, 4).tolist())
    log.info("Joint angles: %s", np.round(joint_angles, 4).tolist())
    log.info("Apple centroids: count=%s", len(apples))
    if apples:
        first = np.asarray(apples[0], dtype=np.float64).reshape(-1)[:3]
        log.info("First apple centroid: %s", np.round(first, 4).tolist())
        print(f"[probe] EE={np.round(ee_pos, 3).tolist()}  joints={np.round(joint_angles, 3).tolist()}")
        print(f"[probe] apples={len(apples)}  first_centroid={np.round(first, 3).tolist()}")
    else:
        print("[probe] WARNING: no apple centroids found")
        log.warning("No apple centroids found on tree %s", tree.id_str)

    camera = next((s for n, s in robot.sensors.items() if "camera" in n.lower()), None)
    if camera is None:
        log.warning("No camera sensor on robot; skipping RGB probe")
        print("[probe] camera: not found on robot.sensors")
        return

    try:
        view_matrix = robot.get_view_mat_at_curr_pose(camera)
        rgb_hwc, _ = robot.get_rgbd_at_cur_pose(
            camera=camera, type="sensor", view_matrix=view_matrix
        )
        shape = np.asarray(rgb_hwc).shape
        log.info("Camera RGB shape (H,W,C): %s", shape)
        print(f"[probe] camera RGB shape={shape}")
    except Exception as e:
        log.warning("Camera RGB probe failed: %s", e)
        print(f"[probe] camera RGB probe failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyBullet scene test (robot + tree) with logs/tests/ output."
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use DIRECT mode (no GUI); overrides simulation_setup.renders",
    )
    parser.add_argument("--tree-id", type=int, default=None, help="Override tree_setup.tree_id")
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=None,
        help="Physics settle steps after load (default: simulation_setup.settle_steps)",
    )
    parser.add_argument(
        "--hold-gui",
        action="store_true",
        help="After the check, keep stepping until Ctrl+C (GUI only)",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Grey tree only (skip bark MTL/texture); room backdrop unchanged",
    )
    parser.add_argument(
        "--preserve-export-mtl",
        action="store_true",
        help="Force v2 *_textured.mtl (default: auto if file exists)",
    )
    parser.add_argument(
        "--tree-scale",
        type=float,
        default=None,
        help="Override tree_setup.scale (default 0.8, matches planner)",
    )
    parser.add_argument(
        "--show-markers",
        action="store_true",
        help="Red spheres at apple centroids (r=0.04) and green leaf vertex samples",
    )
    parser.add_argument(
        "--max-leaf-markers",
        type=int,
        default=DEFAULT_MAX_LEAF_MARKERS,
        help="Max green leaf debug spheres when --show-markers (default: %(default)s)",
    )
    args = parser.parse_args()

    log_path = setup_test_file_logging()
    print(f"[scene] log file: {log_path}")
    if args.no_texture:
        log.info("CLI: --no-texture (grey tree, no OBJ/MTL bark)")
        print("[scene] --no-texture: grey tree (no bark MTL/texture)")
    if args.show_markers:
        log.info("CLI: --show-markers (apple + leaf debug spheres)")
        print("[scene] --show-markers: red apple centroids, green leaf samples")

    config = copy.deepcopy(TEST_CONFIG)
    if args.direct:
        config["simulation_setup"]["renders"] = False
    if args.tree_id is not None:
        config["tree_setup"]["tree_id"] = args.tree_id
    if args.tree_scale is not None:
        config["tree_setup"]["scale"] = float(args.tree_scale)
    if args.preserve_export_mtl:
        config["tree_setup"]["preserve_export_mtl"] = True
    if args.no_texture:
        config["tree_setup"]["apply_texture"] = False
    settle_steps = (
        args.settle_steps
        if args.settle_steps is not None
        else config["simulation_setup"]["settle_steps"]
    )

    pbutils = PyBUtils(renders=config["simulation_setup"]["renders"])
    robot, tree = build_scene(
        pbutils,
        config,
        apply_tree_texture=not args.no_texture,
    )
    log_scene_physics_summary(
        pbutils.pbclient,
        context="scene_after_load",
        tree_body_id=tree.pyb_id,
        robot_body_id=robot.robot,
    )
    n_apples = log_scene_summary(pbutils, robot, tree, config)

    log.info("Settling simulation for %s steps...", settle_steps)
    print(f"[scene] settling {settle_steps} physics steps...")
    for _ in range(settle_steps):
        pbutils.pbclient.stepSimulation()

    if tree.pyb_id is not None and tree.pyb_id >= 0:
        log.info("Tree mesh check after %s settle steps", settle_steps)
        tree.log_pybullet_visual_summary(pbutils.pbclient, phase="after-settle")
        tree.log_tree_parts_integrity(
            pbutils.pbclient,
            context="scene_after_settle",
            require_hidden_base=bool(config["tree_setup"].get("hide_collision_visual", True)),
        )

    if args.show_markers:
        spawn_scene_markers(
            pbutils,
            tree,
            config,
            max_leaf_markers=max(0, args.max_leaf_markers),
        )
        log_scene_physics_summary(
            pbutils.pbclient,
            context="scene_after_markers",
            tree_body_id=tree.pyb_id,
            robot_body_id=robot.robot,
        )
        tree.log_tree_parts_integrity(
            pbutils.pbclient,
            context="scene_after_markers",
            require_hidden_base=bool(config["tree_setup"].get("hide_collision_visual", True)),
        )

    probe_scene(robot, tree, pbutils)

    if n_apples <= 0:
        log.error("Scene test failed: zero apple centroids")
        print("SCENE TEST FAILED (no apples)")
        sys.exit(1)

    log.info("Scene test finished OK.")
    print("SCENE TEST OK")

    renders = config["simulation_setup"]["renders"]
    if renders and args.hold_gui:
        log.info("Holding GUI open; press Ctrl+C to exit.")
        print("[scene] holding GUI (--hold-gui); Ctrl+C to exit")
        try:
            while True:
                pbutils.pbclient.stepSimulation()
                time.sleep(pbutils.step_time)
        except KeyboardInterrupt:
            log.info("Interrupted; exiting.")
    elif renders and not args.hold_gui:
        log.info("GUI mode: exiting (pass --hold-gui to keep window open).")
        print("[scene] GUI loaded; exiting (use --hold-gui to keep open)")
    else:
        log.info("DIRECT mode: exiting.")


if __name__ == "__main__":
    main()
