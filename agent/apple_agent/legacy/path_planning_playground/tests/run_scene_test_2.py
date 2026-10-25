#!/usr/bin/env python3
"""
Scene test v2: load one tree by id, or compare two trees side-by-side (default 8 vs 10).

Run from repository root:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test_2.py --hold-gui

Single tree:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test_2.py --tree-id 10 --hold-gui

Compare tree 8 and 10 (spaced along Y):

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test_2.py --compare --hold-gui

Custom pair / spacing:

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_scene_test_2.py --tree-ids 8,10 --spacing 4 --hold-gui
"""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from zenlog import log

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.pyb_utils import PyBUtils

LOG_SUBDIR = "tests"

APPLE_MARKER_RADIUS = 0.04
APPLE_MARKER_RGBA = [0.9, 0.08, 0.08, 0.9]
LEAF_MARKER_RADIUS = 0.025
LEAF_MARKER_RGBA = [0.2, 0.75, 0.25, 0.65]
DEFAULT_MAX_LEAF_MARKERS = 200

# Side-by-side compare defaults
DEFAULT_COMPARE_TREE_IDS = (8, 10)
DEFAULT_TREE_SPACING_M = 3.0

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
        "tree_type": "envy",
        "tree_namespace": "LPy",
        "scale": 0.6,
        "orientation": np.array([0, 0, 0, 1]),
        "base_x": 0.5,
        "base_z": 0.0,
    },
}


def setup_test_file_logging(suffix: str = "scene_test_2") -> Path:
    base = _REPO_ROOT / "logs" / LOG_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_{suffix}.log").resolve()
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


def parse_tree_ids(text: str) -> list[int]:
    ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("expected at least one tree id, e.g. 8,10")
    return ids


def tree_positions_along_y(
    tree_ids: list[int],
    spacing: float,
    *,
    base_x: float,
    base_z: float,
    center_y: float = 0.0,
) -> dict[int, np.ndarray]:
    """Place tree centers evenly along Y (for side-by-side GUI comparison)."""
    n = len(tree_ids)
    if n == 1:
        return {tree_ids[0]: np.array([base_x, center_y, base_z], dtype=np.float64)}
    half = (n - 1) / 2.0
    return {
        tid: np.array([base_x, center_y + (i - half) * spacing, base_z], dtype=np.float64)
        for i, tid in enumerate(tree_ids)
    }


def resolve_tree_ids(args: argparse.Namespace) -> list[int]:
    if args.compare:
        if args.tree_ids is not None:
            return args.tree_ids
        return list(DEFAULT_COMPARE_TREE_IDS)
    if args.tree_id is not None:
        return [args.tree_id]
    if args.tree_ids is not None:
        return args.tree_ids
    return [10]


def load_robot(pbutils: PyBUtils, config: dict) -> Robot:
    log.info("Loading Robot...")
    robot_conf = config["robot_setup"]
    quat = Rotation.from_euler(
        "xyz", robot_conf["start_orientation_euler_deg"], degrees=True
    ).as_quat()
    robot = Robot(
        pbclient=pbutils.pbclient,
        position=robot_conf["start_position"],
        orientation=quat,
        regenerate_urdf=False,
    )
    if robot_conf.get("start_joint_angles"):
        robot.set_joint_angles_no_collision(robot_conf["start_joint_angles"])
    return robot


def load_trees(
    pbutils: PyBUtils,
    config: dict,
    tree_ids: list[int],
    positions: dict[int, np.ndarray],
    *,
    apply_tree_texture: bool,
) -> list[Tree]:
    tree_conf = config["tree_setup"]
    trees: list[Tree] = []
    for tree_id in tree_ids:
        pos = positions[tree_id]
        log.info("Loading Tree id=%s at position %s", tree_id, np.round(pos, 3).tolist())
        print(f"[scene] loading tree {tree_id} at {np.round(pos, 3).tolist()}")
        tree = Tree(
            pbutils=pbutils,
            tree_id=tree_id,
            tree_type=tree_conf["tree_type"],
            namespace=tree_conf["tree_namespace"],
            scale=tree_conf["scale"],
            position=pos,
            orientation=tree_conf["orientation"],
        )
        tree.load_pybullet_body(
            pbutils.pbclient,
            apply_bark_texture=apply_tree_texture,
        )
        trees.append(tree)
    visual_mode = "bark (OBJ/MTL)" if apply_tree_texture else "grey (--no-texture)"
    log.info("Tree visual mode: %s", visual_mode)
    print(f"[scene] tree visual: {visual_mode}  count={len(trees)}")
    return trees


def log_trees_summary(trees: list[Tree]) -> None:
    log.info("=== Scene test v2 summary (%d trees) ===", len(trees))
    for tree in trees:
        log.info("--- Tree %s ---", tree.id_str)
        log.info("  position: %s", np.round(tree.pos, 4).tolist())
        log.info("  urdf_path: %s", tree.urdf_path)
        log.info("  mesh_path: %s", tree.mesh_path)
        log.info("  pyb_id: %s  scale: %s", tree.pyb_id, tree.scale)
        n_apples = len(tree.get_apple_centroids())
        log.info("  apple_centroids: %s", n_apples)
        print(f"[scene] {tree.id_str}  pyb_id={tree.pyb_id}  apples={n_apples}  pos={np.round(tree.pos, 3).tolist()}")


def create_debug_sphere(pb, position, radius: float, rgba) -> int:
    try:
        vid = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        return pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vid,
            basePosition=list(np.asarray(position, dtype=np.float64).reshape(-1)[:3]),
        )
    except Exception as exc:
        log.warning("Debug sphere failed at %s: %s", position, exc)
        return -1


def sample_leaf_positions(tree: Tree, max_markers: int, seed: int) -> list[np.ndarray]:
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


def spawn_markers_for_trees(
    pbutils: PyBUtils,
    trees: list[Tree],
    *,
    max_leaf_markers: int,
) -> None:
    pb = pbutils.pbclient
    for i, tree in enumerate(trees):
        seed = tree.tree_id if tree.tree_id is not None else i
        for center in tree.get_apple_centroids():
            create_debug_sphere(pb, center, APPLE_MARKER_RADIUS, APPLE_MARKER_RGBA)
        for pos in sample_leaf_positions(tree, max_leaf_markers, seed=seed):
            create_debug_sphere(pb, pos, LEAF_MARKER_RADIUS, LEAF_MARKER_RGBA)
        log.info(
            "Markers for %s: %d apples, up to %d leaf samples",
            tree.id_str,
            len(tree.get_apple_centroids()),
            max_leaf_markers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyBullet scene test v2: one tree or side-by-side compare."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tree-id",
        type=int,
        default=None,
        help="Load a single tree by sim id (e.g. 10)",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help=f"Load trees {DEFAULT_COMPARE_TREE_IDS[0]} and {DEFAULT_COMPARE_TREE_IDS[1]} side by side",
    )
    parser.add_argument(
        "--tree-ids",
        type=parse_tree_ids,
        default=None,
        metavar="IDS",
        help="Comma-separated tree ids (e.g. 8,10). With --compare, overrides default pair.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_TREE_SPACING_M,
        help="Meters between tree centers along Y when loading multiple trees (default: %(default)s)",
    )
    parser.add_argument("--direct", action="store_true", help="DIRECT mode (no GUI)")
    parser.add_argument("--hold-gui", action="store_true", help="Keep GUI open until Ctrl+C")
    parser.add_argument("--no-texture", action="store_true", help="Grey tree mesh (no bark MTL)")
    parser.add_argument("--show-markers", action="store_true", help="Apple (red) and leaf (green) spheres")
    parser.add_argument("--max-leaf-markers", type=int, default=DEFAULT_MAX_LEAF_MARKERS)
    parser.add_argument("--settle-steps", type=int, default=None)
    args = parser.parse_args()

    tree_ids = resolve_tree_ids(args)
    log_path = setup_test_file_logging()
    print(f"[scene] log file: {log_path}")
    print(f"[scene] tree ids: {tree_ids}  spacing={args.spacing} m")

    config = copy.deepcopy(TEST_CONFIG)
    if args.direct:
        config["simulation_setup"]["renders"] = False
    settle_steps = args.settle_steps or config["simulation_setup"]["settle_steps"]

    tconf = config["tree_setup"]
    positions = tree_positions_along_y(
        tree_ids,
        args.spacing,
        base_x=tconf["base_x"],
        base_z=tconf["base_z"],
    )

    pbutils = PyBUtils(renders=config["simulation_setup"]["renders"])
    pbutils.reset_simulation_scene(gravity=config["simulation_setup"]["gravity"])
    robot = load_robot(pbutils, config)
    trees = load_trees(
        pbutils,
        config,
        tree_ids,
        positions,
        apply_tree_texture=not args.no_texture,
    )

    log.info("--- Bodies in client ---")
    pb = pbutils.pbclient
    for i in range(pb.getNumBodies()):
        info = pb.getBodyInfo(i)
        name = info[1].decode("utf-8", errors="replace") if info[1] else "?"
        log.info("  body_index=%s  name=%s", i, name)

    log_trees_summary(trees)

    log.info("Settling simulation for %s steps...", settle_steps)
    for _ in range(settle_steps):
        pb.stepSimulation()

    for tree in trees:
        if tree.pyb_id is not None and tree.pyb_id >= 0:
            tree.log_pybullet_visual_summary(pb, phase="after-settle")

    if args.show_markers:
        spawn_markers_for_trees(
            pbutils,
            trees,
            max_leaf_markers=max(0, args.max_leaf_markers),
        )

    log.info("Scene test v2 finished OK.")
    print("SCENE TEST 2 OK")

    if config["simulation_setup"]["renders"] and args.hold_gui:
        log.info("Holding GUI open; press Ctrl+C to exit.")
        print("[scene] holding GUI (--hold-gui); Ctrl+C to exit")
        try:
            while True:
                pb.stepSimulation()
                time.sleep(pbutils.step_time)
        except KeyboardInterrupt:
            log.info("Interrupted; exiting.")
    elif config["simulation_setup"]["renders"] and not args.hold_gui:
        print("[scene] GUI loaded; use --hold-gui to keep open")


if __name__ == "__main__":
    main()
