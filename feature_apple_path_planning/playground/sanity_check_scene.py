#!/usr/bin/env python3
"""
Lightweight PyBullet sanity check: same Robot + Tree load sequence as
ApplePickingEnv._setup_scene, without the full Gym environment or planner deps.

Use before running path-planning / data-generation scripts to confirm assets
and PyBullet load correctly.

First run for a new tree id can be slow while Tree builds/caches point data
under PKL_PATH; that is expected, not a failure.

Run from repository root (venv activated, dependencies installed):

    PYTHONPATH=. python feature_apple_path_planning/playground/sanity_check_scene.py

Headless (no GUI, exits when done):

    PYTHONPATH=. python feature_apple_path_planning/playground/sanity_check_scene.py --direct

With the default GUI, the scene stays open after the check; press Ctrl+C in the
terminal to quit.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from zenlog import log

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.pyb_utils import PyBUtils

# Mirrors simulation_setup / robot_setup / tree_setup in run_apple_data_generator.CONFIG
MINIMAL_CONFIG = {
    "simulation_setup": {
        "renders": True,
        "gravity": -9.81,
    },
    "robot_setup": {
        "start_position": [0, 1.2, 0],
        "start_orientation_euler_deg": [0, 0, 180],
        "start_joint_angles": [-2.4, -1.57, 1.57, -1.57, 1.57 / 8, 0],
    },
    "tree_setup": {
        "tree_id": 8,
        "tree_type": "envy",
        "tree_namespace": "LPy",
        "scale": 0.6,
        "position": np.array([0.5, 0, 0]),
        "orientation": np.array([0, 0, 0, 1]),
    },
}


def build_scene(pbutils: PyBUtils, config: dict) -> tuple[Robot, Tree]:
    pbutils.reset_simulation_scene(gravity=config["simulation_setup"]["gravity"])

    log.info("Loading Robot...")
    robot_conf = config["robot_setup"]
    robot_start_orientation_quat = Rotation.from_euler(
        "xyz", robot_conf["start_orientation_euler_deg"], degrees=True
    ).as_quat()

    robot = Robot(
        pbclient=pbutils.pbclient,
        position=robot_conf["start_position"],
        orientation=robot_start_orientation_quat,
    )

    log.info("Loading Tree...")
    tree_conf = config["tree_setup"]
    tree = Tree(
        pbutils=pbutils,
        tree_id=tree_conf["tree_id"],
        tree_type=tree_conf["tree_type"],
        namespace=tree_conf["tree_namespace"],
        scale=tree_conf["scale"],
        position=tree_conf["position"],
        orientation=tree_conf["orientation"],
    )
    tree.load_pybullet_body(pbutils.pbclient)
    return robot, tree


def log_scene_summary(pbutils: PyBUtils, robot: Robot, tree: Tree, config: dict) -> None:
    pb = pbutils.pbclient
    sim = config["simulation_setup"]
    mode = "GUI" if sim.get("renders") else "DIRECT"
    log.info("=== PyBullet sanity summary ===")
    log.info(f"Connection mode: {mode} (renders={sim.get('renders')})")
    log.info(f"Gravity (z): {sim['gravity']}  step_time: {pbutils.step_time}")

    log.info("--- Robot ---")
    log.info(f"robot_urdf_path: {robot.robot_urdf_path}")
    log.info(f"body_unique_id: {robot.robot}  num_joints: {pb.getNumJoints(robot.robot)}")
    log.info(f"tool0_link_idx: {robot.tool0_link_idx}")

    log.info("--- Tree ---")
    log.info(f"id_str: {tree.id_str}")
    log.info(f"urdf_path: {tree.urdf_path}")
    log.info(f"mesh_path (unlabeled): {tree.mesh_path}")
    log.info(f"labeled_mesh_path: {tree.labeled_mesh_path}")
    log.info(f"pyb_id: {tree.pyb_id}  scale: {tree.scale}")
    n_apples = len(tree.get_apple_centroids())
    log.info(f"apple_centroids count: {n_apples}")

    log.info("--- Bodies in client ---")
    for i in range(pb.getNumBodies()):
        info = pb.getBodyInfo(i)
        name = info[1].decode("utf-8", errors="replace") if info[1] else "?"
        log.info(f"  body_index={i}  name={name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBullet scene sanity check (robot + tree).")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use DIRECT mode (no GUI), overrides simulation_setup.renders",
    )
    args = parser.parse_args()

    config = copy.deepcopy(MINIMAL_CONFIG)
    if args.direct:
        config["simulation_setup"]["renders"] = False

    pbutils = PyBUtils(renders=config["simulation_setup"]["renders"])
    robot, tree = build_scene(pbutils, config)
    log_scene_summary(pbutils, robot, tree, config)

    n_steps = 100
    log.info(f"Stepping simulation {n_steps} times...")
    for _ in range(n_steps):
        pbutils.pbclient.stepSimulation()
    log.info("Sanity check finished OK.")

    if config["simulation_setup"]["renders"]:
        log.info("Keeping PyBullet open; press Ctrl+C in this terminal to exit.")
        try:
            while True:
                pbutils.pbclient.stepSimulation()
                time.sleep(pbutils.step_time)
        except KeyboardInterrupt:
            log.info("Interrupted; exiting.")
    else:
        log.info("DIRECT mode: no window to hold; exiting.")


if __name__ == "__main__":
    main()
