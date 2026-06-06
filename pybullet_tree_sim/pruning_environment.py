#!/usr/bin/env python3
from pybullet_tree_sim import (
    CONFIG_PATH,
    MESHES_PATH,
    URDF_PATH,
    RGB_LABEL,
    ROBOT_URDF_PATH,
)
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree, TreeException

# from pybullet_tree_sim.utils.ur5_utils import UR5
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.xacro_utils as xutils
import pybullet_tree_sim.utils.yaml_utils as yutils

from pybullet_tree_sim import plot

# from final_approach_controller.cut_point_rotate_axis_controller import CutPointRotateAxisController


from collections import defaultdict
import cv2
import gymnasium as gym
import numpy as np
import os
import skimage.draw as skdraw
import sys
import time
from typing import Optional, Tuple

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import modern_robotics as mr
from numpy.typing import ArrayLike

from zenlog import log
import pprint as pp


class PruningEnvException(Exception):
    def __init__(self, s, *args):
        super().__init__(args)
        self.s = s
        return

    def __str__(self):
        return f"{self.s}"


class PruningEnv(gym.Env):
    rgb_label = RGB_LABEL
    """
        PruningEnv is a custom environment that extends the gym.Env class from OpenAI Gym.
        This environment simulates a pruning task where a robot arm interacts with a tree.
        The robot arm is a UR5 arm and the tree is a 3D model of a tree.
        The environment is used to train a reinforcement learning agent to prune the tree.
    """

    _supports_and_post_xacro_path = os.path.join(URDF_PATH, "supports_and_post", "supports_and_post.urdf.xacro")
    _supports_and_post_urdf_path = os.path.join(URDF_PATH, "supports_and_post", "supports_and_post.urdf")
    _shapes_xacro_dir = os.path.join(URDF_PATH, "shapes")

    def __init__(
        self,
        pbutils: PyBUtils,
        # max_steps: int = 1000,
        # make_trees: bool = False,
        name: str = "PruningEnv",
        # num_trees: int | None = None,
        # renders: bool = False,
        verbose: bool = True,
    ) -> None:
        """Initialize the Pruning Environment

        Parameters
        ----------
            ...
        """
        super().__init__()
        self.pbutils = pbutils

        # Pybullet GUI variables
        self.render_mode = "rgb_array"
        # self.renders = renders
        # self.eval = evaluate

        # Gym variables
        self.name = name
        self.step_counter = 0
        self.global_step_counter = 0
        # self.max_steps = max_steps

        self.verbose = verbose

        self.collision_object_ids = {  # TODO: move to tree.py
            "SPUR": None,
            "TRUNK": None,
            "BRANCH": None,
            "WATER_BRANCH": None,
            "SUPPORT": None,
        }

        self.trees = {}
        self.debouce_time = 0.5
        self.last_button_push_time = time.time()
        return

    def load_tree(  # TODO: Clean up Tree init vs create_tree, probably not needed. Too many file checks.
        self,
        pbutils: PyBUtils,
        scale: float,
        position: np.ndarray = np.array([0, 0, 0]),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        tree_id: int = None,
        tree_type: str = None,
        tree_namespace: str = "",
        tree_urdf_path: str | None = None,
        save_tree_urdf: bool = False,
        randomize_pose: bool = False,
    ) -> str:
        if randomize_pose:
            ...

        # If user supplied URDF file, check if it exists
        if tree_urdf_path is not None:
            if not os.path.exists(tree_urdf_path):
                raise OSError(
                    f"There do not seem to be any files of that name, please check your path: {tree_urdf_path}"
                )

        # Get tree object
        tree = Tree(
            pbutils=pbutils,
            scale=scale,
            position=position,
            orientation=orientation,
            tree_id=tree_id,
            tree_type=tree_type,
            namespace=tree_namespace,
            urdf_path=tree_urdf_path,
            randomize_pose=randomize_pose,
        )

        # Add tree to dict of trees
        self.trees[tree.id_str] = tree
        return tree.id_str

    def get_tree_from_id_str(self, tree_id_str: str) -> Tree:
        try:
            tree = self.trees[tree_id_str]
        except KeyError as e:
            raise TreeException(f"{e}: Tree with ID {tree_id_str} not found")
        return tree

    def activate_tree(
        self,
        tree: Tree,
        include_support_posts: bool = True,
    ) -> None:
        """Activate a tree by object or by tree_id_str. Can include support posts. Must provide either a Tree or tree_id_str.
        @param tree (Tree/None): Tree object to be activated into the pruning environment.
        @param tree_id_str (str/None): String including the identification characteristics of the tree.
        @return None
        """
        if tree:
            if self.verbose:
                log.info("Activating tree")
            tree.load_pybullet_body(self.pbutils.pbclient)
            log.info(f"Tree {tree.id_str} activated with PyBID {tree.pyb_id}")

            if include_support_posts:
                self.activate_support_posts(associated_tree=tree)
        return

    def activate_tree_by_id_str(self, tree_id_str: str, include_support_posts: bool = True) -> None:
        tree = self.get_tree_from_id_str(tree_id_str=tree_id_str)
        self.activate_tree(tree=tree, include_support_posts=include_support_posts)
        return

    def deactivate_tree(self, tree: Tree) -> None:
        """Deactivate a tree object"""
        try:
            self.pbutils.pbclient.removeBody(tree.pyb_id)
            log.info(f"Tree {tree.id_str} with PyBID {tree.pyb_id} deactivated")
        except Exception as e:
            log.error(f"Error deactivating tree: {e}")
        return

    def deactivate_tree_by_id_str(self, tree_id_str: str) -> None:
        tree = self.get_tree_from_id_str(tree_id_str=tree_id_str)
        self.deactivate_tree(tree=tree)
        return

    def activate_support_posts(
        self,
        associated_tree: Tree,
        position: ArrayLike | None = None,
        orientation: ArrayLike | None = None,
    ) -> None:
        # Load support posts
        if position is None:
            position = [
                associated_tree.pos[0],
                associated_tree.pos[1] - 0.05,
                0.0,
            ]
        if orientation is None:
            orientation = Rotation.from_euler("xyz", [np.pi / 2, 0, np.pi / 2]).as_quat()

        if not os.path.exists(
            self._supports_and_post_urdf_path
        ):  # TODO: change this -- it'll take a improperly positioned file and keep using it.'
            urdf_content = xutils.load_urdf_from_xacro(
                xacro_path=self._supports_and_post_xacro_path, mappings=None
            ).toprettyxml()  # TODO: add mappings
            xutils.save_urdf(
                urdf_content=urdf_content,
                urdf_path=self._supports_and_post_urdf_path,
            )

        support_post_id = self.pbutils.pbclient.loadURDF(
            fileName=self._supports_and_post_urdf_path,
            basePosition=position,
            baseOrientation=orientation,
        )
        log.info(f"Supports and post activated with PyBID {support_post_id}")
        self.collision_object_ids["SUPPORT"] = support_post_id
        return

    def deactivate_support_posts(self) -> None:
        # try:
        #     self.pbutils.pbclient.removeBody(self.collision_object_ids["SUPPORT"])
        #     log.info(f"Supports and post deactivated")
        # except Exception as e:
        #     log.error(f"Error deactivating supports and post: {e}")
        return

    def reset_environment(self) -> None:
        """TODO"""
        # self.pbutils.pbclient.resetSimulation()
        return

    def activate_shape(
        self,
        shape: str,
        position: ArrayLike | None = None,
        orientation: ArrayLike | None = None,
        **kwargs,
    ) -> None:
        """Activate a generic cylinder object in the environment.
        @param shape (str): shape type. Currently supported options are: [cylinder]
        @param position (ArrayLike): Vector containing the xyz position of the base. Cylinder default position is the center of the cylinder.
        @param orientation (ArrayLike): Vector containing the roll-pitch-yaw of the base about the world origin.
        @param radius (float, Optional): only valid if shape is "cylinder"
        @return None
        """
        # log.warning(locals())
        shape = shape.strip().lower()
        shape_xacro_path = os.path.join(self._shapes_xacro_dir, shape, f"{shape}.urdf.xacro")
        shape_urdf_path = os.path.join(self._shapes_xacro_dir, shape, f"{shape}.urdf")

        shape_mappings = {}
        for key, value in kwargs.items():
            shape_mappings[key] = str(value)

        # shape position is the center of the shape, move up by half the height
        if position is None:
            position = [0, 0, float(shape_mappings["height"]) / 2]
        if orientation is None:
            orientation = [0, 0, 0, 1]
        else:
            orientation = Rotation.from_euler("xyz", orientation).as_quat()

        # print(position)

        # if shape == "cylinder":
        #     radius = kwargs.get('radius')
        #     height = kwargs.get('height')

        urdf_content = xutils.load_urdf_from_xacro(xacro_path=shape_xacro_path, mappings=shape_mappings).toprettyxml()
        xutils.save_urdf(urdf_content=urdf_content, urdf_path=shape_urdf_path)

        shape_id = self.pbutils.pbclient.loadURDF(
            fileName=shape_urdf_path,
            basePosition=position,
            baseOrientation=orientation,
        )
        log.info(f"{shape.title()} loaded with PyBID {shape_id}")

        return

    def is_reachable(self, robot: Robot, vertex: Tuple[np.ndarray], base_xyz: np.ndarray) -> bool:
        # if vertex[3] != "SPUR":
        #     return False
        ur5_base_pos = np.array(base_xyz)

        # Meta condition
        dist = np.linalg.norm(ur5_base_pos - vertex[0], axis=-1)

        if dist >= 0.98:  # TODO: is this for the UR5? Should it be from a parameter file?
            return False

        j_angles = robot.calculate_ik(vertex[0], None)
        # env.ur5.set_joint_angles(j_angles)
        # for _ in range(100):
        #     pyb.con.stepSimulation()
        # ee_pos, _ = env.ur5.get_current_pose(env.ur5.end_effector_index)
        # dist = np.linalg.norm(np.array(ee_pos) - vertex[0], axis=-1)
        # if dist <= 0.03:
        #     return True

        return False

    def get_reachable_points(self, tree: Tree, env, pyb) -> list:
        # reachable_points = list(filter(lambda x: self.is_reachable(x, env, pyb), self.vertex_and_projection))
        # np.random.shuffle(self.reachable_points)
        # print("Number of reachable points: ", len(self.reachable_points))
        # if len(self.reachable_points) < 1:
        #     print("No points in reachable points", self.urdf_path)
        #     # self.reset_tree()
        reachable_points = []
        return reachable_points

    def get_key_pressed(self, relevant=None) -> list:
        """Return the keys pressed by the user."""
        keys_pressed = []
        events = self.pbutils.pbclient.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            keys_pressed.append(key)
        return keys_pressed


def main():
    from pybullet_tree_sim.time_of_flight import TimeOfFlight
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    from pybullet_tree_sim.robot import Robot
    import numpy as np
    import secrets

    pbutils = PyBUtils(renders=False)
    penv = PruningEnv(pbutils=pbutils)
    tof0 = TimeOfFlight(pbclient=pbutils.pbclient, sensor_name="vl53l8cx")
    robot = Robot(pbclient=pbutils.pbclient)

    depth_data = np.zeros((tof0.depth_width, tof0.depth_height), dtype=float)
    # generator = np.random.default_rng(seed=secrets.randbits(128))
    # data[0,0] = 0.31
    # depth_data[:, :] = tuple(generator.uniform(0.31, 0.35, (tof0.depth_width, tof0.depth_height)))

    start = 0.31
    stop = 0.35
    # Depth data IRL comes in as a C-format nx1 array. start with this IRL
    depth_data[:, 3:5] = np.array(
        [
            np.arange(start, stop, (stop - start) / 8),
            np.arange(start, stop, (stop - start) / 8),
        ]
    ).T
    depth_data[-1, 3] = 0.31
    # Switch to F-format
    depth_data = depth_data.reshape((tof0.depth_width * tof0.depth_height, 1), order="F")

    view_matrix = np.identity(4)
    view_matrix[:3, 3] = -1 * np.array([0, 0, 1])

    world_points = robot.deproject_pixels_to_points(camera=tof0, data=depth_data, view_matrix=view_matrix, debug=True)

    # log.warning(f"joint angles: {penv.ur5.get_joint_angles()}")

    return


if __name__ == "__main__":
    main()
