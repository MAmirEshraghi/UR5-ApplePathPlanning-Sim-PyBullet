"""
apple_picking_env.py

Author: Robin Eshraghi (@MAmirEshraghi)
Affiliation: Robotics Lab, Oregon State University
Supervisor: Prof. Cindy Grimm

Project: Robotic Apple Picking Simulation Environment
Part of Internship Project on Robotic Manipulation and RL-based Motion Planning

Description:
    This module defines the `ApplePickingEnv` class — a Gymnasium-compatible environment
    for simulating apple-picking tasks in PyBullet. It integrates the robot arm, tree model,
    and vision sensors to support reinforcement learning, motion planning, and trajectory
    generation studies. The environment provides camera observations, end-effector states,
    and reward computation for learning-based control.

Usage:
    from apple_picking_env import ApplePickingEnv
    env = ApplePickingEnv(config)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)

Date Started: 2025-01-04
Date completed: 2025-01-08
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import cv2
import os
from scipy.spatial.transform import Rotation
from zenlog import log
import copy

# Import your existing project classes
from pybullet_tree_sim import URDF_PATH
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree

from .utils_conversions import convert_local_action_to_global, convert_global_action_to_local

import time

_CLR_GREEN = "\033[92m"
_CLR_RED = "\033[91m"
_CLR_RESET = "\033[0m"


def resolve_preserve_export_mtl(tree_conf: dict) -> bool:
    """Auto-detect v2 *_textured.mtl when preserve_export_mtl is unset."""
    preserve = tree_conf.get("preserve_export_mtl")
    if preserve is not None:
        return bool(preserve)
    return Tree.has_textured_export_mtl(
        tree_conf["tree_id"],
        tree_conf["tree_type"],
        tree_conf.get("tree_namespace", "LPy"),
    )


def ensure_tree_sim_assets(tree: Tree, tree_conf: dict) -> None:
    """Ensure OBJ/URDF exist; build multi-material URDF when v2 MTL is present."""
    from pathlib import Path

    from pybullet_tree_sim.obj_multimat import (
        parts_mtl_filename,
        parts_subdir,
        ensure_mat_leaf_texture_maps,
    )

    if not os.path.isfile(tree.mesh_path):
        raise FileNotFoundError(
            f"Unlabeled OBJ missing for {tree.id_str}: {tree.mesh_path}\n"
            f"Run: python lpy_treesim/export_pybullet_tree_v2.py --tree-id {tree.tree_id} "
            f"--lpy-index <N> --regenerate-urdf"
        )
    preserve = resolve_preserve_export_mtl(tree_conf)
    textured = Tree.has_textured_export_mtl(
        tree.tree_id, tree.tree_type, tree.tree_namespace
    )
    need_textured_urdf = preserve and textured
    if not os.path.isfile(tree.urdf_path):
        log.info("Generating URDF for %s", tree.id_str)
        if need_textured_urdf:
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
    elif tree_conf.get("regenerate_urdf") and need_textured_urdf:
        log.info("Regenerating multi-material URDF for %s", tree.id_str)
        Tree.regenerate_textured_urdf(
            tree_id=tree.tree_id,
            tree_type=tree.tree_type,
            namespace=tree.tree_namespace,
        )

    textured_mtl = Path(Tree.textured_mtl_path_for_id(tree.id_str))
    if textured_mtl.is_file():
        ensure_mat_leaf_texture_maps(textured_mtl)
    parts_mtl = parts_subdir(Path(tree._tree_meshes_unlabeled_path), tree.id_str) / parts_mtl_filename(
        tree.id_str
    )
    if parts_mtl.is_file():
        ensure_mat_leaf_texture_maps(parts_mtl)


def frame_debug_camera_on_tree(pb_client, tree_position, sim_conf: dict) -> None:
    """Match run_sim_tree_viewer.py so the GUI frames the tree, not the robot base."""
    if not sim_conf.get("frame_tree_camera", True):
        return
    pos = np.asarray(tree_position, dtype=np.float64).reshape(3)
    z_off = float(sim_conf.get("debug_camera_target_z_offset", 1.0))
    target = [float(pos[0]), float(pos[1]), float(pos[2]) + z_off]
    pb_client.resetDebugVisualizerCamera(
        cameraDistance=float(sim_conf.get("debug_camera_distance", 4.5)),
        cameraYaw=float(sim_conf.get("debug_camera_yaw", 55)),
        cameraPitch=float(sim_conf.get("debug_camera_pitch", -22)),
        cameraTargetPosition=target,
    )
    log.info("Debug camera framed on tree at %s", np.round(target, 3).tolist())


def load_tree_pybullet_body(pb_client, tree: Tree, tree_conf: dict, **load_kwargs) -> int:
    """Load tree URDF with v2 multi-material MTL when exported (matches sim viewers)."""
    apply_texture = bool(tree_conf.get("apply_texture", True))
    preserve = resolve_preserve_export_mtl(tree_conf)
    ensure_tree_sim_assets(tree, tree_conf)
    hide_collision_visual = bool(tree_conf.get("hide_collision_visual", False))
    return tree.load_pybullet_body(
        pb_client,
        apply_bark_texture=apply_texture,
        use_obj_mtl=True,
        preserve_exported_mtl=preserve and apply_texture,
        hide_collision_visual=hide_collision_visual,
        **load_kwargs,
    )


def _debug_log_collision_line(step_counter: int, reward_info: dict) -> None:
    """One-line per env step: green if no classified contact, red if acceptable/unacceptable."""
    acc = bool(reward_info.get("collisions_acceptable"))
    unacc = bool(reward_info.get("collisions_unacceptable"))
    if acc or unacc:
        log.debug(
            "%s[collision] step=%s  CONTACT  acceptable=%s  unacceptable=%s%s",
            _CLR_RED, step_counter, acc, unacc, _CLR_RESET,
        )
    else:
        log.debug(
            "%s[collision] step=%s  clear (no classified contacts)%s",
            _CLR_GREEN, step_counter, _CLR_RESET,
        )

class ApplePickingEnv(gym.Env):
    """
    A gymnasium environment for the apple picking task, now with a full feature set
    compatible with the data generation pipeline.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: dict):
        super(ApplePickingEnv, self).__init__()

        self.config = config
        sim_conf = self.config['simulation_setup']
        self._draw_apple_centroid_markers = bool(
            sim_conf.get('draw_apple_centroid_markers', True)
        )
        self._apple_debug_body_ids = []
        robot_conf = self.config['robot_setup']
        planning_conf = self.config['planning']

        # --- Simulation and Robot/Tree Setup ---
        self.pbutils = PyBUtils(renders=sim_conf['renders'])
        self.pb_client = self.pbutils.pbclient
        self._setup_scene()

        self.action_scale = self.config['generator'].get('action_scale', 1.0)
        self.control_time = float(sim_conf.get('control_time', 1.0 / 20.0))
        self.num_control_simulation_steps = max(
            1, int(self.control_time / self.pbutils.step_time)
        )
        log.info(f"Control time: {self.control_time}, pbutils.step.time: {self.pbutils.step_time}, Steps per control: {self.num_control_simulation_steps}")

        self.max_steps = sim_conf.get('max_steps', 500)
        self.distance_threshold = planning_conf.get('fk_pos_tolerance', 0.05)

        # --- Get a reference to the camera sensor from the robot ---
        self.camera_sensor = next((s for n, s in self.robot.sensors.items() if "camera" in n), None)
        if self.camera_sensor is None:
            raise ValueError("No camera sensor found in the robot's configuration.")

        # --- Define Action and Observation Spaces ---
        self.action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)

        cam_height = self.camera_sensor.depth_height
        cam_width = self.camera_sensor.depth_width

        self.observation_space = spaces.Dict({
            "achieved_goal": spaces.Box(low=-5., high=5., shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-5., high=5., shape=(3,), dtype=np.float32),
            "achieved_or": spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32), # 6D orientation
            "joint_angles": spaces.Box(low=-1, high=1, shape=(len(robot_conf['start_joint_angles']) * 2,), dtype=np.float32), # sin/cos encoding
            "relative_distance": spaces.Box(low=-5., high=5., shape=(3,), dtype=np.float32),
            "prev_action_achieved": spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32),
            "rgb": spaces.Box(low=0, high=255, shape=(3, cam_height, cam_width), dtype=np.uint8),
            "prev_rgb": spaces.Box(low=0, high=255, shape=(3, cam_height, cam_width), dtype=np.uint8),
         #   "rgb": spaces.Box(low=0, high=255, shape=(3, cam_height, cam_width), dtype=np.uint8),
         #   "prev_rgb": spaces.Box(low=0, high=255, shape=(3, cam_height, cam_width), dtype=np.uint8),
            "point_mask": spaces.Box(low=0., high=1., shape=(cam_height, cam_width), dtype=np.float32),
        })

        # --- Initialize state tracking variables ---
        self.desired_goal = np.zeros(3)
        self.reward_goal = np.zeros(3)
        self.start_dist_to_goal = 0.0
        self.home_joint_angles = robot_conf['start_joint_angles']
        self.init_pos_ee, self.init_or_ee = self.robot.get_current_pose(self.robot.tool0_link_idx)
        self.reset_env_variables()


    def _setup_scene(self):
        """Loads the robot and tree into the simulation based on the config."""
        sim_setup = self.config["simulation_setup"]
        self.pbutils.reset_simulation_scene(
            gravity=sim_setup["gravity"],
            room_collision=bool(sim_setup.get("room_collision", False)),
        )
        log.info("Loading Robot...")
        robot_conf = self.config['robot_setup']
        robot_start_orientation_quat = Rotation.from_euler(
            'xyz', robot_conf['start_orientation_euler_deg'], degrees=True
        ).as_quat()

        cached_robot_urdf = os.path.join(URDF_PATH, "tmp", "robot.urdf")
        regenerate_robot_urdf = robot_conf.get("regenerate_urdf")
        if regenerate_robot_urdf is None:
            regenerate_robot_urdf = not os.path.isfile(cached_robot_urdf)
        if not regenerate_robot_urdf:
            log.info("Using cached robot URDF: %s", cached_robot_urdf)
        self.robot = Robot(
            pbclient=self.pb_client,
            position=robot_conf['start_position'],
            orientation=robot_start_orientation_quat,
            regenerate_urdf=regenerate_robot_urdf,
        )

        log.info("Loading Tree...")
        tree_conf = self.config['tree_setup']
        self.tree = Tree(
            pbutils=self.pbutils,
            tree_id=tree_conf['tree_id'],
            tree_type=tree_conf['tree_type'],
            namespace=tree_conf['tree_namespace'],
            scale=tree_conf['scale'],
            position=tree_conf['position'],
            orientation=tree_conf['orientation']
        )
        load_tree_pybullet_body(self.pb_client, self.tree, tree_conf)
        self.tree.build_collision_kdtree()
        self.apple_centroids = self.tree.get_apple_centroids()
        self._refresh_apple_debug_markers()
        if self.pbutils.renders:
            frame_debug_camera_on_tree(
                self.pb_client,
                tree_conf["position"],
                self.config["simulation_setup"],
            )

    def _refresh_apple_debug_markers(self):
        """Red spheres offset from apple centroids toward the robot (visible in GUI)."""
        from feature_apple_path_planning.viz_debug import (
            apple_marker_surface_position,
            draw_debug_sphere,
            gui_refresh,
            log_apple_marker_batch,
        )

        if not self._draw_apple_centroid_markers:
            log.info("VIZ_VALIDATE env_centroid_markers: disabled (draw_apple_centroid_markers=False)")
            return
        for bid in self._apple_debug_body_ids:
            if bid is not None and bid >= 0:
                try:
                    self.pb_client.removeBody(bid)
                except Exception:
                    pass
        self._apple_debug_body_ids = []
        vis = self.config.get("visualization", {})
        radius = float(vis.get("goal_sphere_radius", 0.08))
        offset_m = float(vis.get("apple_marker_surface_offset_m", 0.12))
        rgba = list(vis.get("centroid_marker_color", [0.9, 0.08, 0.08, 0.9]))
        robot_pos = np.asarray(self.config["robot_setup"]["start_position"], dtype=np.float64)
        centroids = []
        marker_positions = []
        for center in self.apple_centroids:
            centroid = np.asarray(center, dtype=np.float64).reshape(3)
            centroids.append(centroid)
            marker_pos = apple_marker_surface_position(centroid, robot_pos, offset_m)
            marker_positions.append(marker_pos)
            bid = draw_debug_sphere(self.pb_client, marker_pos, radius, rgba)
            self._apple_debug_body_ids.append(bid)
        log_apple_marker_batch(
            "env_centroid_markers",
            self._apple_debug_body_ids,
            centroids,
            marker_positions,
            radius,
            offset_m,
        )
        gui_refresh(
            self.pb_client,
            self.config["simulation_setup"].get("renders", False),
            steps=int(self.config["simulation_setup"].get("gui_refresh_steps", 1)),
        )

    def reset_env_variables(self):
        """Resets variables that change within an episode."""
        self.step_counter = 0
        self.is_goal_state = False
        self.sum_reward = 0.0
        self.observation = {}
        self.observation_info = {}
        self.prev_observation_info = {}
        self.action = np.zeros(self.action_space.shape)

    def snapshot_episode_start_dist(self):
        """Record EE-to-reward_goal distance at episode start (for training logs)."""
        ee_pos, _ = self.robot.get_current_pose(self.robot.tool0_link_idx)
        self.start_dist_to_goal = float(
            np.linalg.norm(np.asarray(ee_pos, dtype=np.float64) - self.reward_goal)
        )

    def _build_step_info(self, reward_info, terminate_info, terminated, truncated):
        info = {**reward_info, **terminate_info}
        info["start_dist_to_goal"] = float(self.start_dist_to_goal)
        if truncated:
            info["TimeLimit.truncated"] = True
        if terminated or truncated:
            info["episode_length"] = self.step_counter + 1
        return info

    def _compute_deprojected_point_mask(self, view_matrix, proj_matrix_tuple):
        """Projects the 3D desired_goal onto the 2D image plane to create a mask."""
        height = self.camera_sensor.depth_height
        width = self.camera_sensor.depth_width
        point_mask = np.zeros((height, width), dtype=np.float32)

        view_matrix_np = np.array(view_matrix).reshape(4, 4, order='F')
        proj_matrix_np = np.array(proj_matrix_tuple).reshape(4, 4, order='F')
        goal_pos_world = np.array([*self.desired_goal, 1])
        clip_space_pos = proj_matrix_np @ view_matrix_np @ goal_pos_world

        if clip_space_pos[3] != 0:
            ndc_pos = clip_space_pos[:3] / clip_space_pos[3]
        else:
            return point_mask

        if -1 <= ndc_pos[0] <= 1 and -1 <= ndc_pos[1] <= 1 and -1 <= ndc_pos[2] <= 1:
            log.debug(f"Goal is ON-SCREEN. Drawing circle.")
            pixel_x = int(((ndc_pos[0] + 1) / 2) * width)
            pixel_y = int(((1 - ndc_pos[1]) / 2) * height)

            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                cv2.circle(point_mask, (pixel_x, pixel_y), radius=15, color=(1.0,), thickness=-1)
        elif -1 <= ndc_pos[0] <= 1 and -1 <= ndc_pos[1] <= 1:
            log.debug("Goal projection has invalid depth (ndc_z=%s). Mask will be black.", round(float(ndc_pos[2]), 5))
        else:
            # Use the logger to print when the goal is off screen
            log.debug(f"Goal is OFF-SCREEN. Mask will be black.")
            
        return point_mask
    
    def _get_obs(self):
        """
        Gathers all sensor data and state information to construct the observation dictionary.
        This version corrects the handling of previous RGB images to avoid a double-transpose error.
        """
        # 1. Get current robot state and camera info ---
        ee_pos, ee_or_quat = self.robot.get_current_pose(self.robot.tool0_link_idx)
        ee_vel, ee_ang_vel = self.robot.get_current_vel(self.robot.tool0_link_idx)
        joint_angles = self.robot.get_joint_angles()
        view_matrix = self.robot.get_view_mat_at_curr_pose(self.camera_sensor)

        # 2. Handle RGB images correctly ---
        # Get the NEW RGB image from the camera, which is in (H, W, C) format.
        # Transpose it to the (C, H, W) format required by the policy.
        rgb_hwc, _ = self.robot.get_rgbd_at_cur_pose(camera=self.camera_sensor, type="sensor", view_matrix=view_matrix)
        current_rgb_chw = np.transpose(np.array(rgb_hwc * 255, dtype=np.uint8), (2, 0, 1))

        # Get the PREVIOUS RGB image. It's stored in self.observation['rgb'] from the last step
        # and is *already* in the correct (C, H, W) format.
        # If it's the first step, create a zero-array with the correct (C, H, W) shape.
        cam_height, cam_width = self.camera_sensor.depth_height, self.camera_sensor.depth_width
        previous_rgb_chw = self.observation.get('rgb', np.zeros((3, cam_height, cam_width), dtype=np.uint8))

        # 3. Process other data for reward calculation and observation ---
        # The `prev_observation_info` is needed for calculating rewards based on the previous state.
        self.prev_observation_info = copy.deepcopy(self.observation_info)
        self.observation_info['achieved_pos'] = np.array(ee_pos, dtype=np.float32)
        self.observation_info['desired_pos'] = np.array(self.desired_goal, dtype=np.float32)

        # Convert orientation to 6D representation
        achieved_or_mat = np.array(self.pb_client.getMatrixFromQuaternion(ee_or_quat)).reshape(3, 3)
        achieved_or_6d = achieved_or_mat[:, :2].reshape(6,).astype(np.float32)

        # Encode joint angles with sin/cos
        encoded_joint_angles = np.hstack((np.sin(joint_angles), np.cos(joint_angles))).astype(np.float32)

        # Get achieved action in local frame
        achieved_action_global = np.hstack((ee_vel, ee_ang_vel))
        achieved_action_local = convert_global_action_to_local(self.robot, achieved_action_global)

        # Get goal relative to tool
        t_tw, r_tw = self.pb_client.invertTransform(ee_pos, ee_or_quat)
        goal_in_tool_frame, _ = self.pb_client.multiplyTransforms(t_tw, r_tw, self.desired_goal, [0, 0, 0, 1])

        # 4. Construct the final observation dictionary ---
        self.observation = {
            'achieved_goal': (np.array(ee_pos, dtype=np.float32) - self.init_pos_ee).astype(np.float32),
            'desired_goal': (np.array(self.desired_goal, dtype=np.float32) - self.init_pos_ee).astype(np.float32),
            'achieved_or': achieved_or_6d,
            'joint_angles': encoded_joint_angles,
            'relative_distance': np.array(goal_in_tool_frame, dtype=np.float32),
            'prev_action_achieved': achieved_action_local.astype(np.float32),
            'rgb': current_rgb_chw,
            'prev_rgb': previous_rgb_chw,
            'point_mask': self._compute_deprojected_point_mask(view_matrix, self.camera_sensor.depth_proj_mat),
        }
        return self.observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_env_variables()
        self.robot.set_joint_angles_no_collision(self.home_joint_angles)

        # ADD THIS LINE to settle the physics state after teleporting the robot.
        self.pb_client.stepSimulation()
        
        if options and 'target_apple' in options:
            self.desired_goal = options['target_apple']
        else:
            if not self.apple_centroids:
                raise ValueError("No apple centroids found in the tree to set as a goal.")
            self.desired_goal = random.choice(self.apple_centroids)

        # Set initial pose for relative goal calculations
        self.init_pos_ee, self.init_or_ee = self.robot.get_current_pose(self.robot.tool0_link_idx)

        observation = self._get_obs()
        info = {}
        return observation, info


    def step(self, action, f =0):
        """Executes one time step within the environment."""
        self.action = action
        log.debug(f"--- Step {self.step_counter} ---")
        #log.debug(f"  Raw Action    (local): {np.round(action, 3)}")

        # 1. Scale and convert action from local to global frame
        scaled_action = action * self.action_scale
        #log.debug(f"  Scaled Action (local): {np.round(scaled_action, 3)}")
        
        global_action = convert_local_action_to_global(self.robot, scaled_action)
        #log.debug(f"  Global Action (world): {np.round(global_action, 3)}")

        # 2. Calculate joint velocities and step simulation
        #joint_velocities, _ = self.robot.calculate_joint_velocities_from_ee_velocity_dls(global_action, damping_factor = 0.05)
        
        if f == 2:
            log.debug("  CHANGING ORDER ")
            #global_action[0], global_action[1] = global_action[1], global_action[0]
            
            global_action[0], global_action[1], global_action[3] = -global_action[0], -global_action[1], global_action[3]

            joint_velocities, _ = self.robot.calculate_joint_velocities_from_ee_velocity(global_action)
            # sign_correction = np.array([-1, 1, -1, 1, -1, 1])
            # log.debug(f"  Applying Sign Correction Mask: {sign_correction}")
            # joint_velocities = joint_velocities * sign_correction

        else:
            joint_velocities, _ = self.robot.calculate_joint_velocities_from_ee_velocity_dls(global_action)
        
        #log.debug(f"  Target Joint Velocities: {np.round(joint_velocities, 3)}")
        
        self.robot.set_joint_velocities(joint_velocities)

        for _ in range(self.num_control_simulation_steps):
            self.pb_client.stepSimulation()
            #time.sleep(0.01)

        #  if self.config['simulation_setup']['renders']:
        #      time.sleep(0.005)

        # 3. Get new observation
        observation = self._get_obs()

        # 4. Compute reward and check for termination
        reward, reward_info = self._compute_reward()
        _debug_log_collision_line(self.step_counter, reward_info)
        self.sum_reward += reward

        terminated, terminate_info = self._is_task_done()
        truncated = self.step_counter >= self.max_steps

        # Log: Display reward and termination details
        log.debug("  Reward Info: %s", reward_info)

        # time.sleep(0.3)  # Allow some time for the simulation to settle

        info = self._build_step_info(reward_info, terminate_info, terminated, truncated)

        self.step_counter += 1
        return observation, reward, terminated, truncated, info

    def step_joint_delta(self, joint_delta_rad: np.ndarray):
        """Apply one control timestep using joint velocities (joint-space execution).

        Interprets ``joint_delta_rad`` as the intended change in joint coordinates
        over ``control_time`` (same horizon as :meth:`step`). Internally applies
        ``joint_velocity = joint_delta_rad / control_time``, runs the same number
        of physics substeps as :meth:`step`, then uses the same reward and
        termination logic.

        This is the preferred control path for data generation when the policy
        should track planned joint waypoints instead of Cartesian EE actions.

        Parameters
        ----------
        joint_delta_rad :
            Length ``len(robot.control_joints)`` vector (radians per control step).

        Returns
        -------
        Same tuple as :meth:`step`: ``observation, reward, terminated, truncated, info``.
        """
        joint_delta_rad = np.asarray(joint_delta_rad, dtype=np.float64).reshape(-1)
        n_ctrl = len(self.robot.control_joints)
        if joint_delta_rad.size != n_ctrl:
            raise ValueError(
                f"step_joint_delta: expected {n_ctrl} joint deltas, got {joint_delta_rad.size}"
            )

        self.action = joint_delta_rad.astype(np.float32)
        log.debug(
            "--- Joint step %s  Δq=%s ---",
            self.step_counter,
            np.round(joint_delta_rad, 4).tolist(),
        )

        dt = max(float(self.control_time), 1e-9)
        joint_velocities = (joint_delta_rad / dt).tolist()
        self.robot.set_joint_velocities(joint_velocities)

        for _ in range(self.num_control_simulation_steps):
            self.pb_client.stepSimulation()

        observation = self._get_obs()

        reward, reward_info = self._compute_reward()
        _debug_log_collision_line(self.step_counter, reward_info)
        self.sum_reward += reward

        terminated, terminate_info = self._is_task_done()
        truncated = self.step_counter >= self.max_steps

        log.debug("  Reward Info: %s", reward_info)

        info = self._build_step_info(reward_info, terminate_info, terminated, truncated)
        self.step_counter += 1
        return observation, reward, terminated, truncated, info

    def _planning_collision_objects_and_labels(self):
        """Match feature_apple_path_planning planning_config: obstacle ids + acceptable/unacceptable lists."""
        pl = self.config['planning']
        collision_objects = {label: self.tree.pyb_id for label in pl['collision_labels']}
        acceptable = list(pl.get('acceptable_collision_labels', []))
        unacceptable = pl.get('unacceptable_collision_labels')
        if unacceptable is None:
            unacceptable = list(pl.get('collision_labels', [])) + [
                'UNKNOWN_KDTREE_UNAVAILABLE',
                'UNKNOWN_KDTREE_QUERY_ERROR',
                'UNKNOWN_TOO_FAR',
                'UNKNOWN_NO_CLOSE_VERTEX',
            ]
        else:
            unacceptable = list(unacceptable)
        return collision_objects, acceptable, unacceptable

    def _check_collisions_with_planning_config(self):
        collision_objects, acceptable, unacceptable = self._planning_collision_objects_and_labels()
        return self.robot.check_collisions(
            collision_objects,
            tree_object_for_labeling=self.tree,
            acceptable_labels=acceptable,
            unacceptable_labels=unacceptable,
        )

    def _compute_reward(self):
        """Computes the reward for the current state."""
        reward = 0.0
        reward_info = {}

        # Distance reward
        prev_pos = self.prev_observation_info.get('achieved_pos', self.init_pos_ee)
        current_pos = self.observation_info['achieved_pos']
        desired_pos = self.reward_goal #self.observation_info['desired_pos']
     #   log.debug(f"  Desired_pos: {desired_pos}")

        dist_to_goal = np.linalg.norm(current_pos - desired_pos)
        prev_dist_to_goal = np.linalg.norm(prev_pos - desired_pos)
        
        distance_reward = (prev_dist_to_goal - dist_to_goal) * 100
        reward += distance_reward
        reward_info['distance_reward'] = distance_reward

        # Collision penalty (same rules as path planner: planning unacceptable + UNKNOWN_*)
        collision_unacceptable, collision_details = self._check_collisions_with_planning_config()

        reward_info["collisions_acceptable"] = bool(
            collision_details.get("collisions_acceptable", False)
        )
        reward_info["collisions_unacceptable"] = bool(
            collision_details.get("collisions_unacceptable", False)
        )

        collision_penalty = 0.0
        if collision_unacceptable:
            collision_penalty = -1.0
        
        reward += collision_penalty
        reward_info['collision_penalty'] = collision_penalty
        reward_info['self_collision_unacceptable'] = bool(
            collision_details.get('is_self_collision_unacceptable', False)
        )

        # Success reward
        success_reward = 0.0
        if dist_to_goal < self.distance_threshold:
            self.is_goal_state = True
            success_reward = 1.0
        reward += success_reward
        reward_info['success_reward'] = success_reward

        # Slack/time penalty
        slack_reward = -0.01
        reward += slack_reward
        reward_info['slack_reward'] = slack_reward
        reward_info['dist_to_goal'] = float(dist_to_goal)

        return reward, reward_info

    def _is_task_done(self):
        """Checks if the episode should terminate."""
        terminated = False
        terminate_info = {
            'goal_achieved': False,
            'collision_terminated': False,
            'is_success': False,
        }

        # Check for success
        if self.is_goal_state:
            terminated = True
            terminate_info['goal_achieved'] = True
            terminate_info['is_success'] = True
            log.info("[train:episode] Termination: Goal Reached!")

        # Check for collision (aligned with planner acceptable/unacceptable lists)
        collision_unacceptable, collision_details = self._check_collisions_with_planning_config()
        if collision_unacceptable:
            skip_for_self_only = (
                not self.config.get("simulation_setup", {}).get(
                    "terminate_on_self_collision", True
                )
                and collision_details.get("is_self_collision_unacceptable", False)
                and collision_details.get("collided_obstacle_label") is None
            )
            if not skip_for_self_only:
                terminated = True
                terminate_info['collision_terminated'] = True
                log.warning("[train:episode] Termination: Unacceptable Collision!")

        return terminated, terminate_info

    def reconfigure_scene(self, metadata: dict):
        """
        Resets the simulation and reconfigures the robot and tree to match
        the provided metadata from an expert trajectory.
        """
        sim_setup = self.config["simulation_setup"]
        self.pbutils.reset_simulation_scene(
            gravity=sim_setup["gravity"],
            room_collision=bool(sim_setup.get("room_collision", False)),
        )

        robot_conf = self.config['robot_setup']
        robot_quat = Rotation.from_euler(
            'xyz', robot_conf['start_orientation_euler_deg'], degrees=True
        ).as_quat()
        self.robot = Robot(
            pbclient=self.pb_client,
            position=robot_conf['start_position'],
            orientation=robot_quat,
        )

        meta_q = metadata.get('initial_joint_angles')
        if meta_q is not None:
            q = np.asarray(meta_q, dtype=np.float64).reshape(-1).tolist()
            self.robot.set_joint_angles_no_collision(q)
        else:
            self.robot.set_joint_angles_no_collision(self.home_joint_angles)

        tree_scale = float(np.asarray(metadata['tree_scale'], dtype=np.float64).reshape(-1)[0])
        tree_pos = np.asarray(metadata['tree_pos'], dtype=np.float64).reshape(3)
        tree_or = np.asarray(metadata['tree_orientation'], dtype=np.float64).reshape(4)

        self.tree = Tree(
            pbutils=self.pbutils,
            tree_id=self.config['tree_setup']['tree_id'],
            tree_type=self.config['tree_setup']['tree_type'],
            namespace=self.config['tree_setup']['tree_namespace'],
            scale=tree_scale,
            position=tree_pos,
            orientation=tree_or,
        )
        pickle_pos = np.asarray(self.tree.pos, dtype=np.float64).copy()
        pickle_or = np.asarray(self.tree.orientation, dtype=np.float64).copy()
        if (
            np.linalg.norm(pickle_pos - tree_pos) > 1e-5
            or np.max(np.abs(pickle_or - tree_or)) > 1e-5
        ):
            self.tree.rigid_rebase_vertex_points_to_pose(
                pickle_pos, pickle_or, tree_pos, tree_or
            )
            self.tree.pos = tree_pos
            self.tree.orientation = tree_or

        tree_conf = self.config["tree_setup"]
        load_tree_pybullet_body(
            self.pb_client,
            self.tree,
            tree_conf,
            basePosition=self.tree.pos.tolist(),
            baseOrientation=self.tree.orientation.tolist(),
            globalScaling=self.tree.scale,
        )
        self.tree.build_collision_kdtree()
        self.desired_goal = np.asarray(metadata['goal_pos'], dtype=np.float64).reshape(3)
        self.reward_goal = self.desired_goal.copy()
        self.apple_centroids = self.tree.get_apple_centroids()
        self._refresh_apple_debug_markers()
        if self.pbutils.renders:
            frame_debug_camera_on_tree(
                self.pb_client,
                self.tree.pos,
                self.config["simulation_setup"],
            )

    def close(self):
        self.pb_client.disconnect()
