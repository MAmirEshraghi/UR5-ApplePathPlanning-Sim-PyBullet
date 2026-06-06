"""
main.py

Authors: Robin Eshraghi (@MAmirEshraghi)

This project is part of my internship under the supervision of Prof. Cindy Grimm in the Robotics Lab at the Oregon State University.

Description:
This script implements the Apple Path Planning experiment for the
pybullet-tree-sim environment. It performs:

1. Collision-free path planning from the robot's start position
   to a pre-grasp position near the apple using RRT-Connect.
2. Trajectory smoothing and conversion to velocity commands.
3. Generation of trajectory datasets for downstream RL training.

Folder Structure Expected:
- utils/ : helper functions for planning and trajectory generation
- pybullet_tree_sim/ : core simulation environment (provided by team)
- feature_apple_path_planning/ : this experiment's scripts

Usage:
- Run this script to generate data for RL training or visualize
  the robot's approach path in the simulation.
- Ensure PyBullet and required dependencies are installed.

Date Started: 01/04/2025

Date Completed: 01/08/2025


"""

#!/usr/bin/env python3
import logging
import h5py
import numpy as np
import time
import pybullet_planning as pp
from pybullet_tree_sim import robot
from zenlog import log
import argparse
from filelock import FileLock
import json
import sys
from pathlib import Path

# Only repo root on sys.path so ``apple_picking_env`` stays under the package (relative imports work).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from feature_apple_path_planning.apple_picking_env import (
    ApplePickingEnv,
    resolve_preserve_export_mtl,
)
from feature_apple_path_planning.viz_debug import (
    apple_marker_surface_position,
    change_sphere_color,
    draw_debug_sphere,
    gui_refresh,
    log_apple_marker_batch,
)
from feature_apple_path_planning.goal_camera_filter import (
    GoalCameraFilterDebugContext,
    evaluate_goal_camera_filter_method_a,
)
from feature_apple_path_planning.utils_conversions import (
    convert_local_action_to_global,
    convert_global_action_to_local,
)


import pybullet
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import get_sample_fn, get_extend_fn, get_distance_fn, get_difference_fn
from pybullet_planning.motion_planners.smoothing import smooth_path
from scipy.spatial.transform import Rotation
import os
import pickle
import datetime
import cv2
import random
from enum import Enum
from dataclasses import dataclass, field

import copy
import torch as th

from feature_apple_agent.apple_picking_sb3.pruning_gym.optical_flow import OpticalFlow

try:
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    from pybullet_tree_sim.robot import Robot
    from pybullet_tree_sim.tree import Tree
except ImportError as e:
    log.error(f"Failed to import necessary custom modules: {e}")
    log.error("Please ensure pybullet_tree_sim package is installed and accessible in your PYTHONPATH.")
    exit()

# Configuration 
CONFIG = {
    'simulation_setup': {
        'renders': True,
        'gravity': -9.81,
        'control_time': 1.0 / 20.0,
        'max_steps': 600,
        # GUI: frame tree like run_sim_tree_viewer.py (PyBUtils defaults target the robot)
        'frame_tree_camera': True,
        'debug_camera_distance': 0.8,
        'debug_camera_yaw': 55,
        'debug_camera_pitch': -22,
        'debug_camera_target_z_offset': 1.0,
        # Env red spheres at apple centroids; planner also draws goal markers separately
        'draw_apple_centroid_markers': True,
        # stepSimulation after creating debug bodies so the GUI updates
        'gui_refresh_steps': 1,
        # Backdrop boxes are visual-only unless True (planner ignores walls anyway)
        'room_collision': False,
    },
    'robot_setup': {
        'start_position': [0, 1.3, 0],
        'start_orientation_euler_deg': [0, 0, 180],
        'start_joint_angles': [-1.978, -1.51, 2.622, -1.896, 0.579, 0.848],
        # None = use pybullet_tree_sim/urdf/tmp/robot.urdf if present (no ROS ament_index_python)
        'regenerate_urdf': None,
    },
    'tree_setup': {
        'tree_id': 2,
        'tree_type': "envy",
        'tree_namespace': "LPy",
        'scale': 0.6,
        'position': np.array([0.5, 0, 0]),
        'orientation': np.array([0, 0, 0, 1]),
        # v2 export: multi-material OBJ/MTL + parts URDF (see run_sim_tree_viewer.py)
        'apply_texture': True,
        'preserve_export_mtl': None,  # None = auto when *_textured.mtl exists
        'regenerate_urdf': False,
        # False: show full-tree visual on base link (foliage); True: parts-only (thin leaves vanish)
        'hide_collision_visual': True,
    },
    'planning': {
        'max_ee_velocity': 0.25,
        'rrt_max_iterations': 3000,
        'rrt_max_time': 180.0,
        'planner_type': 'rrt_star', #'rrt_star', #'rrt_connect',
        'rrt_radius': 0.4,
        'num_goal_candidates_to_try': 3,
        'goal_position_offset': 0.25,
        # Single source for (1) IK FK acceptance, (2) ApplePickingEnv success radius (distance_threshold),
        # (3) HDF5 attr success_pos_tolerance_m. Keep one value unless you intentionally split IK vs success.
        'fk_pos_tolerance': 0.06,  # relaxed from 0.05 — physics sim drifts ~6 mm short in rollout
        'fk_orn_tolerance': 0.08,  # paired with success_orn_tolerance_rad in HDF5 metadata
        'max_orientation_roll_perturbation_rad': np.pi/4,
        'max_orientation_pitch_yaw_perturbation_rad': np.pi/6,
        'collision_labels': ["TRUNK", "BRANCH", "APPLE", "LEAF", "SPUR", "STEM", "WATER_BRANCH"],
        'acceptable_collision_labels': [],
        'unacceptable_collision_labels': ["TRUNK", "BRANCH", "APPLE", "SPUR", "LEAF", "STEM", "WATER_BRANCH", "UNKNOWN_KDTREE_UNAVAILABLE", "UNKNOWN_KDTREE_QUERY_ERROR", "UNKNOWN_TOO_FAR", "UNKNOWN_NO_CLOSE_VERTEX"],
        'orientation_seed': 3,
        'visibility_seed': 3,
        'enable_smoothing': True,
        'task_space_refinement_threshold': 0.012,
        # Pick best RRT candidate: joint_length + w_smooth*smoothness + w_task*task_regression (see below).
        'path_selection_smoothness_weight': 0.1,
        # Weight on sum of positive Δ||EE−goal|| along raw path (meters); 0 disables (legacy behavior).
        'path_selection_task_regression_weight': 0.15,
        # Run shortcut + task-space refine before writing waypoints to HDF5 (matches generator).
        'save_refined_waypoints_to_hdf5': True,
        'visibility_check': {
            'enable': False,
            'debug_view': True,
            'num_targets': 0,
            'min_visible_pixels': 5,
            'target_color_rgba': [1.0, 0.0, 0.0, 0.8],#[0.1, 1.0, 0.1, 0.9],
            'target_color_rgb': (250, 0, 0), #(25, 255, 25),
            # Add support for multiple target colors (red for goals, blue for active apples)
            'target_colors_rgb': [(250, 0, 0), (0, 0, 240)],  # Red and Blue
            'color_tolerance': (40, 40, 40), #(80,80,80),
            'target_sphere_radius': 0.04 #0.025
        },
        # Pre-RRT only: at q_goal, require enough "apple blue" inside projected disk (Method A).
        # Independent of visibility_check in is_state_valid_fn.
        'goal_camera_filter': {
            'enable': False,
            'min_visible_frac': 0.65,
            # RGB image order matches robot.get_rgbd (RGB not BGR).
            'color_rgb_lower': (0, 0, 140),
            'color_rgb_upper': (100, 100, 255),
            'sphere_radius_m': 0.1, #0.05,
            'disk_radius_scale': 1.15,
            'sim_settle_steps': 1,
            # Extra GOAL_CAM_FILTER fields + log.debug lines (NDC, eye, apple, clip_w).
            'log_projection_debug': True,
            # When True, save s01/s02/s03 PNGs under logs/goal_camera_filter/<run_id>/.
            'save_debug_images': True,
            'debug_subdir': 'goal_camera_filter',
        },
    },
    'generator': {
        'action_scale': 1,
    },
    'visualization': {
        'draw_goal_spheres': True,
        'goal_sphere_radius': 0.035,
        # Push marker spheres outside apple mesh toward robot (meters)
        'apple_marker_surface_offset_m': 0.0,
        'centroid_marker_color': [0.9, 0.08, 0.08, 0.9],
        'goal_sphere_color': [1.0, 0.0, 0.0, 0.8],
        'active_apple_color': [0.0, 0.0, 1.0, 0.9],
        'draw_goal_frames': True,
        'goal_frame_length': 0.09,
        'path_visualization_delay': 0.1,
        'delay_between_apple_attempts': 0.01,
        'draw_collision_item_aabbs': True,
        'collision_item_aabb_color': [0.0, 0.0, 1.0, 0.5],
        'visualize_actual_collision_points': True,
        'collision_point_color_env': [1.0, 0.5, 0.0, 0.9],
        'collision_point_color_self': [1.0, 0.0, 1.0, 0.9],
        'collision_point_radius': 0.01
    },
    'output': {
        'base_dir': "output",
        'waypoints_subdir': "waypoints",
        'agent_data_subdir': "agent_data",
        'waypoints_hdf5_path': None,
        'agent_data_hdf5_path': None
    }
}

DEFAULT_OPTICAL_FLOW_SIZE = (224, 224)


@dataclass
class PlannerStats:
    total_apples: int = 0
    success_count: int = 0
    failure_count: int = 0
    apples: list = field(default_factory=list)
    
    visibility_rejections: int = 0
    goal_camera_filter_rejections: int = 0
    collision_env_rejections: int = 0
    collision_self_rejections: int = 0

    def record_apple(self, index, success, ik_attempts, plan_time_sec, waypoint_count,
                     joint_path_length=None, smoothness_cost=None, failure_reason=None):
        self.apples.append({
            'index': index,
            'success': success,
            'ik_attempts': ik_attempts,
            'plan_time_sec': plan_time_sec,
            'waypoints': waypoint_count,
            'joint_path_length': joint_path_length,
            'smoothness_cost': smoothness_cost,
            'failure_reason': failure_reason,
        })
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def report_lines(self):
        success_rate = (self.success_count / self.total_apples) if self.total_apples else 0.0
        lines = [
            "=" * 60,
            f"Planner Summary @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Apples processed : {self.total_apples}",
            f"Successes        : {self.success_count}",
            f"Failures         : {self.failure_count}",
            f"Success rate     : {success_rate:.2%}",
            f"Visibility rejects : {self.visibility_rejections}",
            f"Goal-cam (pre-RRT) rejects : {self.goal_camera_filter_rejections}",
            f"Env collision rejects : {self.collision_env_rejections}",
            f"Self collision rejects: {self.collision_self_rejections}",
            "-" * 60,
        ]
        for apple in self.apples:
            result = "SUCCESS" if apple['success'] else "FAIL"
            waypoints = apple['waypoints'] if apple['waypoints'] is not None else "N/A"
            path_len = apple.get('joint_path_length', "N/A")
            smooth = apple.get('smoothness_cost', "N/A")
            if isinstance(path_len, float):
                path_len = f"{path_len:.3f}"
            if isinstance(smooth, float):
                smooth = f"{smooth:.3f}"
            failure_reason = apple.get('failure_reason', "N/A") if not apple['success'] else ""
            line = (
                f"Apple #{apple['index']:02d} | {result:<7} | "
                f"IK attempts: {apple['ik_attempts']:>2} | "
                f"Waypoints: {waypoints} | "
                f"PathLen: {path_len} | "
                f"Smooth: {smooth} | "
                f"Plan time: {apple['plan_time_sec']:.2f}s"
            )
            if failure_reason:
                line += f" | Reason: {failure_reason}"
            lines.append(line)
        lines.append("=" * 60)
        return lines


def ensure_optical_flow(obs: dict,
                        optical_flow_model: OpticalFlow = None,
                        default_size: tuple = DEFAULT_OPTICAL_FLOW_SIZE) -> dict:
    """
    Ensure that each observation dictionary carries an 'optical_flow' entry so it
    can be saved for expert training. If an optical flow model is provided we run
    it, otherwise we fall back to zeros so downstream code still finds the key.
    """
    if obs is None:
        return None

    if obs.get('optical_flow') is not None:
        return obs

    target_size = getattr(optical_flow_model, "size", default_size)
    zero_flow = np.zeros((2, target_size[0], target_size[1]), dtype=np.float32)

    if optical_flow_model is None:
        obs['optical_flow'] = zero_flow
        return obs

    try:
        current_rgb = th.tensor(obs['rgb'], dtype=th.float32)
        prev_rgb = th.tensor(obs['prev_rgb'], dtype=th.float32)
        flow_tensor = optical_flow_model.calculate_optical_flow(current_rgb, prev_rgb)
        obs['optical_flow'] = flow_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as exc:
        log.error(f"Failed to compute optical flow for observation: {exc}", exc_info=True)
        obs['optical_flow'] = zero_flow

    return obs


def seed_random_generators(seed_value, label="global"):
    if seed_value is None:
        return
    np.random.seed(seed_value)
    random.seed(seed_value)
    log.info(f"Seeded numpy/random for {label} with seed={seed_value}")

class ResultMode(Enum):
    NO_SOLUTION = 0
    SUCCESS = 1
    NO_PATH = 2


def generate_apple_orientation_quat(apple_center_pos: np.ndarray, 
                                    max_roll_rad: float, 
                                    max_pitch_yaw_rad: float,
                                    approach_axis_world: np.ndarray = np.array([0., 1., 0.]),
                                    gripper_up_axis_world: np.ndarray = np.array([0., 0., 1.])) -> np.ndarray:
    
    eef_z_axis_desired = -np.array(approach_axis_world)
    eef_z_axis_desired /= np.linalg.norm(eef_z_axis_desired)
    eef_x_axis_desired = np.cross(gripper_up_axis_world, eef_z_axis_desired)
    if np.linalg.norm(eef_x_axis_desired) < 1e-3:
        alternative_up = np.array([1., 0., 0.])
        if np.allclose(gripper_up_axis_world, alternative_up) or np.allclose(eef_z_axis_desired, alternative_up) or np.allclose(eef_z_axis_desired, -alternative_up):
            alternative_up = np.array([0., 1., 0.])
        eef_x_axis_desired = np.cross(alternative_up, eef_z_axis_desired)
        if np.linalg.norm(eef_x_axis_desired) < 1e-3:
            eef_x_axis_desired = np.array([0., 1., 0.] if not np.allclose(eef_z_axis_desired, [0, 1, 0]) else [1, 0, 0])
    eef_x_axis_desired /= np.linalg.norm(eef_x_axis_desired)
    eef_y_axis_desired = np.cross(eef_z_axis_desired, eef_x_axis_desired)
    base_rot_matrix = np.column_stack((eef_x_axis_desired, eef_y_axis_desired, eef_z_axis_desired))
    try:
        base_orientation = Rotation.from_matrix(base_rot_matrix)
    except ValueError:
        base_orientation = Rotation.identity()
        if np.allclose(eef_z_axis_desired, [0, 0, -1]):
            base_orientation = Rotation.from_euler('y', np.pi)
        elif np.allclose(eef_z_axis_desired, [0, 0, 1]):
            base_orientation = Rotation.identity()
    roll_perturb, pitch_perturb, yaw_perturb = np.random.uniform(-max_roll_rad, max_roll_rad), np.random.uniform(-max_pitch_yaw_rad, max_pitch_yaw_rad), np.random.uniform(-max_pitch_yaw_rad, max_pitch_yaw_rad)
    perturb_rot = Rotation.from_euler('xyz', [roll_perturb, pitch_perturb, yaw_perturb], degrees=False)
    final_orientation = base_orientation * perturb_rot
    return final_orientation.as_quat()

def get_ik_solutions_for_apple(robot: Robot, 
                               is_state_valid_fn, 
                               apple_center: np.ndarray, robot_ref_pos_for_orient: np.ndarray,
                               config_planning: dict, 
                               pb_client, 
                               initial_robot_config: np.ndarray):
    
    #num_attempts, offset_dist, fk_pos_tol, fk_orn_tol, max_roll, max_pitch_yaw = config_planning['num_goal_candidates_to_try'], config_planning['goal_position_offset'], config_planning['fk_pos_tolerance'], config_planning['fk_orn_tolerance'], config_planning['max_orientation_roll_perturbation_rad'], config_planning['max_orientation_pitch_yaw_perturbation_rad']
    num_attempts = config_planning['num_goal_candidates_to_try']
    offset_dist = config_planning['goal_position_offset']
    fk_pos_tol = config_planning['fk_pos_tolerance']
    fk_orn_tol = config_planning['fk_orn_tolerance']
    max_roll = config_planning['max_orientation_roll_perturbation_rad']
    max_pitch_yaw = config_planning['max_orientation_pitch_yaw_perturbation_rad']

    for attempt_idx in range(num_attempts):
        log.debug(f"    IK Attempt #{attempt_idx + 1}/{num_attempts} for current apple.")
        target_orn_quat = generate_apple_orientation_quat(apple_center, 
                                                          max_roll, 
                                                          max_pitch_yaw, 
                                                          approach_axis_world=np.array([0., 1., 0.]))
        
        rot_matrix_for_offset = Rotation.from_quat(target_orn_quat).as_matrix()
        eef_z_axis_world = rot_matrix_for_offset[:, 2]
        goal_pos_for_ik = np.array(apple_center) - offset_dist * eef_z_axis_world
        if CONFIG['visualization']['draw_goal_frames'] and attempt_idx < 40:
            draw_debug_frame(pb_client, goal_pos_for_ik, target_orn_quat, length=CONFIG['visualization']['goal_frame_length'])
        q_goal_candidate_tuple = robot.calculate_ik(goal_pos_for_ik, target_orn_quat)
        if q_goal_candidate_tuple is None:
            log.debug(f"      IK Attempt #{attempt_idx+1}: IK Calculation FAILED.")
            continue
        q_goal_candidate = np.array(q_goal_candidate_tuple)
        log.debug(f"      IK Attempt #{attempt_idx+1}: IK SUCCEEDED. q_goal_candidate: {np.round(q_goal_candidate, 3)}")
        robot.set_joint_angles_no_collision(q_goal_candidate)
        actual_pos, actual_orn_quat = robot.get_current_pose(robot.tool0_link_idx)
        robot.set_joint_angles_no_collision(initial_robot_config)
        pos_error = np.linalg.norm(np.array(actual_pos) - goal_pos_for_ik)
        orientation_error_angle_rad = (Rotation.from_quat(actual_orn_quat).inv() * Rotation.from_quat(target_orn_quat)).magnitude()
        pos_ok, orn_ok = pos_error < fk_pos_tol, orientation_error_angle_rad < fk_orn_tol
        log.debug(f"      IK Attempt #{attempt_idx+1}: FK Check. PosErr: {pos_error:.4f} (OK:{pos_ok}), OrnErr: {orientation_error_angle_rad:.4f} (OK:{orn_ok})")
        if not (pos_ok and orn_ok):
            log.debug(f"      IK Attempt #{attempt_idx+1}: FK Check FAILED.")
            continue
        log.debug(f"      IK Attempt #{attempt_idx+1}: FK Check PASSED.")
        log.debug(f"      IK Attempt #{attempt_idx+1}: Static Collision Check for q_goal_candidate: {np.round(q_goal_candidate,3)}")
        if is_state_valid_fn(q_goal_candidate):
            log.debug(f"      IK Attempt #{attempt_idx+1}: Static Collision Check FAILED (is_state_valid_fn returned True). q_goal_candidate is in collision.")
            continue
        log.info(f"      IK Attempt #{attempt_idx+1}: Static Collision Check PASSED. Yielding candidate.")
        yield q_goal_candidate, goal_pos_for_ik, target_orn_quat
    yield None, None, None

def draw_debug_frame(pb_client, position, orientation_quat, length=0.1, line_width=2):
    rot_matrix = np.array(pb_client.getMatrixFromQuaternion(orientation_quat)).reshape(3, 3)
    origin = np.array(position)
    x_axis, y_axis, z_axis = origin + rot_matrix @ np.array([length, 0, 0]), origin + rot_matrix @ np.array([0, length, 0]), origin + rot_matrix @ np.array([0, 0, length])
    pb_client.addUserDebugLine(origin, x_axis, [1, 0, 0], lineWidth=line_width)
    pb_client.addUserDebugLine(origin, y_axis, [0, 1, 0], lineWidth=line_width)
    pb_client.addUserDebugLine(origin, z_axis, [0, 0, 1], lineWidth=line_width)

def setup_planning_functions(robot: Robot, 
                             tree_pb_id: int, 
                             planning_config: dict, 
                             visualization_config: dict,
                             pb_client_ref, 
                             tree_object_ref: Tree = None, 
                             camera_sensor=None,
                             metrics: PlannerStats = None):
    
    controllable_joints = robot.control_joint_idxs
    robot_id = robot.robot
    distance_fn = get_distance_fn(robot_id, controllable_joints)
    sample_fn = get_sample_fn(robot_id, controllable_joints)
    #extend_fn = get_extend_fn(robot_id, controllable_joints)
    resolutions = 0.05
    extend_fn = get_extend_fn(robot_id, controllable_joints, resolutions=resolutions)

    obstacle_dict_for_robot_check = {label: tree_pb_id for label in planning_config['collision_labels']}
    acceptable_labels = planning_config.get('acceptable_collision_labels', [])
    unacceptable_labels = planning_config.get('unacceptable_collision_labels', [])
    vis_check_config = planning_config.get('visibility_check', {})

    def is_state_valid_fn(q):
        current_q_robot = robot.get_joint_angles()
        robot.set_joint_angles_no_collision(q)
        
        # Force PyBullet to update collision detection (from old version)
        pb_client_ref.performCollisionDetection()
        
        # 1. --- Enhanced Collision Check with Fallback ---
        is_unacceptable_collision = False
        detailed_col_info = {}
        
        try:
            # Try advanced collision checking first
            if not hasattr(robot, 'check_collisions'):
                raise AttributeError("Robot object does not have 'check_collisions' method.")
                
            is_unacceptable_collision, detailed_col_info = robot.check_collisions(
                obstacle_dict_for_robot_check,
                tree_object_for_labeling=tree_object_ref,
                acceptable_labels=acceptable_labels,
                unacceptable_labels=unacceptable_labels)
            
            # Check if ANY collision occurred (acceptable or unacceptable)
            has_acceptable_collision = detailed_col_info.get("collisions_acceptable", False)
            has_unacceptable_collision = detailed_col_info.get("collisions_unacceptable", False)
            
            # CORRECTED: Only invalidate state for UNACCEPTABLE collisions
            if has_unacceptable_collision:
                col_body = (
                    "self"
                    if detailed_col_info.get("is_self_collision_unacceptable")
                    else "tree"
                )
                log.info(
                    "PLAN_COLLISION reject body=%s label=%s",
                    col_body,
                    detailed_col_info.get("collided_obstacle_label"),
                )
                log.warning("State INVALID: Unacceptable collision detected.")
                if metrics is not None:
                    if detailed_col_info.get('is_self_collision_unacceptable', False):
                        metrics.collision_self_rejections += 1
                    else:
                        metrics.collision_env_rejections += 1
                if visualization_config.get('visualize_actual_collision_points', False):
                    env_contact_point = detailed_col_info.get('contact_point_on_obstacle')
                    collided_label = detailed_col_info.get('collided_obstacle_label')
                    if env_contact_point is not None and collided_label is not None:
                        log.debug(f"Visualizing unacceptable ENV collision with determined label '{collided_label}' at {np.round(env_contact_point, 3)}")
                        draw_debug_sphere(pb_client_ref, env_contact_point, 
                                        visualization_config['collision_point_radius'], 
                                        visualization_config['collision_point_color_env'])
                    self_contact_point = detailed_col_info.get('self_collision_contact_pos')
                    if self_contact_point is not None and detailed_col_info.get('is_self_collision_unacceptable', False):
                        log.debug(f"Visualizing unacceptable SELF collision at {np.round(self_contact_point, 3)}")
                        draw_debug_sphere(pb_client_ref, self_contact_point, 
                                        visualization_config['collision_point_radius'], 
                                        visualization_config['collision_point_color_self'])
                robot.set_joint_angles_no_collision(current_q_robot)
                return True  # State is invalid
            
            # ACCEPTABLE collisions (like APPLE contact) are OK - continue to visibility check
            if has_acceptable_collision:
                log.debug(f"State has acceptable collision with: {detailed_col_info.get('collided_obstacle_label', 'UNKNOWN')}")
                
        except AttributeError as ae:
            # FALLBACK: Use basic PyBullet collision detection (from old version)
            log.warning(f"{ae}. Using generic collision check fallback for environment.")
            
            # Check environment collisions
            contacts_env = pb_client_ref.getContactPoints(bodyA=robot.robot, bodyB=tree_pb_id)
            if contacts_env:
                for contact in contacts_env:
                    if contact[8] < -0.001:  # Penetration threshold from old version
                        is_unacceptable_collision = True
                        log.info("PLAN_COLLISION reject body=tree label=generic_fallback")
                        if visualization_config.get('visualize_actual_collision_points', False):
                            cp = contact[6]  # Contact point on bodyB (tree)
                            log.debug(f"Visualizing generic ENV collision at {np.round(cp, 3)} (due to AttributeError in robot.check_collisions)")
                            draw_debug_sphere(pb_client_ref, cp,
                                            visualization_config['collision_point_radius'],
                                            visualization_config['collision_point_color_env'])
                        break
            
            # Check self-collisions if no environment collision found
            if not is_unacceptable_collision:
                self_contacts = pb_client_ref.getContactPoints(bodyA=robot.robot, bodyB=robot.robot)
                if self_contacts:
                    for sc in self_contacts:
                        if sc[3] != sc[4] and sc[8] < -0.015:  # Different links, deeper penetration threshold
                            is_unacceptable_collision = True
                            if metrics is not None:
                                metrics.collision_self_rejections += 1
                            if visualization_config.get('visualize_actual_collision_points', False):
                                cp_self = sc[5]  # Contact point on bodyA (robot)
                                log.debug(f"Visualizing generic SELF collision at {np.round(cp_self, 3)} (due to AttributeError in robot.check_collisions)")
                                draw_debug_sphere(pb_client_ref, cp_self,
                                                visualization_config['collision_point_radius'],
                                                visualization_config['collision_point_color_self'])
                            break
            
            # If fallback found collision, return invalid
            if is_unacceptable_collision:
                robot.set_joint_angles_no_collision(current_q_robot)
                return True
                
        except Exception as e:
            # Handle any other errors gracefully
            log.error(f"Error during collision check for q={np.round(q, 3)}: {e}", exc_info=True)
            robot.set_joint_angles_no_collision(current_q_robot)
            return True  # Assume invalid state on error
        
        # 2. --- Camera-Based Visibility Check ---
        if vis_check_config.get('enable', False) and camera_sensor is not None:
            try:
                view_matrix = robot.get_view_mat_at_curr_pose(camera_sensor)
                rgb_image_float, _ = robot.get_rgbd_at_cur_pose(camera=camera_sensor, type="sensor", view_matrix=view_matrix)

                rgb_image_uint8 = (np.array(rgb_image_float) * 255).astype(np.uint8)

                # Support multiple target colors (e.g., red for goals, blue for active apples)
                target_colors_rgb = vis_check_config.get('target_colors_rgb', [vis_check_config.get('target_color_rgb', (255, 0, 0))])
                # Backward compatibility: if only single target_color_rgb is provided, convert to list
                if 'target_color_rgb' in vis_check_config and 'target_colors_rgb' not in vis_check_config:
                    target_colors_rgb = [vis_check_config['target_color_rgb']]
                
                tolerance = np.array(vis_check_config['color_tolerance'])
                
                # Create combined mask for all target colors
                combined_mask = np.zeros(rgb_image_uint8.shape[:2], dtype=np.uint8)
                for target_rgb in target_colors_rgb:
                    target_rgb_array = np.array(target_rgb)
                    lower_bound = np.clip(target_rgb_array - tolerance, 0, 255)
                    upper_bound = np.clip(target_rgb_array + tolerance, 0, 255)
                    color_mask = cv2.inRange(rgb_image_uint8, lower_bound, upper_bound)
                    combined_mask = cv2.bitwise_or(combined_mask, color_mask)
                
                visible_pixels = cv2.countNonZero(combined_mask)

         #       log.debug(f"Visibility check: Found {visible_pixels} target pixels.")

                if visible_pixels < vis_check_config.get('min_visible_pixels', 2000):
                    log.warning(f"State INVALID: Visibility check failed. Found {visible_pixels} pixels, need {vis_check_config.get('min_visible_pixels', 2000)}.")
                    if metrics is not None:
                        metrics.visibility_rejections += 1
                    robot.set_joint_angles_no_collision(current_q_robot)
                    return True  # State is invalid due to poor visibility
                    
            except Exception as e:
                log.error(f"Error during visibility check for q={np.round(q, 3)}: {e}", exc_info=True)
                # Continue without visibility check rather than failing completely
                log.warning("Continuing without visibility check due to error.")
        
        # 3. --- If all checks pass, the state is valid ---
        robot.set_joint_angles_no_collision(current_q_robot)
        return False  # State is VALID
    
    # def is_state_valid_fn(q):
    #     current_q_robot = robot.get_joint_angles()
    #     robot.set_joint_angles_no_collision(q)
    #     pb_client_ref.performCollisionDetection()
    #     is_unacceptable_collision = False
    #     try:
    #         if not hasattr(robot, 'check_collisions'):
    #             raise AttributeError("Robot object does not have 'check_collisions' method.")
    #         is_unacceptable_collision, detailed_col_info = robot.check_collisions(obstacle_dict_for_robot_check, tree_object_for_labeling=tree_object_ref)
    #         if is_unacceptable_collision:
    #             log.warning("State INVALID: Unacceptable collision detected.")
    #             if visualization_config.get('visualize_actual_collision_points', False):
    #                 env_contact_point = detailed_col_info.get('contact_point_on_obstacle')
    #                 if env_contact_point is not None:
    #                     draw_debug_sphere(pb_client_ref, env_contact_point, visualization_config['collision_point_radius'], visualization_config['collision_point_color_env'])
    #             robot.set_joint_angles_no_collision(current_q_robot)
    #             return True
    #     except Exception as e:
    #         log.error(f"Error during collision check: {e}", exc_info=True)
    #         robot.set_joint_angles_no_collision(current_q_robot)
    #         return True
    #     if vis_check_config.get('enable', False) and camera_sensor is not None:
    #         try:
    #             view_matrix = robot.get_view_mat_at_curr_pose(camera_sensor)
    #             rgb_image_float, _ = robot.get_rgbd_at_cur_pose(camera=camera_sensor, type="sensor", view_matrix=view_matrix)
    #             rgb_image_uint8 = (np.array(rgb_image_float) * 255).astype(np.uint8)
    #             target_rgb, tolerance = np.array(vis_check_config['target_color_rgb']), np.array(vis_check_config['color_tolerance'])
    #             lower_bound, upper_bound = np.clip(target_rgb - tolerance, 0, 255), np.clip(target_rgb + tolerance, 0, 255)
    #             mask = cv2.inRange(rgb_image_uint8, lower_bound, upper_bound)
    #             visible_pixels = cv2.countNonZero(mask)
    #             if visible_pixels < vis_check_config.get('min_visible_pixels', 2000):
    #                 log.warning(f"State INVALID: Visibility check failed. Found {visible_pixels} pixels.")
    #                 robot.set_joint_angles_no_collision(current_q_robot)
    #                 return True
    #         except Exception as e:
    #             log.error(f"Error during visibility check: {e}", exc_info=True)
    #     robot.set_joint_angles_no_collision(current_q_robot)
    #     return False
    

    return distance_fn, sample_fn, extend_fn, is_state_valid_fn

def visualize_path(robot: Robot, path: list, delay: float, visualization_sensor, pb_client):
    if not path:
        log.warning("Cannot visualize empty path.")
        return
    log.info(f"Visualizing path with {len(path)} waypoints...")
    for q_idx, q_wp in enumerate(path):
        robot.set_joint_angles_no_collision(q_wp)
        if CONFIG['simulation_setup']['renders']:
            time.sleep(delay)
            robot.pbclient.stepSimulation()
        if q_idx == len(path) - 1:
            log.info("Reached the final waypoint in visualize_path.")
            if CONFIG['simulation_setup']['renders']:
                time.sleep(3)
    log.info("Path visualization complete.")


def task_space_distance(robot, q1, q2):
    original_q = robot.get_joint_angles()
    robot.set_joint_angles_no_collision(q1)
    pos1, _ = robot.get_current_pose(robot.tool0_link_idx)
    robot.set_joint_angles_no_collision(q2)
    pos2, _ = robot.get_current_pose(robot.tool0_link_idx)
    robot.set_joint_angles_no_collision(original_q)
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
def get_refine_fn_task(robot, joints, task_threshold):
    difference_fn = get_difference_fn(robot.robot, joints)
    def fn(q1, q2):
        num_steps = int(np.ceil(task_space_distance(robot, q1, q2) / task_threshold))
        if num_steps == 0:
            yield q2
            return
        for i in range(num_steps + 1):
            alpha = i / num_steps
            yield tuple(np.array(q1) + alpha * np.array(difference_fn(q2, q1)))
    return fn
def refine_waypoints_in_task_space(
    robot,
    joints,
    waypoints,
    task_threshold,
    collision_fn=None,
    extend_fn=None,
):
    """Insert intermediate joint poses along each edge (task-space spacing heuristic).

    If ``collision_fn(q)`` is provided, it must match ``is_state_valid_fn`` semantics:
    return **True** if the state is **invalid** (collision / visibility fail).

    Linearly interpolated refinement samples are not guaranteed collision-free even
    when endpoints are valid. If any intermediate sample is invalid, this function
    falls back to ``extend_fn(q1, q2)`` (collision-checked dense joint extension
    from pybullet_planning) for that edge. If the fallback still contains invalid
    states, raises ``RuntimeError``.
    """
    if not waypoints:
        return []
    refine_fn = get_refine_fn_task(robot, joints, task_threshold)
    refined_path = [waypoints[0]]
    for edge_idx, (v1, v2) in enumerate(zip(waypoints, waypoints[1:])):
        candidates = list(refine_fn(v1, v2))[1:]
        use_candidates = candidates
        if collision_fn is not None and candidates:
            invalid_flags = [bool(collision_fn(q)) for q in candidates]
            if any(invalid_flags):
                log.warning(
                    "refine_waypoints_in_task_space: edge %s — %d/%d interpolated states invalid "
                    "(linear task-space refine) — using extend_fn fallback",
                    edge_idx,
                    sum(invalid_flags),
                    len(candidates),
                )
                if extend_fn is None:
                    raise RuntimeError(
                        "Task-space refinement hit invalid interpolated states but extend_fn is None"
                    )
                use_candidates = list(extend_fn(tuple(v1), tuple(v2)))[1:]
                if not use_candidates:
                    use_candidates = [tuple(v2)]
                if any(collision_fn(q) for q in use_candidates):
                    raise RuntimeError(
                        "extend_fn fallback still produced invalid states on edge %s" % edge_idx
                    )
        elif collision_fn is not None and not candidates:
            pass
        refined_path.extend(use_candidates)
    return refined_path

def shortcut_and_refine_path(robot, path, extend_fn, collision_fn, task_threshold, enable_smoothing=True):
    if len(path) < 2: 
        log.warning("Path is too short to refine, returning as is.")
        return path
        
    log.info(f"Refining path. Initial waypoints: {len(path)}")
    
    if enable_smoothing:
        path = smooth_path(path, extend_fn, collision_fn, iterations=20)
        log.info(f"Waypoints after shortcutting: {len(path)}")
    else:
        log.info("Skipping path smoothing per configuration.")
    
    # Task-space refinement (collision-check intermediates; extend_fn fallback on failure)
    path = refine_waypoints_in_task_space(
        robot,
        robot.control_joint_idxs,
        path,
        task_threshold,
        collision_fn=collision_fn,
        extend_fn=extend_fn,
    )
    log.info(f"Waypoints after task-space refinement: {len(path)}")
    
    return path

def convert_ja_to_ee_vel(robot, path, control_freq=2, max_ee_vel=0.7):
    ee_vel_actions = []
    if len(path) < 2: 
        log.warning("Cannot convert path to velocities: path length is less than 2.")
        return []

    log.info(f"Converting {len(path)} waypoints to EE velocity actions...")
    for i in range(len(path) - 1):
        # Calculate required joint velocities

        joint_velocities = (np.array(path[i + 1]) - np.array(path[i])) * control_freq
        log.debug(f"control frequency: {control_freq}")
        #log.debug(f"Step {i}: {np.array(path[i + 1])} - {np.array(path[i])}  = Joint velocities =  {np.round(joint_velocities, 3)}")

        # Set robot to the current waypoint to calculate Jacobian from the correct pose
        robot.set_joint_angles_no_collision(path[i])
        time.sleep(0.02)
        
        robot.pbclient.stepSimulation()
        time.sleep(0.1)

        jacobian = robot.calculate_jacobian()
        
        # Convert joint velocities to global EE velocity
        #global_ee_vel = np.dot(jacobian, joint_velocities)
        global_ee_vel = np.matmul(jacobian, joint_velocities)

   #     log.debug(f"Step {i}: Global EE velocity: {np.round(global_ee_vel, 3)}")
        
        # Convert global EE velocity to the gripper's local frame
        local_ee_vel = convert_global_action_to_local(robot, global_ee_vel)
    #    log.debug(f"Step {i}: Local EE velocity (action)      : {np.round(local_ee_vel, 3)}")
                
        global_ee_vel2 = convert_local_action_to_global(robot, local_ee_vel)
   #     log.debug(f"Step {i}: Global EE velocity (reconverted): {np.round(global_ee_vel2, 3)}")


        # Check if the action exceeds velocity limits and requires sub-steps # if any term in local_ee_vel > 1 or < -1, break the step into smaller steps
        if np.any(np.abs(local_ee_vel) > max_ee_vel * CONFIG['generator']['action_scale']):
            num_steps = int(np.ceil(np.max(np.abs(local_ee_vel / (max_ee_vel * CONFIG['generator']['action_scale'])))))
            log.debug(f"Step {i}: Velocity limit exceeded. Splitting into {num_steps} sub-steps.")
            for _ in range(num_steps):
                ee_vel_actions.append(local_ee_vel / num_steps)
        else:
            ee_vel_actions.append(local_ee_vel)
            
        
        # if np.any(np.abs(local_ee_vel) > max_ee_vel):
        #     num_steps = int(np.ceil(np.max(np.abs(local_ee_vel / max_ee_vel))))
        #     log.debug(f"Step {i}: Velocity limit exceeded. Splitting into {num_steps} sub-steps.")
        #     for _ in range(num_steps):
        #         ee_vel_actions.append(local_ee_vel / num_steps)
        # else:
        #     ee_vel_actions.append(local_ee_vel)
            
    log.info(f"Finished conversion. Produced {len(ee_vel_actions)} total actions.")
    return ee_vel_actions

def visualize_actions_camera(new_obs):
    # =====================VISUALIZATION=============================== #
        # Extract the visual data from the new observation
        rgb_chw = new_obs['rgb']
        prev_rgb_chw = new_obs['prev_rgb']
        mask_hw = new_obs['point_mask']

        # --- Process images for display ---
        # Transpose from (C, H, W) to (H, W, C) and convert from RGB to BGR for OpenCV
        current_rgb_img = cv2.cvtColor(np.transpose(rgb_chw, (1, 2, 0)), cv2.COLOR_RGB2BGR)
        prev_rgb_img = cv2.cvtColor(np.transpose(prev_rgb_chw, (1, 2, 0)), cv2.COLOR_RGB2BGR)

        # Convert the single-channel float mask to a 3-channel BGR image for stacking
        mask_img = (mask_hw * 255).astype(np.uint8)
        mask_img_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

        # Add text labels to each image
        cv2.putText(prev_rgb_img, 'Previous RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(current_rgb_img, 'Current RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(mask_img_bgr, 'Point Mask', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- Combine and display ---
        # Stack the three images horizontally to show in one window
        combined_display = np.hstack([prev_rgb_img, current_rgb_img, mask_img_bgr])
        cv2.imshow("Generator Stage - Visual Data", combined_display)

        # This line is crucial for the window to update. It waits 1ms.
        cv2.waitKey(1)
        # ================================================================= #
        return

def get_transitions_from_ee_vel(env, ee_vel_actions, CONFIG, optical_flow_model: OpticalFlow = None):
    observations = []
    new_observations = []
    rewards = []
    dones = []
    actions = []

    obs, _ = env.reset()
    obs = ensure_optical_flow(obs, optical_flow_model)
    count_in_frame = 0
    action_scale = CONFIG['generator']['action_scale']

    for i, vel in enumerate(ee_vel_actions):
        if np.isclose(vel, np.zeros(6), atol=0.001).all():
            continue  # Skip zero actions

        # Unscale the velocity because env.step() applies the scale internally
        scaled_vel = vel / action_scale
        action = copy.deepcopy(scaled_vel)

        new_obs, reward, terminated, truncated, _ = env.step(scaled_vel)
        new_obs = ensure_optical_flow(new_obs, optical_flow_model)

        visualize_actions_camera(new_obs)

        observations.append(obs)
        new_observations.append(new_obs)
        rewards.append(reward)
        dones.append(terminated)
        actions.append(action)

        if (env.observation['point_mask'] > 0).any():
            count_in_frame += 1

        obs = new_obs  # Move to next observation, even if done/truncated

    return observations, new_observations, rewards, dones, actions, obs, count_in_frame

def execute_path_with_feedback(env,
                               path,
                               control_freq,
                               max_ee_vel,
                               config,
                               optical_flow_model: OpticalFlow = None):
    """
    Executes a path step-by-step, recalculating the required velocity at each step
    based on the robot's actual current state. This is a closed-loop approach.
    """
    # --- Data collection lists ---
    observations = []
    new_observations = []
    rewards = []
    dones = []
    actions = []

    # Get the initial observation from the environment
    obs, _ = env.reset()
    obs = ensure_optical_flow(obs, optical_flow_model)
    # Ensure robot is at the start of the path
    env.robot.set_joint_angles_no_collision(path[0])

    # Loop through each target waypoint in the path (starting from the second one)
    for i in range(len(path) - 1):
        target_joint_angles = path[i + 1]

        # 1. GET CURRENT STATE: Get the robot's *actual* current joint angles
        current_joint_angles = env.robot.get_joint_angles()

        # 2. CALCULATE VELOCITY FOR ONE STEP
        # Calculate required joint velocities to get from ACTUAL to TARGET
        joint_velocities = (np.array(target_joint_angles) - np.array(current_joint_angles)) * control_freq
        
        # Set robot to current pose to calculate the Jacobian correctly
        # (This is already its state, but good practice to be sure)
        env.robot.set_joint_angles_no_collision(current_joint_angles)
        
        jacobian = env.robot.calculate_jacobian()
        global_ee_vel = np.matmul(jacobian, joint_velocities)
        local_ee_vel = convert_global_action_to_local(env.robot, global_ee_vel) # This is our action

        # 3. HANDLE VELOCITY LIMITS (Optional but recommended)
        # This part is simplified; you can reuse the sub-stepping logic if needed.
        # Here, we just cap the velocity for this single step.
        if np.any(np.abs(local_ee_vel) > max_ee_vel):
            local_ee_vel = np.clip(local_ee_vel, -max_ee_vel, max_ee_vel)

        # 4. EXECUTE THE ACTION & STORE DATA
        action = copy.deepcopy(local_ee_vel)
        
        # Take a step in the environment with the calculated action
        new_obs, reward, terminated, truncated, _ = env.step(action)
        new_obs = ensure_optical_flow(new_obs, optical_flow_model)
        
        # Store the transition data
        observations.append(obs)
        new_observations.append(new_obs)
        rewards.append(reward)
        dones.append(terminated)
        actions.append(action)

        # Update the observation for the next loop iteration
        obs = new_obs

        # If the episode ends, stop trying to follow the path
        if terminated or truncated:
            break

    # Return all the collected data from the trajectory
    return observations, new_observations, rewards, dones, actions, obs

def append_transition_using_cartesian_planner(env, metadata, last_obs, observations, new_observations,
                                              rewards, dones, actions, count_in_frame,
                                              max_ee_vel, optical_flow_model: OpticalFlow = None):
    if last_obs is None:
        return observations, new_observations, rewards, dones, actions, count_in_frame

    obs = ensure_optical_flow(last_obs, optical_flow_model)
    control_time = env.config['simulation_setup']['control_time']
    action_scale = env.config['generator']['action_scale']
    goal_pos = metadata['goal_pos']

    # --- Start of Reference Logic ---
    start_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)

    # --- Visualization for Debugging ---
    goal_sphere_id = -1
    start_sphere_id = -1
    intended_path_line_id = -1
    actual_path_line_id = -1 # New variable for the second line

    if env.config['simulation_setup']['renders']:
        goal_sphere_id = draw_debug_sphere(env.pb_client, goal_pos, 0.07, [0, 1, 1, 0.5]) # Cyan Goal
        start_sphere_id = draw_debug_sphere(env.pb_client, start_pos, 0.05, [0, 0.8, 0.3, 0.5]) # Green Start
        # Draw the INTENDED path before any movement
        intended_path_line_id = env.pb_client.addUserDebugLine(start_pos, goal_pos, [1, 0, 1], lineWidth=5) # Magenta Intended Path

    # 1. Calculate the total required velocity ONCE
    ee_vel_global = np.zeros(6)
    ee_vel_global[:3] = (np.array(goal_pos) - np.array(start_pos)) / control_time
    
    # 2. Convert to local frame and determine the scale factor
    ee_vel_local = convert_global_action_to_local(env.robot, ee_vel_global)
    
    scale = 1.0
    if np.any(np.abs(ee_vel_local) > max_ee_vel):
        scale = np.max(np.abs(ee_vel_local)) / max_ee_vel
    
    action_per_step = ee_vel_local / (scale + 1e-6)
    num_steps_to_run = int(1.2* int(scale) / action_scale)

    if not dones or not dones[-1]:
        for j in range(num_steps_to_run):
            if np.isclose(action_per_step, np.zeros(6), atol=0.001).all():
                if dones: dones[-1] = True
                break

            action = copy.deepcopy(action_per_step)
            new_obs, reward, terminated, truncated, info = env.step(copy.deepcopy(action),f=2)
            new_obs = ensure_optical_flow(new_obs, optical_flow_model)
            
            visualize_actions_camera(new_obs)
            
            actions.append(action)
            new_observations.append(new_obs)
            observations.append(obs)
            rewards.append(reward)
            
            if (env.observation['point_mask'] > 0).any():
                count_in_frame += 1

            if terminated:
                dones.append(True)
                break
            elif info.get('collision_unacceptable_reward', 0) < 0 or info.get('collision_acceptable_reward', 0) < 0:
                dones.append(False)
                break
            elif j == num_steps_to_run - 1: # Last step timeout
                dones.append(False)
                break
            else:
                dones.append(terminated)

            obs = new_obs
    
    # --- NEW VISUALIZATION BLOCK ---
    if env.config['simulation_setup']['renders']:
        # Get the final position after all movements are done
        final_pos, _ = env.robot.get_current_pose(env.robot.tool0_link_idx)
        
        # Draw the ACTUAL path taken
        actual_path_line_id = env.pb_client.addUserDebugLine(start_pos, final_pos, [1, 1, 0], lineWidth=5) # Yellow Actual Path
        
        # Add the requested delay to see the lines
        time.sleep(3)

    # --- Cleanup Visualization ---
    if goal_sphere_id != -1:
        env.pb_client.removeBody(goal_sphere_id)
    if start_sphere_id != -1:
        env.pb_client.removeBody(start_sphere_id)
    if intended_path_line_id != -1:
        env.pb_client.removeUserDebugItem(intended_path_line_id)
    if actual_path_line_id != -1: # Clean up the new line
        env.pb_client.removeUserDebugItem(actual_path_line_id)
            
    return observations, new_observations, rewards, dones, actions, count_in_frame


def _compute_joint_path_length(path):
    if not path or len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(np.array(b) - np.array(a)) for a, b in zip(path, path[1:])))


def _compute_smoothness_cost(path):
    if not path or len(path) < 3:
        return 0.0
    diffs = [np.array(path[i + 1]) - np.array(path[i]) for i in range(len(path) - 1)]
    accelerations = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
    return float(sum(np.dot(acc, acc) for acc in accelerations))


def _task_space_goal_regression_cost(robot, path, goal_pos) -> float:
    """
    Sum of positive increments in ||EE - goal|| along the joint path (meters).
    Zero means monotone non-increasing approach to goal in Cartesian distance (matches progress
    component of ApplePickingEnv reward). Evaluated on raw path FK samples only (minimal CPU).
    """
    if path is None or len(path) < 2:
        return 0.0
    goal = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    q_restore = robot.get_joint_angles()
    try:
        dists: list[float] = []
        for q in path:
            robot.set_joint_angles_no_collision(q)
            ee_pos, _ = robot.get_current_pose(robot.tool0_link_idx)
            dists.append(float(np.linalg.norm(np.asarray(ee_pos, dtype=np.float64) - goal)))
        return float(sum(max(0.0, dists[i + 1] - dists[i]) for i in range(len(dists) - 1)))
    finally:
        robot.set_joint_angles_no_collision(q_restore)


def _planner_path_selection_tuple(
    path,
    smoothness_weight: float,
    robot,
    goal_pos,
    task_regression_weight: float,
) -> tuple[float, int, float, float]:
    """
    Sort key: lower is better. Returns (cost, n_wp, jl, task_reg) where cost =
    joint_length + w_smooth * smoothness + w_task *_task_regression.
    """
    jl = _compute_joint_path_length(path)
    sm = _compute_smoothness_cost(path)
    task_reg = _task_space_goal_regression_cost(robot, path, goal_pos)
    cost = float(
        jl
        + float(smoothness_weight) * sm
        + float(task_regression_weight) * task_reg
    )
    n_wp = len(path)
    return (cost, n_wp, jl, task_reg)


def _planner_path_is_better(
    candidate_path,
    best_info: dict | None,
    smoothness_weight: float,
    robot,
    candidate_goal_pos,
    task_regression_weight: float,
) -> bool:
    if best_info is None:
        return True
    t_new = _planner_path_selection_tuple(
        candidate_path,
        smoothness_weight,
        robot,
        candidate_goal_pos,
        task_regression_weight,
    )
    t_old = _planner_path_selection_tuple(
        best_info["path_waypoints"],
        smoothness_weight,
        robot,
        best_info["goal_pos"],
        task_regression_weight,
    )
    # Lexicographic on (cost, n_wp, jl); task_reg only affects cost
    key_new = (t_new[0], t_new[1], t_new[2])
    key_old = (t_old[0], t_old[1], t_old[2])
    return key_new < key_old


def save_rrt_path_to_hdf5(save_path, env_info, fail_mode, waypoints):
    lock_path = save_path + '.lock'
    with FileLock(lock_path):
        mode = 'a' if os.path.exists(save_path) else 'w'
        with h5py.File(save_path, mode) as f:
            name = str(time.time())
            grp = f.create_group(name)
            for key, value in env_info.items():
                if isinstance(value, (list, tuple)): value = np.array(value)
                grp.attrs[key] = value
            grp.attrs['fail_mode'] = fail_mode.value
            grp.create_dataset('waypoints', data=np.stack(waypoints) if waypoints else np.array([]))
    log.info(f"Saved RRT path to {save_path} under group {name}")

def save_agent_data_to_hdf5(observations, actions, rewards, dones, next_observations,
                            info, success, tree_info, robot_pos, robot_or, save_path):
    """Persist one trajectory. Returns the HDF5 group name (from *source_group_key* if provided)."""
    lock_path = save_path + '.lock'
    with FileLock(lock_path):
        mode = 'a' if os.path.exists(save_path) else 'w'
        with h5py.File(save_path, mode) as f:
            base_name = None
            if info is not None:
                raw = info.get("source_group_key")
                if raw is not None and str(raw).strip():
                    base_name = (
                        str(raw).strip()
                        .replace("/", "_")
                        .replace("\\", "_")
                    )
            if not base_name:
                base_name = str(time.time())

            grp_name = base_name
            dup_n = 0
            while grp_name in f:
                dup_n += 1
                grp_name = f"{base_name}__dup{dup_n}"

            grp = f.create_group(grp_name)

            # === Metadata ===
            for key, value in tree_info.items():
                grp.attrs[key] = value
            grp.attrs['robot_pos'] = robot_pos
            grp.attrs['robot_or'] = robot_or
            grp.attrs['success'] = success
            grp.attrs['hdf5_group_key'] = grp_name
            if info is not None:
                for k, v in info.items():
                    grp.attrs[k] = v

            # === Observations ===
            obs_keys = observations[0].keys()
            obs_grp = grp.create_group('observations')
            for key in obs_keys:
                stacked_obs = np.stack([obs[key] for obs in observations])
                obs_grp.create_dataset(key, data=stacked_obs, compression="gzip", compression_opts=4)

            # === Next Observations ===
            next_obs_grp = grp.create_group('next_observations')
            for key in obs_keys:
                stacked_next_obs = np.stack([obs[key] for obs in next_observations])
                next_obs_grp.create_dataset(key, data=stacked_next_obs, compression="gzip", compression_opts=4)

            # === Actions, Rewards, Dones ===
            actions = np.stack(actions)
            rewards = np.asarray(rewards, dtype=np.float64)
            dones = np.stack(dones)

            grp.create_dataset('actions', data=actions, compression="gzip", compression_opts=4)
            grp.create_dataset('rewards', data=rewards, compression="gzip", compression_opts=4)
            grp.create_dataset('dones', data=dones, compression="gzip", compression_opts=4)

    src_key = info.get("source_group_key") if info else None
    log.info(
        "Saved %s transitions to %s under HDF5 group %r (source_group_key=%r)",
        len(actions), save_path, grp_name, src_key,
    )
    return grp_name

def reset_robot_for_new_path(env, metadata, start_waypoints):
        """
        Resets the robot's joint configuration to the start of a path and
        updates the goal without reloading the entire simulation.
        """
        # Set the robot's arm to the starting configuration of the path
        start_joint_angles = start_waypoints[0]
        env.robot.set_joint_angles_no_collision(start_joint_angles)

        # Update the environment's target goal for the new path
        env.desired_goal = metadata['apple_center']
        env.reward_goal = metadata['goal_pos']
        # Give pybullet a moment to update the visual state
        #env.pb_client.stepSimulation()

import time # Make sure 'time' is imported at the top of your file

def tree_id_str_from_config(config: dict | None = None) -> str:
    """PyBullet mesh id, e.g. LPy_envy_00017."""
    cfg = config or CONFIG
    ts = cfg["tree_setup"]
    return f"{ts['tree_namespace']}_{ts['tree_type']}_{int(ts['tree_id']):05d}"


def apply_cli_config(args) -> None:
    """Apply argparse overrides into module-level CONFIG."""
    ts = CONFIG["tree_setup"]
    if getattr(args, "tree_id", None) is not None:
        ts["tree_id"] = int(args.tree_id)
    if getattr(args, "tree_scale", None) is not None:
        ts["scale"] = float(args.tree_scale)
    if getattr(args, "tree_pos", None):
        parts = [float(x.strip()) for x in str(args.tree_pos).split(",")]
        if len(parts) != 3:
            raise ValueError("--tree-pos must be three comma-separated floats, e.g. 0.5,0,0")
        ts["position"] = np.array(parts, dtype=np.float64)
    if getattr(args, "no_render", False):
        CONFIG["simulation_setup"]["renders"] = False
    if getattr(args, "no_texture", False):
        ts["apply_texture"] = False
    if getattr(args, "preserve_export_mtl", False):
        ts["preserve_export_mtl"] = True
    if getattr(args, "regenerate_urdf", False):
        ts["regenerate_urdf"] = True


def waypoints_metadata_matches_config(attrs, config: dict | None = None) -> tuple[bool, str]:
    """Check HDF5 path attrs were planned for the current CONFIG tree."""
    cfg = config or CONFIG
    expected_id = int(cfg["tree_setup"]["tree_id"])
    expected_str = tree_id_str_from_config(cfg)
    saved_id = attrs.get("tree_id")
    if saved_id is not None and int(saved_id) != expected_id:
        return False, f"tree_id {saved_id} != CONFIG {expected_id}"
    urdf = str(attrs.get("tree_urdf", ""))
    if urdf and expected_str not in urdf:
        return False, f"tree_urdf missing {expected_str}: {urdf}"
    saved_str = attrs.get("tree_id_str")
    if saved_str is not None and str(saved_str) != expected_str:
        return False, f"tree_id_str {saved_str} != {expected_str}"
    scale = attrs.get("tree_scale")
    if scale is not None and abs(float(scale) - float(cfg["tree_setup"]["scale"])) > 1e-4:
        return False, f"tree_scale {scale} != CONFIG {cfg['tree_setup']['scale']}"
    return True, ""


def _waypoint_tree_signature(attrs) -> tuple[int, float] | None:
    """(tree_id, tree_scale) from HDF5 group attrs, or None if incomplete."""
    if attrs.get("tree_id") is None or attrs.get("tree_scale") is None:
        return None
    return (int(attrs["tree_id"]), float(attrs["tree_scale"]))


def apply_tree_setup_from_waypoints_hdf5(
    waypoints_path,
    *,
    config: dict | None = None,
    ref_group_key: str | None = None,
) -> str:
    """
    Patch CONFIG tree_setup from a SUCCESS group in the waypoints HDF5.

    Uses *ref_group_key* when set; otherwise the first SUCCESS group.
    Warns if SUCCESS groups disagree on tree_id / tree_scale.
    """
    cfg = config or CONFIG
    ts = cfg["tree_setup"]
    path = Path(waypoints_path)
    with h5py.File(path, "r") as f:
        success_keys = [
            k for k in f.keys()
            if int(f[k].attrs.get("fail_mode", -1)) == ResultMode.SUCCESS.value
        ]
        if not success_keys:
            raise ValueError(f"No SUCCESS groups in {path}")

        if ref_group_key is not None:
            if ref_group_key not in f:
                raise KeyError(f"group {ref_group_key!r} not in {path}")
            ref_key = ref_group_key
        else:
            ref_key = success_keys[0]

        ref_attrs = dict(f[ref_key].attrs)
        ref_sig = _waypoint_tree_signature(ref_attrs)
        if ref_sig is None:
            raise ValueError(
                f"Group {ref_key!r} missing tree_id/tree_scale — cannot auto-apply CONFIG"
            )

        for k in success_keys:
            sig = _waypoint_tree_signature(dict(f[k].attrs))
            if sig is not None and sig != ref_sig:
                log.warning(
                    "Waypoints HDF5 mixed tree setup: %s has tree_id=%s scale=%s "
                    "but ref %s has tree_id=%s scale=%s — using ref group only",
                    k, sig[0], sig[1], ref_key, ref_sig[0], ref_sig[1],
                )

    before = {
        "tree_id": int(ts["tree_id"]),
        "scale": float(ts["scale"]),
        "position": np.asarray(ts["position"], dtype=np.float64).copy(),
    }
    ts["tree_id"] = ref_sig[0]
    ts["scale"] = ref_sig[1]
    if ref_attrs.get("tree_pos") is not None:
        ts["position"] = np.asarray(ref_attrs["tree_pos"], dtype=np.float64).reshape(3)
    if ref_attrs.get("tree_orientation") is not None:
        ts["orientation"] = np.asarray(
            ref_attrs["tree_orientation"], dtype=np.float64
        ).reshape(4)

    log.info(
        "CONFIG tree_setup from HDF5 group %r: tree_id %s→%s  scale %.4f→%.4f  "
        "pos %s→%s  id_str=%s",
        ref_key,
        before["tree_id"],
        ts["tree_id"],
        before["scale"],
        ts["scale"],
        np.round(before["position"], 4).tolist(),
        np.round(np.asarray(ts["position"]), 4).tolist(),
        ref_attrs.get("tree_id_str", tree_id_str_from_config(cfg)),
    )
    return ref_key


# Planner summary text files (per-run results)
LOG_PATH_PLANNER_RESULT_SUBDIR = "path_planner_result"
# Full session transcript for this script (all stages)
LOG_SESSION_SUBDIR = "data_generator_path_planner"


def _ensure_output_dirs():
    base_dir = Path(CONFIG['output']['base_dir'])
    waypoints_dir = base_dir / CONFIG['output']['waypoints_subdir']
    agent_data_dir = base_dir / CONFIG['output']['agent_data_subdir']
    logs_dir = Path("logs")
    planner_results_dir = logs_dir / LOG_PATH_PLANNER_RESULT_SUBDIR
    session_logs_dir = logs_dir / LOG_SESSION_SUBDIR
    for directory in (waypoints_dir, agent_data_dir, logs_dir, planner_results_dir, session_logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return waypoints_dir, agent_data_dir, logs_dir


def _setup_session_logging(stage_label: str) -> Path:
    """Mirror zenlog output to logs/data_generator_path_planner/<timestamp>_<stage>.log"""
    base = Path("logs") / LOG_SESSION_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_{stage_label}.log").resolve()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    # zenlog uses logger name 'pythonConfig' (see zenlog package), not the root logger
    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.addHandler(fh)
    zen_logger.setLevel(logging.DEBUG)
    return log_path


def _resolve_run_id(run_id, directory, should_exist=True):
    directory.mkdir(parents=True, exist_ok=True)
    if run_id:
        target_file = directory / f"{run_id}.hdf5"
        if should_exist and not target_file.exists():
            raise FileNotFoundError(f"Run ID '{run_id}' not found in {directory}.")
        return run_id, target_file
    existing_files = sorted(directory.glob("*.hdf5"))
    if not existing_files:
        raise FileNotFoundError(f"No run outputs found in {directory}.")
    latest_file = existing_files[-1]
    return latest_file.stem, latest_file


def run_visualization_stage(run_id=None):
    """
    Loads saved waypoint paths and visualizes them in the PyBullet GUI
    without generating any new data.
    """
    log.info("="*60)
    log.info("V_DATAGEN: STAGE 3 - VISUALIZE SAVED PATHS")
    log.info("="*60)

    waypoints_dir, _, _ = _ensure_output_dirs()
    try:
        run_id, waypoints_path = _resolve_run_id(run_id, waypoints_dir, should_exist=True)
    except FileNotFoundError as exc:
        log.error(str(exc))
        return
    CONFIG['output']['waypoints_hdf5_path'] = str(waypoints_path)
    log.info(f"Run ID: {run_id}")

    # Initialize the environment with rendering enabled
    env = ApplePickingEnv(config=CONFIG)
    pb_client = env.pb_client

    with h5py.File(CONFIG['output']['waypoints_hdf5_path'], 'r') as f:
        path_keys = list(f.keys())
        log.info(f"Found {len(path_keys)} saved paths to visualize.")

        for key in path_keys:
            log.info(f"--- Visualizing path: {key} ---")
            grp = f[key]
            
            # Skip paths that were not successfully planned
            if grp.attrs['fail_mode'] != ResultMode.SUCCESS.value:
                log.warning(f"Skipping path {key} due to fail_mode: {ResultMode(grp.attrs['fail_mode'])}")
                continue

            waypoints = grp['waypoints'][:]
            if len(waypoints) < 2:
                log.warning(f"Skipping path {key}, not enough waypoints.")
                continue

            # Load the scene configuration for this specific path
            metadata = {k: v for k, v in grp.attrs.items()}
            
            #env.reconfigure_scene(metadata)
            #reset_robot_for_new_path(env, metadata, waypoints)
            
            initial_robot_q = np.array(CONFIG['robot_setup']['start_joint_angles'])
            env.robot.set_joint_angles_no_collision(initial_robot_q)
            

            # Visualize the target apple for this trajectory
            apple_pos = metadata['apple_center']
            goal_sphere_id = draw_debug_sphere(pb_client, apple_pos, 0.05, [0.9, 0.1, 0.1, 0.9])

            # Visualize the goal position (where the gripper goes)
            goal_pos = metadata['goal_pos']
            gripper_goal_id = draw_debug_sphere(pb_client, goal_pos, 0.02, [0.1, 0.9, 0.1, 0.8])
            
            # Give the user a moment to see the start and end points
            time.sleep(2) 

            # Step through each waypoint to visualize the robot's movement
            log.info("waypoint_joints  path=%s  n_waypoints=%d", key, len(waypoints))
            for wp_idx, waypoint in enumerate(waypoints):
                env.robot.set_joint_angles_no_collision(waypoint)
                log.info(
                    "wp %3d/%3d  joints=%s",
                    wp_idx, len(waypoints) - 1,
                    np.round(waypoint, 4).tolist(),
                )
                if CONFIG['simulation_setup']['renders']:
                    time.sleep(0.01)
                    #env.robot.pbclient.stepSimulation()
                time.sleep(0.04) # A small delay to make the motion visible

            # Pause at the end to see the final pose
            time.sleep(2)

            # Clean up the visualization objects before the next trajectory
            pb_client.removeBody(goal_sphere_id)
            pb_client.removeBody(gripper_goal_id)

    log.info("Visualization stage complete.")
    env.close()

def run_prewarm_tree_pkl_stage():
    """Build or load pybullet_tree_sim/pkl/<id_str>_points.pkl without planning (slow first run)."""
    log.info("=" * 60)
    log.info("V_DATAGEN: PREWARM TREE PKL CACHE")
    log.info("=" * 60)
    ts = CONFIG["tree_setup"]
    log.info(
        "Tree: %s  scale=%s  pos=%s",
        tree_id_str_from_config(),
        ts["scale"],
        np.round(ts["position"], 4).tolist(),
    )
    pbutils = PyBUtils(renders=False)
    t0 = time.perf_counter()
    tree = Tree(
        pbutils=pbutils,
        tree_id=ts["tree_id"],
        tree_type=ts["tree_type"],
        namespace=ts["tree_namespace"],
        scale=ts["scale"],
        position=ts["position"],
        orientation=ts["orientation"],
    )
    n_apples = len(tree.get_apple_centroids())
    pkl_name = f"{tree.id_str}_points.pkl"
    log.info(
        "Prewarm done in %.1f s — %s  apples=%d  (cache: pybullet_tree_sim/pkl/%s)",
        time.perf_counter() - t0,
        tree.id_str,
        n_apples,
        pkl_name,
    )
    try:
        pbutils.pbclient.disconnect()
    except Exception:
        pass


def run_planner_stage(num_apples_to_plan, run_id=None):
    """MODIFIED: Stage 1 with advanced planning logic from reference."""
    log.info("="*60)
    log.info("V_DATAGEN: STAGE 1 - ADVANCED RRT PATH PLANNING")
    log.info("="*60)

    planning_cfg = CONFIG['planning']
    planner_type = planning_cfg.get('planner_type', 'rrt_connect')
    visibility_enabled = planning_cfg.get('visibility_check', {}).get('enable', False)
    log.info(f"Planner type      : {planner_type}")
    log.info(f"Visibility check  : {visibility_enabled}")
    log.info(
        "Tree setup        : %s  scale=%s  texture=%s  preserve_mtl=%s",
        tree_id_str_from_config(),
        CONFIG["tree_setup"]["scale"],
        CONFIG["tree_setup"].get("apply_texture", True),
        resolve_preserve_export_mtl(CONFIG["tree_setup"]),
    )
    waypoints_dir, _, logs_dir = _ensure_output_dirs()
    if run_id is None:
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    CONFIG['output']['waypoints_hdf5_path'] = str((waypoints_dir / f"{run_id}.hdf5").resolve())
    planner_results_dir = logs_dir / LOG_PATH_PLANNER_RESULT_SUBDIR
    planner_results_dir.mkdir(parents=True, exist_ok=True)
    log_path = (planner_results_dir / f"{run_id}.log").resolve()
    log.info(f"Run ID: {run_id}")
    orientation_seed = planning_cfg.get('orientation_seed')
    visibility_seed = planning_cfg.get('visibility_seed')
    enable_smoothing = planning_cfg.get('enable_smoothing', True)
    seed_random_generators(orientation_seed, label="orientation/rrt")
    log.info(f"Planning seeds -> orientation_seed: {orientation_seed}, visibility_seed: {visibility_seed}")

    env = ApplePickingEnv(config=CONFIG)
    robot, tree, pb_client = env.robot, env.tree, env.pb_client
    vis_cfg = CONFIG['visualization']
    sim_cfg = CONFIG['simulation_setup']
    tree_cfg = CONFIG['tree_setup']
    log.info(
        "VIZ_VALIDATE planner_config: hide_collision_visual=%s preserve_mtl=%s "
        "env_markers=%s draw_goal_spheres=%s radius_m=%.3f offset_m=%.3f gui_refresh_steps=%s",
        tree_cfg.get('hide_collision_visual'),
        resolve_preserve_export_mtl(tree_cfg),
        sim_cfg.get('draw_apple_centroid_markers'),
        vis_cfg.get('draw_goal_spheres'),
        float(vis_cfg.get('goal_sphere_radius', 0.08)),
        float(vis_cfg.get('apple_marker_surface_offset_m', 0.12)),
        sim_cfg.get('gui_refresh_steps', 1),
    )

    log.info("Accessing robot sensors for visibility check...")
    camera_sensor_name, camera_sensor_obj = next(((name, sensor) for name, sensor in robot.sensors.items() if "camera" in name), (None, None))
    if camera_sensor_obj:
        log.info(f"Found sensor '{camera_sensor_name}' for visibility checks.")
    else:
        log.warning("No camera sensor found. Visibility checks will be disabled.")

    vis_check_config = planning_cfg.get('visibility_check', {})
    if vis_check_config.get('enable', False) and camera_sensor_obj:
        num_targets = vis_check_config.get('num_targets', 0)
        
        # If num_targets is 0, don't create any visibility targets
        if num_targets == 0:
            log.info("num_targets is 0, skipping visibility target creation.")
        else:
            log.info("Creating visibility target spheres on apples...")
            apple_points = [np.array(center) for center in env.apple_centroids]
            
            if apple_points:
                # Use first N apple positions (not random)
                num_targets_to_use = min(num_targets, len(apple_points))
                selected_apple_points = apple_points[:num_targets_to_use]
                
                for point in selected_apple_points:
                    draw_debug_sphere(pb_client, point, vis_check_config['target_sphere_radius'], vis_check_config['target_color_rgba'])
                log.info(f"Created {num_targets_to_use} visibility targets from first {num_targets_to_use} apple positions.")
            else:
                log.warning("No apple points available to create visibility targets.")

    stats = PlannerStats()
    distance_fn, sample_fn, extend_fn, is_state_valid_fn = setup_planning_functions(
        robot, tree.pyb_id, planning_cfg, CONFIG['visualization'], pb_client, tree, camera_sensor_obj, metrics=stats)
    
    initial_robot_q = np.array(CONFIG['robot_setup']['start_joint_angles'])
    if num_apples_to_plan is None or num_apples_to_plan <= 0:
        apples_to_process = env.apple_centroids
        log.info("num_apples<=0 supplied, planning for all available apples.")
    else:
        apples_to_process = env.apple_centroids[:num_apples_to_plan]
    log.info(f"Found {len(apples_to_process)} apple centroids for planning.")
    stats.total_apples = len(apples_to_process)

    gcam_cfg = planning_cfg.get("goal_camera_filter", {})
    gcam_debug_dir = None
    if gcam_cfg.get("enable", False) and gcam_cfg.get("save_debug_images", False):
        gcam_debug_subdir = gcam_cfg.get("debug_subdir", "goal_camera_filter")
        gcam_debug_dir = logs_dir / gcam_debug_subdir / run_id
        gcam_debug_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Goal camera filter debug images -> {gcam_debug_dir.resolve()}")

    robot_pos = np.asarray(CONFIG['robot_setup']['start_position'], dtype=np.float64)
    marker_radius = float(vis_cfg.get('goal_sphere_radius', 0.08))
    marker_offset_m = float(vis_cfg.get('apple_marker_surface_offset_m', 0.12))

    apple_sphere_ids = []
    planner_centroids = []
    planner_marker_positions = []
    if vis_cfg.get('draw_goal_spheres', True):
        log.info(
            "Drawing planner goal spheres (radius=%.3f m, surface_offset=%.3f m toward robot)...",
            marker_radius,
            marker_offset_m,
        )
        for center in apples_to_process:
            centroid = np.asarray(center, dtype=np.float64).reshape(3)
            marker_pos = apple_marker_surface_position(centroid, robot_pos, marker_offset_m)
            planner_centroids.append(centroid)
            planner_marker_positions.append(marker_pos)
            sphere_id = draw_debug_sphere(
                pb_client,
                marker_pos,
                marker_radius,
                vis_cfg['goal_sphere_color'],
            )
            apple_sphere_ids.append(sphere_id)
        log_apple_marker_batch(
            "planner_goal_spheres",
            apple_sphere_ids,
            planner_centroids,
            planner_marker_positions,
            marker_radius,
            marker_offset_m,
        )
        gui_refresh(pb_client, sim_cfg.get('renders', False), steps=int(sim_cfg.get('gui_refresh_steps', 1)))

    for i, apple_center in enumerate(apples_to_process):
        apple_start_time = time.perf_counter()
        ik_attempts = 0
        log.info(f"--- Processing Apple #{i+1}/{len(apples_to_process)} ---")
        current_apple_sphere_id = apple_sphere_ids[i] if i < len(apple_sphere_ids) else -1
        if current_apple_sphere_id != -1:
            change_sphere_color(pb_client, current_apple_sphere_id, vis_cfg['active_apple_color'])
            gui_refresh(pb_client, sim_cfg.get('renders', False), steps=int(sim_cfg.get('gui_refresh_steps', 1)))

        robot.set_joint_angles_no_collision(initial_robot_q)
        path_found, shortest_path_info = False, None
        pick_w = float(planning_cfg.get("path_selection_smoothness_weight", 0.1))
        pick_task_w = float(planning_cfg.get("path_selection_task_regression_weight", 0.0))
        save_refined = bool(planning_cfg.get("save_refined_waypoints_to_hdf5", True))
        gcam_cfg = planning_cfg.get("goal_camera_filter", {})
        gcam_enabled = bool(gcam_cfg.get("enable", False))
        log.info(f"Goal camera filter (pre-RRT) : {gcam_enabled}")
        ik_gen = get_ik_solutions_for_apple(robot, 
                                            is_state_valid_fn, 
                                            apple_center, None, 
                                            CONFIG['planning'], 
                                            pb_client, 
                                            initial_robot_q)

        for q_goal_candidate, goal_pos_ik, target_orn_q in ik_gen:
            if q_goal_candidate is None: break
            ik_attempts += 1

            if gcam_enabled:
                if camera_sensor_obj is None:
                    log.warning("goal_camera_filter enable=True but no camera sensor; skipping filter.")
                else:
                    gcam_debug_ctx = None
                    if gcam_debug_dir is not None:
                        gcam_debug_ctx = GoalCameraFilterDebugContext(
                            output_dir=gcam_debug_dir,
                            apple_id=i + 1,
                            ori_id=ik_attempts,
                            run_id=run_id,
                        )
                    gres = evaluate_goal_camera_filter_method_a(
                        robot,
                        pb_client,  
                        camera_sensor_obj,
                        np.asarray(apple_center, dtype=np.float64),
                        q_goal_candidate,
                        gcam_cfg,
                        debug=gcam_debug_ctx,
                    )
                    if gres.debug_image_paths:
                        log.debug(
                            "GOAL_CAM_FILTER debug images (%d): %s",
                            len(gres.debug_image_paths),
                            ", ".join(gres.debug_image_paths),
                        )
                    proj_extra = ""
                    if gcam_cfg.get("log_projection_debug", True):
                        if gres.ndc_xyz is not None:
                            proj_extra += " ndc=(%.4f,%.4f,%.4f)" % gres.ndc_xyz
                        if gres.clip_div_w is not None:
                            proj_extra += f" clip_w={gres.clip_div_w:.6f}"
                        if gres.dist_cam_apple_m is not None:
                            proj_extra += f" dist_cam_apple_m={gres.dist_cam_apple_m:.4f}"
                        if gres.cam_eye_world is not None:
                            ex, ey, ez = gres.cam_eye_world
                            proj_extra += f" eye_m=({ex:.4f},{ey:.4f},{ez:.4f})"
                    log.info(
                        "GOAL_CAM_FILTER apple=%d/%d ik_try=%d visible_frac=%.4f min_req=%.4f PASS=%s "
                        "uv=(%s,%s) r_px=%.2f blue_px=%d disk_area_px=%.0f reason=%s%s",
                        i + 1,
                        len(apples_to_process),
                        ik_attempts,
                        gres.visible_frac,
                        gres.min_required_frac,
                        gres.passed,
                        gres.u,
                        gres.v,
                        gres.r_px,
                        gres.blue_in_disk_px,
                        gres.disk_area_px,
                        gres.reason,
                        proj_extra,
                    )
                    if not gres.passed:
                        stats.goal_camera_filter_rejections += 1
                        continue
            
            planner_type = CONFIG['planning'].get('planner_type', 'rrt_connect')
            planner_kwargs = dict(max_iterations=CONFIG['planning']['rrt_max_iterations'],
                                  max_time=CONFIG['planning']['rrt_max_time'])

            if planner_type == 'rrt_star':
                planner_kwargs['radius'] = CONFIG['planning'].get('rrt_radius', 0.3)
                path = pp.rrt_star(initial_robot_q, q_goal_candidate, distance_fn, sample_fn, extend_fn,
                                   is_state_valid_fn, **planner_kwargs)
            elif planner_type == 'rrt_star_connect':
                planner_kwargs['radius'] = CONFIG['planning'].get('rrt_radius', 0.3)
                path = pp.rrt_star_connect(initial_robot_q, q_goal_candidate, distance_fn, sample_fn, extend_fn,
                                           is_state_valid_fn, **planner_kwargs)
            else:
                path = pp.rrt_connect(initial_robot_q, q_goal_candidate, distance_fn, sample_fn, extend_fn,
                                      is_state_valid_fn, **planner_kwargs)
                
            if path is not None:
                log.info(f"RRT Path found with {len(path)} waypoints.")
                
                if _planner_path_is_better(
                    path,
                    shortest_path_info,
                    pick_w,
                    robot,
                    goal_pos_ik,
                    pick_task_w,
                ):
                    shortest_path_info = {'path_waypoints': path, 'goal_pos': goal_pos_ik, 'goal_orn': target_orn_q}
                    path_found = True
                    t_sel = _planner_path_selection_tuple(path, pick_w, robot, goal_pos_ik, pick_task_w)
                    log.info(
                        "New best RRT candidate  pick_tuple=(cost=%.4f, n_wp=%d, joint_len=%.4f, task_reg_m=%.4f)  w_smooth=%.3f w_task=%.3f",
                        t_sel[0], t_sel[1], t_sel[2], t_sel[3], pick_w, pick_task_w,
                    )
            else:
                log.info(f"RRT Path Planning FAILED for this IK candidate.")

        plan_time = time.perf_counter() - apple_start_time
        waypoint_count = None
        joint_path_length = None
        smoothness_cost = None
        refined_for_save: list | None = None
        raw_wp_n = 0
        pick_tuple_log: tuple[float, int, float, float] | None = None

        if path_found and shortest_path_info is not None:
            raw_path = shortest_path_info['path_waypoints']
            raw_wp_n = len(raw_path)
            pick_tuple_log = _planner_path_selection_tuple(
                raw_path, pick_w, robot, shortest_path_info["goal_pos"], pick_task_w
            )
            if save_refined:
                refined_for_save = shortcut_and_refine_path(
                    robot,
                    list(raw_path),
                    extend_fn,
                    is_state_valid_fn,
                    planning_cfg['task_space_refinement_threshold'],
                    enable_smoothing=planning_cfg.get('enable_smoothing', True),
                )
                waypoint_count = len(refined_for_save)
                joint_path_length = _compute_joint_path_length(refined_for_save)
                smoothness_cost = _compute_smoothness_cost(refined_for_save)
            else:
                refined_for_save = list(raw_path)
                waypoint_count = raw_wp_n
                joint_path_length = _compute_joint_path_length(raw_path)
                smoothness_cost = _compute_smoothness_cost(raw_path)
            failure_reason = None
        else:
            failure_reason = ("No feasible IK/collision-free goals" if ik_attempts == 0 else "RRT failed")

        stats.record_apple(i + 1, path_found, ik_attempts, plan_time, waypoint_count,
                           joint_path_length, smoothness_cost, failure_reason)

        if path_found:

            robot_pos, robot_or = robot.get_current_pose(-1)
            assert shortest_path_info is not None and refined_for_save is not None
            env_info = {
                'tree_id': int(tree.tree_id),
                'tree_id_str': tree.id_str,
                'tree_urdf': tree.urdf_path, 'tree_pos': tree.pos, 'tree_orientation': tree.orientation,
                'tree_scale': tree.scale, 'robot_pos': robot_pos, 'robot_or': robot_or,
                'apple_center': np.array(apple_center), 
                'goal_pos': np.array(shortest_path_info['goal_pos']), 
                'goal_orn': np.array(shortest_path_info['goal_orn']),
                'n_waypoints_planner_raw': int(raw_wp_n),
                'n_waypoints_saved': int(len(refined_for_save)),
                'waypoints_saved_refined': bool(save_refined),
                'path_pick_smoothness_weight': float(pick_w),
                'path_pick_task_regression_weight': float(pick_task_w),
                'path_pick_cost': float(pick_tuple_log[0]) if pick_tuple_log else 0.0,
                'path_pick_tuple_n_wp': int(pick_tuple_log[1]) if pick_tuple_log else 0,
                'path_pick_tuple_joint_len': float(pick_tuple_log[2]) if pick_tuple_log else 0.0,
                'path_pick_task_regression_m': float(pick_tuple_log[3]) if pick_tuple_log else 0.0,
                'success_pos_tolerance_m': float(planning_cfg['fk_pos_tolerance']),
                'success_orn_tolerance_rad': float(planning_cfg['fk_orn_tolerance']),
            }

            if CONFIG['simulation_setup']['renders']:
                    visualize_path(robot, refined_for_save, CONFIG['visualization']['path_visualization_delay'], camera_sensor_obj, pb_client)
            log.info(
                "PLAN_SAVE_SUMMARY  apple=%d/%d  raw_wp=%d  saved_wp=%d  refined=%s  pick_cost=%.4f  saved_joint_len=%.3f  saved_smooth=%.4f",
                i + 1, len(apples_to_process), raw_wp_n, len(refined_for_save), save_refined,
                float(pick_tuple_log[0]) if pick_tuple_log else 0.0,
                float(joint_path_length or 0.0),
                float(smoothness_cost or 0.0),
            )
                    
            save_rrt_path_to_hdf5(CONFIG['output']['waypoints_hdf5_path'], env_info, ResultMode.SUCCESS, refined_for_save)
        else:
            log.warning(f"FAILED to find any valid RRT path for Apple #{i+1}.")
            save_rrt_path_to_hdf5(CONFIG['output']['waypoints_hdf5_path'], {'apple_center': np.array(apple_center)}, ResultMode.NO_PATH, [])

        if current_apple_sphere_id != -1:
            change_sphere_color(pb_client, current_apple_sphere_id, vis_cfg['goal_sphere_color'])

    successful_apples = [apple for apple in stats.apples if apple['success']]
    avg_plan_time = (sum(apple['plan_time_sec'] for apple in successful_apples) / len(successful_apples)) if successful_apples else 0.0
    avg_path_len = (sum(apple.get('joint_path_length', 0.0) for apple in successful_apples) / len(successful_apples)) if successful_apples else 0.0
    avg_smoothness = (sum(apple.get('smoothness_cost', 0.0) for apple in successful_apples) / len(successful_apples)) if successful_apples else 0.0

    report_lines = stats.report_lines()
    report_lines.insert(2, f"Planner type      : {planner_type}")
    report_lines.insert(3, f"Visibility check  : {visibility_enabled}")
    report_lines.insert(6, f"Average plan time : {avg_plan_time:.2f}s")
    report_lines.insert(7, f"Average path len  : {avg_path_len:.3f}")
    report_lines.insert(8, f"Average smoothness: {avg_smoothness:.3f}")
    full_config_json = json.dumps(CONFIG, indent=2, default=str)
    config_lines = ["Full CONFIG:", full_config_json, "=" * 60]
    report_lines_with_config = report_lines + config_lines
    for line in report_lines_with_config:
        log.info(line)
    report_path = log_path
    report_path.write_text("\n".join(report_lines_with_config))
    log.info(f"Planner summary written to {report_path}")

    log.info("Planner stage complete.")
    env.close()

def run_generator_stage(run_id=None):
    """Stage 2: Load paths, refine them, and generate agent training data."""
    log.info("="*60)
    log.info("V_DATAGEN: STAGE 2 - AGENT DATA GENERATION")
    log.info("="*60)
    waypoints_dir, agent_data_dir, _ = _ensure_output_dirs()
    try:
        run_id, waypoints_path = _resolve_run_id(run_id, waypoints_dir, should_exist=True)
    except FileNotFoundError as exc:
        log.error(str(exc))
        return
    CONFIG['output']['waypoints_hdf5_path'] = str(waypoints_path)
    agent_data_path = (agent_data_dir / f"{run_id}.hdf5").resolve()
    CONFIG['output']['agent_data_hdf5_path'] = str(agent_data_path)
    log.info(f"Run ID: {run_id}")
    try:
        apply_tree_setup_from_waypoints_hdf5(waypoints_path)
    except (ValueError, KeyError, OSError) as exc:
        log.error("Cannot auto-apply tree_setup from waypoints HDF5: %s", exc)
        return

    env = ApplePickingEnv(config=CONFIG)
    robot = env.robot
    pb_client = env.pb_client   

    optical_flow_model = None
    try:
        optical_flow_model = OpticalFlow(size=DEFAULT_OPTICAL_FLOW_SIZE)
    except Exception as exc:
        log.error(f"Failed to initialize optical flow model: {exc}. Falling back to zero-flow observations.", exc_info=True)

    distance_fn, sample_fn, extend_fn, collision_fn = setup_planning_functions(
        robot, env.tree.pyb_id, CONFIG['planning'], CONFIG['visualization'], env.pb_client, env.tree)
        
    with h5py.File(CONFIG['output']['waypoints_hdf5_path'], 'r') as f:
        path_keys = list(f.keys())
        log.info(f"Found {len(path_keys)} saved paths to process.")
        
        for i, key in enumerate(path_keys):
            log.info(f"--- ({i+1}/{len(path_keys)}) Processing saved path: {key} ---")
            grp = f[key]
            
            if grp.attrs['fail_mode'] != ResultMode.SUCCESS.value:
                log.warning(f"Skipping path {key} due to fail_mode: {ResultMode(grp.attrs['fail_mode'])}")
                continue

            ok, reason = waypoints_metadata_matches_config(grp.attrs)
            if not ok:
                log.error(
                    "Skipping path %s: waypoints HDF5 tree mismatch (%s). "
                    "Re-run planner with matching --tree-id or fix CONFIG.",
                    key,
                    reason,
                )
                continue

            waypoints = grp['waypoints'][:]
            log.debug(f"Loaded {len(waypoints)} waypoints from HDF5 file.")
            print(f"DEBUG: Original path length from RRT is {len(waypoints)}") 

            if len(waypoints) < 2:
                log.warning(f"Skipping path {key}, not enough waypoints to form a trajectory.")
                continue
            
            
            metadata = {k: v for k, v in grp.attrs.items()}
            # Visualize the target apple for this trajectory
            apple_pos = metadata['apple_center']
            goal_sphere_id = draw_debug_sphere(pb_client, apple_pos, 0.05, [0.9, 0.1, 0.1, 0.8])
            # Visualize the goal position (where the gripper goes)
            goal_pos = metadata['goal_pos']
            gripper_goal_id = draw_debug_sphere(pb_client, goal_pos, 0.02, [0.1, 0.9, 0.1, 0.9])
            log.debug(f"  Desired_pos: {goal_pos}")
            # Clean up the visualization objects before the next trajectory
            
            waypoints_list = waypoints.tolist()
            metadata = {k: v for k, v in grp.attrs.items()}
            reset_robot_for_new_path(env, metadata, waypoints_list)
            
            refined_path = shortcut_and_refine_path(
                robot,
                waypoints_list,
                extend_fn,
                collision_fn,
                CONFIG['planning']['task_space_refinement_threshold'],
                enable_smoothing=CONFIG['planning'].get('enable_smoothing', True))
            
            if len(refined_path) > CONFIG['simulation_setup']['max_steps']:
                log.warning(f"Skipping path {key}, refined path has too many steps ({len(refined_path)}).")
                continue
                
           #OLD ---------------------------------------------------
            # ee_vel_actions = convert_ja_to_ee_vel(robot, refined_path, 1.0 / CONFIG['simulation_setup']['control_time'], CONFIG['planning']['max_ee_velocity'])
            
            # # Reset robot to the start of the path for simulation
            # reset_robot_for_new_path(env, metadata, waypoints_list)
            
            # log.info("Executing trajectory in simulation to get transitions...")
            # #path_transitions, last_obs = get_transitions_from_ee_vel(env, ee_vel_actions, CONFIG)
            # observations, new_observations, rewards, dones, actions, last_obs, count_in_frame = \
            #     get_transitions_from_ee_vel(env, ee_vel_actions, CONFIG)
           # ----------------------------------------------------------------
           
           # NEW ---------------------------------------------------------
            log.info("Executing trajectory with feedback to get transitions...")
            observations, new_observations, rewards, dones, actions, last_obs = \
                execute_path_with_feedback(env, 
                                        refined_path, 
                                        1.0 / CONFIG['simulation_setup']['control_time'], 
                                        CONFIG['planning']['max_ee_velocity'], 
                                        CONFIG,
                                        optical_flow_model=optical_flow_model)
            # You would need to calculate count_in_frame separately if still needed
            count_in_frame = sum(1 for obs in new_observations if (obs['point_mask'] > 0).any())
           # --------------------------------------------------------------  
            
            if not observations: # Check if the 'observations' list is empty
                log.error(f"No transitions were generated for path {key}. Velocities may have all been zero.")
                continue

            # Append final steps if the trajectory didn't terminate
            if not dones[-1]:
                 log.info("Path did not terminate. Appending final cartesian planner steps...")
                 # Pass all the lists to be appended to
                 observations, new_observations, rewards, dones, actions, count_in_frame = \
                    append_transition_using_cartesian_planner(env, metadata, last_obs, observations, new_observations,
                                                              rewards, dones, actions, count_in_frame,
                                                              CONFIG['planning']['max_ee_velocity'],
                                                              optical_flow_model=optical_flow_model)

            if observations and actions: # <-- MODIFIED THIS LINE
                success = dones[-1] # Success is the last 'done' state
                info = {
                    'count_in_frame': count_in_frame,
                    'path_length': len(actions),
                    'source_group_key': str(key),
                }
                robot_pos, robot_or = metadata['robot_pos'], metadata['robot_or']
                tree_info = {k: v for k, v in metadata.items() if 'robot' not in k}

                log.info(f"Generated {len(actions)} transitions. Final success state: {success}")
                
                # Save all the collected data
                _h5_grp = save_agent_data_to_hdf5(
                    observations, actions, rewards, dones, new_observations,
                    info, success, tree_info, robot_pos, robot_or,
                    CONFIG['output']['agent_data_hdf5_path'],
                )
                log.info("HDF5 group key for path %r -> %r", key, _h5_grp)
            else:
                log.error(f"Failed to generate any valid transitions (observations or actions were empty) for path {key}.")

            pb_client.removeBody(goal_sphere_id)
            pb_client.removeBody(gripper_goal_id)

                
    log.info("Generator stage complete.")
    env.close()

def run_axis_test():
    """
    This function helps diagnose coordinate frame issues by commanding
    a simple movement along the tool's local X-axis.
    """
    log.info("="*60)
    log.info("RUNNING COORDINATE FRAME AXIS TEST")
    log.info("="*60)

    env = ApplePickingEnv(config=CONFIG)
    robot = env.robot
    pb_client = env.pb_client

    # Get the tool's current position and orientation
    tool_pos, tool_orn_quat = robot.get_current_pose(robot.tool0_link_idx)
    
    # Draw the tool's local coordinate frame for reference
    # X-axis = Red, Y-axis = Green, Z-axis = Blue
    log.info("Drawing tool's local coordinate axes (X=Red, Y=Green, Z=Blue)")
    draw_debug_frame(pb_client, tool_pos, tool_orn_quat, length=0.2)

    # Define a simple action: move forward along the tool's local X-axis
    # This is a pure linear velocity with no rotation.
    local_action = np.array([0.2, 0, 0, 0, 0, 0])
    
    log.info(f"Commanding a simple local action: {local_action}")
    log.info("Observe which direction the arm moves relative to the colored lines.")
    
    # Give you time to see the axes before it moves
    time.sleep(5)

    # Execute the action using the environment's step function
    env.step(local_action)

    # Give you time to see the final position
    log.info("Movement complete. The arm should have moved along the RED line.")
    time.sleep(2)
    env.step(local_action)
    log.info("Movement complete. The arm should have moved along the RED line.")
    time.sleep(2)
    env.step(local_action)
    log.info("Movement complete. The arm should have moved along the RED line.")
    time.sleep(2)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apple Picking Data Generation Pipeline")
    parser.add_argument('--stage', type=str, required=False, choices=['prewarm', 'planner', 'generator', 'visualize', 'axis_test'], help="Which stage of the pipeline to run.")
    parser.add_argument('--num_apples', type=int, default=0, help="Number of apples to plan for in the planner stage (<=0 for all).")
    parser.add_argument('--planner_type', type=str, choices=['rrt_connect', 'rrt_star', 'rrt_star_connect'], help="Planner to use in planner stage.")
    parser.add_argument('--visualize_planning', action='store_true', help="Visualize the saved RRT paths without generating data.")
    parser.add_argument('--run_id', type=str, help="Optional run identifier to reuse existing outputs.")
    parser.add_argument('--tree-id', type=int, help="Override CONFIG tree_setup.tree_id (e.g. 17 for LPy_envy_00017).")
    parser.add_argument('--tree-scale', type=float, help="Override CONFIG tree_setup.scale.")
    parser.add_argument('--tree-pos', type=str, help="Override tree position as x,y,z (e.g. 0.5,0,0).")
    parser.add_argument('--no-render', action='store_true', help="Disable PyBullet GUI (faster planner batch runs).")
    parser.add_argument('--no-texture', action='store_true', help="Grey tree load (skip v2 MTL; faster, no camera color fidelity).")
    parser.add_argument('--preserve-export-mtl', action='store_true', help="Force v2 *_textured.mtl (default: auto if file exists).")
    parser.add_argument('--regenerate-urdf', action='store_true', help="Regenerate multi-material URDF before load.")

    args = parser.parse_args()
    apply_cli_config(args)

    _stage_label = None
    if args.stage == "axis_test":
        _stage_label = "axis_test"
    elif args.visualize_planning or args.stage == "visualize":
        _stage_label = "visualize"
    elif args.stage == "prewarm":
        _stage_label = "prewarm"
    elif args.stage == "planner":
        _stage_label = "planner"
    elif args.stage == "generator":
        _stage_label = "generator"
    if _stage_label is not None:
        _session_log_path = _setup_session_logging(_stage_label)
        log.info(f"Session log (full run output): {_session_log_path}")

    if args.stage == 'axis_test':
        run_axis_test()
    elif args.stage == 'prewarm':
        run_prewarm_tree_pkl_stage()
    elif args.visualize_planning or args.stage == 'visualize':
        run_visualization_stage(args.run_id)
    elif args.stage == 'planner':
        if args.planner_type:
            CONFIG['planning']['planner_type'] = args.planner_type
        run_planner_stage(args.num_apples, args.run_id)
    elif args.stage == 'generator':
        run_generator_stage(args.run_id)
    elif args.stage is None:
        parser.print_help()
