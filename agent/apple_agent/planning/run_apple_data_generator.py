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
- agent/apple_agent/ : this experiment's package (envs, planning, data, rl)

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

from apple_agent.envs.apple_picking_env import (
    ApplePickingEnv,
    resolve_preserve_export_mtl,
)
from apple_agent.planning.viz_debug import (
    apple_marker_surface_position,
    draw_debug_sphere,
    gui_refresh,
    log_apple_marker_batch,
    replace_debug_sphere,
)
from apple_agent.planning.goal_camera_filter import (
    GoalCameraFilterDebugContext,
    evaluate_goal_camera_filter_method_a,
)
from apple_agent.envs.utils_conversions import (
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


try:
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    from pybullet_tree_sim.robot import Robot
    from pybullet_tree_sim.tree import Tree
except ImportError as e:
    log.error(f"Failed to import necessary custom modules: {e}")
    log.error("Please ensure pybullet_tree_sim package is installed and accessible in your PYTHONPATH.")
    exit()

from apple_agent.config import CONFIG  # noqa: E402  (shared, mutated in place by the stages below)


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

def pregrasp_from_apple(apple_center: np.ndarray, approach_orn_quat, offset_dist: float) -> np.ndarray:
    """Target contract: pre-grasp = apple centre backed off along the tool z (approach) axis.

    Source-agnostic: apple_center may come from the sim mesh or from perception (detection + depth).
    """
    approach_axis = Rotation.from_quat(approach_orn_quat).as_matrix()[:, 2]
    return np.asarray(apple_center, dtype=np.float64) - offset_dist * approach_axis


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
        
        goal_pos_for_ik = pregrasp_from_apple(apple_center, target_orn_quat, offset_dist)
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


def _planner_path_rank_key(pick_tuple: tuple[float, int, float, float]) -> tuple[float, int, float]:
    """Lexicographic sort key; lower is better (cost, n_wp, joint_len)."""
    return (pick_tuple[0], pick_tuple[1], pick_tuple[2])


def _insert_top_planner_path(
    top_paths: list[dict],
    path,
    goal_pos,
    goal_orn,
    max_k: int,
    smoothness_weight: float,
    robot,
    task_regression_weight: float,
) -> int:
    """Keep up to max_k best paths (sorted best-first). Returns 1-based rank if kept, else 0."""
    pick_tuple = _planner_path_selection_tuple(
        path, smoothness_weight, robot, goal_pos, task_regression_weight
    )
    entry = {
        "path_waypoints": path,
        "goal_pos": goal_pos,
        "goal_orn": goal_orn,
        "pick_tuple": pick_tuple,
    }
    key = _planner_path_rank_key(pick_tuple)
    insert_idx = len(top_paths)
    for j, existing in enumerate(top_paths):
        if key < _planner_path_rank_key(existing["pick_tuple"]):
            insert_idx = j
            break
    top_paths.insert(insert_idx, entry)
    rank = insert_idx + 1
    if len(top_paths) > max_k:
        top_paths.pop()
        return 0
    return rank


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


def discover_available_tree_ids(tree_type: str = "envy", namespace: str = "LPy") -> list[int]:
    """All exported PyBullet tree ids in the unlabeled meshes folder (LPy_envy_000XX.obj → XX)."""
    from pybullet_tree_sim import MESHES_PATH

    obj_dir = Path(MESHES_PATH) / "trees" / tree_type / "unlabeled" / "obj"
    prefix = f"{namespace}_{tree_type}_"
    ids: list[int] = []
    for p in sorted(obj_dir.glob(f"{prefix}*.obj")):
        digits = p.stem[len(prefix):]
        if digits.isdigit():
            ids.append(int(digits))
    return sorted(set(ids))


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
        "env_markers=%s draw_goal_spheres=%s red_radius_m=%.3f blue_radius_m=%.3f "
        "offset_m=%.3f gui_refresh_steps=%s",
        tree_cfg.get('hide_collision_visual'),
        resolve_preserve_export_mtl(tree_cfg),
        sim_cfg.get('draw_apple_centroid_markers'),
        vis_cfg.get('draw_goal_spheres'),
        float(vis_cfg.get('goal_sphere_radius', 0.08)),
        float(vis_cfg.get('active_apple_sphere_radius', vis_cfg.get('goal_sphere_radius', 0.08))),
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
    active_marker_radius = float(vis_cfg.get('active_apple_sphere_radius', marker_radius))
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
        if current_apple_sphere_id != -1 and i < len(planner_marker_positions):
            apple_sphere_ids[i] = replace_debug_sphere(
                pb_client,
                current_apple_sphere_id,
                planner_marker_positions[i],
                active_marker_radius,
                vis_cfg['active_apple_color'],
            )
            gui_refresh(pb_client, sim_cfg.get('renders', False), steps=int(sim_cfg.get('gui_refresh_steps', 1)))

        robot.set_joint_angles_no_collision(initial_robot_q)
        top_paths: list[dict] = []
        max_top_paths = max(1, int(planning_cfg.get("num_top_paths_to_save_per_apple", 1)))
        pick_w = float(planning_cfg.get("path_selection_smoothness_weight", 0.1))
        pick_task_w = float(planning_cfg.get("path_selection_task_regression_weight", 0.0))
        save_refined = bool(planning_cfg.get("save_refined_waypoints_to_hdf5", True))
        gcam_cfg = planning_cfg.get("goal_camera_filter", {})
        gcam_enabled = bool(gcam_cfg.get("enable", False))
        ik_pool = int(planning_cfg["num_goal_candidates_to_try"])
        max_rrt_on_filtered = int(
            gcam_cfg.get("num_camera_filtered_goal_candidates_to_try", 10)
        )
        apply_rrt_cap = gcam_enabled and camera_sensor_obj is not None
        filtered_rrt_count = 0
        log.info(f"Goal camera filter (pre-RRT) : {gcam_enabled}")
        if gcam_enabled:
            log.info(
                "Goal camera filter budgets: IK pool=%d (num_goal_candidates_to_try)  "
                "max RRT on filter-passed=%d",
                ik_pool,
                max_rrt_on_filtered,
            )
            if apply_rrt_cap and ik_pool < max_rrt_on_filtered:
                log.warning(
                    "IK pool (%d) < max RRT on filter-passed (%d); "
                    "raise planning.num_goal_candidates_to_try (e.g. 200).",
                    ik_pool,
                    max_rrt_on_filtered,
                )
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
                    if filtered_rrt_count >= max_rrt_on_filtered:
                        log.info(
                            "GOAL_CAM_FILTER apple=%d/%d: reached max RRT on filter-passed (%d); "
                            "stopping IK/RRT loop.",
                            i + 1,
                            len(apples_to_process),
                            max_rrt_on_filtered,
                        )
                        break
                    filtered_rrt_count += 1
            
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
                rank = _insert_top_planner_path(
                    top_paths,
                    path,
                    goal_pos_ik,
                    target_orn_q,
                    max_top_paths,
                    pick_w,
                    robot,
                    pick_task_w,
                )
                if rank > 0:
                    t_sel = top_paths[rank - 1]["pick_tuple"]
                    log.info(
                        "Top-path insert rank=%d/%d  pick_tuple=(cost=%.4f, n_wp=%d, joint_len=%.4f, task_reg_m=%.4f)  "
                        "w_smooth=%.3f w_task=%.3f",
                        rank,
                        max_top_paths,
                        t_sel[0],
                        t_sel[1],
                        t_sel[2],
                        t_sel[3],
                        pick_w,
                        pick_task_w,
                    )
            else:
                log.info(f"RRT Path Planning FAILED for this IK candidate.")

        if apply_rrt_cap:
            log.info(
                "GOAL_CAM_FILTER apple=%d/%d summary: valid_ik=%d  filter-passed RRT runs=%d/%d",
                i + 1,
                len(apples_to_process),
                ik_attempts,
                filtered_rrt_count,
                max_rrt_on_filtered,
            )

        plan_time = time.perf_counter() - apple_start_time
        path_found = len(top_paths) > 0
        waypoint_count = None
        joint_path_length = None
        smoothness_cost = None
        failure_reason = None

        if path_found:
            best = top_paths[0]
            best_pick = best["pick_tuple"]
            if save_refined:
                best_refined = shortcut_and_refine_path(
                    robot,
                    list(best["path_waypoints"]),
                    extend_fn,
                    is_state_valid_fn,
                    planning_cfg['task_space_refinement_threshold'],
                    enable_smoothing=planning_cfg.get('enable_smoothing', True),
                )
                waypoint_count = len(best_refined)
                joint_path_length = _compute_joint_path_length(best_refined)
                smoothness_cost = _compute_smoothness_cost(best_refined)
            else:
                waypoint_count = len(best["path_waypoints"])
                joint_path_length = _compute_joint_path_length(best["path_waypoints"])
                smoothness_cost = _compute_smoothness_cost(best["path_waypoints"])
        else:
            if apply_rrt_cap and ik_attempts > 0 and filtered_rrt_count == 0:
                failure_reason = "All IK candidates failed goal camera filter"
            elif ik_attempts == 0:
                failure_reason = "No feasible IK/collision-free goals"
            else:
                failure_reason = "RRT failed"

        stats.record_apple(i + 1, path_found, ik_attempts, plan_time, waypoint_count,
                           joint_path_length, smoothness_cost, failure_reason)

        if path_found:
            robot_pos, robot_or = robot.get_current_pose(-1)
            log.info(
                "PLAN_SAVE apple=%d/%d  saving %d top path(s) (configured max=%d)",
                i + 1,
                len(apples_to_process),
                len(top_paths),
                max_top_paths,
            )
            for rank, candidate in enumerate(top_paths, start=1):
                raw_path = candidate["path_waypoints"]
                raw_wp_n = len(raw_path)
                pick_tuple_log = candidate["pick_tuple"]
                if save_refined:
                    refined_for_save = shortcut_and_refine_path(
                        robot,
                        list(raw_path),
                        extend_fn,
                        is_state_valid_fn,
                        planning_cfg['task_space_refinement_threshold'],
                        enable_smoothing=planning_cfg.get('enable_smoothing', True),
                    )
                else:
                    refined_for_save = list(raw_path)

                env_info = {
                    'tree_id': int(tree.tree_id),
                    'tree_id_str': tree.id_str,
                    'tree_urdf': tree.urdf_path, 'tree_pos': tree.pos, 'tree_orientation': tree.orientation,
                    'tree_scale': tree.scale, 'robot_pos': robot_pos, 'robot_or': robot_or,
                    'apple_center': np.array(apple_center),
                    'apple_index': int(i + 1),
                    'goal_pos': np.array(candidate['goal_pos']),
                    'goal_orn': np.array(candidate['goal_orn']),
                    'path_rank': int(rank),
                    'num_top_paths_configured': int(max_top_paths),
                    'n_waypoints_planner_raw': int(raw_wp_n),
                    'n_waypoints_saved': int(len(refined_for_save)),
                    'waypoints_saved_refined': bool(save_refined),
                    'path_pick_smoothness_weight': float(pick_w),
                    'path_pick_task_regression_weight': float(pick_task_w),
                    'path_pick_cost': float(pick_tuple_log[0]),
                    'path_pick_tuple_n_wp': int(pick_tuple_log[1]),
                    'path_pick_tuple_joint_len': float(pick_tuple_log[2]),
                    'path_pick_task_regression_m': float(pick_tuple_log[3]),
                    'success_pos_tolerance_m': float(planning_cfg['fk_pos_tolerance']),
                    'success_orn_tolerance_rad': float(planning_cfg['fk_orn_tolerance']),
                }

                if CONFIG['simulation_setup']['renders'] and rank == 1:
                    visualize_path(
                        robot,
                        refined_for_save,
                        CONFIG['visualization']['path_visualization_delay'],
                        camera_sensor_obj,
                        pb_client,
                    )
                log.info(
                    "PLAN_SAVE_SUMMARY  apple=%d/%d  rank=%d/%d  raw_wp=%d  saved_wp=%d  refined=%s  "
                    "pick_cost=%.4f  saved_joint_len=%.3f  saved_smooth=%.4f",
                    i + 1,
                    len(apples_to_process),
                    rank,
                    len(top_paths),
                    raw_wp_n,
                    len(refined_for_save),
                    save_refined,
                    float(pick_tuple_log[0]),
                    float(_compute_joint_path_length(refined_for_save)),
                    float(_compute_smoothness_cost(refined_for_save)),
                )
                save_rrt_path_to_hdf5(
                    CONFIG['output']['waypoints_hdf5_path'],
                    env_info,
                    ResultMode.SUCCESS,
                    refined_for_save,
                )
        else:
            log.warning(f"FAILED to find any valid RRT path for Apple #{i+1}.")
            save_rrt_path_to_hdf5(CONFIG['output']['waypoints_hdf5_path'], {'apple_center': np.array(apple_center)}, ResultMode.NO_PATH, [])

        if i < len(apple_sphere_ids) and apple_sphere_ids[i] != -1 and i < len(planner_marker_positions):
            apple_sphere_ids[i] = replace_debug_sphere(
                pb_client,
                apple_sphere_ids[i],
                planner_marker_positions[i],
                marker_radius,
                vis_cfg['goal_sphere_color'],
            )

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
    """Removed: the legacy stage-2 writer produced inconsistent observations (prev_rgb == rgb,
    mis-scaled cached flow). Use the schema-v2 generator instead."""
    log.error(
        "--stage generator was removed. Run: python -m apple_agent.data."
        "generate_agent_data_from_waypoints_2 --run-id %s --save-only-success", run_id or "<run_id>",
    )
    sys.exit(2)

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
        if args.tree_id is None:
            ts = CONFIG['tree_setup']
            tree_ids = discover_available_tree_ids(ts['tree_type'], ts['tree_namespace'])
            if not tree_ids:
                log.error("No exported trees found in meshes folder; nothing to plan.")
            else:
                log.info(
                    "No --tree-id given; planning ALL %d discovered trees: %s",
                    len(tree_ids), tree_ids,
                )
                for tid in tree_ids:
                    CONFIG['tree_setup']['tree_id'] = tid
                    per_run_id = (
                        f"tree{tid:02d}" if args.run_id is None
                        else f"{args.run_id}_tree{tid:02d}"
                    )
                    log.info("=== Planning tree_id=%d  run_id=%s ===", tid, per_run_id)
                    run_planner_stage(args.num_apples, per_run_id)
        else:
            run_planner_stage(args.num_apples, args.run_id)
    elif args.stage == 'generator':
        run_generator_stage(args.run_id)
    elif args.stage is None:
        parser.print_help()
