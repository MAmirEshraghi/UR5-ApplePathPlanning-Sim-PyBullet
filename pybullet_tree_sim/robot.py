"""
updates 12/8/2025 (Robin Eshraghi) include:

modifications:
    _assign_control_joints
    get_current_pose
    calculate_ik
    check_collisions

added:
    skew_symmetric
    make_adjoint
    convert_local_action_to_global
    find_closest_vertex_label_kdtree

"""

#!/usr/bin/env python3
from pybullet_tree_sim.sensors.camera import Camera
from pybullet_tree_sim.sensors.time_of_flight import TimeOfFlight
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.camera_helpers as ch
import pybullet_tree_sim.utils.xacro_utils as xutils
import pybullet_tree_sim.utils.yaml_utils as yutils
from pybullet_tree_sim import CONFIG_PATH, MESHES_PATH, URDF_PATH

import pybullet_tree_sim.plot as plot


import glob
from typing import List, Optional, Tuple
import modern_robotics as mr
import numpy as np
from pathlib import Path
import pybullet
import os
import time

from zenlog import log
import pprint as pp
from scipy.spatial.transform import Rotation


class Robot:

    _robot_configs_path = os.path.join(CONFIG_PATH, "description", "robot")
    _robot_xacro_path = os.path.join(URDF_PATH, "robot", "generic", "robot.urdf.xacro")
    _urdf_tmp_path = os.path.join(URDF_PATH, "tmp")

    print("----CONFIG_PATH",_robot_configs_path)
    print("----MESHES_PATH",_robot_xacro_path)
    print("----URDF_PATH",_urdf_tmp_path)

    def __init__(
        self,
        pbclient,
        position=(0, 0, 0),
        orientation=(0, 0, 0, 1),
        init_joint_angles: Optional[list] = None,
        randomize_pose=False,
        verbose=True,
        regenerate_urdf: bool = True,
    ) -> None:
        self.pbclient = pbclient
        self.verbose = verbose
        self.position = position
        self.orientation = orientation
        self.randomize_pose = randomize_pose  # TODO: This isn't set up anymore... fix
        # self.init_joint_angles = (  # TODO: dynamically assign from defaults (i.e. if linear-slider is loaded in)
        #     (
        #         -np.pi / 2 + np.pi / 4,
        #         -np.pi * 2 / 3,
        #         np.pi * 2 / 3,
        #         -np.pi,
        #         -np.pi / 2,
        #         0,
        #     )
        #     if init_joint_angles is None
        #     else init_joint_angles
        # )

        self.init_joint_angles = ( 
            (-1.978, -1.51, 2.622, -1.896, 0.579, 0.848)
            if init_joint_angles is None
            else init_joint_angles
        )
        

        # Robot setup
        self.robot = None
        self.robot_conf = {}
        if regenerate_urdf:
            self._generate_robot_urdf()
        else:
            self._load_robot_conf()
            self.robot_urdf_path = os.path.join(self._urdf_tmp_path, "robot.urdf")
            if not os.path.isfile(self.robot_urdf_path):
                raise FileNotFoundError(
                    f"Cached robot URDF not found: {self.robot_urdf_path}. "
                    "Generate it once (e.g. from a ROS environment) or construct Robot with "
                    "regenerate_urdf=True after installing ament_index_python."
                )
            log.info("Using cached robot URDF (xacro skipped): %s", self.robot_urdf_path)
        self._setup_robot()
        self.num_joints = self.pbclient.getNumJoints(self.robot)
        self.robot_stack: list = self.robot_conf["robot_stack"]

        # Links
        self.links = self._get_links()
        self.robot_collision_filter_idxs = self._assign_collision_links()
        self._apply_self_collision_config_from_yaml()
        self.tool0_link_idx = self._get_tool0_link_idx()

        # Joints
        self.joints = self._get_joints()
        self.control_joints, self.control_joint_idxs = self._assign_control_joints(self.joints)
        self.reset_robot()

        # Sensors
        self.sensors = self._get_sensors()
        self.debounce_time = time.time()

        # Robot action parameters
        self.action = None
        self.set_joint_angles_no_collision(self.init_joint_angles)
        self.pbclient.stepSimulation()
        return

    def _load_robot_conf(self) -> None:
        """Load robot.yaml + per-part YAML mappings (no xacro)."""
        self.robot_conf.update(yutils.load_yaml(os.path.join(self._robot_configs_path, "robot.yaml")))
        self.robot_conf.update({"robot_stack_qty": str(len(self.robot_conf["robot_stack"]))})
        self.robot_conf.update(
            {
                "mesh_base_path": MESHES_PATH,
                "urdf_base_path": URDF_PATH,
            }
        )
        for i, robot_part in enumerate(self.robot_conf["robot_stack"]):
            robot_part = robot_part.strip().lower()
            if i == 0:
                self.robot_conf.update({f"parent{i}": "world"})
            else:
                self.robot_conf.update({f"parent{i}": self.robot_conf["robot_stack"][i - 1]})
            self.robot_conf.update({f"robot_part{i}": self.robot_conf["robot_stack"][i]})
            part_conf = yutils.load_yaml(os.path.join(self._robot_configs_path, f"{robot_part}.yaml"))
            if part_conf is not None:
                self.robot_conf.update(part_conf)
            else:
                raise ValueError(f"Robot part {robot_part} not found in {self._robot_configs_path}")

    def _generate_robot_urdf(self) -> None:
        self._load_robot_conf()
        log.warn(self.robot_conf)
        # Generate URDF from mappings
        robot_urdf = xutils.load_urdf_from_xacro(
            xacro_path=self._robot_xacro_path,
            mappings=self.robot_conf,  # for some reason, this adds in the rest of the args from the xacro.
        )
        robot_urdf = robot_urdf.toprettyxml()

        # UR_description uses filename="package://<>" for meshes, and this doesn't work with pybullet
        for i, robot_part in enumerate(self.robot_conf["robot_stack"]):
            if robot_part.startswith("ur"):
                ur_absolute_mesh_path = "/opt/ros/humble/share/ur_description/meshes"
                robot_urdf = robot_urdf.replace(
                    f'filename="package://ur_description/meshes',
                    f'filename="{ur_absolute_mesh_path}',
                )
        # Save the generated URDF
        self.robot_urdf_path = os.path.join(self._urdf_tmp_path, "robot.urdf")
        xutils.save_urdf(robot_urdf, urdf_path=self.robot_urdf_path)
        return

    def _setup_robot(self):
        if self.robot is not None:
            self.pbclient.removeBody(self.robot)
            del self.robot
        flags = self.pbclient.URDF_USE_SELF_COLLISION

        if self.randomize_pose:
            delta_pos = np.random.rand(3) * 0.0
            delta_orientation = pybullet.getQuaternionFromEuler(np.random.rand(3) * np.pi / 180 * 5)
        else:
            delta_pos = np.array([0.0, 0.0, 0.0])
            delta_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])

        self.position, self.orientation = self.pbclient.multiplyTransforms(
            self.position, self.orientation, delta_pos, delta_orientation
        )
        self.robot = self.pbclient.loadURDF(  # TODO: change to PyB_ID, this isn't a robot
            self.robot_urdf_path,
            self.position,
            self.orientation,
            flags=flags,
            useFixedBase=True,
        )
        self.num_joints = self.pbclient.getNumJoints(self.robot)

        # # self.init_pos_ee = self.get_current_pose(self.end_effector_index)
        # # self.init_pos_base = self.get_current_pose(self.base_index)
        # # self.init_pos_eebase = self.get_current_pose(self.success_link_index)
        # # self.action = np.zeros(len(self.init_joint_angles), dtype=np.float32)
        # self.joint_angles = np.array(self.init_joint_angles).astype(np.float32)
        # # self.achieved_pos = np.array(self.get_current_pose(self.end_effector_index)[0])
        # # base_pos, base_or = self.get_current_pose(self.base_index)
        return

    def _get_joints(self) -> dict:
        """Return a dict of joint information for the robot"""
        joints = {}

        for i in range(self.num_joints):
            info = self.pbclient.getJointInfo(self.robot, i)
            joint_name = info[1].decode("utf-8")
            joints.update(
                {
                    joint_name: {
                        "id": i,
                        "type": info[2],
                        "lower_limit": info[8],
                        "upper_limit": info[9],
                        "max_force": info[10],
                        "max_velocity": info[11],
                    }
                }
            )
        log.warn(joints)
        return joints

    #(edit_robin)
    def _assign_control_joints(self, joints: dict) -> tuple[list]: 
        """Get list of controllable joints from the joint dict by joint type"""
        control_joints = []
        control_joint_idxs = []
        self.control_joint_lower_limits = []
        self.control_joint_upper_limits = []
        self.control_joint_ranges = []
        for joint, joint_info in joints.items():
            if joint_info["type"] == 0:  # TODO: Check if this works for prismatic joints or just revolute
                control_joints.append(joint)
                control_joint_idxs.append(joint_info["id"])

                # (robin) Population Logic 
                lower_limit = joint_info["lower_limit"]
                upper_limit = joint_info["upper_limit"]
                self.control_joint_lower_limits.append(lower_limit)
                self.control_joint_upper_limits.append(upper_limit)
                joint_range = upper_limit - lower_limit
                
                if joint_range <= 0:    # Handle non-positive range if necessary
                    if joint_info["type"] == pybullet.JOINT_REVOLUTE: joint_range = 2 * np.pi
                    else: joint_range = 10 # Adjust default as needed
                self.control_joint_ranges.append(joint_range)
                # -----------------------------------

                # self.joint_upper_limits,
                # self.joint_lower_limits,
                # self.joint_ranges,  # ,
            # Check if lists were populated correctly
            num_ctrl_joints = len(control_joint_idxs)
            if not (len(self.control_joint_lower_limits) == num_ctrl_joints and \
                    len(self.control_joint_upper_limits) == num_ctrl_joints and \
                    len(self.control_joint_ranges) == num_ctrl_joints):
                        log.error("Mismatch in populating joint limits/ranges!")

        return control_joints, control_joint_idxs

    def _get_links(self) -> dict:
        links = {}
        for i in range(self.num_joints):
            info = self.pbclient.getJointInfo(self.robot, i)
            # log.debug(info)
            child_link_name = info[12].decode("utf-8")
            links.update({child_link_name: {"id": i, "tf_from_parent": info[14]}})
        return links

    def _assign_collision_links(self) -> list:
        """Find tool0/base pairs, add to collision filter list.
        Requires that the robot part is ordered from base to tool0.

        TODO: Clean this up, there must be a better way.
        """
        robot_collision_filter_idxs = []
        for i, robot_part in enumerate(self.robot_conf["robot_stack"]):
            joint_info = self.pbclient.getJointInfo(self.robot, i)
            if i == 0:
                continue
            else:
                if (
                    robot_part + "__base" in self.links.keys()
                    and self.robot_conf["robot_stack"][i - 1] + "__tool0" in self.links.keys()
                ):
                    robot_collision_filter_idxs.append(
                        (
                            self.links[robot_part + "__base"]["id"],
                            self.links[self.robot_conf["robot_stack"][i - 1] + "__tool0"]["id"],
                        )
                    )
        return robot_collision_filter_idxs

    def _get_tool0_link_idx(self):
        """TODO: Clean up, find a better way?"""
        return self.links[self.robot_conf["robot_stack"][-1] + "__tool0"]["id"]

    def _get_sensors(self) -> dict:
        """Get sensors on robot based on runtime config files"""
        sensors = {}
        robot_part_runtime_conf_path = os.path.join(CONFIG_PATH, "runtime")
        for robot_part in self.robot_stack:
            robot_part_runtime_conf_file = os.path.join(robot_part_runtime_conf_path, f"{robot_part}.yaml")
            robot_part_conf = yutils.load_yaml(robot_part_runtime_conf_file)
            if robot_part_conf is None:
                log.warn(f"Could not load configuration for {robot_part_runtime_conf_file}")
                continue
            try:
                sensor_conf = robot_part_conf["sensors"]
            except KeyError:
                log.warn(f"No sensor configuration found for {robot_part} in {robot_part_runtime_conf_file}")
                continue
            # Create sensors
            for sensor_name, metadata in sensor_conf.items():
                if metadata["type"] == "camera":
                    sensors.update(
                        {
                            sensor_name: Camera(
                                pbclient=self.pbclient,
                                sensor_name=metadata["name"],
                            )
                        }
                    )
                elif metadata["type"] == "tof":
                    sensors.update(
                        {
                            sensor_name: TimeOfFlight(
                                pbclient=self.pbclient,
                                sensor_name=metadata["name"],
                            )
                        }
                    )
                # Assign TF frame and pybullet frame id to sensor
                sensors[sensor_name].tf_frame = (
                    robot_part + "__" + metadata["tf_frame"]
                )  # TODO: find a better way to get the prefix. If
                # from robot_conf, need standard for all robots TODO: log an error if robot_part doesn't have all the right frames. Xacro utils?
                sensors[sensor_name].tf_id = self.links[sensors[sensor_name].tf_frame]["id"]
                sensors[sensor_name].tf_from_parent = self.links[sensors[sensor_name].tf_frame]["tf_from_parent"]
                sensors[sensor_name].pan = metadata["pan"]  # TODO: Are these only for cameras/toFs? If so, needs reorg
                sensors[sensor_name].tilt = metadata["tilt"]
            # for key, value in yamlcontent.items():
            #     sensors.update({Path(file).stem: yamlcontent})
        return sensors

    def _get_sensor_attributes(self) -> dict:
        """TODO: Delete? This is not used"""
        sensor_attributes = {}
        # Cameras
        camera_configs_path = os.path.join(
            CONFIG_PATH,
            "description",
            "camera",
        )
        camera_configs_files = glob.glob(os.path.join(camera_configs_path, "*.yaml"))
        for file in camera_configs_files:
            yamlcontent = yutils.load_yaml(file)
            # for key, value in yamlcontent.items():
            sensor_attributes.update({Path(file).stem: yamlcontent})
        # ToFs: TODO: lots of repetitive code here, refactor
        tof_configs_path = os.path.join(CONFIG_PATH, "description", "tof")
        tof_configs_files = glob.glob(os.path.join(tof_configs_path, "*.yaml"))
        for file in tof_configs_files:
            yamlcontent = yutils.load_yaml(file)
            for key, value in yamlcontent.items():
                sensor_attributes.update({Path(file).stem: yamlcontent})
        return sensor_attributes

    def reset_robot(self, joint_angles: tuple = None) -> None:
        if self.robot is None:
            return
        self.init_joint_angles = (
            (
                -np.pi / 2 + np.pi / 4,
                -np.pi * 2 / 3,
                np.pi * 2 / 3,
                -np.pi,
                -np.pi / 2,
                0,
            )
            if joint_angles is None
            else joint_angles
        )
        self.set_joint_angles_no_collision(self.init_joint_angles)
        # self.pbclient.resetJointState() # TODO Fill out
        return

    def remove_robot(self):
        self.pbclient.removeBody(self.robot)
        self.robot = None
        return

    def set_joint_angles_no_collision(self, joint_angles) -> None:
        assert len(joint_angles) == len(self.control_joints)
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            self.pbclient.resetJointState(self.robot, joint["id"], joint_angles[i], targetVelocity=0)
        return

    def set_joint_angles(self, joint_angles) -> None:
        """Set joint angles using pybullet motor control"""
        assert len(joint_angles) == len(self.control_joints)
        poses = []
        indices = []
        forces = []

        for i, name in enumerate(self.control_joints):
            # joint = self.robot_conf["joint_info"][name]
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indices.append(joint["id"])
            forces.append(joint["max_force"])

        self.pbclient.setJointMotorControlArray(
            self.robot,
            indices,
            self.pbclient.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.05] * len(poses),
            forces=forces,
        )
        return

    def set_joint_velocities(self, joint_velocities) -> None:
        """Set joint velocities using pybullet motor control"""
        assert len(joint_velocities) == len(self.control_joints)
        velocities = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            velocities.append(joint_velocities[i])
            indexes.append(joint["id"])
            forces.append(joint["max_force"])

        self.pbclient.setJointMotorControlArray(
            self.robot,
            indexes,
            controlMode=self.pbclient.VELOCITY_CONTROL,
            targetVelocities=joint_velocities,
        )
        return

    # TODO: Use proprty decorator for getters?
    def get_joint_velocities(self):
        j = self.pbclient.getJointStates(self.robot, self.control_joint_idxs)
        joints = tuple((i[1] for i in j))
        return joints  # type: ignore

    def get_joint_angles(self):
        """Return joint angles"""
        # print(self.robot_conf["control_joints"], self.controllable_joints_idxs)
        j = self.pbclient.getJointStates(self.robot, self.control_joint_idxs)
        joints = tuple((i[0] for i in j))
        return joints

    # (edit_robin)
    def get_current_pose(self, index):
        """
        Returns the current pose of the specified link index.
        MODIFIED: Handles the base link (-1) using getBasePositionAndOrientation for robustness.
        """
        if index == -1:
            # Use the dedicated function for getting the base position and orientation
            position, orientation = self.pbclient.getBasePositionAndOrientation(self.robot)
        else:
            # Use getLinkState for all other links
            link_state = self.pbclient.getLinkState(self.robot, index, computeForwardKinematics=True)
            position, orientation = link_state[4], link_state[5]
    
        return position, orientation

    def get_current_vel(self, index):
        """Returns current pose of the index."""
        link_state = self.pbclient.getLinkState(
            self.robot,
            index,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
        )
        trans, ang = link_state[6], link_state[7]
        return trans, ang

    def get_condition_number(self):
        """Get condition number of the jacobian"""
        jacobian = self.calculate_jacobian()
        condition_number = np.linalg.cond(jacobian)
        return condition_number

    # (edit_robin)
    def calculate_ik(self, position, orientation): 
        """Calculates joint angles from end effector position and orientation using inverse kinematics"""

        # # --- LOGGING for Monitoring ---
        # log.debug("--- calculate_ik called with parameters: ---")
        # log.debug(f"  Target Position: {np.round(position, 4)}")
        # log.debug(f"  Target Orientation (quat): {np.round(orientation, 4)}")
        # log.debug(f"  tool0_link_idx: {self.tool0_link_idx}")
        current_joint_damping = [0.01] * len(self.control_joints) 
        # log.debug(f"  Joint Damping: {np.round(current_joint_damping, 4)}")
        # log.debug(f"  Control Joint Lower Limits: {np.round(self.control_joint_lower_limits, 4)}")
        # log.debug(f"  Control Joint Upper Limits: {np.round(self.control_joint_upper_limits, 4)}")
        # log.debug(f"  Control Joint Ranges: {np.round(self.control_joint_ranges, 4)}")
        # log.debug(f"  Rest Poses (if used): {self.init_joint_angles}") # Uncomment if you add restPoses

        joint_angles = self.pbclient.calculateInverseKinematics(
            self.robot,
            self.tool0_link_idx,
            position,
            orientation,
            jointDamping=current_joint_damping,             # Use the logged variable
            upperLimits=self.control_joint_upper_limits,
            lowerLimits=self.control_joint_lower_limits,
            jointRanges=self.control_joint_ranges,
            #restPoses=self.init_joint_angles               # ERROR: considering this line resulted in an error!
        )
        return joint_angles

    # (add_robin)
    def skew_symmetric(self, vector): 
        """
        Compute the skew-symmetric matrix of a vector.
        """
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    # (add_robin)
    def make_adjoint(self, rotation_matrix, translation_vector):
        """
        Create the adjoint transformation matrix.
        """
        p_skew = self.skew_symmetric(translation_vector)
        adjoint = np.block([
            [rotation_matrix, np.zeros((3, 3))],
            [np.dot(p_skew, rotation_matrix), rotation_matrix]
        ])
        return adjoint

    # (add_robin)
    def convert_local_action_to_global(self, action):
        """Convert local action to global action using the tool0 link as the reference frame."""
        # Get the current pose of the reference link (tool0)
        pos, orient = self.get_current_pose(self.tool0_link_idx)
        
        # Create the adjoint transformation matrix from the current pose
        adjoint = self.make_adjoint(np.array(self.pbclient.getMatrixFromQuaternion(orient)).reshape(3, 3), pos)
        
        # Transform the local action vector to a global action vector
        global_action = np.dot(adjoint, action)

        return global_action
    
    def calculate_jacobian(self):
        jacobian = self.pbclient.calculateJacobian(
            self.robot,
            self.tool0_link_idx,
            [0, 0, 0],
            self.get_joint_angles(),
            [0] * len(self.control_joints),
            [0] * len(self.control_joints),
        )
        jacobian = np.vstack(jacobian)
        return jacobian

    #check outputs
    def calculate_joint_velocities_from_ee_velocity(self, end_effector_velocity):
        """Calculate joint velocities from end effector velocity using jacobian using least squares"""
        jacobian = self.calculate_jacobian()
        inv_jacobian = np.linalg.pinv(jacobian)
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity).astype(np.float32)
        return joint_velocities, jacobian  # (-joint_velocities)

    def calculate_joint_velocities_from_ee_velocity_dls(self, end_effector_velocity, damping_factor: float = 0.05):
        """Calculate joint velocities from end effector velocity using damped least squares"""
        jacobian = self.calculate_jacobian()
        identity_matrix = np.eye(jacobian.shape[0])
        damped_matrix = jacobian @ jacobian.T + (damping_factor**2) * identity_matrix
        damped_matrix_inv = np.linalg.inv(damped_matrix)
        dls_inv_jacobian = jacobian.T @ damped_matrix_inv
        joint_velocities = dls_inv_jacobian @ end_effector_velocity
        return joint_velocities, jacobian   # (-joint_velocities)

    # TODO: Make camera a separate class?
    def create_camera_transform(self, world_position, world_orientation, camera: Camera | None) -> np.ndarray:
        """Create rotation matrix for camera"""
        base_offset_tf = np.identity(4)

        ee_transform = np.identity(4)
        ee_rot_mat = np.array(self.pbclient.getMatrixFromQuaternion(world_orientation)).reshape(3, 3)

        ee_transform[:3, :3] = ee_rot_mat
        ee_transform[:3, 3] = world_position

        tilt_tf = np.identity(4)
        pan_tf = np.identity(4)
        if camera is None:
            tilt = 0
            pan = 0
        else:
            tilt = camera.tilt
            pan = camera.tilt
            base_offset_tf[:3, 3] = camera.xyz_offset

        tilt_rot = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt), -np.sin(tilt)],
                [0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        tilt_tf[:3, :3] = tilt_rot

        pan_rot = np.array(
            [
                [np.cos(pan), 0, np.sin(pan)],
                [0, 1, 0],
                [-np.sin(pan), 0, np.cos(pan)],
            ]
        )
        pan_tf[:3, :3] = pan_rot

        tf = ee_transform @ pan_tf @ tilt_tf @ base_offset_tf
        return tf

    def _apply_self_collision_config_from_yaml(self) -> None:
        """Apply robot.yaml ``disable_self_collisions`` (default off → URDF + mount filters only)."""
        raw = self.robot_conf.get("disable_self_collisions", False)
        if isinstance(raw, str):
            disable_all_self = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            disable_all_self = bool(raw)

        if disable_all_self:
            self.disable_self_collision()
            log.info(
                "Robot self-collision: disabled for all link pairs (robot.yaml disable_self_collisions)."
            )
        else:
            self.set_collision_filter(self.robot_collision_filter_idxs)
            log.info(
                "Robot self-collision: URDF defaults + mount filter pairs disabled (count=%s).",
                len(self.robot_collision_filter_idxs),
            )

    def set_collision_filter(self, robot_collision_filter_idxs) -> None:
        """Disable collision between pruner and arm"""
        for i in robot_collision_filter_idxs:
            self.pbclient.setCollisionFilterPair(self.robot, self.robot, i[0], i[1], 0)
        return

    def unset_collision_filter(self):
        """Enable collision between pruner and arm"""
        for i in self.robot_collision_filter_idxs:
            self.pbclient.setCollisionFilterPair(self.robot, self.robot, i[0], i[1], 1)
        return

    def disable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.pbclient.setCollisionFilterPair(self.robot, self.robot, i, j, 0)
        return

    def enable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.pbclient.setCollisionFilterPair(self.robot, self.robot, i, j, 1)
        return

    # (add_robin) helper for "check_collisions" func.
    def find_closest_vertex_label_kdtree(self, 
                                         contact_point_world: np.ndarray, 
                                         tree_object_with_kdtree: 'Tree', # Expects the Tree object
                                         distance_upper_bound: float = 0.1) -> str | None: # Optional max distance
        """
        Finds the label of the closest vertex in the tree's K-D tree
        to the contact_point_world.
        Args:
            contact_point_world: The [x,y,z] of the contact.
            tree_object_with_kdtree: The Tree object which should have a pre-built .kdtree
                                     and .kdtree_vertex_labels.
            distance_upper_bound: Optional. Max distance to consider a vertex "close".
                                  Helps to avoid spurious matches if contact is far from any tree vertex.
        Returns:
            The string label of the closest vertex, or None if not found or too far.
        """
        if tree_object_with_kdtree is None or \
           not hasattr(tree_object_with_kdtree, 'kdtree') or \
           tree_object_with_kdtree.kdtree is None or \
           not hasattr(tree_object_with_kdtree, 'kdtree_vertex_labels') or \
           not tree_object_with_kdtree.kdtree_vertex_labels:
            
            log.warning("find_closest_vertex_label_kdtree: Tree object or its K-D tree/labels not available.")
            return "UNKNOWN_KDTREE_UNAVAILABLE" 

        try:
            # Query the K-D tree for the nearest neighbor / k=1 : find the 1st nearest neighbor / 
            distance_sq, index = tree_object_with_kdtree.kdtree.query(
                contact_point_world, k=1, distance_upper_bound=distance_upper_bound
            )

            if index < len(tree_object_with_kdtree.kdtree_vertex_labels): # Check if a neighbor was found within distance_upper_bound
                closest_label = tree_object_with_kdtree.kdtree_vertex_labels[index]
                # log.debug(f"KDTree Query: Contact {np.round(contact_point_world,3)}, "
                #           f"Closest Vertex Index: {index}, "
                #           f"Coord: {np.round(tree_object_with_kdtree.kdtree_vertex_coords_np[index],3)}, "
                #           f"Dist: {np.sqrt(distance_sq):.4f}, Label: {closest_label}")
                return closest_label
            else: # No point found within distance_upper_bound
                # log.debug(f"KDTree Query: No vertex found within distance_upper_bound ({distance_upper_bound}) "
                #           f"for contact point {np.round(contact_point_world,3)}")
                return "UNKNOWN_TOO_FAR" 

        except Exception as e:
            log.error(f"Error during K-D tree query: {e}", exc_info=True)
            return "UNKNOWN_KDTREE_QUERY_ERROR"

    @staticmethod
    def _vertex_label_matches_config(label: Optional[str], patterns: List[str]) -> bool:
        """Match KD-tree labels to planner config (e.g. APPLE_12 counts as APPLE)."""
        if label is None or not patterns:
            return False
        if label in patterns:
            return True
        if isinstance(label, str) and label.startswith("APPLE_") and "APPLE" in patterns:
            return True
        return False
    
    # (edit_robin) Major overhaul to use K-D tree for labeling
    def check_collisions(self,  # Eddited By Robben
                         collision_objects: dict,
                         tree_object_for_labeling: 'Tree' = None,
                         acceptable_labels: Optional[List[str]] = None,
                         unacceptable_labels: Optional[List[str]] = None) -> Tuple[bool, dict]:
        collision_info = {
            "collisions_acceptable": False,
            "collisions_unacceptable": False,
            "contact_point_on_obstacle": None,
            "collided_obstacle_label": None, #will be determined by closest vertex
            "self_collision_contact_pos": None,
            "is_self_collision_unacceptable": False
        }

        # `collision_objects` here is still expected to map your high-level labels 
        # (like "TRUNK", "BRANCH" from CONFIG) to the PyBullet ID of the *whole tree*.
        # Example: {'TRUNK': whole_tree_id, 'BRANCH': whole_tree_id}
        #
        # Optional lists align with feature_apple_path_planning/run_apple_data_generator.py
        # planning_config keys acceptable_collision_labels / unacceptable_collision_labels.

        if acceptable_labels is not None:
            collision_acceptable_list = list(acceptable_labels)
        else:
            collision_acceptable_list = []

        if unacceptable_labels is not None:
            collision_unacceptable_env_list = list(unacceptable_labels)
        else:
            collision_unacceptable_env_list = ["BRANCH", "TRUNK", "APPLE", "LEAF", "SPUR"]
            collision_unacceptable_env_list += [f"APPLE_{i}" for i in range(41)]

        
        # 1. Check for "Acceptable" Environmental Collisions
        for acc_type in collision_acceptable_list:
            # We assume 'APPLE' might also map to the whole tree_id if you don't have separate apple bodies
            # Or, if you *do* have separate apple bodies, collision_objects[acc_type] would be the apple's pb_id.
            if acc_type in collision_objects and collision_objects[acc_type] is not None:
                contacts = self.pbclient.getContactPoints(bodyA=self.robot, bodyB=collision_objects[acc_type])
                if contacts:
                    for contact_point_detail in contacts:
                        if contact_point_detail[8] < -0.001: 
                            collision_info["collisions_acceptable"] = True
                            # If it's an APPLE collision, we can directly use the acc_type as the label
                            collision_info["contact_point_on_obstacle"] = contact_point_detail[6] 
                            collision_info["collided_obstacle_label"] = acc_type 
                            log.warning(f"---- ACCEPTABLE collision with {acc_type} ----")
                            break
                if collision_info["collisions_acceptable"]:
                    break
        
        # 2. Check for "Unacceptable" Environmental Collisions (TRUNK, BRANCH) ---
        # This section now primarily serves to trigger the check against the whole tree.
        # The actual label ("TRUNK" or "BRANCH") will be determined by querying vertex data.
        if not collision_info["collisions_unacceptable"] and not collision_info["collisions_acceptable"]:
            # We only need to check against the tree once if TRUNK and BRANCH map to the same tree_id
            # Get the tree_id from one of the unacceptable labels if available
            tree_pb_id_to_check = None
            for label in collision_unacceptable_env_list:
                if label in collision_objects and collision_objects[label] is not None:
                    tree_pb_id_to_check = collision_objects[label]
                    break
            
            if tree_pb_id_to_check is not None:
                contacts_with_tree = self.pbclient.getContactPoints(bodyA=self.robot, bodyB=tree_pb_id_to_check)
                if contacts_with_tree:
                    for contact_point_detail in contacts_with_tree:
                        if contact_point_detail[8] < -0.0001: # Penetration: 1cm
                            contact_pos_on_tree = contact_point_detail[6] # Position on BodyB (tree)
                            collision_info["contact_point_on_obstacle"] = contact_pos_on_tree
                            
                            determined_label = "UNKNOWN_NO_CLOSE_VERTEX" # Default if labeling fails
                            if tree_object_for_labeling and hasattr(tree_object_for_labeling, 'transformed_vertices'):
                                determined_label = self.find_closest_vertex_label_kdtree(
                                    np.array(contact_pos_on_tree), 
                                    tree_object_for_labeling    #.transformed_vertices
                                )
                                if determined_label is None: determined_label = "UNKNOWN_NO_CLOSE_VERTEX"
                            
                            collision_info["collided_obstacle_label"] = determined_label
                            
                            # Now decide if this determined_label is unacceptable / acceptable
                            if self._vertex_label_matches_config(
                                    determined_label, collision_unacceptable_env_list):
                                collision_info["collisions_unacceptable"] = True
                                log.error(f"---- UNACCEPTABLE collision with determined label '{determined_label}' at {np.round(contact_pos_on_tree, 3)} ----")
                            elif self._vertex_label_matches_config(
                                    determined_label, collision_acceptable_list):
                                collision_info["collisions_acceptable"] = True
                                log.warning(f"---- ACCEPTABLE collision with determined label '{determined_label}' at {np.round(contact_pos_on_tree, 3)} ----")
                            else:
                                # Unlisted mesh label or unknown: treat as unacceptable (safe default).
                                collision_info["collisions_unacceptable"] = True
                                log.error(
                                    f"---- UNACCEPTABLE collision (unclassified tree label) "
                                    f"'{determined_label}' at {np.round(contact_pos_on_tree, 3)} ----"
                                )
                            break 


        # 3. Check for Self-Collisions
        if not collision_info["collisions_unacceptable"]: 
            self_contact_points = self.pbclient.getContactPoints(bodyA=self.robot, bodyB=self.robot)
            for contact_point_detail in self_contact_points:
                link_a_index = contact_point_detail[3]
                link_b_index = contact_point_detail[4]
                contact_distance = contact_point_detail[8]
                if link_a_index == link_b_index: continue
                if contact_distance < -0.015:   #-0.015: 
                    collision_info["collisions_unacceptable"] = True
                    collision_info["is_self_collision_unacceptable"] = True
                    collision_info["self_collision_contact_pos"] = contact_point_detail[5]
                    log.error(f"---- UNACCEPTABLE SELF_COLLISION ---- ...")
                    break
        
        #log.debug(f"Collision Check Result (after vertex query): {collision_info}")
        return collision_info["collisions_unacceptable"], collision_info

    def check_success_collision(self, body_b) -> bool:
        """Check if there are any collisions between the robot and the environment
        Returns: Bool
        """
        collisions_success = self.pbclient.getContactPoints(
            bodyA=self.robot, bodyB=body_b, linkIndexA=self.success_link_index
        )
        for i in range(len(collisions_success)):
            if collisions_success[i][-6] < 0.05:
                if self.verbose > 1:
                    print("DEBUG: Success Collision")
                return True

        return False

    def set_collision_filter_tree(self, collision_objects):
        for i in collision_objects.values():
            for j in range(self.num_joints):
                self.pbclient.setCollisionFilterPair(self.robot, i, j, 0, 0)
        return

    def unset_collision_filter_tree(self, collision_objects):
        for i in collision_objects.values():
            for j in range(self.num_joints):
                self.pbclient.setCollisionFilterPair(self.robot, i, j, 0, 1)
        return

    # TODO: Better types for getCameraImage
    def get_view_mat_at_curr_pose(self, camera: Camera | TimeOfFlight) -> np.ndarray:
        """Get view matrix at current pose"""
        # tf_id = camera.tf_id
        # log.error(camera.tf_frame)
        # log.error(camera.tf_id)
        pos, orientation = self.get_current_pose(camera.tf_id)
        # log.debug(f"{camera.tf_frame} Pose: {pos}, Orientation: {Rotation.from_quat(orientation).as_euler('xyz')}")

        camera_tf = self.create_camera_transform(pos, orientation, camera)

        # Initial vectors
        camera_vector = np.array([0, 0, 1]) @ camera_tf[:3, :3].T  #
        up_vector = np.array([0, 1, 0]) @ camera_tf[:3, :3].T  #

        # log.debug(f"cam vec, up vec:\n{camera_vector}, {up_vector}")

        view_matrix = self.pbclient.computeViewMatrix(
            cameraEyePosition=camera_tf[:3, 3],
            cameraTargetPosition=camera_tf[:3, 3] + 0.1 * camera_vector,
            cameraUpVector=up_vector,
        )
        return view_matrix

    def get_view_mat_by_id_at_curr_pose(self, id) -> np.ndarray:
        pos, orientation = self.get_current_pose(id)
        camera_tf = self.create_camera_transform(pos, orientation, camera=None)
        # log.debug(f"End effector Pose: {pos}, Orientation: {Rotation.from_quat(orientation).as_euler('xyz')}")
        # log.debug(f"camera_tf:\n{camera_tf}")
        # Initial vectors
        camera_vector = np.array([0, 0, 1]) @ camera_tf[:3, :3].T
        up_vector = np.array([0, 1, 0]) @ camera_tf[:3, :3].T

        # log.debug(f"camera_vector: {camera_vector}")
        # log.debug(f"up_vector: {up_vector}")

        view_matrix = self.pbclient.computeViewMatrix(
            cameraEyePosition=camera_tf[:3, 3],
            cameraTargetPosition=camera_tf[:3, 3] + 0.1 * camera_vector,
            cameraUpVector=up_vector,
        )
        # log.warn(np.asarray(view_matrix).reshape((4,4), order="F"))
        return view_matrix

    def get_rgbd_at_cur_pose(self, camera, type, view_matrix) -> Tuple:
        """Get RGBD image at current pose
        @param camera (Camera): Camera object
        @param type (str): either 'robot' or 'viz'
        @param view_matrix (tuple): 16x1 tuple representing the view matrix

        @return (rgb, depth) (tuple): RGB and depth images
        """
        # cur_p = self.ur5.get_current_pose(self.camera_link_index)
        rgbd = self.get_image_at_curr_pose(camera, type, view_matrix)
        # log.warn(rgbd)
        rgb, depth = ch.seperate_rgbd_rgb_d(rgbd, height=camera.depth_height, width=camera.depth_width)
        depth = depth.astype(np.float32)
        # log.debug(f"depth_before_lin: {depth}")
        # log.debug(camera.far_val)
        # log.debug(camera.near_val)
        depth = PyBUtils.linearize_depth(depth, camera.far_val, camera.near_val)
        # log.debug(f"depth_after_lin: {depth}")

        return rgb, depth

    def get_image_at_curr_pose(self, camera, type, view_matrix=None) -> list:
        """Take the current pose of the sensor and capture an image
        TODO: Add support for different types of sensors? For now, full rgbd
        TOOD: Move sensor/viz view to different methods, viz to pruning env?"""
        if type == "sensor":
            if view_matrix is None:
                raise ValueError("view_matrix cannot be None for sensor view")
            return self.pbclient.getCameraImage(
                width=camera.depth_width,  # TODO: how to work with depth + RGB?
                height=camera.depth_height,
                viewMatrix=view_matrix,
                projectionMatrix=camera.depth_proj_mat,  # TODO: ^ same
                renderer=self.pbclient.ER_BULLET_HARDWARE_OPENGL,
                flags=self.pbclient.ER_NO_SEGMENTATION_MASK,
                lightDirection=[1, 1, 1],
            )
        elif type == "viz":
            return self.pbclient.getCameraImage(
                width=camera.depth_width,
                height=camera.depth_height,
                viewMatrix=self.viz_view_matrix,
                projectionMatrix=self.viz_proj_matrix,
                renderer=self.pbclient.ER_BULLET_HARDWARE_OPENGL,
                flags=self.pbclient.ER_NO_SEGMENTATION_MASK,
                lightDirection=[1, 1, 1],
            )

    # def get_camera_location(
    #     self, camera: Camera
    # ):  # TODO: get transform from dictionary. choose between rgb or tof frames
    #     pose, orientation = self.get_current_pose(camera.tf_id)
    #     tilt = camera.tilt

    #     camera_tf = self.create_camera_transform(camera=camera)
    #     return camera_tf

    # Collision checking
    #
    def deproject_pixels_to_points(
        self,
        sensor,
        data: np.ndarray,
        view_matrix: np.ndarray,
        return_frame: str = "world",
        debug=False,
    ) -> np.ndarray:
        """Compute frame XYZ from image XY and measured depth. Default frame is 'world'.
        (pixel_coords -- [u,v]) -> (film_coords -- [x,y]) -> (camera_coords -- [X, Y, Z]) -> (world_coords -- [U, V, W])

        https://ksimek.github.io/2013/08/13/intrinsic/
        https://gachiemchiep.github.io/cheatsheet/camera_calibration/
        https://stackoverflow.com/questions/4124041/is-opengl-coordinate-system-left-handed-or-right-handed

        @param data: nx1 array of depth values, Fortran order (column-first)
        @param view_matrix: 4x4 matrix (world -> camera transform)
        @param return_frame: str, either 'camera' or 'world'

        @return: nx4 array of world XYZ coordinates
        """

        # log.debug(f"View matrix:\n{view_matrix}")
        # log.debug(f"Data\n{data}")

        # Flip the y and z axes to convert from OpenGL camera frame to standard camera frame.
        # https://stackoverflow.com/questions/4124041/is-opengl-coordinate-system-left-handed-or-right-handed
        # https://github.com/bitlw/LearnProjMatrix/blob/main/doc/OpenGL_Projection.md#introduction
        # view_matrix[1:3, :] = -view_matrix[1:3, :]
        #

        proj_matrix = np.asarray(sensor.depth_proj_mat).reshape([4, 4], order="F")
        # log.warning(f'{proj_matrix}')
        # proj_matrix = camera.depth_proj_mat

        # rgb, depth = self.pbutils.get_rgbd_at_cur_pose(type='robot', view_matrix=view_matrix)
        # data = depth.reshape((self.cam_width * self.cam_height, 1), order="F")

        # Get camera intrinsics from projection matrix. If square camera, these should be the same.
        fx = proj_matrix[0, 0]
        fy = proj_matrix[1, 1]  # if square camera, these should be the same

        # Get camera coordinates from film-plane coordinates. Scale, add z (depth), then homogenize the matrix.
        sensor_coords = np.divide(np.multiply(sensor.depth_film_coords, data), [fx, fy])
        sensor_coords = np.concatenate(
            (
                sensor_coords,
                data,
                np.ones((sensor.depth_width * sensor.depth_height, 1)),
            ),
            axis=1,
        )

        return_frame = return_frame.strip().lower()
        if return_frame == "sensor":
            return sensor_coords
        elif return_frame == "world":
            world_coords = (mr.TransInv(view_matrix) @ sensor_coords.T).T
            log.err("HELLO WORLD")
            if debug:
                plot.debug_deproject_pixels_to_points(
                    sensor=sensor,
                    data=data,
                    cam_coords=sensor_coords,
                    world_coords=world_coords,
                    view_matrix=view_matrix,
                )
            return world_coords
        else:
            raise ValueError("Invalid return frame. Must be 'camera' or 'world'.")

    def get_cam_to_frame_coords(
        self,
        cam_coords: np.ndarray,
        start_frame: str,
        end_frame: str = "world",
        view_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert camera coordinates to other frame coordinates. Default is world.
        @param cam_coords: nx4 array of camera XYZ coordinates
        @param start_frame: str
        @param end_frame: str (default = 'world')
        @param view_matrix: 4x4 matrix (frame -> camera transform)

        @return: nx4 array of world XYZ coordinates
        """
        end_frame = end_frame.strip().lower()
        if end_frame == "world" and view_matrix is None:
            raise ValueError("View matrix required for world frame conversion.")
        elif end_frame == "world":
            return (mr.TransInv(view_matrix) @ cam_coords.T).T

        start_frame = start_frame.strip().lower()
        end_frame_coords = (mr.TransInv(self.static_frames[f"{start_frame}_to_{end_frame}"]) @ cam_coords.T).T

        return end_frame_coords

    def compute_deprojected_point_mask(self):
        point_mask = []
        # TODO: Make this function nicer
        # Function. Be Nice.
        # """Find projection stuff in 'treefitting'. Simpole depth to point mask conversion."""
        # point_mask = np.zeros((self.pbutils.cam_height, self.pbutils.cam_width), dtype=np.float32)

        # proj_matrix = np.asarray(self.pbutils.proj_mat).reshape([4, 4], order="F")
        # view_matrix = np.asarray(
        # self.ur5.get_view_mat_at_curr_pose(pan=self.cam_pan, tilt=self.cam_tilt, xyz_offset=self.cam_xyz_offset)
        # ).reshape([4, 4], order="F")
        # projection = (
        #     proj_matrix
        #     @ view_matrix
        #     @ np.array([0.5, 0.5, 1, 1])
        #     # @ np.array([self.tree_goal_pos[0], self.tree_goal_pos[1], self.tree_goal_pos[2], 1])
        # )

        # # Normalize by w -> homogeneous coordinates
        # projection = projection / projection[3]
        # # log.info(f"View matrix: {view_matrix}")
        # log.info(f"Projection: {projection}")
        # # if projection within 1,-1, set point mask to 1
        # if projection[0] < 1 and projection[0] > -1 and projection[1] < 1 and projection[1] > -1:
        #     projection = (projection + 1) / 2
        #     row = self.pbutils.cam_height - 1 - int(projection[1] * (self.pbutils.cam_height))
        #     col = int(projection[0] * self.pbutils.cam_width)
        #     radius = 5  # TODO: Make this a variable proportional to distance
        #     # modern scikit uses a tuple for center
        #     rr, cc = skdraw.disk((row, col), radius)
        #     print(rr, cc)
        #     point_mask[
        #         np.clip(0, rr, self.pbutils.cam_height - 1), np.clip(0, cc, self.pbutils.cam_width - 1)
        #     ] = 1  # TODO: This is a hack, numbers shouldnt exceed max and min anyways

        # # resize point mask to algo_height, algo_width
        # point_mask_resize = cv2.resize(point_mask, dsize=(self.algo_width, self.algo_height))
        # point_mask = np.expand_dims(point_mask_resize, axis=0).astype(np.float32)
        return point_mask

    def get_key_move_action(self, keys_pressed: list) -> np.ndarray:
        """Return an action based on the keys pressed."""
        action = np.zeros((6, 1), dtype=float)
        if keys_pressed:
            if ord("a") in keys_pressed:
                action[0, 0] += 0.01
            if ord("d") in keys_pressed:
                action[0, 0] += -0.01
            if ord("s") in keys_pressed:
                action[1, 0] += 0.01
            if ord("w") in keys_pressed:
                action[1, 0] += -0.01
            if ord("q") in keys_pressed:
                action[2, 0] += 0.01
            if ord("e") in keys_pressed:
                action[2, 0] += -0.01
            if ord("z") in keys_pressed:
                action[3, 0] += 0.01
            if ord("c") in keys_pressed:
                action[3, 0] += -0.01
            if ord("x") in keys_pressed:
                action[4, 0] += 0.01
            if ord("v") in keys_pressed:
                action[4, 0] += -0.01
            if ord("r") in keys_pressed:
                action[5, 0] += 0.05
            if ord("f") in keys_pressed:
                action[5, 0] += -0.05
        return action

    def get_key_sensor_action(self, keys_pressed: list) -> dict | None:
        if keys_pressed:
            if ord("p") in keys_pressed:
                if time.time() - self.debounce_time > 0.1:
                    sensor_data = {}
                    for sensor_name, sensor in self.sensors.items():
                        if sensor_name.startswith("tof"):
                            view_matrix = self.get_view_mat_at_curr_pose(camera=sensor)
                            rgb, depth = self.get_rgbd_at_cur_pose(
                                camera=sensor,
                                type="sensor",
                                view_matrix=view_matrix,
                            )
                            view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
                            depth = depth.reshape(
                                (sensor.depth_width * sensor.depth_height, 1),
                                order="F",
                            )

                            camera_points = self.deproject_pixels_to_points(
                                sensor=sensor,
                                data=depth,
                                view_matrix=view_matrix,
                                return_frame="world",
                                debug=False,
                            )

                            sensor_data.update(
                                {
                                    sensor_name: {
                                        "data": camera_points,
                                        "tf_frame": sensor.tf_frame,
                                        "view_matrix": view_matrix,
                                        "sensor": sensor,
                                    }
                                }
                            )
                    # plot.debug_sensor_world_data(sensor_data)
                    self.debounce_time = time.time()
                    return sensor_data
                else:
                    return
        return

    def get_key_action(self, keys_pressed: list):
        move_action = self.get_key_move_action(keys_pressed=keys_pressed)
        sensor_data = self.get_key_sensor_action(keys_pressed=keys_pressed)
        # controller_action = self.get_key_controller_action(keys_pressed=keys_pressed)

        return


def main():
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    import time

    pbutils = PyBUtils(renders=True)

    robot = Robot(
        pbclient=pbutils.pbclient,
    )

    return


if __name__ == "__main__":
    main()
