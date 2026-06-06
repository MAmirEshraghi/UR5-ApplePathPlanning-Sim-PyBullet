#!/usr/bin/env python3
from __future__ import annotations

"""
Author (s): Abhinav Jain, Luke Strohbehn
"""
import os
from typing import List, Tuple

import numpy as np
import pybullet as pybullet
from nptyping import NDArray
from pybullet_utils import bullet_client as bc
from scipy.constants import g as grav

from pybullet_tree_sim import MESHES_PATH, URDF_PATH, TEXTURES_PATH

# from pybullet_tree_sim.camera import Camera
# from pybullet_tree_sim.utils.camera_helpers import get_fov_from_dfov

from zenlog import log


def create_room_backdrop(pbclient, *, collision: bool = False) -> list[int]:
    """Create textured room box (floor + walls + ceiling). Returns body ids."""
    wall_texture_path = os.path.join(TEXTURES_PATH, "leaves-dead.png")
    tex_id = pbclient.loadTexture(wall_texture_path)
    if not collision:
        log.info("Room backdrop: visual-only (no wall/floor collision for planning)")

    def _wall(wall_dim, wall_pos, euler_rotation):
        wall_viz = pbclient.createVisualShape(
            shapeType=pbclient.GEOM_BOX,
            halfExtents=wall_dim,
        )
        wall_col = (
            pbclient.createCollisionShape(
                shapeType=pbclient.GEOM_BOX,
                halfExtents=wall_dim,
            )
            if collision
            else -1
        )
        wall_id = pbclient.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=wall_viz,
            baseCollisionShapeIndex=wall_col,
            basePosition=wall_pos,
            baseOrientation=list(pbclient.getQuaternionFromEuler(euler_rotation)),
        )
        pbclient.changeVisualShape(
            objectUniqueId=wall_id,
            linkIndex=-1,
            textureUniqueId=tex_id,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )
        return wall_id

    return [
        _wall([0.01, 5, 5], [0, 0, 0], [0, np.pi / 2, 0]),
        _wall([0.01, 5, 5], [0, -2, 5], [np.pi / 2, 0, np.pi / 2]),
        _wall([0.01, 5, 5], [-5, 0, 5], [0, 0, 0]),
        _wall([0.01, 5, 5], [5, 0, 5], [0, 0, 0]),
        _wall([0.01, 5, 5], [0, 0, 10], [0, np.pi / 2, 0]),
    ]


class PyBUtils:
    def __init__(self, renders: bool = False) -> None:
        self.viz_view_matrix = None
        self.viz_proj_matrix = None
        self.renders = renders
        # self.cam_height = cam_height
        # self.cam_width = cam_width
        self.near_val = 0.02
        self.far_val = 4.0
        self.step_time = 1 / 240  #1/4 (old)

        # Debug parameters
        self.debug_items_step = []
        self.debug_items_reset = []

        self._setup_pybullet()
        return

    def _setup_pybullet(self) -> None:
        # New class for pybullet
        # obj_path=os.path.jo
        if self.renders:
            self.pbclient = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.pbclient = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.pbclient.setTimeStep(self.step_time)
        # self.enable_gravity()
        self.disable_gravity()
        self.pbclient.setRealTimeSimulation(False)

        # Set the viewing camera
        self.pbclient.resetDebugVisualizerCamera(
            cameraDistance=1.56,
            cameraYaw=-120.3,
            cameraPitch=-12.48,
            cameraTargetPosition=[-0.3, -0.06, 1.0],
        )
        if self.renders:
            self.configure_debug_visualizer_lighting()
        self.create_background()
        # self.setup_bird_view_visualizer()
        return

    def disable_gravity(self):
        self.pbclient.setGravity(0, 0, 0)
        log.info("Gravity disabled.")
        return

    def enable_gravity(self):
        self.pbclient.setGravity(0, 0, -grav)
        log.info(f"Gravity enabled ({-grav} m/s^2).")
        return

    def configure_debug_visualizer_lighting(self) -> None:
        """GUI tweaks supported by this PyBullet build (no light*Coeff kwargs)."""
        pb = self.pbclient
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        if hasattr(pb, "COV_ENABLE_SHADOWS"):
            pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        # Nov 2023 bindings accept lightPosition / shadowMapResolution only (not lightAmbientCoeff).
        pb.configureDebugVisualizer(lightPosition=[2.0, 3.0, 4.0])
        pb.configureDebugVisualizer(shadowMapResolution=2048)

    def reset_simulation_scene(
        self,
        gravity: float = 0.0,
        *,
        room_collision: bool = False,
        room_backdrop: bool = True,
    ) -> None:
        """Clear the world and optionally recreate the textured room backdrop.

        Call after PyBUtils() whenever code uses resetSimulation() before loading
        robot/tree (ApplePickingEnv, scene tests, etc.). resetSimulation removes
        bodies created in __init__, including the background from create_background().

        room_collision: when False (default for apple planning), backdrop is visual-only.
        room_backdrop: when False, skip create_background (robot-only debug in tree viewer).
        """
        self.pbclient.resetSimulation()
        self.pbclient.setGravity(0, 0, gravity)
        if self.renders:
            self.configure_debug_visualizer_lighting()
        if room_backdrop:
            self.create_background(collision=room_collision)
        else:
            log.info(
                "Room backdrop: disabled (room_backdrop=False); robot/tree only for debug"
            )

    def create_wall_with_texture(
        self,
        wall_dim: List,
        wall_pos: List,
        euler_rotation: List,
        wall_texture: int,
        *,
        collision: bool = True,
    ):
        wall_viz = self.pbclient.createVisualShape(
            shapeType=self.pbclient.GEOM_BOX,
            halfExtents=wall_dim,
        )
        wall_col = (
            self.pbclient.createCollisionShape(
                shapeType=self.pbclient.GEOM_BOX,
                halfExtents=wall_dim,
            )
            if collision
            else -1
        )
        wall_id = self.pbclient.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=wall_viz,
            baseCollisionShapeIndex=wall_col,
            basePosition=wall_pos,
            baseOrientation=list(self.pbclient.getQuaternionFromEuler(euler_rotation)),
        )
        self.pbclient.changeVisualShape(
            objectUniqueId=wall_id,
            linkIndex=-1,
            textureUniqueId=wall_texture,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )
        return wall_id

    def create_background(self, *, collision: bool = True) -> None:
        ids = create_room_backdrop(self.pbclient, collision=collision)
        self.floor_id = ids[0]
        self.wall_id = ids[1]
        self.side_wall_1_id = ids[2]
        self.side_wall_2_id = ids[3]
        self.ceil_id = ids[4]
        return

    def remove_debug_items(self, where) -> None:
        if where == "step":
            for item in self.debug_items_step:
                self.pbclient.removeUserDebugItem(item)
            self.debug_items_step = []
        elif where == "reset":
            for item in self.debug_items_reset:
                self.pbclient.removeUserDebugItem(item)
            self.debug_items_reset = []
        return

    def add_debug_item(self, type, where, **kwargs):
        debug_item = self.pbclient.addUserDebugLine(**kwargs)
        if where == "step":
            self.debug_items_step.append(debug_item)
        elif where == "reset":
            self.debug_items_reset.append(debug_item)
        return debug_item

    def remove_body(self, body_id: int) -> None:
        self.pbclient.removeBody(body_id)

    def add_sphere(self, radius: float, pos: List, rgba: List) -> int:
        colSphereId = -1
        visualShapeId = self.pbclient.createVisualShape(self.pbclient.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        sphereUid = self.pbclient.createMultiBody(0.0, colSphereId, visualShapeId, pos, [0, 0, 0, 1])
        return sphereUid

    # def setup_bird_view_visualizer(self):
    #     self.viz_view_matrix = self.pbclient.computeViewMatrixFromYawPitchRoll(
    #         cameraTargetPosition=[-0.3, -0.06, 1.3], distance=1.06, yaw=-120.3, pitch=-12.48, roll=0, upAxisIndex=2
    #     )
    #     self.viz_proj_matrix = self.pbclient.computeProjectionMatrixFOV(
    #         fov=60, aspect=float(8 / 8), nearVal=0.1, farVal=100.0
    #     )
    #     return

    @staticmethod
    def linearize_depth(depth: NDArray, far_val: float, near_val: float):
        """OpenGL returns contracted depth, linearize it"""
        try:
            depth_linearized = far_val * near_val / (far_val - (far_val - near_val) * depth)
        except ZeroDivisionError:
            log.warning("Encountered division by 0 in depth linearization.")
            depth_linearized = None
        return depth_linearized

    def visualize_points(self, points: List, type: str) -> None:
        dx = 0.1
        for point in points:
            if type == "curriculum":
                loc = point[1][0]
                branch = point[1][1]
                normal_vec = point[1][2]
            elif type == "reachable":
                loc = point[0]
                branch = point[1]
                normal_vec = point[2]
            #         self.debug_branch = self.pyb_con.con.addUserDebugLine(point,
            #                                                       point + 5 * normal_vec / np.linalg.norm(normal_vec),
            #                                                       [1, 0, 0], 2)

            self.add_debug_item(
                "sphere",
                "step",
                lineFromXYZ=[loc[0], loc[1], loc[2]],
                lineToXYZ=[loc[0] + 0.005, loc[1] + 0.005, loc[2] + 0.005],
                lineColorRGB=[1, 0, 0],
                lineWidth=200,
            )
            self.add_debug_item(
                "sphere",
                "step",
                lineFromXYZ=[
                    loc[0] - dx * branch[0],
                    loc[1] - dx * branch[1],
                    loc[2] - dx * branch[2],
                ],
                lineToXYZ=[
                    loc[0] + dx * branch[0],
                    loc[1] + dx * branch[1],
                    loc[2] + dx * branch[2],
                ],
                lineColorRGB=[1, 1, 0],
                lineWidth=200,
            )
        return

    def visualize_rot_mat(self, rot_mat: List, pos):
        # if rot_mat is Tuple:
        if isinstance(rot_mat, tuple) or len(rot_mat) == 4:
            rot_mat = np.array(self.pbclient.getMatrixFromQuaternion(rot_mat)).reshape(3, 3)
        dx = 0.1
        colors = np.eye(3)
        for i in range(3):
            self.add_debug_item(
                "sphere",
                "step",
                lineFromXYZ=[pos[0], pos[1], pos[2]],
                lineToXYZ=[
                    pos[0] + rot_mat[0][i] * dx,
                    pos[1] + rot_mat[1][i] * dx,
                    pos[2] + rot_mat[2][i] * dx,
                ],
                lineColorRGB=colors[i],
                lineWidth=200,
            )
        return
