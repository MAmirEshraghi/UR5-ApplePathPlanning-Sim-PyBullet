"""Shared PyBullet GUI debug markers (apple spheres, GUI refresh)."""

from __future__ import annotations

import numpy as np
from zenlog import log


def apple_marker_surface_position(
    apple_centroid: np.ndarray,
    robot_position: np.ndarray,
    surface_offset_m: float,
) -> np.ndarray:
    """Offset marker from apple centroid toward robot so it sits outside the mesh."""
    centroid = np.asarray(apple_centroid, dtype=np.float64).reshape(3)
    robot = np.asarray(robot_position, dtype=np.float64).reshape(3)
    delta = robot - centroid
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        # Fallback: nudge toward world +Y if robot coincides with centroid
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        direction = delta / norm
    return centroid + surface_offset_m * direction


def draw_debug_sphere(pb_client, position, radius, color_rgba):
    """Visual-only sphere; returns body id or -1 on failure."""
    try:
        pos = np.asarray(position, dtype=np.float64).reshape(3).tolist()
        visual_shape_id = pb_client.createVisualShape(
            shapeType=pb_client.GEOM_SPHERE,
            radius=float(radius),
            rgbaColor=list(color_rgba),
        )
        return pb_client.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=pos,
        )
    except Exception as exc:
        log.error(
            "VIZ_DEBUG sphere create failed pos=%s r=%.4f: %s",
            np.round(np.asarray(position, dtype=np.float64), 3).tolist(),
            radius,
            exc,
        )
        return -1


def change_sphere_color(pb_client, sphere_id, color_rgba):
    if sphere_id is not None and sphere_id >= 0:
        try:
            pb_client.changeVisualShape(sphere_id, -1, rgbaColor=list(color_rgba))
        except Exception as exc:
            log.error("VIZ_DEBUG change_sphere_color id=%s failed: %s", sphere_id, exc)


def gui_refresh(pb_client, renders: bool, steps: int = 1) -> None:
    """Step simulation so the GUI picks up new/updated bodies."""
    if not renders or steps <= 0:
        return
    for _ in range(int(steps)):
        pb_client.stepSimulation()


def log_apple_marker_batch(
    label: str,
    body_ids: list,
    centroids: list,
    marker_positions: list,
    radius: float,
    surface_offset_m: float,
) -> None:
    """One INFO line per marker batch for log validation."""
    ok_ids = [int(b) for b in body_ids if b is not None and int(b) >= 0]
    fail_n = len(body_ids) - len(ok_ids)
    if not centroids:
        log.info("VIZ_VALIDATE %s: no apples (skipped)", label)
        return
    c0 = np.asarray(centroids[0], dtype=np.float64).reshape(3)
    m0 = np.asarray(marker_positions[0], dtype=np.float64).reshape(3) if marker_positions else c0
    log.info(
        "VIZ_VALIDATE %s: created=%d failed=%d radius_m=%.3f surface_offset_m=%.3f "
        "sample_centroid=%s sample_marker_pos=%s",
        label,
        len(ok_ids),
        fail_n,
        radius,
        surface_offset_m,
        np.round(c0, 3).tolist(),
        np.round(m0, 3).tolist(),
    )
    if ok_ids:
        preview = ok_ids[:5]
        suffix = "" if len(ok_ids) <= 5 else f" ... +{len(ok_ids) - 5} more"
        log.info("VIZ_VALIDATE %s body_ids=%s%s", label, preview, suffix)


def log_scene_physics_summary(
    pb_client,
    *,
    context: str = "scene",
    tree_body_id: int | None = None,
    robot_body_id: int | None = None,
) -> None:
    """Log all bodies in the client so robot/tree/marker ids do not clash (grep VIZ_VALIDATE SCENE_BODIES)."""
    try:
        n = int(pb_client.getNumBodies())
    except Exception as exc:
        log.warning("VIZ_VALIDATE SCENE_BODIES context=%s failed: %s", context, exc)
        return
    body_ids: list[int] = []
    for i in range(n):
        try:
            body_ids.append(int(pb_client.getBodyUniqueId(i)))
        except Exception:
            pass
    tree_ok = tree_body_id is None or tree_body_id in body_ids
    robot_ok = robot_body_id is None or robot_body_id in body_ids
    dup = len(body_ids) != len(set(body_ids))
    status = "PASS" if tree_ok and robot_ok and not dup else "FAIL"
    log.info(
        "VIZ_VALIDATE SCENE_BODIES context=%s status=%s body_count=%d ids=%s "
        "tree_id=%s tree_in_scene=%s robot_id=%s robot_in_scene=%s",
        context,
        status,
        n,
        body_ids,
        tree_body_id,
        tree_ok,
        robot_body_id,
        robot_ok,
    )
    if dup:
        log.warning("VIZ_VALIDATE SCENE_BODIES context=%s duplicate body ids detected", context)


def log_viewer_scene_mode(
    *,
    with_robot: bool,
    room_backdrop: bool,
    gravity_z: float,
    hide_collision_visual: bool,
    preserve_mtl: bool,
    scale: float,
) -> None:
    """One-line scene recipe for A/B tests (grep VIZ_VALIDATE VIEWER_SCENE)."""
    log.info(
        "VIZ_VALIDATE VIEWER_SCENE with_robot=%s room_backdrop=%s gravity_z=%s "
        "hide_collision_visual=%s preserve_mtl=%s scale=%s",
        with_robot,
        room_backdrop,
        gravity_z,
        hide_collision_visual,
        preserve_mtl,
        scale,
    )
