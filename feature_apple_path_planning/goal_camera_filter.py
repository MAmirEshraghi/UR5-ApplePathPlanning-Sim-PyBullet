"""
Pre-RRT goal-only camera filter (Method A: RGB color mask + circular ROI).

Independent from ``visibility_check`` inside ``is_state_valid_fn``. Intended to run
once per IK candidate at q_goal before spending effort on RRT.

Improve later with depth gating (Method B) or scoring in the projection module only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from zenlog import log

from pybullet_tree_sim.robot import Robot


@dataclass
class GoalCameraFilterResult:
    passed: bool
    visible_frac: float
    min_required_frac: float
    u: int | None
    v: int | None
    r_px: float
    blue_in_disk_px: int
    disk_area_px: float
    reason: str
    # Filled for debugging projection / false off-screen investigations
    ndc_xyz: tuple[float, float, float] | None = None
    clip_div_w: float | None = None
    cam_eye_world: tuple[float, float, float] | None = None
    dist_cam_apple_m: float | None = None
    debug_image_paths: tuple[str, ...] = ()


@dataclass
class GoalCameraFilterDebugContext:
    """Optional debug context; when set, step PNGs are written under ``output_dir``."""

    output_dir: Path
    apple_id: int
    ori_id: int
    run_id: str | None = None
    saved_paths: list[str] = field(default_factory=list)


def _as_ndc_tuple(ndc: np.ndarray | None) -> tuple[float, float, float] | None:
    if ndc is None:
        return None
    return (float(ndc[0]), float(ndc[1]), float(ndc[2]))


def _world_to_pixel(
    view_matrix,
    proj_matrix_tuple,
    world_xyz: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int] | None, np.ndarray | None, str, float | None]:
    """Same NDC convention as ApplePickingEnv._compute_deprojected_point_mask."""
    view_matrix_np = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
    proj_matrix_np = np.asarray(proj_matrix_tuple, dtype=np.float64).reshape(4, 4, order="F")
    w = np.array([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0], dtype=np.float64)
    clip = proj_matrix_np @ view_matrix_np @ w
    clip_w = float(clip[3])
    if abs(clip_w) < 1e-12:
        return None, None, "clip_w_zero", clip_w
    ndc = clip[:3] / clip_w
    if not (-1.0 <= float(ndc[0]) <= 1.0 and -1.0 <= float(ndc[1]) <= 1.0):
        return None, ndc, "off_screen", clip_w
    if not (-1.0 <= float(ndc[2]) <= 1.0):
        return None, ndc, "behind_or_clipped_z", clip_w
    u = int(round(((ndc[0] + 1.0) / 2.0) * width))
    v = int(round(((1.0 - ndc[1]) / 2.0) * height))
    if not (0 <= u < width and 0 <= v < height):
        return None, ndc, "pixel_oob", clip_w
    return (u, v), ndc, "ok", clip_w


def _camera_world_position(view_matrix) -> np.ndarray:
    """Eye position in world frame from PyBullet view matrix."""
    v = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
    inv_v = np.linalg.inv(v)
    return np.asarray(inv_v[:3, 3], dtype=np.float64).reshape(3)


def _rgb_to_bgr(rgb_u8: np.ndarray) -> np.ndarray:
    if rgb_u8.ndim == 2:
        return cv2.cvtColor(rgb_u8, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def _sanitize_reason(reason: str) -> str:
    return reason.replace("/", "_").replace(":", "_").replace(" ", "_")[:48]


def _gcam_debug_filename(
    ctx: GoalCameraFilterDebugContext,
    step: str,
    passed: bool | None,
    reason: str,
) -> Path:
    apple = f"apple{ctx.apple_id:02d}"
    ori = f"ori{ctx.ori_id:02d}"
    if passed is True:
        verdict = "PASS"
    elif passed is False:
        verdict = "FAIL"
    else:
        verdict = "RAW"
    return ctx.output_dir / f"{apple}_{ori}_{step}_{verdict}_{_sanitize_reason(reason)}.png"


def _draw_text_banner(img_bgr: np.ndarray, lines: list[str], passed: bool | None) -> None:
    if passed is True:
        color = (0, 200, 0)
    elif passed is False:
        color = (0, 0, 255)
    else:
        color = (220, 220, 220)
    y = 20
    for line in lines:
        cv2.putText(img_bgr, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 18


def _gcam_save_step(
    ctx: GoalCameraFilterDebugContext,
    step: str,
    rgb_u8: np.ndarray,
    passed: bool | None,
    reason: str,
    label_lines: list[str] | None = None,
) -> str | None:
    try:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = _gcam_debug_filename(ctx, step, passed, reason)
        img_bgr = _rgb_to_bgr(rgb_u8)
        lines = list(label_lines or [])
        if passed is True:
            lines.insert(0, f"PASS {reason}")
        elif passed is False:
            lines.insert(0, f"FAIL {reason}")
        else:
            lines.insert(0, reason)
        _draw_text_banner(img_bgr, lines, passed)
        if not cv2.imwrite(str(out_path), img_bgr):
            log.warning("goal_camera_filter: failed to write debug image %s", out_path)
            return None
        path_str = str(out_path.resolve())
        ctx.saved_paths.append(path_str)
        return path_str
    except Exception as exc:
        log.warning("goal_camera_filter: debug save error step=%s: %s", step, exc)
        return None


def _gcam_overlay_projection(
    rgb_u8: np.ndarray,
    u: int | None,
    v: int | None,
    ndc_t: tuple[float, float, float] | None,
    preason: str,
    projection_ok: bool,
) -> np.ndarray:
    overlay = rgb_u8.copy()
    if projection_ok and u is not None and v is not None:
        cv2.drawMarker(
            overlay, (u, v), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2
        )
        cv2.circle(overlay, (u, v), 4, (255, 255, 0), thickness=1)
    return overlay


def _gcam_overlay_disk(
    rgb_u8: np.ndarray,
    u: int,
    v: int,
    r_px: float,
    vis_mask: np.ndarray,
    passed: bool,
    reason: str,
    visible_frac: float,
    min_frac: float,
    blue_in_disk: int,
) -> np.ndarray:
    overlay = rgb_u8.copy()
    disk_color = (0, 255, 0) if passed else (0, 0, 255)
    cv2.circle(overlay, (u, v), int(round(r_px)), disk_color, thickness=2)
    cv2.drawMarker(
        overlay, (u, v), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1
    )
    match_idx = vis_mask > 0
    if np.any(match_idx):
        tint = overlay.copy()
        tint[match_idx] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0.0)
    return overlay


def evaluate_goal_camera_filter_method_a(
    robot: Robot,
    pb_client,
    camera_sensor,
    apple_center_world: np.ndarray,
    q_goal,
    cfg: dict,
    debug: GoalCameraFilterDebugContext | None = None,
) -> GoalCameraFilterResult:
    """
    Move robot to q_goal, render RGB, measure blue pixels inside projected disk.

    Parameters
    ----------
    cfg
        ``goal_camera_filter`` dict from planning CONFIG. Expected keys:
        min_visible_frac, color_rgb_lower, color_rgb_upper, sphere_radius_m,
        disk_radius_scale, sim_settle_steps,
        log_projection_debug (enable ``log.debug`` projection lines and extra INFO fields).
    debug
        When set, writes step PNGs under ``debug.output_dir`` (s01 rgb, s02 projection, s03 disk).
    """
    min_frac = float(cfg.get("min_visible_frac", 0.8))
    r_world = float(cfg.get("sphere_radius_m", 0.05))
    disk_scale = float(cfg.get("disk_radius_scale", 1.15))
    settle = max(0, int(cfg.get("sim_settle_steps", 2)))
    log_projection_detail = bool(cfg.get("log_projection_debug", True))

    lower = np.array(cfg.get("color_rgb_lower", (0, 0, 140)), dtype=np.uint8)
    upper = np.array(cfg.get("color_rgb_upper", (80, 80, 255)), dtype=np.uint8)

    height = int(camera_sensor.depth_height)
    width = int(camera_sensor.depth_width)
    vfov_deg = float(camera_sensor.depth_vfov)
    vfov_rad = np.deg2rad(vfov_deg)
    fy = (height / 2.0) / max(np.tan(vfov_rad / 2.0), 1e-9)

    debug_paths: list[str] = []
    rgb_u8: np.ndarray | None = None

    q_save = robot.get_joint_angles()
    try:
        robot.set_joint_angles_no_collision(q_goal)
        # for _ in range(settle):
        #     pb_client.stepSimulation()
            
        import time
        time.sleep(0.8)   # 100 ms pause so GUI shows teleported pose
        # pb_client.stepSimulation()

        view_matrix = robot.get_view_mat_at_curr_pose(camera_sensor)
        rgb_f, _depth = robot.get_rgbd_at_cur_pose(
            camera=camera_sensor, type="sensor", view_matrix=view_matrix
        )
        rgb_u8 = (np.clip(np.asarray(rgb_f, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)

        if debug is not None:
            p = _gcam_save_step(debug, "s01_rgb", rgb_u8, None, "raw", ["s01: wrist RGB"])
            if p:
                debug_paths.append(p)

        ac = np.asarray(apple_center_world, dtype=np.float64).reshape(3)
        uv, ndc_arr, preason, clip_w = _world_to_pixel(
            view_matrix, camera_sensor.depth_proj_mat, ac, width, height
        )
        cam_pos = _camera_world_position(view_matrix)
        dist_ca = float(np.linalg.norm(ac - cam_pos))
        ndc_t = _as_ndc_tuple(ndc_arr)
        eye_t = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        dbg = dict(
            ndc_xyz=ndc_t,
            clip_div_w=clip_w,
            cam_eye_world=eye_t,
            dist_cam_apple_m=dist_ca,
        )

        projection_ok = uv is not None
        if uv is not None:
            u_proj, v_proj = uv
        else:
            u_proj, v_proj = None, None

        if debug is not None:
            proj_overlay = _gcam_overlay_projection(
                rgb_u8, u_proj, v_proj, ndc_t, preason, projection_ok
            )
            proj_lines = [f"s02: projection reason={preason}"]
            if ndc_t is not None:
                proj_lines.append(f"ndc=({ndc_t[0]:.3f},{ndc_t[1]:.3f},{ndc_t[2]:.3f})")
            if u_proj is not None and v_proj is not None:
                proj_lines.append(f"uv=({u_proj},{v_proj})")
            p = _gcam_save_step(
                debug,
                "s02_proj",
                proj_overlay,
                projection_ok,
                preason,
                proj_lines,
            )
            if p:
                debug_paths.append(p)

        if uv is None:
            if log_projection_detail:
                log.debug(
                    "goal_camera_filter projection_fail reason=%s ndc=%s clip_div_w=%s "
                    "dist_cam_apple_m=%.5f cam_eye=%s apple_center=%s image=%dx%d vfov_deg=%.2f",
                    preason,
                    ndc_t,
                    clip_w,
                    dist_ca,
                    np.round(cam_pos, 5).tolist(),
                    np.round(ac, 5).tolist(),
                    width,
                    height,
                    vfov_deg,
                )
            return GoalCameraFilterResult(
                False, 0.0, min_frac, None, None, 0.0, 0, 0.0, preason, **dbg,
                debug_image_paths=tuple(debug_paths),
            )

        u, v = uv
        if dist_ca < 1e-6:
            if log_projection_detail:
                log.debug(
                    "goal_camera_filter zero_dist cam_eye=%s apple_center=%s",
                    np.round(cam_pos, 5).tolist(),
                    np.round(ac, 5).tolist(),
                )
            return GoalCameraFilterResult(
                False, 0.0, min_frac, u, v, 0.0, 0, 0.0, "zero_dist", **dbg,
                debug_image_paths=tuple(debug_paths),
            )

        dist = dist_ca
        r_px = fy * r_world / dist * disk_scale
        r_px = float(np.clip(r_px, 2.0, float(max(width, height)) * 2.0))

        mask_blue = cv2.inRange(rgb_u8, lower, upper)
        disk_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(disk_mask, (u, v), int(round(r_px)), 255, thickness=-1)
        vis = cv2.bitwise_and(mask_blue, disk_mask)

        blue_in_disk = int(cv2.countNonZero(vis))
        disk_area_px = float(np.pi * (r_px**2))
        visible_frac = blue_in_disk / max(disk_area_px, 1.0)
        passed = bool(visible_frac >= min_frac)
        reason = "ok" if passed else "below_min_frac"
        if log_projection_detail and not passed:
            log.debug(
                "goal_camera_filter color_fail reason=%s visible_frac=%.4f min_req=%.4f "
                "uv=(%s,%s) r_px=%.2f blue_px=%d ndc=%s dist_cam=%.4f",
                reason,
                visible_frac,
                min_frac,
                u,
                v,
                r_px,
                blue_in_disk,
                ndc_t,
                dist_ca,
            )

        if debug is not None:
            disk_overlay = _gcam_overlay_disk(
                rgb_u8, u, v, r_px, vis, passed, reason, visible_frac, min_frac, blue_in_disk
            )
            disk_lines = [
                f"s03: disk r_px={r_px:.1f}",
                f"frac={visible_frac:.4f} min={min_frac:.4f}",
                f"blue_px={blue_in_disk}",
            ]
            p = _gcam_save_step(debug, "s03_disk", disk_overlay, passed, reason, disk_lines)
            if p:
                debug_paths.append(p)

        return GoalCameraFilterResult(
            passed,
            visible_frac,
            min_frac,
            u,
            v,
            r_px,
            blue_in_disk,
            disk_area_px,
            reason,
            **dbg,
            debug_image_paths=tuple(debug_paths),
        )
    except Exception as e:  # pragma: no cover
        log.warning("goal_camera_filter: exception %s", e, exc_info=True)
        if debug is not None and rgb_u8 is not None:
            p = _gcam_save_step(
                debug, "s01_rgb", rgb_u8, False, f"error_{e!s}", [f"exception: {e!s}"]
            )
            if p:
                debug_paths.append(p)
        return GoalCameraFilterResult(
            False, 0.0, min_frac, None, None, 0.0, 0, 0.0, f"error:{e!s}",
            debug_image_paths=tuple(debug_paths),
        )
    finally:
        robot.set_joint_angles_no_collision(q_save)
