#!/usr/bin/env python3
"""
Visualize generated agent-data HDF5 trajectories.

Run from repository root:

    PYTHONPATH=. python3 feature_apple_path_planning/playground/visualize_agent_data_hdf5.py \
        --run-id 20260504_203032

Optional controls:

    --group-index 0           # select trajectory by sorted index
    --group-key <hdf5_key>    # select exact trajectory key
    --fps 12                  # autoplay speed
    --no-overlay-mask         # show raw RGB only

Keyboard:
    Right / Left : next / previous frame
    Space         : toggle autoplay
    N             : toggle obs <-> next_obs source
    M             : toggle point-mask overlay
    Up / Down     : next / previous trajectory group
    Home / End    : first / last frame
    Q or Esc      : quit
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import logging
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from zenlog import log

_FP_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _FP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = importlib.util.spec_from_file_location(
    "apple_data_generator_72",
    _FP_ROOT / "run_apple_data_generator.py",
)
dg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dg)

WINDOW_NAME = "agent_data_hdf5_visualizer"


def setup_logging() -> Path:
    base = _REPO_ROOT / "logs" / dg.LOG_SESSION_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (base / f"{ts}_agent_data_visualizer.log").resolve()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    zen_logger = logging.getLogger("pythonConfig")
    zen_logger.addHandler(fh)
    zen_logger.setLevel(logging.DEBUG)
    log.info("Agent-data visualizer log file: %s", log_path)
    return log_path


def sort_group_keys(keys: list[str]) -> list[str]:
    def _sort_key(k: str):
        try:
            return (0, float(k))
        except ValueError:
            return (1, str(k))

    return sorted(keys, key=_sort_key)


def resolve_hdf5_path(run_id: str | None, hdf5_path: str | None) -> Path:
    if hdf5_path:
        p = Path(hdf5_path).expanduser().resolve()
    elif run_id:
        p = (_REPO_ROOT / "output" / "agent_data" / f"{run_id}.hdf5").resolve()
    else:
        raise ValueError("Provide either --run-id or --hdf5-path.")

    if not p.exists():
        raise FileNotFoundError(f"Agent data HDF5 not found: {p}")
    return p


def to_bgr_u8_chw_or_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")

    # (C, H, W) -> (H, W, C)
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # Ensure 3 channels
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        # Heuristic: if already [0,1], scale to [0,255].
        if float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def mask_to_u8(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {m.shape}")
    if np.issubdtype(m.dtype, np.floating):
        m = np.clip(m, 0.0, 1.0) * 255.0
    else:
        m = (m > 0).astype(np.uint8) * 255
    return m.astype(np.uint8)


def draw_text_lines(canvas: np.ndarray, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        y += 22


def read_group_payload(f: h5py.File, group_key: str) -> dict:
    grp = f[group_key]
    payload: dict[str, object] = {"attrs": dict(grp.attrs.items())}

    payload["actions"] = np.asarray(grp["actions"]) if "actions" in grp else None
    payload["rewards"] = np.asarray(grp["rewards"]) if "rewards" in grp else None
    payload["dones"] = np.asarray(grp["dones"]) if "dones" in grp else None

    obs = {}
    if "observations" in grp:
        for k in grp["observations"].keys():
            obs[k] = np.asarray(grp["observations"][k])
    next_obs = {}
    if "next_observations" in grp:
        for k in grp["next_observations"].keys():
            next_obs[k] = np.asarray(grp["next_observations"][k])
    payload["observations"] = obs
    payload["next_observations"] = next_obs

    # Determine canonical trajectory length.
    length = 0
    if payload["actions"] is not None:
        length = int(payload["actions"].shape[0])
    elif obs:
        some_key = next(iter(obs.keys()))
        length = int(obs[some_key].shape[0])
    payload["length"] = length
    return payload


def format_attr(attrs: dict, key: str) -> str:
    if key not in attrs:
        return "n/a"
    v = attrs[key]
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return str(v.reshape(-1)[0])
        return f"array(shape={v.shape})"
    return str(v)


def make_frame(
    payload: dict,
    frame_idx: int,
    group_key: str,
    group_index: int,
    group_total: int,
    use_next_obs: bool,
    overlay_mask: bool,
    fps: float,
    autoplay: bool,
) -> np.ndarray:
    obs_dict = payload["next_observations"] if use_next_obs else payload["observations"]
    if "rgb" not in obs_dict:
        raise KeyError("This trajectory does not contain observations/rgb.")

    rgb = obs_dict["rgb"][frame_idx]
    image = to_bgr_u8_chw_or_hwc(rgb)

    mask_panel = np.zeros_like(image)
    mask_sum = 0.0
    if "point_mask" in obs_dict:
        pm = mask_to_u8(obs_dict["point_mask"][frame_idx])
        mask_sum = float(np.sum(pm > 0))
        heat = cv2.applyColorMap(pm, cv2.COLORMAP_SUMMER)
        mask_panel = heat
        if overlay_mask:
            image = cv2.addWeighted(image, 0.75, heat, 0.25, 0.0)

    canvas = np.hstack([image, mask_panel])

    actions = payload["actions"]
    rewards = payload["rewards"]
    dones = payload["dones"]
    attrs = payload["attrs"]
    action_norm = float(np.linalg.norm(actions[frame_idx])) if actions is not None else float("nan")
    reward = float(rewards[frame_idx]) if rewards is not None else float("nan")
    done = bool(dones[frame_idx]) if dones is not None else False

    rel_dist = "n/a"
    if "relative_distance" in obs_dict:
        rd = np.asarray(obs_dict["relative_distance"][frame_idx], dtype=np.float64).reshape(-1)
        rel_dist = f"[{rd[0]:.4f}, {rd[1]:.4f}, {rd[2]:.4f}]"

    lines = [
        f"group {group_index + 1}/{group_total}: {group_key}",
        f"frame {frame_idx + 1}/{payload['length']} | source={'next_obs' if use_next_obs else 'obs'}",
        f"reward={reward:.6f} done={done} action_norm={action_norm:.6f} mask_px={mask_sum:.0f}",
        f"success_attr={format_attr(attrs, 'success')} fail_mode={format_attr(attrs, 'fail_mode')}",
        f"relative_distance={rel_dist}",
        f"autoplay={'ON' if autoplay else 'OFF'} fps={fps:.1f} overlay_mask={'ON' if overlay_mask else 'OFF'}",
        "keys: <- -> step | space play | n obs/next | m mask | up/down group | home/end | q/esc",
    ]
    draw_text_lines(canvas, lines)
    return canvas


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize generated agent-data HDF5 trajectories.")
    ap.add_argument("--run-id", type=str, default=None, help="Run ID for output/agent_data/<run_id>.hdf5")
    ap.add_argument("--hdf5-path", type=str, default=None, help="Explicit path to agent-data HDF5")
    ap.add_argument("--group-index", type=int, default=0, help="Sorted trajectory index to open first")
    ap.add_argument("--group-key", type=str, default=None, help="Exact trajectory group key to open first")
    ap.add_argument("--fps", type=float, default=12.0, help="Autoplay FPS")
    ap.add_argument("--window-name", type=str, default=WINDOW_NAME, help="OpenCV window title")
    ap.add_argument("--no-overlay-mask", action="store_true", help="Disable mask overlay on RGB by default")
    return ap.parse_args()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def main() -> int:
    args = parse_args()
    setup_logging()

    h5_path = resolve_hdf5_path(args.run_id, args.hdf5_path)
    log.info("Opening agent-data file: %s", h5_path)

    with h5py.File(h5_path, "r") as f:
        group_keys = sort_group_keys(list(f.keys()))
        if not group_keys:
            raise RuntimeError(f"No trajectory groups found in {h5_path}")

        if args.group_key is not None:
            if args.group_key not in f:
                raise KeyError(f"group-key {args.group_key!r} not found in file.")
            current_group_idx = group_keys.index(args.group_key)
        else:
            current_group_idx = clamp(int(args.group_index), 0, len(group_keys) - 1)

        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        overlay_mask = not bool(args.no_overlay_mask)
        use_next_obs = False
        autoplay = False
        frame_idx = 0
        last_tick = time.perf_counter()

        while True:
            gk = group_keys[current_group_idx]
            payload = read_group_payload(f, gk)
            traj_len = int(payload["length"])
            if traj_len <= 0:
                raise RuntimeError(f"Group {gk!r} has no frames.")
            frame_idx = clamp(frame_idx, 0, traj_len - 1)

            canvas = make_frame(
                payload=payload,
                frame_idx=frame_idx,
                group_key=gk,
                group_index=current_group_idx,
                group_total=len(group_keys),
                use_next_obs=use_next_obs,
                overlay_mask=overlay_mask,
                fps=max(0.1, float(args.fps)),
                autoplay=autoplay,
            )
            cv2.imshow(args.window_name, canvas)

            if autoplay:
                dt = time.perf_counter() - last_tick
                step_every = 1.0 / max(0.1, float(args.fps))
                wait_ms = max(1, int((step_every - dt) * 1000))
            else:
                wait_ms = 0

            key = cv2.waitKeyEx(wait_ms)
            now = time.perf_counter()
            if autoplay and (now - last_tick) >= (1.0 / max(0.1, float(args.fps))):
                frame_idx = clamp(frame_idx + 1, 0, traj_len - 1)
                last_tick = now

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord(" "),):
                autoplay = not autoplay
                last_tick = time.perf_counter()
            elif key in (ord("n"), ord("N")):
                use_next_obs = not use_next_obs
            elif key in (ord("m"), ord("M")):
                overlay_mask = not overlay_mask
            elif key in (83, 65363):  # right
                frame_idx = clamp(frame_idx + 1, 0, traj_len - 1)
            elif key in (81, 65361):  # left
                frame_idx = clamp(frame_idx - 1, 0, traj_len - 1)
            elif key in (82, 65362):  # up
                current_group_idx = clamp(current_group_idx + 1, 0, len(group_keys) - 1)
                frame_idx = 0
            elif key in (84, 65364):  # down
                current_group_idx = clamp(current_group_idx - 1, 0, len(group_keys) - 1)
                frame_idx = 0
            elif key in (80, 65360):  # home
                frame_idx = 0
            elif key in (87, 65367):  # end
                frame_idx = traj_len - 1

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
