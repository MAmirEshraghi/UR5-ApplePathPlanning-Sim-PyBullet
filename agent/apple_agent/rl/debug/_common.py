"""Shared helpers for feature_apple_agent/debug diagnostics."""
import datetime
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np
import torch as th

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from feature_apple_path_planning.apple_picking_env import ApplePickingEnv
from feature_apple_path_planning.run_apple_data_generator import CONFIG as APPLE_PICKING_CONFIG
from apple_picking_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAEWithExpert
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from feature_apple_path_planning import agent_data
from eval_apple_picking_expert import (
    _load_expert_metadata,
    checkpoint_image_channels,
    _reconfigure_to_trajectory,
    _vec_obs_from_env,
)
from train_apple_picking_expert import _apply_tree_setup_for_training, _log_train_data

LOG_DIR = Path("logs") / "debug"
JSON_DIR = LOG_DIR / "json"

TABULAR_OBS_KEYS = (
    "achieved_goal",
    "achieved_or",
    "desired_goal",
    "joint_angles",
    "relative_distance",
    "pregrasp_in_tool",
    "prev_action_achieved",
)
IMAGE_OBS_KEYS = ("rgb", "prev_rgb", "depth", "point_mask")


def agent_data_path(run_id: str) -> Path:
    return Path("output") / "agent_data" / f"{run_id}.hdf5"


def training_config(renders: bool = False, image_channels=None) -> dict:
    return {
        **APPLE_PICKING_CONFIG,
        "simulation_setup": {
            **APPLE_PICKING_CONFIG["simulation_setup"],
            "image_channels": image_channels,  # None = emit every image key
            "renders": renders,
            "draw_apple_centroid_markers": False,
            "draw_trajectory_target_markers": False,
        },
    }


def sample_episode_keys(run_id: str, n_episodes: int, seed: int) -> Tuple[List[str], Dict[str, dict]]:
    path = agent_data_path(run_id)
    trajectory_keys, metadata_cache = _load_expert_metadata(str(path.resolve()))
    rng = random.Random(seed)
    keys = rng.sample(trajectory_keys, min(n_episodes, len(trajectory_keys)))
    return keys, metadata_cache


def setup_env(run_id: str, renders: bool = False, checkpoint: str = None):
    """Env emitting the checkpoint's image keys (all keys when no checkpoint is given)."""
    path = agent_data_path(run_id)
    channels = checkpoint_image_channels(Path(checkpoint).resolve()) if checkpoint else None
    cfg = training_config(renders=renders, image_channels=channels)
    _apply_tree_setup_for_training(cfg, run_id, path)
    _log_train_data(path, cfg)
    env = make_vec_env(
        ApplePickingEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"config": cfg},
    )
    return env, cfg, path


def resolve_device(request: str = "auto") -> str:
    """Pick torch device; auto tries CUDA and falls back to CPU if busy/unavailable."""
    req = (request or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req not in ("auto", "cuda"):
        raise ValueError(f"device must be auto, cuda, or cpu — got {request!r}")

    if not th.cuda.is_available():
        if req == "cuda":
            raise RuntimeError(
                "CUDA requested but PyTorch cannot use CUDA (is_available()=False). "
                "Common cause: PyBullet initialized before CUDA — resolve device BEFORE setup_env(). "
                "If GPU looks free in nvidia-smi, retry; otherwise use --device cpu."
            )
        print("[debug:device] CUDA not available — using CPU", flush=True)
        return "cpu"

    try:
        th.cuda.empty_cache()
        probe = th.zeros(1, device="cuda")
        del probe
        th.cuda.synchronize()
        return "cuda"
    except Exception as exc:
        if req == "cuda":
            raise RuntimeError(
                "CUDA device busy or unavailable (another job may be using the GPU). "
                "Stop the other process or run with --device cpu."
            ) from exc
        print(
            f"[debug:device] CUDA busy/unavailable ({exc}) — falling back to CPU",
            flush=True,
        )
        return "cpu"


def load_policy(
    checkpoint: str,
    env,
    hdf5_path: Path,
    *,
    device: str = "auto",
):
    if device in ("cuda", "cpu"):
        resolved = device
    else:
        resolved = resolve_device(device)
    print(f"[debug:device] loading checkpoint on device={resolved}", flush=True)
    return RecurrentPPOAEWithExpert.load(
        str(Path(checkpoint).resolve()),
        env=env,
        use_online_data=False,
        use_offline_data=False,
        use_ppo_offline=False,
        use_online_bc=True,
        use_awac=False,
        path_trajectories=str(hdf5_path.resolve()),
        algo_size=(224, 224),
        bc_only=False,
        device=resolved,
    )


def hdf5_obs_step0(h5file: h5py.File, traj_key: str) -> Dict[str, np.ndarray]:
    grp = h5file[traj_key]
    frames = agent_data.load_frames(grp, agent_data.frame_keys(grp))
    return {k: v[0] for k, v in frames.items()}


def vec_obs_from_dict(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.stack([v]) for k, v in obs.items()}


def hdf5_obs_for_policy(model, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Stack HDF5 t=0 obs with the keys of the policy's observation space."""
    return {k: np.stack([np.asarray(raw_obs[k])]) for k in model.observation_space.spaces.keys()}


def predict_policy(model, obs: Dict[str, np.ndarray], **kwargs):
    return model.predict(obs, **kwargs)


def sim_obs_step0(env, metadata: dict) -> Dict[str, np.ndarray]:
    _reconfigure_to_trajectory(env, metadata)
    stacked = _vec_obs_from_env(env)
    return {k: stacked[k][0] for k in stacked}


def compare_obs_arrays(expert: np.ndarray, sim: np.ndarray, key: str) -> dict:
    e = np.asarray(expert)
    s = np.asarray(sim)
    out = {"key": key, "expert_shape": list(e.shape), "sim_shape": list(s.shape)}
    if e.shape != s.shape:
        out["shape_match"] = False
        return out
    out["shape_match"] = True
    diff = np.abs(e.astype(np.float64) - s.astype(np.float64))
    out["mean_abs_diff"] = float(diff.mean())
    out["max_abs_diff"] = float(diff.max())
    if key in TABULAR_OBS_KEYS:
        out["l2_diff"] = float(np.linalg.norm(e.astype(np.float64) - s.astype(np.float64)))
    return out


def compare_obs_pair(expert_obs: dict, sim_obs: dict) -> Tuple[List[dict], dict]:
    rows = []
    for key in sorted(set(expert_obs) | set(sim_obs)):
        if key not in expert_obs:
            rows.append({"key": key, "missing": "expert"})
            continue
        if key not in sim_obs:
            rows.append({"key": key, "missing": "sim"})
            continue
        rows.append(compare_obs_arrays(expert_obs[key], sim_obs[key], key))
    tab = [r for r in rows if r.get("key") in TABULAR_OBS_KEYS and r.get("shape_match")]
    mean_tab = float(np.mean([r["mean_abs_diff"] for r in tab])) if tab else float("nan")
    rgb_rows = [r for r in rows if r.get("key") in ("rgb", "prev_rgb") and r.get("shape_match")]
    mean_rgb = float(np.mean([r["mean_abs_diff"] for r in rgb_rows])) if rgb_rows else float("nan")
    summary = {
        "mean_tabular_mean_abs_diff": mean_tab,
        "mean_rgb_mean_abs_diff": mean_rgb,
        "expert_has_optical_flow": "optical_flow" in expert_obs,
        "sim_has_optical_flow": "optical_flow" in sim_obs,
    }
    return rows, summary


def action_l2(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    mse = float(np.mean((a - b) ** 2))
    l2 = float(np.linalg.norm(a - b))
    return mse, l2


def run_with_session_log(script_name: str, run_id: str, fn: Callable[[], dict]) -> dict:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (LOG_DIR / f"{ts}_{script_name}_{run_id}.log").resolve()
    json_path = (JSON_DIR / f"{ts}_{script_name}_{run_id}.json").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout

    class _Tee:
        def write(self, data):
            orig_stdout.write(data)
            log_fp.write(data)

        def flush(self):
            orig_stdout.flush()
            log_fp.flush()

    sys.stdout = _Tee()
    print(f"Session log: {log_path}", flush=True)
    try:
        payload = fn()
    finally:
        sys.stdout = orig_stdout
        log_fp.close()

    payload["log_file"] = str(log_path)
    payload["json_file"] = str(json_path)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"[debug:output] log={log_path}", file=orig_stdout, flush=True)
    print(f"[debug:output] json={json_path}", file=orig_stdout, flush=True)
    return payload
