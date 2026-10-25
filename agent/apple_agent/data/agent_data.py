"""
Agent-data HDF5 I/O — schema v2. One HDF5 group per expert trajectory:

    <group>/frames/<obs_key>  (T+1, ...)  observation at every frame; frame 0 = episode start
    <group>/actions           (T, A)      float32, action applied between frame t and t+1
    <group>/rewards           (T,)        float64
    <group>/terminals         (T,)        bool, env `terminated` after action t
    <group>/truncateds        (T,)        bool, env `truncated` after action t
    <group>/step_info/<key>   (T,)        per-step env info (reward relabelling later)
    attrs: schema_version, trajectory metadata (tree_*, apple_center, goal_pos, goal_orn,
           initial_joint_angles, camera_*, render_mode, ...)

`prev_rgb` is not stored: the previous image of frame t is rgb[t-1] (rgb[0] at t=0), which is
exactly what ApplePickingEnv emits. Transition t is (frames[t], actions[t], rewards[t], frames[t+1]),
and the end of a trajectory is always an episode boundary. Storing each frame once makes
observation / next-observation / previous-frame inconsistencies impossible by construction.
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import h5py
import numpy as np
from filelock import FileLock

SCHEMA_VERSION = 2
STEP_INFO_KEYS = ("dist_to_goal", "collisions_unacceptable")


def check_schema(grp: h5py.Group) -> None:
    version = int(grp.attrs.get("schema_version", 1))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{grp.file.filename}:{grp.name} has agent-data schema v{version}, expected v{SCHEMA_VERSION}. "
            "Regenerate it with python -m apple_agent.data.generate_agent_data_from_waypoints_2."
        )


def write_trajectory(
    path: str,
    group_key: str,
    frames: Sequence[Dict[str, np.ndarray]],
    actions: Sequence[np.ndarray],
    rewards: Sequence[float],
    terminals: Sequence[bool],
    truncateds: Sequence[bool],
    step_info: Dict[str, Sequence],
    attrs: Dict,
) -> str:
    """Validate and append one trajectory; returns the HDF5 group name actually used."""
    n = len(actions)
    if n == 0 or len(frames) != n + 1:
        raise ValueError(f"need T>0 actions and T+1 frames, got {n} actions / {len(frames)} frames")
    if not (len(rewards) == len(terminals) == len(truncateds) == n):
        raise ValueError("rewards / terminals / truncateds length != number of actions")
    if any(set(fr) != set(frames[0]) for fr in frames):
        raise ValueError("inconsistent observation keys across frames")
    keys = sorted(k for k in frames[0] if k != "prev_rgb")
    stacked = {}
    for k in keys:
        arr = np.stack([np.asarray(fr[k]) for fr in frames])
        if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            raise ValueError(f"non-finite values in frames/{k}")
        stacked[k] = arr
    actions_arr = np.asarray(actions, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    if not (np.all(np.isfinite(actions_arr)) and np.all(np.isfinite(rewards_arr))):
        raise ValueError("non-finite actions or rewards")

    with FileLock(path + ".lock"), h5py.File(path, "a") as f:
        for existing in f.values():
            check_schema(existing)  # never mix schema versions in one file
        name, dup = group_key, 0
        while name in f:
            dup += 1
            name = f"{group_key}__dup{dup}"
        grp = f.create_group(name)
        for k, v in attrs.items():
            grp.attrs[k] = v
        grp.attrs["schema_version"] = SCHEMA_VERSION
        grp.attrs["hdf5_group_key"] = name
        fgrp = grp.create_group("frames")
        for k, arr in stacked.items():
            fgrp.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
        grp.create_dataset("actions", data=actions_arr)
        grp.create_dataset("rewards", data=rewards_arr)
        grp.create_dataset("terminals", data=np.asarray(terminals, dtype=bool))
        grp.create_dataset("truncateds", data=np.asarray(truncateds, dtype=bool))
        igrp = grp.create_group("step_info")
        for k, v in step_info.items():
            igrp.create_dataset(k, data=np.asarray(v))
    return name


def trajectory_metadata(grp: h5py.Group) -> dict:
    """Group attrs as a dict (what ApplePickingEnv.reconfigure_scene expects)."""
    check_schema(grp)
    return {k: grp.attrs[k] for k in grp.attrs.keys()}


def load_frames(grp: h5py.Group, keys: Iterable[str]) -> Dict[str, np.ndarray]:
    """Read the requested observation keys for all T+1 frames, rebuilding prev_rgb from rgb."""
    check_schema(grp)
    keys = list(keys)
    fgrp = grp["frames"]
    out = {k: fgrp[k][:] for k in keys if k != "prev_rgb"}
    if "prev_rgb" in keys:
        rgb = out["rgb"] if "rgb" in out else fgrp["rgb"][:]
        out["prev_rgb"] = np.concatenate([rgb[:1], rgb[:-1]], axis=0)
    return out


def iter_transitions(
    grp: h5py.Group, keys: Iterable[str]
) -> Iterator[Tuple[dict, np.ndarray, float, bool, dict]]:
    """Yield (obs_t, action_t, reward_t, episode_end_t, obs_t+1) for one trajectory."""
    frames = load_frames(grp, keys)
    actions = grp["actions"][:]
    rewards = grp["rewards"][:]
    n = len(actions)
    for t in range(n):
        obs = {k: v[t] for k, v in frames.items()}
        nxt = {k: v[t + 1] for k, v in frames.items()}
        yield obs, actions[t], float(rewards[t]), t == n - 1, nxt


def frame_keys(grp: h5py.Group) -> List[str]:
    """Observation keys available for this trajectory (prev_rgb is derived from rgb)."""
    check_schema(grp)
    keys = list(grp["frames"].keys())
    return keys + ["prev_rgb"] if "rgb" in keys else keys
