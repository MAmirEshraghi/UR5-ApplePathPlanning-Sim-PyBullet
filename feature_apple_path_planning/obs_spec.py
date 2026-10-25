"""
Observation spec shared by ApplePickingEnv, agent-data I/O and the policy.

Image observation keys (all stored once per frame in agent-data schema v2):
    rgb         uint8   (3, H, W)
    prev_rgb    uint8   (3, H, W)   rgb of the previous step (= rgb at episode start);
                                    never stored, rebuilt from rgb[t-1] on load
    depth       uint16  (H, W)      metric depth in millimetres
    point_mask  uint8   (H, W)      1 inside a disc around the projected apple centre

Policy image channels are chosen at train time and map onto those keys.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

# Channel name -> env observation keys it needs, and channels it contributes to the CNN input.
CHANNEL_KEYS = {"flow": ("rgb", "prev_rgb"), "depth": ("depth",), "mask": ("point_mask",)}
CHANNEL_DIMS = {"flow": 2, "depth": 1, "mask": 1}
CHANNEL_ORDER = ("flow", "depth", "mask")
IMAGE_KEYS = frozenset({"rgb", "prev_rgb", "depth", "point_mask"})

DEPTH_UNITS_PER_M = 1000.0  # depth stored as uint16 millimetres
DEPTH_MAX_M = 2.0           # policy clips depth to [0, DEPTH_MAX_M] before normalising
FLOW_SCALE = 50.0           # RAFT flow (pixels / algo_size) has p95 ~0.015-0.02 on tree08 expert data -> ~[-1, 1]


def parse_image_channels(spec: Optional[Iterable[str] | str]) -> Tuple[str, ...]:
    """'depth,mask' / ['mask', 'depth'] / '' / 'none' -> canonical tuple, e.g. ('depth', 'mask')."""
    if spec is None:
        return ()
    if isinstance(spec, str):
        spec = [] if spec.strip().lower() in ("", "none") else spec.split(",")
    names = {s.strip().lower() for s in spec if s.strip()}
    unknown = names - set(CHANNEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown image channel(s) {sorted(unknown)}; choose from {CHANNEL_ORDER}")
    return tuple(c for c in CHANNEL_ORDER if c in names)


def image_keys_for_channels(channels: Iterable[str]) -> Tuple[str, ...]:
    keys: list[str] = []
    for c in parse_image_channels(list(channels)):
        keys.extend(k for k in CHANNEL_KEYS[c] if k not in keys)
    return tuple(keys)


def depth_m_to_mm(depth_m: np.ndarray) -> np.ndarray:
    return np.clip(np.round(depth_m * DEPTH_UNITS_PER_M), 0, np.iinfo(np.uint16).max).astype(np.uint16)
