import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))


# Global URDF path pointing to robot and supports URDFs
PKL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "pkl"))
MESHES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "meshes"))
URDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "urdf"))
TEXTURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "textures"))
ROBOT_URDF_PATH = os.path.join(URDF_PATH, "ur5e", "ur5e_cutter_new_calibrated_precise_level.urdf")
CONFIG_PATH = os.path.join(os.path.join(os.path.dirname(__file__), "config"))
CAMERAS_PATH = os.path.join(CONFIG_PATH, "cameras")
TOFS_PATH = os.path.join(CONFIG_PATH, "tofs")


def _rgb255_to_norm(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


def _label_color_slot(slot: int) -> int:
    """Match lpy_treesim Camp_Envy c(slot): slot * 0.1 * 255 rounded."""
    return int(round(slot * 0.1 * 255))


# Camp_Envy_tie_prune_label6_2 — wood/leaf/stem: (0, 0, B) only
RGB_LABEL = {
    _rgb255_to_norm(0, 0, _label_color_slot(1)): "TRUNK",
    _rgb255_to_norm(0, 0, _label_color_slot(2)): "BRANCH",
    _rgb255_to_norm(0, 0, _label_color_slot(3)): "WATER_BRANCH",
    _rgb255_to_norm(0, 0, _label_color_slot(4)): "SPUR",
    _rgb255_to_norm(0, 0, _label_color_slot(5)): "LEAF",
    _rgb255_to_norm(0, 0, _label_color_slot(6)): "STEM",
}

# Apples: 200 IDs — same order as Camp_Envy_tie_prune_label6_2.lpy _apple_label_colors()
APPLE_LABEL_LEVELS = (51, 102, 153, 204, 255)
APPLE_LABEL_LEVELS_NORM = tuple(v / 255.0 for v in APPLE_LABEL_LEVELS)
APPLE_LEVEL_TOLERANCE_NORM = 8.0 / 255.0
APPLE_ZERO_CHANNEL_MAX_NORM = 20.0 / 255.0


def _apple_label_colors_255() -> tuple[tuple[int, int, int], ...]:
    L = APPLE_LABEL_LEVELS
    out: list[tuple[int, int, int]] = []
    for r in L:
        for g in L:
            out.append((r, g, 0))
    for g in L:
        for b in L:
            out.append((0, g, b))
    for r in L:
        for b in L:
            out.append((r, 0, b))
    for r in L:
        for g in L:
            for b in L:
                out.append((r, g, b))
    return tuple(out)


APPLE_COLORS_255 = _apple_label_colors_255()
APPLE_COLORS_NORMALIZED = tuple(_rgb255_to_norm(r, g, b) for r, g, b in APPLE_COLORS_255)
_APPLE_INDEX_BY_RGB = {APPLE_COLORS_NORMALIZED[i]: i for i in range(len(APPLE_COLORS_NORMALIZED))}


def _at_apple_level_norm(value: float) -> bool:
    for level in APPLE_LABEL_LEVELS_NORM:
        if abs(value - level) <= APPLE_LEVEL_TOLERANCE_NORM:
            return True
    return False


def is_apple_label_normalized(rgb) -> bool:
    """75 two-channel + 125 full RGB on label levels; excludes wood (0,0,B)."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    if r <= APPLE_ZERO_CHANNEL_MAX_NORM and g <= APPLE_ZERO_CHANNEL_MAX_NORM:
        return False
    zero_ch = sum(v <= APPLE_ZERO_CHANNEL_MAX_NORM for v in (r, g, b))
    on_level = (_at_apple_level_norm(r), _at_apple_level_norm(g), _at_apple_level_norm(b))
    if zero_ch == 1:
        return sum(on_level) == 2
    if zero_ch == 0:
        return all(on_level)
    return False


def _snap_apple_channel_norm(value: float) -> float | None:
    nearest = min(APPLE_LABEL_LEVELS_NORM, key=lambda level: abs(value - level))
    if abs(value - nearest) <= APPLE_LEVEL_TOLERANCE_NORM:
        return nearest
    if value <= APPLE_ZERO_CHANNEL_MAX_NORM:
        return 0.0
    return None


def apple_label_index_from_normalized(rgb) -> int | None:
    """Return APPLE_i index (0..199) or None. Matches LPY / add_texture_by_color_v2."""
    if not is_apple_label_normalized(rgb):
        return None
    snapped = []
    for ch in (float(rgb[0]), float(rgb[1]), float(rgb[2])):
        s = _snap_apple_channel_norm(ch)
        if s is None:
            return None
        snapped.append(s)
    return _APPLE_INDEX_BY_RGB.get((snapped[0], snapped[1], snapped[2]))
