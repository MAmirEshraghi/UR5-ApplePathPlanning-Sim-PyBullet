"""
Tree asset preparation and loading helpers shared by environments and the tree viewers
(tools/tree_viewers). Tree config keys: tree_id, tree_type, tree_namespace, preserve_export_mtl,
regenerate_urdf, apply_texture, hide_collision_visual.
"""
import os

import numpy as np
from zenlog import log

from pybullet_tree_sim.tree import Tree


def resolve_preserve_export_mtl(tree_conf: dict) -> bool:
    """Auto-detect v2 *_textured.mtl when preserve_export_mtl is unset."""
    preserve = tree_conf.get("preserve_export_mtl")
    if preserve is not None:
        return bool(preserve)
    return Tree.has_textured_export_mtl(
        tree_conf["tree_id"],
        tree_conf["tree_type"],
        tree_conf.get("tree_namespace", "LPy"),
    )


def ensure_tree_sim_assets(tree: Tree, tree_conf: dict) -> None:
    """Ensure OBJ/URDF exist; build multi-material URDF when v2 MTL is present."""
    from pathlib import Path

    from pybullet_tree_sim.obj_multimat import (
        parts_mtl_filename,
        parts_subdir,
        ensure_mat_leaf_texture_maps,
    )

    if not os.path.isfile(tree.mesh_path):
        raise FileNotFoundError(
            f"Unlabeled OBJ missing for {tree.id_str}: {tree.mesh_path}\n"
            f"Run: python lpy_treesim/export_pybullet_tree_v2.py --tree-id {tree.tree_id} "
            f"--lpy-index <N> --regenerate-urdf"
        )
    preserve = resolve_preserve_export_mtl(tree_conf)
    textured = Tree.has_textured_export_mtl(
        tree.tree_id, tree.tree_type, tree.tree_namespace
    )
    need_textured_urdf = preserve and textured
    if not os.path.isfile(tree.urdf_path):
        log.info("Generating URDF for %s", tree.id_str)
        if need_textured_urdf:
            Tree.regenerate_textured_urdf(
                tree_id=tree.tree_id,
                tree_type=tree.tree_type,
                namespace=tree.tree_namespace,
            )
        else:
            Tree.regenerate_urdf_from_xacro(
                tree_id=tree.tree_id,
                tree_type=tree.tree_type,
                namespace=tree.tree_namespace,
                use_visual_material=True,
            )
    elif tree_conf.get("regenerate_urdf") and need_textured_urdf:
        log.info("Regenerating multi-material URDF for %s", tree.id_str)
        Tree.regenerate_textured_urdf(
            tree_id=tree.tree_id,
            tree_type=tree.tree_type,
            namespace=tree.tree_namespace,
        )

    textured_mtl = Path(Tree.textured_mtl_path_for_id(tree.id_str))
    if textured_mtl.is_file():
        ensure_mat_leaf_texture_maps(textured_mtl)
    parts_mtl = parts_subdir(Path(tree._tree_meshes_unlabeled_path), tree.id_str) / parts_mtl_filename(
        tree.id_str
    )
    if parts_mtl.is_file():
        ensure_mat_leaf_texture_maps(parts_mtl)


def frame_debug_camera_on_tree(pb_client, tree_position, sim_conf: dict) -> None:
    """Point the GUI debug camera at the tree (not the robot base), as the tree viewers do."""
    if not sim_conf.get("frame_tree_camera", True):
        return
    pos = np.asarray(tree_position, dtype=np.float64).reshape(3)
    z_off = float(sim_conf.get("debug_camera_target_z_offset", 1.0))
    target = [float(pos[0]), float(pos[1]), float(pos[2]) + z_off]
    pb_client.resetDebugVisualizerCamera(
        cameraDistance=float(sim_conf.get("debug_camera_distance", 4.5)),
        cameraYaw=float(sim_conf.get("debug_camera_yaw", 55)),
        cameraPitch=float(sim_conf.get("debug_camera_pitch", -22)),
        cameraTargetPosition=target,
    )
    log.info("Debug camera framed on tree at %s", np.round(target, 3).tolist())


def load_tree_pybullet_body(pb_client, tree: Tree, tree_conf: dict, **load_kwargs) -> int:
    """Load tree URDF with v2 multi-material MTL when exported (matches sim viewers)."""
    apply_texture = bool(tree_conf.get("apply_texture", True))
    preserve = resolve_preserve_export_mtl(tree_conf)
    ensure_tree_sim_assets(tree, tree_conf)
    hide_collision_visual = bool(tree_conf.get("hide_collision_visual", False))
    return tree.load_pybullet_body(
        pb_client,
        apply_bark_texture=apply_texture,
        use_obj_mtl=True,
        preserve_exported_mtl=preserve and apply_texture,
        hide_collision_visual=hide_collision_visual,
        **load_kwargs,
    )

