#!/usr/bin/env python3
from __future__ import annotations

"""
tree.py
authors: Abhinav Jain, Luke Strohbehn, Robin Eshraghi
modified 12/10/2025 (Robin) functions: label_vertex_by_color / get_all_points / get_apple_centroids

Generates a tree in PyBullet

Modified to include apple path planning features and new tree manipulation utilities.

"""
from collections import defaultdict
import glob
import math
import os
from pathlib import Path
import pickle
import time
from typing import Optional, Tuple, List
import secrets
import numpy as np
import pybullet
import pywavefront

# from nptyping import NDArray, Shape, Float
from numpy.typing import ArrayLike
from pybullet_tree_sim import (
    APPLE_COLORS_NORMALIZED,
    RGB_LABEL,
    URDF_PATH,
    MESHES_PATH,
    PKL_PATH,
    TEXTURES_PATH,
    apple_label_index_from_normalized,
)
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.camera_helpers import (
    compute_perpendicular_projection_vector,
)
from pybullet_tree_sim.utils import math_helpers as mh
import pybullet_tree_sim.utils.xacro_utils as xutils
from scipy.spatial.transform import Rotation
import xacro
import xml.etree.ElementTree as ET

from zenlog import log

# v2 multi-material URDF part stems (grep VIZ_VALIDATE PARTS_CHECK)
VIZ_EXPECTED_PART_STEMS = frozenset(
    {
        "mat_apple",
        "mat_branch",
        "mat_leaf",
        "mat_nontrunk",
        "mat_spur",
        "mat_stem",
        "mat_trunk",
    }
)


# from pruning_sb3.pruning_gym.helpers import roundup, rounddown
# from memory_profiler import profile


class TreeException(Exception):
    pass


class Tree:
    """This class is used to create a tree object by loading the urdf file and the obj file
    along with the labelled obj file. This class is used to filter the points on the tree # and create a curriculum of points
    to be used in training.

    To create a tree object, the following parameters are required:
    urdf_path: The path to the urdf file
    obj_path: The path to the obj file
    labelled_obj_path: The path to the labelled obj file
    pos: The position of the tree
    orientation: The orientation of the tree
    scale: The scale of the tree
    """

    _tree_xacro_path = os.path.join(URDF_PATH, "trees", "envy", "tree.urdf.xacro")
    _tree_generated_urdf_path = os.path.join(URDF_PATH, "trees", "envy", "generated")
    _tree_meshes_unlabeled_path = os.path.join(MESHES_PATH, "trees", "envy", "unlabeled", "obj")
    _tree_meshes_labeled_path = os.path.join(MESHES_PATH, "trees", "envy", "labeled", "obj")
    _pkl_path = PKL_PATH
    # Candidate diffuse maps (first existing file wins).
    _bark_texture_candidates = (
        os.path.join("bark_willow_02", "bark_willow_02_diff_4k.jpg"),
        os.path.join("bark_willow", "bark_willow_diff_4k.jpg"),
    )
    _grey_visual_rgba = [0.92, 0.92, 0.92, 1.0]
    _obj_mtl_material_name = "mat_bark_willow"  # must match usemtl in envy OBJ files

    def obj_mtl_path(self) -> str:
        """Path referenced by mtllib in unlabeled OBJ (e.g. LPy_envy_00008_labeled.mtl)."""
        return os.path.join(self._tree_meshes_unlabeled_path, f"{self.id_str}_labeled.mtl")

    @classmethod
    def resolve_bark_texture_path(cls) -> str | None:
        for rel in cls._bark_texture_candidates:
            path = os.path.join(TEXTURES_PATH, rel)
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def pybullet_load_flags(pbclient, *, use_obj_mtl: bool = True) -> int:
        """URDF load flags; use MTL from OBJ when visual has no URDF material override."""
        flags = 0
        if use_obj_mtl and hasattr(pbclient, "URDF_USE_MATERIAL_COLORS_FROM_MTL"):
            flags |= pbclient.URDF_USE_MATERIAL_COLORS_FROM_MTL
        if use_obj_mtl and hasattr(pbclient, "URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL"):
            flags |= pbclient.URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL
        return flags

    def resolve_urdf_mesh_path(self) -> str | None:
        """Resolve the visual/collision mesh path declared in the tree URDF."""
        try:
            root = ET.parse(self.urdf_path).getroot()
            for mesh in root.iter("mesh"):
                filename = mesh.get("filename")
                if filename:
                    return os.path.normpath(
                        os.path.join(os.path.dirname(self.urdf_path), filename)
                    )
        except Exception as exc:
            log.warning("Could not parse mesh filename from URDF %s: %s", self.urdf_path, exc)
        return None

    @staticmethod
    def read_obj_mtllib_name(obj_path: str) -> str | None:
        """Return mtllib filename from OBJ header (first ~30 lines)."""
        try:
            with open(obj_path, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i > 30:
                        break
                    if line.startswith("mtllib "):
                        return line.split(None, 1)[1].strip()
        except OSError as exc:
            log.warning("Could not read OBJ mtllib from %s: %s", obj_path, exc)
        return None

    def obj_mtllib_path(self) -> str | None:
        """Absolute path to MTL file referenced by the unlabeled OBJ mtllib line."""
        name = self.read_obj_mtllib_name(self.mesh_path)
        if not name:
            return None
        return os.path.join(os.path.dirname(self.mesh_path), name)

    def log_pybullet_mesh_preflight(self) -> None:
        """Log URDF/OBJ/MTL paths before loadURDF (verify files PyBullet will use)."""
        log.info("--- Tree PyBullet mesh preflight (%s) ---", self.id_str)
        log.info("urdf_path: %s exists=%s", self.urdf_path, os.path.isfile(self.urdf_path))

        resolved = self.resolve_urdf_mesh_path()
        if resolved:
            exists = os.path.isfile(resolved)
            size_mb = os.path.getsize(resolved) / (1024 * 1024) if exists else 0.0
            log.info(
                "URDF mesh resolved: %s exists=%s size=%.1f MB",
                resolved,
                exists,
                size_mb,
            )
            if not exists:
                log.warning("URDF mesh file missing — PyBullet visual will be empty")
        else:
            log.warning("Could not resolve URDF mesh path from %s", self.urdf_path)

        mesh_exists = os.path.isfile(self.mesh_path)
        log.info("mesh_path (unlabeled OBJ): %s exists=%s", self.mesh_path, mesh_exists)

        mtllib_name = self.read_obj_mtllib_name(self.mesh_path)
        obj_mtl = self.obj_mtllib_path()
        auto_mtl = self.obj_mtl_path()
        log.info("OBJ mtllib: %s", mtllib_name or "(none)")
        if obj_mtl:
            log.info(
                "OBJ-linked MTL (PyBullet loads via mtllib): %s exists=%s",
                obj_mtl,
                os.path.isfile(obj_mtl),
            )
        log.info(
            "Auto-written MTL (ensure_obj_mtl): %s exists=%s",
            auto_mtl,
            os.path.isfile(auto_mtl),
        )
        if obj_mtl and auto_mtl and os.path.normpath(obj_mtl) != os.path.normpath(auto_mtl):
            log.warning(
                "Tree %s: PyBullet uses OBJ mtllib %s; ensure_obj_mtl writes %s",
                self.id_str,
                os.path.basename(obj_mtl),
                os.path.basename(auto_mtl),
            )

    def log_pybullet_visual_summary(self, pbclient, *, phase: str = "post-load") -> None:
        """Log visual shapes and AABB after loadURDF (detect empty/broken meshes)."""
        log.info("--- Tree PyBullet mesh postflight (%s) [%s] ---", self.id_str, phase)
        if self.pyb_id is None or self.pyb_id < 0:
            log.warning("loadURDF failed: invalid pyb_id=%s", self.pyb_id)
            return

        log.info(
            "pyb_id=%s basePosition=%s globalScaling=%s",
            self.pyb_id,
            np.round(np.asarray(self.pos), 4).tolist(),
            self.scale,
        )
        try:
            visual_data = pbclient.getVisualShapeData(self.pyb_id)
            log.info("visual shape count: %s", len(visual_data))
            if not visual_data:
                log.warning("No visual shapes on tree body — mesh did not load for rendering")
            for i, vs in enumerate(visual_data):
                mesh_file = vs[4]
                if isinstance(mesh_file, bytes):
                    mesh_file = mesh_file.decode("utf-8", errors="replace")
                short_mesh = os.path.basename(mesh_file.replace("\\", "/"))
                log.info(
                    "  visual[%s] link=%s: geom_type=%s mesh=%s rgba=%s",
                    i,
                    vs[1],
                    vs[2],
                    short_mesh,
                    tuple(round(float(x), 4) for x in vs[7]),
                )
        except Exception as exc:
            log.warning("getVisualShapeData failed for tree %s: %s", self.id_str, exc)

        try:
            aabb_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
            aabb_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
            link_indices = [-1]
            try:
                link_indices.extend(range(pbclient.getNumJoints(self.pyb_id)))
            except Exception:
                pass
            for link in link_indices:
                link_min, link_max = pbclient.getAABB(self.pyb_id, link)
                link_min = np.asarray(link_min, dtype=np.float64)
                link_max = np.asarray(link_max, dtype=np.float64)
                aabb_min = np.minimum(aabb_min, link_min)
                aabb_max = np.maximum(aabb_max, link_max)
            extent = aabb_max - aabb_min
            log.info(
                "AABB (all links) min=%s max=%s extent=%s",
                np.round(aabb_min, 4).tolist(),
                np.round(aabb_max, 4).tolist(),
                np.round(extent, 4).tolist(),
            )
            max_extent = float(np.max(extent))
            if max_extent < 1e-4:
                log.warning(
                    "Tree AABB extent ~0 — visual mesh likely empty or failed to load"
                )
            elif max_extent < 0.2:
                log.warning(
                    "Tree AABB max extent %.4f m is very small (scale=%s) — "
                    "mesh may not have loaded (expect ~1–2 m for envy trees)",
                    max_extent,
                    self.scale,
                )
            else:
                log.info("Tree AABB OK (max extent %.3f m)", max_extent)
        except Exception as exc:
            log.warning("getAABB failed for tree %s: %s", self.id_str, exc)

    def ensure_obj_mtl(self, *, grey: bool = False) -> str:
        """Write MTL beside the unlabeled OBJ (required: OBJ mtllib points here).

        grey=True: flat bright material, no map_Kd (--no-texture).
        grey=False: bark diffuse map_Kd for textured runs.
        """
        mtl_path = self.obj_mtl_path()
        mat = self._obj_mtl_material_name
        if grey:
            content = (
                "# Auto-generated for PyBullet grey mode (no texture map)\n"
                f"newmtl {mat}\n"
                "Ns 96.0\n"
                "Ka 1.0 1.0 1.0\n"
                "Kd 0.92 0.92 0.92\n"
                "Ks 0.0 0.0 0.0\n"
                "d 1.0\n"
                "illum 1\n"
            )
        else:
            tex_path = self.resolve_bark_texture_path()
            if tex_path is None:
                log.warning(
                    "Cannot write bark MTL for %s: no texture under %s",
                    self.id_str,
                    TEXTURES_PATH,
                )
                return mtl_path
            rel_tex = os.path.relpath(tex_path, self._tree_meshes_unlabeled_path).replace("\\", "/")
            content = (
                "# Auto-generated for PyBullet (map_Kd relative to this MTL file)\n"
                f"newmtl {mat}\n"
                "Ns 225.0\n"
                "Ka 1.0 1.0 1.0\n"
                "Kd 0.85 0.85 0.85\n"
                "Ks 0.5 0.5 0.5\n"
                "d 1.0\n"
                "illum 2\n"
                f"map_Kd {rel_tex}\n"
            )
        with open(mtl_path, "w", encoding="utf-8") as f:
            f.write(content)
        if grey:
            log.info("Wrote grey tree MTL for PyBullet: %s", mtl_path)
        else:
            log.info("Wrote bark tree MTL for PyBullet: %s (map_Kd=%s)", mtl_path, rel_tex)
        if not os.path.isfile(mtl_path):
            log.warning("Tree MTL missing after write: %s", mtl_path)
        return mtl_path

    def load_pybullet_body(
        self,
        pbclient,
        *,
        apply_bark_texture: bool = True,
        use_obj_mtl: bool = True,
        preserve_exported_mtl: bool = False,
        hide_collision_visual: bool = False,
        **load_kwargs,
    ) -> int:
        """Load tree URDF; grey or bark via OBJ MTL (OBJ mtllib requires *_labeled.mtl on disk).

        preserve_exported_mtl: if True and apply_bark_texture, do not call ensure_obj_mtl()
        (use mtllib from OBJ, e.g. export_pybullet_tree_v2 *_textured.mtl).
        hide_collision_visual: if False, keep base-link full-mesh visual (debug).
        """
        if apply_bark_texture:
            load_with_mtl = use_obj_mtl
            if load_with_mtl and not preserve_exported_mtl:
                self.ensure_obj_mtl(grey=False)
        else:
            load_with_mtl = True
            self.ensure_obj_mtl(grey=True)

        flags = load_kwargs.pop("flags", self.pybullet_load_flags(pbclient, use_obj_mtl=load_with_mtl))
        base_position = load_kwargs.pop("basePosition", self.pos)
        base_orientation = load_kwargs.pop("baseOrientation", self.orientation)
        global_scaling = load_kwargs.pop("globalScaling", self.scale)
        use_fixed_base = load_kwargs.pop("useFixedBase", True)

        auto_mtl_path = self.obj_mtl_path()
        obj_mtl_path = self.obj_mtllib_path()
        if load_with_mtl and obj_mtl_path and not os.path.isfile(obj_mtl_path):
            log.warning(
                "Tree %s OBJ mtllib file missing: %s — mesh may render black",
                self.id_str,
                obj_mtl_path,
            )
        if load_with_mtl and not preserve_exported_mtl and not os.path.isfile(auto_mtl_path):
            log.warning(
                "Tree %s ensure_obj_mtl target missing: %s",
                self.id_str,
                auto_mtl_path,
            )

        self.log_pybullet_mesh_preflight()
        t0 = time.perf_counter()
        self.pyb_id = pbclient.loadURDF(
            self.urdf_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            globalScaling=global_scaling,
            useFixedBase=use_fixed_base,
            flags=flags,
            **load_kwargs,
        )
        log.info(
            "loadURDF(%s) -> pyb_id=%s in %.0f ms",
            os.path.basename(self.urdf_path),
            self.pyb_id,
            (time.perf_counter() - t0) * 1000.0,
        )
        self.log_pybullet_visual_summary(pbclient)

        if not apply_bark_texture:
            self.apply_grey_visual_all_links(pbclient)
            if hide_collision_visual:
                self._hide_collision_mesh_visuals(pbclient)
            log.info(
                "Tree %s loaded grey (urdf flags=%s, obj_mtl=%s, auto_mtl=%s)",
                self.id_str,
                flags,
                obj_mtl_path,
                auto_mtl_path,
            )
            self.log_tree_parts_integrity(
                pbclient,
                context="post-load-grey",
                require_hidden_base=hide_collision_visual,
            )
            return self.pyb_id

        if load_with_mtl:
            mode = "multi-material OBJ/MTL" if preserve_exported_mtl else "OBJ/MTL (single bark)"
            log.info(
                "Tree %s loaded with %s (urdf flags=%s, obj_mtl=%s, auto_mtl=%s, preserve_exported_mtl=%s)",
                self.id_str,
                mode,
                flags,
                obj_mtl_path,
                auto_mtl_path,
                preserve_exported_mtl,
            )
            if preserve_exported_mtl:
                if not hide_collision_visual:
                    log.info(
                        "Tree %s: keeping base-link full-mesh visual (hide_collision_visual=False)",
                        self.id_str,
                    )
                if hide_collision_visual:
                    self._hide_collision_mesh_visuals(pbclient)
                self._apply_leaf_part_visual_fallback(pbclient)
                log.info(
                    "Tree %s: part visuals use URDF/OBJ mtllib only (no double-sided, no rebind)",
                    self.id_str,
                )
                if hide_collision_visual:
                    self.log_pybullet_visual_validation(pbclient, phase="post-hide")
                else:
                    self.log_pybullet_visual_validation(pbclient, phase="post-load")
                self.log_tree_parts_integrity(
                    pbclient,
                    context="post-load",
                    require_hidden_base=hide_collision_visual,
                )
        else:
            self.apply_bark_texture(pbclient, fallback=False)
        return self.pyb_id

    @staticmethod
    def _parse_mtl_map_kd_by_material(mtl_path: str) -> dict[str, str]:
        """Parse newmtl blocks; return material name -> absolute map_Kd path."""
        path = Path(mtl_path)
        if not path.is_file():
            return {}
        base = path.parent.resolve()
        current: str | None = None
        out: dict[str, str] = {}
        with path.open(encoding="utf-8", errors="replace") as fin:
            for raw_line in fin:
                stripped = raw_line.strip()
                if stripped.startswith("newmtl "):
                    current = stripped.split(None, 1)[1].strip()
                    continue
                if current and stripped.lower().startswith("map_kd "):
                    rel = stripped.split(None, 1)[1].strip().strip('"')
                    tex = Path(rel)
                    abs_tex = tex if tex.is_absolute() else (base / rel).resolve()
                    if abs_tex.is_file():
                        out[current] = str(abs_tex)
        return out

    def _apply_leaf_part_visual_fallback(self, pbclient) -> None:
        """Solid green on mat_leaf only when leaf2 map_Kd was not bound."""
        from pybullet_tree_sim.obj_multimat import LEAF_OPAQUE_KD, parts_mtl_filename, parts_subdir

        if self.pyb_id is None or self.pyb_id < 0:
            return
        parts_dir = parts_subdir(Path(self._tree_meshes_unlabeled_path), self.id_str)
        mtl_path = parts_dir / parts_mtl_filename(self.id_str)
        if "mat_leaf" in self._parse_mtl_map_kd_by_material(str(mtl_path)):
            log.info("Tree %s: mat_leaf uses leaf2 map_Kd; skipping green fallback", self.id_str)
            return
        rgba = [LEAF_OPAQUE_KD[0], LEAF_OPAQUE_KD[1], LEAF_OPAQUE_KD[2], 1.0]
        try:
            for vs in pbclient.getVisualShapeData(self.pyb_id):
                mesh_file = vs[4]
                if isinstance(mesh_file, bytes):
                    mesh_file = mesh_file.decode("utf-8", errors="replace")
                if "mat_leaf.obj" not in mesh_file.replace("\\", "/"):
                    continue
                pbclient.changeVisualShape(
                    self.pyb_id,
                    vs[1],
                    textureUniqueId=-1,
                    rgbaColor=rgba,
                )
            log.info("Tree %s: leaf part visual fallback (opaque green)", self.id_str)
        except Exception as exc:
            log.warning("Leaf visual fallback failed for %s: %s", self.id_str, exc)

    def _hide_collision_mesh_visuals(self, pbclient) -> None:
        """Hide base-link full-mesh visual so part links (trunk/leaf/spur) are visible."""
        if self.pyb_id is None or self.pyb_id < 0:
            return
        full_mesh_name = f"{self.id_str}.obj"
        hidden_links: list[int] = []
        invisible_visual = -1
        try:
            invisible_visual = pbclient.createVisualShape(
                shapeType=pbclient.GEOM_BOX,
                halfExtents=[1e-6, 1e-6, 1e-6],
                rgbaColor=[0.0, 0.0, 0.0, 0.0],
            )
        except Exception as exc:
            log.warning("Tree %s: could not create invisible visual shape: %s", self.id_str, exc)

        try:
            for vs in pbclient.getVisualShapeData(self.pyb_id):
                mesh_file = vs[4]
                if isinstance(mesh_file, bytes):
                    mesh_file = mesh_file.decode("utf-8", errors="replace")
                mesh_norm = mesh_file.replace("\\", "/")
                if "parts/" in mesh_norm:
                    continue
                link_index = int(vs[1])
                # Base link: hide collision-mesh ghost and placeholder box (not part children).
                is_base_link = link_index in (0, -1)
                is_full_tree_mesh = mesh_norm.endswith(full_mesh_name)
                if not is_base_link and not is_full_tree_mesh:
                    continue
                method = "none"
                if invisible_visual >= 0:
                    try:
                        pbclient.changeVisualShape(
                            self.pyb_id,
                            link_index,
                            visualShapeData=invisible_visual,
                            rgbaColor=[0.0, 0.0, 0.0, 0.0],
                            textureUniqueId=-1,
                        )
                        method = "visualShapeData"
                    except TypeError:
                        invisible_visual = -1
                if method == "none":
                    try:
                        pbclient.changeVisualShape(
                            self.pyb_id,
                            link_index,
                            rgbaColor=[0.0, 0.0, 0.0, 0.0],
                            textureUniqueId=-1,
                            meshScale=[0.0, 0.0, 0.0],
                        )
                        method = "meshScale"
                    except TypeError:
                        pbclient.changeVisualShape(
                            self.pyb_id,
                            link_index,
                            rgbaColor=[0.0, 0.0, 0.0, 0.0],
                            textureUniqueId=-1,
                        )
                        method = "rgba_only"
                hidden_links.append(link_index)
                log.info(
                    "VIZ_VALIDATE hide_base tree=%s link=%s method=%s",
                    self.id_str,
                    link_index,
                    method,
                )
            if hidden_links:
                log.info(
                    "Tree %s: disabled collision-mesh visual on link(s) %s",
                    self.id_str,
                    hidden_links,
                )
            else:
                log.warning(
                    "Tree %s: hide_collision_visual=True but no full-mesh visual on link 0",
                    self.id_str,
                )
        except Exception as exc:
            log.warning("Could not hide collision mesh visual for %s: %s", self.id_str, exc)

    def log_pybullet_visual_validation(self, pbclient, *, phase: str = "validation") -> None:
        """Log part-only visuals and texture ids (grep VIZ_VALIDATE in session logs)."""
        if self.pyb_id is None or self.pyb_id < 0:
            return
        part_rows: list[str] = []
        hidden_rows: list[str] = []
        try:
            for vs in pbclient.getVisualShapeData(self.pyb_id):
                mesh_file = vs[4]
                if isinstance(mesh_file, bytes):
                    mesh_file = mesh_file.decode("utf-8", errors="replace")
                mesh_short = os.path.basename(mesh_file.replace("\\", "/"))
                rgba = tuple(round(float(x), 3) for x in vs[7])
                tex_id = int(vs[8]) if len(vs) > 8 else -1
                scale = vs[3] if len(vs) > 3 else None
                row = (
                    f"link={vs[1]} mesh={mesh_short} rgba={rgba} tex_id={tex_id} scale={scale}"
                )
                if "parts/" in mesh_file.replace("\\", "/"):
                    part_rows.append(row)
                elif mesh_short.endswith(f"{self.id_str}.obj") or (
                    int(vs[1]) in (0, -1) and "parts/" not in mesh_file.replace("\\", "/")
                ):
                    hidden_rows.append(row)
            log.info(
                "VIZ_VALIDATE tree=%s phase=%s part_visuals=%d hidden_full_mesh=%d",
                self.id_str,
                phase,
                len(part_rows),
                len(hidden_rows),
            )
            for row in part_rows[:8]:
                log.info("VIZ_VALIDATE   part %s", row)
            if len(part_rows) > 8:
                log.info("VIZ_VALIDATE   ... +%d more part visuals", len(part_rows) - 8)
            for row in hidden_rows:
                log.info("VIZ_VALIDATE   base %s", row)
        except Exception as exc:
            log.warning("VIZ_VALIDATE tree=%s failed: %s", self.id_str, exc)

    def log_tree_parts_integrity(
        self,
        pbclient,
        *,
        context: str = "integrity",
        require_hidden_base: bool = True,
    ) -> bool:
        """Validate all v2 part meshes present, base shell hidden, per-link AABB non-degenerate."""
        if self.pyb_id is None or self.pyb_id < 0:
            log.warning(
                "VIZ_VALIDATE PARTS_CHECK tree=%s context=%s status=FAIL reason=no_pybullet_body",
                self.id_str,
                context,
            )
            return False

        found_stems: set[str] = set()
        base_alpha: float | None = None
        base_visible_warn = False
        link_rows: list[str] = []
        issues: list[str] = []

        try:
            for vs in pbclient.getVisualShapeData(self.pyb_id):
                mesh_file = vs[4]
                if isinstance(mesh_file, bytes):
                    mesh_file = mesh_file.decode("utf-8", errors="replace")
                mesh_short = os.path.basename(mesh_file.replace("\\", "/"))
                stem = Path(mesh_short).stem
                link_index = int(vs[1])
                rgba = vs[7]
                alpha = float(rgba[3]) if len(rgba) > 3 else 1.0

                if "parts/" in mesh_file.replace("\\", "/"):
                    found_stems.add(stem)
                    link_rows.append(f"link={link_index} mesh={stem} visual=ok")
                elif mesh_short.endswith(f"{self.id_str}.obj") or (
                    link_index in (0, -1) and "parts/" not in mesh_file.replace("\\", "/")
                ):
                    if base_alpha is None or alpha < base_alpha:
                        base_alpha = alpha
                    if require_hidden_base and alpha > 0.05:
                        issues.append(
                            f"base_shell_visible link={link_index} alpha={round(alpha, 3)} "
                            f"(expected <=0.05)"
                        )
                    if alpha > 0.05:
                        base_visible_warn = True

            missing = sorted(VIZ_EXPECTED_PART_STEMS - found_stems)
            extra = sorted(found_stems - VIZ_EXPECTED_PART_STEMS)
            if missing:
                issues.append(f"missing_parts={missing}")
            if extra:
                issues.append(f"unexpected_parts={extra}")

            num_joints = pbclient.getNumJoints(self.pyb_id)
            tree_max_extent: float | None = None
            try:
                aabb_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
                aabb_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
                for link in range(-1, num_joints):
                    link_min, link_max = pbclient.getAABB(self.pyb_id, link)
                    aabb_min = np.minimum(aabb_min, np.asarray(link_min, dtype=np.float64))
                    aabb_max = np.maximum(aabb_max, np.asarray(link_max, dtype=np.float64))
                tree_max_extent = float(np.max(aabb_max - aabb_min))
                if tree_max_extent < 0.5:
                    issues.append(
                        f"tree_aabb_small max_extent={tree_max_extent:.4f}m (expected >=0.5)"
                    )
            except Exception as aabb_exc:
                issues.append(f"tree_aabb_fail: {aabb_exc}")

            status = "PASS" if not issues else "FAIL"
            log.info(
                "VIZ_VALIDATE PARTS_CHECK tree=%s context=%s status=%s "
                "parts=%d/%d base_alpha=%s joints=%d tree_max_extent_m=%s hide_base_required=%s",
                self.id_str,
                context,
                status,
                len(found_stems),
                len(VIZ_EXPECTED_PART_STEMS),
                round(base_alpha, 3) if base_alpha is not None else "n/a",
                num_joints,
                round(tree_max_extent, 3) if tree_max_extent is not None else "n/a",
                require_hidden_base,
            )
            for row in link_rows:
                log.info("VIZ_VALIDATE PARTS_CHECK   %s", row)
            if base_visible_warn and not require_hidden_base:
                log.info(
                    "VIZ_VALIDATE PARTS_CHECK   base_shell alpha=%s (hide_collision_visual=False)",
                    round(base_alpha, 3) if base_alpha is not None else "n/a",
                )
            if issues:
                log.warning(
                    "VIZ_VALIDATE PARTS_CHECK tree=%s issues=%s",
                    self.id_str,
                    "; ".join(issues),
                )
            return status == "PASS"
        except Exception as exc:
            log.warning(
                "VIZ_VALIDATE PARTS_CHECK tree=%s context=%s status=FAIL exception=%s",
                self.id_str,
                context,
                exc,
            )
            return False

    def apply_grey_visual(self, pbclient) -> None:
        """Backup flat shading on base link only (legacy). Prefer apply_grey_visual_all_links."""
        if self.pyb_id is None or self.pyb_id < 0:
            return
        pbclient.changeVisualShape(
            self.pyb_id,
            -1,
            textureUniqueId=-1,
            rgbaColor=self._grey_visual_rgba,
            specularColor=[0.0, 0.0, 0.0],
        )
        log.info("Tree %s: grey visual overlay (no texture)", self.id_str)

    def apply_grey_visual_all_links(self, pbclient) -> None:
        """Flat grey on every visual shape (multi-material URDF or single mesh)."""
        if self.pyb_id is None or self.pyb_id < 0:
            return
        updated = 0
        try:
            for vs in pbclient.getVisualShapeData(self.pyb_id):
                link_index = int(vs[1])
                pbclient.changeVisualShape(
                    self.pyb_id,
                    link_index,
                    textureUniqueId=-1,
                    rgbaColor=self._grey_visual_rgba,
                    specularColor=[0.0, 0.0, 0.0],
                )
                updated += 1
        except Exception as exc:
            log.warning("Grey visual all-links failed for %s: %s", self.id_str, exc)
            return
        log.info("Tree %s: grey visual on %d link(s) (no texture)", self.id_str, updated)

    def apply_bark_texture(self, pbclient, *, fallback: bool = False) -> None:
        """Apply bark diffuse via changeVisualShape (primary or fallback after OBJ/MTL)."""
        if self.pyb_id is None or self.pyb_id < 0:
            return
        tex_path = self.resolve_bark_texture_path()
        if tex_path is None:
            log.warning("Bark texture not found under %s", TEXTURES_PATH)
            return
        try:
            tex_id = pbclient.loadTexture(tex_path)
            if tex_id < 0:
                log.warning("loadTexture failed for %s (tex_id=%s)", tex_path, tex_id)
                return
            pbclient.changeVisualShape(
                self.pyb_id,
                -1,
                textureUniqueId=tex_id,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                specularColor=[0.4, 0.4, 0.4],
            )
            mode = "fallback overlay" if fallback else "overlay"
            log.info("Applied bark texture (%s): %s (tex_id=%s)", mode, tex_path, tex_id)
        except Exception as exc:
            log.warning("Failed to apply bark texture to tree %s: %s", self.id_str, exc)

    @classmethod
    def textured_mtl_path_for_id(cls, id_str: str) -> str:
        """Path to v2 multi-material MTL beside the unlabeled OBJ."""
        return os.path.join(cls._tree_meshes_unlabeled_path, f"{id_str}_textured.mtl")

    @classmethod
    def has_textured_export_mtl(cls, tree_id: int, tree_type: str = "envy", namespace: str = "LPy") -> bool:
        id_str = f"{namespace}_{tree_type}_{tree_id:05d}"
        return os.path.isfile(cls.textured_mtl_path_for_id(id_str))

    @classmethod
    def regenerate_urdf_from_xacro(
        cls,
        tree_id: int,
        tree_type: str = "envy",
        namespace: str = "LPy",
        parent: str = "world",
        *,
        use_visual_material: bool | None = None,
    ) -> str:
        """Regenerate one generated envy tree URDF from tree_macro (e.g. after xacro edits).

        use_visual_material: URDF <material LightGrey> on the visual mesh. Default True.
        Set False (or auto when *_textured.mtl exists) so PyBullet uses OBJ/MTL map_Kd per group.
        """
        _tree_id = str(tree_id).zfill(5)
        id_str = f"{namespace}_{tree_type}_{_tree_id}"
        urdf_path = os.path.join(cls._tree_generated_urdf_path, id_str + ".urdf")
        os.makedirs(cls._tree_generated_urdf_path, exist_ok=True)
        if use_visual_material is None:
            use_visual_material = not cls.has_textured_export_mtl(tree_id, tree_type, namespace)
        if not use_visual_material:
            return cls.regenerate_textured_urdf(
                tree_id, tree_type=tree_type, namespace=namespace, parent=parent
            )
        urdf_mappings = {
            "namespace": namespace,
            "tree_id": _tree_id,
            "tree_type": tree_type,
            "parent": parent,
            "xyz": "0.0 0.0 0.0",
            "rpy": "0.0 0.0 0.0",
            "use_visual_material": "true" if use_visual_material else "false",
        }
        urdf_content = xutils.load_urdf_from_xacro(
            xacro_path=cls._tree_xacro_path, mappings=urdf_mappings
        ).toprettyxml()
        xutils.save_urdf(urdf_content=urdf_content, urdf_path=urdf_path)
        return urdf_path

    @classmethod
    def regenerate_textured_urdf(
        cls,
        tree_id: int,
        tree_type: str = "envy",
        namespace: str = "LPy",
        parent: str = "world",
        *,
        resplit_obj: bool = False,
    ) -> str:
        """Build URDF with one fixed child link per OBJ material (PyBullet limitation)."""
        from pybullet_tree_sim.obj_multimat import (
            ensure_mat_leaf_texture_maps,
            order_materials_for_urdf,
            parts_mtl_filename,
            parts_subdir,
            split_obj_by_usemtl,
            write_multimaterial_tree_urdf,
            refresh_parts_obj_mtllib,
            write_parts_mtl,
        )

        _tree_id = str(tree_id).zfill(5)
        id_str = f"{namespace}_{tree_type}_{_tree_id}"
        urdf_path = os.path.join(cls._tree_generated_urdf_path, id_str + ".urdf")
        obj_path = os.path.join(cls._tree_meshes_unlabeled_path, id_str + ".obj")
        src_mtl_path = Path(cls.textured_mtl_path_for_id(id_str))
        parts_dir = parts_subdir(Path(cls._tree_meshes_unlabeled_path), id_str)
        parts_mtl_name = parts_mtl_filename(id_str)

        if not os.path.isfile(obj_path):
            raise FileNotFoundError(f"Unlabeled OBJ missing for textured URDF: {obj_path}")
        if not src_mtl_path.is_file():
            raise FileNotFoundError(f"Textured MTL missing for {id_str}: {src_mtl_path}")

        ensure_mat_leaf_texture_maps(src_mtl_path)

        if resplit_obj or not parts_dir.is_dir() or not any(parts_dir.glob("*.obj")):
            log.info("Splitting %s by usemtl for multi-texture URDF", id_str)
            materials = split_obj_by_usemtl(
                Path(obj_path),
                parts_dir,
                src_mtl=src_mtl_path,
                id_str=id_str,
            )
        else:
            materials = sorted(p.stem for p in parts_dir.glob("*.obj"))
            write_parts_mtl(src_mtl_path, parts_dir / parts_mtl_name, parts_dir)
            refresh_parts_obj_mtllib(parts_dir, parts_mtl_name)

        materials = order_materials_for_urdf(materials)
        log.info("URDF part link order for %s: %s", id_str, materials)

        mesh_base = f"../../../../meshes/trees/{tree_type}/unlabeled/obj"
        collision_rel = f"{mesh_base}/{id_str}.obj"
        visual_by_mat = {
            mat: f"{mesh_base}/parts/{id_str}/{mat}.obj" for mat in materials
        }
        return write_multimaterial_tree_urdf(
            urdf_path=urdf_path,
            id_str=id_str,
            tree_type=tree_type,
            namespace=namespace,
            materials=materials,
            collision_mesh_rel=collision_rel,
            visual_mesh_rel_by_mat=visual_by_mat,
            parent=parent,
        )

    def __init__(
        self,
        pbutils: PyBUtils,
        tree_id: int | None = None,
        tree_type: str | None = None,
        namespace: str = "",
        parent: str = "world",
        urdf_path: str | None = None,
        obj_path: str | None = None,
        labeled_tree_obj_path: str | None = None,
        position: np.ndarray = np.array([0, 0, 0]),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        scale: float = 1.0,
        randomize_pose: bool = False,
        verbose: bool = True,
        seed: int | None = None,
    ) -> None:
        log.info("Creating Tree object")
        self.pbclient = pbutils.pbclient
        # Set seed
        if seed is not None:
            self.seed = seed
        else:
            self.seed = secrets.randbits(128)
        self.generator: np.random.Generator = np.random.default_rng(seed=self.seed)
        self.verbose = verbose

        # Tree specific parameters
        self.scale = scale
        self.tree_namespace = namespace
        self.tree_id = tree_id
        self.tree_type = tree_type
        self.id_str = self.create_id_string(
            tree_id=tree_id, tree_type=tree_type, namespace=namespace, urdf_path=urdf_path
        )
        self.urdf_path = os.path.join(self._tree_generated_urdf_path, self.id_str + ".urdf")
        self.mesh_path = os.path.join(self._tree_meshes_unlabeled_path, self.id_str + ".obj")
        self.labeled_mesh_path = os.path.join(self._tree_meshes_labeled_path, self.id_str + "_labeled.obj")
        self.init_pos = position
        self.init_orientation = orientation

        # URDF
        self.load_tree_urdf(scale=scale, parent=parent)

        self.rgb_label = RGB_LABEL
        self.pyb_id: int = None

        if randomize_pose:
            new_pos, new_orientation = self._randomize_pose()
        else:
            new_pos = position
            new_orientation = orientation

        self.pos = new_pos
        self.orientation = new_orientation

        self.vertex_and_projection = []
        self.projection_mean = np.array(0.0)
        self.projection_std = np.array(0.0)
        self.projection_sum_x = np.array(0.0)
        self.projection_sum_x2 = np.array(0.0)
        self.reachable_points = []

        pkl_filepath = os.path.join(self._pkl_path, Path(self.urdf_path).stem + "_points.pkl")
        if self._try_load_full_pkl_cache(pkl_filepath):
            return

        log.info("Building tree point cache from OBJ meshes (first run or cache miss)...")
        tree_obj = self.load_tree_obj()
        labeled_tree_obj = self.load_labeled_tree_obj()

        log.info("Inspecting structure of 'unlabeled_wavefront_obj'")
        if tree_obj:
            log.info(f"  Wavefront Meshes: {len(list(tree_obj.meshes.keys()))}")
            log.info(f"  Wavefront Materials: {len(list(tree_obj.materials.keys()))}")
            if hasattr(tree_obj, "mesh_list") and tree_obj.mesh_list:
                log.info(f"  Wavefront mesh_list items: {[m.name for m in tree_obj.mesh_list]}")
        log.info("Inspecting structure of 'labeled_wavefront_obj'")
        if labeled_tree_obj:
            log.info(f"  Wavefront Meshes: {len(list(labeled_tree_obj.meshes.keys()))}")
            log.info(f"  Wavefront Materials: {len(list(labeled_tree_obj.materials.keys()))}")
            if hasattr(labeled_tree_obj, "mesh_list") and labeled_tree_obj.mesh_list:
                log.info(f"  Wavefront mesh_list items: {[m.name for m in labeled_tree_obj.mesh_list]}")

        log.info(
            "Apple color targets (Camp_Envy label6_2): %s label IDs",
            len(APPLE_COLORS_NORMALIZED),
        )

        vertex_to_label = self.label_vertex_by_color(
            self.rgb_label,
            tree_obj.vertices,
            labeled_tree_obj.vertices,
            color_dist_threshold=0.1,
        )
        
        
        # append the label to each vertex
        tree_obj_vertices_labeled = []
        for i, vertex in enumerate(tree_obj.vertices):
            tree_obj_vertices_labeled.append(vertex + (vertex_to_label[vertex],))

        # # # TODO: begin() method or something
        self.transformed_vertices = list(
            map(
                lambda x: self.transform_tree_obj_vertex(x),
                tree_obj_vertices_labeled,
            )
        )

        loaded_projection_only = False
        if os.path.isfile(pkl_filepath):
            try:
                with open(pkl_filepath, "rb") as f:
                    _cached = pickle.load(f)
                if (
                    isinstance(_cached, (list, tuple))
                    and len(_cached) == 3
                    and self._saved_pose_matches_current(_cached[0], _cached[1])
                    and self._pkl_scale_matches(None)
                ):
                    self.vertex_and_projection = _cached[2]
                    loaded_projection_only = True
                    log.info(
                        "Loaded vertex_and_projection from legacy 3-field PKL %s (%d points); skipped get_all_points.",
                        pkl_filepath,
                        len(self.vertex_and_projection),
                    )
            except Exception as exc:
                log.warning("Legacy PKL load failed (%s); recomputing points.", exc)

        if not loaded_projection_only:
            self.get_all_points(tree_obj)
            self.filter_outliers()
            self.filter_trunk_points()
            if self.verbose > 0:
                print(f"INFO: Number of points: {len(self.vertex_and_projection)}")

        try:
            with open(pkl_filepath, "wb") as f:
                pickle.dump(
                    (
                        self.pos,
                        self.orientation,
                        self.scale,
                        self.vertex_and_projection,
                        self.transformed_vertices,
                    ),
                    f,
                )
            if self.verbose > 0:
                print(f"INFO: Saved tree cache PKL (pose + points + labels) {pkl_filepath}")
        except Exception as exc:
            log.warning("Could not write PKL %s: %s", pkl_filepath, exc)

        # # Make different meshes for each label
        # # make bins
        self.or_bins = self.create_bins(18, 36)
        # # Go through vertex_and_projection and assign to bins
        self.populate_bins(self.vertex_and_projection)

        #del self.vertex_and_projection

        return

    def _try_load_full_pkl_cache(self, pkl_filepath: str) -> bool:
        """Load 4-field PKL when pose matches; skip OBJ load and mesh labeling."""
        if not os.path.isfile(pkl_filepath):
            log.info("Tree PKL cache miss: no file at %s", pkl_filepath)
            return False
        try:
            with open(pkl_filepath, "rb") as f:
                _cached = pickle.load(f)
            if isinstance(_cached, (list, tuple)) and len(_cached) >= 5:
                saved_scale = float(_cached[2])
                proj_idx, vert_idx = 3, 4
            elif isinstance(_cached, (list, tuple)) and len(_cached) >= 4:
                saved_scale = 1.0
                proj_idx, vert_idx = 2, 3
                log.info(
                    "Tree PKL legacy format (no scale field) at %s; treating saved scale as 1.0",
                    pkl_filepath,
                )
            else:
                log.info("Tree PKL cache miss: expected >=4 fields in %s", pkl_filepath)
                return False
            if not self._saved_pose_matches_current(_cached[0], _cached[1]):
                log.info(
                    "Tree PKL cache miss: pose mismatch for %s (config pos=%s)",
                    pkl_filepath,
                    np.round(self.pos, 4).tolist(),
                )
                return False
            if not self._pkl_scale_matches(saved_scale):
                log.info(
                    "Tree PKL cache miss: scale mismatch for %s (saved=%s config=%s)",
                    pkl_filepath,
                    saved_scale,
                    self.scale,
                )
                return False
            self.vertex_and_projection = _cached[proj_idx]
            self.transformed_vertices = _cached[vert_idx]
            log.info(
                "Tree PKL cache hit: %s (scale=%s, %d projections, %d labeled verts); skipped OBJ load.",
                pkl_filepath,
                saved_scale,
                len(self.vertex_and_projection),
                len(self.transformed_vertices),
            )
            self.or_bins = self.create_bins(18, 36)
            self.populate_bins(self.vertex_and_projection)
            return True
        except Exception as exc:
            log.warning("Tree PKL cache load failed (%s); building from mesh.", exc)
            return False

    def _saved_pose_matches_current(self, saved_pos, saved_or, atol: float = 1e-4) -> bool:
        """True if saved pose matches this tree's config pose (quaternion q ~ ±q)."""
        sp = np.asarray(saved_pos, dtype=np.float64).reshape(3)
        sq = np.asarray(saved_or, dtype=np.float64).reshape(4)
        cp = np.asarray(self.pos, dtype=np.float64).reshape(3)
        cq = np.asarray(self.orientation, dtype=np.float64).reshape(4)
        if not np.allclose(sp, cp, atol=atol):
            return False
        return bool(
            np.allclose(sq, cq, atol=atol) or np.allclose(sq, -cq, atol=atol)
        )

    def _pkl_scale_matches(self, saved_scale: float | None, atol: float = 1e-5) -> bool:
        """True if PKL scale matches this tree (None saved_scale => legacy 1.0)."""
        if saved_scale is None:
            saved_scale = 1.0
        return bool(np.isclose(float(saved_scale), float(self.scale), rtol=0.0, atol=atol))

    def create_id_string(
        self,
        tree_id: int | None = None,
        tree_type: str | None = None,
        namespace: str | None = None,
        urdf_path: str | None = None,
    ) -> str:
        if tree_id is None and urdf_path is None:
            # log.error("Both urdf_path and tree parameters cannot be None.")
            raise TreeException("Both urdf_path and tree parameters cannot be None.")

        if tree_id is not None:
            tree_id = str(tree_id).zfill(5)

        if urdf_path is None:
            id_str = f"{namespace}_{tree_type}_{tree_id}"
        else:
            id_str = Path(urdf_path).stem
            id_str_components = id_str.split("_")
            self.tree_namespace = id_str_components[0]
            self.tree_type = id_str_components[1]
            self.tree_id = str(id_str_components[2]).zfill(5)
        return id_str

    def _load_points_from_pickle(self, pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.pos = data[0]
            self.orientation = data[1]
            self.vertex_and_projection = data[2]
        if self.verbose > 0:
            log.info(f"Loaded points from pickle file {pkl_path}")
            log.info(f"Number of points: {len(self.vertex_and_projection)}")
        return

    def _randomize_pose(self) -> tuple:
        # TODO: Randomize position to bounds?
        new_position = np.array([0, 0, 0])
        # TODO: Multiply orientation with initial orientation
        new_orientation = pybullet.getQuaternionFromEuler(
            self.generator.uniform(low=-1, high=1, size=(3,)) * np.pi / 180 * 5
        )
        return new_position, new_orientation

    @staticmethod
    def create_bins(num_latitude_bins, num_longitude_bins):
        """
        Create bins separated by 10 degrees on a unit sphere.

        Parameters:
            num_latitude_bins (int): Number of bins along the latitude direction.
            num_longitude_bins (int): Number of bins along the longitude direction.

        Returns:
            list of tuples: List of tuples where each tuple represents a bin defined by
                            (latitude_min, latitude_max, longitude_min, longitude_max).
        """
        bin_size = np.deg2rad(10)  # Convert degrees to radians
        bins = {}
        for i in range(num_latitude_bins):
            lat_min = np.rad2deg(-np.pi / 2 + i * bin_size)
            lat_max = np.rad2deg(-np.pi / 2 + (i + 1) * bin_size)
            for j in range(num_longitude_bins):
                lon_min = np.rad2deg(-np.pi + j * bin_size)
                lon_max = np.rad2deg(-np.pi + (j + 1) * bin_size)
                bins[
                    (
                        round((lat_min + lat_max) / 2),
                        round((lon_min + lon_max) / 2),
                    )
                ] = []
        return bins

    def transform_tree_obj_vertex(self, vertex: ArrayLike) -> Tuple[np.ndarray, float]:
        """
        Transform a vertex from the tree object to the world frame.
        """
        vertex_pos = np.array(vertex[0:3]) * self.scale
        vertex_orientation = [0, 0, 0, 1]  # Dont care about orientation

        vertex_w_transform: Tuple[tuple, tuple] = self.pbclient.multiplyTransforms(
            self.pos, self.orientation, vertex_pos, vertex_orientation
        )
        # vertex_w_transform = np.concatenate((final_position, final_orientation))

        return (np.array(vertex_w_transform[0]), vertex[3])

    def populate_bins(self, points):
        """
        Populate the bins based on a list of direction vectors.

        Parameters:
            direction_vectors (list of numpy.ndarray): List of direction vectors.
            bins (list of tuples): List of bins where each tuple represents a bin defined by
                                   (latitude_min, latitude_max, longitude_min, longitude_max).

        Returns:
            list of lists: List of lists where each sublist represents the indices of direction vectors
                           assigned to the corresponding bin.
        """
        offset = 1e-3
        for i, point in enumerate(points):

            direction_vector = point[1]
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            lat_angle = np.rad2deg(np.arcsin(direction_vector[2])) + offset
            lon_angle = np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0])) + offset
            lat_angle_min = mh.rounddown(lat_angle)
            lat_angle_max = mh.roundup(lat_angle)
            lon_angle_min = mh.rounddown(lon_angle)
            lon_angle_max = mh.roundup(lon_angle)
            bin_key = (
                round((lat_angle_min + lat_angle_max) / 2),
                round((lon_angle_min + lon_angle_max) / 2),
            )
            # if bin_key[0] not in between -85 and 85 set as 85 or -85
            # if bin_keyp[1] not in between -175 and 175 set as 175 or -175

            if bin_key[0] > 85:
                bin_key = (85, bin_key[1])
            elif bin_key[0] < -85:
                bin_key = (-85, bin_key[1])
            if bin_key[1] > 175:
                bin_key = (bin_key[0], 175)
            elif bin_key[1] < -175:
                bin_key = (bin_key[0], -175)
            self.or_bins[bin_key].append(
                (self.urdf_path, point, self.orientation, self.scale)
            )  # Add collision meshes for each tree here
        return

    # (edit_robin) Updated function to prioritize apple color matching
    def label_vertex_by_color(
        self,
        labels,
        unlabelled_vertices,
        labelled_vertices,
        *,
        color_dist_threshold=0.05,
        default_label="UNKNOWN",
    ):
        """Label vertices: wood/leaf (RGB_LABEL), apples (200-level Camp_Envy scheme)."""
        final_vertex_labels = {}
        known_colors_list = [(np.array(color), label) for color, label in labels.items()]

        for vertex_data in labelled_vertices:
            coords = tuple(vertex_data[:3])
            color_np = np.array(vertex_data[3:], dtype=np.float64)

            best_label = default_label
            for known_color, label in known_colors_list:
                if np.linalg.norm(color_np - known_color) < color_dist_threshold:
                    best_label = label
                    break
            else:
                apple_idx = apple_label_index_from_normalized(color_np)
                if apple_idx is not None:
                    best_label = f"APPLE_{apple_idx}"

            final_vertex_labels[coords] = best_label

        for vertex_coord in unlabelled_vertices:
            coord_tuple = tuple(vertex_coord)
            if coord_tuple not in final_vertex_labels:
                final_vertex_labels[coord_tuple] = default_label

        return final_vertex_labels

    @staticmethod
    def _is_apple_pick_label(label) -> bool:
        """Camp_Envy uses APPLE_0..APPLE_39; legacy code used the literal APPLE."""
        return label == "APPLE" or (
            isinstance(label, str) and label.startswith("APPLE_")
        )

    # (edit_robin) Handling of potential numerical issues like zero-length vectors, NaN propagation, division by zero, and negative arguments to square roots
    def get_all_points(self, tree_obj):
        for num, face in enumerate(tree_obj.mesh_list[0].faces):
            
            # Order the sides of the face by length
            ab_data = ( # Renamed to avoid conflict if you use 'ab' as a vector later
                face[0],
                face[1],
                np.linalg.norm(self.transformed_vertices[face[0]][0] - self.transformed_vertices[face[1]][0]),
            )
            ac_data = ( # Renamed
                face[0],
                face[2],
                np.linalg.norm(self.transformed_vertices[face[0]][0] - self.transformed_vertices[face[2]][0]),
            )
            bc_data = ( # Renamed
                face[1],
                face[2],
                np.linalg.norm(self.transformed_vertices[face[1]][0] - self.transformed_vertices[face[2]][0]),
            )

            normal_vec = np.cross(
                self.transformed_vertices[ac_data[0]][0] - self.transformed_vertices[ac_data[1]][0], # Use ac_data
                self.transformed_vertices[bc_data[0]][0] - self.transformed_vertices[bc_data[1]][0], # Use bc_data
            )
            # Only front facing faces
            if np.dot(normal_vec, [0, 1, 0]) < 0:
                continue
            
            sides = [ab_data, ac_data, bc_data]
            # argsort sorts in ascending order of side lengths (element at index 2)
            sorted_indices = np.argsort([s[2] for s in sides])
            
            # s_ac, s_ab, s_bc refer to the side data tuples (vertex1_idx, vertex2_idx, length)
            s_bc = sides[sorted_indices[0]] # Shortest side data
            s_ab = sides[sorted_indices[1]] # Middle side data
            s_ac = sides[sorted_indices[2]] # Longest side data

            # Define the vectors for projection
            # Vector along the longest side (AC)
            vec_ac_edge = self.transformed_vertices[s_ac[0]][0] - self.transformed_vertices[s_ac[1]][0]
            # Vector along the shortest side (BC) - this is likely the one causing issues if it's zero length
            vec_bc_edge = self.transformed_vertices[s_bc[0]][0] - self.transformed_vertices[s_bc[1]][0]

            # CHECK FOR ZERO-LENGTH VECTOR
            # Check if the length of the shortest side (s_bc[2]) is effectively zero. means vec_bc_edge will be a zero vector.
            if s_bc[2] < 1e-9:  # Using the pre-calculated length s_bc[2]
                # print(f"DEBUG: Face {face_indices} has a zero-length shortest edge {s_bc[0]}-{s_bc[1]}. Setting perpendicular_projection to zero.")
                perpendicular_projection = np.array([0.0, 0.0, 0.0])
            else:
                perpendicular_projection = compute_perpendicular_projection_vector(
                    vec_ac_edge,
                    vec_bc_edge,
                )

            # Check if perpendicular_projection itself became NaN for any other unexpected reason
            if np.isnan(perpendicular_projection).any():
                # print(f"DEBUG: perpendicular_projection is NaN for face with original vertices {face}. Skipping summation.")
                perpendicular_projection = np.array([0.0, 0.0, 0.0]) 
                # continue # Or skip this point entirely

            scale = np.random.uniform()
            # Use s_ab (middle side) for defining tree_point as per your original logic
            tree_point = (1 - scale) * self.transformed_vertices[s_ab[0]][0] + scale * self.transformed_vertices[s_ab[1]][0]

            labels = [
                self.transformed_vertices[s_ab[0]][1], self.transformed_vertices[s_ab[1]][1],
                self.transformed_vertices[s_ac[0]][1], self.transformed_vertices[s_ac[1]][1],
                self.transformed_vertices[s_bc[0]][1], self.transformed_vertices[s_bc[1]][1],
            ]

            if len(set(labels)) == 1:
                label = labels[0]
            else:
                label = "JOINT"
            if not self._is_apple_pick_label(label) and label != "SPUR":
                continue
            
            self.vertex_and_projection.append((tree_point, perpendicular_projection, normal_vec, label))
            
            current_norm = np.linalg.norm(perpendicular_projection)
            if np.isnan(current_norm):
                # print(f"DEBUG: Norm of perpendicular_projection is NaN. Skipping sums for this point.")
                pass # Do not add NaN to sums
            else:
                self.projection_sum_x += current_norm
                self.projection_sum_x2 += current_norm ** 2

        # Calculation of mean and std
        N = len(self.vertex_and_projection)
        if N > 0:
            self.projection_mean = self.projection_sum_x / N
            # print(f"DEBUG: N (len(self.vertex_and_projection)) = {N}")
            # print(f"DEBUG: self.projection_sum_x = {self.projection_sum_x}")
            # print(f"DEBUG: self.projection_sum_x2 = {self.projection_sum_x2}")
            # print(f"DEBUG: calculated self.projection_mean = {self.projection_mean}")
            val_E_X_sq = self.projection_sum_x2 / N
            val_E_X_whole_sq = self.projection_mean ** 2
            variance_before_sqrt = val_E_X_sq - val_E_X_whole_sq
            # print(f"DEBUG: E[X^2] (sum_x2/N) = {val_E_X_sq}")
            # print(f"DEBUG: (E[X])^2 (mean^2) = {val_E_X_whole_sq}")
            # print(f"DEBUG: Variance before sqrt (E[X^2] - (E[X])^2) = {variance_before_sqrt}")
            self.projection_std = np.sqrt(max(0, variance_before_sqrt))
            # print(f"DEBUG: Calculated self.projection_std = {self.projection_std}")
        else:
            # print("DEBUG: N is 0 in get_all_points, no points added. Setting mean/std to 0.")
            self.projection_mean = np.array(0.)
            self.projection_std = np.array(0.)
        
        return

    def filter_outliers(self):
        # Filter out outliers
        print(
            "Number of points before filtering: ",
            len(self.vertex_and_projection),
        )
        self.vertex_and_projection = list(
            filter(
                lambda x: np.linalg.norm(x[1]) > self.projection_mean + 0.5 * self.projection_std,
                self.vertex_and_projection,
            )
        )
        print(
            "Number of points after filtering: ",
            len(self.vertex_and_projection),
        )
        return

    def filter_points_below_base(self, base_xyz):
        # Filter out points below the base of the arm
        self.vertex_and_projection = list(filter(lambda x: x[0][2] > base_xyz[2], self.vertex_and_projection))
        return

    def filter_trunk_points(self):
        self.vertex_and_projection = list(
            filter(
                lambda x: abs(x[0][0] - self.pos[0]) > 0.8,
                self.vertex_and_projection,
            )
        )
        print(
            "Number of points after filtering trunk points: ",
            len(self.vertex_and_projection),
        )
        return

    def build_collision_kdtree(self) -> None:
        """KD-tree over world-frame labeled vertices for robot contact labeling."""
        from scipy.spatial import cKDTree

        verts = getattr(self, "transformed_vertices", None)
        if not verts:
            log.warning("build_collision_kdtree: no transformed_vertices on %s", self.id_str)
            self.kdtree = None
            self.kdtree_vertex_labels = []
            self.kdtree_vertex_coords_np = np.empty((0, 3), dtype=np.float64)
            return
        coords = np.asarray([v[0] for v in verts], dtype=np.float64)
        self.kdtree_vertex_coords_np = coords
        self.kdtree_vertex_labels = [v[1] for v in verts]
        self.kdtree = cKDTree(coords)
        log.info(
            "Collision KD-tree built for %s (%d vertices)",
            self.id_str,
            len(self.kdtree_vertex_labels),
        )

    def rigid_rebase_vertex_points_to_pose(
        self,
        old_pos: ArrayLike,
        old_or: ArrayLike,
        new_pos: ArrayLike,
        new_or: ArrayLike,
    ) -> None:
        """Re-express cached world vertices from one tree pose to another."""
        if not getattr(self, "transformed_vertices", None):
            return
        old_pos = np.asarray(old_pos, dtype=np.float64).reshape(3)
        old_or = np.asarray(old_or, dtype=np.float64).reshape(4)
        new_pos = np.asarray(new_pos, dtype=np.float64).reshape(3)
        new_or = np.asarray(new_or, dtype=np.float64).reshape(4)
        inv_pos, inv_or = self.pbclient.invertTransform(
            old_pos.tolist(), old_or.tolist()
        )
        rebased = []
        for coords, label in self.transformed_vertices:
            local_pos, _ = self.pbclient.multiplyTransforms(
                inv_pos, inv_or, np.asarray(coords, dtype=np.float64).reshape(3).tolist(), [0, 0, 0, 1]
            )
            world_pos, _ = self.pbclient.multiplyTransforms(
                new_pos.tolist(), new_or.tolist(), local_pos, [0, 0, 0, 1]
            )
            rebased.append((np.asarray(world_pos, dtype=np.float64), label))
        self.transformed_vertices = rebased

   # (add_robin) New function to get apple centroids based on unique labels
    def get_apple_centroids(self): # No parameters needed
        """
        Identifies individual apple centroids based on their unique "APPLE_i" label.
        This method replaces the need for spatial clustering.
        """
        if not hasattr(self, 'transformed_vertices') or not self.transformed_vertices:
            log.warning("get_apple_centroids: 'transformed_vertices' not available or empty.")
            return []

        # Group points by their unique apple label
        apple_points_by_label = defaultdict(list)
        for coords_np, label_str in self.transformed_vertices:
            if isinstance(label_str, str) and label_str.startswith("APPLE_"):
                apple_points_by_label[label_str].append(coords_np)

        if not apple_points_by_label:
            log.info("get_apple_centroids: No points with 'APPLE_' labels found.")
            return []

        # Calculate the centroid for each group of points
        centroids = []
        log.info(f"get_apple_centroids: Found {len(apple_points_by_label)} unique apple labels. Calculating centroids...")
        for label, points_list in sorted(apple_points_by_label.items()):
            points_np = np.array(points_list)
            center_of_cluster = np.mean(points_np, axis=0)
            centroids.append(center_of_cluster)
            log.debug(f"  {label}: Centroid {center_of_cluster.round(3)}, found from {len(points_list)} points.")
            
        return centroids

    _VECTOR_EPSILON = 1e-6
    
    def load_tree_urdf(
        self,
        scale: float,
        parent: str = "world",
        position: str = "0.0 0.0 0.0",
        orientation: str = "0.0 0.0 0.0",
        save_urdf: bool = True,
        regenerate_urdf: bool = False,  # TODO: make save/regenerate work well together. Will need to add delete URDF function
    ) -> str:
        """Load a tree URDF from a given path or generate a tree URDF from a xacro file. If content is generated, by default saves the content to /urdf/trees/<tree_type>/generated Returns the URDF content.

        Returns
        -------
            None
        """
        if not os.path.exists(self.urdf_path):
            log.info(f"Could not find file '{self.urdf_path}'. Generating URDF from xacro.")

            if not os.path.isdir(Tree._tree_generated_urdf_path):
                os.mkdir(Tree._tree_generated_urdf_path)

            _tree_id = str(self.tree_id).zfill(5)
            use_visual_material = not self.has_textured_export_mtl(
                self.tree_id, self.tree_type, self.tree_namespace
            )
            urdf_mappings = {
                "namespace": self.tree_namespace,
                "tree_id": _tree_id,
                "tree_type": self.tree_type,
                "parent": parent,
                "xyz": position,
                "rpy": orientation,
                "use_visual_material": "true" if use_visual_material else "false",
            }

            # If the tree macro information doesn't describe a generated file, generate it using the generic tree xacro.
            urdf_content = xutils.load_urdf_from_xacro(
                xacro_path=Tree._tree_xacro_path, mappings=urdf_mappings
            ).toprettyxml()
            if save_urdf:
                xutils.save_urdf(urdf_content=urdf_content, urdf_path=self.urdf_path)
        else:
            urdf_content = xutils.load_urdf_from_xacro(xacro_path=self.urdf_path).toprettyxml()
            log.info(f"Loaded URDF from file '{self.urdf_path}'.")

        return urdf_content

    def _load_wavefront_obj(self, path: str) -> pywavefront.Wavefront:
        """Load OBJ via pywavefront; log rounded ms (suppress library seconds print)."""
        import logging as py_logging

        wf_logger = py_logging.getLogger("pywavefront")
        prev_level = wf_logger.level
        wf_logger.setLevel(py_logging.WARNING)
        t0 = time.perf_counter()
        try:
            return pywavefront.Wavefront(path, create_materials=True, collect_faces=True)
        finally:
            wf_logger.setLevel(prev_level)
            log.info("%s: load time %.0f ms", path, (time.perf_counter() - t0) * 1000.0)

    def load_tree_obj(self):
        """Loads a mesh .obj mesh file with its path defined by the tree_id_str"""
        if not os.path.exists(self.mesh_path):
            raise TreeException(f"Could not find file '{self.mesh_path}.")
        return self._load_wavefront_obj(self.mesh_path)

    def load_labeled_tree_obj(self):
        """Loads a labeled mesh .obj mesh file with its path defined by the tree_id_str"""
        if not os.path.exists(self.labeled_mesh_path):
            raise TreeException(f"Could not find the file {self.labeled_mesh_path}")
        return self._load_wavefront_obj(self.labeled_mesh_path)

    @staticmethod
    def make_trees_from_ids(
        pbutils: PyBUtils,
        tree_ids: list[int],
        namespace: str = "",
        pos: np.ndarray = np.array([0, 0, 0]),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        scale: float = 1.0,
        randomize_pose: bool = False,
    ) -> list[Tree]:
        trees: list[Tree] = []

        for tree_id in tree_ids:
            trees.append(
                Tree(
                    pbutils=pbutils,
                    tree_id=tree_id,
                )
            )
        return trees

    @staticmethod
    def make_trees_from_folder(
        pbutils: PyBUtils,
        trees_urdf_path: str,
        trees_obj_path: str,
        trees_labelled_path: str,
        pos: np.ndarray,
        orientation: np.ndarray,
        scale: float,
        num_trees: int,
        randomize_pose: bool = False,
    ) -> list[Tree]:
        trees: list[Tree] = []
        # for urdfs, objs in zip(
        #     sorted(glob.glob(trees_urdf_path + "/*.urdf")), sorted(glob.glob(trees_obj_path + "/*.obj"))
        # ):
        #     print(urdfs, objs, labelled_obj)
        #     if len(trees) >= num_trees:
        #         break
        #     trees.append(
        #         Tree(
        #             urdf_path=urdfs,
        #             obj_path=objs,
        #             pos=pos,
        #             orientation=orientation,
        #             scale=scale,
        #             randomize_pose=randomize_pose,
        #         )
        #     )

        # self,
        # pbutils: PyBUtils,
        # tree_id: int | None = None,
        # tree_type: str | None = None,
        # namespace: str = "",
        # parent: str = "world",
        # urdf_path: str | None = None,
        # obj_path: str | None = None,
        # labeled_tree_obj_path: str | None = None,
        # position=np.array([0, 0, 0]),
        # orientation=np.array([0, 0, 0, 1]),
        # scale: float = 1.0,
        # randomize_pose: bool = False,
        # verbose: bool = True,
        # seed: int | None = None,
        return trees


def main():

    return


if __name__ == "__main__":
    main()
