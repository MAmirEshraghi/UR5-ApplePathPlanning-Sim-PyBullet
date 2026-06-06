#!/usr/bin/env python3
"""Export lpy_treesim dataset meshes into pybullet_tree_sim layout.

Reads:
  - dataset/textured_obj/tree_{lpy_index}.obj  (Blender v2, quads + multi-MTL)
  - dataset/obj/tree_{lpy_index}.obj           (MeshLab labeled RGB)

Writes:
  - pybullet_tree_sim/meshes/trees/envy/unlabeled/obj/LPy_envy_{tree_id:05d}.obj
    (Blender-style: keeps v/vt/vn and quad faces — required for PyBullet URDF visuals)
  - pybullet_tree_sim/meshes/trees/envy/unlabeled/obj/LPy_envy_{tree_id:05d}_labeled.mtl
  - pybullet_tree_sim/meshes/trees/envy/labeled/obj/LPy_envy_{tree_id:05d}_labeled.obj

When sim tree_id differs from the LPY filename index (e.g. tree_19.obj → LPy_envy_00009),
pass both --tree-id and --lpy-index.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from pathlib import Path

LPY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LPY_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from pybullet_tree_sim import MESHES_PATH, TEXTURES_PATH  # noqa: E402
from pybullet_tree_sim.tree import Tree  # noqa: E402

log = logging.getLogger(__name__)

DATASET_OBJ = LPY_ROOT / "dataset" / "v3" / "obj"
TEXTURED_OBJ = LPY_ROOT / "dataset" / "v3" / "textured_obj"
PB_UNLABELED = Path(MESHES_PATH) / "trees" / "envy" / "unlabeled" / "obj"
PB_LABELED = Path(MESHES_PATH) / "trees" / "envy" / "labeled" / "obj"

TREE_OBJ_RE = re.compile(r"^tree_(\d+)\.obj$", re.IGNORECASE)
OBJ_MTL_MATERIAL = Tree._obj_mtl_material_name


def make_id_str(
    tree_id: int,
    *,
    namespace: str = "LPy",
    tree_type: str = "envy",
) -> str:
    return f"{namespace}_{tree_type}_{tree_id:05d}"


def lpy_textured_path(lpy_index: int) -> Path:
    return TEXTURED_OBJ / f"tree_{lpy_index}.obj"


def lpy_labeled_path(lpy_index: int) -> Path:
    return DATASET_OBJ / f"tree_{lpy_index}.obj"


def pb_unlabeled_obj_path(id_str: str) -> Path:
    return PB_UNLABELED / f"{id_str}.obj"


def pb_unlabeled_mtl_path(id_str: str) -> Path:
    return PB_UNLABELED / f"{id_str}_labeled.mtl"


def pb_labeled_obj_path(id_str: str) -> Path:
    return PB_LABELED / f"{id_str}_labeled.obj"


def resolve_bark_texture() -> Path | None:
    path = Tree.resolve_bark_texture_path()
    return Path(path) if path else None


def write_pybullet_mtl(mtl_path: Path, *, unlabeled_dir: Path) -> None:
    """Write MTL matching Tree.ensure_obj_mtl() (relative map_Kd, no EXR)."""
    tex = resolve_bark_texture()
    if tex is None:
        raise FileNotFoundError(
            f"No bark texture under {TEXTURES_PATH} "
            f"(candidates: {Tree._bark_texture_candidates})"
        )
    rel_tex = os.path.relpath(tex, unlabeled_dir).replace("\\", "/")
    content = (
        "# Written by export_pybullet_tree.py (PyBullet-safe MTL)\n"
        f"newmtl {OBJ_MTL_MATERIAL}\n"
        "Ns 225.0\n"
        "Ka 1.0 1.0 1.0\n"
        "Kd 0.85 0.85 0.85\n"
        "Ks 0.5 0.5 0.5\n"
        "d 1.0\n"
        "illum 2\n"
        f"map_Kd {rel_tex}\n"
    )
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path.write_text(content, encoding="utf-8")
    log.info("Wrote MTL: %s (map_Kd=%s)", mtl_path, rel_tex)


def count_obj_stats(obj_path: Path) -> tuple[int, int]:
    verts = faces = 0
    with obj_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                verts += 1
            elif line.startswith("f "):
                faces += 1
    return verts, faces


def update_bbox(
    bbox: list[list[float]] | None,
    line: str,
) -> list[list[float]]:
    parts = line.split()
    if len(parts) < 4:
        return bbox or [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
    if bbox is None:
        return [[x, y, z], [x, y, z]]
    for i in range(3):
        bbox[0][i] = min(bbox[0][i], (x, y, z)[i])
        bbox[1][i] = max(bbox[1][i], (x, y, z)[i])
    return bbox


def export_unlabeled_textured(
    src: Path,
    dst_obj: Path,
    id_str: str,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Stream Blender OBJ → PyBullet URDF mesh (preserve v/vt/vn + quad faces).

    Stripping vt/vn and triangulating to v-only triangles breaks PyBullet rendering
    (trees 9/10 invisible). Tree 8 uses full Blender-style faces — match that here.
    """
    if not src.is_file():
        raise FileNotFoundError(src)

    mtl_name = f"{id_str}_labeled.mtl"
    dst_mtl = pb_unlabeled_mtl_path(id_str)

    if dry_run:
        verts, faces_in = count_obj_stats(src)
        log.info(
            "[dry-run] unlabeled %s → %s (%d v, %d f; preserves v/vt/vn faces)",
            src,
            dst_obj,
            verts,
            faces_in,
        )
        return {"vertices": verts, "faces_in": faces_in, "dry_run": True}

    dst_obj.parent.mkdir(parents=True, exist_ok=True)
    verts = 0
    faces_out = 0
    bbox: list[list[float]] | None = None
    usemtl_written = False
    _skip = ("mtllib ", "o ", "usemtl ", "s ", "g ")

    with src.open(encoding="utf-8", errors="replace") as fin, dst_obj.open(
        "w", encoding="utf-8"
    ) as fout:
        fout.write("# Exported by export_pybullet_tree.py (PyBullet / Blender-style OBJ)\n")
        fout.write(f"mtllib {mtl_name}\n")
        fout.write(f"o tree_{id_str}\n")

        for line in fin:
            if line.startswith(_skip):
                continue
            if not (
                line.startswith("v ")
                or line.startswith("vn ")
                or line.startswith("vt ")
                or line.startswith("f ")
            ):
                continue
            if line.startswith("f "):
                if not usemtl_written:
                    fout.write(f"usemtl {OBJ_MTL_MATERIAL}\n")
                    fout.write("s 1\n")
                    usemtl_written = True
                fout.write(line)
                faces_out += 1
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) == 4:
                    verts += 1
                    bbox = update_bbox(bbox, line)
            fout.write(line)

    write_pybullet_mtl(dst_mtl, unlabeled_dir=PB_UNLABELED)
    report = {
        "vertices": verts,
        "faces_out": faces_out,
        "obj": str(dst_obj),
        "mtl": str(dst_mtl),
    }
    if bbox:
        report["bbox_min"] = bbox[0]
        report["bbox_max"] = bbox[1]
    log.info(
        "Unlabeled export %s: %d vertices, %d faces (v/vt/vn) → %s",
        id_str,
        verts,
        faces_out,
        dst_obj,
    )
    return report


def export_labeled_meshlab(
    src: Path,
    dst: Path,
    id_str: str,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Copy MeshLab labeled OBJ with renamed header (vertex RGB preserved)."""
    if not src.is_file():
        raise FileNotFoundError(src)

    verts, faces = count_obj_stats(src)
    if dry_run:
        log.info(
            "[dry-run] labeled %s → %s (%d v, %d f)",
            src,
            dst,
            verts,
            faces,
        )
        return {"vertices": verts, "faces": faces, "dry_run": True}

    dst.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "####\n"
        "#\n"
        "# OBJ File Generated by Meshlab (exported for PyBullet)\n"
        "#\n"
        "####\n"
        f"# Object {id_str}_labeled.obj\n"
        "#\n"
        f"# Vertices: {verts}\n"
        f"# Faces: {faces}\n"
        "#\n"
        "####\n"
    )

    with src.open(encoding="utf-8", errors="replace") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        fout.write(header)
        skip = True
        for line in fin:
            if skip:
                if line.startswith("vn ") or line.startswith("v "):
                    skip = False
                    fout.write(line)
                continue
            fout.write(line)

    log.info(
        "Labeled export %s: %d vertices, %d faces → %s",
        id_str,
        verts,
        faces,
        dst,
    )
    return {"vertices": verts, "faces": faces, "obj": str(dst)}


def export_one_tree(
    tree_id: int,
    lpy_index: int,
    *,
    namespace: str = "LPy",
    tree_type: str = "envy",
    labeled: bool = True,
    unlabeled: bool = True,
    regenerate_urdf: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    id_str = make_id_str(tree_id, namespace=namespace, tree_type=tree_type)
    result: dict[str, object] = {
        "tree_id": tree_id,
        "lpy_index": lpy_index,
        "id_str": id_str,
    }

    if labeled:
        src = lpy_labeled_path(lpy_index)
        dst = pb_labeled_obj_path(id_str)
        result["labeled"] = export_labeled_meshlab(src, dst, id_str, dry_run=dry_run)

    if unlabeled:
        src = lpy_textured_path(lpy_index)
        dst = pb_unlabeled_obj_path(id_str)
        result["unlabeled"] = export_unlabeled_textured(
            src, dst, id_str, dry_run=dry_run
        )

    if regenerate_urdf and not dry_run:
        urdf = Tree.regenerate_urdf_from_xacro(
            tree_id=tree_id, tree_type=tree_type, namespace=namespace
        )
        result["urdf"] = urdf
        log.info("Regenerated URDF: %s", urdf)

    return result


def discover_lpy_indices(obj_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in sorted(obj_dir.glob("tree_*.obj")):
        m = TREE_OBJ_RE.match(path.name)
        if m:
            indices.append(int(m.group(1)))
    return indices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tree-id", type=int, help="Sim / repo tree ID (e.g. 9 → LPy_envy_00009)")
    p.add_argument(
        "--lpy-index",
        type=int,
        help="LPY dataset filename index (tree_N.obj). Defaults to --tree-id.",
    )
    p.add_argument("--namespace", default="LPy")
    p.add_argument("--tree-type", default="envy")
    p.add_argument(
        "--batch-labeled",
        action="store_true",
        help="Export all dataset/obj/tree_*.obj → labeled/obj (tree_id = filename index)",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Batch labeled + textured exports (tree_id = filename index)",
    )
    p.add_argument(
        "--labeled-only",
        action="store_true",
        help="Only export dataset/obj → labeled/obj",
    )
    p.add_argument(
        "--unlabeled-only",
        action="store_true",
        help="Only export textured_obj → unlabeled/obj",
    )
    p.add_argument(
        "--regenerate-urdf",
        action="store_true",
        help="Regenerate URDF from xacro after export",
    )
    p.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    labeled = unlabeled = True
    if args.labeled_only:
        unlabeled = False
    if args.unlabeled_only:
        labeled = False

    if args.batch or args.batch_labeled:
        indices = discover_lpy_indices(DATASET_OBJ)
        if not indices:
            log.error("No tree_*.obj files in %s", DATASET_OBJ)
            return 1
        for lpy_index in indices:
            export_one_tree(
                lpy_index,
                lpy_index,
                namespace=args.namespace,
                tree_type=args.tree_type,
                labeled=True,
                unlabeled=args.batch and not args.batch_labeled,
                regenerate_urdf=args.regenerate_urdf,
                dry_run=args.dry_run,
            )
        log.info("Batch complete: %d trees", len(indices))
        return 0

    if args.tree_id is None:
        parser.error("Provide --tree-id or use --batch / --batch-labeled")
        return 2

    lpy_index = args.lpy_index if args.lpy_index is not None else args.tree_id
    export_one_tree(
        args.tree_id,
        lpy_index,
        namespace=args.namespace,
        tree_type=args.tree_type,
        labeled=labeled,
        unlabeled=unlabeled,
        regenerate_urdf=args.regenerate_urdf,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
