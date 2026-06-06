#!/usr/bin/env python3
"""Export lpy_treesim meshes for PyBullet with **full Blender materials** (v3).

Same outputs as v2 (same paths — current viewer/planner need no changes). v3 adds
**decimated mat_leaf** part mesh for faster PyBullet GUI (v2 unchanged for rollback).

Unlike export_pybullet_tree.py (single mat_bark_willow), v3 keeps:
  - All usemtl groups (mat_apple, mat_leaf, mat_trunk, …)
  - v / vt / vn and quad faces
  - Per-material map_Kd, map_Bump, map_Ns (relative paths; PNG/JPG/EXR like Blender)

Reads dataset/textured_obj/tree_{lpy_index}.obj from add_texture_by_color_v2.py.

Writes:
  - unlabeled/obj/LPy_envy_{tree_id:05d}.obj          → mtllib …_textured.mtl
  - unlabeled/obj/LPy_envy_{tree_id:05d}_textured.mtl
  - labeled/obj/LPy_envy_{tree_id:05d}_labeled.obj

PyBullet loads textures from the OBJ mtllib line (*_textured.mtl). Because PyBullet
supports only one material per link, v2 also splits the OBJ into parts/…/{mat}.obj and
--regenerate-urdf builds a multi-link URDF (one visual link per material).

Example:

    python lpy_treesim/export_pybullet_tree_v3.py --tree-id 17 --lpy-index 17 --regenerate-urdf

    PYTHONPATH=. python feature_apple_path_planning/playground/tests/run_sim_tree_viewer.py \\
        --tree-id 17 --texture --with-robot --no-room --hold-gui

Rollback to v2 meshes: re-run export_pybullet_tree_v2.py for the same --tree-id.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

LPY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LPY_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LPY_ROOT))

from pybullet_tree_sim import TEXTURES_PATH  # noqa: E402
from pybullet_tree_sim.tree import Tree  # noqa: E402

from export_pybullet_tree import (  # noqa: E402
    DATASET_OBJ,
    PB_UNLABELED,
    TEXTURED_OBJ,
    count_obj_stats,
    discover_lpy_indices,
    export_labeled_meshlab,
    lpy_labeled_path,
    lpy_textured_path,
    make_id_str,
    pb_labeled_obj_path,
    pb_unlabeled_obj_path,
    update_bbox,
)

log = logging.getLogger(__name__)

TEXTURES_MARKER = "/textures/"
MTL_MAP_LINES = ("map_Kd", "map_Bump", "map_Ns")
MTL_SCALAR_KEYS = ("Ns", "Ka", "Kd", "Ks", "d", "illum", "Ni", "Ke")

MATERIAL_TEXTURE_DIRS: dict[str, str] = {
    "mat_apple": "apple2",
    "mat_leaf": "leaf2",
    "mat_branch": "bark_willow",
    "mat_stem": "bark_willow",
    "mat_trunk": "bark_willow",
    "mat_spur": "bark_willow",
    "mat_nontrunk": "bark_willow_02",
}

MAP_SEARCH: dict[str, list[str]] = {
    "map_Kd": ["_diff_4k.jpg", "_diff_4k.png", "_diff.jpg", "_diff.png", "albedo", "basecolor"],
    "map_Bump": ["_nor_gl_4k.png", "_nor_gl_4k.exr", "_nor_gl_4k.jpg", "_nor_gl", "_normal"],
    "map_Ns": ["_rough_4k.jpg", "_rough_4k.png", "_rough_4k.exr", "_rough"],
}


def pb_textured_mtl_path(id_str: str) -> Path:
    return PB_UNLABELED / f"{id_str}_textured.mtl"


def lpy_source_mtl_path(lpy_index: int) -> Path:
    return TEXTURED_OBJ / f"tree_{lpy_index}.mtl"


def _list_texture_files(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(
        n
        for n in os.listdir(folder)
        if n.lower().endswith((".jpg", ".jpeg", ".png", ".exr"))
    )


def find_texture_in_folder(folder: Path, patterns: list[str]) -> Path | None:
    names = _list_texture_files(folder)
    for pattern in patterns:
        for name in names:
            if pattern in name.lower():
                path = folder / name
                if path.is_file():
                    return path
    return None


def resolve_texture_file(raw_path: str, *, mtl_dir: Path) -> Path | None:
    raw = raw_path.strip().strip('"')
    if not raw:
        return None

    normalized = raw.replace("\\", "/")
    full: Path | None = None
    lower = normalized.lower()

    if TEXTURES_MARKER in lower:
        idx = lower.index("/textures/") + len("/textures/")
        full = Path(TEXTURES_PATH) / normalized[idx:]
    elif os.path.isabs(normalized):
        candidate = Path(normalized)
        if candidate.is_file():
            full = candidate
        else:
            matches = list(Path(TEXTURES_PATH).rglob(Path(normalized).name))
            if matches:
                full = matches[0]
    else:
        candidate = (mtl_dir / normalized).resolve()
        if candidate.is_file():
            full = candidate

    if full is not None and full.is_file():
        return full.resolve()
    return None


def relative_texture_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace("\\", "/")


def material_texture_folder(mat_name: str) -> Path | None:
    rel = MATERIAL_TEXTURE_DIRS.get(mat_name)
    if not rel:
        return None
    folder = Path(TEXTURES_PATH) / rel
    return folder if folder.is_dir() else None


def resolve_material_maps(
    mat_name: str,
    block_lines: list[str],
    *,
    mtl_dir: Path,
) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for line in block_lines:
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        key, rest = parts[0], parts[1]
        if key not in MTL_MAP_LINES:
            continue
        resolved = resolve_texture_file(rest, mtl_dir=mtl_dir)
        if resolved is not None:
            found[key] = resolved

    folder = material_texture_folder(mat_name)
    if folder is not None:
        for key, patterns in MAP_SEARCH.items():
            if key in found:
                continue
            hit = find_texture_in_folder(folder, patterns)
            if hit is not None:
                found[key] = hit.resolve()

    return found


def convert_blender_mtl_to_pybullet(src_mtl: Path, dst_mtl: Path) -> list[str]:
    """Parse Blender MTL → unified MTL (map_Kd/Bump/Ns, paths relative to OBJ dir)."""
    if not src_mtl.is_file():
        raise FileNotFoundError(f"Source MTL missing: {src_mtl}")

    dst_mtl.parent.mkdir(parents=True, exist_ok=True)
    mtl_dir = dst_mtl.parent.resolve()
    material_names: list[str] = []
    out_lines: list[str] = [
        "# Written by export_pybullet_tree_v3.py (unified MTL: viewer + PyBullet)\n",
        f"# Source: {src_mtl.name}\n",
        f"# Textures root: {TEXTURES_PATH}\n",
    ]

    current_name: str | None = None
    block_lines: list[str] = []

    def flush_block() -> None:
        nonlocal current_name, block_lines
        if current_name is None:
            block_lines = []
            return
        material_names.append(current_name)
        out_lines.append(f"newmtl {current_name}\n")

        for line in block_lines:
            parts = line.split(None, 1)
            if len(parts) < 1:
                continue
            key = parts[0]
            if key in MTL_SCALAR_KEYS:
                out_lines.append(line if line.endswith("\n") else line + "\n")

        maps = resolve_material_maps(current_name, block_lines, mtl_dir=mtl_dir)
        for key in MTL_MAP_LINES:
            if key in maps:
                out_lines.append(f"{key} {relative_texture_path(maps[key], mtl_dir)}\n")
        if "map_Kd" not in maps:
            log.warning("Material %s has no map_Kd after conversion (may render flat)", current_name)
        out_lines.append("\n")
        current_name = None
        block_lines = []

    with src_mtl.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("newmtl "):
                flush_block()
                current_name = stripped.split(None, 1)[1].strip()
                continue
            if current_name is not None:
                block_lines.append(line if line.endswith("\n") else line + "\n")
        flush_block()

    dst_mtl.write_text("".join(out_lines), encoding="utf-8")
    from pybullet_tree_sim.obj_multimat import ensure_mat_leaf_texture_maps

    ensure_mat_leaf_texture_maps(dst_mtl)
    log.info(
        "Wrote textured MTL: %s (%d materials: %s)",
        dst_mtl,
        len(material_names),
        ", ".join(material_names),
    )
    return material_names


def export_unlabeled_multimat(
    src_obj: Path,
    src_mtl: Path,
    dst_obj: Path,
    dst_mtl: Path,
    id_str: str,
    *,
    dry_run: bool = False,
    decimate_leaf: bool = True,
    leaf_target_faces: int = 20_000,
) -> dict[str, object]:
    """Stream Blender OBJ preserving usemtl + v/vt/vn; write PyBullet *_textured.mtl."""
    if not src_obj.is_file():
        raise FileNotFoundError(src_obj)

    mtl_name = dst_mtl.name

    if dry_run:
        verts, faces = count_obj_stats(src_obj)
        mats: list[str] = []
        if src_mtl.is_file():
            with src_mtl.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("newmtl "):
                        mats.append(line.split(None, 1)[1].strip())
        log.info(
            "[dry-run] v3 unlabeled %s → %s (%d v, %d f, materials: %s)",
            src_obj,
            dst_obj,
            verts,
            faces,
            mats,
        )
        return {"vertices": verts, "faces": faces, "materials": mats, "dry_run": True}

    materials = convert_blender_mtl_to_pybullet(src_mtl, dst_mtl)

    dst_obj.parent.mkdir(parents=True, exist_ok=True)
    verts = 0
    faces = 0
    usemtl_seen: set[str] = set()
    bbox = None
    _skip = ("mtllib ", "o ")

    with src_obj.open(encoding="utf-8", errors="replace") as fin, dst_obj.open(
        "w", encoding="utf-8"
    ) as fout:
        fout.write("# Exported by export_pybullet_tree_v3.py (multi-material PyBullet mesh)\n")
        fout.write(f"mtllib {mtl_name}\n")
        fout.write(f"o tree_{id_str}\n")

        for line in fin:
            if line.startswith(_skip):
                continue
            if line.startswith("usemtl "):
                name = line.split(None, 1)[1].strip()
                usemtl_seen.add(name)
                fout.write(line)
                continue
            if line.startswith("f "):
                faces += 1
                fout.write(line)
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) == 4:
                    verts += 1
                    bbox = update_bbox(bbox, line)
                    fout.write(line)
                continue
            if line.startswith(("vn ", "vt ", "s ")):
                fout.write(line)
                continue

    unknown = usemtl_seen - set(materials)
    if unknown:
        log.warning(
            "OBJ usemtl not in converted MTL: %s (faces may render without map_Kd)",
            sorted(unknown),
        )

    report: dict[str, object] = {
        "vertices": verts,
        "faces": faces,
        "materials": materials,
        "usemtl_in_obj": sorted(usemtl_seen),
        "obj": str(dst_obj),
        "mtl": str(dst_mtl),
    }
    if bbox:
        report["bbox_min"] = bbox[0]
        report["bbox_max"] = bbox[1]
    log.info(
        "Unlabeled v3 export %s: %d vertices, %d faces, %d materials → %s",
        id_str,
        verts,
        faces,
        len(materials),
        dst_obj,
    )

    from pybullet_tree_sim.obj_multimat import (
        decimate_part_obj,
        parts_subdir,
        split_obj_by_usemtl,
    )

    parts_dir = parts_subdir(PB_UNLABELED, id_str)
    mat_order = split_obj_by_usemtl(
        dst_obj,
        parts_dir,
        src_mtl=dst_mtl,
        id_str=id_str,
    )
    report["material_parts"] = mat_order
    report["parts_dir"] = str(parts_dir)

    if decimate_leaf and "mat_leaf" in mat_order:
        leaf_path = parts_dir / "mat_leaf.obj"
        report["leaf_decimate"] = decimate_part_obj(
            leaf_path,
            target_faces=leaf_target_faces,
            material="mat_leaf",
        )

    return report


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
    decimate_leaf: bool = True,
    leaf_target_faces: int = 20_000,
) -> dict[str, object]:
    id_str = make_id_str(tree_id, namespace=namespace, tree_type=tree_type)
    result: dict[str, object] = {
        "tree_id": tree_id,
        "lpy_index": lpy_index,
        "id_str": id_str,
    }

    if labeled:
        result["labeled"] = export_labeled_meshlab(
            lpy_labeled_path(lpy_index),
            pb_labeled_obj_path(id_str),
            id_str,
            dry_run=dry_run,
        )

    if unlabeled:
        result["unlabeled"] = export_unlabeled_multimat(
            lpy_textured_path(lpy_index),
            lpy_source_mtl_path(lpy_index),
            pb_unlabeled_obj_path(id_str),
            pb_textured_mtl_path(id_str),
            id_str,
            dry_run=dry_run,
            decimate_leaf=decimate_leaf,
            leaf_target_faces=leaf_target_faces,
        )

    if regenerate_urdf and not dry_run:
        urdf = Tree.regenerate_textured_urdf(
            tree_id=tree_id,
            tree_type=tree_type,
            namespace=namespace,
        )
        result["urdf"] = urdf
        log.info("Regenerated multi-material URDF: %s", urdf)

    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tree-id", type=int, help="Sim tree ID (e.g. 12)")
    p.add_argument("--lpy-index", type=int, help="LPY tree_N index (default: --tree-id)")
    p.add_argument("--namespace", default="LPy")
    p.add_argument("--tree-type", default="envy")
    p.add_argument("--batch-labeled", action="store_true")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--labeled-only", action="store_true")
    p.add_argument("--unlabeled-only", action="store_true")
    p.add_argument("--regenerate-urdf", action="store_true")
    p.add_argument(
        "--no-decimate-leaf",
        action="store_true",
        help="Keep full mat_leaf part mesh (v2-sized; slow in PyBullet)",
    )
    p.add_argument(
        "--leaf-target-faces",
        type=int,
        default=20_000,
        help="Target faces for mat_leaf.obj after split (default 20000)",
    )
    p.add_argument("--dry-run", action="store_true")
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

    decimate_leaf = not args.no_decimate_leaf
    leaf_target_faces = max(1000, int(args.leaf_target_faces))

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
                decimate_leaf=decimate_leaf,
                leaf_target_faces=leaf_target_faces,
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
        decimate_leaf=decimate_leaf,
        leaf_target_faces=leaf_target_faces,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
