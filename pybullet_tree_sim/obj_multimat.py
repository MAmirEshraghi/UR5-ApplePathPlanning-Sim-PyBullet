"""Split multi-material OBJs for PyBullet (one texture per URDF link)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from zenlog import log


def parts_subdir(unlabeled_obj_dir: Path, id_str: str) -> Path:
    return unlabeled_obj_dir / "parts" / id_str


# PyBullet loads URDF child links in file order (GUI appears progressively).
URDF_PART_LINK_ORDER = (
    
    "mat_apple",
    "mat_branch",
    "mat_trunk",
    "mat_stem",
    "mat_spur",
    "mat_nontrunk",
    "mat_leaf",
)


def order_materials_for_urdf(materials: list[str]) -> list[str]:
    """Sort part stems for URDF write: small parts first, heavy mat_leaf last."""
    rank = {m: i for i, m in enumerate(URDF_PART_LINK_ORDER)}
    return sorted(materials, key=lambda m: (rank.get(m, len(URDF_PART_LINK_ORDER)), m))


def _parse_face_corners(face_line: str) -> list[tuple[int, int, int]]:
    tokens = face_line.strip().split()[1:]
    corners: list[tuple[int, int, int]] = []
    for token in tokens:
        parts = token.split("/")
        vi = int(parts[0])
        vti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        vni = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        corners.append((vi, vti, vni))
    return corners


_MTL_MAP_PREFIXES = ("map_Kd", "map_Bump", "map_Ns", "map_Ks", "map_Ka", "map_d", "map_bump")

# Fallback solid green when mat_leaf has no map_Kd (optional; prefer leaf2 textures).
LEAF_OPAQUE_KD = (0.22, 0.52, 0.18)
LEAF2_MAP_FILES = {
    "map_Kd": "leaf2_diff_4k.png",
    "map_Bump": "leaf2_nor_gl_4k.png",
    "map_Ns": "leaf2_rough_4k.png",
}


def _leaf2_map_paths_relative_to(mtl_dir: Path) -> dict[str, str]:
    from pybullet_tree_sim import TEXTURES_PATH

    leaf2_dir = Path(TEXTURES_PATH) / "leaf2"
    out: dict[str, str] = {}
    for key, fname in LEAF2_MAP_FILES.items():
        tex = leaf2_dir / fname
        if tex.is_file():
            out[key] = os.path.relpath(tex.resolve(), mtl_dir.resolve()).replace("\\", "/")
    return out


def ensure_mat_leaf_texture_maps(mtl_path: Path) -> bool:
    """Ensure mat_leaf block has leaf2 map_* (same style as apple2 / bark)."""
    if not mtl_path.is_file():
        return False
    lines = mtl_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    mtl_dir = mtl_path.parent.resolve()
    leaf_maps = _leaf2_map_paths_relative_to(mtl_dir)
    if not leaf_maps.get("map_Kd"):
        log.warning("leaf2 map_Kd missing under textures/leaf2 for %s", mtl_path)
        return False

    blocks: list[tuple[str | None, list[str]]] = []
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current, buf
        if current is not None or buf:
            blocks.append((current, buf))
        current = None
        buf = []

    for line in lines:
        if line.strip().startswith("newmtl "):
            flush()
            current = line.strip().split(None, 1)[1].strip()
            buf = [line]
        elif current is not None:
            buf.append(line)
        else:
            if blocks and blocks[-1][0] is None:
                blocks[-1][1].append(line)
            else:
                blocks.append((None, [line]))
    flush()

    changed = False
    new_lines: list[str] = []
    for name, block in blocks:
        if name != "mat_leaf":
            new_lines.extend(block)
            continue
        has_map_kd = any(
            ln.strip().lower().startswith("map_kd ") for ln in block
        )
        new_lines.extend(block)
        if not has_map_kd:
            if block and not block[-1].endswith("\n"):
                new_lines[-1] = new_lines[-1] + "\n"
            for key in ("map_Kd", "map_Bump", "map_Ns"):
                if key in leaf_maps:
                    new_lines.append(f"{key} {leaf_maps[key]}\n")
            changed = True
    if changed:
        mtl_path.write_text("".join(new_lines), encoding="utf-8")
        log.info("Restored leaf2 map_* on mat_leaf in %s", mtl_path)
    return changed


def patch_mtl_leaf_opaque_for_pybullet(mtl_path: Path) -> bool:
    """Legacy: strip mat_leaf maps for solid Kd only. Prefer ensure_mat_leaf_texture_maps."""
    if not mtl_path.is_file():
        return False
    lines = mtl_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    current: str | None = None
    out: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("newmtl "):
            current = stripped.split(None, 1)[1].strip()
            out.append(line)
            continue
        if current == "mat_leaf":
            low = stripped.lower()
            if any(low.startswith(p.lower() + " ") for p in _MTL_MAP_PREFIXES):
                changed = True
                continue
            if low.startswith("kd "):
                out.append(f"Kd {LEAF_OPAQUE_KD[0]:.6f} {LEAF_OPAQUE_KD[1]:.6f} {LEAF_OPAQUE_KD[2]:.6f}\n")
                changed = True
                continue
            if low.startswith("d "):
                out.append("d 1.0\n")
                changed = True
                continue
            if low.startswith("illum "):
                out.append("illum 1\n")
                changed = True
                continue
        out.append(line)
    if changed:
        mtl_path.write_text("".join(out), encoding="utf-8")
        log.info("Patched %s for PyBullet opaque mat_leaf", mtl_path)
    return changed


def parts_mtl_filename(id_str: str) -> str:
    return f"{id_str}_parts.mtl"


def write_parts_mtl(src_mtl: Path, dst_mtl: Path, parts_dir: Path) -> None:
    """Copy MTL for part OBJs; rewrite map_* paths relative to parts_dir."""
    parts_dir = parts_dir.resolve()
    src_base = src_mtl.parent.resolve()
    if not src_mtl.is_file():
        raise FileNotFoundError(f"Source MTL missing: {src_mtl}")

    ensure_mat_leaf_texture_maps(src_mtl)

    out_lines: list[str] = [
        f"# Written for part OBJs in {parts_dir.name}/ (map_* relative to parts dir)\n",
        f"# Source: {src_mtl.name}\n",
    ]
    missing: list[str] = []

    with src_mtl.open(encoding="utf-8", errors="replace") as fin:
        for line in fin:
            stripped = line.strip()
            rewritten = False
            for prefix in _MTL_MAP_PREFIXES:
                if stripped.lower().startswith(prefix.lower() + " "):
                    raw = stripped.split(None, 1)[1].strip().strip('"')
                    tex_path = Path(raw)
                    if tex_path.is_absolute():
                        abs_tex = tex_path
                    else:
                        abs_tex = (src_base / raw).resolve()
                    if abs_tex.is_file():
                        rel = os.path.relpath(abs_tex, parts_dir).replace("\\", "/")
                        out_lines.append(f"{prefix} {rel}\n")
                    else:
                        missing.append(raw)
                        out_lines.append(line)
                    rewritten = True
                    break
            if not rewritten:
                out_lines.append(line)

    dst_mtl.parent.mkdir(parents=True, exist_ok=True)
    with dst_mtl.open("w", encoding="utf-8") as fout:
        fout.writelines(out_lines)

    ensure_mat_leaf_texture_maps(dst_mtl)

    if missing:
        log.warning(
            "Parts MTL %s: %d map_* paths not found (from %s parent)",
            dst_mtl.name,
            len(missing),
            src_mtl.name,
        )
    log.info("Wrote parts MTL for PyBullet part meshes: %s", dst_mtl)


def refresh_parts_obj_mtllib(parts_dir: Path, parts_mtl_name: str) -> None:
    """Point existing part OBJs at the co-located *_parts.mtl."""
    for obj_path in parts_dir.glob("*.obj"):
        lines = obj_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        with obj_path.open("w", encoding="utf-8") as fout:
            for line in lines:
                if line.startswith("mtllib "):
                    fout.write(f"mtllib {parts_mtl_name}\n")
                else:
                    fout.write(line)


def _count_obj_faces(obj_path: Path) -> int:
    n = 0
    with obj_path.open(encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if line.startswith("f "):
                n += 1
    return n


def _decimate_obj_face_stride(obj_path: Path, *, target_faces: int) -> tuple[int, int]:
    """Keep every Nth face and compact vertices (no extra deps). Returns (before, after)."""
    header: list[str] = []
    v_lines: dict[int, str] = {}
    vt_lines: dict[int, str] = {}
    vn_lines: dict[int, str] = {}
    all_corners: list[list[tuple[int, int, int]]] = []

    with obj_path.open(encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if line.startswith("v "):
                v_lines[len(v_lines) + 1] = line
            elif line.startswith("vt "):
                vt_lines[len(vt_lines) + 1] = line
            elif line.startswith("vn "):
                vn_lines[len(vn_lines) + 1] = line
            elif line.startswith("f "):
                all_corners.append(_parse_face_corners(line))
            else:
                header.append(line)

    before = len(all_corners)
    if before <= target_faces or before == 0:
        return before, before

    stride = max(1, before // target_faces)
    kept_corners = [all_corners[i] for i in range(0, before, stride)][:target_faces]

    used_v: set[int] = set()
    used_vt: set[int] = set()
    used_vn: set[int] = set()
    for corners in kept_corners:
        for vi, vti, vni in corners:
            used_v.add(vi)
            if vti:
                used_vt.add(vti)
            if vni:
                used_vn.add(vni)

    v_map = {old: new for new, old in enumerate(sorted(used_v), start=1)}
    vt_map = {old: new for new, old in enumerate(sorted(used_vt), start=1)}
    vn_map = {old: new for new, old in enumerate(sorted(used_vn), start=1)}

    out: list[str] = []
    for line in header:
        if not line.startswith(("v ", "vt ", "vn ", "f ")):
            out.append(line if line.endswith("\n") else line + "\n")
    for old in sorted(used_v):
        out.append(v_lines[old])
    for old in sorted(used_vt):
        out.append(vt_lines[old])
    for old in sorted(used_vn):
        out.append(vn_lines[old])
    for corners in kept_corners:
        new_corners = [
            (v_map[vi], vt_map.get(vti, 0) if vti else 0, vn_map.get(vni, 0) if vni else 0)
            for vi, vti, vni in corners
        ]
        out.append(_format_face(new_corners))
    obj_path.write_text("".join(out), encoding="utf-8")
    return before, len(kept_corners)


def decimate_part_obj(
    obj_path: Path,
    *,
    target_faces: int = 20_000,
    material: str = "mat_leaf",
) -> dict[str, int | str | bool]:
    """Reduce part OBJ face count (PyBullet GUI). Uses trimesh if installed, else stride."""
    if not obj_path.is_file():
        raise FileNotFoundError(obj_path)
    before = _count_obj_faces(obj_path)
    if before <= target_faces:
        log.info(
            "Decimate %s: %d faces <= target %d (unchanged)",
            obj_path.name,
            before,
            target_faces,
        )
        return {
            "path": str(obj_path),
            "material": material,
            "faces_before": before,
            "faces_after": before,
            "method": "none",
        }

    method = "stride"
    after = before
    try:
        import trimesh

        mesh = trimesh.load(str(obj_path), process=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        simplified = mesh.simplify_quadric_decimation(int(target_faces))
        simplified.export(str(obj_path))
        after = _count_obj_faces(obj_path)
        method = "trimesh_quadric"
    except ImportError:
        before, after = _decimate_obj_face_stride(obj_path, target_faces=target_faces)
    except Exception as exc:
        log.warning("trimesh decimate failed for %s (%s); using stride", obj_path, exc)
        before, after = _decimate_obj_face_stride(obj_path, target_faces=target_faces)

    log.info(
        "Decimate %s: %d → %d faces (target %d, method=%s)",
        obj_path.name,
        before,
        after,
        target_faces,
        method,
    )
    return {
        "path": str(obj_path),
        "material": material,
        "faces_before": before,
        "faces_after": after,
        "method": method,
    }


def _format_face(corners: list[tuple[int, int, int]]) -> str:
    parts: list[str] = []
    for vi, vti, vni in corners:
        if vti and vni:
            parts.append(f"{vi}/{vti}/{vni}")
        elif vti:
            parts.append(f"{vi}/{vti}")
        else:
            parts.append(str(vi))
    return "f " + " ".join(parts) + "\n"


def split_obj_by_usemtl(
    src_obj: Path,
    parts_dir: Path,
    *,
    src_mtl: Path,
    id_str: str,
) -> list[str]:
    """Write one OBJ per usemtl group (vertex remapped). Returns material names in file order."""
    v_lines: dict[int, str] = {}
    vt_lines: dict[int, str] = {}
    vn_lines: dict[int, str] = {}
    faces_by_mat: dict[str, list[str]] = {}
    mat_order: list[str] = []
    current_mat: str | None = None

    with src_obj.open(encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if line.startswith("v "):
                idx = len(v_lines) + 1
                v_lines[idx] = line
            elif line.startswith("vt "):
                idx = len(vt_lines) + 1
                vt_lines[idx] = line
            elif line.startswith("vn "):
                idx = len(vn_lines) + 1
                vn_lines[idx] = line
            elif line.startswith("usemtl "):
                current_mat = line.split(None, 1)[1].strip()
                if current_mat not in faces_by_mat:
                    faces_by_mat[current_mat] = []
                    mat_order.append(current_mat)
            elif line.startswith("f ") and current_mat is not None:
                faces_by_mat[current_mat].append(line)

    if not mat_order:
        raise ValueError(f"No usemtl groups in {src_obj}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    parts_mtl_name = parts_mtl_filename(id_str)
    write_parts_mtl(src_mtl, parts_dir / parts_mtl_name, parts_dir)

    for mat in mat_order:
        faces = faces_by_mat[mat]
        v_map: dict[int, int] = {}
        vt_map: dict[int, int] = {}
        vn_map: dict[int, int] = {}
        out_faces: list[str] = []

        for face_line in faces:
            corners = _parse_face_corners(face_line)
            new_corners: list[tuple[int, int, int]] = []
            for vi, vti, vni in corners:
                if vi not in v_map:
                    v_map[vi] = len(v_map) + 1
                if vti and vti not in vt_map:
                    vt_map[vti] = len(vt_map) + 1
                if vni and vni not in vn_map:
                    vn_map[vni] = len(vn_map) + 1
                new_corners.append(
                    (
                        v_map[vi],
                        vt_map.get(vti, 0) if vti else 0,
                        vn_map.get(vni, 0) if vni else 0,
                    )
                )
            out_faces.append(_format_face(new_corners))

        part_path = parts_dir / f"{mat}.obj"
        rev_v = {new: old for old, new in v_map.items()}
        rev_vt = {new: old for old, new in vt_map.items()}
        rev_vn = {new: old for old, new in vn_map.items()}

        with part_path.open("w", encoding="utf-8") as fout:
            fout.write(f"# Split from {src_obj.name} for PyBullet (one material per link)\n")
            fout.write(f"mtllib {parts_mtl_name}\n")
            fout.write(f"o {id_str}_{mat}\n")
            for i in range(1, len(v_map) + 1):
                fout.write(v_lines[rev_v[i]])
            for i in range(1, len(vt_map) + 1):
                fout.write(vt_lines[rev_vt[i]])
            for i in range(1, len(vn_map) + 1):
                fout.write(vn_lines[rev_vn[i]])
            fout.write(f"usemtl {mat}\n")
            fout.writelines(out_faces)

        log.info(
            "Material part %s: %d faces → %s",
            mat,
            len(faces),
            part_path,
        )

    return mat_order


def write_multimaterial_tree_urdf(
    *,
    urdf_path: str,
    id_str: str,
    tree_type: str,
    namespace: str,
    materials: list[str],
    collision_mesh_rel: str,
    visual_mesh_rel_by_mat: dict[str, str],
    parent: str = "world",
    mass: float = 8.0,
) -> str:
    """URDF: one base link (collision + inertial), fixed child links per material visual."""
    robot = ET.Element("robot", name="tree")
    link_name = id_str

    base = ET.SubElement(robot, "link", name=link_name)
    collision = ET.SubElement(ET.SubElement(base, "collision", concave="true"), "geometry")
    ET.SubElement(
        collision,
        "mesh",
        filename=collision_mesh_rel,
        scale="1 1 1",
    )
    # Tiny placeholder visual prevents PyBullet from mirroring the collision mesh as a
    # second full-tree visual (depth shell that occludes part-link meshes).
    base_vis = ET.SubElement(base, "visual")
    ET.SubElement(base_vis, "origin", rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
    base_geom = ET.SubElement(base_vis, "geometry")
    ET.SubElement(base_geom, "box", size="0.001 0.001 0.001")
    base_mat = ET.SubElement(base_vis, "material", name="hidden_base_placeholder")
    ET.SubElement(base_mat, "color", rgba="0 0 0 0")
    inertial = ET.SubElement(base, "inertial")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(inertial, "origin", rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
    inertia = ET.SubElement(inertial, "inertia")
    inertia.set("ixx", "0.00443333156")
    inertia.set("ixy", "0.0")
    inertia.set("ixz", "0.0")
    inertia.set("iyy", "0.00443333156")
    inertia.set("iyz", "0.0")
    inertia.set("izz", "0.0072")

    for mat in materials:
        vis_link = f"{link_name}_vis_{mat}"
        vlink = ET.SubElement(robot, "link", name=vis_link)
        inertial = ET.SubElement(vlink, "inertial")
        ET.SubElement(inertial, "mass", value="0.001")
        ET.SubElement(inertial, "origin", rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
        inertia = ET.SubElement(inertial, "inertia")
        inertia.set("ixx", "1e-6")
        inertia.set("ixy", "0.0")
        inertia.set("ixz", "0.0")
        inertia.set("iyy", "1e-6")
        inertia.set("iyz", "0.0")
        inertia.set("izz", "1e-6")
        visual = ET.SubElement(vlink, "visual")
        geom = ET.SubElement(visual, "geometry")
        ET.SubElement(
            geom,
            "mesh",
            filename=visual_mesh_rel_by_mat[mat],
            scale="1 1 1",
        )

        joint = ET.SubElement(robot, "joint", name=f"{vis_link}_joint", type="fixed")
        ET.SubElement(joint, "parent", link=link_name)
        ET.SubElement(joint, "child", link=vis_link)
        ET.SubElement(joint, "origin", rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")

    ET.SubElement(robot, "link", name="world")
    root_joint = ET.SubElement(robot, "joint", name=f"{link_name}_joint", type="fixed")
    ET.SubElement(root_joint, "parent", link=parent)
    ET.SubElement(root_joint, "child", link=link_name)
    ET.SubElement(root_joint, "origin", rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")

    xml_str = minidom.parseString(ET.tostring(robot, encoding="unicode")).toprettyxml(indent="\t")
    xml_str = re.sub(r'<\?xml version="1.0" \?>\n', '<?xml version="1.0" ?>\n', xml_str)
    os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    log.info("Saved multi-material URDF to '%s' (%d visual links)", urdf_path, len(materials))
    return urdf_path
