"""
Blender script: assign PBR textures per tree component using vertex colors
from labeled OBJ/PLY exports (Camp_Envy_tie_prune_label6_2.lpy).

Run inside Blender (Scripting workspace) or:
  blender --background --python add_texture_by_color_v2.py
"""
import glob
import os
import bpy
from math import sqrt

# =============================================================================
# CONFIG — edit paths and component → texture mapping
# =============================================================================

TREE_INPUT_DIR = "/home/robben/codes/TreeSim/ok/lpy_treesim/dataset/obj"
TREE_OUTPUT_DIR = "/home/robben/codes/TreeSim/ok/lpy_treesim/dataset/textured_obj"
INPUT_EXT = ".obj"  # ".obj" or ".ply"

TEXTURE_ROOT = "/home/robben/codes/TreeSim/ok/pybullet-tree-sim/pybullet_tree_sim/textures"

# Label scheme (Camp_Envy_tie_prune_label6_2.lpy): B-only wood/leaf/stem; apples = 200 label IDs.
# Levels 51,102,153,204,255: 75 two-channel (one zero) + 125 full RGB.
COMPONENT_MAP = [
    {
        "name": "apple",
        "rgb": (51, 51, 51),
        "match_apple_label": True,
        "texture_dir": os.path.join(TEXTURE_ROOT, "apple2"),
    },
    {
        "name": "leaf",
        "rgb": (0, 0, 128),
        "tolerance": 15,
        "texture_dir": os.path.join(TEXTURE_ROOT, "leaf2"),
    },
    {
        "name": "stem",
        "rgb": (0, 0, 153),
        "tolerance": 15,
        "texture_dir": os.path.join(TEXTURE_ROOT, "bark_willow"),
    },
    {
        "name": "trunk",
        "rgb": (0, 0, 26),
        "tolerance": 15,
        "texture_dir": os.path.join(TEXTURE_ROOT, "bark_willow"),
    },
    {
        "name": "branch",
        "rgb": (0, 0, 51),
        "tolerance": 15,
        "texture_dir": os.path.join(TEXTURE_ROOT, "bark_willow_02"),
    },
    {
        "name": "nontrunk",
        "rgb": (0, 0, 77),
        "tolerance": 15,
        "texture_dir": os.path.join(TEXTURE_ROOT, "bark_willow_02"),
    },
    {
        "name": "spur",
        "rgb": (0, 0, 102),
        "tolerance": 15,
        "texture_dir": os.path.join(TEXTURE_ROOT, "bark_willow"),
    },
]

COLOR_TOLERANCE_DEFAULT = 15
APPLE_ZERO_CHANNEL_MAX = 20
APPLE_LABEL_LEVELS = (51, 102, 153, 204, 255)
APPLE_LEVEL_TOLERANCE = 8
# Log face counts per component and diffuse paths used.
DEBUG = True
# Use [] to read files directly from TREE_INPUT_DIR; or ["train", "test"] for subfolders.
SUBDIRS = []

# Never use these as diffuse (height/roughness/normal maps).
_NON_DIFFUSE_SUBSTR = ("disp", "rough", "nor", "normal", "arm", "ao", "metal", "opacity", "bump")


def set_image_noncolor_colorspace(image):
    """Blender 3.0 has Linear/sRGB only; 3.4+ uses Non-Color for data maps."""
    for name in ("Non-Color", "Linear"):
        try:
            image.colorspace_settings.name = name
            return
        except TypeError:
            continue


def _list_texture_files(folder):
    if not os.path.isdir(folder):
        return []
    return [n for n in os.listdir(folder) if n.lower().endswith((".jpg", ".jpeg", ".png", ".exr"))]


def _find_map(folder, patterns, exclude_non_diffuse=False):
    """Pick first filename matching patterns (ordered). patterns are substrings."""
    if not os.path.isdir(folder):
        return None
    names = sorted(_list_texture_files(folder))
    for pattern in patterns:
        for name in names:
            lower = name.lower()
            if exclude_non_diffuse and any(s in lower for s in _NON_DIFFUSE_SUBSTR):
                continue
            if pattern in lower:
                return os.path.join(folder, name)
    return None


def find_diffuse_map(folder):
    path = _find_map(
        folder,
        ["_diff_4k.jpg", "_diff_4k.png", "_diff.jpg", "_diff.png", "albedo", "basecolor", "_col_4k"],
        exclude_non_diffuse=True,
    )
    if path:
        return path
    for name in _list_texture_files(folder):
        lower = name.lower()
        if any(s in lower for s in _NON_DIFFUSE_SUBSTR):
            continue
        return os.path.join(folder, name)
    return None


def create_pbr_material(mat_name, texture_dir):
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    diff = find_diffuse_map(texture_dir)
    if diff is None:
        raise FileNotFoundError(f"No diffuse (*_diff_* / albedo) in {texture_dir}")
    if DEBUG:
        print(f"  material {mat_name}: diffuse <- {os.path.basename(diff)}")

    tex_diff = nodes.new("ShaderNodeTexImage")
    tex_diff.image = bpy.data.images.load(diff, check_existing=True)
    links.new(tex_diff.outputs["Color"], bsdf.inputs["Base Color"])

    nor = _find_map(texture_dir, ["_nor_gl_4k.exr", "_nor_gl_4k.png", "_nor_gl", "_normal"])
    if nor:
        tex_n = nodes.new("ShaderNodeTexImage")
        tex_n.image = bpy.data.images.load(nor, check_existing=True)
        set_image_noncolor_colorspace(tex_n.image)
        nmap = nodes.new("ShaderNodeNormalMap")
        links.new(tex_n.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    rough = _find_map(texture_dir, ["_rough_4k.exr", "_rough_4k.jpg", "_rough"])
    if rough:
        tex_r = nodes.new("ShaderNodeTexImage")
        tex_r.image = bpy.data.images.load(rough, check_existing=True)
        set_image_noncolor_colorspace(tex_r.image)
        links.new(tex_r.outputs["Color"], bsdf.inputs["Roughness"])

    disp = _find_map(texture_dir, ["_disp_4k.png", "_disp_4k.jpg", "_disp", "displacement", "height"])
    if disp:
        tex_d = nodes.new("ShaderNodeTexImage")
        tex_d.image = bpy.data.images.load(disp, check_existing=True)
        set_image_noncolor_colorspace(tex_d.image)
        disp_n = nodes.new("ShaderNodeDisplacement")
        links.new(tex_d.outputs["Color"], disp_n.inputs["Height"])
        links.new(disp_n.outputs["Displacement"], out.inputs["Displacement"])

    return mat


def color_distance(c1, c2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def _at_label_level(value):
    for level in APPLE_LABEL_LEVELS:
        if abs(value - level) <= APPLE_LEVEL_TOLERANCE:
            return True
    return False


def _is_apple_label(r, g, b):
    """75 pair (one channel ~0) or 125 triple; excludes wood (0,0,B). Matches .lpy APPLE_COLORS."""
    if r <= APPLE_ZERO_CHANNEL_MAX and g <= APPLE_ZERO_CHANNEL_MAX:
        return False
    z = sum(v <= APPLE_ZERO_CHANNEL_MAX for v in (r, g, b))
    on = (_at_label_level(r), _at_label_level(g), _at_label_level(b))
    if z == 1:
        return sum(on) == 2
    if z == 0:
        return all(on)
    return False


def match_component(rgb):
    r, g, b = rgb
    if _is_apple_label(r, g, b):
        return "apple"
    # Wood / leaf / stem: (0, 0, B) — match on blue channel
    if r <= APPLE_ZERO_CHANNEL_MAX and g <= APPLE_ZERO_CHANNEL_MAX:
        best = None
        best_d = 1e9
        for comp in COMPONENT_MAP:
            if comp.get("match_apple_label"):
                continue
            cb = comp["rgb"][2]
            d = abs(b - cb)
            tol = comp.get("tolerance", COLOR_TOLERANCE_DEFAULT)
            if d <= tol and d < best_d:
                best_d = d
                best = comp
        if best:
            return best["name"]
    return "default"


def _pos_key(x, y, z, decimals=4):
    m = 10 ** decimals
    return (int(round(x * m)), int(round(y * m)), int(round(z * m)))


def parse_obj_colored_vertices(filepath):
    """MeshLab OBJ: v x y z r g b (RGB 0-1). Returns [(x,y,z,(r,g,b)), ...] in file order."""
    verts = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
            verts.append(
                (x, y, z, (int(r * 255), int(g * 255), int(b * 255)))
            )
    return verts if verts else None


def vertex_colors_from_obj_positions(mesh, filepath):
    """Map labels by vertex position (Blender reorders vertices vs OBJ file order)."""
    obj_verts = parse_obj_colored_vertices(filepath)
    if not obj_verts:
        return None

    color_by_pos = {}
    for x, y, z, rgb in obj_verts:
        key = _pos_key(x, y, z)
        if key not in color_by_pos:
            color_by_pos[key] = rgb

    cols = []
    matched = 0
    for v in mesh.vertices:
        co = v.co
        rgb = color_by_pos.get(_pos_key(co.x, co.y, co.z))
        if rgb is None:
            cols.append((180, 180, 180))
        else:
            cols.append(rgb)
            matched += 1

    if DEBUG:
        print(
            f"  vertex colors: matched {matched}/{len(mesh.vertices)} "
            f"from {len(obj_verts)} OBJ vertices"
        )
    return cols


def get_vertex_colors_from_mesh(mesh):
    n = len(mesh.vertices)
    cols = [(180, 180, 180)] * n

    if mesh.vertex_colors:
        layer = mesh.vertex_colors.active or mesh.vertex_colors[0]
        for poly in mesh.polygons:
            for i, vi in enumerate(poly.vertices):
                c = layer.data[poly.loop_indices[i]].color
                cols[vi] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
        return cols

    for attr_name in ("Col", "color", "CD"):
        if attr_name in mesh.attributes:
            attr = mesh.attributes[attr_name]
            if attr.domain == "CORNER":
                for poly in mesh.polygons:
                    for li in poly.loop_indices:
                        vi = mesh.loops[li].vertex_index
                        c = attr.data[li].color
                        cols[vi] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            elif attr.domain == "POINT":
                for i in range(n):
                    c = attr.data[i].color
                    cols[i] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            return cols
    return None


def resolve_vertex_colors(mesh, source_path):
    if source_path.lower().endswith(".obj"):
        vcols = vertex_colors_from_obj_positions(mesh, source_path)
        if vcols is not None:
            return vcols
    vcols = get_vertex_colors_from_mesh(mesh)
    if vcols is not None:
        if DEBUG:
            print("  vertex colors: from mesh color attribute / vertex_colors layer")
        return vcols
    return None


def assign_materials_by_vertex_color(obj, vcols):
    mesh = obj.data
    if vcols is None:
        raise RuntimeError(
            "No vertex colors found. Use labeled OBJ from convert_ply_to_obj (MeshLab) or labeled PLY."
        )

    name_to_slot = {}
    mesh.materials.clear()

    for comp in COMPONENT_MAP:
        mat = create_pbr_material(f"mat_{comp['name']}", comp["texture_dir"])
        name_to_slot[comp["name"]] = len(mesh.materials)
        mesh.materials.append(mat)

    trunk_entry = next(c for c in COMPONENT_MAP if c["name"] == "trunk")
    default_mat = create_pbr_material("mat_default", trunk_entry["texture_dir"])
    slot_default = len(mesh.materials)
    mesh.materials.append(default_mat)

    face_counts = {c["name"]: 0 for c in COMPONENT_MAP}
    face_counts["default"] = 0

    for poly in mesh.polygons:
        rs, gs, bs, cnt = 0, 0, 0, 0
        for vi in poly.vertices:
            if vi >= len(vcols):
                continue
            r, g, b = vcols[vi]
            rs += r
            gs += g
            bs += b
            cnt += 1
        if cnt == 0:
            poly.material_index = slot_default
            continue
        avg = (rs // cnt, gs // cnt, bs // cnt)
        cname = match_component(avg)
        if cname in name_to_slot:
            poly.material_index = name_to_slot[cname]
            face_counts[cname] = face_counts.get(cname, 0) + 1
        else:
            poly.material_index = slot_default
            face_counts["default"] += 1

    if DEBUG:
        total = len(mesh.polygons)
        print(f"  face assignment ({total} faces):")
        for name, count in sorted(face_counts.items(), key=lambda x: -x[1]):
            if count:
                print(f"    {name}: {count} ({100.0 * count / total:.1f}%)")


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=path)
        except (AttributeError, RuntimeError):
            bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".ply":
        try:
            bpy.ops.wm.ply_import(filepath=path)
        except (AttributeError, RuntimeError):
            bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise ValueError(f"Unsupported input format: {ext}")


def export_obj(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        bpy.ops.wm.obj_export(
            filepath=path,
            export_uv=True,
            export_materials=True,
            export_colors=False,
        )
    except (AttributeError, RuntimeError):
        bpy.ops.export_scene.obj(filepath=path, use_uvs=True, use_materials=True)


def join_mesh_objects(objects):
    if len(objects) == 1:
        return objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objects:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def process_one_tree(in_path, out_path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    import_mesh(in_path)
    mesh_objects = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh imported from {in_path}")

    obj = join_mesh_objects(mesh_objects)
    obj.name = "tree"
    bpy.context.view_layer.objects.active = obj

    vcols = resolve_vertex_colors(obj.data, in_path)
    assign_materials_by_vertex_color(obj, vcols)

    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.cube_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    export_obj(out_path)
    print("Saved:", out_path)


def collect_input_files():
    pattern = "*" + INPUT_EXT
    if SUBDIRS:
        files = []
        for sub in SUBDIRS:
            files.extend(glob.glob(os.path.join(TREE_INPUT_DIR, sub, pattern)))
        return sorted(files)
    return sorted(glob.glob(os.path.join(TREE_INPUT_DIR, pattern)))


def main():
    os.makedirs(TREE_OUTPUT_DIR, exist_ok=True)
    input_files = collect_input_files()
    if not input_files:
        print(f"No {INPUT_EXT} files found in", TREE_INPUT_DIR)
        return
    for in_path in input_files:
        base = os.path.splitext(os.path.basename(in_path))[0]
        if SUBDIRS:
            sub = os.path.basename(os.path.dirname(in_path))
            out_path = os.path.join(TREE_OUTPUT_DIR, sub, base + ".obj")
        else:
            out_path = os.path.join(TREE_OUTPUT_DIR, base + ".obj")
        print("Processing:", in_path)
        process_one_tree(in_path, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
