"""
Keep robot.urdf portable: copy every mesh it references by absolute path into robot_meshes/ next to the
URDF and rewrite the reference as a relative path (PyBullet resolves relative mesh paths from the URDF's
folder). Robot._generate_robot_urdf calls this after regenerating from xacro; already-relative references
are left alone, so calling it again is safe.
"""
import re
import shutil
from pathlib import Path

# Source path marker -> folder under robot_meshes/. The part of the path after the marker is kept.
SOURCES = {
    "/ur_description/meshes/ur5e/": "ur5e/",
    "/meshes/base_mounts/farm-ng/amiga/": "amiga/",
    "/pybullet_tree_sim/meshes/end_effectors/": "end_effectors/",
}


def bundle(urdf: Path) -> int:
    text = urdf.read_text()
    out_root = urdf.parent / "robot_meshes"
    changed = 0
    for ref in sorted(set(re.findall(r'filename="([^"]+)"', text))):
        src = ref[len("file://"):] if ref.startswith("file://") else ref
        if not src.startswith("/"):
            continue
        marker = next((m for m in SOURCES if m in src), None)
        if marker is None:
            raise SystemExit(f"No robot_meshes/ folder known for {src}; add its marker to SOURCES.")
        rel = SOURCES[marker] + src.split(marker, 1)[1]
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        text = text.replace(f'filename="{ref}"', f'filename="robot_meshes/{rel}"')
        changed += 1
    urdf.write_text(text)
    return changed
