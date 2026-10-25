#!/usr/bin/env python3
"""
Make pybullet_tree_sim/urdf/tmp/robot.urdf portable: copy every mesh it references by absolute path
into urdf/tmp/robot_meshes/ and rewrite the reference as a relative path (PyBullet resolves relative
mesh paths from the URDF's folder). Run after regenerating robot.urdf from xacro, which needs ROS 2
and writes absolute paths. Already-relative references are left alone, so re-running is safe.

    python tools/bundle_robot_meshes.py [--urdf pybullet_tree_sim/urdf/tmp/robot.urdf]
"""
import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = Path(__file__).resolve().parents[1] / "pybullet_tree_sim" / "urdf" / "tmp" / "robot.urdf"
    parser.add_argument("--urdf", type=Path, default=default)
    args = parser.parse_args()
    print(f"{bundle(args.urdf)} absolute mesh reference(s) bundled in {args.urdf}")


if __name__ == "__main__":
    main()
