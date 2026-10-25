#!/usr/bin/env python3
"""
Make pybullet_tree_sim/urdf/tmp/robot.urdf portable (bundle absolute mesh references into
urdf/tmp/robot_meshes/ and use relative paths). Robot._generate_robot_urdf already does this after
regenerating; this command is for URDFs produced some other way. Safe to re-run.

    python tools/bundle_robot_meshes.py [--urdf pybullet_tree_sim/urdf/tmp/robot.urdf]
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pybullet_tree_sim.utils.robot_meshes import bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=_REPO_ROOT / "pybullet_tree_sim" / "urdf" / "tmp" / "robot.urdf")
    args = parser.parse_args()
    print(f"{bundle(args.urdf)} absolute mesh reference(s) bundled in {args.urdf}")


if __name__ == "__main__":
    main()
