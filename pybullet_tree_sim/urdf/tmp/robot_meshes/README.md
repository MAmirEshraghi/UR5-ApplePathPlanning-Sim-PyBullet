# Robot meshes referenced by ../robot.urdf

`robot.urdf` references these files by relative path, so it loads on any machine without ROS or
other workspaces. PyBullet resolves the paths from the URDF's folder.

| Folder | Source | License |
|---|---|---|
| `ur5e/` | ROS 2 package `ur_description` (Universal Robots, github.com/UniversalRobots/Universal_Robots_ROS2_Description), `meshes/ur5e/` | BSD-3-Clause (`ur5e/LICENSE`) |
| `amiga/` | `branch_detection_system_description` (github.com/lukestroh/branch_detection_ws), `meshes/base_mounts/farm-ng/amiga/` | BSD-3-Clause (`amiga/LICENSE`) |
| `end_effectors/` | this simulator's mesh set (`pybullet_tree_sim/meshes/end_effectors/`) | this repository's license |

`robot.urdf` itself is generated from `urdf/robot/generic/robot.urdf.xacro`, which needs ROS 2 (xacro with
ament). After regenerating it, run the path fix again so it references these bundled meshes.
