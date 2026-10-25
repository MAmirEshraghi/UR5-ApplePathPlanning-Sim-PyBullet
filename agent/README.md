# Perception-Guided Motion Planning and Deep Reinforcement Learning for Robotic Fruit Harvesting

### 🎥 Demo Video
Click below to watch the demo:

[![Watch the video](https://img.youtube.com/vi/AE1qpv3drOY/0.jpg)](https://youtu.be/AE1qpv3drOY)

## (1) Simulation Environment Development: 
Extended the PyBullet-based tree simulation to support a 6-DoF UR5 manipulator with an end-effector gripper, RGB-D camera sensor, and a detailed tree model with apples and leaves, enabling realistic orchard manipulation scenarios. 

## (2) Perception-Integrated Path Planning: 
Developed a collision-free RRT-Connect planner with a novel camera-based visibility validation system to ensure the end-effector maintains visual contact with targets throughout the trajectory and to ensure natural arm movement for the imitation learning task. The planner incorporates collision checking, multiple inverse kinematics candidate generation with orientation perturbation, and task-space path refinement for smooth trajectories. 
## Quantitative results (minimal) 
Path planning with RRT-connect and having a visibility check. These results for a few samples show that using the visibility check leads to smooth and shorter paths.

<img width="1600" height="1067" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/e6b586bb-6699-4a35-af90-d80e48d677c7" />
<img width="1600" height="533" alt="Code_Generated_Image (3)" src="https://github.com/user-attachments/assets/0a9441a6-18b1-4d79-bae9-27bc6700f19e" />
<img width="1600" height="533" alt="Code_Generated_Image (2)" src="https://github.com/user-attachments/assets/55efbf3d-d26a-4d99-a6df-313dfa7a7e4b" />
<img width="1400" height="800" alt="Code_Generated_Image (1)" src="https://github.com/user-attachments/assets/c52dea6d-ebed-46d0-9e2b-fc212a6277ca" />

## (3) Expert Data Generation Pipeline: 
Implemented a two-stage pipeline (planner and generator) that converts joint-space waypoints to end-effector velocity commands using Jacobian-based transformations, executes closed-loop trajectories with feedback, and stores multi-modal observations (RGB images, point masks, joint states, goal positions) with optical flow in HDF5 format for efficient dataset management. 


## (4) Deep Reinforcement Learning Training: 
Designed and implemented a RecurrentPPOAEWithExpert architecture combining Proximal Policy Optimization (PPO) with Long Short-Term Memory (LSTM) networks for temporal reasoning, integrated with online behavioral cloning (BC) using offline expert demonstrations. The model employs a convolutional encoder for RGB feature extraction, supports optical flow for motion understanding, uses a dual-LSTM architecture (actor and critic), and includes a custom imitation learning callback that dynamically reconfigures scene conditions from expert trajectories during training. 
### RL Agent training results (minimal)
Trained only on 4 trajectories.
<img width="1773" height="925" alt="image" src="https://github.com/user-attachments/assets/a349a3c4-370a-4f58-99cc-1edd840a6f18" />
<img width="1765" height="973" alt="image" src="https://github.com/user-attachments/assets/9fbf6f2b-6ba7-4dc6-8105-bfbab07a4b8e" />



### Demo (YouTube) [7/10/2025]:
[![Watch the video](https://img.youtube.com/vi/L9cdALAOvzs/maxresdefault.jpg)](https://www.youtube.com/watch?v=L9cdALAOvzs)





## Script Description: 

### 1: ```apple_agent/planning/run_apple_data_generator.py``` 
This is the main executable script for the data generation pipeline. It uses command-line arguments to run different stages of the process, including:

- Planner (Stage 1): Loads the simulation, finds apples, and uses RRT-Connect to plan collision-free paths to them, saving the paths to ```rrt_paths.hdf5```.

- Generator (Stage 2): moved to ```apple_agent/data/generate_expert_data.py``` (see Stage 2 below); `--stage generator` now exits with that pointer.

- Visualization: Loads and replays the saved RRT paths in the PyBullet GUI for visual inspection.

#### 2: ```apple_agent/envs/apple_picking_env.py```: 
This script defines the core simulation environment, ApplePickingEnv, using the gymnasium (formerly Gym) API. It manages the PyBullet physics, loads the robot and tree models, provides sensor data (like camera images and joint states), and calculates rewards. It's the "world" in which the robot operates and is essential for both data generation and (later) RL agent training.


#### 3: ```apple_agent/envs/utils_conversions.py```: 

This is a utility script containing mathematical helper functions for coordinate frame transformations. Its primary functions, convert_local_action_to_global and convert_global_action_to_local, are used to translate velocity commands between the robot's local end-effector frame (e.g., "move 0.1 m/s along the gripper's 'forward' direction") and the simulation's global world frame.

#### 4: ```pybullet_tree_sim/tree.py``` (simulator dependency): 
This script defines the ```Tree``` class used in the simulation. It is responsible for loading the tree's URDF (for physics) and its high-resolution ```.obj``` mesh (for geometry). Its most important function is to process a labeled mesh file where vertex colors are used to identify tree parts. It contains the logic to parse these colors and generate a list of 3D coordinates for each individual apple's "centroid" (```get_apple_centroids```), which are then used as goals for the path planner.

#### 5: ```pybullet_tree_sim/robot.py``` (simulator dependency): 
This script defines the core ```Robot``` class. It is responsible for loading the robot arm model (from ```.xacro``` and config files) into the simulation. It provides the essential API for robot control, such as setting joint velocities (```set_joint_velocities```), calculating inverse kinematics (```calculate_ik```), and getting sensor data (like camera images from ``get_rgbd_at_cur_pose``). It also contains critical modified functions for the project, including a sophisticated ```check_collisions``` method that uses a K-D tree to identify the label of a collided point (e.g., "TRUNK" vs. "APPLE").

## How to Use ```run_apple_data_generator.py```:

### Stage 1: Run the Path Planner 
This stage loads the environment and uses the RRT-Connect algorithm to find collision-free joint-space paths from the robot's start position to a pre-grasp pose near each apple. These paths are saved to ```rrt_paths.hdf5```

```
python -m apple_agent.planning.run_apple_data_generator --stage planner
```
### Stage 2: Run the Data Generator 
Loads the planner waypoints (`output/waypoints/<run_id>.hdf5`), refines them and executes them with a closed-loop Jacobian controller in `ApplePickingEnv` (DIRECT mode, same renderer as training). Each trajectory is saved in agent-data schema v2 (`apple_agent/data/agent_data.py`): every frame once (rgb, depth in mm, apple mask, proprio, `pregrasp_in_tool`), plus actions, rewards and env terminal/truncated flags. Then validate the file before training:

```
python -m apple_agent.data.generate_expert_data --run-id tree08 --path-rank 1 --out-id tree08_rank1_v2 --save-only-success
python -m apple_agent.data.validate_agent_data --run-id tree08_rank1_v2 --policy-smoke --replay 2
```

Training picks the policy's image inputs from the same data: `python -m apple_agent.rl.train_apple_picking_expert --run_id tree08_rank1_v2 --image_channels depth,mask` (any of `flow`, `depth`, `mask`, or `none` for proprio only).

## Utility & Debugging Commands

### 1: Visualize Saved Paths 
If you want to visually inspect the paths generated by the planner stage without generating new agent data, use this command. It loads rrt_paths.hdf5 and plays back the robot's movement in the GUI.
```
python -m apple_agent.planning.run_apple_data_generator --visualize_planning
```
### 2: Run Coordinate Frame Axis Test 
This is a debugging tool to check that the robot's coordinate frames are set up correctly. It loads the robot and commands a simple movement along its local X-axis (the red line). This helps verify that "move forward" in the tool's frame translates to the correct motion in the simulation.
```
python -m apple_agent.planning.run_apple_data_generator --stage axis_test
```

## Setup and running

This package (`apple_agent`) depends on the simulator package `pybullet_tree_sim`. Today both live in one
repository: the simulator at the repo root, this package under `agent/`. It is prepared to move into its own
repository later, with the simulator installed as a pinned dependency (see `agent/pyproject.toml`).

```bash
# from the repository root (the workspace), venv activated
export PYTHONPATH=$PWD/agent            # or: pip install -e agent   (and pip install -e . for the simulator)
python -m apple_agent.planning.run_apple_data_generator --stage planner      # RRT waypoints -> output/waypoints/
python -m apple_agent.data.generate_expert_data --run-id tree08 --out-id tree08_v2
python -m apple_agent.data.validate_agent_data --run-id tree08_v2 --policy-smoke --replay 2
python -m apple_agent.rl.train_apple_picking_expert --run_id tree08_v2 --run_name <name> --image_channels depth,mask
python -m apple_agent.rl.eval_apple_picking_expert --run_id tree08_v2 --checkpoint logs/<name>/model_20000_steps.zip --eval_all --max_steps 150
```

Commands run from the **workspace root**: `output/` (waypoints, agent data), `logs/` and `runs/` are
relative to the current directory.

If the simulator is installed read-only, point `PYBULLET_TREE_SIM_DATA` at a folder with the simulator's
data layout (`meshes/`, `textures/`, `pkl/`, `urdf/`).

### Package layout
- `apple_agent/config.py`: shared `CONFIG` (simulation, robot, tree, planning, generator, output)
- `apple_agent/envs/`: `ApplePickingEnv` and frame conversions
- `apple_agent/planning/`: RRT-Connect planner (stage 1), goal camera filter, debug visualisation
- `apple_agent/data/`: agent-data schema v2 I/O, observation spec, expert generator, validator, merge tool
- `apple_agent/rl/`: train / eval / replay entry points and debug diagnostics
- `apple_agent/apple_picking_sb3/`: vendored SB3 algorithms (PPO+BC with LSTM) and helpers
- `apple_agent/legacy/`: old playground and archive scripts, kept for reference only
