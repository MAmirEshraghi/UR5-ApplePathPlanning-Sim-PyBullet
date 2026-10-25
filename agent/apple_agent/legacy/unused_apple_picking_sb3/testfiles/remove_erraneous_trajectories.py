# Previous location: agent/apple_agent/apple_picking_sb3/testfiles/remove_erraneous_trajectories.py (originally feature_apple_agent/apple_picking_sb3/testfiles/remove_erraneous_trajectories.py). Unused by the pipeline; kept for reference.
import os
import sys
import glob
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


if __name__ == "__main__":
    expert_trajectory_path = "expert_trajectories"
    print("Number of expert trajectories: ", len(glob.glob(expert_trajectory_path + "/*.pkl")))
    expert_trajectories = glob.glob(expert_trajectory_path + "/*.pkl")
    for expert_trajectory in expert_trajectories:
        with open(expert_trajectory, "rb") as f:
            try:
                expert_data = pickle.load(f)
            except:
                print("Error loading expert trajectory: ", expert_trajectory)
                os.remove(expert_trajectory)
                continue