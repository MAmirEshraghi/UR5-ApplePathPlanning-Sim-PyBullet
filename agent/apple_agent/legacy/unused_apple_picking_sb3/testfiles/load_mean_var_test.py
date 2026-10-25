# Previous location: agent/apple_agent/apple_picking_sb3/testfiles/load_mean_var_test.py (originally feature_apple_agent/apple_picking_sb3/testfiles/load_mean_var_test.py). Unused by the pipeline; kept for reference.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pickle

with open("C:/Users/abhin/PycharmProjects/sb3bleeding/logs/test/mean_std.obj", "rb") as f:
    mean_std = pickle.load(f)
print(mean_std)
print(mean_std.mean)
print(mean_std.var)
