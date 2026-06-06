import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
import glob
from helpers import convert_ply_to_obj



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='dataset/')
    parser.add_argument('--input_dir', type=str, default='examples/UFO_tie_prune_label.lpy')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    print("1")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("2")

    for file in glob.glob(os.path.join(args.input_dir, '*.ply')):
        output_path = os.path.join(args.output_dir, os.path.basename(file).replace('.ply', '.obj'))
        print("3")
        if args.verbose:
            print("INFO: Converting file: ", file, "and saving to: ", output_path)
        convert_ply_to_obj(file, output_path)



