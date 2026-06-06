import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import random as rd
from openalea.lpy import Lsystem
from lpy_treesim.tree_generation.helpers import write
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trees', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='dataset/')
    parser.add_argument('--lpy_file', type=str, default='examples/UFO_tie_prune_label.lpy')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    num_trees = args.num_trees
    output_dir = args.output_dir
    lpy_file = args.lpy_file

    for i in range(num_trees):
        if args.verbose:
            print("INFO: Generating tree number: ", i)
        rand_seed = rd.randint(0,1000)
        variables = {'label': False, 'seed_val': rand_seed}
        l = Lsystem(lpy_file, variables)
        lstring = l.axiom
        for time in range(l.derivationLength):
            lstring = l.derive(lstring, time, 1)
            l.plot(lstring)
        # l.plot()
        scene = l.sceneInterpretation(lstring)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if args.verbose:
            print("INFO: Writing tree number: ", i)
        # scene.save("{}/tree_{}.obj".format(output_dir, i))
        write("{}/tree_{}.ply".format(output_dir, i), scene)
        del scene
        del lstring
        del l
