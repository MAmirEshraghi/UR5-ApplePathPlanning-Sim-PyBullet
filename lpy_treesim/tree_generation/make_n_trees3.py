import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import random as rd
from openalea.lpy import Lsystem
import argparse
from openalea.plantgl.all import Scene # Import Scene to handle the object

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trees', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='dataset/')
    # Use your corrected L-system file
    parser.add_argument('--lpy_file', type=str, default='examples/Camp_Envy_tie_prune_label.text')
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
        
        # This loop runs the full derivation.
        for time in range(l.derivationLength):
            lstring = l.derive(lstring, time, 1)
            # The plot call is necessary to build the internal scene state step-by-step
            l.plot(lstring)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if args.verbose:
            print("INFO: Writing tree number: ", i)

        # ============================================================================== #
        # === THE FIX: Get the final scene after the plotting loop is complete       === #
        # ============================================================================== #
        # The sceneInterpretation() method gets the final geometry from the derivation.
        # This call should now work correctly because the plot() calls have built the scene.
        final_scene = l.sceneInterpretation(lstring)
        
        # The scene.save() method will create both the .obj and the necessary .mtl file.
        output_filename = "{}/tree_{}.obj".format(output_dir, i)
        final_scene.save(output_filename)
        print(f"Successfully saved complete tree to {output_filename}")
        
        del final_scene
        del lstring
        del l

