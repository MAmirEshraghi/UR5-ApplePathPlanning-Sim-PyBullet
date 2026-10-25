import argparse
import os
import sys
import numpy as np
import torch as th
import h5py
import random
from pathlib import Path
import psutil

# --- Adapt paths to your project structure ---
# This assumes the script is run from a location where both projects are accessible.
# e.g., sys.path.insert(0, '/path/to/your/projects/')
# For simplicity, ensure the required files are in the same directory or in the Python path.

# Add paths to apple picking project and the pruning_sb3 project
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/robben/codes/TreeSim/ok/pybullet-tree-sim')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/robben/codes/TreeSim/pruning_sb3')))

# --- Imports from Apple Picking Project ---
from feature_apple_path_planning.apple_picking_env import ApplePickingEnv
from feature_apple_path_planning.run_apple_data_generator import CONFIG as APPLE_PICKING_CONFIG

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/.../.../apple_picking_sb3')))

# --- Imports from Pruning SB3 Project ---
from apple_picking_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
#from pruning_sb3.ppo_recurrent_ae import RecurrentPPOAEWithExpert
from apple_picking_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAEWithExpert
#from pruning_sb3.policies import RecurrentActorCriticPolicy
#from pruning_sb3.models import Encoder
from apple_picking_sb3.pruning_gym.models import Encoder

#from train_callbacks
from pruning_sb3.pruning_gym.callbacks.train_callbacks import PruningCheckpointCallback # This one is reusable

from stable_baselines3.common.callbacks import BaseCallback

# --- Other necessary imports ---
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common import utils
from stable_baselines3.common.vec_env import SubprocVecEnv

# =================================================================================
# 0. MEMORY MONITORING FUNCTION
# =================================================================================
def log_memory():
    """Log current memory usage of the process."""
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        mem_gb = mem_mb / 1024
        print(f"Memory usage: {mem_mb:.1f} MB ({mem_gb:.2f} GB)")
        return mem_mb, mem_gb
    except Exception as e:
        print(f"Warning: Could not log memory usage: {e}")
        return None, None

# =================================================================================
# 1. DEFINE THE NEW IMITATION LEARNING CALLBACK
# =================================================================================
class ApplePickingImitationCallback(BaseCallback):
    """
    Callback for imitation learning. On each episode reset, it loads a random
    expert trajectory's starting conditions into the environment.
    """
    def __init__(self, hdf5_path: str, verbose=0):
        super(ApplePickingImitationCallback, self).__init__(verbose)
        self.hdf5_path = hdf5_path
        self.trajectory_keys = []
        self.metadata_cache = {}

    def _on_training_start(self) -> None:
        """Load all trajectory keys and their metadata from the HDF5 file."""
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                self.trajectory_keys = list(f.keys())
                for key in self.trajectory_keys:
                    self.metadata_cache[key] = {attr: f[key].attrs[attr] for attr in f[key].attrs.keys()}
            if self.verbose > 0:
                print(f"Callback initialized. Found {len(self.trajectory_keys)} expert trajectories.")
        except Exception as e:
            print(f"Error initializing callback: Could not read HDF5 file at {self.hdf5_path}. Error: {e}")

    def _on_step(self) -> bool:
        """
        Check for finished episodes and reset the corresponding environments
        to a new expert starting state.
        """
        for i, done in enumerate(self.locals['dones']):
            if done:
                # If an episode is done, select a new random expert trajectory
                if not self.trajectory_keys:
                    print("Callback Warning: No expert trajectory keys found to reset environment.")
                    continue
                
                random_key = random.choice(self.trajectory_keys)
                scene_metadata = self.metadata_cache[random_key]
                
                if self.verbose > 1:
                    print(f"Env {i} finished. Resetting to state from trajectory: {random_key}")
                    print("Memory usage before scene reconfiguration:")
                    log_memory()
                
                # Use env_method to call the new reconfigure_scene method on the specific environment
                self.training_env.env_method("reconfigure_scene", indices=i, metadata=scene_metadata)
                
                if self.verbose > 1:
                    print("Memory usage after scene reconfiguration:")
                    log_memory()
        return True

# =================================================================================
# 2. MAIN TRAINING SCRIPT
# =================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train apple-picking expert policy with PPO + imitation.")
    parser.add_argument("--run_id", required=True, help="Run ID corresponding to output/agent_data/{run_id}.hdf5")
    args = parser.parse_args()

    agent_data_dir = Path("output") / "agent_data"
    agent_data_dir.mkdir(parents=True, exist_ok=True)
    expert_data_path = agent_data_dir / f"{args.run_id}.hdf5"
    if not expert_data_path.exists():
        raise FileNotFoundError(f"Agent data file not found: {expert_data_path}")

    # --- Basic Configuration ---
    run_name = "apple_picking_imit_run_final"
    n_envs = 1
    # FIX 2: Increased training duration for meaningful learning
    # For production: use 2_000_000 or more
    # For testing: use 50_000 - 100_000
    total_timesteps = 50_000  # Changed from 100 to 50,000 for better learning
    
    # Path to your generated expert data file
    expert_data_filepath = str(expert_data_path.resolve())

    # --- Environment Setup ---
    env_kwargs = {'config': APPLE_PICKING_CONFIG}
    env = make_vec_env(ApplePickingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    
    #env = VecTransposeImage(env)

    new_logger = utils.configure_logger(verbose=1, tensorboard_log=f"./runs/{run_name}", reset_num_timesteps=True)

    # --- Callbacks ---
    # Use the new, more powerful callback
    imitation_callback = ApplePickingImitationCallback(hdf5_path=expert_data_filepath, verbose=1)
    # FIX 2: Adjusted checkpoint frequency to match new training duration
    checkpoint_callback = PruningCheckpointCallback(save_freq=10000,  # More frequent checkpoints
                                                    save_path=f"./logs/{run_name}",
                                                    name_prefix="model", verbose=1)
    callback_list = [imitation_callback, checkpoint_callback]

    # --- Policy and Model Setup ---
    policy_kwargs = {
        'features_extractor_class': Encoder,
        'features_extractor_kwargs': dict(in_channels=3, size=(240, 424)), # Match env camera
        'net_arch': dict(pi=[128], vf=[128]),  # Reduce from 64 to 32 (50% less memory)
        'activation_fn': th.nn.ReLU,
        'lstm_hidden_size': 128,  # Reduce from 256 to 128 (50% less memory)
        'enable_critic_lstm': True,
        'use_optical_flow': True, # Apple picking env doesn't use optical flow
        'squash_output': True  # Force actions to be in [-1, 1] range using tanh
    }

    # The RecurrentPPOAEWithExpert model from the pruning project is now fully compatible
    model = RecurrentPPOAEWithExpert(
        policy=RecurrentActorCriticPolicy,
        env=env,
        path_trajectories=os.path.dirname(os.path.abspath(expert_data_filepath)), # Pass directory
        use_online_data=False,
        use_offline_data=False,
        use_ppo_offline=False,
        use_online_bc=True, # Use online PPO with offline Behavioral Cloning
        use_awac=False,
        # FIX 3: Adjusted BC coefficient for better balance
        bc_coeff=0.3,  # Reduced from 0.5 to 0.3 to allow more exploration
        algo_size=(224, 224), # Standard size for the algo's internal processing
        use_cached_optical_flow=True,
        # FIX 3: Reduced learning rate for more stable training
        learning_rate=1e-4,  # Reduced from 3e-4 to 1e-4 for stability
        n_steps=32,#1024, # Reduce from 64 to 32 (50% less memory)
        batch_size=16,  # Reduce from 32 to 16 (50% less memory)
        # FIX 3: Increased epochs for better sample efficiency
        n_epochs=4,  # Increased from 1 to 4 for better learning
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        # FIX 4: Reduced value function coefficient to help with value learning
        vf_coef=0.25,  # Reduced from 0.5 to 0.25 to reduce value loss impact
        max_grad_norm=0.5,  # Already set, verified working
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"./runs/{run_name}"
    )

    model.set_logger(new_logger)

    print("INFO: Starting training for apple picking...")
    print("=" * 60)
    print("Initial memory usage:")
    log_memory()
    print("=" * 60)
    
    model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
    
    print("=" * 60)
    print("Final memory usage after training:")
    log_memory()
    print("=" * 60)

    # --- Save final model ---
    save_dir = Path("./logs") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Store original dataloader and data_iter references
    original_dataloader = getattr(model, "dataloader", None)
    original_data_iter = getattr(model, "data_iter", None)
    
    # Clear unpicklable objects before saving
    if original_dataloader is not None:
        model.dataloader = None
    if original_data_iter is not None:
        model.data_iter = None

    try:
        model.save(str(save_dir / "final_model.zip"))
        print("INFO: Training complete. Final model saved.")
    except NotImplementedError as exc:
        print(f"WARNING: Failed to save model due to unpicklable object: {exc}")
    finally:
        # Restore original references after saving
        if original_dataloader is not None:
            model.dataloader = original_dataloader
        if original_data_iter is not None:
            model.data_iter = original_data_iter
