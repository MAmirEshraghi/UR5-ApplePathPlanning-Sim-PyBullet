import argparse
import datetime
import os
import sys
import traceback
from typing import Optional, Tuple
import numpy as np
import torch as th
import h5py
import random
from pathlib import Path
import psutil

# --- Adapt paths: repo root (pybullet_tree_sim, feature_apple_path_planning) then this dir (apple_picking_sb3) ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_AGENT_DIR = os.path.abspath(os.path.dirname(__file__))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

# --- Imports from Apple Picking Project ---
from feature_apple_path_planning.apple_picking_env import ApplePickingEnv
from feature_apple_path_planning.run_apple_data_generator import (
    CONFIG as APPLE_PICKING_CONFIG,
    apply_tree_setup_from_waypoints_hdf5,
    tree_id_str_from_config,
)

# --- Imports from Pruning SB3 Project ---
from apple_picking_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
#from pruning_sb3.ppo_recurrent_ae import RecurrentPPOAEWithExpert
from apple_picking_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAEWithExpert
#from pruning_sb3.policies import RecurrentActorCriticPolicy
#from pruning_sb3.models import Encoder
from apple_picking_sb3.pruning_gym.models import Encoder

#from train_callbacks
from apple_picking_sb3.pruning_gym.callbacks.train_callbacks import PruningCheckpointCallback

from stable_baselines3.common.callbacks import BaseCallback

# --- Other necessary imports ---
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common import utils
from stable_baselines3.common.vec_env import DummyVecEnv


def _first_hdf5_group_metadata(hdf5_path: Path) -> dict:
    """Read attrs from the first group in an agent-data or waypoints HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No groups in {hdf5_path}")
        return {k: np.asarray(v) for k, v in f[keys[0]].attrs.items()}


def _patch_tree_setup_from_metadata(config: dict, metadata: dict) -> None:
    """Align tree_setup with recorded trajectory metadata (matches generator / replay)."""
    ts = config["tree_setup"]
    if metadata.get("tree_id") is not None:
        ts["tree_id"] = int(np.asarray(metadata["tree_id"]).reshape(-1)[0])
    if metadata.get("tree_scale") is not None:
        ts["scale"] = float(np.asarray(metadata["tree_scale"]).reshape(-1)[0])
    if metadata.get("tree_pos") is not None:
        ts["position"] = np.asarray(metadata["tree_pos"], dtype=np.float64).reshape(3)
    if metadata.get("tree_orientation") is not None:
        ts["orientation"] = np.asarray(metadata["tree_orientation"], dtype=np.float64).reshape(4)


def _apply_tree_setup_for_training(config: dict, run_id: str, agent_data_path: Path) -> None:
    """
    Patch config tree_setup from the waypoints run (preferred) or agent-data metadata.

    Matches generate_agent_data_from_waypoints_2.py so ApplePickingEnv loads the same tree
    as the expert trajectories (tree_id, scale, pose).
    """
    waypoints_path = Path("output") / "waypoints" / f"{run_id}.hdf5"
    if waypoints_path.is_file():
        try:
            ref_key = apply_tree_setup_from_waypoints_hdf5(
                waypoints_path, config=config
            )
            print(
                f"INFO: tree_setup from waypoints {waypoints_path.name} "
                f"(ref group {ref_key!r}, id={tree_id_str_from_config(config)})",
                flush=True,
            )
            return
        except (ValueError, KeyError, OSError) as exc:
            print(
                f"WARNING: waypoints tree_setup failed ({exc}); "
                "falling back to agent-data metadata.",
                flush=True,
            )

    metadata = _first_hdf5_group_metadata(agent_data_path)
    if metadata.get("tree_id") is None or metadata.get("tree_scale") is None:
        raise ValueError(
            f"Agent data {agent_data_path} missing tree_id/tree_scale attrs; "
            "cannot align training env with expert data."
        )
    _patch_tree_setup_from_metadata(config, metadata)
    print(
        f"INFO: tree_setup from agent-data metadata: "
        f"tree_id={config['tree_setup']['tree_id']} "
        f"scale={config['tree_setup']['scale']:.4f} "
        f"pos={np.round(np.asarray(config['tree_setup']['position']), 4).tolist()} "
        f"id_str={tree_id_str_from_config(config)}",
        flush=True,
    )


class _TeeIO:
    """Write to terminal + log file; tqdm/Rich progress redraws stay terminal-only."""

    def __init__(self, terminal_stream, log_stream, filter_progress: bool = True):
        self._terminal = terminal_stream
        self._log = log_stream
        self._filter_progress = filter_progress

    @staticmethod
    def _is_progress_spam(data: str) -> bool:
        if not data or not data.strip():
            return True
        # In-place bar updates (no newline) — main source of multi-MB log lines
        if "\r" in data and "\n" not in data:
            return True
        # Rich/tqdm ANSI progress lines
        if "\x1b[2K" in data or "\x1b[?25" in data:
            return True
        if "%" in data and "it/s" in data and ("\x1b[" in data or "━━" in data):
            return True
        return False

    def write(self, data):
        self._terminal.write(data)
        self._terminal.flush()
        if self._filter_progress and self._is_progress_spam(data):
            return
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def fileno(self):
        return self._terminal.fileno()

    def isatty(self):
        return self._terminal.isatty()


# =================================================================================
# 0. MEMORY MONITORING
# =================================================================================
def log_memory(phase: str, num_timesteps: Optional[int] = None) -> Tuple[Optional[float], Optional[float]]:
    """Log RSS (and CUDA alloc when available) with a phase tag for debugging leaks."""
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        mem_gb = mem_mb / 1024
        ts_part = f" t={num_timesteps}" if num_timesteps is not None else ""
        cuda_part = ""
        if th.cuda.is_available():
            cuda_gb = th.cuda.memory_allocated() / (1024 ** 3)
            cuda_part = f" | cuda_alloc={cuda_gb:.2f} GB"
        print(
            f"[train:mem]{ts_part} phase={phase} RSS={mem_mb:.1f} MB ({mem_gb:.2f} GB){cuda_part}",
            flush=True,
        )
        return mem_mb, mem_gb
    except Exception as e:
        print(f"Warning: Could not log memory ({phase}): {e}", flush=True)
        return None, None


def _summarize_expert_hdf5(hdf5_path: Path) -> dict:
    """Read expert HDF5 once for [train:data] startup lines."""
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        step_counts: list[int] = []
        for key in keys:
            grp = f[key]
            if "actions" in grp:
                step_counts.append(int(grp["actions"].shape[0]))
            elif "rewards" in grp:
                step_counts.append(int(grp["rewards"].shape[0]))
        ref = f[keys[0]] if keys else None
        tree_id_str = str(ref.attrs.get("tree_id_str", "")) if ref is not None else ""
        tree_scale = float(ref.attrs.get("tree_scale", 0.0)) if ref is not None else 0.0
    steps = np.asarray(step_counts, dtype=np.int64) if step_counts else np.array([], dtype=np.int64)
    return {
        "n_trajectories": len(keys),
        "step_min": int(steps.min()) if steps.size else 0,
        "step_max": int(steps.max()) if steps.size else 0,
        "step_mean": float(steps.mean()) if steps.size else 0.0,
        "tree_id_str": tree_id_str,
        "tree_scale": tree_scale,
        "size_mb": hdf5_path.stat().st_size / (1024 * 1024),
    }


def _log_train_data(hdf5_path: Path, training_config: dict) -> None:
    summary = _summarize_expert_hdf5(hdf5_path)
    tree = training_config.get("tree_setup", {})
    print(
        f"[train:data] expert_hdf5={hdf5_path.resolve()} size_mb={summary['size_mb']:.2f} "
        f"trajectories={summary['n_trajectories']} "
        f"steps_min={summary['step_min']} steps_max={summary['step_max']} "
        f"steps_mean={summary['step_mean']:.1f}",
        flush=True,
    )
    print(
        f"[train:data] tree_id_str={summary['tree_id_str'] or tree.get('tree_id', '')} "
        f"scale={summary['tree_scale'] or tree.get('scale', 0):.4f} "
        f"pos={np.round(np.asarray(tree.get('position', [0, 0, 0])), 4).tolist()}",
        flush=True,
    )


def _log_train_config(
    *,
    args,
    log_path: Path,
    run_name: str,
    total_timesteps: int,
    n_envs: int,
    training_config: dict,
    policy_kwargs: dict,
    train_kwargs: dict,
) -> None:
    sim = training_config.get("simulation_setup", {})
    print(
        f"[train:config] run_id={args.run_id} log_file={log_path} "
        f"total_timesteps={total_timesteps} progress_bar={args.progress_bar} "
        f"memory_debug={args.memory_debug}",
        flush=True,
    )
    print(
        f"[train:config] bc_coeff={args.bc_coeff} ent_coef={train_kwargs['ent_coef']} "
        f"log_std_init={policy_kwargs['log_std_init']} "
        f"terminate_on_self_collision={sim.get('terminate_on_self_collision', True)} "
        f"n_envs={n_envs} device={train_kwargs['device']}",
        flush=True,
    )
    print(
        f"[train:config] learning_rate={train_kwargs['learning_rate']} "
        f"n_steps={train_kwargs['n_steps']} batch_size={train_kwargs['batch_size']} "
        f"n_epochs={train_kwargs['n_epochs']} gamma={train_kwargs['gamma']} "
        f"gae_lambda={train_kwargs['gae_lambda']} clip_range={train_kwargs['clip_range']}",
        flush=True,
    )
    print(
        f"[train:config] use_online_bc={train_kwargs['use_online_bc']} "
        f"use_offline_data={train_kwargs['use_offline_data']} "
        f"lstm_hidden_size={policy_kwargs['lstm_hidden_size']} "
        f"checkpoint_dir=logs/{run_name} tensorboard=./runs/{run_name}",
        flush=True,
    )
    print(
        "[train:config] log_groups: [train:init] [train:data] [train:episode] "
        "[train:rollout] [train:summary] [train:mem]",
        flush=True,
    )


class MemoryDebugCallback(BaseCallback):
    """Periodic RSS/CUDA snapshots during rollouts to spot per-step vs per-episode growth."""

    def __init__(self, log_interval: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = max(1, int(log_interval))

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.log_interval == 0:
            log_memory("rollout_step", num_timesteps=self.num_timesteps)
        return True


# =================================================================================
# 1. DEFINE THE NEW IMITATION LEARNING CALLBACK
# =================================================================================
def _empty_rollout_stats() -> dict:
    return {
        "success": [],
        "collision": [],
        "timeout": [],
        "ep_len": [],
        "ep_rew": [],
        "final_ee_dist": [],
        "start_ee_dist": [],
        "goal_ep_len": [],
    }


class ApplePickingImitationCallback(BaseCallback):
    """
    Callback for imitation learning. On each episode reset, it loads a random
    expert trajectory's starting conditions into the environment.

    Logs use grouped prefixes for easy grep/monitoring:
      [train:init]    — startup / scene setup
      [train:episode] — one line per finished episode
      [train:rollout] — aggregated stats each PPO rollout
      [train:summary] — cumulative stats at end of training
    """
    def __init__(self, hdf5_path: str, verbose=0, memory_debug: bool = False):
        super(ApplePickingImitationCallback, self).__init__(verbose)
        self.hdf5_path = hdf5_path
        self.trajectory_keys = []
        self.metadata_cache = {}
        self.memory_debug = memory_debug
        self._active_traj_keys: list[Optional[str]] = []
        self._ep_return: list[float] = []
        self._rollout_stats = _empty_rollout_stats()
        self._total_stats = _empty_rollout_stats()

    def _on_rollout_start(self) -> None:
        self._rollout_stats = _empty_rollout_stats()
        n_envs = self.training_env.num_envs if self.training_env is not None else 1
        self._ep_return = [0.0] * n_envs
        if len(self._active_traj_keys) != n_envs:
            self._active_traj_keys = [None] * n_envs

    def _log_episode_end(self, env_idx: int, info: dict) -> None:
        goal = bool(info.get("goal_achieved", False))
        collision = bool(info.get("collision_terminated", False))
        timeout = not goal and not collision
        if goal:
            reason = "GOAL"
        elif collision:
            reason = "COLLISION"
        else:
            reason = "TIMEOUT"

        ep_len = int(info.get("episode_length", 0))
        ep_rew = float(self._ep_return[env_idx])
        final_dist = float(info.get("dist_to_goal", float("nan")))
        start_dist = float(info.get("start_dist_to_goal", float("nan")))
        delta_dist = start_dist - final_dist if np.isfinite(start_dist) and np.isfinite(final_dist) else float("nan")
        traj_key = self._active_traj_keys[env_idx] if env_idx < len(self._active_traj_keys) else None

        if self.verbose > 0:
            print(
                f"[train:episode] t={self.num_timesteps} env={env_idx} reason={reason} "
                f"len={ep_len} rew={ep_rew:.2f} ee_dist={final_dist:.4f} "
                f"start_dist={start_dist:.4f} delta_dist={delta_dist:+.4f} "
                f"traj={traj_key}",
                flush=True,
            )

        self._rollout_stats["success"].append(float(goal))
        self._rollout_stats["collision"].append(float(collision))
        self._rollout_stats["timeout"].append(float(timeout))
        self._rollout_stats["ep_len"].append(ep_len)
        self._rollout_stats["ep_rew"].append(ep_rew)
        self._rollout_stats["final_ee_dist"].append(final_dist)
        self._rollout_stats["start_ee_dist"].append(start_dist)
        if goal:
            self._rollout_stats["goal_ep_len"].append(ep_len)

        self._total_stats["success"].append(float(goal))
        self._total_stats["collision"].append(float(collision))
        self._total_stats["timeout"].append(float(timeout))
        self._total_stats["ep_len"].append(ep_len)
        self._total_stats["ep_rew"].append(ep_rew)
        self._total_stats["final_ee_dist"].append(final_dist)
        self._total_stats["start_ee_dist"].append(start_dist)
        if goal:
            self._total_stats["goal_ep_len"].append(ep_len)

        self._ep_return[env_idx] = 0.0

    def _log_rollout_summary(self) -> None:
        stats = self._rollout_stats
        n = len(stats["success"])
        if n == 0:
            return

        success_rate = float(np.mean(stats["success"]))
        collision_rate = float(np.mean(stats["collision"]))
        timeout_rate = float(np.mean(stats["timeout"]))
        mean_final_dist = float(np.mean(stats["final_ee_dist"]))
        mean_start_dist = float(np.mean(stats["start_ee_dist"]))
        mean_ep_rew = float(np.mean(stats["ep_rew"]))
        mean_ep_len = float(np.mean(stats["ep_len"]))
        goal_ep_len_mean = (
            float(np.mean(stats["goal_ep_len"])) if stats["goal_ep_len"] else float("nan")
        )

        if self.verbose > 0:
            print(
                f"[train:rollout] t={self.num_timesteps} episodes={n} "
                f"success_rate={success_rate:.3f} collision_rate={collision_rate:.3f} "
                f"timeout_rate={timeout_rate:.3f} mean_final_ee_dist={mean_final_dist:.4f} "
                f"mean_start_ee_dist={mean_start_dist:.4f} mean_ep_len={mean_ep_len:.1f} "
                f"goal_ep_len_mean={goal_ep_len_mean:.1f} mean_ep_rew={mean_ep_rew:.2f}",
                flush=True,
            )

        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/collision_rate", collision_rate)
        self.logger.record("rollout/timeout_rate", timeout_rate)
        self.logger.record("rollout/mean_final_ee_dist", mean_final_dist)
        self.logger.record("rollout/mean_start_ee_dist", mean_start_dist)
        self.logger.record("rollout/mean_ep_len_task", mean_ep_len)
        self.logger.record("rollout/mean_ep_rew_task", mean_ep_rew)
        if stats["goal_ep_len"]:
            self.logger.record("rollout/goal_ep_len_mean", goal_ep_len_mean)

    def _log_training_summary(self) -> None:
        stats = self._total_stats
        n = len(stats["success"])
        if n == 0:
            print("[train:summary] no episodes completed", flush=True)
            return

        success_rate = float(np.mean(stats["success"]))
        collision_rate = float(np.mean(stats["collision"]))
        timeout_rate = float(np.mean(stats["timeout"]))
        mean_final_dist = float(np.mean(stats["final_ee_dist"]))
        goal_ep_len_mean = (
            float(np.mean(stats["goal_ep_len"])) if stats["goal_ep_len"] else float("nan")
        )
        n_goals = int(np.sum(stats["success"]))

        print(
            f"[train:summary] total_episodes={n} goals_reached={n_goals} "
            f"success_rate={success_rate:.3f} collision_rate={collision_rate:.3f} "
            f"timeout_rate={timeout_rate:.3f} mean_final_ee_dist={mean_final_dist:.4f} "
            f"goal_ep_len_mean={goal_ep_len_mean:.1f}",
            flush=True,
        )

    def _on_training_start(self) -> None:
        """Load all trajectory keys and their metadata from the HDF5 file."""
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                self.trajectory_keys = list(f.keys())
                for key in self.trajectory_keys:
                    grp = f[key]
                    meta = {attr: grp.attrs[attr] for attr in grp.attrs.keys()}
                    if (
                        "initial_joint_angles" not in meta
                        and "observations" in grp
                        and "joint_angles" in grp["observations"]
                    ):
                        enc = np.asarray(
                            grp["observations"]["joint_angles"][0],
                            dtype=np.float64,
                        ).reshape(-1)
                        half = len(enc) // 2
                        meta["initial_joint_angles"] = np.arctan2(
                            enc[:half], enc[half:]
                        ).astype(np.float64)
                    self.metadata_cache[key] = meta
            if self.verbose > 0:
                print(
                    f"[train:init] callback ready expert_trajectories={len(self.trajectory_keys)}",
                    flush=True,
                )
            # Phase 0: first rollout must use expert goal/pose (VecEnv reset() alone leaves reward_goal at origin).
            if self.trajectory_keys and self.training_env is not None:
                n_envs = self.training_env.num_envs
                self._active_traj_keys = [None] * n_envs
                self._ep_return = [0.0] * n_envs
                start_key = self._reset_to_expert_trajectory(env_idx=0)
                if start_key is not None:
                    self._active_traj_keys[0] = start_key
                if self.verbose > 0 and start_key is not None:
                    print(
                        f"[train:init] initial_scene traj={start_key} t=0",
                        flush=True,
                    )
        except Exception as e:
            print(f"[train:init] error loading HDF5 {self.hdf5_path}: {e}")

    def _reset_to_expert_trajectory(
        self, trajectory_key: Optional[str] = None, env_idx: int = 0
    ) -> Optional[str]:
        """Reconfigure env to a reachable expert start and sync model obs after VecEnv auto-reset."""
        if not self.trajectory_keys or self.training_env is None:
            return None
        key = trajectory_key or random.choice(self.trajectory_keys)
        scene_metadata = self.metadata_cache[key]
        self.training_env.env_method("reconfigure_scene", metadata=scene_metadata)
        self.training_env.env_method("reset_env_variables")
        self.training_env.env_method("snapshot_episode_start_dist")
        if self.model is not None:
            obs_list = self.training_env.env_method("_get_obs")
            self.model._last_obs = {
                k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]
            }
            self.model._last_episode_starts = np.ones(
                (self.training_env.num_envs,), dtype=bool
            )
        return key

    def _on_step(self) -> bool:
        """
        Check for finished episodes and reset the corresponding environments
        to a new expert starting state.
        """
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i, done in enumerate(dones):
            self._ep_return[i] += float(rewards[i])
            if not done:
                continue

            if not self.trajectory_keys:
                print(
                    "[train:init] warning: no expert trajectory keys for reset",
                    flush=True,
                )
                continue

            self._log_episode_end(i, infos[i])

            if self.memory_debug:
                log_memory("before_reconfigure", num_timesteps=self.num_timesteps)

            random_key = self._reset_to_expert_trajectory(env_idx=i)
            if random_key is not None and i < len(self._active_traj_keys):
                self._active_traj_keys[i] = random_key

            if self.memory_debug:
                log_memory("after_reconfigure", num_timesteps=self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        self._log_rollout_summary()

    def _on_training_end(self) -> None:
        self._log_training_summary()

# =================================================================================
# 2. MAIN TRAINING SCRIPT
# =================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train apple-picking expert policy with PPO + imitation.")
    parser.add_argument("--run_id", required=True, help="Run ID corresponding to output/agent_data/{run_id}.hdf5")
    parser.add_argument(
        "--log_file",
        default=None,
        help="Append full stdout/stderr session to this file. Default: logs/train_apple_picking_expert/<ts>_<run_id>.log",
    )
    parser.add_argument(
        "--memory_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log RSS/CUDA at milestones, every N steps, and around scene reconfigure (default: on).",
    )
    parser.add_argument(
        "--memory_log_interval",
        type=int,
        default=50,
        help="When --memory_debug, log every N env timesteps during rollouts (default: 50).",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=4086,
        help="Training length in env steps. Default 2048 (~1h smoke); was hardcoded 10_000 (~4h).",
    )
    parser.add_argument(
        "--bc_coeff",
        type=float,
        default=1.5,
        help="BC loss multiplier. Default 1.5; was 0.6.",
    )
    parser.add_argument(
        "--progress_bar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Live tqdm bar in terminal only (filtered from log). Default: off.",
    )
    args = parser.parse_args()

    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    _log_fp = None
    if args.log_file:
        log_path = Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = (
            Path("logs") / "train_apple_picking_expert" / f"{_ts}_{args.run_id}.log"
        ).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    _log_fp = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _TeeIO(_orig_stdout, _log_fp)
    sys.stderr = _TeeIO(_orig_stderr, _log_fp)

    try:
        print(
            f"Session log (Python stdout/stderr): {log_path}\n"
            "(Note: some native libraries write directly to the terminal FD and may not appear in this file.)",
            flush=True,
        )

        agent_data_dir = Path("output") / "agent_data"
        agent_data_dir.mkdir(parents=True, exist_ok=True)
        expert_data_path = agent_data_dir / f"{args.run_id}.hdf5"
        if not expert_data_path.exists():
            raise FileNotFoundError(f"Agent data file not found: {expert_data_path}")

        # --- Basic Configuration ---
        run_name = "apple_picking_imit_run_final"
        n_envs = 1
        total_timesteps = args.total_timesteps
        expert_data_filepath = str(expert_data_path.resolve())

        training_config = {
            **APPLE_PICKING_CONFIG,
            "simulation_setup": {
                **APPLE_PICKING_CONFIG["simulation_setup"],
                "renders": False,  # DIRECT mode — much lower RAM, no GUI leak
                "terminate_on_self_collision": True,  # was False — align with expert data gen default
            },
        }
        _apply_tree_setup_for_training(training_config, args.run_id, expert_data_path)
        _log_train_data(expert_data_path, training_config)

        # --- Environment Setup ---
        env_kwargs = {'config': training_config}
        print("INFO: Building vectorized env (PyBullet load can take 1–3+ minutes)…", flush=True)
        env = make_vec_env(ApplePickingEnv, n_envs=n_envs, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)
        print("INFO: Environment ready.", flush=True)
        if args.memory_debug:
            log_memory("after_env")

        #env = VecTransposeImage(env)

        new_logger = utils.configure_logger(verbose=1, tensorboard_log=f"./runs/{run_name}", reset_num_timesteps=True)

        # --- Callbacks ---
        imitation_callback = ApplePickingImitationCallback(
            hdf5_path=expert_data_filepath,
            verbose=1,
            memory_debug=args.memory_debug,
        )
        checkpoint_callback = PruningCheckpointCallback(save_freq=10000,  # More frequent checkpoints
                                                        save_path=f"./logs/{run_name}",
                                                        name_prefix="model", verbose=1)
        callback_list = [imitation_callback, checkpoint_callback]
        if args.memory_debug:
            callback_list.insert(
                0,
                MemoryDebugCallback(log_interval=args.memory_log_interval, verbose=1),
            )

        # --- Policy and Model Setup ---
        policy_kwargs = {
            'features_extractor_class': Encoder,
            'features_extractor_kwargs': dict(in_channels=3, size=(240, 424)), # Match env camera
            'net_arch': dict(pi=[128], vf=[128]),  # Reduce from 64 to 32 (50% less memory)
            'activation_fn': th.nn.ReLU,
            'lstm_hidden_size': 128,  # Reduce from 256 to 128 (50% less memory)
            'enable_critic_lstm': True,
            'use_optical_flow': True, # Apple picking env doesn't use optical flow
            'squash_output': True,  # Force actions to be in [-1, 1] range using tanh
            'log_std_init': -1.5,  # was 0.0 (std≈1.0) — closer to expert Jacobian action scale
        }

        # The RecurrentPPOAEWithExpert model from the pruning project is now fully compatible
        print("INFO: Building PPO+BC model (loads expert dataset)…", flush=True)
        train_kwargs = dict(
            learning_rate=1e-4,
            n_steps=512,
            batch_size=16,
            n_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.25,
            max_grad_norm=0.5,
            use_online_data=False,
            use_offline_data=False,
            use_ppo_offline=False,
            use_online_bc=True,
            use_awac=False,
            device="cuda",
        )
        model = RecurrentPPOAEWithExpert(
            policy=RecurrentActorCriticPolicy,
            env=env,
            path_trajectories=os.path.abspath(expert_data_filepath),
            bc_coeff=args.bc_coeff,
            algo_size=(224, 224),
            use_cached_optical_flow=True,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"./runs/{run_name}",
            **train_kwargs,
        )

        model.set_logger(new_logger)
        if args.memory_debug:
            log_memory("after_model")

        print("INFO: Model ready. Starting learning loop…", flush=True)
        _log_train_config(
            args=args,
            log_path=log_path,
            run_name=run_name,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            training_config=training_config,
            policy_kwargs=policy_kwargs,
            train_kwargs=train_kwargs,
        )
        if args.memory_debug:
            print(
                f"INFO: Memory debug on (interval={args.memory_log_interval} steps, "
                "before/after reconfigure on episode end).",
                flush=True,
            )
            print("=" * 60)
            log_memory("before_learn", num_timesteps=0)
            print("=" * 60)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=args.progress_bar,
        )

        if args.memory_debug:
            print("=" * 60)
            log_memory("after_learn", num_timesteps=model.num_timesteps)
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

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        if _log_fp is not None:
            _log_fp.close()
        print(f"Session log saved to {log_path}", file=_orig_stdout, flush=True)
