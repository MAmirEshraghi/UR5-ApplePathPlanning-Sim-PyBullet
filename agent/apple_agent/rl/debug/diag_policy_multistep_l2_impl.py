"""Implementation for multistep policy L2 diagnostic (import after CUDA bootstrap)."""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from debug._common import (
    action_l2,
    agent_data_path,
    load_policy,
    run_with_session_log,
    setup_env,
)
from eval_apple_picking_expert import (
    _classify_episode,
    _load_expert_metadata,
    _reconfigure_to_trajectory,
    _vec_obs_from_env,
)


def _episode_keys(
    hdf5_path: Path,
    *,
    eval_all: bool,
    n_episodes: int,
    seed: int,
) -> Tuple[List[str], Dict[str, dict]]:
    trajectory_keys, metadata_cache = _load_expert_metadata(str(hdf5_path.resolve()))
    if eval_all:
        return list(trajectory_keys), metadata_cache
    rng = random.Random(seed)
    n = min(n_episodes, len(trajectory_keys))
    return rng.sample(trajectory_keys, n), metadata_cache


def _l2_at_step(steps: List[dict], step: int) -> Optional[float]:
    for row in steps:
        if row["step"] == step:
            return float(row["l2"])
    return None


def _first_spike_step(steps: List[dict], threshold: float) -> Optional[int]:
    for row in steps:
        if row["l2"] >= threshold:
            return int(row["step"])
    return None


def run_multistep_episode(
    model,
    env,
    hdf5_path: Path,
    traj_key: str,
    metadata: dict,
    *,
    max_steps: int,
    max_log_steps: int,
    l2_threshold: float,
    deterministic: bool,
    log_every: int,
) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        expert_actions = np.asarray(f[traj_key]["actions"][:], dtype=np.float32)
    expert_len = len(expert_actions)
    n_steps = min(max_steps, expert_len, max_log_steps)

    _reconfigure_to_trajectory(env, metadata)
    obs = _vec_obs_from_env(env)
    lstm_states = None
    episode_start = np.array([True])
    step_rows: List[dict] = []
    last_info: dict = {}

    for step_i in range(n_steps):
        expert_action = expert_actions[step_i]
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        episode_start = np.array([False])
        policy_action = np.asarray(action[0], dtype=np.float32)
        mse, l2 = action_l2(policy_action, expert_action)

        obs, _reward, done, info = env.step(action)
        last_info = dict(info[0])
        ee_dist = float(last_info.get("dist_to_goal", float("nan")))

        row = {
            "step": step_i,
            "l2": l2,
            "mse": mse,
            "ee_dist": ee_dist,
            "policy_action": policy_action.tolist(),
            "expert_action": expert_action.tolist(),
        }
        step_rows.append(row)

        if step_i == 0 or step_i % log_every == 0 or l2 >= l2_threshold or bool(done[0]):
            print(
                f"[diag:multistep] traj={traj_key} step={step_i}/{n_steps} "
                f"l2={l2:.4f} ee_dist={ee_dist:.4f} done={bool(done[0])}",
                flush=True,
            )

        if bool(done[0]):
            break

    reason = _classify_episode(last_info)
    final_ee = float(last_info.get("dist_to_goal", float("nan")))
    ep_len = int(last_info.get("episode_length", len(step_rows)))
    l2_values = [r["l2"] for r in step_rows]
    ee_values = [r["ee_dist"] for r in step_rows if np.isfinite(r["ee_dist"])]

    summary = {
        "traj": traj_key,
        "reason": reason,
        "expert_len": expert_len,
        "steps_logged": len(step_rows),
        "episode_length": ep_len,
        "final_ee_dist": final_ee,
        "l2_step0": _l2_at_step(step_rows, 0),
        "l2_step5": _l2_at_step(step_rows, 5),
        "l2_step10": _l2_at_step(step_rows, 10),
        "l2_step20": _l2_at_step(step_rows, 20),
        "l2_mean": float(np.mean(l2_values)) if l2_values else None,
        "l2_max": float(np.max(l2_values)) if l2_values else None,
        "first_l2_spike_step": _first_spike_step(step_rows, l2_threshold),
        "l2_threshold": l2_threshold,
        "ee_dist_min": float(np.min(ee_values)) if ee_values else None,
        "ee_dist_at_last_step": float(step_rows[-1]["ee_dist"]) if step_rows else None,
    }
    l2s0 = summary["l2_step0"]
    l2mean = summary["l2_mean"]
    l2max = summary["l2_max"]
    print(
        f"[diag:multistep:traj] traj={traj_key} reason={reason} ep_len={ep_len} "
        f"final_ee={final_ee:.4f} l2_step0={l2s0:.4f} "
        f"l2_mean={l2mean:.4f} l2_max={l2max:.4f} "
        f"first_spike@={summary['first_l2_spike_step']}"
        if l2s0 is not None and l2mean is not None and l2max is not None
        else f"[diag:multistep:traj] traj={traj_key} reason={reason} ep_len={ep_len} "
        f"final_ee={final_ee:.4f} (no steps logged)",
        flush=True,
    )
    return {"summary": summary, "steps": step_rows}


def _aggregate(episodes: List[dict], l2_threshold: float) -> dict:
    if not episodes:
        return {"n_episodes": 0}

    def _mean_key(key: str) -> Optional[float]:
        vals = [e["summary"][key] for e in episodes if e["summary"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    goals = sum(1 for e in episodes if e["summary"]["reason"] == "GOAL")
    spikes_before_10 = sum(
        1
        for e in episodes
        if e["summary"]["first_l2_spike_step"] is not None and e["summary"]["first_l2_spike_step"] < 10
    )

    return {
        "n_episodes": len(episodes),
        "success_rate": goals / len(episodes),
        "n_goals": goals,
        "mean_l2_step0": _mean_key("l2_step0"),
        "mean_l2_step5": _mean_key("l2_step5"),
        "mean_l2_step10": _mean_key("l2_step10"),
        "mean_l2_step20": _mean_key("l2_step20"),
        "mean_l2_mean": _mean_key("l2_mean"),
        "mean_l2_max": _mean_key("l2_max"),
        "frac_spike_before_step10": spikes_before_10 / len(episodes),
        "l2_threshold": l2_threshold,
    }


def _interpret(agg: dict) -> str:
    if agg.get("n_episodes", 0) == 0:
        return "NO DATA"
    s5 = agg.get("mean_l2_step5")
    s0 = agg.get("mean_l2_step0")
    if s5 is not None and s0 is not None and s5 > s0 + 0.15:
        return "CLOSED_LOOP_DIVERGENCE — L2 grows quickly after t=0 (shift likely main blocker)"
    if agg.get("frac_spike_before_step10", 0) > 0.5:
        return "EARLY_DIVERGENCE — most trajs exceed L2 threshold before step 10"
    if agg.get("success_rate", 0) >= 0.15 and agg.get("mean_l2_step0", 1.0) < 0.35:
        return "MIXED — t=0 marginal; failures may be hard-traj or horizon, check per-traj"
    return "INCONCLUSIVE — inspect per-traj logs and compare with eval GOAL trajs"


def run_diagnostic(args, resolved_device: str) -> None:
    def _run() -> dict:
        print(f"[debug:device] policy/RAFT on device={resolved_device}", flush=True)

        hdf5_path = agent_data_path(args.run_id)
        episode_keys, metadata_cache = _episode_keys(
            hdf5_path,
            eval_all=args.eval_all,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
        env, cfg, hdf5_path = setup_env(args.run_id, checkpoint=args.checkpoint)
        max_steps = int(cfg["simulation_setup"].get("max_steps", 500))
        ckpt = Path(args.checkpoint).resolve()
        print(
            f"[diag:multistep:config] checkpoint={ckpt} n={len(episode_keys)} "
            f"max_steps={max_steps} max_log_steps={args.max_log_steps} "
            f"l2_threshold={args.l2_threshold} deterministic={args.deterministic} "
            f"device={resolved_device}",
            flush=True,
        )

        model = load_policy(
            str(ckpt),
            env,
            hdf5_path,
            device=resolved_device,
        )
        model.policy.set_training_mode(False)

        episodes: List[dict] = []
        try:
            for traj_key in episode_keys:
                episodes.append(
                    run_multistep_episode(
                        model,
                        env,
                        hdf5_path,
                        traj_key,
                        metadata_cache[traj_key],
                        max_steps=max_steps,
                        max_log_steps=args.max_log_steps,
                        l2_threshold=args.l2_threshold,
                        deterministic=args.deterministic,
                        log_every=args.log_every,
                    )
                )
        finally:
            env.close()

        agg = _aggregate(episodes, args.l2_threshold)
        verdict = _interpret(agg)
        agg["verdict"] = verdict
        print(f"[diag:multistep:summary] {agg}", flush=True)
        return {
            "script": "diag_policy_multistep_l2",
            "run_id": args.run_id,
            "checkpoint": str(ckpt),
            "seed": args.seed,
            "config": {
                "max_steps": max_steps,
                "max_log_steps": args.max_log_steps,
                "l2_threshold": args.l2_threshold,
                "deterministic": args.deterministic,
                "device": resolved_device,
            },
            "summary": agg,
            "episodes": episodes,
        }

    run_with_session_log("diag_policy_multistep_l2", args.run_id, _run)
