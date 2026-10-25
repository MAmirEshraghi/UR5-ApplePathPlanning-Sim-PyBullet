#!/usr/bin/env python3
"""
Sanity checks for agent-data schema v2 (run before training on a newly generated file).

    python feature_apple_path_planning/validate_agent_data.py --run-id tree08_rank1_v2 \
        [--policy-smoke] [--replay 2]

Data checks (always):
  schema   schema v2, required attrs, T+1 frames, dtypes, finite values, DIRECT render mode
  depth    metric depth within the camera's [near, far]
  motion   consecutive frames differ (prev_rgb carries real motion); RAFT flow magnitude stats
  target   observation targets recomputed from raw pose + attrs match the stored ones, and the
           last frame of every successful trajectory is within the success tolerance
Optional:
  --policy-smoke  build the policy for every image_channels config and run one BC backward pass
  --replay N      replay N trajectories' actions in the env and compare env obs with stored frames
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "feature_apple_agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from feature_apple_path_planning import agent_data  # noqa: E402
from feature_apple_path_planning.obs_spec import CHANNEL_ORDER, IMAGE_KEYS  # noqa: E402

REQUIRED_ATTRS = (
    "apple_center", "goal_pos", "goal_orn", "initial_joint_angles", "tree_id", "tree_scale",
    "tree_pos", "tree_orientation", "render_mode", "camera_near_m", "camera_far_m",
    "depth_units_per_m", "control_time", "success",
)
FRAME_DTYPES = {"rgb": np.uint8, "depth": np.uint16, "point_mask": np.uint8}
TABULAR_TARGET_KEYS = ("relative_distance", "pregrasp_in_tool")

failures: list[str] = []


def check(ok: bool, msg: str) -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok:
        failures.append(msg)
    return ok


def targets_in_tool_frame(frames: dict, attrs: dict) -> dict:
    """Recompute apple / pre-grasp positions in the tool frame from achieved pose + attrs."""
    init_pos = np.asarray(attrs["apple_center"], np.float64) - frames["desired_goal"][0].astype(np.float64)
    ee = frames["achieved_goal"].astype(np.float64) + init_pos
    or6 = frames["achieved_or"].astype(np.float64)        # rotation matrix[:, :2] flattened row-major
    c0, c1 = or6[:, 0::2], or6[:, 1::2]
    rot = np.stack([c0, c1, np.cross(c0, c1)], axis=2)    # (T+1, 3, 3), columns = tool axes in world
    out = {}
    for key, attr in (("relative_distance", "apple_center"), ("pregrasp_in_tool", "goal_pos")):
        delta = np.asarray(attrs[attr], np.float64)[None] - ee
        out[key] = np.einsum("tji,tj->ti", rot, delta)    # R^T (p - ee)
    return out


def check_trajectory(key: str, grp: h5py.Group, flow_model) -> None:
    print(f"- {key}", flush=True)
    try:
        attrs = agent_data.trajectory_metadata(grp)
    except ValueError as exc:
        check(False, str(exc))
        return
    missing = [a for a in REQUIRED_ATTRS if a not in attrs]
    check(not missing, f"required attrs present {missing or ''}")
    check(str(attrs.get("render_mode")) == "DIRECT", f"render_mode={attrs.get('render_mode')} (DIRECT matches training)")

    n = len(grp["actions"])
    frames = agent_data.load_frames(grp, agent_data.frame_keys(grp))
    lengths = {k: len(v) for k, v in frames.items()}
    check(all(v == n + 1 for v in lengths.values()), f"T={n}, all frame keys have T+1 entries")
    check(all(len(grp[d]) == n for d in ("rewards", "terminals", "truncateds")), "per-step arrays have T entries")
    check(all(frames[k].dtype == dt for k, dt in FRAME_DTYPES.items() if k in frames),
          "image dtypes " + str({k: str(frames[k].dtype) for k in FRAME_DTYPES if k in frames}))
    floats = [k for k, v in frames.items() if v.dtype.kind == "f"]
    check(all(np.isfinite(frames[k]).all() for k in floats), "all float observations finite")
    check(not grp["truncateds"][:].any(), "no truncated steps")

    # depth
    depth_m = frames["depth"].astype(np.float64) / float(attrs["depth_units_per_m"])
    near, far = float(attrs["camera_near_m"]), float(attrs["camera_far_m"])
    check(depth_m.min() >= near - 1e-3 and depth_m.max() <= far + 1e-3,
          f"depth in [{near}, {far}] m (min {depth_m.min():.3f}, median {np.median(depth_m):.3f}, max {depth_m.max():.3f})")

    # motion: frame t's prev_rgb is frame t-1's rgb, and the arm actually moves the image
    rgb = frames["rgb"].astype(np.int16)
    still = [int(np.abs(rgb[t] - rgb[t - 1]).max() == 0) for t in range(1, n + 1)]
    check(np.array_equal(frames["prev_rgb"][1:], frames["rgb"][:-1]) and np.array_equal(frames["prev_rgb"][0], frames["rgb"][0]),
          "prev_rgb[t] == rgb[t-1] (prev_rgb[0] == rgb[0])")
    check(np.mean(still) < 0.2, f"consecutive frames differ ({100 * (1 - np.mean(still)):.0f}% of steps move the image)")
    if flow_model is not None:
        import torch as th
        idx = np.linspace(1, n, num=min(n, 8), dtype=int)
        flow = flow_model.calculate_optical_flow(th.from_numpy(frames["rgb"][idx]), th.from_numpy(frames["prev_rgb"][idx]))
        mag = np.abs(flow.cpu().numpy())
        print(f"  [info] |flow| (normalised by algo_size) p50={np.percentile(mag, 50):.4f} "
              f"p95={np.percentile(mag, 95):.4f} p99={np.percentile(mag, 99):.4f}", flush=True)
        check(np.percentile(mag, 95) > 1e-3, "flow carries motion (p95 > 1e-3)")

    # targets: stored == recomputed from raw pose; last frame within tolerance when successful
    recomputed = targets_in_tool_frame(frames, attrs)
    for k in TABULAR_TARGET_KEYS:
        err = np.abs(recomputed[k] - frames[k]).max()
        check(err < 1e-3, f"{k} matches pose-based recomputation (max err {err:.2e} m)")
    tol = float(attrs.get("success_pos_tolerance_m", 0.06))
    final = float(np.linalg.norm(frames["pregrasp_in_tool"][-1]))
    if bool(attrs["success"]):
        check(final <= tol + 1e-3, f"success: final |pregrasp_in_tool| = {final:.4f} m <= {tol} m")
        n_coll = int(np.sum(grp["step_info"]["collisions_unacceptable"][:]))
        check(n_coll == 0, f"success: no unacceptable collisions ({n_coll} colliding steps)")
    print(f"  [info] start |pregrasp_in_tool| = {np.linalg.norm(frames['pregrasp_in_tool'][0]):.3f} m, "
          f"apple visible in {100 * np.mean(frames['point_mask'].reshape(n + 1, -1).max(1) > 0):.0f}% of frames, "
          f"terminal steps {int(grp['terminals'][:].sum())}", flush=True)


def policy_smoke(path: Path) -> None:
    """Build the policy for each image_channels config and run one BC forward/backward on real data."""
    import torch as th
    from gymnasium import spaces
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    from apple_picking_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
    from apple_picking_sb3.pruning_gym.models import Encoder
    from feature_apple_path_planning.obs_spec import image_keys_for_channels

    with h5py.File(path, "r") as f:
        grp = f[next(iter(f.keys()))]
        all_keys = agent_data.frame_keys(grp)
        batch = list(agent_data.iter_transitions(grp, all_keys))[:8]
    obs_np = {k: np.stack([b[0][k] for b in batch]) for k in all_keys}
    actions = th.as_tensor(np.stack([b[1] for b in batch]))
    action_space = spaces.Box(-1.0, 1.0, shape=(actions.shape[1],), dtype=np.float32)

    configs = [(), ("depth", "mask"), ("flow", "mask"), CHANNEL_ORDER]
    print("- policy smoke test", flush=True)
    for channels in configs:
        keys = [k for k in all_keys if k not in IMAGE_KEYS] + list(image_keys_for_channels(channels))
        obs_space = spaces.Dict({k: spaces.Box(0, 1, shape=obs_np[k].shape[1:], dtype=obs_np[k].dtype) for k in keys})
        policy = RecurrentActorCriticPolicy(
            obs_space, action_space, lr_schedule=lambda _: 1e-4, lr_schedule_ae=lambda _: 1e-4,
            lr_schedule_logstd=lambda _: 1e-4, net_arch=dict(pi=[128], vf=[128]), activation_fn=th.nn.ReLU,
            features_extractor_class=Encoder, features_extractor_kwargs=dict(size=(240, 424)),
            lstm_hidden_size=128, enable_critic_lstm=True, squash_output=True, log_std_init=-1.5,
            image_channels=channels,
        )
        obs = {k: th.as_tensor(obs_np[k]) for k in keys}
        zeros = th.zeros(1, len(batch), 128)
        states = RNNStates((zeros, zeros), (zeros, zeros))
        starts = th.zeros(len(batch))
        starts[0] = 1.0
        loss = policy.bc_action_mse(obs, states, starts, actions)
        loss.backward()
        enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.features_extractor.parameters())
        check(np.isfinite(loss.item()) and (enc_grad or not channels),
              f"image_channels={channels or '()'}: bc_loss={loss.item():.4f}, encoder grads={'yes' if enc_grad else 'n/a'}")


def replay_parity(path: Path, run_id: str, n_traj: int) -> None:
    """Replay stored actions from the stored start state; env observations must match stored frames."""
    from feature_apple_path_planning.apple_picking_env import ApplePickingEnv
    from feature_apple_path_planning.run_apple_data_generator import CONFIG
    from train_apple_picking_expert import _apply_tree_setup_for_training

    config = {**CONFIG, "simulation_setup": {**CONFIG["simulation_setup"], "renders": False, "image_channels": None}}
    _apply_tree_setup_for_training(config, run_id, path)
    env = ApplePickingEnv(config=config)
    print("- replay parity", flush=True)
    with h5py.File(path, "r") as f:
        for key in list(f.keys())[:n_traj]:
            grp = f[key]
            meta = agent_data.trajectory_metadata(grp)
            frames = agent_data.load_frames(grp, agent_data.frame_keys(grp))
            actions, rewards = grp["actions"][:], grp["rewards"][:]
            env.reconfigure_scene(meta)
            env.reset_env_variables()
            sim = [env._get_obs()]
            sim_rewards, info = [], {}
            for a in actions:
                obs, r, terminated, truncated, info = env.step(a.astype(np.float64))
                sim.append(obs)
                sim_rewards.append(r)
            # PyBullet replays drift slightly (sub-mm) on the first rollout after a scene (re)load because of
            # internal solver state, so tolerances are physics-level; semantic bugs (wrong frame, wrong target,
            # wrong prev image) show up as cm-level / whole-image errors.
            per_frame = {k: [float((np.mean if k in IMAGE_KEYS else np.max)(
                np.abs(sim[t][k].astype(np.float64) - frames[k][t].astype(np.float64)))) for t in range(len(sim))]
                for k in frames}
            print(f"  {key}: worst |env - stored| per key: "
                  + ", ".join(f"{k}={max(v):.3g}" for k, v in sorted(per_frame.items())), flush=True)
            env_prev_ok = np.array_equal(sim[0]["prev_rgb"], sim[0]["rgb"]) and all(
                np.array_equal(sim[t]["prev_rgb"], sim[t - 1]["rgb"]) for t in range(1, len(sim)))
            check(env_prev_ok, f"{key}: env prev_rgb[t] == env rgb[t-1] (same rule as the loader)")
            pose_err = max(max(per_frame[k]) for k in ("achieved_goal", "pregrasp_in_tool", "relative_distance"))
            check(pose_err < 2e-3, f"{key}: proprio/targets reproduce (max err {pose_err * 1000:.2f} mm < 2 mm)")
            # images: fraction of changed pixels env-vs-stored must be well below that of one real step
            for k, tol in (("rgb", 8), ("depth", 10)):
                changed = lambda a, b: float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32)) > tol))
                replay = np.median([changed(sim[t][k], frames[k][t]) for t in range(1, len(sim))])
                motion = np.median([changed(frames[k][t], frames[k][t - 1]) for t in range(1, len(sim))])
                check(replay < 0.5 * motion,
                      f"{key}: {k} reproduces (changed pixels: replay {100 * replay:.1f}% vs one step of motion {100 * motion:.1f}%)")
            reward_err = np.abs(np.asarray(sim_rewards) - rewards).max()
            check(reward_err < 0.05, f"{key}: rewards reproduce (max |diff| {reward_err:.4f} < 0.05)")
            check(bool(info.get("goal_achieved", False)) == bool(grp["terminals"][-1]), f"{key}: termination reproduces")
    env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="output/agent_data/<run-id>.hdf5")
    p.add_argument("--no-flow-stats", action="store_true", help="Skip RAFT flow statistics.")
    p.add_argument("--policy-smoke", action="store_true")
    p.add_argument("--replay", type=int, default=0, metavar="N")
    args = p.parse_args()

    # mirror stdout to logs/validate_agent_data/<timestamp>_<run_id>.log
    import datetime
    log_path = _REPO_ROOT / "logs" / "validate_agent_data" / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{args.run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w", buffering=1, encoding="utf-8")
    terminal = sys.stdout

    class _Tee:
        def write(self, data):
            terminal.write(data)
            log_fp.write(data)

        def flush(self):
            terminal.flush()
            log_fp.flush()

    sys.stdout = _Tee()
    print(f"log: {log_path}")

    path = _REPO_ROOT / "output" / "agent_data" / f"{args.run_id}.hdf5"
    if not path.is_file():
        print(f"not found: {path}")
        return 1
    flow_model = None
    if not args.no_flow_stats:
        from apple_picking_sb3.pruning_gym.optical_flow import OpticalFlow
        flow_model = OpticalFlow(size=(224, 224))
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"{path}: {len(keys)} trajectories", flush=True)
        for key in keys:
            check_trajectory(key, f[key], flow_model)
    if args.policy_smoke:
        policy_smoke(path)
    if args.replay > 0:
        replay_parity(path, args.run_id, args.replay)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}", flush=True)
    for msg in failures:
        print(f"  - {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
