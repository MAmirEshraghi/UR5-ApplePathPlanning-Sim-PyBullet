# Unused vendored modules (from apple_agent.apple_picking_sb3)

Moved here on 2026-10-25 because the pipeline never imports them (old OSU pruning-sb3 code: test files,
training scripts, baselines, viz, PPOAE, SACAE, pruning env). Imports inside still use
`apple_agent.apple_picking_sb3.*` paths and are not expected to work from here. Delete together with legacy/.
Files that cannot carry a comment (notebook, profiler output, odd file names) are listed here only.

| File (here) | Previous location |
|---|---|
| `README.md` | `agent/apple_agent/apple_picking_sb3/README.md` |
| `algo/PPOAE/__init__.py` | `agent/apple_agent/apple_picking_sb3/algo/PPOAE/__init__.py` |
| `algo/PPOAE/policies.py` | `agent/apple_agent/apple_picking_sb3/algo/PPOAE/policies.py` |
| `algo/PPOAE/ppo_ae.py` | `agent/apple_agent/apple_picking_sb3/algo/PPOAE/ppo_ae.py` |
| `algo/SACAE/__init__.py` | `agent/apple_agent/apple_picking_sb3/algo/SACAE/__init__.py` |
| `algo/SACAE/policies.py` | `agent/apple_agent/apple_picking_sb3/algo/SACAE/policies.py` |
| `algo/SACAE/sac_ae.py` | `agent/apple_agent/apple_picking_sb3/algo/SACAE/sac_ae.py` |
| `args/__init__.py` | `agent/apple_agent/apple_picking_sb3/args/__init__.py` |
| `args/args.py` | `agent/apple_agent/apple_picking_sb3/args/args.py` |
| `args/args_test.py` | `agent/apple_agent/apple_picking_sb3/args/args_test.py` |
| `baselines/__init__.py` | `agent/apple_agent/apple_picking_sb3/baselines/__init__.py` |
| `baselines/baselines_callbacks.py` | `agent/apple_agent/apple_picking_sb3/baselines/baselines_callbacks.py` |
| `baselines/load_and_smooth_paths.py` | `agent/apple_agent/apple_picking_sb3/baselines/load_and_smooth_paths.py` |
| `baselines/rrt_connect_with_tolerance.py` | `agent/apple_agent/apple_picking_sb3/baselines/rrt_connect_with_tolerance.py` |
| `baselines/rrt_star_with_tolerance.py` | `agent/apple_agent/apple_picking_sb3/baselines/rrt_star_with_tolerance.py` |
| `baselines/run_baseline.py` | `agent/apple_agent/apple_picking_sb3/baselines/run_baseline.py` |
| `img_to_vid.py` | `agent/apple_agent/apple_picking_sb3/img_to_vid.py` |
| `pruning_gym/callbacks/eval_callbacks.py` | `agent/apple_agent/apple_picking_sb3/pruning_gym/callbacks/eval_callbacks.py` |
| `pruning_gym/make_urdf.py` | `agent/apple_agent/apple_picking_sb3/pruning_gym/make_urdf.py` |
| `pruning_gym/pruning_env.py` | `agent/apple_agent/apple_picking_sb3/pruning_gym/pruning_env.py` |
| `pruning_gym/pyb_utils.py` | `agent/apple_agent/apple_picking_sb3/pruning_gym/pyb_utils.py` |
| `pruning_gym/reward_utils.py` | `agent/apple_agent/apple_picking_sb3/pruning_gym/reward_utils.py` |
| `pruning_gym/tree.py` | `agent/apple_agent/apple_picking_sb3/pruning_gym/tree.py` |
| `pruning_gym/ur5_utils.py` | `agent/apple_agent/apple_picking_sb3/pruning_gym/ur5_utils.py` |
| `testfiles/__init__.py` | `agent/apple_agent/apple_picking_sb3/testfiles/__init__.py` |
| `testfiles/compress_checkpoints.py` | `agent/apple_agent/apple_picking_sb3/testfiles/compress_checkpoints.py` |
| `testfiles/create_urdf_from_obj.py` | `agent/apple_agent/apple_picking_sb3/testfiles/create_urdf_from_obj.py` |
| `testfiles/detached_action_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/detached_action_test.py` |
| `testfiles/flow_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/flow_test.py` |
| `testfiles/load_mean_var_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/load_mean_var_test.py` |
| `testfiles/make_temp_expert_traj.py` | `agent/apple_agent/apple_picking_sb3/testfiles/make_temp_expert_traj.py` |
| `testfiles/make_trees_from_lpy.py` | `agent/apple_agent/apple_picking_sb3/testfiles/make_trees_from_lpy.py` |
| `testfiles/optical_flow_colorized.py` | `agent/apple_agent/apple_picking_sb3/testfiles/optical_flow_colorized.py` |
| `testfiles/optical_flow_from_file.py` | `agent/apple_agent/apple_picking_sb3/testfiles/optical_flow_from_file.py` |
| `testfiles/plot_optical_flow.py` | `agent/apple_agent/apple_picking_sb3/testfiles/plot_optical_flow.py` |
| `testfiles/plot_points.py` | `agent/apple_agent/apple_picking_sb3/testfiles/plot_points.py` |
| `testfiles/ply_to_obj.py` | `agent/apple_agent/apple_picking_sb3/testfiles/ply_to_obj.py` |
| `testfiles/raft_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/raft_test.py` |
| `testfiles/remove_erraneous_trajectories.py` | `agent/apple_agent/apple_picking_sb3/testfiles/remove_erraneous_trajectories.py` |
| `testfiles/result_ppo_lstm_expert_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/result_ppo_lstm_expert_test.py` |
| `testfiles/split_obj_by_color.py` | `agent/apple_agent/apple_picking_sb3/testfiles/split_obj_by_color.py` |
| `testfiles/test_collision.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_collision.py` |
| `testfiles/test_condition_number.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_condition_number.py` |
| `testfiles/test_env.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_env.py` |
| `testfiles/test_expert_rollouts.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_expert_rollouts.py` |
| `testfiles/test_expert_trajectories.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_expert_trajectories.py` |
| `testfiles/test_hdf5.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_hdf5.py` |
| `testfiles/test_ik.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_ik.py` |
| `testfiles/test_motion copy.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_motion copy.py` |
| `testfiles/test_motion.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_motion.py` |
| `testfiles/test_motion_continous.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_motion_continous.py` |
| `testfiles/test_optical_flow.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_optical_flow.py` |
| `testfiles/test_perp_reward.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_perp_reward.py` |
| `testfiles/test_point_distribution.py` | `agent/apple_agent/apple_picking_sb3/testfiles/test_point_distribution.py` |
| `testfiles/train_ppo_lstm_expert_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/train_ppo_lstm_expert_test.py` |
| `testfiles/train_ppo_lstm_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/train_ppo_lstm_test.py` |
| `testfiles/train_ppo_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/train_ppo_test.py` |
| `testfiles/train_sac_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/train_sac_test.py` |
| `testfiles/visualizations.ipynb` | `agent/apple_agent/apple_picking_sb3/testfiles/visualizations.ipynb` |
| `testfiles/vonmises_test.py` | `agent/apple_agent/apple_picking_sb3/testfiles/vonmises_test.py` |
| `tests/__init__.py` | `agent/apple_agent/apple_picking_sb3/tests/__init__.py` |
| `tests/test_policies.py` | `agent/apple_agent/apple_picking_sb3/tests/test_policies.py` |
| `tests/test_ppo_recurrent_ae.py` | `agent/apple_agent/apple_picking_sb3/tests/test_ppo_recurrent_ae.py` |
| `tests/test_pruning_env.py` | `agent/apple_agent/apple_picking_sb3/tests/test_pruning_env.py` |
| `tests/test_reward_utils.py` | `agent/apple_agent/apple_picking_sb3/tests/test_reward_utils.py` |
| `tests/test_running_mean_std.py` | `agent/apple_agent/apple_picking_sb3/tests/test_running_mean_std.py` |
| `tests/test_tree.py` | `agent/apple_agent/apple_picking_sb3/tests/test_tree.py` |
| `tests/test_ur5_utils.py` | `agent/apple_agent/apple_picking_sb3/tests/test_ur5_utils.py` |
| `training_files/__init__.py` | `agent/apple_agent/apple_picking_sb3/training_files/__init__.py` |
| `training_files/eval_ppo_lstm_checkpoints.py` | `agent/apple_agent/apple_picking_sb3/training_files/eval_ppo_lstm_checkpoints.py` |
| `training_files/result_ppo_lstm.py` | `agent/apple_agent/apple_picking_sb3/training_files/result_ppo_lstm.py` |
| `training_files/train_ppo.py` | `agent/apple_agent/apple_picking_sb3/training_files/train_ppo.py` |
| `training_files/train_ppo_lstm.prof` | `agent/apple_agent/apple_picking_sb3/training_files/train_ppo_lstm.prof` |
| `training_files/train_ppo_lstm.py` | `agent/apple_agent/apple_picking_sb3/training_files/train_ppo_lstm.py` |
| `training_files/train_ppo_lstm_expert.py` | `agent/apple_agent/apple_picking_sb3/training_files/train_ppo_lstm_expert.py` |
| `training_files/train_ppo_lstm_expert.py0:-21:683:44` | `agent/apple_agent/apple_picking_sb3/training_files/train_ppo_lstm_expert.py0:-21:683:44` |
| `training_files/train_sac.py` | `agent/apple_agent/apple_picking_sb3/training_files/train_sac.py` |
| `viz/real_world_to_viz.py` | `agent/apple_agent/apple_picking_sb3/viz/real_world_to_viz.py` |
| `viz/viz_2d_task.py` | `agent/apple_agent/apple_picking_sb3/viz/viz_2d_task.py` |
| `viz/viz_points_distribution.py` | `agent/apple_agent/apple_picking_sb3/viz/viz_points_distribution.py` |
| `viz/viz_points_on_sphere.py` | `agent/apple_agent/apple_picking_sb3/viz/viz_points_on_sphere.py` |
| `viz/viz_smooth_paths.py` | `agent/apple_agent/apple_picking_sb3/viz/viz_smooth_paths.py` |
| `viz/viz_success.py` | `agent/apple_agent/apple_picking_sb3/viz/viz_success.py` |
| `viz/viz_violin_plot.py` | `agent/apple_agent/apple_picking_sb3/viz/viz_violin_plot.py` |
