"""
Shared configuration for the apple-picking task: simulation, robot, tree, planning, generator, output.

One mutable dict shared by planner, expert generator, env construction, training and evaluation.
Paths in CONFIG['output'] are relative to the workspace (the directory commands are run from).
"""
import numpy as np

CONFIG = {
    'simulation_setup': {
        'renders': True,
        'gravity': -9.81,
        'control_time': 1.0 / 20.0,
        'max_steps': 350,
        # GUI: frame tree like run_sim_tree_viewer.py (PyBUtils defaults target the robot)
        'frame_tree_camera': True,
        'debug_camera_distance': 0.8,
        'debug_camera_yaw': 55,
        'debug_camera_pitch': -22,
        'debug_camera_target_z_offset': 1.0,
        # Env red spheres at apple centroids; planner also draws goal markers separately
        'draw_apple_centroid_markers': False,
        # stepSimulation after creating debug bodies so the GUI updates
        'gui_refresh_steps': 1,
        # Backdrop boxes are visual-only unless True (planner ignores walls anyway)
        'room_collision': False,
    },
    'robot_setup': {
        'start_position': [0, 1.2, 0],
        'start_orientation_euler_deg': [0, 0, 180],
        'start_joint_angles': [-1.978, -1.51, 2.622, -1.896, 0.579, 0.848],
        # None = use pybullet_tree_sim/urdf/tmp/robot.urdf if present (no ROS ament_index_python)
        'regenerate_urdf': None,
    },
    'tree_setup': {
        'tree_id': 0,
        'tree_type': "envy",
        'tree_namespace': "LPy",
        'scale': 0.7,
        'position': np.array([0.3, 0, 0]),
        'orientation': np.array([0, 0, 0, 1]),
        # v2 export: multi-material OBJ/MTL + parts URDF (see run_sim_tree_viewer.py)
        'apply_texture': True,
        'preserve_export_mtl': None,  # None = auto when *_textured.mtl exists
        'regenerate_urdf': False,
        # False: show full-tree visual on base link (foliage); True: parts-only (thin leaves vanish)
        'hide_collision_visual': True,
    },
    'planning': {
        'max_ee_velocity': 0.25,
        'rrt_max_iterations': 3000,
        'rrt_max_time': 90.0,
        'planner_type': 'rrt_star', #'rrt_star', #'rrt_connect',
        'rrt_radius': 0.4,
        'num_goal_candidates_to_try': 200,
        'goal_position_offset': 0.2,
        # Single source for (1) IK FK acceptance, (2) ApplePickingEnv success radius (distance_threshold),
        # (3) HDF5 attr success_pos_tolerance_m. Keep one value unless you intentionally split IK vs success.
        'fk_pos_tolerance': 0.06,  # relaxed from 0.05 — physics sim drifts ~6 mm short in rollout
        'fk_orn_tolerance': 0.08,  # paired with success_orn_tolerance_rad in HDF5 metadata
        'max_orientation_roll_perturbation_rad': np.pi/4,
        'max_orientation_pitch_yaw_perturbation_rad': np.pi/6,
        'collision_labels': ["TRUNK", "BRANCH", "APPLE", "LEAF", "SPUR", "STEM", "WATER_BRANCH"],
        'acceptable_collision_labels': [],
        'unacceptable_collision_labels': ["TRUNK", "BRANCH", "APPLE", "SPUR", "LEAF", "STEM", "WATER_BRANCH", "UNKNOWN_KDTREE_UNAVAILABLE", "UNKNOWN_KDTREE_QUERY_ERROR", "UNKNOWN_TOO_FAR", "UNKNOWN_NO_CLOSE_VERTEX"],
        'orientation_seed': 44,
        'visibility_seed': 44,
        'enable_smoothing': True,
        'task_space_refinement_threshold': 0.012,
        # Pick best RRT candidate: joint_length + w_smooth*smoothness + w_task*task_regression (see below).
        'path_selection_smoothness_weight': 0.1,
        # Weight on sum of positive Δ||EE−goal|| along raw path (meters); 0 disables (legacy behavior).
        'path_selection_task_regression_weight': 0.15,
        # Run shortcut + task-space refine before writing waypoints to HDF5 (matches generator).
        'save_refined_waypoints_to_hdf5': True,
        # Save up to N best RRT paths per apple as separate HDF5 groups (1 = legacy single-best).
        'num_top_paths_to_save_per_apple': 2,
        'visibility_check': {
            'enable': False,
            'debug_view': True,
            'num_targets': 0,
            'min_visible_pixels': 5,
            'target_color_rgba': [1.0, 0.0, 0.0, 0.8],#[0.1, 1.0, 0.1, 0.9],
            'target_color_rgb': (250, 0, 0), #(25, 255, 25),
            # Add support for multiple target colors (red for goals, blue for active apples)
            'target_colors_rgb': [(250, 0, 0), (0, 0, 240)],  # Red and Blue
            'color_tolerance': (40, 40, 40), #(80,80,80),
            'target_sphere_radius': 0.04 #0.025
        },
        # Pre-RRT only: at q_goal, require enough "apple blue" inside projected disk (Method A).
        # Independent of visibility_check in is_state_valid_fn.
        'goal_camera_filter': {
            'enable': True,
            # When enable=True: RRT runs only on up to this many filter-passed IK goals.
            # Raise planning.num_goal_candidates_to_try (IK pool, e.g. 200) separately.
            'num_camera_filtered_goal_candidates_to_try': 5,
            'min_visible_frac': 0.85,
            # RGB image order matches robot.get_rgbd (RGB not BGR).
            'color_rgb_lower': (0, 0, 140),
            'color_rgb_upper': (100, 100, 255),
            'sphere_radius_m': 0.052, 
            'disk_radius_scale': 1.0,
            'sim_settle_steps': 1,
            # Extra GOAL_CAM_FILTER fields + log.debug lines (NDC, eye, apple, clip_w).
            'log_projection_debug': True,
            # When True, save s01/s02/s03 PNGs under logs/goal_camera_filter/<run_id>/.
            'save_debug_images': False,
            'debug_subdir': 'goal_camera_filter',
        },
    },
    'generator': {
        'action_scale': 1,
    },
    'visualization': {
        'draw_goal_spheres': True,
        'goal_sphere_radius': 0.042,
        # Push marker spheres outside apple mesh toward robot (meters)
        'apple_marker_surface_offset_m': 0.0,
        'centroid_marker_color': [0.9, 0.08, 0.08, 0.9],
        'goal_sphere_color': [1.0, 0.0, 0.0, 0.8],
        'active_apple_color': [0.0, 0.0, 1.0, 0.9],
        'active_apple_sphere_radius': 0.053,
        'pregrasp_marker_color': [0.0, 1.0, 1.0, 0.9],
        'pregrasp_sphere_radius': 0.04,
        'draw_goal_frames': True,
        'goal_frame_length': 0.09,
        'path_visualization_delay': 0.1,
        'delay_between_apple_attempts': 0.05,
        'draw_collision_item_aabbs': True,
        'collision_item_aabb_color': [0.0, 0.0, 1.0, 0.5],
        'visualize_actual_collision_points': True,
        'collision_point_color_env': [1.0, 0.5, 0.0, 0.9],
        'collision_point_color_self': [1.0, 0.0, 1.0, 0.9],
        'collision_point_radius': 0.01
    },
    'output': {
        'base_dir': "output",
        'waypoints_subdir': "waypoints",
        'agent_data_subdir': "agent_data",
        'waypoints_hdf5_path': None,
        'agent_data_hdf5_path': None
    }
}
