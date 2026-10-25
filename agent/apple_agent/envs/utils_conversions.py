# utils_conversions.py
import numpy as np

def skew_symmetric(vector):
    """
    Compute the skew-symmetric matrix of a vector.
    :param vector: np.array, shape (3,)
    :return: np.array, shape (3, 3)
    """
    x, y, z = vector
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def make_adjoint(rotation_matrix, translation_vector):
    """
    Create the adjoint transformation matrix.
    :param rotation_matrix: np.array, shape (3, 3), rotation matrix R
    :param translation_vector: np.array, shape (3,), translation vector p
    :return: np.array, shape (6, 6), adjoint transformation matrix
    """
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
    translation_vector = np.array(translation_vector).reshape(3)
    p_skew = skew_symmetric(translation_vector)
    adjoint = np.block([
        [rotation_matrix, np.zeros((3, 3))],
        [np.dot(p_skew, rotation_matrix), rotation_matrix]
    ])
    return adjoint

def convert_local_action_to_global(robot, local_action):
    """
    Converts a local end-effector velocity action to a global velocity.
    :param robot: The Robot instance.
    :param local_action: np.array, shape (6,), action in the tool's local frame.
    :return: np.array, shape (6,), action in the global world frame.
    """
    pos, orient = robot.get_current_pose(robot.tool0_link_idx)
    rot_matrix = np.array(robot.pbclient.getMatrixFromQuaternion(orient)).reshape(3, 3)
    adjoint = make_adjoint(rot_matrix, pos)
    global_action = np.dot(adjoint, local_action)
    return global_action

def convert_global_action_to_local(robot, action):
    """
    Convert a global action vector to the local frame of the robot.
    :param robot: robot instance with `get_current_pose` and `pbclient.getMatrixFromQuaternion`
    :param action: np.array, shape (6,), action in global frame
    :return: np.array, shape (6,), action in local frame
    """
    pos, orient = robot.get_current_pose(robot.tool0_link_idx)
    rot_matrix = np.array(robot.pbclient.getMatrixFromQuaternion(orient)).reshape(3, 3)
    adjoint = make_adjoint(rot_matrix, pos)
    local_action = np.dot(np.linalg.inv(adjoint), action)
    return local_action
