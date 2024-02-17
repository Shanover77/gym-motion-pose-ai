import numpy as np
from constant import JOINT_CONSTANTS


def calculate_joint_angles(landmark_positions):
    """
    Calculate the angles between joints based on landmark positions.

    Args:
        landmark_positions (dict): Dictionary containing landmark positions for all joints.

    Returns:
        dict: Dictionary containing joint names as keys and corresponding angles in degrees as values.
    """
    joint_angles = {}

    # Iterate over each joint constant
    for joint_name, joint_indices in JOINT_CONSTANTS.items():
        # Unpack the joint indices
        idx1, idx2, idx3 = joint_indices

        # Get landmark positions for the joints
        joint1 = landmark_positions[idx1]
        joint2 = landmark_positions[idx2]
        joint3 = landmark_positions[idx3]

        # Calculate vectors between the joints
        vector1 = np.array([joint1.x - joint2.x, joint1.y - joint2.y])
        vector2 = np.array([joint3.x - joint2.x, joint3.y - joint2.y])

        # Calculate dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Calculate cosine of the angle
        cosine_angle = dot_product / (magnitude1 * magnitude2)

        # Calculate angle in radians
        angle_radians = np.arccos(cosine_angle)

        # Convert angle to degrees
        angle_degrees = np.degrees(angle_radians)

        # Store the calculated angle in the dictionary
        joint_angles[joint_name.lower()] = round(angle_degrees, 2)

    return joint_angles
