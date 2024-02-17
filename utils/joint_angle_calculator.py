import numpy as np
from constant import JOINT_CONSTANTS


class DotDict:
    """
    DotDict wraps a dictionary and provides attribute-style access to its items.
    """

    def __init__(self, data):
        """
        Initialize DotDict with the provided dictionary.

        Args:
            data (dict): Dictionary containing key-value pairs.
        """
        self.data = data

    def __getattr__(self, name):
        """
        Get the value of a dictionary item using dot notation.

        Args:
            name (str): Name of the item to access.

        Returns:
            any: Value corresponding to the provided name.

        Raises:
            AttributeError: If the provided name does not exist in the dictionary.
        """
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


def calculate_angle(joint1, joint2, joint3):
    """
    Calculate the angle between three joints.

    Args:
        joint1 (DotDict): First joint.
        joint2 (DotDict): Second joint (vertex of the angle).
        joint3 (DotDict): Third joint.

    Returns:
        float: Angle between the joints in degrees, rounded to 3 decimal places.
    """
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

    # Convert angle to degrees and round to 3 decimal places
    angle_degrees = np.degrees(angle_radians)
    return round(angle_degrees, 3)


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

        # Calculate angle between the joints
        angle_degrees = calculate_angle(joint1, joint2, joint3)

        # Store the calculated angle in the dictionary
        joint_angles[joint_name] = angle_degrees

    # Calculate angle between the legs
    right_hip = landmark_positions[24]
    left_hip = landmark_positions[23]

    # Calculate mid-point between the hips
    mid_point_hip = {
        "x": (right_hip.x + left_hip.x) / 2,
        "y": (right_hip.y + left_hip.y) / 2,
    }

    # Convert the dictionary into a DotDict for easy access and convert similar like landmark objects
    mid_point_of_hip = DotDict(mid_point_hip)

    # Calculate angles for leg spred
    joint_angles["leg_spread"] = calculate_angle(
        landmark_positions[26], mid_point_of_hip, landmark_positions[25]
    )

    return joint_angles
