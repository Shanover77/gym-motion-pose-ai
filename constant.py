# Joint Constants
# This dictionary defines constants representing groups of joints.
# Each key represents a group of joints, and the corresponding value is a tuple containing the joint indices.

# RE_SH_H: Tuple representing the indices of landmarks for the right elbow, shoulder, and hip joints.
# LE_SH_H: Tuple representing the indices of landmarks for the left elbow, shoulder, and hip joints.
# RW_E_SH: Tuple representing the indices of landmarks for the right wrist, elbow, and shoulder joints.
# LW_E_SH: Tuple representing the indices of landmarks for the left wrist, elbow, and shoulder joints.
# RH_K_A: Tuple representing the indices of landmarks for the right hip, knee, and ankle joints.
# LH_K_A: Tuple representing the indices of landmarks for the left hip, knee, and ankle joints.

JOINT_CONSTANTS = {
    "RIGHT_ELBOW_SHOULDER_HIP": (
        14,
        12,
        24,
    ),  # Indices of right elbow, shoulder, and hip joints
    "LEFT_ELBOW_SHOULDER_HIP": (
        13,
        11,
        23,
    ),  # Indices of left elbow, shoulder, and hip joints
    "RIGHT_WRIST_ELBOW_SHOULDER": (
        16,
        14,
        12,
    ),  # Indices of right wrist, elbow, and shoulder joints
    "LEFT_WRIST_ELBOW_SHOULDER": (
        15,
        13,
        11,
    ),  # Indices of left wrist, elbow, and shoulder joints
    "RIGHT_HIP_KNEE_ANKLE": (
        24,
        26,
        28,
    ),  # Indices of right hip, knee, and ankle joints
    "LEFT_HIP_KNEE_ANKLE": (23, 25, 27),  # Indices of left hip, knee, and ankle joints
}
