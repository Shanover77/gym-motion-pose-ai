import numpy as np
import math
from mediapipe.python.solutions.pose import PoseLandmark

class PoseProcessor:
    """Processes pose landmarks from MediaPipe."""

    # constructor
    def __init__(self):
        self.features = []
        self.joint_landmarks_combinations = [(14, 12, 24), (13, 11, 23), (16, 14, 12), (15, 13, 11), (12, 24, 26), (11, 23, 25),
                           (24, 26, 28), (23, 25, 27), (11, 12, 24), (12, 11, 23), (26, 24, 23), (25, 23, 24),
                           (26, 24, 23, 25)]

    # process pose landmarks
    def process(self, landmarks):
        # calculate features
        self.features = self.calculate_features(landmarks)

    def calculate_angle(self, a, b, c, mode='degree'):
        """Calculates the angle between three 2D points."""
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point
        c = np.array(c)  # End point

        # Calculate vector differences
        v1 = b - a
        v2 = c - b

        # Normalize vectors (optional for improved stability)
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Calculate cosine of the angle
        cosine_angle = np.dot(v1_norm, v2_norm)

        if mode == 'degree':
            # Convert cosine to degree
            angle = np.degrees(np.arccos(cosine_angle))
        else:
            angle = cosine_angle

        return angle