import numpy as np

class PoseProcessor:
    """Processes pose landmarks from MediaPipe."""

    def __init__(self):
        self.all_angles = [(14, 12, 24), (13, 11, 23), (16, 14, 12), (15, 13, 11), (12, 24, 26), (11, 23, 25),
                           (24, 26, 28), (23, 25, 27), (11, 12, 24), (12, 11, 23), (26, 24, 23), (25, 23, 24),
                           (26, 24, 23, 25)]

    def process(self, results, with_index=False):
        """
        Extracts and formats keypoints from MediaPipe pose landmark data.

        Args:
          results: A MediaPipe pose detection results object.

        Returns:
          A list of dictionaries, where each dictionary represents a keypoint
          with the following properties:
            - X: X-coordinate of the keypoint.
            - Y: Y-coordinate of the keypoint.
            - Z: Z-coordinate of the keypoint (if available).
            - Visibility: Visibility score of the keypoint (0: invisible, 1: visible).
            - Landmark: Index of the keypoint.
        """

        if results.pose_landmarks:
            keypoints = []
            index = 0
            for data_point in results.pose_landmarks.landmark:
                this_xyz = {
                    'X': data_point.x,
                    'Y': data_point.y,
                    'Z': data_point.z if hasattr(data_point, 'z') else None,  # Handle optional Z-coordinate
                    'Visibility': data_point.visibility,
                    'Landmark': index,
                }
                keypoints.append(this_xyz)
                index += 1

            # return self.calculate_angles(keypoints, self.all_angles, mode='degree')
            if with_index:
                return self.calculate_angles_with_ind(keypoints, self.all_angles, mode='degree')
            else:
                return self.calculate_angles(keypoints, self.all_angles, mode='degree')
            
        else:
            # Handle the case where no pose landmarks are detected (optional)
            return None  # Or return an empty list, etc.

    def calculate_angle(self, a, b, c, mode='cosine'):
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

    def calculate_leg_spread(self, a, b, c, d, mode='cosine'):
        # Find the midpoint between b and c
        mid_point_bc = [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2]

        # Calculate the angle between a, (b, c), and d using the midpoint as the vertex
        angle = self.calculate_angle(a, mid_point_bc, d, mode)

        return angle

    def get_landmark_xyz(self, coordinates, landmark):
        """Retrieves the X and Y coordinates of a specific landmark."""
        for coord in coordinates:
            if coord['Landmark'] == landmark:
                return [coord['X'], coord['Y']]

        return None  # Handle the case where the landmark is not found

    def calculate_angles(self, coords, joints, mode='cosine'):
        """Calculates cosine radian values or degree values for a list of joint configurations."""
        angles = []
        for joint in joints:
            if len(joint) == 3:
                landmark_first = self.get_landmark_xyz(coords, joint[0])
                landmark_mid = self.get_landmark_xyz(coords, joint[1])
                landmark_end = self.get_landmark_xyz(coords, joint[2])

                angle = self.calculate_angle(landmark_first, landmark_mid, landmark_end, mode)
            else:
                landmark_l_knee = self.get_landmark_xyz(coords, joint[0])
                landmark_l_hip = self.get_landmark_xyz(coords, joint[1])
                landmark_r_hip = self.get_landmark_xyz(coords, joint[2])
                landmark_r_knee = self.get_landmark_xyz(coords, joint[3])

                angle = self.calculate_leg_spread(landmark_l_knee, landmark_l_hip, landmark_r_hip, landmark_r_knee,
                                                  mode)
            angles.append(angle)

        return angles
    
    def calculate_angles_with_ind(self, coords, joints, mode='cosine'):
        """Calculates cosine radian values or degree values for a list of joint configurations."""
        angles = []
        for joint_index, joint in enumerate(joints):
            if len(joint) == 3:
                landmark_first = self.get_landmark_xyz(coords, joint[0])
                landmark_mid = self.get_landmark_xyz(coords, joint[1])
                landmark_end = self.get_landmark_xyz(coords, joint[2])

                angle = self.calculate_angle(landmark_first, landmark_mid, landmark_end, mode)
            else:
                landmark_l_knee = self.get_landmark_xyz(coords, joint[0])
                landmark_l_hip = self.get_landmark_xyz(coords, joint[1])
                landmark_r_hip = self.get_landmark_xyz(coords, joint[2])
                landmark_r_knee = self.get_landmark_xyz(coords, joint[3])

                angle = self.calculate_leg_spread(landmark_l_knee, landmark_l_hip, landmark_r_hip, landmark_r_knee,
                                                mode)
            named_angle = ['left_arm', 'right_arm', 'left_elbow', 'right_elbow',
       'left_waist_leg', 'right_waist_leg', 'left_knee', 'right_knee',
       'leftup_chest_inside', 'rightup_chest_inside', 'leftlow_chest_inside',
       'rightlow_chest_inside', 'leg_spread']
            
            angles.append((named_angle[joint_index], angle))

        return angles
