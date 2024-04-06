import numpy as np

class PoseProcessor:
    """Processes pose landmarks from MediaPipe."""

    def __init__(self):
        self.all_angles = [(14, 12, 24), (13, 11, 23), (16, 14, 12), (15, 13, 11), (12, 24, 26), (11, 23, 25),
                           (24, 26, 28), (23, 25, 27), (11, 12, 24), (12, 11, 23), (26, 24, 23), (25, 23, 24),
                           (26, 24, 23, 25)]

    def show_lengths_from_results(self, results):
        # Get the pose landmarks of arms
        left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_heel = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]

        # vertical distance shoulder to heel
        midpoint_shoulders = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
        midpoint_heel = [(left_heel.x + right_heel.x) / 2, (left_heel.y + right_heel.y) / 2]
        shoulder_heel_distance = self.get_length(midpoint_shoulders, midpoint_heel)

        # midpoint of left shoulder left hip
        midpoint_left_shoulder_hip = [(left_shoulder.x + left_hip.x) / 2, (left_shoulder.y + left_hip.y) / 2]
        # midpoint of right shoulder right hip
        midpoint_right_shoulder_hip = [(right_shoulder.x + right_hip.x) / 2, (right_shoulder.y + right_hip.y) / 2]
        # distance between the midpoints
        shoulder_hip_distance = self.get_length(midpoint_left_shoulder_hip, midpoint_right_shoulder_hip)

        # ratio of the distances
        shoulder_heel_hip_ratio = shoulder_heel_distance / shoulder_hip_distance
        
        # Calculate the length of the arms
        left_arm_length = self.get_length(left_shoulder, left_elbow) + self.get_length(left_elbow, left_wrist)
        right_arm_length = self.get_length(right_shoulder, right_elbow) + self.get_length(right_elbow, right_wrist)

        # Ratio of the length of the arms
        arm_length_ratio = left_arm_length / right_arm_length
        print("Ratio of the length of the arms:", arm_length_ratio)

        # Length from shoulder to hip
        left_shoulder_to_hip = self.get_length(left_shoulder, left_hip)
        right_shoulder_to_hip = self.get_length(right_shoulder, right_hip)

        # Ratio of the length from shoulder to hip
        shoulder_to_hip_ratio = left_shoulder_to_hip / right_shoulder_to_hip

        # process the results
        angles = self.process(results)

        return angles + [arm_length_ratio, shoulder_to_hip_ratio, shoulder_heel_hip_ratio]

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
