import cv2
import mediapipe as mp

class PoseLandmarks:
    def __init__(self, image_path):
        self.image_path = image_path
        # Initialize the pose landmark model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)

    def get_length(self, point1, point2):
        return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** 0.5
    
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
        


        # Calculate the length of the arms
        left_arm_length = self.get_length(left_shoulder, left_elbow) + self.get_length(left_elbow, left_wrist)
        right_arm_length = self.get_length(right_shoulder, right_elbow) + self.get_length(right_elbow, right_wrist)

        # Print the length of the arms
        print("Length of the arms:")
        print("Left arm:", left_arm_length)
        print("Right arm:", right_arm_length)

        # Ratio of the length of the arms
        arm_length_ratio = left_arm_length / right_arm_length
        print("Ratio of the length of the arms:", arm_length_ratio)

        # Length from shoulder to hip
        left_shoulder_to_hip = self.get_length(left_shoulder, left_hip)
        right_shoulder_to_hip = self.get_length(right_shoulder, right_hip)

        # Print the length from shoulder to hip
        print("Length from shoulder to hip:")
        print("Left shoulder to hip:", left_shoulder_to_hip)
        print("Right shoulder to hip:", right_shoulder_to_hip)

        # Ratio of the length from shoulder to hip
        shoulder_to_hip_ratio = left_shoulder_to_hip / right_shoulder_to_hip
        print("Ratio of the length from shoulder to hip:", shoulder_to_hip_ratio)


    def show_pose_landmarks(self):
        # Load the image
        image = cv2.imread(self.image_path)

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get the pose landmarks
        results = self.pose.process(image_rgb)

        # Draw the pose landmarks on the image
        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            self.show_lengths_from_results(results)

            # Draw the pose landmarks of the arms
            mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))


            left_heel = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
            right_heel = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
            left_foot_index = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot_index = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

            # Show a blue dot on left foot index and right foot index
            cv2.circle(image, (int(left_foot_index.x * image.shape[1]), int(left_foot_index.y * image.shape[0])), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(right_foot_index.x * image.shape[1]), int(right_foot_index.y * image.shape[0])), 5, (255, 0, 0), -1)

            # Draw lines from left heel to left foot index and right heel to right foot index
            cv2.line(image, (int(left_heel.x * image.shape[1]), int(left_heel.y * image.shape[0])), (int(right_heel.x * image.shape[1]), int(right_heel.y * image.shape[0])), (255, 0, 0), 2)
            cv2.line(image, (int(right_foot_index.x * image.shape[1]), int(right_foot_index.y * image.shape[0])), (int(left_foot_index.x * image.shape[1]), int(left_foot_index.y * image.shape[0])), (255, 0, 0), 2)

            
        # Show the image with pose landmarks
        cv2.imshow("Pose Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Path to the image file
image_path = "images/dew0.jpg"

# Create an instance of the PoseLandmarks class
pose_landmarks = PoseLandmarks(image_path)

# Call the show_pose_landmarks method
pose_landmarks.show_pose_landmarks()