import cv2
import mediapipe as mp
import math
import numpy as np

class PoseLandmarks:
    def __init__(self, video_path):
        self.video_path = video_path
        # Initialize the pose landmark model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False)

    def get_length(self, point1, point2):
        return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** 0.5
    
    def rotate_point(self, point, angle):
        # Function to rotate a point around the x-axis by a given angle
        x, y, z = point
        rotated_y = y * math.cos(angle) - z * math.sin(angle)
        rotated_z = y * math.sin(angle) + z * math.cos(angle)
        return x, rotated_y, rotated_z

    def plot_axes(self, frame, angle):
        # Plot y-axis and z-axis after rotation
        height, width, _ = frame.shape

        # Original y-axis and z-axis points
        origin = np.array([width // 2, height // 2, 0])
        y_axis_end = np.array([width // 2, 0, 0])
        z_axis_end = np.array([width // 2, height // 2, -100])

        # Rotate the points
        rotated_origin = self.rotate_point(origin, angle)
        rotated_y_axis_end = self.rotate_point(y_axis_end, angle)
        rotated_z_axis_end = self.rotate_point(z_axis_end, angle)

        # Draw lines on the frame
        cv2.line(frame, (int(rotated_origin[0]), int(rotated_origin[1])),
                 (int(rotated_y_axis_end[0]), int(rotated_y_axis_end[1])), (0, 255, 0), 2)  # Green line for y-axis
        cv2.line(frame, (int(rotated_origin[0]), int(rotated_origin[1])),
                 (int(rotated_z_axis_end[0]), int(rotated_z_axis_end[1])), (0, 0, 255), 2)  # Red line for z-axis

    
    def angle_from_bottom_horizon(self, point1, point2, frame):

        toggle_vertical_line = False

        # find the bottom point
        bottom_point = point1 if point1.y > point2.y else point2
        bottom_point_vert = point1 if point1.x > point2.x else point2

        # represent a horizontal line crossing the bottom point
        horizontal_line = {'start': (bottom_point.x, bottom_point.y), 'end': (bottom_point.x + 1, bottom_point.y)}
        
        # draw horizontal line on the frame
        cv2.line(frame, (int(horizontal_line['start'][0] * frame.shape[1]), int(horizontal_line['start'][1] * frame.shape[0])), (int(horizontal_line['end'][0] * frame.shape[1]), int(horizontal_line['end'][1] * frame.shape[0])), (100, 120, 25), 1)

        # find angle of line from point1 to point2 with the horizontal line
        angle = math.atan2(point2.y - point1.y, point2.x - point1.x) - math.atan2(horizontal_line['end'][1] - horizontal_line['start'][1], horizontal_line['end'][0] - horizontal_line['start'][0])


        self.plot_axes(frame, angle)


        # draw a dim line from point 1 to point 2 on frame
        cv2.line(frame, (int(point1.x * frame.shape[1]), int(point1.y * frame.shape[0])), (int(point2.x * frame.shape[1]), int(point2.y * frame.shape[0])), (200, 202, 210), 1)

        # Convert angle to degrees
        angle_degrees = math.degrees(angle)
        
        # Check if vertical line drawing is toggled on
        if toggle_vertical_line:
            # represent a vertical line crossing the bottom point
            vertical_line = {'start': (bottom_point_vert.x, bottom_point_vert.y), 'end': (bottom_point_vert.x, bottom_point_vert.y + 1)}

            # draw vertical line on the frame
            cv2.line(frame, (int(vertical_line['start'][0] * frame.shape[1]), int(vertical_line['start'][1] * frame.shape[0])), (int(vertical_line['end'][0] * frame.shape[1]), int(vertical_line['end'][1] * frame.shape[0])), (100, 120, 25), 1)

            # find angle of line from point1 to point2 with the vertical line
            angle_vert = math.atan2(point2.y - point1.y, point2.x - point1.x) - math.atan2(vertical_line['end'][1] - vertical_line['start'][1], vertical_line['end'][0] - vertical_line['start'][0])

            # Convert angle to degrees
            angle_vert_degrees = math.degrees(angle_vert)
        else:
            # Set vertical angle to 0 if vertical line drawing is toggled off
            angle_vert_degrees = 0

        return angle_degrees, angle_vert_degrees
    
    def show_lengths_from_results(self, results, frame):
        # Get the pose landmarks of arms
        left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_foot_index = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot_index = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

        # Find angle of shoulders with the horizontal line
        shoulder_h, shoulder_v = self.angle_from_bottom_horizon(left_shoulder, right_shoulder, frame=frame)

        # find hip angle with the horizontal line
        hip_h, hip_v = self.angle_from_bottom_horizon(left_hip, right_hip, frame=frame)

        # find the angle of the foot index with the horizontal line
        foot_h, foot_v = self.angle_from_bottom_horizon(left_foot_index, right_foot_index, frame=frame)

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
       

        # draw ratios on the frame on top left corner
        cv2.putText(frame, "Arm Length Ratio: {:.2f}".format(arm_length_ratio), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Shoulder to Hip Ratio: {:.2f}".format(shoulder_to_hip_ratio), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw horitonatl and vertical angles on the frame on top right corner
        cv2.putText(frame, "Shoulder Angle: {:.2f}, {:.2f}".format(shoulder_h, shoulder_v), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Hip Angle: {:.2f}, {:.2f}".format(hip_h, hip_v), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Foot Angle: {:.2f}, {:.2f}".format(foot_h, foot_v), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        return frame

    def show_pose_landmarks(self):
        # Open the video file
        video = cv2.VideoCapture(self.video_path)

        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the frame rate of the video
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))

        # Calculate the duration of the video in seconds
        video_duration = total_frames / frame_rate

        # Repeat the video 3 times if the duration is less than 3 seconds
        if video_duration < 5:
            repeat_times = 3
        else:
            repeat_times = 1

        while repeat_times > 0:
            # Reset the video to the beginning
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while True:
                # Read a frame from the video
                ret, frame = video.read()

                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame and get the pose landmarks
                results = self.pose.process(frame_rgb)

                # Draw the pose landmarks on the frame
                if results.pose_landmarks:
                    mp_drawing = mp.solutions.drawing_utils
                    # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    frame = self.show_lengths_from_results(results, frame)

                    # Draw the pose landmarks of the arms
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                # Show the frame with pose landmarks
                cv2.imshow("Pose Landmarks", frame)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            repeat_times -= 1

        # Release the video file and close the windows
        video.release()
        cv2.destroyAllWindows()

# Path to the video file
video_path = "videos/deadlift.mp4"

# Create an instance of the PoseLandmarks class
pose_landmarks = PoseLandmarks(video_path)

# Call the show_pose_landmarks method
pose_landmarks.show_pose_landmarks()