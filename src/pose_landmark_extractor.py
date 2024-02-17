import cv2
import mediapipe as mp
import os
from .data_saver import DataSaver
from utils.folder_manager import ensure_folder_exists
from utils.joint_angle_calculator import calculate_joint_angles


class PoseLandmarkExtractor:
    """
    PoseLandmarkExtractor extracts pose landmarks from a video and saves the data to a CSV file.

    Args:
        video_file_name (str): Name of the input video file.
        project_root_path (str): Absolute path to the project root.
        video_full_path (str): Absolute path to the input video file.

    Attributes:
        video_file_name (str): Name of the input video file.
        project_root_path (str): Absolute path to the project root.
        video_full_path (str): Absolute path to the input video file.
        mp_drawing (mediapipe.solutions.drawing_utils): MediaPipe drawing utilities.
        mp_pose (mediapipe.solutions.pose): MediaPipe Pose estimator.
        pose_estimator (mediapipe.solutions.pose.Pose): Pose estimator instance.
        data_saver (DataSaver): DataSaver instance for saving pose data to CSV.
    """

    def __init__(self, video_file_name, project_root_path, video_full_path):
        self.video_file_name = video_file_name
        self.project_root_path = project_root_path
        self.video_full_path = video_full_path
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.data_saver = DataSaver(project_root_path, video_file_name)

    def process_frame(self, frame, pose_data_list, frame_index, fps):
        """
        Process a single frame to extract pose landmarks.

        Args:
            frame (numpy.ndarray): Input frame as a NumPy array.
            pose_data_list (list): List to store pose data.
            frame_index (int): Index of the current frame.
            fps (float): Frames per second of the video.

        Returns:
            numpy.ndarray: Processed frame with pose landmarks drawn.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_estimator.process(rgb_frame)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            angle_of_joint = calculate_joint_angles(results.pose_landmarks.landmark)

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                frame_time = frame_index / fps
                pose_data_list.append(
                    [
                        {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility,
                            "landmark": idx,
                            "time": frame_time,  # Time in Seconds
                            "frame_index": frame_index,
                        }
                    ]
                    + list(angle_of_joint.values())
                )
        return frame, angle_of_joint

    def draw_table(self, frame, angle_of_joint):
        """
        Draw a table on the input frame showing joint names and corresponding angles.

        Args:
            frame (numpy.ndarray): Input frame as a NumPy array.
            angle_of_joint (dict): Dictionary containing joint names as keys and corresponding angles as values.

        Returns:
            numpy.ndarray: Frame with the table drawn.

        Comments:
            This function draws a table on the input frame to display joint names and their corresponding angles.
            The table is positioned at the top-left corner of the frame.
            Each row of the table consists of a joint name and its angle.
            The table parameters such as cell width, cell height, font scale, and font thickness can be adjusted for customization.
        """
        # Define table parameters
        table_start_x = 20
        table_start_y = 30
        cell_width = 500  # Increased cell width for more space
        cell_height = 30
        text_color = (0, 0, 0)  # Black color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # Increased font scale for bigger text
        font_thickness = 2  # Added font thickness for bold text
        column_spacing = 30  # Increased spacing between columns

        # Draw table headers
        cv2.putText(
            frame,
            "Joint",
            (table_start_x, table_start_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Angle",
            (table_start_x + cell_width + column_spacing, table_start_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        # Draw angle data
        row_index = 1
        for joint, angle in angle_of_joint.items():
            cv2.putText(
                frame,
                joint,
                (table_start_x, table_start_y + row_index * cell_height),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(angle),
                (
                    table_start_x + cell_width + column_spacing,
                    table_start_y + row_index * cell_height,
                ),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            row_index += 1

        return frame

    def run_extraction(self):
        """
        Run the pose landmark extraction process on the input video.

        Returns:
            bool: True if extraction is successful, False otherwise.
        """

        # Define the output video folder path
        output_video_folder = os.path.join(
            self.project_root_path, "data", "output_videos"
        )

        # Ensure that the output video folder exists
        if not ensure_folder_exists(output_video_folder):
            return False

        output_path = os.path.join(
            self.project_root_path,
            "data",
            "output_videos",
            "annotated_" + self.video_file_name,
        )

        cap = cv2.VideoCapture(self.video_full_path)

        if not cap.isOpened():
            print("Error: Could not open the video file.")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        pose_data_list = []
        angle_of_joint = {}
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            processed_frame, angle_of_joint = self.process_frame(
                frame, pose_data_list, frame_index, fps
            )

            processed_frame = self.draw_table(processed_frame, angle_of_joint)

            cv2.imshow("Processed Frame", processed_frame)
            out.write(processed_frame)

            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        self.pose_estimator.close()
        cv2.destroyAllWindows()

        self.data_saver.save_pose_data_to_csv(pose_data_list)

        return True
