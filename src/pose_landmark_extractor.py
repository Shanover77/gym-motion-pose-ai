import cv2
import mediapipe as mp
import os
from .data_saver import DataSaver


class PoseLandmarkExtractor:
    """
    PoseLandmarkExtractor extracts pose landmarks from a video and saves the data to a CSV file.

    Args:
        video_file_name (str): Name of the input video file.
        project_root_path (str): Absolute path to the project root.

    Attributes:
        video_file_name (str): Name of the input video file.
        project_root_path (str): Absolute path to the project root.
        mp_drawing (mediapipe.solutions.drawing_utils): MediaPipe drawing utilities.
        mp_pose (mediapipe.solutions.pose): MediaPipe Pose estimator.
        pose_estimator (mediapipe.solutions.pose.Pose): Pose estimator instance.
        data_saver (DataSaver): DataSaver instance for saving pose data to CSV.
    """

    def __init__(self, video_file_name, project_root_path):
        self.video_file_name = video_file_name
        self.project_root_path = project_root_path
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.data_saver = DataSaver(project_root_path)

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

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                frame_time = frame_index / fps
                pose_data_list.append(
                    [
                        frame_index,
                        idx,
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                        frame_time,
                    ]
                )

        return frame

    def run_extraction(self):
        """
        Run the pose landmark extraction process on the input video.

        Returns:
            bool: True if extraction is successful, False otherwise.
        """
        video_path = os.path.join(
            self.project_root_path, "data", "input_videos", self.video_file_name
        )
        output_path = os.path.join(
            self.project_root_path,
            "data",
            "output_videos",
            "annotated_" + self.video_file_name,
        )

        cap = cv2.VideoCapture(video_path)

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
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            processed_frame = self.process_frame(
                frame, pose_data_list, frame_index, fps
            )

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
