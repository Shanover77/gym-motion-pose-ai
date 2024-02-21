import os
import csv
from utils.folder_manager import ensure_folder_exists


class DataSaver:
    """
    A class responsible for saving pose data to a CSV file.

    Attributes:
        project_root_path (str): The root path of the project.
        video_file_name (str): The name of the video file.
    """

    def __init__(self, project_root_path, video_file_name):
        """
        Initializes the DataSaver.

        Args:
            project_root_path (str): The root path of the project.
            video_file_name (str): The name of the video file.
        """
        self.project_root_path = project_root_path
        self.video_file_name = video_file_name

    def save_pose_data_to_csv(self, pose_data_list):
        """
        Saves pose data to a CSV file.

        Args:
            pose_data_list (list): List containing pose data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Define the output CSV folder path
            output_csv_folder = os.path.join(
                self.project_root_path, "data", "output_csv"
            )

            # Ensure that the output CSV folder exists; create if it doesn't
            if not ensure_folder_exists(output_csv_folder):
                return False

            # Split the name of video_file_name
            file_name = self.video_file_name.split(".")[0]

            # Define the output CSV path
            output_csv_path = os.path.join(
                output_csv_folder, "annotated_" + file_name + ".csv"
            )

            # Write the header and data to the CSV file
            with open(output_csv_path, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                header = [
                    "data",
                    "frame",
                    "left_arm",
                    "right_arm",
                    "left_elbow",
                    "right_elbow",
                    "left_waist_leg",
                    "right_waist_leg",
                    "left_knee",
                    "right_knee",
                    "leftup_chest_inside",
                    "rightup_chest_inside",
                    "leftlow_chest_inside",
                    "rightlow_chest_inside",
                    "leg_spread",
                ]
                csv_writer.writerow(header)

                for pose_entry in pose_data_list:
                    csv_writer.writerow(pose_entry)

            print(f"Pose data saved to {output_csv_path}")
            return True

        except Exception as e:
            print(f"Error saving pose data: {e}")
            return False
