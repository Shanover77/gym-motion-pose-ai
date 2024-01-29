import pandas as pd
import os
from utils.folder_manager import ensure_folder_exists


class DataSaver:
    """
    A class responsible for saving pose data to a CSV file.

    Attributes:
        project_root_path (str): The root path of the project.
    """

    def __init__(self, project_root_path, video_file_name):
        """
        Initializes the DataSaver.

        Args:
            project_root_path (str): The root path of the project.
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

            # Ensure that the output CSV folder exists
            if not ensure_folder_exists(output_csv_folder):
                return False

            # Split the name of video_file_name
            file_name = self.video_file_name.split(".")[0]

            # Define the output CSV path
            output_csv_path = os.path.join(
                output_csv_folder, "annotated_" + file_name + "_pose_data.csv"
            )

            # Create a DataFrame from the pose data list
            columns = [
                "Frame Index",
                "Landmark Index",
                "X",
                "Y",
                "Z",
                "Visibility",
                "Time (seconds)",
            ]
            df = pd.DataFrame(pose_data_list, columns=columns)

            # Save the DataFrame to a CSV file
            df.to_csv(output_csv_path, index=False)

            print(f"Pose data saved to {output_csv_path}")
            return True

        except Exception as e:
            print(f"Error saving pose data: {e}")
            return False
