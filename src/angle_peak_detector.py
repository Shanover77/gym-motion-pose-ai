import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from constant import JOINTS_NAME
from src.project_root_locator import find_project_root


def get_annotated_csv_path(video_file_name):
    """
    Get the path of the annotated CSV file based on the given video file name.

    Args:
        video_file_name (str): The name of the video file.

    Returns:
        tuple: A tuple containing a boolean indicating the success status and the output CSV path.
    """
    try:
        # Extract file name without extension
        file_name = os.path.splitext(video_file_name)[0]

        # Define the path to the annotated CSV file
        annotated_csv_path = os.path.join(
            find_project_root(), "data", "output_csv", f"annotated_{file_name}.csv"
        )

        # Check if the file exists
        if os.path.exists(annotated_csv_path):
            return True, annotated_csv_path
        else:
            return False, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return False, None


class AnglePeaksDetector:
    def __init__(self, filename, window_size=100):
        """
        Initialize AnglePeaksDetector with the specified parameters.

        Args:
            filename (str): The name of the video file.
            window_size (int, optional): The size of the window for peak detection. Defaults to 100.
        """
        self.window_size = window_size
        self.filename = filename

    def find_angle_peaks(self):
        """
        Find angle peaks in the DataFrame for specified JOINTS_NAME.
        Apply rolling mean to specified JOINTS_NAME before detecting peaks.
        """
        try:
            # Retrieve annotated CSV file path
            status, filepath = get_annotated_csv_path(self.filename)

            # Check if the file exists
            if not status:
                print(f"Error: CSV file not found for {self.filename}")
                return None

            # Load CSV data into DataFrame
            df = pd.read_csv(filepath)

            # Create an empty list to store angle peaks
            angle_peaks = []

            # Apply rolling mean to specified JOINTS_NAME
            df[JOINTS_NAME] = df[JOINTS_NAME].rolling(window=90, min_periods=1).mean()

            # Iterate through each joint column
            for i, col in enumerate(JOINTS_NAME):
                y = df[col]  # Extract the y data for the current joint

                # Find relative maxima (peaks) within a window of frames
                peaks_indices = argrelextrema(
                    y.values, np.greater, order=self.window_size
                )[0]

                # Find relative minima (valleys) within a window of frames
                valleys_indices = argrelextrema(
                    y.values, np.less, order=self.window_size
                )[0]

                # Combine peaks and valleys indices
                apex_indices = np.sort(np.concatenate([peaks_indices, valleys_indices]))

                # Convert all values to integers using list comprehension
                apex_indices = [int(x) for x in apex_indices]

                # Add joint's peak indices to angle_peaks list
                angle_peaks.append(
                    {"column": df.columns[i + 2], "indices": apex_indices}
                )

            # Save angle peaks to a new CSV file
            angle_peaks_filepath = filepath.replace("annotated_", "angle_peaks_")
            pd.DataFrame(angle_peaks).to_csv(angle_peaks_filepath, index=False)
            return angle_peaks
        except KeyError as e:
            print(f"Column '{e.args[0]}' not found in the DataFrame.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
