import os
import shutil
from src.project_root_locator import find_project_root


def cleanup_temp_images(all_angle_indices):
    """
    Cleans up temporary images in the output folder, moving files related to angle indices
    specified in `all_angle_indices` to a different folder.

    Args:
        all_angle_indices (list): List of angle indices to keep.

    Returns:
        bool: True if cleanup is successful, False otherwise.
    """
    try:
        # Find and store the project root path
        project_root_path = find_project_root()

        # Define the source and destination folder paths
        source_folder = os.path.join(
            project_root_path, "data", "output_images", "temp_images"
        )
        destination_folder = os.path.join(
            project_root_path, "data", "output_images", "angle_peaks_images"
        )

        # Check if the source folder exists
        if not os.path.exists(source_folder):
            raise FileNotFoundError(f"Source folder '{source_folder}' does not exist.")

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Get a list of all files in the source folder
        all_files = os.listdir(source_folder)

        # Iterate through all files in the source folder
        for file_name in all_files:
            # Extract the frame index from the file name
            parts = file_name.rsplit(".", 2)
            if len(parts) < 2:
                print(f"Invalid file name format: {file_name}")
                continue

            file_name_without_extension = parts[0]
            frame_index = int(file_name_without_extension.rsplit("_", 1)[-1])

            file_path = os.path.join(source_folder, file_name)
            dest_file_path = os.path.join(destination_folder, file_name)

            # Check if the file is not in the list of files to keep
            if frame_index not in all_angle_indices:
                # Delete the file
                os.remove(file_path)
            else:
                # If the destination folder already contains a file with the same name,
                # skip moving the file
                if os.path.exists(dest_file_path):
                    continue

                # Move the file to the destination folder
                shutil.move(file_path, destination_folder)

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
