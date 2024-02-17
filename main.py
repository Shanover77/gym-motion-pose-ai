import os
from dotenv import load_dotenv
from src.project_root_locator import find_project_root
from src.pose_landmark_extractor import PoseLandmarkExtractor
from utils.check_file_existence import check_file_existence


def load_env_variables():
    """
    Load environment variables from the .env file.

    Returns:
    - bool: True if successful, False otherwise.
    """
    try:
        load_dotenv()
        return True
    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return False


def initialize_pose_extractor():
    """
    Initialize the PoseLandmarkExtractor instance with required parameters.

    Returns:
    - tuple: (PoseLandmarkExtractor instance, success flag).
    """
    try:
        # Find and store the project root path
        project_root_path = find_project_root()

        # Access the video file name from environment variables
        video_file_name = os.getenv("VIDEO_FILE_NAME")

        if video_file_name is None:
            print("Error: No environment variable named VIDEO_FILE_NAME found.")
            return None, False

        video_path = os.path.join(
            project_root_path, "data", "input_videos", video_file_name
        )

        if check_file_existence(video_path):
            # Create an instance of the PoseLandmarkExtractor class
            pose_extractor = PoseLandmarkExtractor(
                video_file_name, project_root_path, video_path
            )
            return pose_extractor, True
        else:
            print("Error: Input video file not found.")
            print(f"Path: {video_path}")
            return None, False
    except Exception as e:
        print(f"Error initializing PoseLandmarkExtractor: {e}")
        return None, False


def main():
    """
    Main function to run the pose landmark extraction process.
    """
    try:
        # Load environment variables
        if not load_env_variables():
            print("Error: Failed to load environment variables.")
            return

        # Initialize PoseLandmarkExtractor
        pose_extractor, success = initialize_pose_extractor()
        if not success or pose_extractor is None:
            print("Error: Failed to initialize PoseLandmarkExtractor.")
            return

        # Run the pose landmark extraction process
        pose_extractor.run_extraction()
        print("Success: Pose landmark extraction completed.")
    except Exception as e:
        print(f"Error: {e}")


# Check if the script is being run directly
if __name__ == "__main__":
    # Call the main function
    main()
