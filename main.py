import os
from dotenv import load_dotenv
from src.project_root_locator import find_project_root
from src.pose_landmark_extractor import PoseLandmarkExtractor


def load_environment_variables():
    """Load environment variables from the .env file."""
    try:
        load_dotenv()
        return True
    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return False


def initialize_pose_landmark_extractor():
    """Initialize the PoseLandmarkExtractor instance with required parameters."""
    try:
        # Find and store the project root path
        project_root_path = find_project_root()

        # Access the video file name from environment variables
        video_file_name = os.getenv("VIDEO_FILE_NAME")

        # Create an instance of the PoseLandmarkExtractor class
        pose_extractor = PoseLandmarkExtractor(video_file_name, project_root_path)
        return pose_extractor, True
    except Exception as e:
        print(f"Error initializing PoseLandmarkExtractor: {e}")
        return None, False


def main():
    """Main function to run the pose landmark extraction process."""
    try:
        # Load environment variables
        if not load_environment_variables():
            print("Exiting due to error in loading environment variables.")
            return

        # Initialize PoseLandmarkExtractor
        pose_extractor, success = initialize_pose_landmark_extractor()
        if not success or pose_extractor is None:
            print("Exiting due to error in initializing PoseLandmarkExtractor.")
            return

        # Run the pose landmark extraction process
        pose_extractor.run_extraction()
        print("Pose landmark extraction completed successfully.")
    except Exception as e:
        print(f"Error in main function: {e}")


# Check if the script is being run directly
if __name__ == "__main__":
    # Call the main function
    main()
