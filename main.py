# Import the necessary module for finding the project root
from src.project_root_locator import find_project_root

# Import the PoseLandmarkExtractor class from the pose_landmark_extractor module
from src.pose_landmark_extractor import PoseLandmarkExtractor


def main():
    # Find and store the project root path
    project_root_path = find_project_root()

    # Specify the video file name
    video_file_name = "barbell_row.mp4"

    # Create an instance of the PoseLandmarkExtractor class
    pose_extractor = PoseLandmarkExtractor(video_file_name, project_root_path)

    # Run the pose landmark extraction process
    pose_extractor.run_extraction()


# Check if the script is being run directly
if __name__ == "__main__":
    # Call the main function
    main()
