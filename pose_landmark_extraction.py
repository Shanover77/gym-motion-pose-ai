import cv2
import mediapipe as mp
import pandas as pd

# Load the MediaPipe modules for pose estimation and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define a function to process each frame in a video
def process_frame(frame, pose_estimator, frame_index, fps, data_list):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the MediaPipe Pose estimator
    results = pose_estimator.process(rgb_frame)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Process each landmark
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Calculate time for the current frame
            frame_time = frame_index / fps
            
            print("Frame Index:", frame_index)
            print("idx: ", idx)
            print("x: ", landmark.x)
            print("y: ", landmark.y)
            print("z: ", landmark.z)
            print("visibility: ", landmark.visibility)
            print("Time (seconds):", frame_time)

            # Append data to the list
            data_list.append([frame_index, idx, landmark.x, landmark.y, landmark.z, landmark.visibility, frame_time])

    return frame

# Initialize MediaPipe Pose and Drawing modules
pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video file paths
video_path = "barbell_row.mp4"
output_path = "annotated_" + video_path

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
else:
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the processed video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Data list to store pose information
    pose_data_list = []

    # Process each frame in the video
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()

        # Break the loop if the video is finished
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame, pose_estimator, frame_index, fps, pose_data_list)

        # Display the processed frame (optional)
        cv2.imshow("Processed Frame", processed_frame)

        # Write the processed frame to the output video file
        out.write(processed_frame)

        # Increment frame index
        frame_index += 1

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture, VideoWriter, and Pose objects
    cap.release()
    out.release()
    pose_estimator.close()

    # Destroy any OpenCV windows
    cv2.destroyAllWindows()

    # Convert the data list to a pandas DataFrame
    columns = ["Frame Index", "Landmark Index", "X", "Y", "Z", "Visibility", "Time (seconds)"]
    df = pd.DataFrame(pose_data_list, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv("pose_data.csv", index=False)
