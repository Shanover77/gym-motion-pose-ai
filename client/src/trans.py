import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(nose, left_eye, right_eye):
    # Calculate the midpoint between the eyes
    mid_eye = (left_eye + right_eye) / 2
    
    # Calculate the direction vector from the nose to the midpoint of the eyes
    direction_vector = mid_eye - nose
    
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # Define a reference vector pointing to the right
    reference_vector = np.array([1, 0, 0])
    
    # Calculate the angle between the direction vector and the reference vector
    angle = np.arccos(np.dot(direction_vector, reference_vector))
    
    # Convert the angle from radians to degrees
    angle = np.degrees(angle)
    
    return angle

# Replace 'path/to/your/video.mp4' with the actual path to your video file
video_path = 'src/spin_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Extract nose, left eye, and right eye coordinates
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y, 0])
        left_eye = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y, 0])
        right_eye = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y, 0])

        # Calculate the angle of translation
        angle = calculate_angle(nose, left_eye, right_eye)

        # Display the angle on the frame
        cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
