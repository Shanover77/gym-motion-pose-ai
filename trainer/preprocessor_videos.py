import cv2
import mediapipe as mp
from src.poseMethods import PoseProcessor
import os
import json
import pandas as pd
import os
from scipy.signal import argrelextrema
import numpy as np

class Pipeline_V2D:

    def __init__(self, mp4_file_path, label):
        self.mp4_file_path = mp4_file_path
        self.mp_pose = mp.solutions.pose.Pose()
        self.pose_processor = PoseProcessor()
        self.label = label

        self.sequence_directory = 'sequences'
        self.trainable_dir = 'trainable_data'
        self.columns = ['Frame', 'left_arm', 'right_arm', 'left_elbow', 'right_elbow',
                        'left_waist_leg', 'right_waist_leg', 'left_knee', 'right_knee',
                        'leftup_chest_inside', 'rightup_chest_inside', 'leftlow_chest_inside',
                        'rightlow_chest_inside', 'leg_spread']

    def create_directories(self, directory_names):
        """
        Create directories if they don't exist.
        """
        for directory_name in directory_names:
            directory_path = os.path.join(os.getcwd(), directory_name)

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Directory '{directory_name}' created.")
            else:
                print(f"Directory '{directory_name}' already exists.")

    def getIndexOfColumn(self, column):
        """
        Get the index of a column in the DataFrame.
        """
        return self.columns.index(column)

    def save_keypoints_to_json(self, filename='keypoints.json'):
        """
        Save keypoints to a JSON file.
        """
        path = f'json/{self.mp4_file_path.split("/")[-1].split(".")[0]}_keypoints.json'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as json_file:
            json.dump(self.keypoints, json_file)

    def load_keypoints_from_json(self, filename='keypoints.json'):
        """
        Load keypoints from a JSON file.
        """
        path = f'json/{self.mp4_file_path.split("/")[-1].split(".")[0]}_keypoints.json'
        try:
            with open(os.path.join(path, filename), 'r') as json_file:
                self.keypoints = json.load(json_file)
            print(f"Keypoints loaded from {os.path.join(path, filename)}.")
        except FileNotFoundError:
            print(f"Error: File {os.path.join(path, filename)} not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {os.path.join(path, filename)}. The file may be corrupted.")

    def separate_non_continuous_numbers(self, numbers):
        """
        Separate non-continuous numbers into clusters.
        """
        clusters = []
        current_cluster = []

        for num in sorted(numbers):
            if not current_cluster or num == current_cluster[-1] + 1:
                current_cluster.append(num)
            else:
                clusters.append(current_cluster)
                current_cluster = [num]

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def save_sequence(self, numbers, filename):
        """
        Save a sequence of numbers to a file.
        """
        full_path = os.path.join(os.getcwd(), self.sequence_directory, filename)

        with open(full_path, 'w') as json_file:
            json.dump(numbers, json_file)

        print(f'Numbers saved to {full_path}')

    def save_apex(self, numbers, filename):
        """
        Save a sequence of numbers to a file.
        """
        full_path = os.path.join(os.getcwd(), 'apex_points', filename)

        with open(full_path, 'w') as json_file:
            json.dump(numbers, json_file)

        print(f'Apex indices data saved to {full_path}')


    def extendListOfNumbers(self, input_list, window=15):
        print('Extending list...', len(input_list))
        extended_set = set()
        extended_list = []

        for num in input_list:
            num = int(num)
            # Add numbers to the left
            extended_set.update(range(max(0, num - window), num))

            # Add the current number
            extended_set.add(num)

            # Add numbers to the right
            extended_set.update(range(num + 1, num + 31))

        # Convert the set back to a sorted list
        extended_list = sorted(extended_set)

        return extended_list

    def step1_load_mp4(self):
        """
        Step 1: Load the MP4 file.
        """
        if os.path.exists(self.mp4_file_path):
            print(f"MP4 file '{self.mp4_file_path}' exists. Loading...")
            # Add your loading logic here
        else:
            print(f"Error: MP4 file '{self.mp4_file_path}' not found. Please provide a valid file path.")

    def step2_extract_frames_and_create_pose(self):
        """
        Step 2: Extract frames and create human pose.
        """
        cap = cv2.VideoCapture(self.mp4_file_path)
        keypoints = []

        if not cap.isOpened():
            print(f"Error: Could not open video file '{self.mp4_file_path}'.")
            return

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb_image)

            keypoints.append(self.pose_processor.processFrame(results, with_index=True))

            frame_count += 1

        cap.release()
        self.keypoints = keypoints
        self.save_keypoints_to_json()
        return len(keypoints)

    def step4_calculate_major_exercises(self):
        """
        Step 4: Calculate major exercises.
        """
        df = pd.DataFrame(columns=self.columns)

        for index, row in enumerate(self.keypoints):
            if row is None:
                continue

            vals = []
            vals.append(index)
            for i in row:
                if i is None:
                    continue
                val = i[1]
                vals.append(val)

            new_row_df = pd.DataFrame([vals], columns=df.columns)
            df = pd.concat([df, new_row_df], ignore_index=True)

        self.df = df

        df_prop = df.iloc[:, 1:]
        percentage_change = (df_prop.std() / df_prop.mean()) * 100
        sorted_columns = percentage_change.sort_values(ascending=False)

        NOS_COLUMNS_TO_CONSIDER = 1
        print(sorted_columns)        
        self.top_changing_columns = sorted_columns[:NOS_COLUMNS_TO_CONSIDER].index.tolist()
        return self.top_changing_columns

    def step6_applyPeakValley(self):
        # use scikit argrelextrema to get the peaks and valleys
        for col in self.top_changing_columns:
            df = self.df
            
            # apply rolling mean to smooth the data
            df[col] = df[col].rolling(window=90).mean()

            # find all peaks and valleys using scikit argrelextrema
            peaks = df.iloc[argrelextrema(df[col].values, np.greater, order=5)]
            valleys = df.iloc[argrelextrema(df[col].values, np.less, order=5)]
            
            print("Peaks:", len(peaks), "Valleys:", len(valleys))

            # indices of peaks and valleys
            peaks_indices = peaks.index.tolist()
            valleys_indices = valleys.index.tolist()
            sorted_apex_indices = sorted(peaks_indices + valleys_indices)

            self.apex_indices = sorted_apex_indices
            print(sorted_apex_indices)

            # save apex indices to json
            self.save_apex(sorted_apex_indices, f'{col}_apex_indices.json')

    def step8_extract_trainable_data(self):

        keypoints = self.keypoints
        apex_indices = self.apex_indices
        print('Indices type', type(apex_indices), len(apex_indices))
              
        # Expand apex_indices to include the frames before and after the apex frames, n=30
        expanded_apex_indices = self.extendListOfNumbers(apex_indices, window=15)        
                   
        # Check if the expanded apex indices are within the range of the keypoints
        expanded_apex_indices = [i for i in expanded_apex_indices if i < len(keypoints)]
        print('Expanded apex indices:', len(expanded_apex_indices))
        
        # Extract keypoints for apex frames
        apex_keypoints = []
        for frame_number in expanded_apex_indices:
            selected_keypoint = keypoints[frame_number]
            if selected_keypoint is not None:
                selected_keypoint = [i[1] for i in selected_keypoint]
                apex_keypoints.append(selected_keypoint)

        # Create a DataFrame with keypoints and a label column
        apex_df = pd.DataFrame(apex_keypoints, columns=self.columns[1:])
        
        # exercise label from mp4 filename
        # mp4_filename = self.mp4_file_path.split("/")[-1].split(".")[0]
        apex_df['label'] = self.label

        # Optionally, you can reset the index if needed
        apex_df.reset_index(drop=True, inplace=True)

        # Save the apex data to a CSV file
        mp4_filename = self.mp4_file_path.split("/")[-1].split(".")[0]
        csv_file_path = os.path.join(self.trainable_dir, f'{mp4_filename}_apex_data.csv')
        
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            combined_df = pd.concat([existing_df, apex_df], ignore_index=True)
            combined_df.to_csv(csv_file_path, index=False)
        else:
            apex_df.to_csv(csv_file_path, index=False)

        print(f"Apex data saved to {os.path.join(self.trainable_dir, f'{mp4_filename}_apex_data.csv')}")


    # Function to create video clips from specified frames
    def create_video_clips(self, frame_numbers, input_video_path, output_folder, subfix='clip'):
        # Read the input video
        video_capture = cv2.VideoCapture(input_video_path)

        # Get video properties
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate through each list of frame numbers
        for i, frame_list in enumerate(frame_numbers):
            # Create a VideoWriter object for the output video clip
            output_filename = f'{self.mp4_file_path.split("/")[-1].split(".")[0]}_{subfix}_{i+1}.mp4'
            output_path = os.path.join(output_folder, output_filename)
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

            # Iterate through each frame number in the list
            for frame_number in frame_list:
                # Set the video capture to the specified frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

                # Read and write the frame to the output video clip
                ret, frame = video_capture.read()
                if ret:
                    video_writer.write(frame)

            # Release the VideoWriter object
            video_writer.release()

        # Release the VideoCapture object
        video_capture.release()


if __name__ == "__main__":

    video_files = []
    for root, dirs, files in os.walk("videos"):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file).replace("\\", "/"))

   
    for video_file in video_files:
        label = video_file.split("/")[-2]
        # remove underscores from label
        label = label.replace("_", " ")
        mp4_file_path = video_file
        
        pipeline = Pipeline_V2D(mp4_file_path, label)

        pipeline.create_directories(['json', 'sequences', 'trainable_data', 'apex_points'])
        print("Directories created.")

        pipeline.step1_load_mp4()
        print("MP4 file loaded.")

        points = pipeline.step2_extract_frames_and_create_pose()
        print("Frames extracted and pose created.")

        top_angles = pipeline.step4_calculate_major_exercises()
        print('Top angle column:', top_angles)

        pipeline.step6_applyPeakValley()
        print("Peaks and valleys saved.")

        pipeline.step8_extract_trainable_data()
        print("Trainable data saved.")