import cv2
import mediapipe as mp
from poseMethods import PoseProcessor
import os
import json
import pandas as pd
from sklearn.cluster import KMeans

class Pipeline_V2D:
    """
    Generates a pipeline for V2D:
    Step 1: Load the mp4 file
    Step 2: Extract the frames and create human pose using mediapipe
    Step 3: Calculate the 13 angles 
    Step 4: Calculate Major exercises by using the angles deviation percentage
    Step 5: Calculate peaks and valleys for each major angle and save for threshold model
    Step 6: K-means clustering for each major angle (k=3)
    Step 7: Get the Entry cluster (m=0), Exit cluster (m=1) 
    Step 8: Get sequence of frames from Entry/Exit clusters and save to csv with label 'Exit' or 'Entry'
    """

    def __init__(self, mp4_file_path):
        self.mp4_file_path = mp4_file_path
        self.mp_pose = mp.solutions.pose.Pose()
        self.pose_processor = PoseProcessor()

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
        path = f'json/{self.mp4_file_path.split("/")[-1].split(".")[0]}keypoints.json'
        with open(path, 'w') as json_file:
            json.dump(self.keypoints, json_file)

    def load_keypoints_from_json(self, filename='keypoints.json'):
        """
        Load keypoints from a JSON file.
        """
        path = f'json/{self.mp4_file_path.split("/")[-1].split(".")[0]}keypoints.json'
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
        full_path = os.path.join(self.sequence_directory, filename)

        with open(full_path, 'w') as json_file:
            json.dump(numbers, json_file)

        print(f'Numbers saved to {full_path}')

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

            keypoints.append(self.pose_processor.process(results, with_index=True))

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

    def step6_kmeans_clustering(self, k=3):
        """
        Step 6: Perform K-means clustering.
        """
        for col in self.top_changing_columns:
            column_data = self.df[col].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(column_data)
            cluster_column_name = f'cluster'
            self.df[cluster_column_name] = cluster_labels

    def step7_get_entry_exit_clusters_and_finalize(self, create_video_clips=False):
        """
        Step 7: Get entry and exit clusters.
        """
        seq_length = 30
        entry_cluster = 0
        exit_cluster = 1

        entry_cluster_data = self.df[self.df['cluster'] == entry_cluster]
        entry_frames = entry_cluster_data['Frame'][seq_length:].tolist()

        entry_non_continous = self.separate_non_continuous_numbers(entry_frames)
        entry_valid = []
        for entry_li in entry_non_continous:
            if len(entry_li) > seq_length:
                entry_valid.append(entry_li)

        self.entry_sequence = entry_valid
        entry_file_prefix = 'entry_'
        entry_filename = f'{entry_file_prefix}{self.mp4_file_path.split("/")[-1].split(".")[0]}_seq.json'
        self.save_sequence(entry_valid, entry_filename)

        exit_cluster_data = self.df[self.df['cluster'] == exit_cluster]
        exit_frames = exit_cluster_data['Frame'][seq_length:].tolist()

        exit_non_continous = self.separate_non_continuous_numbers(exit_frames)
        exit_valid = []
        for exit_li in exit_non_continous:
            if len(exit_li) > seq_length:
                exit_valid.append(exit_li)

        self.exit_sequence = exit_valid
        exit_file_prefix = 'exit_'
        exit_filename = f'{exit_file_prefix}{self.mp4_file_path.split("/")[-1].split(".")[0]}_seq.json'
        self.save_sequence(exit_valid, exit_filename)

        # Call create_video_clips function for both entry and exit frames
        if create_video_clips:
            input_video_path = self.mp4_file_path
            output_folder = 'seq_clips'
            self.create_video_clips(entry_valid, input_video_path, output_folder, subfix='entry')
            self.create_video_clips(exit_valid, input_video_path, output_folder, subfix='exit')

    def step8_extract_trainable_data(self):

        keypoints = self.keypoints
        entry_frames = self.entry_sequence
        exit_frames = self.exit_sequence
        
        # Extract keypoints for entry frames
        entry_keypoints = []
        for sequence in entry_frames:
            for frame_number in sequence:
                selected_keypoint = keypoints[frame_number]
                # select keypoint is collection of 13 list of numbers like ["left_arm", 148.73277432475436], ["right_arm", 154.17960283981398], ["left_elbow", 2.040557170703907], ["right_elbow", 6.575634946201725], ["left_waist_leg", 11.75987017095535], ["right_waist_leg", 11.78409467667364], ["left_knee", 1.0046045256151428], ["right_knee", 4.645123554163573], ["leftup_chest_inside", 98.14604135044894], ["rightup_chest_inside", 98.11155727779096], ["leftlow_chest_inside", 83.67468502022136], ["rightlow_chest_inside", 89.03894876038952], ["leg_spread", 144.3427620913677]
                # extract only the values and append to entry_keypoints
                selected_keypoint = [i[1] for i in selected_keypoint]
                entry_keypoints.append(selected_keypoint)

        # Create a DataFrame with keypoints and a label column
        entry_df = pd.DataFrame(entry_keypoints, columns=self.columns[1:])
        entry_df['Label'] = 'Entry'

        # Assuming you want to add a column for sequence numbers
        entry_df['Sequence'] = [i for i, sequence in enumerate(entry_frames) for _ in sequence]

        # Optionally, you can reset the index if needed
        entry_df.reset_index(drop=True, inplace=True)

        # Extract keypoints for exit frames
        exit_keypoints = []
        for sequence in exit_frames:
            for frame_number in sequence:
                selected_keypoint = keypoints[frame_number]
                selected_keypoint = [i[1] for i in selected_keypoint]
                exit_keypoints.append(selected_keypoint)

        # Create a DataFrame with keypoints and a label column
        exit_df = pd.DataFrame(exit_keypoints, columns=self.columns[1:])
        exit_df['Label'] = 'Exit'

        # Assuming you want to add a column for sequence numbers
        exit_df['Sequence'] = [i for i, sequence in enumerate(exit_frames) for _ in sequence]

        # Optionally, you can reset the index if needed
        exit_df.reset_index(drop=True, inplace=True)

        # Concatenate entry and exit DataFrames
        trainable_data = pd.concat([entry_df, exit_df], ignore_index=True)

        # Save the trainable data to a CSV file
        mp4_filename = self.mp4_file_path.split("/")[-1].split(".")[0]
        trainable_data.to_csv(os.path.join(self.trainable_dir, f'{mp4_filename}_trainable_data.csv'), index=False)

        print(f"Trainable data saved to {os.path.join(self.trainable_dir, f'{mp4_filename}_trainable_data.csv')}")

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
    video_files = [
        "videos/train_dumbbell_biceps_curls.mp4",
        "videos/test_dumbbell_biceps_curls.mp4",
        "videos/validate_dumbbell_curl_trifecta.mp4"
    ]

    for mp4_file_path in video_files:
        pipeline = Pipeline_V2D(mp4_file_path)

        pipeline.create_directories(['json', 'sequences','clips'])
        print("Directories created.")

        pipeline.step1_load_mp4()
        print("MP4 file loaded.")

        points = pipeline.step2_extract_frames_and_create_pose()
        print("Frames extracted and pose created.")

        top_angles = pipeline.step4_calculate_major_exercises()
        print('Top angle column:', top_angles)

        pipeline.step6_kmeans_clustering()
        print("K-means clustering performed.")

        pipeline.step7_get_entry_exit_clusters_and_finalize()
        print("Entry and exit clusters obtained.")

        pipeline.step8_extract_trainable_data()
        print("Trainable data extracted.")
