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
    def create_directories(self, directory_names):
        for directory_name in directory_names:
            directory_path = os.path.join(os.getcwd(), directory_name)

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Directory '{directory_name}' created.")
            else:
                print(f"Directory '{directory_name}' already exists.")


    def __init__(self, mp4_file_path):
        self.mp4_file_path = mp4_file_path
        self.mp_pose = mp.solutions.pose.Pose()
        self.pose_processor = PoseProcessor()

        self.sequence_directory = 'sequences'
        self.columns = ['Frame', 'left_arm', 'right_arm', 'left_elbow', 'right_elbow',
                'left_waist_leg', 'right_waist_leg', 'left_knee', 'right_knee',
                'leftup_chest_inside', 'rightup_chest_inside', 'leftlow_chest_inside',
                'rightlow_chest_inside', 'leg_spread']

    def getIndexOfColumn(self, column):
        return self.columns.index(column)
    
    def save_keypoints_to_json(self, filename='keypoints.json'):
        path = f'json/{self.mp4_file_path.split("/")[-1].split(".")[0]}keypoints.json'
        """Save keypoints to a JSON file."""
        with open(path, 'w') as json_file:
            json.dump(self.keypoints, json_file)

    def load_keypoints_from_json(self, filename='keypoints.json'):
        """Load keypoints from a JSON file."""
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
        full_path = os.path.join(self.sequence_directory, filename)

        with open(full_path, 'w') as json_file:
            json.dump(numbers, json_file)

        print(f'Numbers saved to {full_path}')
    
    # STEP METHODS

    def step1_load_mp4(self):
        # Check if the MP4 file exists
        if os.path.exists(self.mp4_file_path):
            print(f"MP4 file '{self.mp4_file_path}' exists. Loading...")
            # Add your loading logic here
        else:
            print(f"Error: MP4 file '{self.mp4_file_path}' not found. Please provide a valid file path.")

    def step2_extract_frames_and_create_pose(self):
        cap = cv2.VideoCapture(self.mp4_file_path)
        keypoints = []

        # Check if the video capture is successful
        if not cap.isOpened():
            print(f"Error: Could not open video file '{self.mp4_file_path}'.")
            return

        frame_count = 0

        while True:
            ret, frame = cap.read()

            # Break the loop if there are no more frames
            if not ret:
                break

            # Process the frame using OpenCV and MediaPipe
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb_image)

            # Assuming you have a pose_processor instance with a process method
            keypoints.append(self.pose_processor.process(results))

            frame_count += 1

        # Release the video capture object
        cap.release()
        self.keypoints = keypoints
        self.save_keypoints_to_json()
        return len(keypoints)

    #   NOTE: NOT REQUIRED
    def step3_calculate_angles(self):
        # Implementation for calculating the 13 angles
        pass

    def step4_calculate_major_exercises(self):
        df = pd.DataFrame(columns=self.columns)


        for index, row in enumerate(self.keypoints):
            if row is None: # Check if row is None
                continue # Skip this iteration if row is None
            
            vals = []
            vals.append(index)
            for i in row:
                if i is None: # Check if i is None
                    continue # Skip this iteration if i is None
                val = i[1]
                vals.append(val)
            
            # Create a new DataFrame for the current row
            new_row_df = pd.DataFrame([vals], columns=df.columns)
            
            # Concatenate the new DataFrame with the existing DataFrame
            df = pd.concat([df, new_row_df], ignore_index=True)


        self.df = df # Save the DataFrame to the object

        # Calculate the rolling mean for each column
        # self.df[self.columns] = self.df[self.columns].rolling(window=90, min_periods=1, axis=0).mean()
        # print('Columns names after rolling:', self.df.head(1))
        # NOTE: KEEP THIS COMMENTED BECAUSE IT HAS SEVERE EFFECTS ON THE CLUSTER SEQUENCE EXTRACTION STEP 7
        
        # Calculate the overall deviation percentage for each column
        df_prop = df.iloc[:, 1:]
        
        overall_deviation_percentage = ((df_prop.max() - df_prop.min()) / df_prop.mean()) * 100
        sorted_columns = overall_deviation_percentage.sort_values(ascending=False)

        NOS_COLUMNS_TO_CONSIDER = 1
        print(sorted_columns[:NOS_COLUMNS_TO_CONSIDER])
        self.top_changing_columns = sorted_columns[:NOS_COLUMNS_TO_CONSIDER].index.tolist()
        return self.top_changing_columns


    #  NOTE: NOT REQUIRED
    def step5_calculate_peaks_and_valleys(self):
        # Implementation for calculating peaks and valleys for each major angle and save for threshold model
        pass

    def step6_kmeans_clustering(self, k=3):
        for col in self.top_changing_columns:
            # Extract the column values for clustering
            column_data = self.df[col].values.reshape(-1, 1)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(column_data)

            # Add cluster labels to the DataFrame
            # cluster_column_name = f'{col}_cluster'
            cluster_column_name = f'cluster'
            self.df[cluster_column_name] = cluster_labels

    def step7_get_entry_exit_clusters(self):
        # Implementation for getting Entry cluster (m=0) and Exit cluster (m=1)
        seq_length = 30  # You can try with 10, 20, 30, 40, 50

        # Desired clusters to extract
        entry_cluster = 0
        exit_cluster = 1

        # Filter the data based on the cluster label for Entry cluster (m=0)
        entry_cluster_data = self.df[self.df['cluster'] == entry_cluster]

        # Extract corresponding frames for Entry cluster
        entry_frames = entry_cluster_data['Frame'][seq_length:].tolist()

        entry_non_continous = self.separate_non_continuous_numbers(entry_frames)
        entry_valid = []
        for entry_li in entry_non_continous:
            if len(entry_li) > seq_length:
                entry_valid.append(entry_li)

        # Save the sequences for Entry cluster
        entry_file_prefix = 'entry_'
        entry_filename = f'{entry_file_prefix}{self.mp4_file_path.split("/")[-1].split(".")[0]}_seq.json'
        self.save_sequence(entry_valid, entry_filename)

        # Filter the data based on the cluster label for Exit cluster (m=1)
        exit_cluster_data = self.df[self.df['cluster'] == exit_cluster]

        # Extract corresponding frames for Exit cluster
        exit_frames = exit_cluster_data['Frame'][seq_length:].tolist()

        exit_non_continous = self.separate_non_continuous_numbers(exit_frames)
        exit_valid = []
        for exit_li in exit_non_continous:
            if len(exit_li) > seq_length:
                exit_valid.append(exit_li)

        # Save the sequences for Exit cluster
        exit_file_prefix = 'exit_'
        exit_filename = f'{exit_file_prefix}{self.mp4_file_path.split("/")[-1].split(".")[0]}_seq.json'
        self.save_sequence(exit_valid, exit_filename)


    def step8_save_to_csv_with_label(self):
        # Implementation for saving sequence of frames from Entry/Exit clusters to csv with label 'Exit' or 'Entry'
        pass

if __name__ == "__main__":

    
    mp4_file_path = "videos/squat.mp4"

    # Create an instance of the Pipeline_V2D class
    pipeline = Pipeline_V2D(mp4_file_path)

    # Create the required directories
    pipeline.create_directories(['json', 'sequences'])

    # Step 1: Load the MP4 file
    pipeline.step1_load_mp4()

    # Step 2: Extract frames and create human pose
    points = pipeline.step2_extract_frames_and_create_pose()

    # Step 4: Calculate major exercises
    top_angles = pipeline.step4_calculate_major_exercises()
    print('Top angle column:', top_angles)

    # Step 6: Perform K-means clustering
    pipeline.step6_kmeans_clustering()

    # Step 7: Get entry and exit clusters
    pipeline.step7_get_entry_exit_clusters()

    # Step 8: Save sequences to CSV with labels
    pipeline.step8_save_to_csv_with_label()