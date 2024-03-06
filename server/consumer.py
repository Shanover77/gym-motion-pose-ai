import pika, sys, os
import numpy as np
from tensorflow import keras
import json
from scipy.signal import argrelextrema
import pandas as pd

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')  

# Load the saved model (replace 'models/my_model.h5' with your model path if different)
# loaded_model = keras.models.load_model('models/exer-5-cosine-keras.h5')
loaded_model = keras.models.load_model('models/peak5-exer-combine-ep10-b32.h5')


batch_size = 400
data_batch = []

def find_peaks_and_valleys(df, window_size=100):
        angle_peaks = []

        if df is not None:
            for i, col in enumerate(df.columns):
                if col != 'Frame' and col != 'Data':
                    # Get the x and y data for the current column
                    x = df.index
                    y = df[col]

                    # Find relative maxima (peaks) within a window of 100 frames
                    peaks_indices = argrelextrema(y.values, np.greater, order=window_size)[0]

                    # Find relative minima (valleys) within a window of 100 frames
                    valleys_indices = argrelextrema(y.values, np.less, order=window_size)[0]

                    # Combine peaks and valleys indices
                    apex_indices = np.sort(np.concatenate([peaks_indices, valleys_indices]))

                    # Add to angle peaks
                    angle_peaks.append({'column': col, 'indices': apex_indices})

        return angle_peaks

def preprocess_batch(batch, window_size=100):
    data_points = []
    apex_indices = []

    for i, data_point in enumerate(batch):
        # Assuming 'features' is a key in each data point
        features = data_point['features']
        data_points.append(features)

    columns = [
    'left_arm_cos', 'right_arm_cos', 'left_elbow_cos', 'right_elbow_cos',
    'left_waist_leg_cos', 'right_waist_leg_cos', 'left_knee_cos',
    'right_knee_cos', 'leftup_chest_inside_cos', 'rightup_chest_inside_cos',
    'leftlow_chest_inside_cos', 'rightlow_chest_inside_cos', 'leg_spread_cos'
    ]

    # Create DataFrame
    df = pd.DataFrame(data_points, columns=columns)

    columns_to_smooth = [col for col in df.columns if col != 'Data']

    # Smooth the values of specified columns
    df[columns_to_smooth] = df[columns_to_smooth].rolling(window=90, min_periods=1).mean()
    
    return find_peaks_and_valleys(df)

def combine_and_get_mean(arrays):
  """
  Combines and gets the average mean of arrays of varying lengths.

  Args:
      arrays: A list of NumPy arrays of varying lengths.

  Returns:
      A list containing the mean of each column across all arrays.
  """

  # Find the maximum length of any array
  max_len = max(len(arr) for arr in arrays)

  # Create an empty array to store the combined means
  combined_means = np.zeros(max_len)

  # Iterate over each column index
  for col_idx in range(max_len):
    # Get the values at the current column index from each array
    column_values = [arr[col_idx] if len(arr) > col_idx else np.nan for arr in arrays]

    # Calculate the mean of the column values (handling NaNs)
    combined_means[col_idx] = np.nanmean(column_values)

  return combined_means.tolist()

def process_batch(batch):

    # Preprocess the batch
    peaks = preprocess_batch(batch)

    indices = [peak['indices'] for peak in peaks if len(peak['indices']) > 0]
    
    combined = combine_and_get_mean(indices)
    
    # Get integer indices
    combined = [int(i) for i in combined]

    # Get the data points for the combined indices
    X_new = np.array([point['features'] for i, point in enumerate(batch) if i in combined])

    # Convert the degree angles to radians of cosine\
    X_new = np.cos(np.radians(X_new))

    print("The XNEW from Batch indices", X_new)
    
    predictions = loaded_model.predict(X_new)
    print("The predictions from the model:", predictions)

    label_mapping = {0: 'dumbbell_biceps_curls',
 1: 'dumbbell_overhead_shoulder_press',
 2: 'dumbbell_reverse_lunge'}

    threshold = 0.3  # Set a threshold for class probability

    for i, prediction in enumerate(predictions):
        # Check if any class probability exceeds the threshold
        if np.max(prediction) > threshold:
            predicted_class = int(prediction.argmax())
            print(f'Predicted class for data point {i + 1}: {label_mapping[predicted_class]}')
        else:
            print(f'Predicted class for data point {i + 1}: Uncertain (Low probability)')


    print('Processed batch')


def callback(ch, method, properties, body):
    # Convert data to json
    data = json.loads(body)
    data_batch.append(data)

    if len(data_batch) == batch_size:
        process_batch(data_batch)
        data_batch.clear()

def main():  
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
