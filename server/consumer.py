import pika, sys, os
import numpy as np
from tensorflow import keras
import json
from scipy.signal import argrelextrema
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
channel = connection.channel()
channel.queue_declare(queue="hello")


# Load the saved model (replace 'models/my_model.h5' with your model path if different)
# loaded_model = keras.models.load_model('models/exer-5-cosine-keras.h5')
# loaded_model = keras.models.load_model('models/exer7_s05_front_lstm_enex.h5')

MODEL_PATH = "models/exer24_apex_pipe_w16degtorad.h5"
loaded_model = keras.models.load_model(MODEL_PATH)

batch_size = 100
data_batch = []


def find_peaks_and_valleys(df, window_size=10):
    angle_peaks = []

    if df is not None:
        for i, col in enumerate(df.columns):
            if col != "Frame" and col != "Data":
                # Get the x and y data for the current column
                x = df.index
                y = df[col]

                # Find relative maxima (peaks) within a window of 100 frames
                peaks_indices = argrelextrema(y.values, np.greater, order=window_size)[
                    0
                ]

                # Find relative minima (valleys) within a window of 100 frames
                valleys_indices = argrelextrema(y.values, np.less, order=window_size)[0]

                # Combine peaks and valleys indices
                apex_indices = np.sort(np.concatenate([peaks_indices, valleys_indices]))

                # Add to angle peaks
                angle_peaks.append({"column": col, "indices": apex_indices})

    return angle_peaks


def preprocess_batch(batch, window_size=100):
    data_points = []
    apex_indices = []

    for i, data_point in enumerate(batch):
        # Assuming 'features' is a key in each data point
        features = data_point["features"]
        data_points.append(features)

    columns = [
        "left_arm_cos",
        "right_arm_cos",
        "left_elbow_cos",
        "right_elbow_cos",
        "left_waist_leg_cos",
        "right_waist_leg_cos",
        "left_knee_cos",
        "right_knee_cos",
        "leftup_chest_inside_cos",
        "rightup_chest_inside_cos",
        "leftlow_chest_inside_cos",
        "rightlow_chest_inside_cos",
        "leg_spread_cos",
    ]

    # Create DataFrame
    df = pd.DataFrame(data_points, columns=columns)

    columns_to_smooth = [col for col in df.columns if col != "Data"]

    # Smooth the values of specified columns
    df[columns_to_smooth] = (
        df[columns_to_smooth].rolling(window=90, min_periods=1).mean()
    )

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
        column_values = [
            arr[col_idx] if len(arr) > col_idx else np.nan for arr in arrays
        ]

        # Calculate the mean of the column values (handling NaNs)
        combined_means[col_idx] = np.nanmean(column_values)

    return combined_means.tolist()


def process_batch(batch):

    # Preprocess the batch
    peaks = preprocess_batch(batch)

    indices = [peak["indices"] for peak in peaks if len(peak["indices"]) > 0]

    combined = combine_and_get_mean(indices)

    # Get integer indices
    combined = [int(i) for i in combined]

    # Get the data points for the combined indices
    X_raw = np.array(
        [point["features"] for i, point in enumerate(batch) if i in combined]
    )

    # Scale the data using MinMaxScaler
    X_new = np.radians(X_raw)

    print("The XNEW from Batch indices", X_new)

    predictions = loaded_model.predict(X_new)
    print("The predictions from the model:", predictions[0])

    label_mapping = {
        0: "band_pull_apart",
        1: "barbell_dead_row",
        2: "barbell_row",
        3: "barbell_shrug",
        4: "burpees",
        5: "clean_and_press",
        6: "deadlift",
        7: "diamond_pushup",
        8: "drag_curl",
        9: "dumbbell_biceps_curls",
        10: "dumbbell_curl_trifecta",
        11: "dumbbell_hammer_curls",
        12: "dumbbell_high_pulls",
        13: "dumbbell_overhead_shoulder_press",
        14: "dumbbell_reverse_lunge",
        15: "dumbbell_scaptions",
        16: "man_maker",
        17: "mule_kick",
        18: "neutral_overhead_shoulder_press",
        19: "one_arm_row",
        20: "overhead_extension_thruster",
        21: "overhead_trap_raises",
        22: "pushup",
        23: "side_lateral_raise",
    }

    threshold = 0.3  # Set a threshold for class probability

    for i, prediction in enumerate(predictions):
        # Check if any class probability exceeds the threshold
        if np.max(prediction) > threshold:
            predicted_class = int(prediction.argmax())
            print(
                f"Predicted class for data point {i + 1}: {label_mapping[predicted_class]}"
            )
        else:
            print(
                f"Predicted class for data point {i + 1}: Uncertain (Low probability)"
            )

    # Most likely class having more frequency
    most_likely_class = np.argmax(np.bincount(np.argmax(predictions, axis=1)))
    print(f"Most likely class: {label_mapping[most_likely_class]}")

    # Send the most likely class to the next queue
    channel.basic_publish(
        exchange="",
        routing_key="prediction",
        body=json.dumps({"class": label_mapping[most_likely_class]}),
    )
    
    print("____Processed batch____")


def callback(ch, method, properties, body):
    # Convert data to json
    data = json.loads(body)
    data_batch.append(data)

    if len(data_batch) == batch_size:
        process_batch(data_batch)
        data_batch.clear()


def main():
    channel.basic_consume(queue="hello", on_message_callback=callback, auto_ack=True)

    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
