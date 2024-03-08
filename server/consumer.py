import pika
import numpy as np
from tensorflow import keras
import json
from scipy.signal import argrelextrema
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Consumer:
    def __init__(self, model_path, batch_size=100, window_size=10):
        self.model_path = model_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.data_batch = []
        self.loaded_model = None
        self.channel = None

    def connect_to_rabbitmq(self, host="localhost"):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = connection.channel()
        self.channel.queue_declare(queue="hello")

    def load_model(self):
        self.loaded_model = keras.models.load_model(self.model_path)

    def find_peaks_and_valleys(self, df):
        angle_peaks = []

        if df is not None:
            for i, col in enumerate(df.columns):
                if col != "Frame" and col != "Data":
                    x = df.index
                    y = df[col]

                    peaks_indices = argrelextrema(y.values, np.greater, order=self.window_size)[0]
                    valleys_indices = argrelextrema(y.values, np.less, order=self.window_size)[0]
                    apex_indices = np.sort(np.concatenate([peaks_indices, valleys_indices]))

                    angle_peaks.append({"column": col, "indices": apex_indices})

        return angle_peaks

    def preprocess_batch(self, batch):
        data_points = []
        apex_indices = []

        for i, data_point in enumerate(batch):
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

        df = pd.DataFrame(data_points, columns=columns)
        columns_to_smooth = [col for col in df.columns if col != "Data"]
        df[columns_to_smooth] = df[columns_to_smooth].rolling(window=90, min_periods=1).mean()

        return self.find_peaks_and_valleys(df)

    def combine_and_get_mean(self, arrays):
        max_len = max(len(arr) for arr in arrays)
        combined_means = np.zeros(max_len)

        for col_idx in range(max_len):
            column_values = [arr[col_idx] if len(arr) > col_idx else np.nan for arr in arrays]
            combined_means[col_idx] = np.nanmean(column_values)

        return combined_means.tolist()

    def process_batch(self, batch):
        peaks = self.preprocess_batch(batch)
        indices = [peak["indices"] for peak in peaks if len(peak["indices"]) > 0]
        combined = self.combine_and_get_mean(indices)
        combined = [int(i) for i in combined]
        X_raw = np.array([point["features"] for i, point in enumerate(batch) if i in combined])
        X_new = np.radians(X_raw)

        predictions = self.loaded_model.predict(X_new)

        label_mapping = {
            0: "band_pull_apart",
            1: "barbell_dead_row",
            2: "barbell_row",
            # ... add more labels here ...
        }

        threshold = 0.3

        for i, prediction in enumerate(predictions):
            if np.max(prediction) > threshold:
                predicted_class = int(prediction.argmax())
                print(f"Predicted class for data point {i + 1}: {label_mapping[predicted_class]}")
            else:
                print(f"Predicted class for data point {i + 1}: Uncertain (Low probability)")

        most_likely_class = np.argmax(np.bincount(np.argmax(predictions, axis=1)))
        print(f"Most likely class: {label_mapping[most_likely_class]}")

        # Publish the most likely class to the "prediction" queue
        self.channel.basic_publish(
            exchange="",
            routing_key="prediction",
            body=json.dumps({"class": label_mapping[most_likely_class]}),
        )
        print(" [*] Predictions sent to the 'prediction' queue")

    def callback(self, ch, method, properties, body):
        data = json.loads(body)
        self.data_batch.append(data)

        if len(self.data_batch) == self.batch_size:
            self.process_batch(self.data_batch)
            self.data_batch.clear()

    def start_consuming(self):
        self.channel.basic_consume(queue="hello", on_message_callback=self.callback, auto_ack=True)
        print(" [*] Waiting for messages. To exit press CTRL+C")
        self.channel.start_consuming()

if __name__ == "__main__":
    model_path = "models/exer24_apex_pipe_w16degtorad.h5"
    consumer = Consumer(model_path)
    consumer.connect_to_rabbitmq()
    consumer.load_model()
    consumer.start_consuming()
