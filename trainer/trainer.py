import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import sklearn
import os
from contextlib import redirect_stdout

# Define the custom activation function
def inertia_activation(x, threshold=0.2, inertia_factor=0.1, decay_factor=0.2):
    inertia_term = tf.keras.activations.sigmoid(inertia_factor * x)
    significant_change = tf.keras.activations.relu(x - threshold)
    after_effects = tf.keras.activations.exponential(decay_factor * (x - 0.1))
    return x + inertia_term * significant_change * after_effects

class ModelPipeline:
    def __init__(self, model_name_prefix):
        self.model = None
        self.label_mapping = None
        self.model_name_prefix = model_name_prefix
        self.work_dir = 'temp'
        self.create_work_dir()

    def create_work_dir(self):
        os.makedirs(self.work_dir, exist_ok=True)
    
    def load_data(self, file_paths):
        dfs = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            dfs.append(df)
        combined_df = pd.concat(dfs)
        return combined_df
    
    def save_data(self, df, file_path):
        # Save to working directory
        file_path = os.path.join(self.work_dir, file_path)
        df.to_csv(file_path, index=False)
    
    def preprocess_data(self, df):
        df = df.dropna()
        X_inp = df.iloc[:, :13].values
        y_inp = df['label'].values
        
        X_float = np.empty_like(X_inp, dtype=float)
        for i in range(X_inp.shape[0]):
            for j in range(X_inp.shape[1]):
                try:
                    X_float[i, j] = float(X_inp[i, j])
                except ValueError:
                    X_float[i, j] = np.nan
        
        X = np.radians(X_float)
        
        le = LabelEncoder()
        y = le.fit_transform(y_inp)
        self.label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        return X, y
    
    def save_label_mapping(self, file_path):
        # Save to working directory
        file_path = os.path.join(self.work_dir, file_path)
        with open(file_path, 'w') as f:
            for key, value in self.label_mapping.items():
                f.write(f'{key}: {value}\n')
    
    def saveModelPlot(self, model, file_path):
        # Save to working directory
        file_path = os.path.join(self.work_dir, file_path)
        keras.utils.plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
        
    def save_loss_data(self, history, file_path):
        # Save to working directory
        file_path = os.path.join(self.work_dir, file_path)
        loss_df = pd.DataFrame(history.history)
        loss_df.to_csv(file_path, index=False)

        # Save plots of the loss and accuracy
        loss_df.plot()
        file_path = os.path.join(self.work_dir, f'{self.model_name_prefix}_loss_plot.png')
        plt.savefig(file_path)
        plt.close()

    def save_model_summary(self, file_path, model):
        # Define the filename where you want to save the summary
        filename = file_path

        # Open the file in write mode
        with open(filename, 'w') as f:
            # Redirect the stdout to the file
            with redirect_stdout(f):
                # Print the summary of the model
                model.summary()


    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Sequential()
        model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.20))
        model.add(Dense(65, activation=inertia_activation, name='INERTIA_ACTIVATION'))
        model.add(Dense(len(self.label_mapping), activation='softmax'))        
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.save_model_summary(f'{self.model_name_prefix}_summary.txt', model)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1)
        
        # Generate the model plot
        total_classes = len(set(y))
        # self.saveModelPlot(model, f'{self.model_name_prefix}_{total_classes}_plot.png')

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])
        
        self.model = model

        # Generate the model file name        
        model_file_name = f'{self.model_name_prefix}_{total_classes}.keras'

        # Save the trained model to a file
        self.save_model(model_file_name)
        self.save_loss_data(history, f'{self.model_name_prefix}_loss.csv')
        print(f'Model trained and saved to {model_file_name}')

        self.evaluate_model(X_test, y_test)
        self.save_evaluation('evaluation.csv', X_test, y_test)
        # self.generate_save_classification_report(X_test, y_test)

    def generate_save_classification_report(self, X_test, y_test):
        # Using from sklearn.metrics import classification_report
        y_pred = self.predict(X_test)
        report = sklearn.metrics.classification_report(y_test, y_pred)
        
        # Save to working directory
        file_path = os.path.join(self.work_dir, 'classification_report.txt')
        with open(file_path, 'w') as f:
            f.write(report)        

    def evaluate_model(self, X_test, y_test):
        _, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Model accuracy: {accuracy}')

    def save_evaluation(self, file_path, X_test, y_test):
        predictions = self.model.predict(X_test)
        softmax_predictions = np.argmax(predictions, axis=1)
        predicted_labels = [list(self.label_mapping.keys())[idx] for idx in softmax_predictions]
        evaluation_df = pd.DataFrame({'predicted_label': predicted_labels, 'actual_label': y_test})
        
        # Save to working directory
        file_path = os.path.join(self.work_dir, file_path)
        evaluation_df.to_csv(file_path, index=False)
    
    def save_model(self, file_path):
        # Save to working directory
        file_path = os.path.join(self.work_dir, file_path)        
        self.model.save(file_path)
        print(f'Model saved to {file_path}')
    
    def load_model(self, file_path):
        self.model = load_model(file_path)
    
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        softmax_predictions = np.argmax(predictions, axis=1)
        predicted_labels = [list(self.label_mapping.keys())[idx] for idx in softmax_predictions]
        return predicted_labels


# Example usage
# Set the model name prefix and label mapping file name
model_name_prefix = 'ep50lstm128dense64intertia020102_ex'
label_mapping_file = 'label_mapping.txt'

# Create an instance of the ModelPipeline class
model_pipeline = ModelPipeline(model_name_prefix)

# Load the data from the specified file paths
# Set the directory path where the CSV files are located
directory_path = 'trainable_data/'

# Get all CSV file names in the directory
file_names = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Load the data from the CSV files
df = model_pipeline.load_data([os.path.join(directory_path, file) for file in file_names])

# List of strings that label have as substring to ignore
ignore_labels = ['warmup']
df = df[~df['label'].str.contains('|'.join(ignore_labels))]

# Get the unique labels from the data
len_labels = len(df['label'].unique())
print('Labels:', len_labels)

# Generate the combined data file name
combined_data_file = f'{model_name_prefix}_{len_labels}_combined_data.csv'

# Save the combined data to a file
model_pipeline.save_data(df, combined_data_file)

# Preprocess the data
X, y = model_pipeline.preprocess_data(df)

# Calculate the total number of classes
total_classes = len(set(y))

# Generate the label mapping file name
label_file_name = f'{model_name_prefix}_{total_classes}_label_mapping.txt'

# Save the label mapping to a file
model_pipeline.save_label_mapping(label_file_name)

# Train the model
model_pipeline.train_model(X, y)