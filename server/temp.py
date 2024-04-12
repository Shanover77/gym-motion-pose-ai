from flask import Flask, jsonify, request, send_file
import numpy as np
from tensorflow import keras
import json
import tempfile
import atexit

app = Flask(__name__)

# Load the saved model (replace 'models/my_model.h5' with your model path if different)
loaded_model = keras.models.load_model('models/exer-5-cosine-keras.h5')

# List to store predictions
predictions_history = []

# Path to the temporary JSON file
temp_json_file = tempfile.NamedTemporaryFile(delete=False)

def save_predictions_to_json():
    with open(temp_json_file.name, 'w') as json_file:
        json.dump(predictions_history, json_file)

# Register the function to be called on program exit
atexit.register(save_predictions_to_json)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    # Preprocess data if needed (e.g., convert to numpy array)
    X_new = np.array([data['features']])  # Assuming data has a 'features' key

    # Make predictions
    prediction = loaded_model.predict(X_new)

    predicted_class = int(prediction.argmax())  # Get the predicted class index

    label_mapping = {'0': 'barbell_dead_row', '1': 'dumbbell_biceps_curls', '2': 'dumbbell_overhead_shoulder_press', '3': 'dumbbell_reverse_lunge'} 

    if label_mapping.get(str(predicted_class)) is not None:  # Check if key exists
        original_label = label_mapping[str(predicted_class)]

        # Store the prediction in the history list
        predictions_history.insert(0, {'prediction': original_label})

        return jsonify({'prediction': original_label})
    else:
        return jsonify({'error': 'Invalid class index'})

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    # Return the list of predictions as JSON
    return jsonify(predictions_history)

@app.route('/get_temp_json', methods=['GET'])
def get_temp_json():
    # Return the temporary JSON file
    return send_file(temp_json_file.name, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
