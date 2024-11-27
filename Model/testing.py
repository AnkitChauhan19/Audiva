import tensorflow as tf
from predict import extract_mfcc
import joblib
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")

# Allocate tensors (necessary step to work with the interpreter)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()

# Get output details
output_details = interpreter.get_output_details()

file_path = "C:/Users/ankit/Downloads/Recording.wav"

mfccs = extract_mfcc(file_path)
scaler = joblib.load("scaler.pkl")

mfccs_reshaped = mfccs.reshape(mfccs.shape[0], -1)
mfccs_scaled = scaler.transform(mfccs_reshaped)

mfccs_mean = np.mean(mfccs_scaled, axis=0)

# Reshape to match the model input shape: [1, 13, 1]
mfccs_input = np.expand_dims(mfccs_mean, axis=0)  # Add batch dimension
mfccs_input = np.expand_dims(mfccs_input, axis=-1)  # Add channel dimension

# Ensure data type matches model input type
mfccs_input = mfccs_input.astype(input_details[0]['dtype'])

# Step 4: Set the input tensor and run inference
interpreter.set_tensor(input_details[0]['index'], mfccs_input)

# Run inference
interpreter.invoke()

# Step 5: Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Model prediction:", output_data)