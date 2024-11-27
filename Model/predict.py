import os
import numpy as np
import librosa
import joblib
import tensorflow as tf

def split_audio(audio, sr, duration=2):
    chunk_length = int(duration * sr)  # Number of samples for 2 seconds
    return [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length) if len(audio[i:i+chunk_length]) == chunk_length]

def extract_mfcc(file_path):
    sample_rate = 22050
    n_mfcc = 13

    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None
    
    # Split the audio into chunks
    y_chunks = split_audio(y, sr)
    
    # Extract MFCCs for each chunk
    mfccs = []
    for chunk in y_chunks:
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        mfccs.append(np.mean(mfcc.T, axis=0))  # Take the mean over time (this is fine for features)
    
    mfccs = np.array(mfccs)
    return mfccs  # Return the full MFCC array for all chunks

def predict(file_path):
    model = tf.keras.models.load_model("C:/Users/ankit/Desktop/Ankit/Audiva/model.h5")
    scaler = joblib.load("C:/Users/ankit/Desktop/Ankit/Audiva/scaler.pkl")

    if not os.path.exists(file_path):
        print("Error: The specified file does not exist.")
        return
    elif not (file_path.lower().endswith(".wav") or file_path.lower().endswith(".mp3"), file_path.lower().endswith(".flac")):
        print("Error: The specified file is not a .wav/.mp3/.flac file.")
        return
    
    mfccs = extract_mfcc(file_path)

    if mfccs is not None:
        # Reshape for scaling: make sure each chunk is a row of 13 MFCC features
        mfccs_reshaped = mfccs.reshape(mfccs.shape[0], -1)
        # Scale the features (scale each chunk independently)
        mfccs_scaled = scaler.transform(mfccs_reshaped)

        # Now, predict using the model
        prediction = model.predict(mfccs_scaled)
        prediction = np.array(prediction)
        mean_prediction = np.mean(prediction)
        prediction_class = 1 if mean_prediction > 0.5 else 0

        if prediction_class == 0:
            print(mean_prediction)
            print("The input audio is classified as AI-generated.")
        else:
            print("The input audio is classified as real.")
    else:
        print("Error: Unable to process the input audio.")

def main():
    file_path = "C:/Users/ankit/Downloads/aihindivoice.wav"
    predict(file_path)

if __name__ == "__main__":
    main()