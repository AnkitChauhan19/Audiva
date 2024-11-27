import numpy as np
import librosa

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
    return mfccs