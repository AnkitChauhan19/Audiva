import os
import numpy as np
import librosa
from concurrent.futures import ProcessPoolExecutor

# MFCC and model parameters
n_mfcc = 13
sample_rate = 22050  # Commonly used sample rate

# Paths to folders
data_path = "C:/Users/ankit/Desktop/Ankit/Audiva/Data/for-2seconds/train_test_combined"
val_path = "C:/Users/ankit/Desktop/Ankit/Audiva/Data/for-2seconds/validation"

# Function to extract and preprocess MFCCs
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Function to process each file
def process_file(file_path, label):
    mfcc_feature = preprocess_audio(file_path)
    if mfcc_feature is None:
        return None
    return mfcc_feature, label

# Function to load dataset
def load_dataset_parallel(path):
    features = []
    labels = []

    for label in ["fake", "real"]:
        label_dir = os.path.join(path, label)
        file_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith(".wav")]
        
        # Use executor with global function
        with ProcessPoolExecutor() as executor:
            results = executor.map(process_file, file_paths, [label] * len(file_paths))
            for result in results:
                if result is not None:
                    feature, lbl = result
                    features.append(feature)
                    if lbl == "fake":
                        labels.append(0)
                    else:
                        labels.append(1)
    
    combined = list(zip(features, labels))
    np.random.shuffle(combined)
    features, labels = zip(*combined)

    return np.array(features, dtype = object), np.array(labels)

def main():
    for dataset_type, path in zip(['dataset', 'val'], [data_path, val_path]):
        X, y = load_dataset_parallel(path)
        np.savez_compressed(f"{dataset_type}_data.npz", features=X, labels=y)

    print("Data preprocessing complete and saved to .npz files.")
    
# Main script
if __name__ == '__main__':
    main()