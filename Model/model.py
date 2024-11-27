import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report
import joblib

def check_missing_or_infinite(data, data_name="Dataset"):
    if np.any(np.isnan(data)):
        print(f"{data_name} contains NaN values.")
    else:
        print(f"{data_name} has no NaN values.")
        
    if np.any(np.isinf(data)):
        print(f"{data_name} contains infinite values.")
    else:
        print(f"{data_name} has no infinite values.")

def train_model(X, y, X_val, y_val):
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Normalisation of data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshaping datasets for LSTM compatibility
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    '''# Checking for missing or infinite values in the dataset after scaling(to be executed only once)
    check_missing_or_infinite(X_train, "X_train")
    check_missing_or_infinite(X_val, "X_val")
    check_missing_or_infinite(X_test, "X_test")'''

    # Define model architecture
    input_shape = (X_train.shape[1], 1)
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Predictions and Classification Report
    y_pred = (model.predict(X_test) > 0.6).astype("int32")
    print(classification_report(y_test, y_pred, target_names=["fake", "real"]))

    # Saving the model
    model.save("model.h5")
    joblib.dump(scaler, "scaler.pkl")

def main():
    # Load preprocessed data
    data = np.load("dataset_data.npz", allow_pickle=True)
    X, y = data['features'], data['labels']

    val_data = np.load("val_data.npz", allow_pickle=True)
    X_val, y_val = val_data['features'], val_data['labels']

    train_model(X, y, X_val, y_val)

if __name__ == "__main__":
    main()