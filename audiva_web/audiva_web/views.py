from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
import os
import numpy as np
import librosa
import joblib
import tensorflow as tf

def split_audio(audio, sr, duration=2):
    chunk_length = int(duration * sr)
    return [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length) if len(audio[i:i+chunk_length]) == chunk_length]

def extract_mfcc(file_path):
    sample_rate = 22050
    n_mfcc = 13

    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None
    
    y_chunks = split_audio(y, sr)
    
    mfccs = []
    for chunk in y_chunks:
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        mfccs.append(np.mean(mfcc.T, axis=0))
    
    mfccs = np.array(mfccs)
    return mfccs

def predict(file_path):
    model = tf.keras.models.load_model("C:/Users/ankit/Desktop/Ankit/Audiva/model.h5")
    scaler = joblib.load("C:/Users/ankit/Desktop/Ankit/Audiva/scaler.pkl")

    if not os.path.exists(file_path):
        print("Error: The specified file does not exist.")
        return None, None
    elif not (file_path.lower().endswith(('.wav', '.mp3', '.flac'))):
        print("Error: The specified file is not a .wav/.mp3/.flac file.")
        return None, None
    
    mfccs = extract_mfcc(file_path)

    if mfccs is not None:
        # Reshape for scaling: make sure each chunk is a row of 13 MFCC features
        mfccs_reshaped = mfccs.reshape(mfccs.shape[0], -1)
        # Scale the features (scale each chunk independently)
        mfccs_scaled = scaler.transform(mfccs_reshaped)

        prediction = model.predict(mfccs_scaled)
        prediction = np.array(prediction)
        mean_prediction = np.mean(prediction)
        prediction_class = 1 if mean_prediction > 0.5 else 0

        return mean_prediction, prediction_class
    
    else:
        print("Error: Unable to process the input audio.")
        return None, None

def home(request):
    pred_class = 0
    prediction = 0
    error_message = None

    if request.method == 'POST':
        if 'audio_file' not in request.FILES:
            error_message = "Please upload an audio file."
        else:
            audio_file = request.FILES['audio_file']
            if not audio_file.name.lower().endswith(('.wav', '.mp3', '.flac')):
                error_message = "Please upload a valid audio file (.wav)"
            else:
                file_path = f"/tmp/{audio_file.name}"
                with open(file_path, 'wb+') as destination:
                    for chunk in audio_file.chunks():
                        destination.write(chunk)

                prediction, pred_class = predict(file_path)
                prediction = prediction * 100
                prediction = np.round(prediction, 2)

    data = {
        "prediction": prediction,
        "pred_class": pred_class,
        "error_message": error_message
    }

    return render(request, 'home.html', data)

def about(request):
    return render(request, 'about.html')

def signup_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password == confirm_password:
            try:
                User.objects.create_user(username=username, password=password)
                messages.success(request, 'Account created successfully!')
                return redirect('login')
            except:
                messages.error(request, 'Username already exists.')
        else:
            messages.error(request, 'Passwords do not match.')
    return render(request, 'signup.html')

def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')

def logout_user(request):
    logout(request)
    return redirect('login')

def dashboard(request):
    pred_class = 0
    prediction = 0
    error_message = None

    if request.method == 'POST':
        if 'audio_file' not in request.FILES:
            error_message = "Please upload an audio file."
        else:
            audio_file = request.FILES['audio_file']
            if not audio_file.name.lower().endswith(('.wav', '.mp3', '.flac')):
                error_message = "Please upload a valid audio file (.wav)"
            else:
                file_path = f"/tmp/{audio_file.name}"
                with open(file_path, 'wb+') as destination:
                    for chunk in audio_file.chunks():
                        destination.write(chunk)

                prediction, pred_class = predict(file_path)
                prediction = prediction * 100
                prediction = np.round(prediction, 2)

    data = {
        "prediction": prediction,
        "pred_class": pred_class,
        "error_message": error_message
    }

    return render(request, 'dashboard.html', data)