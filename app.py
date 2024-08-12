import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
import librosa
import joblib

# Load the trained model and label encoder
model = tf.keras.models.load_model('emotion_model_cnn.h5')
le = joblib.load('label_encoder_cnn.joblib')

# Function to load and process the audio file
def process_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfccs.shape[1] < 130:
            mfccs = np.pad(mfccs, ((0, 0), (0, 130 - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :130]
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        return mfccs
    except Exception as e:
        messagebox.showerror("Error", f"Could not process audio file: {str(e)}")
        return None

# Function to predict emotion
def predict_emotion():
    file_path = filedialog.askopenfilename()
    if file_path:
        mfccs = process_audio(file_path)
        if mfccs is not None:
            prediction = model.predict(mfccs)
            predicted_label = le.inverse_transform([np.argmax(prediction)])
            result_var.set(f"Predicted Emotion: {predicted_label[0]}")

# Create the main window
root = tk.Tk()
root.title("Emotion Recognition from Audio")
root.geometry("400x200")

# Result variable
result_var = tk.StringVar()

# GUI Elements
label = tk.Label(root, text="Click 'Load Audio' to predict the emotion:")
label.pack(pady=10)

predict_button = tk.Button(root, text="Load Audio", command=predict_emotion)
predict_button.pack(pady=20)

result_label = tk.Label(root, textvariable=result_var)
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
