import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

def collect_green_channel(video_path):
    cap = cv2.VideoCapture(video_path)
    green_values = []  # List to store green channel values

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract green channel (assuming BGR color space)
        green_frame = frame[:, :, 1]  # Green channel

        # Collect mean value of green channel
        mean_green = np.mean(green_frame)
        green_values.append(mean_green)

    cap.release()
    return green_values

def mean_normalize(signal):
    mean_signal = np.mean(signal)
    normalized_signal = signal - mean_signal
    return normalized_signal

def visualize_signal(signal):
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label="Normalized Green Channel")
    plt.title("Mean-Normalized Heart Rate Signal")
    plt.xlabel("Frame Index")
    plt.ylabel("Normalized Intensity")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
video_path_resting = "C:/Users/yazhi/Desktop/CV_Builder_Series/projects/video_20240628_141429.mp4"
green_values_resting = collect_green_channel(video_path_resting)
normalized_resting_signal = mean_normalize(green_values_resting)
visualize_signal(normalized_resting_signal)
