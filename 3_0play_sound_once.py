import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import numpy as np
import time

# --- Initialize pygame mixer ---
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

def play_tone(frequency, duration=0.3, volume=0.7):
    """Play a stereo tone of given frequency (Hz) and duration (seconds)."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    wave = np.sin(2 * np.pi * frequency * t)  # sine wave
    wave = np.int16(wave * 32767 * volume)
    
    # Make stereo by stacking two channels
    stereo_wave = np.column_stack((wave, wave))
    
    sound = pygame.sndarray.make_sound(stereo_wave)
    sound.play()

# Map gestures to tone frequencies
gesture_tones = {
    "Open_Palm": 440,
    "Thumb_Up": 1600,
    "Closed_Fist": 800,
    "Victory": 1200,
}

# Track gesture state
current_gesture = None
gesture_start_time = None
sound_played = False

# --- Load model ---
model_path = "gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("🎥 Starting gesture recognition. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = recognizer.recognize(mp_image)

    detected_gesture = None
    if result.gestures:
        detected_gesture = result.gestures[0][0].category_name
        confidence = result.gestures[0][0].score
        text = f"{detected_gesture} ({confidence:.2f})"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        print(f"Detected gesture: {text}")

    # --- Gesture timing logic ---
    current_time = time.time()

    if detected_gesture != current_gesture:
        # Gesture changed, reset timers
        current_gesture = detected_gesture
        gesture_start_time = current_time if detected_gesture else None
        sound_played = False
    elif current_gesture:
        # Gesture held, check duration
        held_time = current_time - gesture_start_time
        if held_time > 0.5 and not sound_played:
            # Play sound after 1 second of initial detection
            if held_time >= 1.0 and current_gesture in gesture_tones:
                play_tone(gesture_tones[current_gesture])
                sound_played = True

    cv2.imshow("Live Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
