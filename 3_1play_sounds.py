import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import numpy as np
import time

# --- Initialize pygame mixer ---
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def generate_tone(frequency, duration=1.0, volume=0.7):
    """Generate a pygame Sound object with the given frequency and duration."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    wave = np.sin(2 * np.pi * frequency * t)
    wave = np.int16(wave * 32767 * volume)
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo_wave)

# Map gestures to tone frequencies and create Sound objects
gesture_tones = {
    "Open_Palm": 440,
    "Thumb_Up": 1600,
    "Closed_Fist": 800,
    "Victory": 1200,
}

gesture_sounds = {g: generate_tone(f, duration=1.0) for g, f in gesture_tones.items()}

# Track gesture state
current_gesture = None
gesture_start_time = None
sound_playing = None

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

    current_time = time.time()

    if detected_gesture != current_gesture:
        # Gesture changed: stop previous sound
        if sound_playing:
            sound_playing.stop()
            sound_playing = None

        current_gesture = detected_gesture
        gesture_start_time = current_time if detected_gesture else None

    elif current_gesture:
        # Gesture held
        held_time = current_time - gesture_start_time
        if held_time > 0.5 and not sound_playing:
            if current_gesture in gesture_sounds:
                # Play the sound in a loop (-1 = infinite loop)
                sound_playing = gesture_sounds[current_gesture]
                sound_playing.play(loops=-1)

    cv2.imshow("Live Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
if sound_playing:
    sound_playing.stop()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
