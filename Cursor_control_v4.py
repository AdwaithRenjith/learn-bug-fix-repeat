import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Gesture Recognizer ---
model_path = "gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# --- Hands for tracking ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

frame_count = 0
detected_gesture = None

# --- Relative cursor variables ---
prev_x, prev_y = None, None
sensitivity = 3 
delta_queue_size = 5  # moving average window size
dx_queue = deque(maxlen=delta_queue_size)
dy_queue = deque(maxlen=delta_queue_size)

# --- Action cooldowns ---
gesture_last_action_time = {}
action_cooldown = 1.0  # seconds

print("🎮 Smoothed trackpad-like gesture mouse started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Hand tracking ---
    results = hands.process(rgb)
    current_x, current_y = None, None

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        landmarks = handLms.landmark

        # Index fingertip
        current_x = int(landmarks[8].x * w)
        current_y = int(landmarks[8].y * h)
        cv2.circle(frame, (current_x, current_y), 10, (255, 0, 0), -1)

    # --- Gesture recognition every 10 frames ---
    frame_count += 1
    if frame_count % 10 == 0 and current_x is not None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)
        if result.gestures:
            detected_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score
            # print(f"Detected gesture: {detected_gesture} ({confidence:.2f})")

    current_time = time.time()

    # --- Relative cursor movement for "Pointing_Up" ---
    if detected_gesture == "Pointing_Up" and prev_x is not None and current_x is not None:
        dx = (current_x - prev_x) * sensitivity
        dy = (current_y - prev_y) * sensitivity

        # Append to deque for smoothing
        dx_queue.append(dx)
        dy_queue.append(dy)

        # Compute smoothed delta
        smooth_dx = np.mean(dx_queue)
        smooth_dy = np.mean(dy_queue)

        # Move cursor
        cur_pos = pyautogui.position()
        new_x = np.clip(cur_pos[0] + smooth_dx, 0, screen_w)
        new_y = np.clip(cur_pos[1] + smooth_dy, 0, screen_h)
        pyautogui.moveTo(new_x, new_y, duration=0)

    # --- Gesture actions with cooldown ---
    gesture_actions = {
        "Closed_Fist": lambda: pyautogui.click(),
        "Thumb_Up": lambda: pyautogui.rightClick(),
        "Victory": lambda: pyautogui.scroll(200)
    }

    if detected_gesture in gesture_actions:
        last_time = gesture_last_action_time.get(detected_gesture, 0)
        if current_time - last_time >= action_cooldown:
            gesture_actions[detected_gesture]()
            gesture_last_action_time[detected_gesture] = current_time

    # Update previous hand position
    if current_x is not None:
        prev_x, prev_y = current_x, current_y

    cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Smoothed Trackpad Gesture Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
