import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Initialize Gesture Recognizer ---
model_path = "gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# --- Initialize Hands for cursor tracking ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

frame_count = 0
detected_gesture = None

print("🎮 Gesture-controlled mouse started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Hand tracking for cursor control ---
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        landmarks = handLms.landmark

        # Index tip
        ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
        cv2.circle(frame, (ix, iy), 10, (255, 0, 0), -1)

        # --- Only move cursor if gesture is "Pointing_Up" ---
        if detected_gesture == "Pointing_Up":
            screen_x = np.interp(ix, [0, w], [0, screen_w])
            screen_y = np.interp(iy, [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y, duration=0)

    # --- Gesture recognition (every 5th frame for efficiency) ---
    frame_count += 1
    if frame_count % 5 == 0:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        if result.gestures:
            detected_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score
            print(f"Detected gesture: {detected_gesture} ({confidence:.2f})")

    # --- Map gestures to actions ---
    if detected_gesture == "Closed_Fist":
        pyautogui.click()
    elif detected_gesture == "Thumb_Up":
        pyautogui.rightClick()
    elif detected_gesture == "Victory":
        pyautogui.scroll(800)
    elif detected_gesture == "Open_Palm":
        pass  # idle state

    cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture-Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
