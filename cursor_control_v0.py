import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)

recognizer = vision.GestureRecognizer.create_from_options(options)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for natural movement
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index fingertip landmark (id=8)
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # Draw circle for visualization
            cv2.circle(frame, (x, y), 10, (255, 0, 0), cv2.FILLED)

            # Map to screen coordinates
            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])

            # Move cursor
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    detected_gesture = None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = recognizer.recognize(mp_image)

    if result.gestures:
        detected_gesture = result.gestures[0][0].category_name
        print(f"Detected Gesture: {detected_gesture}")
    if detected_gesture == "Closed_Fist":
        pyautogui.click()
    elif detected_gesture == "Thumb_Up":
        pyautogui.rightClick()
    elif detected_gesture == "Victory":
        pyautogui.scroll(200)


cap.release()
cv2.destroyAllWindows()
