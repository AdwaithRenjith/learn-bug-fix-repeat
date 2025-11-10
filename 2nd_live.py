import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

def print_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        gesture = result.gestures[0][0].category_name
        confidence = result.gestures[0][0].score
        print(f"[{timestamp_ms}] Gesture: {gesture} ({confidence:.2f})")

model_path = "gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,  # callback is required!
)

recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp += 33  # roughly 30 FPS
    recognizer.recognize_async(mp_image, timestamp)

    cv2.imshow("Gesture Recognition (Async)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
