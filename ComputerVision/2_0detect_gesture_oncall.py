import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Step 1: Load model ---
model_path = "gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)

# --- Step 2: Configure options ---
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)

# --- Step 3: Create recognizer ---
recognizer = vision.GestureRecognizer.create_from_options(options)

# --- Step 4: Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# --- Step 5: Run loop ---
print("🎥 Starting gesture recognition. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame (OpenCV BGR → RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run recognition
    result = recognizer.recognize(mp_image)

    # Display gesture results
    if result.gestures:
        gesture = result.gestures[0][0].category_name
        confidence = result.gestures[0][0].score
        text = f"{gesture} ({confidence:.2f})"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        print(f"Detected gesture: {text}")

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
