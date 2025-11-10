import cv2, mediapipe as mp, pyautogui, numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        landmarks = handLms.landmark

        # Index tip & thumb tip
        ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
        tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)
        cv2.circle(frame, (ix, iy), 10, (255, 0, 0), -1)

        # Move cursor
        screen_x = np.interp(ix, [0, w], [0, screen_w])
        screen_y = np.interp(iy, [0, h], [0, screen_h])
        pyautogui.moveTo(screen_x, screen_y, duration=0)

        # Distance between thumb and index finger
        dist = np.hypot(tx - ix, ty - iy)
        if dist < 40:
            pyautogui.click()
            cv2.circle(frame, (ix, iy), 15, (0, 255, 0), -1)

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
