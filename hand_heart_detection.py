import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# masukkan gambar hati kamu
original_heart_img = Image.open('love.png') 


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            distance = np.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)

            if distance < 0.1:  # Batas threshold yang lebih besar
                # Hitung ukuran love berdasarkan jarak antara ibu jari dan telunjuk
                new_size = int(distance * w * 2)  # Sesuaikan skala agar ukuran lebih proporsional
                resized_heart_img = original_heart_img.resize((new_size, new_size))

                heart_np = cv2.cvtColor(np.array(resized_heart_img), cv2.COLOR_RGBA2BGRA)

                center_x = int((thumb_tip.x + index_finger_tip.x) * w // 2)
                center_y = int((thumb_tip.y + index_finger_tip.y) * h // 2)

                heart_x = max(0, min(center_x - new_size // 2, w - new_size))
                heart_y = max(0, min(center_y - new_size // 2, h - new_size))

                for i in range(new_size):
                    for j in range(new_size):
                        if heart_np[i, j][3] != 0:  # Periksa alpha channel
                            frame[heart_y + i, heart_x + j] = heart_np[i, j][:3]

    cv2.imshow("Heart Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
