import cv2
import mediapipe as mp
import csv
import os

# Set up MediaPipe for single-hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7
)

# Set to 0 for local laptop camera testing; change to "http://<ESP32-IP>/stream" when hardware arrives
STREAM_URL = 0  

cap = cv2.VideoCapture(STREAM_URL)

TARGET_GESTURES = ["WATER", "YES", "NO", "SICK", "BATHROOM", "PLEASE"]
print("Target Gestures available:", TARGET_GESTURES)
label = input("Enter gesture label to collect (must match one above): ").strip().upper()

if label not in TARGET_GESTURES:
    print(f"Warning: '{label}' is not in your primary list, but it will still be saved.")

csv_file = "hand_gestures.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        # 1 hand * 21 landmarks * 3 coordinates (x, y, z) = 63 features total
        header = ['label'] + [f'feature_{i}' for i in range(63)]
        writer.writerow(header)

    count = 0
    print(f"\n--- COLLECTING DATA FOR: {label} ---")
    print("Hold gesture in front of chest. Press SPACEBAR to capture frame. Press 'q' to stop.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # EGOCENTRIC ADJUSTMENT: Keep raw frame orientation (do NOT flip horizontally)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        keypoints = []
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[0]
            
            # Normalize every landmark relative to the wrist (landmark 0)
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {label} | Saved: {count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Chest-View Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if len(keypoints) == 63:
                writer.writerow([label] + keypoints)
                count += 1
                print(f"Saved sample #{count} for {label}")
            else:
                print("No hand detected in frame! Keep hand in view.")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()