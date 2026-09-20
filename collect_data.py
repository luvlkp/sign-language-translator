"""
Collect training data for the sign classifier.

Uses the same camera setup as server/app.py (ESP32 /capture URL or a webcam index)
and the same features: 21 hand landmarks relative to the wrist (63 numbers).

Run from the repo root:   py collect_data.py
Controls (click the preview window first):  SPACE = save sample,  Q = quit

Don't run this at the same time as app.py - they would split the camera's frames.
"""
import os
import csv
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ============================ SETTINGS ============================
# Keep these the same as in server/app.py so the training data matches what the app sees
CAMERA_SOURCE = "http://192.168.137.50/capture"   # or 0 for the laptop webcam
ROTATE = None             # e.g. cv2.ROTATE_180 if the camera is mounted upside down
FLIP_HORIZONTAL = False   # egocentric (chest) view: keep the raw orientation
DISPLAY_WIDTH = 640       # preview window width (the ESP image is small, so it's enlarged)
# ==================================================================

TARGET_GESTURES = ["WATER", "YES", "NO", "SICK", "BATHROOM", "PLEASE",
                   "HELP", "HURT", "HOT", "COLD", "MORE", "MEDICINE", "STOP", "ME", "YOU"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "hand_gestures.csv")
LANDMARKER_PATH = os.path.join(BASE_DIR, "server", "hand_landmarker.task")
WINDOW = "Chest-View Data Collector"

# Which landmarks to join with lines when drawing the hand
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]


class FrameSource:
    """Reads frames from the ESP32's /capture URL, or from a local webcam index."""

    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source) if isinstance(source, int) else None

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        try:
            with urllib.request.urlopen(self.source, timeout=2) as resp:
                data = resp.read()
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[camera] could not get a frame from {self.source}: {e}")
            return None

    def release(self):
        if self.cap is not None:
            self.cap.release()


def draw_hand(image, landmarks):
    h, w = image.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, points[a], points[b], (255, 255, 255), 2)
    for p in points:
        cv2.circle(image, p, 4, (0, 0, 255), -1)


def put_label(image, text, color=(0, 255, 0)):
    cv2.putText(image, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(image, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    if not os.path.exists(LANDMARKER_PATH):
        raise FileNotFoundError(f"Missing {LANDMARKER_PATH} - download hand_landmarker.task into the server folder.")

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
        )
    )

    print("Target gestures:", ", ".join(TARGET_GESTURES))
    label = input("Enter gesture label to collect (must match one above): ").strip().upper()
    if label not in TARGET_GESTURES:
        print(f"Warning: '{label}' is not in your primary list, but it will still be saved.")

    source = FrameSource(CAMERA_SOURCE)
    file_exists = os.path.isfile(CSV_PATH)

    start = time.monotonic()
    last_ts = -1
    count = 0

    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # 1 hand * 21 landmarks * 3 coordinates (x, y, z) = 63 features total
            writer.writerow(["label"] + [f"feature_{i}" for i in range(63)])

        print(f"\n--- COLLECTING DATA FOR: {label} ---")
        print(f"Camera: {CAMERA_SOURCE}")
        print("Click the preview window, hold the gesture in front of your chest,")
        print("press SPACE to save a sample, and Q to stop.\n")

        while True:
            frame = source.read()

            if frame is None:
                waiting = np.zeros((DISPLAY_WIDTH * 3 // 4, DISPLAY_WIDTH, 3), np.uint8)
                put_label(waiting, "Waiting for camera...", (0, 200, 255))
                cv2.imshow(WINDOW, waiting)
                if (cv2.waitKey(500) & 0xFF) == ord("q"):
                    break
                continue

            if ROTATE is not None:
                frame = cv2.rotate(frame, ROTATE)
            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode needs strictly increasing timestamps (milliseconds)
            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            # Enlarge for display (drawing happens on the enlarged copy so it stays sharp)
            scale = DISPLAY_WIDTH / frame.shape[1]
            display = cv2.resize(frame, None, fx=scale, fy=scale)

            keypoints = []
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                wrist = landmarks[0]
                # Normalize every landmark relative to the wrist (landmark 0), same as app.py
                for lm in landmarks:
                    keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                draw_hand(display, landmarks)

            hand_text = "hand OK" if keypoints else "no hand"
            put_label(display, f"{label} | Saved: {count} | {hand_text}",
                      (0, 255, 0) if keypoints else (0, 200, 255))
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if len(keypoints) == 63:
                    writer.writerow([label] + keypoints)
                    f.flush()
                    count += 1
                    print(f"Saved sample #{count} for {label}")
                else:
                    print("No hand detected in frame! Keep hand in view.")
            elif key == ord("q"):
                break

            # Stop if the window was closed with the X button
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    source.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Saved {count} samples for {label} to {CSV_PATH}")


if __name__ == "__main__":
    main()
