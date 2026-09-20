"""
Sign Language Translator - laptop server

  ESP32-S3 + ArduCAM  --JPEG photos over Wi-Fi-->  this laptop (MediaPipe + your model)
                      --Socket.IO text-->  any phone/computer browser on the same Wi-Fi

Run from the repo root:   py server/app.py
"""
import os
import time
import pickle
import socket
import warnings
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ============================ SETTINGS ============================
# Where frames come from:
#   ESP32:          "http://<ESP IP>/capture"   (IP is printed in the Arduino Serial Monitor)
#   Laptop webcam:  0                           (handy for testing without the ESP)
CAMERA_SOURCE = CAMERA_SOURCE = "http://192.168.137.50/capture"

ROTATE = None              # e.g. cv2.ROTATE_180 or cv2.ROTATE_90_CLOCKWISE if the camera is mounted sideways
FLIP_HORIZONTAL = False    # True if your training data was recorded mirrored (selfie view)
CONFIDENCE_THRESHOLD = 0.85
REPEAT_COOLDOWN_SEC = 1.5  # holding the same sign re-sends it at most this often
PORT = 5000
# ==================================================================

# Model trained on a pandas DataFrame warns on every prediction otherwise
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------- Sign classifier ----------
model = None
model_path = os.path.join(BASE_DIR, "model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("ML model loaded.")
else:
    print("WARNING: server/model.pkl not found - run train_model.py first.")

# ---------- MediaPipe hand landmarker ----------
landmarker_path = os.path.join(BASE_DIR, "hand_landmarker.task")
if not os.path.exists(landmarker_path):
    raise FileNotFoundError(f"Missing {landmarker_path} - download hand_landmarker.task first.")

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=landmarker_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
    )
)

# ---------- Status message shown on the web page ----------
current_status = "Starting..."


def set_status(msg):
    global current_status
    if msg != current_status:
        current_status = msg
        print(f"[status] {msg}")
        socketio.emit("status", {"msg": msg})


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


def camera_loop():
    source = FrameSource(CAMERA_SOURCE)
    print(f"[camera] reading from {CAMERA_SOURCE}")

    start = time.monotonic()
    last_ts = -1
    last_sign, last_sign_time = None, 0.0
    failures = 0
    frame_count, fps_timer = 0, time.monotonic()

    while True:
        frame = source.read()
        if frame is None:
            failures += 1
            if failures >= 3:
                set_status("Camera not responding - check the ESP32")
            socketio.sleep(1.0)
            continue
        failures = 0

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

        if not result.hand_landmarks:
            set_status("Camera OK - show your hand")
            last_sign = None  # hand left the frame, so the same sign can be sent again
        elif model is None:
            set_status("Hand detected, but model.pkl is missing")
        else:
            landmarks = result.hand_landmarks[0]
            wrist = landmarks[0]
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            probs = model.predict_proba([keypoints])[0]
            best = int(np.argmax(probs))
            confidence = float(probs[best])

            sign = str(model.classes_[best])
            # Show the model's best guess on every frame, even below the threshold
            set_status(f"Hand detected - best guess: {sign} ({confidence:.0%})")

            if confidence > CONFIDENCE_THRESHOLD:
                now = time.monotonic()
                if sign != last_sign or now - last_sign_time > REPEAT_COOLDOWN_SEC:
                    print(f"[sign] {sign} ({confidence:.0%})")
                    socketio.emit("new_sign", {"sign": sign, "confidence": round(confidence * 100, 1)})
                    last_sign, last_sign_time = sign, now

        frame_count += 1
        if time.monotonic() - fps_timer >= 5:
            print(f"[camera] {frame_count / (time.monotonic() - fps_timer):.1f} frames/sec")
            frame_count, fps_timer = 0, time.monotonic()

        socketio.sleep(0.01)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("[web] a browser connected")
    emit("status", {"msg": current_status})


def get_lan_ips():
    """All of this laptop's IPv4 addresses (Wi-Fi, Ethernet, Mobile Hotspot...)."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


if __name__ == "__main__":
    print("\n" + "=" * 62)
    print(f"  On this laptop:  http://localhost:{PORT}")
    print("  On a phone, use the address on the same network as the phone:")
    for ip in get_lan_ips():
        note = "   <- laptop's Mobile Hotspot" if ip.startswith("192.168.137.") else ""
        print(f"     http://{ip}:{PORT}{note}")
    print("=" * 62 + "\n")
    socketio.start_background_task(camera_loop)
    socketio.run(app, host="0.0.0.0", port=PORT, allow_unsafe_werkzeug=True)
