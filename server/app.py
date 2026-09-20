from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp

# Safe compatibility import for MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = None
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("ML Model loaded successfully.")
else:
    print("Warning: model.pkl not found! Run train_model.py first.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7
)

@app.route('/')
def index():
    return render_template('index.html')

def generate_predictions():
    # Set to 0 for local laptop camera testing; replace with ESP32 URL later
    STREAM_URL = 0 
    cap = cv2.VideoCapture(STREAM_URL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            socketio.sleep(0.03)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and model:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[0]
            
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            if len(keypoints) == 63:
                probs = model.predict_proba([keypoints])[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]

                # 85% confidence threshold to minimize false detections
                if confidence > 0.85:
                    predicted_sign = model.classes_[best_idx]
                    
                    socketio.emit('new_sign', {
                        'sign': predicted_sign, 
                        'confidence': round(float(confidence) * 100, 1)
                    })

        socketio.sleep(0.05)

@socketio.on('connect')
def handle_connect():
    print("iPhone connected to WebSocket server!")
    socketio.start_background_task(generate_predictions)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)