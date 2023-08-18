from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import mediapipe as mp
import json
import time
import Angle_cal as angle
from datetime import datetime
import threading

app = Flask(__name__)


# ... (rest of your imports and utility functions here)

@app.route("/")
def index():
    return render_template('index.html')


def process_frame(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    # Convert the BGR frame to RGB for Mediapipe processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(imgRGB)

    # Draw pose landmarks on the frame
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )

    right_elbow_angle, left_elbow_angle, right_arm_angle, left_arm_angle = angle.elbow_shoulder(results, mp_pose)

    # (Your original code does not provide details on how you manage these angles, so I am just drawing them)
    if right_elbow_angle < 110:
        warn1 = "keep your right hand straight"
        cv2.putText(frame, warn1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if left_elbow_angle < 110:
        warn2 = "keep your left hand straight"
        cv2.putText(frame, warn2, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Here, the exercise count logic and other components would come in.
    # I'm omitting it for brevity, but you'd integrate them as per your original function.

    cv2.putText(frame, f" Right Elbow angle: {right_elbow_angle}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(frame, f" Left Elbow angle: {left_elbow_angle}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(frame, f" Right Arm angle: {right_arm_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(frame, f" Left Arm angle: {left_arm_angle}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                2)
    # cv2.putText(img, f"Exercise count: {count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Note: The exercise counting logic is omitted for brevity.

    return frame


@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    # Decode the image from base64 format
    image_data = request.form['image'].split(',', 1)[1]
    decoded_image = base64.b64decode(image_data)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)
    frame = cv2.imdecode(image_np, flags=1)

    # Process the frame
    frame = process_frame(frame)

    # Encode the processed frame back to send to the client
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': "data:image/jpeg;base64," + frame_encoded})


if __name__ == '__main__':
    app.run(host='10.5.49.148', port=5000, ssl_context=('cert.pem', 'key.pem'))
