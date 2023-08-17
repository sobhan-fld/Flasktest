from flask import Flask, render_template, Response
import mediapipe as mp
import cv2
import json
import time
import Angle_cal as angle
from datetime import datetime
import threading
from flask import request

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)
@app.route("/")
def index():
    return render_template('index.html')

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data = self.data[1:]

    def get_average(self):
        if self.data:
            return sum(self.data) / len(self.data)
        else:
            return 0
def save_data_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def threaded_save_data_to_json(data, filename):
    """Start a new thread to save data."""
    thread = threading.Thread(target=save_data_to_json, args=(data, filename))
    thread.start()
def detect_body():
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    count = 0
    is_count = False

    right_elbow_angle = MovingAverage(window_size=10)
    left_elbow_angle = MovingAverage(window_size=10)
    right_arm_angle = MovingAverage(window_size=10)
    left_arm_angle = MovingAverage(window_size=10)

    data1 = []
    save_interval = 1  # Save Data every one second
    prev_save_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            # Convert the BGR frame to RGB for Mediapipe processing
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe
            results = pose.process(imgRGB)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            right_elbow_angle, left_elbow_angle, right_arm_angle, left_arm_angle = angle.elbow_shoulder(results, mpPose)
            angle_data = {
                "time": datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S'),
                # "time": curr_time,
                "right_elbow_angle": right_elbow_angle,
                "left_elbow_angle": left_elbow_angle,
                "right_arm_angle": right_arm_angle,
                "left_arm_angle": left_arm_angle,
                "exercise_count": count
            }

            if right_elbow_angle < 110:
                warn1 = "keep your right hand straight"
                cv2.putText(img, warn1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if left_elbow_angle < 110:
                warn2 = "keep your left hand straight"
                cv2.putText(img, warn2, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if right_arm_angle > 25 and left_arm_angle > 25 and right_elbow_angle > 110 and left_elbow_angle > 110:
                if not is_count:
                    count += 1
                    is_count = True
            else:
                is_count = False

            cv2.putText(img, f" Right Elbow angle: {right_elbow_angle}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.putText(img, f" Left Elbow angle: {left_elbow_angle}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.putText(img, f" Right Arm angle: {right_arm_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.putText(img, f" Left Arm angle: {left_arm_angle}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        2)
            cv2.putText(img, f"Excersice count: {count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            data1.append(angle_data)
            # Convert the frame back to BGR for displaying in OpenCV
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Encode the frame as JPEG for streaming via Flask
            ret, buffer = cv2.imencode('.jpg', frame_rgb)
            frame_bytes = buffer.tobytes()

            # Use the 'yield' keyword to return the frame in real-time
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if time.time() - prev_save_time >= save_interval:
                threaded_save_data_to_json(data1, "angle_data.json")
                prev_save_time = time.time()
    cap.release()



@app.route('/video_feed', methods=['POST'])
def video_feed():
    def generate_frames():
        while True:
            frame = request.data  # Get frame data from the POST request
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')