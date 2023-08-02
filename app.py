from flask import Flask, render_template, Response
import mediapipe as mp
import cv2

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.route("/")
def index():
    return render_template('index.html')

def detect_body():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR frame to RGB for Mediapipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe
            results = pose.process(image_rgb)

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Convert the frame back to BGR for displaying in OpenCV
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Encode the frame as JPEG for streaming via Flask
            ret, buffer = cv2.imencode('.jpg', frame_rgb)
            frame_bytes = buffer.tobytes()

            # Use the 'yield' keyword to return the frame in real-time
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(detect_body(), mimetype='multipart/x-mixed-replace; boundary=frame')
