
import math as m
import numpy as np


def absolute_difference(x, y):
    return abs(x - y)


def findAngle(x1, y1, x2, y2, x3, y3):
    vector1_length = m.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    vector2_length = m.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    ad = absolute_difference(vector1_length, vector2_length)
    # print("absolute dif:", ad)

    dot_product = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
    cosine_theta = dot_product / (vector1_length * vector2_length)
    theta_radians = m.acos(cosine_theta)
    theta_degrees = m.degrees(theta_radians)

    if ad <= 90 and ad >= 70:
        theta_degrees = theta_degrees + 65

    if ad <= 40 and ad >= 50:
        theta_degrees = theta_degrees + 5

    if ad >= 150:
        theta_degrees = theta_degrees - 8

    theta_degrees = int(theta_degrees)

    return theta_degrees


def regular_Angle(x1, y1, x2, y2, x3, y3):
    vector1_length = m.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    vector2_length = m.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    ad = absolute_difference(vector1_length, vector2_length)
    # print("absolute dif:", ad)

    dot_product = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
    cosine_theta = dot_product / (vector1_length * vector2_length)
    theta_radians = m.acos(cosine_theta)
    theta_degrees = m.degrees(theta_radians)

    theta_degrees = int(theta_degrees)

    return theta_degrees


def findAngle_v4(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


def waist(results, mpPose):
    # Right Landmarks
    right_shoulder_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
    right_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]
    right_knee_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE]

    # Left Landmarks
    left_shoulder_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
    left_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]
    left_knee_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE]

    # Waist angle calculation
    left_waist_angle = regular_Angle(left_shoulder_landmark.x, left_shoulder_landmark.y,
                                     left_hip_landmark.x, left_hip_landmark.y,
                                     left_knee_landmark.x, left_knee_landmark.y)

    right_waist_angle = regular_Angle(right_shoulder_landmark.x, right_shoulder_landmark.y,
                                      right_hip_landmark.x, right_hip_landmark.y,
                                      right_knee_landmark.x, right_knee_landmark.y)

    if right_waist_angle < left_waist_angle:
        waist_angle = right_waist_angle
    elif left_waist_angle < right_waist_angle:
        waist_angle = left_waist_angle
    else:
        waist_angle = left_waist_angle

    return waist_angle


def elbow_shoulder(results, mpPose):
    right_shoulder_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW]
    right_wrist_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
    right_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]

    left_shoulder_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
    left_elbow_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW]
    left_wrist_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]
    left_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]

    # Calculate the elbow angle
    right_elbow_angle = findAngle(right_shoulder_landmark.x, right_shoulder_landmark.y,
                                  right_elbow_landmark.x, right_elbow_landmark.y,
                                  right_wrist_landmark.x, right_wrist_landmark.y)

    left_elbow_angle = findAngle(left_shoulder_landmark.x, left_shoulder_landmark.y,
                                 left_elbow_landmark.x, left_elbow_landmark.y,
                                 left_wrist_landmark.x, left_wrist_landmark.y)

    right_arm_angle = regular_Angle(right_hip_landmark.x, right_hip_landmark.y,
                                    right_shoulder_landmark.x, right_shoulder_landmark.y,
                                    right_elbow_landmark.x, right_elbow_landmark.y)

    left_arm_angle = regular_Angle(left_hip_landmark.x, left_hip_landmark.y,
                                   left_shoulder_landmark.x, left_shoulder_landmark.y,
                                   left_elbow_landmark.x, left_elbow_landmark.y)

    return right_elbow_angle, left_elbow_angle, right_arm_angle, left_arm_angle


def knee_angle(results, mpPose):
    right_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]
    right_knee_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE]
    right_ankle_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ANKLE]

    left_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]
    left_knee_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE]
    left_ankle_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ANKLE]

    right_knee_angle = regular_Angle(right_hip_landmark.x, right_hip_landmark.y,
                                     right_knee_landmark.x, right_knee_landmark.y,
                                     right_ankle_landmark.x, right_ankle_landmark.y)

    left_knee_angle = regular_Angle(left_hip_landmark.x, left_hip_landmark.y,
                                    left_knee_landmark.x, left_knee_landmark.y,
                                    left_ankle_landmark.x, left_ankle_landmark.y)

    return right_knee_angle, left_knee_angle


def side_angle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


def neck(results, mpPose):
    right_shoulder_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
    left_shoulder_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
    nose_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE]
    left_ear_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_EAR]
    right_ear_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_EAR]
    left_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]
    right_hip_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]

    bsh_x = (right_shoulder_landmark.x + left_shoulder_landmark.x) / 2
    bsh_y = (right_shoulder_landmark.y + left_shoulder_landmark.y) / 2

    body_landmark_x = (right_shoulder_landmark.x + left_shoulder_landmark.x) / 2
    body_landmark_y = ((right_shoulder_landmark.y + left_shoulder_landmark.y) / 2) - 0.15

    a = absolute_difference(left_shoulder_landmark.x, right_shoulder_landmark.x)
    if a > 0.1:
        body_view = "Front"
        degree = regular_Angle(nose_landmark.x, nose_landmark.y, bsh_x, bsh_y, body_landmark_x, body_landmark_y)
        return body_view, degree
    elif a < 0.18:
        body_view = "Side"
        left_degree = regular_Angle(left_ear_landmark.x, left_ear_landmark.y,
                                    left_shoulder_landmark.x, left_shoulder_landmark.y,
                                    left_hip_landmark.x, left_hip_landmark.y)
        right_degree = regular_Angle(right_ear_landmark.x, right_ear_landmark.y,
                                     right_shoulder_landmark.x, right_shoulder_landmark.y,
                                     right_hip_landmark.x, right_hip_landmark.y)

        if right_ear_landmark.x < left_ear_landmark.x:
            degree = right_degree
        elif right_ear_landmark.x > left_ear_landmark.x:
            degree = left_degree
        else:
            degree = right_degree

        return body_view, abs(degree - 180)
    else:
        body_view = "not detected"
        return body_view, None
