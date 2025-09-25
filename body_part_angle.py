import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
from utils import *


class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_left_arm(self):
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        l_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        l_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        return calculate_angle(l_shoulder, l_elbow, l_wrist)

    def angle_of_the_right_arm(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        r_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        r_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        return calculate_angle(r_shoulder, r_elbow, r_wrist)

    def angle_of_the_left_leg(self):
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        l_ankle = detection_body_part(self.landmarks, "LEFT_ANKLE")
        return calculate_angle(l_hip, l_knee, l_ankle)

    def angle_of_the_right_leg(self):
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        r_ankle = detection_body_part(self.landmarks, "RIGHT_ANKLE")
        return calculate_angle(r_hip, r_knee, r_ankle)

    def angle_of_the_neck(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        r_mouth = detection_body_part(self.landmarks, "MOUTH_RIGHT")
        l_mouth = detection_body_part(self.landmarks, "MOUTH_LEFT")
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")

        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]
        mouth_avg = [(r_mouth[0] + l_mouth[0]) / 2,
                     (r_mouth[1] + l_mouth[1]) / 2]
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        return abs(180 - calculate_angle(mouth_avg, shoulder_avg, hip_avg))

    def angle_of_the_abdomen(self):
        # calculate angle of the avg shoulder
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]

        # calculate angle of the avg hip
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        # calculate angle of the avg knee
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        knee_avg = [(r_knee[0] + l_knee[0]) / 2, (r_knee[1] + l_knee[1]) / 2]

        return calculate_angle(shoulder_avg, hip_avg, knee_avg)

    def calculate_angles(self, exercise_type):
        """
        Calculate key angles/distances & detect up to three common form issues.
        """
        # Example: Sit-up form analysis
        results = {}

        # 1. Lower back straightness (use abdomen angle)
        abdomen_angle = self.angle_of_the_abdomen()
        results["abdomen_angle"] = abdomen_angle
        if abdomen_angle < 140:
            results["form_issue"] = "Lower back not straight"

        # 2. Elbows touching knees (full range)
        # You may define elbow–knee horizontal/vertical difference at contraction pose
        l_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        r_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        # Use the smaller elbow–knee distance as proxy
        left_elbow_knee_dist = np.linalg.norm(np.array(l_elbow) - np.array(l_knee))
        right_elbow_knee_dist = np.linalg.norm(np.array(r_elbow) - np.array(r_knee))
        min_dist = min(left_elbow_knee_dist, right_elbow_knee_dist)
        results["elbow_knee_min_distance"] = min_dist
        # Threshold should be tuned (e.g., < 0.1 normalized for 'touching')
        if "form_issue" not in results and min_dist > 0.15:  # tune 0.15 for your scale
            results["form_issue"] = "Elbows not touching knees"

        # 3. Head pulled forward (neck angle analysis)
        neck_angle = self.angle_of_the_neck()
        results["neck_angle"] = neck_angle
        if "form_issue" not in results and neck_angle < 100:
            results["form_issue"] = "Head pulled forward during movement"

        # If no issues, return None
        if "form_issue" not in results:
            results["form_issue"] = None

        return results