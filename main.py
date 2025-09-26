## import packages
import cv2
import argparse
import mediapipe as mp
import numpy as np

from utils import *
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from feedback_engine import FeedbackAnalyzer
from cheat_detection_system import ComprehensiveCheatDetector
from cheat_messages import EnhancedCheatMessages
# from user_registration import UserRegistration


## setup argparse
ap = argparse.ArgumentParser()
ap.add_argument("-t",
                "--exercise_type",
                type=str,
                help='Type of activity to do',
                required=True)
ap.add_argument("-vs",
                "--video_source",
                type=str,
                help='Path to input video',
                required=False)
args = vars(ap.parse_args())

## instantiate FeedbackAnalyzer (before loop)
analyzer = FeedbackAnalyzer(args["exercise_type"])

## setup mediapipe drawing and pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## setting the video source
if args["video_source"] is not None:
    cap = cv2.VideoCapture(args["video_source"])
else:
    cap = cv2.VideoCapture(0)  # webcam

cap.set(3, 800)  # width
cap.set(4, 480)  # height

user_id = "test_user_123"  # Get from your authentication system
registered_photo = r"C:\Users\nanin\OneDrive\Pictures\Camera Roll\WIN_20250926_11_56_12_Pro.jpg"

# Initialize systems
cheat_detector = ComprehensiveCheatDetector(user_id, registered_photo)
message_handler = EnhancedCheatMessages()

## setup mediapipe pose detector
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    counter = 0  # movement of exercise
    status = True  # state of move

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    args["exercise_type"], counter, status)
                
                            # CHEAT DETECTION INTEGRATION
                detection_results = cheat_detector.process_frame(frame)

                if not detection_results['session_active']:
                    # Session blocked
                    block_message = message_handler.get_blocked_message()
                    cv2.putText(frame, block_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print(f"🚫 {block_message}")
                    break
                
                # Display appropriate messages
                if detection_results['violations'] or detection_results['warnings']:
                    message = message_handler.format_comprehensive_message(detection_results)
                    cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_results['overlay_color'], 2)
                    print(f"⚠️ {message}")


                if detection_results['face_verified']:
                    # -- FEEDBACK SYSTEM INTEGRATION START --
                    # Select relevant rep angle for each exercise
                    if args["exercise_type"] == "sit-up":
                        angle = BodyPartAngle(landmarks).angle_of_the_abdomen()
                    elif args["exercise_type"] == "push-up":
                        angle = BodyPartAngle(landmarks).angle_of_the_left_arm()
                    elif args["exercise_type"] == "pull-up":
                        angle = BodyPartAngle(landmarks).angle_of_the_left_arm()  # or right arm if preferred
                    elif args["exercise_type"] == "squat":
                        angle = BodyPartAngle(landmarks).angle_of_the_left_leg()
                    elif args["exercise_type"] == "vertical jump":
                        angle = BodyPartAngle(landmarks).angle_of_the_left_leg()
                    elif args["exercise_type"] == "run":
                        angle = BodyPartAngle(landmarks).angle_of_the_left_leg()  # Ideally stride info; adjust as needed
                    else:
                        angle = BodyPartAngle(landmarks).angle_of_the_abdomen()

                    feedback = analyzer.analyze_rep_performance(angle, status, counter)
                    print(f"Coach feedback: {feedback}")

                    cv2.putText(frame, f"{feedback}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    # -- FEEDBACK SYSTEM INTEGRATION END --

            score_table(args["exercise_type"], counter, status)
            
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
            )

        except Exception as e:
            print("Error during feedback logic:", e)

        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("counter: " + str(counter))
            break

    # Print session summary AFTER exiting video loop
    print(analyzer.generate_session_summary())

    cap.release()
    cv2.destroyAllWindows()
