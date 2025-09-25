import time
import numpy as np
from collections import deque
from utils import *

class FeedbackAnalyzer:
    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.rep_times = []
        self.rep_start_time = None
        self.angle_histories = {'main': deque(maxlen=30)}  # Dictionary to support multiple angles per exercise
        self.form_scores = []
        self.current_feedback = ""
        self.last_status = True
        self.session_start_time = time.time()
        self.rep_aux_metrics = []  # Extra metrics per rep

        # Thresholds by exercise type
        self.rules = {
            "sit-up": {
                'ideal_rep_time_min': 2.0, 'ideal_rep_time_max': 4.0,
                'min_angle': 45,     # lowest torso angle at bottom
                'max_angle': 115,    # highest angle at top
            },
            "push-up": {
                'ideal_rep_time_min': 1.0, 'ideal_rep_time_max': 3.0,
                'min_angle': 50,     # elbow angle at bottom
                'max_angle': 160,    # elbow/arm angle at top
            },
            "pull-up": {
                'ideal_rep_time_min': 1.5, 'ideal_rep_time_max': 3.5,
                'min_angle': 20,     # arm angle at max contraction
                'max_angle': 140,    # arm angle at full extension
            },
            "squat": {
                'ideal_rep_time_min': 2.0, 'ideal_rep_time_max': 4.0,
                'min_angle': 80,    # knee angle at bottom
                'max_angle': 170,   # standing knee angle
            },
            "vertical jump": {
                'ideal_rep_time_min': 1.0, 'ideal_rep_time_max': 2.5,
                'min_angle': 90,    # knee angle at takeoff
                'max_angle': 170,   # standing position
            },
            "run": {
                'ideal_rep_time_min': 0.5, 'ideal_rep_time_max': 1.5,
                'stride_min': 1.0, "stride_max": 2.2,  # seconds per step (running)
            }
        }
        self.exercise_config = self.rules.get(exercise_type, self.rules["sit-up"])

    def analyze_rep_performance(self, angle, status, counter, aux_metrics=None):
        current_time = time.time()
        if "main" not in self.angle_histories:
            self.angle_histories['main'] = deque(maxlen=30)
        self.angle_histories['main'].append(angle)
        if aux_metrics:
            self.rep_aux_metrics.append(aux_metrics)

        if status != self.last_status:
            if status == False:  # Starting rep
                self.rep_start_time = current_time
            elif status == True and self.rep_start_time:
                rep_duration = current_time - self.rep_start_time
                self.rep_times.append(rep_duration)
                form_score = self._analyze_rep_form()
                self.form_scores.append(form_score)
        self.last_status = status

        self.current_feedback = self._generate_realtime_feedback(angle, status, counter, aux_metrics)
        return self.current_feedback

    def _analyze_rep_form(self):
        if len(self.angle_histories['main']) < 10:
            return 50
        angles = list(self.angle_histories['main'])
        min_angle = min(angles)
        max_angle = max(angles)
        form_score = 100
        # Per-exercise form rules
        if self.exercise_type == "sit-up":
            if min_angle > self.exercise_config['min_angle'] + 10:
                form_score -= 25
            if max_angle < self.exercise_config['max_angle'] - 10:
                form_score -= 20
        elif self.exercise_type == "push-up":
            if min_angle > self.exercise_config['min_angle'] + 10:
                form_score -= 30
            if max_angle < self.exercise_config['max_angle'] - 20:
                form_score -= 25
        elif self.exercise_type == "pull-up":
            if min_angle > self.exercise_config['min_angle'] + 10:
                form_score -= 25
            if max_angle < self.exercise_config['max_angle'] - 20:
                form_score -= 20
        elif self.exercise_type == "squat":
            if min_angle > self.exercise_config['min_angle'] + 10:
                form_score -= 30
            if max_angle < self.exercise_config['max_angle'] - 15:
                form_score -= 20
        elif self.exercise_type == "vertical jump":
            if min_angle > self.exercise_config['min_angle'] + 10:
                form_score -= 30
            # Could also check for soft landing if you track it
        angle_variance = np.var(angles)
        if angle_variance > 500:
            form_score -= 20
        return max(0, form_score)

    def _generate_realtime_feedback(self, angle, status, counter, aux_metrics=None):
        feedback_messages = []
        ex = self.exercise_config
        # Sit-up feedback
        if self.exercise_type == "sit-up":
            if status == False:  # Going down
                if angle > ex['min_angle'] + 10:
                    feedback_messages.append("Go lower with your sit-up!")
                elif angle < ex['min_angle'] - 5:
                    feedback_messages.append("Perfect sit-up depth!")
            else:
                if angle < ex['max_angle'] - 10:
                    feedback_messages.append("Come up higher for full sit-up!")
                elif angle > ex['max_angle']:
                    feedback_messages.append("Great full range of motion!")
        # Push-up feedback
        elif self.exercise_type == "push-up":
            if status == False:
                if angle > ex['min_angle'] + 10:
                    feedback_messages.append("Lower your chest closer to the ground in push-up.")
                elif angle < ex['min_angle'] - 5:
                    feedback_messages.append("Perfect push-up depth!")
            else:
                if angle < ex['max_angle'] - 20:
                    feedback_messages.append("Extend arms fully to finish the push-up.")
                else:
                    feedback_messages.append("Great lockout at the top position!")            
        # Pull-up feedback
        elif self.exercise_type == "pull-up":
            if status == False:
                if angle > ex['min_angle'] + 10:
                    feedback_messages.append("Pull higher, try to get your chin above the bar in pull-up.")
            else:
                if angle < ex['max_angle'] - 20:
                    feedback_messages.append("Lower yourself to full extension in pull-up.")
                else:
                    feedback_messages.append("Strong top and bottom positions!")
        # Squat feedback
        elif self.exercise_type == "squat":
            if status == False:
                if angle > ex['min_angle'] + 10:
                    feedback_messages.append("Go lower for full squat depth.")
                elif angle < ex['min_angle']:
                    feedback_messages.append("Excellent squat depth!")
            else:
                if angle < ex['max_angle'] - 15:
                    feedback_messages.append("Come up fully to finish the squat.")
                else:
                    feedback_messages.append("Good full-range squat rep!")
        # Vertical jump feedback
        elif self.exercise_type == "vertical jump":
            if status == False:
                if angle > ex['min_angle'] + 10:
                    feedback_messages.append("Bend your knees deeper before takeoff for more height.")
                elif angle < ex['min_angle']:
                    feedback_messages.append("Great jump position!")
            else:
                feedback_messages.append("Focus on landing softly after your jump.")
        # Running feedback (use aux_metrics for stride_time)
        elif self.exercise_type == "run":
            if aux_metrics and 'stride_time' in aux_metrics:
                stride_time = aux_metrics['stride_time']
                if stride_time < ex['stride_min']:
                    feedback_messages.append("Slow down your steps for better running rhythm.")
                elif stride_time > ex['stride_max']:
                    feedback_messages.append("Pick up your cadence for efficient running.")
                else:
                    feedback_messages.append("Excellent running cadence!")
        # Rep timing feedback
        if len(self.rep_times) > 0:
            avg_rep_time = np.mean(self.rep_times[-5:])
            if avg_rep_time < ex['ideal_rep_time_min']:
                feedback_messages.append("Try to slow down for better control.")
            elif avg_rep_time > ex['ideal_rep_time_max']:
                feedback_messages.append("Steady your pace for best performance.")
        # Form consistency feedback
        if len(self.form_scores) >= 3:
            recent_scores = self.form_scores[-3:]
            if all(score > 80 for score in recent_scores):
                feedback_messages.append("Excellent form rep after rep.")
            elif all(score < 60 for score in recent_scores):
                feedback_messages.append("Focus on improving rep form quality.")
        if feedback_messages:
            return feedback_messages[0]
        else:
            return "Keep going!"

    def get_performance_stats(self):
        stats = {
            'total_reps': len(self.rep_times),
            'average_rep_time': np.mean(self.rep_times) if self.rep_times else 0,
            'average_form_score': np.mean(self.form_scores) if self.form_scores else 0,
            'session_duration': time.time() - self.session_start_time,
            'current_feedback': self.current_feedback
        }
        return stats

    def generate_session_summary(self):
        if not self.rep_times:
            return "No complete reps detected in this session."
        total_reps = len(self.rep_times)
        avg_rep_time = np.mean(self.rep_times)
        avg_form_score = np.mean(self.form_scores) if self.form_scores else 0
        session_duration = time.time() - self.session_start_time
        summary = [f"=== SESSION SUMMARY ===",
                   f"Total Reps: {total_reps}",
                   f"Session Duration: {session_duration:.1f} seconds",
                   f"Average Rep Time: {avg_rep_time:.1f}s",
                   f"Average Form Score: {avg_form_score:.1f}/100",
                   "\n=== PERFORMANCE ANALYSIS ==="]
        # Timing
        if avg_rep_time < self.exercise_config['ideal_rep_time_min']:
            summary.append("⚡ You're moving too fast - focus on controlled movements")
        elif avg_rep_time > self.exercise_config['ideal_rep_time_max']:
            summary.append("🐌 Try to maintain a steadier, slightly faster pace")
        else:
            summary.append("✅ Great rep timing!")
        # Form
        if avg_form_score >= 80:
            summary.append("✅ Excellent form throughout the session!")
        elif avg_form_score >= 60:
            summary.append("👍 Good form, but room for improvement")
        else:
            summary.append("⚠️  Focus on improving your form quality")
        # Consistency and improvement
        summary.append("\n=== IMPROVEMENT SUGGESTIONS ===")
        if len(self.form_scores) >= 5:
            form_trend = np.polyfit(range(len(self.form_scores)), self.form_scores, 1)[0]
            if form_trend > 0:
                summary.append("📈 Your form improved during the session!")
            elif form_trend < -5:
                summary.append("📉 Take breaks if form is decreasing; rest can help you maintain quality.")
        consistency_score = 100 - np.std(self.form_scores) if len(self.form_scores) > 1 else 100
        if consistency_score < 70:
            summary.append("🎯 Work on maintaining consistent form across all reps.")
        if total_reps < 6:
            summary.append("💡 Try to increase repetitions next time for greater endurance.")
        return "\n".join(summary)
