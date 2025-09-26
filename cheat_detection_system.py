# cheat_detection_system.py
import cv2
import numpy as np
import face_recognition
import pickle
import time
from datetime import datetime
import json
import os
from typing import Tuple, Dict, List
import logging

class ComprehensiveCheatDetector:
    def __init__(self, user_id: str, registered_photo_path: str = None):
        self.user_id = user_id
        self.registered_encoding = None
        self.session_log = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Thresholds and counters
        self.face_match_threshold = 0.6  # Lower = stricter, but tolerance increased to pose/lighting variations
        self.replay_confidence_threshold = 0.7
        self.max_violations = 3
        self.max_no_face_frames = 90  # 3 second at 30fps
        
        # Violation tracking
        self.violation_counts = {
            'wrong_person': 0,
            'no_face_detected': 0,
            'multiple_faces': 0,
            'replay_detected': 0,
            'poor_lighting': 0
        }
        
        # Session state
        self.session_active = True
        self.last_face_detection = time.time()
        self.no_face_frame_count = 0
        
        # Load registered face if provided
        if registered_photo_path and os.path.exists(registered_photo_path):
            self.load_registered_face(registered_photo_path)
        
        
    def load_registered_face(self, photo_path: str) -> bool:
        """Load and encode the registered user's photo"""
        try:
            # Load image
            image = face_recognition.load_image_file(photo_path)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                self.logger.error("No face found in registered photo")
                return False
            elif len(encodings) > 1:
                self.logger.warning("Multiple faces found, using the first one")
            
            self.registered_encoding = encodings[0]
            
            # Save encoding for future use
            encoding_file = f"user_encodings/{self.user_id}_encoding.pkl"
            os.makedirs("user_encodings", exist_ok=True)
            with open(encoding_file, 'wb') as f:
                pickle.dump(self.registered_encoding, f)
            
            self.logger.info(f"Successfully loaded registered face for user {self.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading registered face: {str(e)}")
            return False
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect all faces in the current frame"""
        faces = []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade (faster) and face_recognition (more accurate)
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            faces.append({
                'location': (top, right, bottom, left),
                'encoding': encoding,
                'center': ((left + right) // 2, (top + bottom) // 2),
                'size': (right - left) * (bottom - top)
            })
        
        return faces
    
    def verify_identity(self, current_face_encoding: np.ndarray) -> Tuple[bool, float]:
        """Verify if the current face matches the registered user"""
        if self.registered_encoding is None:
            return False, 0.0
        
        # Compare faces
        distance = face_recognition.face_distance([self.registered_encoding], current_face_encoding)[0]
        is_match = distance <= self.face_match_threshold
        confidence = 1 - distance  # Convert distance to confidence score
        
        return is_match, confidence
    
    def detect_replay_attack(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect if the video is a replay/screen recording"""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        replay_indicators = []
        
        # 1. Screen reflection detection
        bright_pixels = np.sum(gray > 240) / gray.size
        replay_indicators.append(bright_pixels > 0.15)  # Too many bright pixels
        
        # 2. Edge sharpness (screens have artificial sharp edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        replay_indicators.append(edge_density > 0.08)  # Too many sharp edges
        
        # 3. Color temperature analysis (screens have blue tint)
        b, g, r = cv2.split(frame)
        blue_dominance = np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
        replay_indicators.append(blue_dominance > 0.4)  # Too much blue
        
        # 4. Rectangular region detection (screen borders)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_rectangles = 0
        for contour in contours:
            if cv2.contourArea(contour) > 10000:  # Large area
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:  # Rectangle
                    large_rectangles += 1
        replay_indicators.append(large_rectangles > 2)
        
        # 5. Moiré pattern detection
        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        high_freq_energy = np.sum(magnitude[magnitude.shape[0]//4:, :]) / np.sum(magnitude)
        replay_indicators.append(high_freq_energy > 0.3)
        
        # Calculate confidence based on indicators
        confidence = sum(replay_indicators) / len(replay_indicators)
        is_replay = confidence > 0.6  # 60% or more indicators triggered
        
        return is_replay, confidence
    
    def analyze_lighting_quality(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Analyze if lighting is adequate for face recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness statistics
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        if mean_brightness < 50:
            return False, "Too dark - please improve lighting"
        elif mean_brightness > 200:
            return False, "Too bright - reduce lighting or move away from light source"
        elif brightness_std < 20:
            return False, "Poor contrast - adjust lighting for better face visibility"
        else:
            return True, "Good lighting conditions"
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Main processing function for each frame"""
        if not self.session_active:
            return self._get_blocked_response()
        
        results = {
            'session_active': True,
            'violations': [],
            'warnings': [],
            'face_verified': False,
            'confidence': 0.0,
            'message': "Monitoring...",
            'overlay_color': (0, 255, 0)  # Green by default
        }
        
        # 1. Check lighting quality
        good_lighting, lighting_msg = self.analyze_lighting_quality(frame)
        if not good_lighting:
            results['warnings'].append(lighting_msg)
            results['overlay_color'] = (0, 255, 255)  # Yellow
        
        # 2. Detect replay attacks
        is_replay, replay_confidence = self.detect_replay_attack(frame)
        if is_replay:
            self.violation_counts['replay_detected'] += 1
            results['violations'].append(f"Video replay detected (confidence: {replay_confidence:.2f})")
            results['overlay_color'] = (0, 0, 255)  # Red
        
        # 3. Detect faces
        faces = self.detect_faces_in_frame(frame)
        
        if len(faces) == 0:
            self.no_face_frame_count += 1
            if self.no_face_frame_count > self.max_no_face_frames:
                self.violation_counts['no_face_detected'] += 1
                results['violations'].append("No face detected for extended period")
                results['message'] = "Please ensure your face is clearly visible"
                results['overlay_color'] = (0, 165, 255)  # Orange
        
        elif len(faces) > 1:
            self.violation_counts['multiple_faces'] += 1
            results['violations'].append("Multiple faces detected")
            results['message'] = "Only one person should be visible"
            results['overlay_color'] = (0, 0, 255)  # Red
        
        else:
            # Single face detected - verify identity
            self.no_face_frame_count = 0
            face = faces[0]
            
            # Draw face rectangle
            top, right, bottom, left = face['location']
            cv2.rectangle(frame, (left, top), (right, bottom), results['overlay_color'], 2)
            
            # Verify identity
            if self.registered_encoding is not None:
                is_match, confidence = self.verify_identity(face['encoding'])
                results['face_verified'] = is_match
                results['confidence'] = confidence
                
                if is_match:
                    results['message'] = f"Identity verified ({confidence:.2f})"
                    self.last_face_detection = time.time()
                else:
                    self.violation_counts['wrong_person'] += 1
                    results['violations'].append(f"Identity mismatch (confidence: {confidence:.2f})")
                    results['message'] = "Unrecognized person detected"
                    results['overlay_color'] = (0, 0, 255)  # Red
        
        # Check if maximum violations exceeded
        total_violations = sum(self.violation_counts.values())
        if total_violations >= self.max_violations:
            self.session_active = False
            results = self._get_blocked_response()
        
        # Log the frame analysis
        self._log_frame_analysis(results)
        
        return results
    
    def _get_blocked_response(self) -> Dict:
        """Return response when session is blocked"""
        return {
            'session_active': False,
            'violations': ["Session terminated due to multiple violations"],
            'warnings': [],
            'face_verified': False,
            'confidence': 0.0,
            'message': "Assessment blocked - Contact administrator",
            'overlay_color': (0, 0, 255)
        }
    
    def _log_frame_analysis(self, results: Dict):
        """Log frame analysis for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
            'face_verified': results['face_verified'],
            'confidence': results['confidence'],
            'violations': results['violations'],
            'total_violations': sum(self.violation_counts.values())
        }
        self.session_log.append(log_entry)
    
    def get_session_report(self) -> Dict:
        """Generate comprehensive session report"""
        return {
            'user_id': self.user_id,
            'session_duration': len(self.session_log),
            'total_violations': sum(self.violation_counts.values()),
            'violation_breakdown': self.violation_counts.copy(),
            'session_valid': self.session_active and sum(self.violation_counts.values()) < self.max_violations,
            'log_entries': self.session_log[-10:]  # Last 10 entries
        }
