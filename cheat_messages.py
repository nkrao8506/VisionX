# cheat_messages.py
import random
from datetime import datetime
from typing import Dict

class EnhancedCheatMessages:
    def __init__(self):
        self.identity_messages = {
            'mismatch': [
                "🚨 Identity verification failed. Only the registered user can take this assessment.",
                "⚠️ Face doesn't match registration photo. Please ensure the correct person is taking the test.",
                "🔍 Unrecognized individual detected. This assessment is linked to a specific user account.",
                "❌ Identity mismatch detected. Contact support if this is an error.",
                "🛡️ Security check failed - face doesn't match registered user profile."
            ],
            'no_photo': [
                "📸 No registration photo found. Please upload a clear photo in your profile settings.",
                "⚠️ Identity verification requires a registered photo. Please complete your profile.",
                "🔍 Cannot verify identity - no reference photo available in your account."
            ]
        }
        
        self.replay_messages = [
            "📺 Screen recording detected! Please use live camera feed only, not recorded videos.",
            "🚨 Video replay attack identified. This assessment requires real-time performance.",
            "⚠️ Digital screen detected in video. Ensure you're not recording another device.",
            "🔍 Suspicious video artifacts found. Use your device's camera directly.",
            "🛡️ Anti-cheat system triggered - potential pre-recorded video detected."
        ]
        
        self.multiple_face_messages = [
            "👥 Multiple people detected. Only the registered user should be visible.",
            "⚠️ Additional persons in frame. Please ensure you're alone during assessment.",
            "🔍 Multiple faces detected. This is a single-user assessment only.",
            "🚨 Unauthorized individuals present. Clear the area before continuing."
        ]
        
        self.no_face_messages = [
            "😕 Face not visible. Please position yourself in front of the camera.",
            "📹 No face detected. Ensure proper lighting and camera positioning.",
            "⚠️ You've moved out of camera view. Stay visible throughout the assessment.",
            "🔍 Face detection lost. Maintain clear visibility for identity verification."
        ]
        
        self.lighting_messages = {
            'too_dark': [
                "🔦 Lighting too dim. Please improve room lighting for better face detection.",
                "💡 Insufficient light detected. Move to a brighter area or add lighting.",
                "⚠️ Poor visibility due to darkness. Increase ambient lighting."
            ],
            'too_bright': [
                "☀️ Too much glare detected. Reduce lighting or move away from bright sources.",
                "⚠️ Overexposed video. Adjust lighting to reduce brightness.",
                "🔍 Excessive brightness interfering with face detection."
            ]
        }
        
        self.violation_warnings = [
            "⚠️ Violation #{count}/3 detected: {violation}. Assessment may be terminated.",
            "🚨 Security alert #{count}/3: {violation}. Please comply immediately.",
            "🛡️ Cheat detection warning #{count}/3: {violation}. Final warning approaching.",
        ]
        
        self.session_blocked_messages = [
            "🚫 Assessment terminated due to multiple security violations.",
            "🔒 Session blocked. Contact your administrator for assistance.",
            "⛔ Too many violations detected. This attempt has been invalidated.",
            "🚨 Assessment ended - security protocols triggered.",
            "🛡️ Session terminated for violating assessment integrity guidelines."
        ]
        
        self.guidance_tips = [
            "💡 Tip: Sit in a well-lit area with your face clearly visible.",
            "📱 Tip: Use your device's main camera, not a secondary screen.",
            "🔍 Tip: Ensure you're alone and no one else is visible in the frame.",
            "💡 Tip: Maintain steady positioning throughout the assessment.",
            "🛡️ Tip: Avoid using virtual backgrounds or filters during assessment."
        ]
    
    def get_identity_message(self, violation_type: str = 'mismatch') -> str:
        return random.choice(self.identity_messages.get(violation_type, self.identity_messages['mismatch']))
    
    def get_replay_message(self) -> str:
        return random.choice(self.replay_messages)
    
    def get_multiple_face_message(self) -> str:
        return random.choice(self.multiple_face_messages)
    
    def get_no_face_message(self) -> str:
        return random.choice(self.no_face_messages)
    
    def get_lighting_message(self, condition: str) -> str:
        return random.choice(self.lighting_messages.get(condition, self.lighting_messages['too_dark']))
    
    def get_violation_warning(self, count: int, violation: str) -> str:
        return random.choice(self.violation_warnings).format(count=count, violation=violation)
    
    def get_blocked_message(self) -> str:
        return random.choice(self.session_blocked_messages)
    
    def get_guidance_tip(self) -> str:
        return random.choice(self.guidance_tips)
    
    def format_comprehensive_message(self, results: Dict) -> str:
        """Format a comprehensive message based on detection results"""
        messages = []
        
        if not results['session_active']:
            return self.get_blocked_message()
        
        if results['violations']:
            for violation in results['violations']:
                if 'Identity mismatch' in violation:
                    messages.append(self.get_identity_message())
                elif 'replay detected' in violation:
                    messages.append(self.get_replay_message())
                elif 'Multiple faces' in violation:
                    messages.append(self.get_multiple_face_message())
                elif 'No face detected' in violation:
                    messages.append(self.get_no_face_message())
        
        if results['warnings']:
            for warning in results['warnings']:
                if 'dark' in warning.lower():
                    messages.append(self.get_lighting_message('too_dark'))
                elif 'bright' in warning.lower():
                    messages.append(self.get_lighting_message('too_bright'))
        
        return messages[0] if messages else results['message']
