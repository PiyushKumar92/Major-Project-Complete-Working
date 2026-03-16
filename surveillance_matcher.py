import cv2
import face_recognition
import numpy as np
import json
import os
from datetime import datetime

class SurveillanceMatcher:
    def __init__(self):
        self.confidence_threshold = 0.6
        
    def load_case_profile(self, case_id):
        """Load the face profile for a case"""
        try:
            profile_path = os.path.join('app', 'static', 'analysis', str(case_id), 'face_profile.json')
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading case profile {case_id}: {str(e)}")
            return None
    
    def analyze_surveillance_footage(self, case_id, footage_path):
        """Analyze surveillance footage against case profile"""
        case_profile = self.load_case_profile(case_id)
        if not case_profile or not case_profile.get('primary_encoding'):
            return {'error': 'No face profile found for case'}
        
        # Load known face encoding
        known_encoding = np.array(case_profile['primary_encoding'])
        confidence_threshold = case_profile.get('confidence_threshold', 0.6)
        
        # Process video
        cap = cv2.VideoCapture(footage_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections = []
        frame_interval = max(1, int(fps))  # Check every second
        
        for frame_num in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for i, face_encoding in enumerate(face_encodings):
                # Compare with known face
                distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                confidence = 1 - distance  # Convert distance to confidence
                
                if confidence >= confidence_threshold:
                    timestamp = frame_num / fps
                    
                    detection = {
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'confidence': float(confidence),
                        'face_location': face_locations[i],
                        'detection_time': datetime.now().isoformat()
                    }
                    
                    detections.append(detection)
                    
                    # Save detection frame
                    self._save_detection_frame(case_id, frame, detection, frame_num)
        
        cap.release()
        
        return {
            'case_id': case_id,
            'footage_analyzed': footage_path,
            'total_detections': len(detections),
            'detections': detections,
            'analysis_completed': datetime.now().isoformat()
        }
    
    def _save_detection_frame(self, case_id, frame, detection, frame_num):
        """Save frame where detection occurred"""
        try:
            detection_dir = os.path.join('app', 'static', 'detections', str(case_id))
            os.makedirs(detection_dir, exist_ok=True)
            
            frame_filename = f"detection_{frame_num}_{detection['confidence']:.2f}.jpg"
            frame_path = os.path.join(detection_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            detection['frame_path'] = f"detections/{case_id}/{frame_filename}"
            
        except Exception as e:
            print(f"Error saving detection frame: {str(e)}")

def match_case_against_footage(case_id, footage_path):
    """Main function to match a case against surveillance footage"""
    matcher = SurveillanceMatcher()
    return matcher.analyze_surveillance_footage(case_id, footage_path)