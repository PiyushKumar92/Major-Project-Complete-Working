"""
Professional Person Detection System
Handles all real-world CCTV scenarios with maximum accuracy and minimal false positives
"""

import cv2
import numpy as np
import face_recognition
import os
import json
from datetime import datetime
from collections import defaultdict
import logging
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class ProfessionalPersonDetector:
    def __init__(self, case_id):
        self.case_id = case_id
        
        # STRICT Professional thresholds for HIGH accuracy
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70  # 70%+ = Auto-approve (VERIFIED)
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.55  # 55-70% = Need manual review
        self.LOW_CONFIDENCE_THRESHOLD = 0.50  # Below 50% = REJECT completely
        
        # CRITICAL: Minimum acceptable confidence
        self.ABSOLUTE_MINIMUM_THRESHOLD = 0.50  # Nothing below 50% should pass
        
        # Frame processing optimization
        self.FRAME_SKIP_INTERVAL = 30  # Process every 30th frame (1 sec at 30fps)
        self.MIN_DETECTION_DURATION = 2.0  # Minimum 2 seconds appearance
        self.MAX_GAP_BETWEEN_DETECTIONS = 5.0  # Max 5 seconds gap
        
        # Face detection parameters
        self.MIN_FACE_SIZE = (30, 30)
        self.MAX_FACE_SIZE = (300, 300)
        
        # Load target person data
        self.target_encodings = self._load_target_encodings()
        self.target_features = self._extract_target_features()
        
        # Detection results storage
        self.detections = []
        self.confirmation_needed = []
        
    def _load_target_encodings(self):
        """Load and process all target person photos"""
        from models import Case, TargetImage
        
        case = Case.query.get(self.case_id)
        if not case:
            return []
        
        encodings = []
        
        for target_image in case.target_images:
            try:
                image_path = os.path.join('static', target_image.image_path)
                if os.path.exists(image_path):
                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Multiple encoding methods for robustness
                    face_encodings = face_recognition.face_encodings(image, num_jitters=10, model='large')
                    
                    if face_encodings:
                        encodings.extend(face_encodings)
                        logger.info(f"Loaded {len(face_encodings)} encodings from {target_image.image_path}")
                    
            except Exception as e:
                logger.error(f"Error loading target image {target_image.image_path}: {e}")
        
        # Remove duplicate encodings
        if len(encodings) > 1:
            encodings = self._remove_duplicate_encodings(encodings)
        
        logger.info(f"Total target encodings: {len(encodings)}")
        return encodings
    
    def _extract_target_features(self):
        """Extract additional features from target photos for better matching"""
        from models import Case
        
        case = Case.query.get(self.case_id)
        features = {
            'age_range': self._estimate_age_range(case.age),
            'clothing_description': case.clothing_description or "",
            'physical_features': case.details or ""
        }
        
        return features
    
    def _estimate_age_range(self, age):
        """Estimate age range for better filtering"""
        if not age:
            return (18, 60)  # Default range
        
        return (max(age - 5, 0), age + 5)
    
    def analyze_footage_professional(self, footage_id):
        """Professional footage analysis with accuracy focus"""
        from models import SurveillanceFootage
        
        footage = SurveillanceFootage.query.get(footage_id)
        if not footage:
            return {"error": "Footage not found"}
        
        video_path = os.path.join('static', footage.video_path)
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        logger.info(f"Starting professional analysis of {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video: {total_frames} frames, {fps} FPS, {duration:.1f} seconds")
        
        # Process video with optimized frame skipping
        frame_count = 0
        processed_frames = 0
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for optimization
            if frame_count % self.FRAME_SKIP_INTERVAL == 0:
                timestamp = frame_count / fps
                
                # Process frame
                frame_detections = self._process_frame_professional(frame, timestamp)
                all_detections.extend(frame_detections)
                processed_frames += 1
                
                # Progress logging
                if processed_frames % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% - {len(all_detections)} detections")
            
            frame_count += 1
        
        cap.release()
        
        # Post-process detections
        final_results = self._post_process_detections(all_detections, fps)
        
        # Save results
        self._save_professional_results(final_results, footage_id)
        
        return {
            "footage_id": footage_id,
            "total_frames_processed": processed_frames,
            "total_detections": len(all_detections),
            "high_confidence_detections": len([d for d in all_detections if d['confidence'] >= self.HIGH_CONFIDENCE_THRESHOLD]),
            "confirmation_needed": len([d for d in all_detections if self.MEDIUM_CONFIDENCE_THRESHOLD <= d['confidence'] < self.HIGH_CONFIDENCE_THRESHOLD]),
            "appearances": final_results['appearances'],
            "analysis_complete": True
        }
    
    def _process_frame_professional(self, frame, timestamp):
        """Process single frame with professional accuracy"""
        detections = []
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in frame
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        
        if not face_locations:
            return detections
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check face size (filter out too small/large faces)
            face_width = right - left
            face_height = bottom - top
            
            if (face_width < self.MIN_FACE_SIZE[0] or face_height < self.MIN_FACE_SIZE[1] or
                face_width > self.MAX_FACE_SIZE[0] or face_height > self.MAX_FACE_SIZE[1]):
                continue
            
            # Match against target encodings with STRICT filtering
            confidence = self._calculate_match_confidence(face_encoding)
            
            # CRITICAL: Reject anything below absolute minimum
            if confidence < self.ABSOLUTE_MINIMUM_THRESHOLD:
                continue  # Skip this detection completely
            
            if confidence >= self.LOW_CONFIDENCE_THRESHOLD:
                # Extract face region for confirmation if needed
                face_region = frame[top:bottom, left:right]
                
                detection = {
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'bbox': (left, top, right - left, bottom - top),
                    'face_region': face_region,
                    'face_size': (face_width, face_height),
                    'method': 'face_recognition_professional'
                }
                
                # Categorize detection
                if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                    detection['status'] = 'confirmed'
                elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                    detection['status'] = 'needs_confirmation'
                    # Save frame for manual verification
                    detection['confirmation_frame'] = self._save_confirmation_frame(face_region, timestamp)
                else:
                    detection['status'] = 'low_confidence'
                
                detections.append(detection)
        
        return detections
    
    def _calculate_match_confidence(self, face_encoding):
        """Calculate confidence with multiple target encodings"""
        if not self.target_encodings:
            return 0.0
        
        best_confidence = 0.0
        
        for target_encoding in self.target_encodings:
            try:
                # Face recognition distance
                distance = face_recognition.face_distance([target_encoding], face_encoding)[0]
                confidence = max(0, 1.0 - distance)
                
                # Additional cosine similarity check
                cosine_sim = 1 - cosine(target_encoding, face_encoding)
                
                # Combined confidence
                combined_confidence = (confidence * 0.7) + (cosine_sim * 0.3)
                
                best_confidence = max(best_confidence, combined_confidence)
                
            except Exception as e:
                logger.error(f"Error calculating confidence: {e}")
                continue
        
        return best_confidence
    
    def _save_confirmation_frame(self, face_region, timestamp):
        """Save frame that needs manual confirmation"""
        try:
            timestamp_str = f"{int(timestamp)}_{int((timestamp % 1) * 1000)}"
            filename = f"confirmation_{self.case_id}_{timestamp_str}.jpg"
            
            confirmation_dir = os.path.join('static', 'confirmations')
            os.makedirs(confirmation_dir, exist_ok=True)
            
            file_path = os.path.join(confirmation_dir, filename)
            cv2.imwrite(file_path, face_region)
            
            return f"confirmations/{filename}"
            
        except Exception as e:
            logger.error(f"Error saving confirmation frame: {e}")
            return None
    
    def _post_process_detections(self, all_detections, fps):
        """Post-process detections to create appearance timeline"""
        if not all_detections:
            return {'appearances': [], 'total_duration': 0}
        
        # Sort by timestamp
        sorted_detections = sorted(all_detections, key=lambda x: x['timestamp'])
        
        # Group into appearances
        appearances = []
        current_appearance = None
        
        for detection in sorted_detections:
            timestamp = detection['timestamp']
            
            if current_appearance is None:
                # Start new appearance
                current_appearance = {
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'detections': [detection],
                    'max_confidence': detection['confidence'],
                    'status': detection['status'],
                    'confirmation_frames': []
                }
                
                if detection.get('confirmation_frame'):
                    current_appearance['confirmation_frames'].append(detection['confirmation_frame'])
            else:
                # Check if this continues current appearance
                time_gap = timestamp - current_appearance['end_time']
                
                if time_gap <= self.MAX_GAP_BETWEEN_DETECTIONS:
                    # Continue current appearance
                    current_appearance['end_time'] = timestamp
                    current_appearance['detections'].append(detection)
                    current_appearance['max_confidence'] = max(
                        current_appearance['max_confidence'], 
                        detection['confidence']
                    )
                    
                    # Update status to highest confidence level
                    if detection['confidence'] >= self.HIGH_CONFIDENCE_THRESHOLD:
                        current_appearance['status'] = 'confirmed'
                    elif (detection['confidence'] >= self.MEDIUM_CONFIDENCE_THRESHOLD and 
                          current_appearance['status'] != 'confirmed'):
                        current_appearance['status'] = 'needs_confirmation'
                    
                    if detection.get('confirmation_frame'):
                        current_appearance['confirmation_frames'].append(detection['confirmation_frame'])
                else:
                    # Finalize current appearance
                    current_appearance['duration'] = (
                        current_appearance['end_time'] - current_appearance['start_time']
                    )
                    
                    if current_appearance['duration'] >= self.MIN_DETECTION_DURATION:
                        appearances.append(current_appearance)
                    
                    # Start new appearance
                    current_appearance = {
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'detections': [detection],
                        'max_confidence': detection['confidence'],
                        'status': detection['status'],
                        'confirmation_frames': []
                    }
                    
                    if detection.get('confirmation_frame'):
                        current_appearance['confirmation_frames'].append(detection['confirmation_frame'])
        
        # Don't forget the last appearance
        if current_appearance:
            current_appearance['duration'] = (
                current_appearance['end_time'] - current_appearance['start_time']
            )
            if current_appearance['duration'] >= self.MIN_DETECTION_DURATION:
                appearances.append(current_appearance)
        
        # Calculate total duration
        total_duration = sum(app['duration'] for app in appearances)
        
        return {
            'appearances': appearances,
            'total_duration': total_duration
        }
    
    def _save_professional_results(self, results, footage_id):
        """Save professional analysis results"""
        from models import PersonDetection, LocationMatch
        from __init__ import db
        
        try:
            # Find or create location match
            location_match = LocationMatch.query.filter_by(
                case_id=self.case_id,
                footage_id=footage_id
            ).first()
            
            if not location_match:
                location_match = LocationMatch(
                    case_id=self.case_id,
                    footage_id=footage_id,
                    match_score=0.8,
                    status='completed'
                )
                db.session.add(location_match)
                db.session.commit()
            
            # Save each appearance as detection
            for appearance in results['appearances']:
                detection = PersonDetection(
                    location_match_id=location_match.id,
                    timestamp=appearance['start_time'],
                    confidence_score=appearance['max_confidence'],
                    analysis_method='final_correct_matching',
                    detection_box=json.dumps({
                        'start_time': appearance['start_time'],
                        'end_time': appearance['end_time'],
                        'duration': appearance['duration']
                    }),
                    verified=(appearance['status'] == 'confirmed'),
                    notes=f"Status: {appearance['status']}, Duration: {appearance['duration']:.1f}s"
                )
                
                db.session.add(detection)
            
            # Update location match
            location_match.person_found = len(results['appearances']) > 0
            location_match.detection_count = len(results['appearances'])
            location_match.confidence_score = max(
                [app['max_confidence'] for app in results['appearances']], 
                default=0.0
            )
            
            db.session.commit()
            
            logger.info(f"Saved {len(results['appearances'])} professional detections")
            
        except Exception as e:
            logger.error(f"Error saving professional results: {e}")
            db.session.rollback()
    
    def _remove_duplicate_encodings(self, encodings):
        """Remove duplicate encodings using clustering"""
        try:
            if len(encodings) <= 1:
                return encodings
            
            # Calculate pairwise distances
            distances = []
            for i, enc1 in enumerate(encodings):
                for j, enc2 in enumerate(encodings[i+1:], i+1):
                    dist = face_recognition.face_distance([enc1], enc2)[0]
                    distances.append([i, j, dist])
            
            # Keep only unique encodings (distance > 0.4)
            unique_encodings = []
            used_indices = set()
            
            for i, encoding in enumerate(encodings):
                if i not in used_indices:
                    unique_encodings.append(encoding)
                    # Mark similar encodings as used
                    for j, other_encoding in enumerate(encodings[i+1:], i+1):
                        if j not in used_indices:
                            dist = face_recognition.face_distance([encoding], other_encoding)[0]
                            if dist < 0.4:  # Similar encoding
                                used_indices.add(j)
            
            return unique_encodings
        except:
            return encodings


# Global function for easy access
def analyze_footage_professional(case_id, footage_id):
    """Professional footage analysis with maximum accuracy"""
    try:
        detector = ProfessionalPersonDetector(case_id)
        return detector.analyze_footage_professional(footage_id)
    except Exception as e:
        logger.error(f"Error in professional analysis: {e}")
        return {"error": str(e)}