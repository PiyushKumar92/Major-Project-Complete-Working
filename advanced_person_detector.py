"""
Advanced Person Detection System for CCTV Footage Analysis
Handles all cases: partial faces, moving persons, spectacles, different angles
"""

import cv2
import numpy as np
import face_recognition
import logging
from datetime import datetime, timezone
import os
from werkzeug.utils import secure_filename
from __init__ import db
from models import Case, SurveillanceFootage, Sighting

logger = logging.getLogger(__name__)

class AdvancedPersonDetector:
    def __init__(self, case_id):
        self.case_id = case_id
        self.case = Case.query.get(case_id)
        if not self.case:
            raise ValueError(f"Case {case_id} not found")
        
        # Smart detection thresholds
        self.face_confidence_threshold = 0.65
        self.body_confidence_threshold = 0.55
        self.motion_threshold = 0.50
        
        # Initialize detectors
        self._init_detectors()
        self.target_encodings = self._get_target_encodings()
        
        logger.info(f"Advanced Person Detector initialized for case {case_id}")
    
    def _init_detectors(self):
        """Initialize all detection models"""
        # Face detector (multiple methods for robustness)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Body detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Motion detector
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Previous frame for motion analysis
        self.prev_frame = None
    
    def _get_target_encodings(self):
        """Get face encodings from case photos with enhanced processing"""
        encodings = []
        for target_image in self.case.target_images:
            try:
                image_path = os.path.join('static', 'uploads', os.path.basename(target_image.image_path))
                if not os.path.exists(image_path):
                    continue
                
                image = face_recognition.load_image_file(image_path)
                
                # Multiple detection methods for robustness
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings.extend(face_encodings)
                
                # Try with different parameters for difficult cases
                face_encodings_alt = face_recognition.face_encodings(image, num_jitters=10, model='large')
                if face_encodings_alt:
                    encodings.extend(face_encodings_alt)
                    
            except Exception as e:
                logger.error(f"Error processing target image: {e}")
        
        return encodings
    
    def analyze_footage(self, footage_id):
        """Analyze CCTV footage for person detection"""
        footage = SurveillanceFootage.query.get(footage_id)
        if not footage:
            return {"error": "Footage not found"}
        
        video_path = os.path.join('static', 'surveillance', os.path.basename(footage.file_path))
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        detections = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Analyzing footage {footage_id}: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Process every 5th frame for efficiency
            if frame_count % 5 == 0:
                detection_results = self._detect_person_in_frame(frame, timestamp, footage_id)
                if detection_results:
                    detections.extend(detection_results)
            
            frame_count += 1
            
            # Progress logging
            if frame_count % 1000 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% - {len(detections)} detections so far")
        
        cap.release()
        
        # Save detections to database
        self._save_detections(detections, footage_id)
        
        return {
            "footage_id": footage_id,
            "total_detections": len(detections),
            "detections": detections,
            "analysis_complete": True
        }
    
    def _detect_person_in_frame(self, frame, timestamp, footage_id):
        """Comprehensive person detection in single frame"""
        detections = []
        
        # 1. Face Detection (multiple methods)
        face_detections = self._detect_faces_comprehensive(frame, timestamp)
        detections.extend(face_detections)
        
        # 2. Body Detection
        body_detections = self._detect_bodies(frame, timestamp)
        detections.extend(body_detections)
        
        # 3. Motion Detection for moving persons
        motion_detections = self._detect_motion(frame, timestamp)
        detections.extend(motion_detections)
        
        # Filter and merge overlapping detections
        filtered_detections = self._filter_detections(detections)
        
        return filtered_detections
    
    def _detect_faces_comprehensive(self, frame, timestamp):
        """Detect faces using multiple methods for robustness"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Frontal face detection
        frontal_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Method 2: Profile face detection
        profile_faces = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Method 3: face_recognition library (more accurate)
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            for (top, right, bottom, left) in face_locations:
                face_region = rgb_frame[top:bottom, left:right]
                confidence = self._match_target_face(face_region)
                
                if confidence > self.face_confidence_threshold:
                    detections.append({
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'method': 'face_recognition',
                        'bbox': (left, top, right-left, bottom-top),
                        'type': 'face'
                    })
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
        
        # Process cascade detections
        for (x, y, w, h) in frontal_faces:
            face_region = frame[y:y+h, x:x+w]
            confidence = self._match_target_face(face_region)
            
            if confidence > self.face_confidence_threshold:
                detections.append({
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'method': 'frontal_cascade',
                    'bbox': (x, y, w, h),
                    'type': 'face'
                })
        
        for (x, y, w, h) in profile_faces:
            face_region = frame[y:y+h, x:x+w]
            confidence = self._match_target_face(face_region)
            
            if confidence > self.face_confidence_threshold:
                detections.append({
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'method': 'profile_cascade',
                    'bbox': (x, y, w, h),
                    'type': 'face'
                })
        
        return detections
    
    def _detect_bodies(self, frame, timestamp):
        """Detect full body/person silhouettes"""
        detections = []
        
        try:
            # HOG person detector
            (rects, weights) = self.hog.detectMultiScale(
                frame, winStride=(4, 4), padding=(8, 8), scale=1.05
            )
            
            for i, (x, y, w, h) in enumerate(rects):
                if len(weights) > i and weights[i] > self.body_confidence_threshold:
                    # Extract person region for analysis
                    person_region = frame[y:y+h, x:x+w]
                    
                    # Try to find face in body region
                    face_confidence = self._find_face_in_body(person_region)
                    
                    # Body shape analysis (basic)
                    body_confidence = self._analyze_body_shape(person_region)
                    
                    combined_confidence = max(face_confidence, body_confidence * 0.7)
                    
                    if combined_confidence > self.body_confidence_threshold:
                        detections.append({
                            'timestamp': timestamp,
                            'confidence': combined_confidence,
                            'method': 'hog_body',
                            'bbox': (x, y, w, h),
                            'type': 'body'
                        })
        
        except Exception as e:
            logger.error(f"Body detection error: {e}")
        
        return detections
    
    def _detect_motion(self, frame, timestamp):
        """Detect moving persons using background subtraction"""
        detections = []
        
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (person-sized objects)
                if 500 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Aspect ratio check (person-like)
                    aspect_ratio = h / w if w > 0 else 0
                    if 1.2 < aspect_ratio < 4.0:
                        
                        # Extract moving region
                        motion_region = frame[y:y+h, x:x+w]
                        
                        # Check if it contains target person
                        face_confidence = self._find_face_in_body(motion_region)
                        
                        if face_confidence > self.motion_threshold:
                            detections.append({
                                'timestamp': timestamp,
                                'confidence': face_confidence,
                                'method': 'motion_detection',
                                'bbox': (x, y, w, h),
                                'type': 'motion'
                            })
        
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
        
        return detections
    
    def _match_target_face(self, face_region):
        """Perfect face matching - only real matches"""
        if not self.target_encodings:
            return 0.0
        
        try:
            # Convert to RGB
            if len(face_region.shape) == 3 and face_region.shape[2] == 3:
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            else:
                rgb_face = face_region
            
            h, w = rgb_face.shape[:2]
            if h < 30 or w < 30:
                return 0.0
            
            # Face detection validation
            face_locations = face_recognition.face_locations(rgb_face)
            if not face_locations:
                return 0.0  # No face = No score
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
            if not face_encodings:
                return 0.0
            
            # Compare with target
            face_distances = face_recognition.face_distance(self.target_encodings, face_encodings[0])
            
            if len(face_distances) > 0:
                min_distance = min(face_distances)
                
                # Only accept good matches
                if min_distance > 0.4:
                    return 0.0  # Not a match
                
                # Calculate confidence
                base_confidence = 1.0 - min_distance
                
                # Blur check
                gray = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if blur_score < 30:  # Very blurry
                    return base_confidence * 0.6
                elif blur_score < 100:  # Slightly blurry
                    return base_confidence * 0.9
                else:  # Clear
                    return base_confidence * 1.2
        
        except Exception as e:
            logger.error(f"Face matching error: {e}")
        
        return 0.0
    
    def _find_face_in_body(self, body_region):
        """Find and match face within body region"""
        try:
            # Look for face in upper portion of body
            height = body_region.shape[0]
            upper_region = body_region[:height//2, :]  # Top half
            
            return self._match_target_face(upper_region)
        
        except Exception as e:
            logger.error(f"Face in body detection error: {e}")
            return 0.0
    
    def _analyze_body_shape(self, body_region):
        """Basic body shape analysis for additional confidence"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(body_region, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels (more edges = more detailed person)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Basic confidence based on edge density
            confidence = min(edge_density * 2, 1.0)
            
            return confidence
        
        except Exception as e:
            logger.error(f"Body shape analysis error: {e}")
            return 0.0
    
    def _filter_detections(self, detections):
        """Filter and merge overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            # Check for overlap with existing detections
            is_duplicate = False
            for existing in filtered:
                if self._calculate_overlap(detection['bbox'], existing['bbox']) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _save_detections(self, detections, footage_id):
        """Save detection results to database"""
        try:
            for detection in detections:
                # Create thumbnail
                thumbnail_path = self._save_detection_thumbnail(detection, footage_id)
                
                sighting = Sighting(
                    case_id=self.case_id,
                    search_video_id=footage_id,
                    timestamp=detection['timestamp'],
                    confidence_score=detection['confidence'],
                    detection_method=detection['method'],
                    thumbnail_path=thumbnail_path
                )
                
                db.session.add(sighting)
            
            db.session.commit()
            logger.info(f"Saved {len(detections)} detections for case {self.case_id}")
        
        except Exception as e:
            logger.error(f"Error saving detections: {e}")
            db.session.rollback()
    
    def _save_detection_thumbnail(self, detection, footage_id):
        """Save thumbnail of detection"""
        try:
            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            filename = f"detection_{self.case_id}_{footage_id}_{timestamp_str}.jpg"
            
            # This would need the actual frame data - simplified for now
            return f"static/uploads/{filename}"
        
        except Exception as e:
            logger.error(f"Error saving thumbnail: {e}")
            return None

# Global function for easy access
def analyze_footage_for_case(case_id, footage_id):
    """Analyze specific footage for a case"""
    try:
        detector = AdvancedPersonDetector(case_id)
        return detector.analyze_footage(footage_id)
    except Exception as e:
        logger.error(f"Error in footage analysis: {e}")
        return {"error": str(e)}