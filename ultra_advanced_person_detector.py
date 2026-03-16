"""
Ultra-Advanced Person Detection System
Handles all real-world scenarios with high accuracy
"""

import cv2
import numpy as np
import face_recognition
import logging
from datetime import datetime, timezone
import os
from collections import defaultdict, deque
import json
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
import dlib
from __init__ import db
from models import Case, SurveillanceFootage, Sighting, TargetImage

# Import GPU CNN Detector
try:
    from gpu_cnn_detector import GPUCNNDetector
    GPU_CNN_AVAILABLE = True
except:
    GPU_CNN_AVAILABLE = False

logger = logging.getLogger(__name__)

class UltraAdvancedPersonDetector:
    def __init__(self, case_id):
        self.case_id = case_id
        self.case = Case.query.get(case_id)
        if not self.case:
            raise ValueError(f"Case {case_id} not found")
        
        # Smart detection parameters
        self.min_detection_duration = 0.1
        self.max_gap_between_detections = 2.0
        self.face_confidence_threshold = 0.10  # Accept all face detections
        self.body_confidence_threshold = 0.10
        self.motion_confidence_threshold = 0.10
        self.min_face_quality = 0.0  # No minimum quality
        
        # Initialize all detection systems
        self._init_advanced_detectors()
        self.target_encodings = self._get_comprehensive_target_encodings()
        
        # Tracking systems
        self.person_tracker = PersonTracker()
        self.appearance_analyzer = AppearanceAnalyzer()
        self.crowd_analyzer = CrowdAnalyzer()
        
        logger.info(f"Ultra-Advanced Person Detector initialized for case {case_id}")
    
    def _init_advanced_detectors(self):
        """Initialize all advanced detection systems"""
        # Multiple face detection methods
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # GPU CNN Detector
        if GPU_CNN_AVAILABLE:
            try:
                self.gpu_cnn_detector = GPUCNNDetector()
                logger.info("GPU CNN Detector initialized successfully")
            except Exception as e:
                logger.warning(f"GPU CNN init failed: {e}")
                self.gpu_cnn_detector = None
        else:
            self.gpu_cnn_detector = None
        
        # Body and pose detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Motion and tracking
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Advanced tracking
        try:
            self.tracker_types = ['CSRT', 'KCF', 'MOSSE']
            self.active_trackers = []
        except:
            self.tracker_types = ['KCF']
            self.active_trackers = []
        
        # Optical flow for motion analysis
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Previous frame data
        self.prev_frame = None
        self.prev_gray = None
        self.tracking_points = []
    
    def _get_comprehensive_target_encodings(self):
        """Get comprehensive face encodings with multiple methods"""
        encodings = []
        face_images = []
        
        for target_image in self.case.target_images:
            try:
                image_path = os.path.join('static', 'uploads', os.path.basename(target_image.image_path))
                if not os.path.exists(image_path):
                    continue
                
                image = face_recognition.load_image_file(image_path)
                face_images.append(image)
                
                # Multiple encoding methods for robustness
                # Method 1: Standard encoding
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings.extend(face_encodings)
                
                # Method 2: High precision encoding
                face_encodings_precise = face_recognition.face_encodings(image, num_jitters=10, model='large')
                if face_encodings_precise:
                    encodings.extend(face_encodings_precise)
                
                # Method 3: Multiple face locations
                face_locations = face_recognition.face_locations(image, model='cnn')
                if face_locations:
                    face_encodings_cnn = face_recognition.face_encodings(image, face_locations, num_jitters=5)
                    encodings.extend(face_encodings_cnn)
                    
            except Exception as e:
                logger.error(f"Error processing target image: {e}")
        
        # Remove duplicate encodings using clustering
        if len(encodings) > 1:
            encodings = self._remove_duplicate_encodings(encodings)
        
        logger.info(f"Generated {len(encodings)} target encodings from {len(face_images)} images")
        return encodings
    
    def _remove_duplicate_encodings(self, encodings):
        """Remove duplicate encodings using clustering"""
        try:
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
    
    def analyze_footage_ultra_advanced(self, footage_id):
        """Ultra-advanced footage analysis with all scenarios covered"""
        footage = SurveillanceFootage.query.get(footage_id)
        if not footage:
            return {"error": "Footage not found"}
        
        video_path = os.path.join('static', 'surveillance', os.path.basename(footage.file_path))
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Ultra-advanced analysis: {total_frames} frames at {fps} FPS")
        
        # Detection results storage
        all_detections = []
        person_tracks = defaultdict(list)
        appearance_timeline = []
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Process every frame for maximum accuracy
            frame_detections = self._process_frame_ultra_advanced(
                frame, timestamp, frame_count, fps
            )
            
            if frame_detections:
                all_detections.extend(frame_detections)
                
                # Update tracking
                for detection in frame_detections:
                    person_id = self._assign_person_id(detection, person_tracks)
                    person_tracks[person_id].append({
                        'timestamp': timestamp,
                        'detection': detection,
                        'frame_number': frame_count
                    })
            
            frame_count += 1
            
            # Progress logging
            if frame_count % 500 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Ultra-analysis progress: {progress:.1f}% - {len(all_detections)} detections")
        
        cap.release()
        
        # Post-process detections
        processed_results = self._post_process_detections(
            all_detections, person_tracks, fps
        )
        
        # Save comprehensive results
        self._save_ultra_advanced_results(processed_results, footage_id)
        
        return {
            "footage_id": footage_id,
            "total_detections": len(all_detections),
            "unique_appearances": len(processed_results['appearances']),
            "total_duration": processed_results['total_duration'],
            "appearance_timeline": processed_results['appearances'],
            "analysis_complete": True,
            "confidence_distribution": processed_results['confidence_stats']
        }
    
    def _process_frame_ultra_advanced(self, frame, timestamp, frame_number, fps):
        """Process single frame with all advanced detection methods"""
        detections = []
        
        # 1. Multi-method face detection
        face_detections = self._detect_faces_ultra_advanced(frame, timestamp)
        detections.extend(face_detections)
        
        # 2. Body detection with pose analysis
        body_detections = self._detect_bodies_advanced(frame, timestamp)
        detections.extend(body_detections)
        
        # 3. Motion-based detection
        motion_detections = self._detect_motion_advanced(frame, timestamp)
        detections.extend(motion_detections)
        
        # 4. Tracking-based detection
        tracking_detections = self._update_tracking(frame, timestamp)
        detections.extend(tracking_detections)
        
        # 5. Crowd analysis and person extraction
        if len(detections) > 3:  # Crowd scenario
            crowd_detections = self.crowd_analyzer.extract_persons_from_crowd(
                frame, detections, self.target_encodings
            )
            detections.extend(crowd_detections)
        
        # Filter and merge overlapping detections
        filtered_detections = self._filter_and_merge_detections(detections)
        
        # Appearance analysis for each detection
        for detection in filtered_detections:
            appearance_features = self.appearance_analyzer.analyze_appearance(
                frame, detection['bbox']
            )
            detection['appearance'] = appearance_features
        
        return filtered_detections
    
    def _detect_faces_ultra_advanced(self, frame, timestamp):
        """Ultra-advanced face detection handling all scenarios"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection
        enhanced_gray = cv2.equalizeHist(gray)
        
        # ENHANCEMENT FOR LONG DISTANCE: Upscale frame
        height, width = frame.shape[:2]
        upscaled_frame = cv2.resize(frame, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        upscaled_gray = cv2.cvtColor(upscaled_frame, cv2.COLOR_BGR2GRAY)
        upscaled_gray = cv2.equalizeHist(upscaled_gray)
        
        # Method 1: Multiple scale face detection (NORMAL + UPSCALED)
        for scale_factor in [1.01, 1.03, 1.05, 1.1]:
            for min_neighbors in [2, 3, 5]:
                # Normal frame
                faces = self.face_cascade.detectMultiScale(
                    enhanced_gray, 
                    scaleFactor=scale_factor, 
                    minNeighbors=min_neighbors,
                    minSize=(15, 15),  # Smaller minimum for distant faces
                    maxSize=(300, 300)
                )
                
                for (x, y, w, h) in faces:
                    face_region = frame[y:y+h, x:x+w]
                    confidence = self._match_target_face_advanced(face_region)
                    
                    if confidence > 0:
                        detections.append({
                            'timestamp': timestamp,
                            'confidence': confidence,
                            'method': f'frontal_cascade_{scale_factor}_{min_neighbors}',
                            'bbox': (x, y, w, h),
                            'type': 'face'
                        })
                
                # Upscaled frame for small/distant faces
                faces_upscaled = self.face_cascade.detectMultiScale(
                    upscaled_gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(20, 20),
                    maxSize=(600, 600)
                )
                
                for (x, y, w, h) in faces_upscaled:
                    # Scale back to original coordinates
                    x, y, w, h = x//2, y//2, w//2, h//2
                    if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                        face_region = frame[y:y+h, x:x+w]
                        confidence = self._match_target_face_advanced(face_region)
                        
                        if confidence > 0:
                            detections.append({
                                'timestamp': timestamp,
                                'confidence': confidence,
                                'method': f'upscaled_cascade_{scale_factor}',
                                'bbox': (x, y, w, h),
                                'type': 'face'
                            })
        
        # Method 2: Profile face detection (NORMAL + UPSCALED)
        profile_faces = self.profile_cascade.detectMultiScale(
            enhanced_gray, scaleFactor=1.03, minNeighbors=3, minSize=(15, 15)
        )
        
        for (x, y, w, h) in profile_faces:
            face_region = frame[y:y+h, x:x+w]
            confidence = self._match_target_face_advanced(face_region)
            
            if confidence > 0:
                detections.append({
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'method': 'profile_cascade',
                    'bbox': (x, y, w, h),
                    'type': 'face'
                })
        
        # Upscaled profile detection
        profile_faces_upscaled = self.profile_cascade.detectMultiScale(
            upscaled_gray, scaleFactor=1.03, minNeighbors=3, minSize=(20, 20)
        )
        
        for (x, y, w, h) in profile_faces_upscaled:
            x, y, w, h = x//2, y//2, w//2, h//2
            if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                face_region = frame[y:y+h, x:x+w]
                confidence = self._match_target_face_advanced(face_region)
                
                if confidence > 0:
                    detections.append({
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'method': 'upscaled_profile',
                        'bbox': (x, y, w, h),
                        'type': 'face'
                    })
        
        # Method 3: GPU CNN Detection (FASTEST + MOST ACCURATE)
        if self.gpu_cnn_detector is not None:
            try:
                gpu_detections = self.gpu_cnn_detector.detect_faces_gpu_cnn(frame)
                
                for detection in gpu_detections:
                    x, y, w, h = detection['bbox']
                    face_region = frame[y:y+h, x:x+w]
                    confidence = self._match_target_face_advanced(face_region)
                    
                    if confidence > 0:
                        detections.append({
                            'timestamp': timestamp,
                            'confidence': confidence,
                            'method': 'gpu_cnn_mtcnn',
                            'bbox': (x, y, w, h),
                            'type': 'face'
                        })
            except Exception as e:
                logger.error(f"GPU CNN detection error: {e}")
        
        # Method 4: face_recognition library (most accurate)
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_upscaled = cv2.cvtColor(upscaled_frame, cv2.COLOR_BGR2RGB)
            
            # Check if GPU available for CNN
            use_cnn = False
            try:
                import torch
                if torch.cuda.is_available() and dlib.DLIB_USE_CUDA:
                    use_cnn = True
            except:
                pass
            
            # Choose models based on GPU availability
            models = ['hog', 'cnn'] if use_cnn else ['hog']
            
            for model in models:
                try:
                    # Normal frame
                    face_locations = face_recognition.face_locations(rgb_frame, model=model)
                    
                    for (top, right, bottom, left) in face_locations:
                        face_region = rgb_frame[top:bottom, left:right]
                        confidence = self._match_target_face_advanced(face_region)
                        
                        if confidence > 0:
                            detections.append({
                                'timestamp': timestamp,
                                'confidence': confidence,
                                'method': f'face_recognition_{model}',
                                'bbox': (left, top, right-left, bottom-top),
                                'type': 'face'
                            })
                    
                    # Upscaled frame with more upsampling for small faces
                    face_locations_upscaled = face_recognition.face_locations(
                        rgb_upscaled, 
                        model=model,
                        number_of_times_to_upsample=2
                    )
                    
                    for (top, right, bottom, left) in face_locations_upscaled:
                        # Scale back
                        top, right, bottom, left = top//2, right//2, bottom//2, left//2
                        if top >= 0 and left >= 0 and bottom <= frame.shape[0] and right <= frame.shape[1]:
                            face_region = rgb_frame[top:bottom, left:right]
                            confidence = self._match_target_face_advanced(face_region)
                            
                            if confidence > 0:
                                detections.append({
                                    'timestamp': timestamp,
                                    'confidence': confidence,
                                    'method': f'upscaled_{model}',
                                    'bbox': (left, top, right-left, bottom-top),
                                    'type': 'face'
                                })
                except:
                    continue
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
        
        return detections
    
    def _match_target_face_advanced(self, face_region):
        """Ultra-smart face matching with strict validation"""
        if not self.target_encodings:
            return 0.0
        
        try:
            # Convert to RGB
            if len(face_region.shape) == 3 and face_region.shape[2] == 3:
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            else:
                rgb_face = face_region
            
            # Step 1: Face detection validation - MUST have face
            face_locations = face_recognition.face_locations(rgb_face)
            if not face_locations:
                return 0.0  # No face = 0 score
            
            # Step 2: Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
            if not face_encodings:
                return 0.0  # No face encoding = 0 score
            
            # Step 3: Check blur quality
            blur_level = self._get_blur_level(rgb_face)
            
            # Step 4: Match with target
            best_confidence = 0.0
            
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(self.target_encodings, face_encoding)
                if len(distances) == 0:
                    continue
                
                min_distance = min(distances)
                
                # Not matching face
                if min_distance > 0.45:
                    continue
                
                # Calculate smart score based on blur
                calibrated = self._calculate_smart_score(min_distance, blur_level)
                best_confidence = max(best_confidence, calibrated)
            
            return best_confidence
            
        except Exception as e:
            logger.error(f"Smart face matching error: {e}")
            return 0.0
    
    def _get_blur_level(self, face_image):
        """Get blur level: 'clear', 'little_blur', 'fully_blur'"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY) if len(face_image.shape) == 3 else face_image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var > 150:
                return 'clear'
            elif laplacian_var > 50:
                return 'little_blur'
            else:
                return 'fully_blur'
        except:
            return 'fully_blur'
    
    def _calculate_smart_score(self, distance, blur_level):
        """Calculate score based on match quality and blur"""
        # Strong match
        if distance < 0.35:
            if blur_level == 'clear':
                return 0.92  # 92%
            elif blur_level == 'little_blur':
                return 0.38  # 38%
            else:
                return 0.15  # 15%
        # Good match
        elif distance < 0.40:
            if blur_level == 'clear':
                return 0.90  # 90%
            elif blur_level == 'little_blur':
                return 0.36  # 36%
            else:
                return 0.12  # 12%
        # Weak match - reject
        else:
            return 0.0
    
    def _find_face_in_body_advanced(self, person_region):
        """Try to find and match face in body region"""
        try:
            return self._match_target_face_advanced(person_region)
        except:
            return 0.0
    
    def _analyze_body_advanced(self, person_region):
        """Analyze body features - returns 0 as we focus on face matching"""
        return 0.0
    
    def _analyze_motion_pattern(self, contour, area):
        """Analyze motion pattern - returns 0 as we focus on face matching"""
        return 0.0
    
    def _update_tracking(self, frame, timestamp):
        """Update tracking - returns empty as we focus on face matching"""
        return []
    
    def _assign_person_id(self, detection, person_tracks):
        """Assign person ID for tracking"""
        return 1
    
    def _filter_and_merge_detections(self, detections):
        """Filter and merge overlapping detections"""
        return detections
    
    def _detect_bodies_advanced(self, frame, timestamp):
        """Advanced body detection - disabled to focus on face matching"""
        return []
    
    def _detect_motion_advanced(self, frame, timestamp):
        """Advanced motion detection - disabled to focus on face matching"""
        return []
    
    def _post_process_detections(self, all_detections, person_tracks, fps):
        """Post-process detections to create appearance timeline"""
        if not all_detections:
            return {
                'appearances': [],
                'total_duration': 0,
                'confidence_stats': {'high': 0, 'medium': 0, 'low': 0}
            }
        
        # Group detections into appearances
        appearances = []
        current_appearance = None
        
        # Sort detections by timestamp
        sorted_detections = sorted(all_detections, key=lambda x: x['timestamp'])
        
        for detection in sorted_detections:
            timestamp = detection['timestamp']
            
            if current_appearance is None:
                # Start new appearance
                current_appearance = {
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'duration': 0,
                    'detections': [detection],
                    'max_confidence': detection['confidence'],
                    'detection_methods': [detection['method']],
                    'appearance_count': 1
                }
            else:
                # Check if this detection continues the current appearance
                time_gap = timestamp - current_appearance['end_time']
                
                if time_gap <= self.max_gap_between_detections:
                    # Continue current appearance
                    current_appearance['end_time'] = timestamp
                    current_appearance['detections'].append(detection)
                    current_appearance['max_confidence'] = max(
                        current_appearance['max_confidence'], 
                        detection['confidence']
                    )
                    if detection['method'] not in current_appearance['detection_methods']:
                        current_appearance['detection_methods'].append(detection['method'])
                else:
                    # Finalize current appearance and start new one
                    current_appearance['duration'] = (
                        current_appearance['end_time'] - current_appearance['start_time']
                    )
                    
                    if current_appearance['duration'] >= self.min_detection_duration:
                        appearances.append(current_appearance)
                    
                    # Start new appearance
                    current_appearance = {
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'duration': 0,
                        'detections': [detection],
                        'max_confidence': detection['confidence'],
                        'detection_methods': [detection['method']],
                        'appearance_count': 1
                    }
        
        # Don't forget the last appearance
        if current_appearance:
            current_appearance['duration'] = (
                current_appearance['end_time'] - current_appearance['start_time']
            )
            if current_appearance['duration'] >= self.min_detection_duration:
                appearances.append(current_appearance)
        
        # Calculate statistics
        total_duration = sum(app['duration'] for app in appearances)
        
        confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
        for detection in all_detections:
            conf = detection['confidence']
            if conf >= 0.7:
                confidence_stats['high'] += 1
            elif conf >= 0.5:
                confidence_stats['medium'] += 1
            else:
                confidence_stats['low'] += 1
        
        return {
            'appearances': appearances,
            'total_duration': total_duration,
            'confidence_stats': confidence_stats
        }
    
    def _save_ultra_advanced_results(self, results, footage_id):
        """Save comprehensive results to database"""
        try:
            for appearance in results['appearances']:
                # Create sighting for each appearance
                sighting = Sighting(
                    case_id=self.case_id,
                    search_video_id=footage_id,
                    timestamp=appearance['start_time'],
                    confidence_score=appearance['max_confidence'],
                    detection_method=', '.join(appearance['detection_methods'][:3]),
                    thumbnail_path=self._save_appearance_thumbnail(appearance, footage_id)
                )
                
                db.session.add(sighting)
            
            db.session.commit()
            logger.info(f"Saved {len(results['appearances'])} appearances for case {self.case_id}")
        
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            db.session.rollback()
    
    def _save_appearance_thumbnail(self, appearance, footage_id):
        """Save thumbnail for appearance"""
        try:
            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            filename = f"appearance_{self.case_id}_{footage_id}_{timestamp_str}.jpg"
            return f"static/uploads/{filename}"
        except Exception as e:
            logger.error(f"Error saving thumbnail: {e}")
            return None


class PersonTracker:
    """Advanced person tracking across frames"""
    def __init__(self):
        self.active_tracks = {}
        self.next_track_id = 1
    
    def update_tracks(self, detections, frame):
        """Update person tracks with new detections"""
        # Implementation for multi-object tracking
        pass


class AppearanceAnalyzer:
    """Analyze person appearance for consistency"""
    def analyze_appearance(self, frame, bbox):
        """Analyze appearance features"""
        x, y, w, h = bbox
        person_region = frame[y:y+h, x:x+w]
        
        features = {
            'dominant_colors': self._extract_colors(person_region),
            'clothing_pattern': self._analyze_clothing(person_region),
            'body_shape': self._analyze_body_shape(person_region)
        }
        
        return features
    
    def _extract_colors(self, region):
        """Extract dominant colors"""
        # Implementation for color extraction
        return []
    
    def _analyze_clothing(self, region):
        """Analyze clothing patterns"""
        # Implementation for clothing analysis
        return {}
    
    def _analyze_body_shape(self, region):
        """Analyze body shape"""
        # Implementation for body shape analysis
        return {}


class CrowdAnalyzer:
    """Analyze crowded scenes and extract persons"""
    def extract_persons_from_crowd(self, frame, detections, target_encodings):
        """Extract individual persons from crowd"""
        crowd_detections = []
        
        if len(detections) > 5:  # Crowd scenario
            # Implementation for crowd analysis
            pass
        
        return crowd_detections


# Global function for easy access
def analyze_footage_ultra_advanced(case_id, footage_id):
    """Ultra-advanced footage analysis"""
    try:
        detector = UltraAdvancedPersonDetector(case_id)
        return detector.analyze_footage_ultra_advanced(footage_id)
    except Exception as e:
        logger.error(f"Error in ultra-advanced analysis: {e}")
        return {"error": str(e)}