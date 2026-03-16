"""
Intelligent Footage Analysis System
Advanced computer vision pipeline for comprehensive person tracking and analysis
"""

import cv2
import numpy as np
import face_recognition
from datetime import datetime, timedelta
import os
import json
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import threading
import queue

class IntelligentFootageAnalyzer:
    """
    Advanced computer vision pipeline for intelligent footage analysis
    """
    
    def __init__(self):
        self.tracking_threshold = 0.3
        self.appearance_change_threshold = 0.4
        self.behavioral_analysis_window = 30  # seconds
        self.crowd_density_threshold = 5  # persons per frame
        
        # Initialize trackers and analyzers
        self.multi_tracker = MultiPersonTracker()
        self.temporal_analyzer = TemporalAnalyzer()
        self.appearance_analyzer = AppearanceChangeDetector()
        self.behavioral_analyzer = BehaviorAnalyzer()
        self.crowd_analyzer = CrowdAnalyzer()
        
    def analyze_footage_comprehensive(self, video_path, case_reference_data):
        """
        Comprehensive footage analysis with all advanced features
        """
        results = {
            'multi_person_tracking': {},
            'temporal_patterns': {},
            'appearance_changes': [],
            'behavioral_analysis': {},
            'crowd_analysis': {},
            'person_extractions': [],
            'summary': {}
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Cannot open video file'}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Initialize analysis components
            self.multi_tracker.initialize(case_reference_data)
            self.temporal_analyzer.initialize(duration)
            self.appearance_analyzer.initialize(case_reference_data)
            self.behavioral_analyzer.initialize()
            self.crowd_analyzer.initialize()
            
            frame_number = 0
            processing_interval = max(1, int(fps / 2))  # Process 2 frames per second
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % processing_interval == 0:
                    timestamp = frame_number / fps
                    
                    # 1. Multi-person tracking across camera networks
                    tracking_results = self.multi_tracker.track_frame(frame, timestamp)
                    results['multi_person_tracking'][timestamp] = tracking_results
                    
                    # 2. Temporal analysis for movement patterns
                    temporal_data = self.temporal_analyzer.analyze_movement(tracking_results, timestamp)
                    results['temporal_patterns'][timestamp] = temporal_data
                    
                    # 3. Clothing/appearance change detection
                    appearance_changes = self.appearance_analyzer.detect_changes(frame, tracking_results, timestamp)
                    if appearance_changes:
                        results['appearance_changes'].extend(appearance_changes)
                    
                    # 4. Behavioral analysis
                    behavioral_data = self.behavioral_analyzer.analyze_behavior(tracking_results, timestamp)
                    results['behavioral_analysis'][timestamp] = behavioral_data
                    
                    # 5. Crowd analysis and person extraction
                    crowd_data = self.crowd_analyzer.analyze_crowd(frame, tracking_results, timestamp)
                    results['crowd_analysis'][timestamp] = crowd_data
                    
                    # Extract persons of interest
                    extractions = self._extract_persons_of_interest(frame, tracking_results, timestamp)
                    if extractions:
                        results['person_extractions'].extend(extractions)
                
                frame_number += 1
            
            cap.release()
            
            # Generate comprehensive summary
            results['summary'] = self._generate_analysis_summary(results, duration)
            
        except Exception as e:
            results['error'] = f"Analysis failed: {str(e)}"
        
        return results
    
    def _extract_persons_of_interest(self, frame, tracking_results, timestamp):
        """Extract high-quality images of persons of interest"""
        extractions = []
        
        for person_id, person_data in tracking_results.items():
            if person_data.get('confidence', 0) > 0.6:  # High confidence threshold
                bbox = person_data.get('bbox')
                if bbox:
                    x, y, w, h = bbox
                    person_crop = frame[y:y+h, x:x+w]
                    
                    # Quality assessment
                    quality_score = self._assess_image_quality(person_crop)
                    
                    if quality_score > 0.7:  # High quality threshold
                        extraction = {
                            'person_id': person_id,
                            'timestamp': timestamp,
                            'bbox': bbox,
                            'quality_score': quality_score,
                            'confidence': person_data.get('confidence'),
                            'image_data': person_crop,
                            'features': person_data.get('features', {})
                        }
                        extractions.append(extraction)
        
        return extractions
    
    def _assess_image_quality(self, image):
        """Assess quality of extracted person image"""
        if image is None or image.size == 0:
            return 0.0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 1000, 1.0)
        
        # 2. Brightness
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 127) / 127
        
        # 3. Contrast
        contrast = gray.std()
        contrast_score = min(contrast / 50, 1.0)
        
        # 4. Size adequacy
        height, width = image.shape[:2]
        size_score = min((width * height) / (100 * 150), 1.0)  # Optimal at 100x150
        
        return (sharpness_score * 0.4 + brightness_score * 0.2 + 
                contrast_score * 0.2 + size_score * 0.2)
    
    def _generate_analysis_summary(self, results, duration):
        """Generate comprehensive analysis summary"""
        summary = {
            'total_duration': duration,
            'persons_tracked': len(set().union(*[list(frame_data.keys()) 
                                               for frame_data in results['multi_person_tracking'].values()])),
            'appearance_changes_detected': len(results['appearance_changes']),
            'behavioral_anomalies': sum(1 for frame_data in results['behavioral_analysis'].values() 
                                      if frame_data.get('anomalies')),
            'crowd_scenes_detected': sum(1 for frame_data in results['crowd_analysis'].values() 
                                       if frame_data.get('crowd_density', 0) > self.crowd_density_threshold),
            'high_quality_extractions': len([e for e in results['person_extractions'] 
                                           if e.get('quality_score', 0) > 0.8]),
            'analysis_confidence': self._calculate_overall_confidence(results)
        }
        
        return summary
    
    def _calculate_overall_confidence(self, results):
        """Calculate overall analysis confidence"""
        confidences = []
        
        # Collect confidence scores from all analysis components
        for frame_data in results['multi_person_tracking'].values():
            for person_data in frame_data.values():
                if 'confidence' in person_data:
                    confidences.append(person_data['confidence'])
        
        return np.mean(confidences) if confidences else 0.0


class MultiPersonTracker:
    """Advanced multi-person tracking across camera networks"""
    
    def __init__(self):
        self.active_tracks = {}
        self.track_id_counter = 0
        self.max_track_age = 30  # frames
        self.similarity_threshold = 0.7
        
    def initialize(self, reference_data):
        """Initialize tracker with reference person data"""
        self.reference_encodings = reference_data.get('face_encodings', [])
        self.reference_features = reference_data.get('visual_features', {})
        
    def track_frame(self, frame, timestamp):
        """Track all persons in current frame"""
        # Detect persons in frame
        person_detections = self._detect_persons(frame)
        
        # Extract features for each detection
        detection_features = []
        for detection in person_detections:
            features = self._extract_person_features(frame, detection)
            detection_features.append(features)
        
        # Associate detections with existing tracks
        tracking_results = self._associate_detections(detection_features, timestamp)
        
        # Update track states
        self._update_tracks(timestamp)
        
        return tracking_results
    
    def _detect_persons(self, frame):
        """Detect all persons in frame using multiple methods"""
        detections = []
        
        # Method 1: HOG person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        try:
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), 
                                                   padding=(8, 8), scale=1.05)
            
            for i, (x, y, w, h) in enumerate(rects):
                if weights[i] > 0.5:  # Confidence threshold
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': float(weights[i]),
                        'method': 'hog'
                    })
        except Exception as e:
            print(f"HOG detection error: {e}")
        
        # Method 2: Face detection as person indicator
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for (top, right, bottom, left) in face_locations:
                # Expand face bbox to approximate person bbox
                face_height = bottom - top
                person_height = int(face_height * 6)  # Approximate person height
                person_width = int(face_height * 2)   # Approximate person width
                
                # Center the person bbox around the face
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                x = max(0, center_x - person_width // 2)
                y = max(0, center_y - person_height // 3)  # Face is in upper third
                w = min(frame.shape[1] - x, person_width)
                h = min(frame.shape[0] - y, person_height)
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,  # High confidence for face-based detection
                    'method': 'face_based',
                    'face_location': (top, right, bottom, left)
                })
        except Exception as e:
            print(f"Face-based detection error: {e}")
        
        return detections
    
    def _extract_person_features(self, frame, detection):
        """Extract comprehensive features for person detection"""
        bbox = detection['bbox']
        x, y, w, h = bbox
        person_roi = frame[y:y+h, x:x+w]
        
        features = {
            'bbox': bbox,
            'confidence': detection['confidence'],
            'method': detection['method'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Face encoding if available
        if 'face_location' in detection:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, [detection['face_location']])
                if face_encodings:
                    features['face_encoding'] = face_encodings[0].tolist()
            except Exception as e:
                print(f"Face encoding error: {e}")
        
        # Color histogram for clothing
        try:
            hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            features['color_histogram'] = hist.flatten().tolist()
        except Exception as e:
            print(f"Color histogram error: {e}")
        
        # Size and position features
        features['size'] = (w, h)
        features['center'] = (x + w//2, y + h//2)
        features['aspect_ratio'] = w / h if h > 0 else 0
        
        return features
    
    def _associate_detections(self, detection_features, timestamp):
        """Associate current detections with existing tracks"""
        tracking_results = {}
        
        # Calculate similarity matrix between detections and existing tracks
        similarities = self._calculate_similarities(detection_features)
        
        # Assign detections to tracks using Hungarian algorithm (simplified)
        assignments = self._assign_detections(similarities)
        
        # Update existing tracks and create new ones
        for det_idx, track_id in assignments.items():
            features = detection_features[det_idx]
            
            if track_id is None:  # New track
                track_id = self._create_new_track(features, timestamp)
            else:  # Update existing track
                self._update_track(track_id, features, timestamp)
            
            # Check if this track matches reference person
            match_confidence = self._match_reference_person(features)
            
            tracking_results[track_id] = {
                'bbox': features['bbox'],
                'confidence': features['confidence'],
                'match_confidence': match_confidence,
                'features': features,
                'is_target': match_confidence > 0.6
            }
        
        return tracking_results
    
    def _calculate_similarities(self, detection_features):
        """Calculate similarity matrix between detections and tracks"""
        similarities = {}
        
        for det_idx, features in enumerate(detection_features):
            similarities[det_idx] = {}
            
            for track_id, track_data in self.active_tracks.items():
                similarity = self._calculate_feature_similarity(features, track_data['features'])
                similarities[det_idx][track_id] = similarity
        
        return similarities
    
    def _calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        similarity_score = 0.0
        weight_sum = 0.0
        
        # Face encoding similarity (highest weight)
        if 'face_encoding' in features1 and 'face_encoding' in features2:
            try:
                enc1 = np.array(features1['face_encoding'])
                enc2 = np.array(features2['face_encoding'])
                face_distance = np.linalg.norm(enc1 - enc2)
                face_similarity = max(0, 1 - face_distance)
                similarity_score += face_similarity * 0.5
                weight_sum += 0.5
            except:
                pass
        
        # Color histogram similarity
        if 'color_histogram' in features1 and 'color_histogram' in features2:
            try:
                hist1 = np.array(features1['color_histogram'])
                hist2 = np.array(features2['color_histogram'])
                hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                similarity_score += max(0, hist_similarity) * 0.3
                weight_sum += 0.3
            except:
                pass
        
        # Size similarity
        if 'size' in features1 and 'size' in features2:
            size1 = features1['size']
            size2 = features2['size']
            size_diff = abs(size1[0] - size2[0]) + abs(size1[1] - size2[1])
            max_size = max(size1[0] + size1[1], size2[0] + size2[1])
            size_similarity = 1 - (size_diff / max_size) if max_size > 0 else 0
            similarity_score += size_similarity * 0.2
            weight_sum += 0.2
        
        return similarity_score / weight_sum if weight_sum > 0 else 0.0
    
    def _assign_detections(self, similarities):
        """Assign detections to tracks (simplified Hungarian algorithm)"""
        assignments = {}
        used_tracks = set()
        
        # Sort detections by best similarity
        for det_idx in similarities:
            best_track = None
            best_similarity = 0
            
            for track_id, similarity in similarities[det_idx].items():
                if track_id not in used_tracks and similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_track = track_id
            
            assignments[det_idx] = best_track
            if best_track is not None:
                used_tracks.add(best_track)
        
        return assignments
    
    def _create_new_track(self, features, timestamp):
        """Create new person track"""
        track_id = f"person_{self.track_id_counter}"
        self.track_id_counter += 1
        
        self.active_tracks[track_id] = {
            'id': track_id,
            'features': features,
            'created_at': timestamp,
            'last_seen': timestamp,
            'age': 0,
            'trajectory': [features['center']]
        }
        
        return track_id
    
    def _update_track(self, track_id, features, timestamp):
        """Update existing track with new detection"""
        if track_id in self.active_tracks:
            track = self.active_tracks[track_id]
            track['features'] = features
            track['last_seen'] = timestamp
            track['age'] = 0
            track['trajectory'].append(features['center'])
            
            # Limit trajectory length
            if len(track['trajectory']) > 100:
                track['trajectory'] = track['trajectory'][-100:]
    
    def _update_tracks(self, timestamp):
        """Update track ages and remove old tracks"""
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            track['age'] += 1
            if track['age'] > self.max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
    
    def _match_reference_person(self, features):
        """Check if detection matches reference person"""
        if not self.reference_encodings:
            return 0.0
        
        max_confidence = 0.0
        
        # Face matching
        if 'face_encoding' in features:
            try:
                detection_encoding = np.array(features['face_encoding'])
                for ref_encoding in self.reference_encodings:
                    distance = np.linalg.norm(detection_encoding - np.array(ref_encoding))
                    confidence = max(0, 1 - distance)
                    max_confidence = max(max_confidence, confidence)
            except:
                pass
        
        return max_confidence


class TemporalAnalyzer:
    """Temporal analysis for movement patterns"""
    
    def __init__(self):
        self.movement_history = defaultdict(list)
        self.pattern_window = 30  # seconds
        
    def initialize(self, video_duration):
        """Initialize temporal analyzer"""
        self.video_duration = video_duration
        self.movement_history.clear()
    
    def analyze_movement(self, tracking_results, timestamp):
        """Analyze movement patterns for all tracked persons"""
        temporal_data = {
            'movement_vectors': {},
            'speed_analysis': {},
            'direction_changes': {},
            'loitering_detection': {},
            'trajectory_patterns': {}
        }
        
        for person_id, person_data in tracking_results.items():
            # Update movement history
            center = person_data['features']['center']
            self.movement_history[person_id].append({
                'timestamp': timestamp,
                'position': center,
                'bbox': person_data['bbox']
            })
            
            # Analyze movement for this person
            person_analysis = self._analyze_person_movement(person_id, timestamp)
            
            if person_analysis:
                temporal_data['movement_vectors'][person_id] = person_analysis.get('movement_vector')
                temporal_data['speed_analysis'][person_id] = person_analysis.get('speed')
                temporal_data['direction_changes'][person_id] = person_analysis.get('direction_changes')
                temporal_data['loitering_detection'][person_id] = person_analysis.get('loitering')
                temporal_data['trajectory_patterns'][person_id] = person_analysis.get('pattern')
        
        return temporal_data
    
    def _analyze_person_movement(self, person_id, current_timestamp):
        """Analyze movement patterns for a specific person"""
        history = self.movement_history[person_id]
        
        if len(history) < 2:
            return None
        
        # Filter recent history within analysis window
        recent_history = [
            h for h in history 
            if current_timestamp - h['timestamp'] <= self.pattern_window
        ]
        
        if len(recent_history) < 2:
            return None
        
        analysis = {}
        
        # Calculate movement vector
        start_pos = recent_history[0]['position']
        end_pos = recent_history[-1]['position']
        movement_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        analysis['movement_vector'] = movement_vector
        
        # Calculate speed
        total_distance = 0
        for i in range(1, len(recent_history)):
            prev_pos = recent_history[i-1]['position']
            curr_pos = recent_history[i]['position']
            distance = euclidean(prev_pos, curr_pos)
            total_distance += distance
        
        time_span = recent_history[-1]['timestamp'] - recent_history[0]['timestamp']
        speed = total_distance / time_span if time_span > 0 else 0
        analysis['speed'] = speed
        
        # Detect direction changes
        direction_changes = self._detect_direction_changes(recent_history)
        analysis['direction_changes'] = direction_changes
        
        # Detect loitering (staying in same area)
        loitering = self._detect_loitering(recent_history)
        analysis['loitering'] = loitering
        
        # Classify movement pattern
        pattern = self._classify_movement_pattern(analysis)
        analysis['pattern'] = pattern
        
        return analysis
    
    def _detect_direction_changes(self, history):
        """Detect significant direction changes in movement"""
        if len(history) < 3:
            return 0
        
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(history)):
            curr_pos = history[i]['position']
            prev_pos = history[i-1]['position']
            
            # Calculate direction vector
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            if dx == 0 and dy == 0:
                continue
            
            # Calculate angle
            angle = np.arctan2(dy, dx)
            
            if prev_direction is not None:
                angle_diff = abs(angle - prev_direction)
                # Normalize angle difference
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                # Significant direction change threshold (45 degrees)
                if angle_diff > np.pi / 4:
                    direction_changes += 1
            
            prev_direction = angle
        
        return direction_changes
    
    def _detect_loitering(self, history):
        """Detect if person is loitering in an area"""
        if len(history) < 5:
            return False
        
        positions = [h['position'] for h in history]
        
        # Calculate bounding box of all positions
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        
        # If person stays within small area for extended time, it's loitering
        area_threshold = 50  # pixels
        time_threshold = 10  # seconds
        
        time_span = history[-1]['timestamp'] - history[0]['timestamp']
        
        return (bbox_width < area_threshold and 
                bbox_height < area_threshold and 
                time_span > time_threshold)
    
    def _classify_movement_pattern(self, analysis):
        """Classify movement pattern based on analysis"""
        speed = analysis.get('speed', 0)
        direction_changes = analysis.get('direction_changes', 0)
        loitering = analysis.get('loitering', False)
        
        if loitering:
            return 'loitering'
        elif speed < 5:  # Very slow movement
            return 'stationary'
        elif speed < 20:  # Normal walking speed
            if direction_changes > 3:
                return 'wandering'
            else:
                return 'walking'
        else:  # Fast movement
            if direction_changes > 2:
                return 'erratic'
            else:
                return 'running'


class AppearanceChangeDetector:
    """Detect clothing/appearance changes"""
    
    def __init__(self):
        self.appearance_history = defaultdict(list)
        self.change_threshold = 0.4
        
    def initialize(self, reference_data):
        """Initialize appearance detector"""
        self.reference_appearance = reference_data.get('appearance_features', {})
        self.appearance_history.clear()
    
    def detect_changes(self, frame, tracking_results, timestamp):
        """Detect appearance changes for tracked persons"""
        changes = []
        
        for person_id, person_data in tracking_results.items():
            if person_data.get('is_target', False):  # Only analyze target person
                # Extract current appearance features
                current_appearance = self._extract_appearance_features(frame, person_data)
                
                # Compare with previous appearances
                change_detected = self._compare_appearances(person_id, current_appearance, timestamp)
                
                if change_detected:
                    changes.append({
                        'person_id': person_id,
                        'timestamp': timestamp,
                        'change_type': change_detected['type'],
                        'confidence': change_detected['confidence'],
                        'details': change_detected['details']
                    })
        
        return changes
    
    def _extract_appearance_features(self, frame, person_data):
        """Extract appearance features from person detection"""
        bbox = person_data['bbox']
        x, y, w, h = bbox
        person_roi = frame[y:y+h, x:x+w]
        
        features = {}
        
        # Color analysis
        try:
            # Divide person into upper and lower body
            upper_body = person_roi[:h//2, :]  # Upper half
            lower_body = person_roi[h//2:, :]  # Lower half
            
            # Extract color histograms
            features['upper_body_colors'] = self._extract_color_histogram(upper_body)
            features['lower_body_colors'] = self._extract_color_histogram(lower_body)
            features['overall_colors'] = self._extract_color_histogram(person_roi)
            
        except Exception as e:
            print(f"Color extraction error: {e}")
        
        # Texture analysis
        try:
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern for texture
            lbp = self._calculate_lbp(gray_roi)
            features['texture_pattern'] = lbp
            
        except Exception as e:
            print(f"Texture extraction error: {e}")
        
        return features
    
    def _extract_color_histogram(self, image_roi):
        """Extract color histogram from image region"""
        if image_roi is None or image_roi.size == 0:
            return None
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [30, 32, 32], [0, 180, 0, 256, 0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    
    def _calculate_lbp(self, gray_image):
        """Calculate Local Binary Pattern for texture analysis"""
        if gray_image is None or gray_image.size == 0:
            return None
        
        # Simple LBP implementation
        rows, cols = gray_image.shape
        lbp_image = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = gray_image[i, j]
                code = 0
                
                # 8-neighborhood
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp_image[i-1, j-1] = code
        
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def _compare_appearances(self, person_id, current_appearance, timestamp):
        """Compare current appearance with history"""
        # Add to history
        self.appearance_history[person_id].append({
            'timestamp': timestamp,
            'features': current_appearance
        })
        
        # Keep only recent history (last 60 seconds)
        self.appearance_history[person_id] = [
            h for h in self.appearance_history[person_id]
            if timestamp - h['timestamp'] <= 60
        ]
        
        history = self.appearance_history[person_id]
        
        if len(history) < 2:
            return None
        
        # Compare with previous appearance
        prev_appearance = history[-2]['features']
        
        # Calculate appearance similarity
        similarity = self._calculate_appearance_similarity(current_appearance, prev_appearance)
        
        if similarity < self.change_threshold:
            # Significant change detected
            change_details = self._analyze_change_details(current_appearance, prev_appearance)
            
            return {
                'type': 'appearance_change',
                'confidence': 1 - similarity,
                'details': change_details
            }
        
        return None
    
    def _calculate_appearance_similarity(self, appearance1, appearance2):
        """Calculate similarity between two appearance feature sets"""
        similarities = []
        
        # Color similarity
        for color_key in ['upper_body_colors', 'lower_body_colors', 'overall_colors']:
            if color_key in appearance1 and color_key in appearance2:
                hist1 = appearance1[color_key]
                hist2 = appearance2[color_key]
                if hist1 is not None and hist2 is not None:
                    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    similarities.append(max(0, similarity))
        
        # Texture similarity
        if 'texture_pattern' in appearance1 and 'texture_pattern' in appearance2:
            texture1 = appearance1['texture_pattern']
            texture2 = appearance2['texture_pattern']
            if texture1 is not None and texture2 is not None:
                # Chi-square distance for histograms
                chi_square = np.sum((texture1 - texture2) ** 2 / (texture1 + texture2 + 1e-7))
                texture_similarity = 1 / (1 + chi_square)
                similarities.append(texture_similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _analyze_change_details(self, current, previous):
        """Analyze what specifically changed in appearance"""
        details = {}
        
        # Analyze color changes
        color_changes = []
        for color_key in ['upper_body_colors', 'lower_body_colors']:
            if color_key in current and color_key in previous:
                hist1 = current[color_key]
                hist2 = previous[color_key]
                if hist1 is not None and hist2 is not None:
                    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    if similarity < 0.5:
                        body_part = 'upper_body' if 'upper' in color_key else 'lower_body'
                        color_changes.append(f"{body_part}_color_change")
        
        details['color_changes'] = color_changes
        
        # Analyze texture changes
        if 'texture_pattern' in current and 'texture_pattern' in previous:
            texture1 = current['texture_pattern']
            texture2 = previous['texture_pattern']
            if texture1 is not None and texture2 is not None:
                chi_square = np.sum((texture1 - texture2) ** 2 / (texture1 + texture2 + 1e-7))
                if chi_square > 0.5:
                    details['texture_change'] = True
        
        return details


class BehaviorAnalyzer:
    """Analyze behavioral patterns and detect anomalies"""
    
    def __init__(self):
        self.behavior_history = defaultdict(list)
        self.normal_patterns = {}
        
    def initialize(self):
        """Initialize behavior analyzer"""
        self.behavior_history.clear()
        
    def analyze_behavior(self, tracking_results, timestamp):
        """Analyze behavioral patterns for all tracked persons"""
        behavioral_data = {
            'movement_patterns': {},
            'interaction_analysis': {},
            'anomaly_detection': {},
            'activity_classification': {}
        }
        
        for person_id, person_data in tracking_results.items():
            # Analyze individual behavior
            behavior_analysis = self._analyze_individual_behavior(person_id, person_data, timestamp)
            
            if behavior_analysis:
                behavioral_data['movement_patterns'][person_id] = behavior_analysis.get('movement')
                behavioral_data['activity_classification'][person_id] = behavior_analysis.get('activity')
                
                # Check for anomalies
                anomalies = self._detect_behavioral_anomalies(person_id, behavior_analysis)
                if anomalies:
                    behavioral_data['anomaly_detection'][person_id] = anomalies
        
        # Analyze interactions between persons
        interactions = self._analyze_interactions(tracking_results, timestamp)
        behavioral_data['interaction_analysis'] = interactions
        
        return behavioral_data
    
    def _analyze_individual_behavior(self, person_id, person_data, timestamp):
        """Analyze behavior of individual person"""
        # Update behavior history
        behavior_entry = {
            'timestamp': timestamp,
            'position': person_data['features']['center'],
            'bbox': person_data['bbox'],
            'confidence': person_data['confidence']
        }
        
        self.behavior_history[person_id].append(behavior_entry)
        
        # Keep only recent history (last 2 minutes)
        self.behavior_history[person_id] = [
            h for h in self.behavior_history[person_id]
            if timestamp - h['timestamp'] <= 120
        ]
        
        history = self.behavior_history[person_id]
        
        if len(history) < 3:
            return None
        
        analysis = {}
        
        # Movement analysis
        movement_analysis = self._analyze_movement_behavior(history)
        analysis['movement'] = movement_analysis
        
        # Activity classification
        activity = self._classify_activity(history, movement_analysis)
        analysis['activity'] = activity
        
        return analysis
    
    def _analyze_movement_behavior(self, history):
        """Analyze movement behavior patterns"""
        if len(history) < 2:
            return {}
        
        positions = [h['position'] for h in history]
        timestamps = [h['timestamp'] for h in history]
        
        # Calculate movement metrics
        total_distance = 0
        speeds = []
        
        for i in range(1, len(positions)):
            distance = euclidean(positions[i], positions[i-1])
            time_diff = timestamps[i] - timestamps[i-1]
            
            total_distance += distance
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else 0
        speed_variance = np.var(speeds) if speeds else 0
        
        # Calculate path efficiency (straight line vs actual path)
        if len(positions) >= 2:
            straight_distance = euclidean(positions[0], positions[-1])
            path_efficiency = straight_distance / total_distance if total_distance > 0 else 0
        else:
            path_efficiency = 1.0
        
        return {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'speed_variance': speed_variance,
            'path_efficiency': path_efficiency,
            'movement_consistency': 1 - (speed_variance / (avg_speed + 1e-7))
        }
    
    def _classify_activity(self, history, movement_analysis):
        """Classify person's activity based on movement patterns"""
        avg_speed = movement_analysis.get('average_speed', 0)
        path_efficiency = movement_analysis.get('path_efficiency', 0)
        consistency = movement_analysis.get('movement_consistency', 0)
        
        # Activity classification rules
        if avg_speed < 2:
            if consistency > 0.8:
                return 'standing'
            else:
                return 'fidgeting'
        elif avg_speed < 10:
            if path_efficiency > 0.7:
                return 'walking_purposeful'
            else:
                return 'walking_casual'
        elif avg_speed < 25:
            if path_efficiency > 0.8:
                return 'fast_walking'
            else:
                return 'pacing'
        else:
            return 'running'
    
    def _detect_behavioral_anomalies(self, person_id, behavior_analysis):
        """Detect unusual behavioral patterns"""
        anomalies = []
        
        movement = behavior_analysis.get('movement', {})
        activity = behavior_analysis.get('activity', '')
        
        # Anomaly detection rules
        
        # 1. Erratic movement (high speed variance)
        speed_variance = movement.get('speed_variance', 0)
        avg_speed = movement.get('average_speed', 0)
        if avg_speed > 0 and speed_variance / avg_speed > 2:
            anomalies.append({
                'type': 'erratic_movement',
                'severity': 'medium',
                'description': 'Person showing erratic movement patterns'
            })
        
        # 2. Inefficient path (possible confusion or searching)
        path_efficiency = movement.get('path_efficiency', 1)
        if path_efficiency < 0.3 and avg_speed > 5:
            anomalies.append({
                'type': 'inefficient_path',
                'severity': 'low',
                'description': 'Person taking inefficient path, possibly lost or searching'
            })
        
        # 3. Sudden activity change
        if hasattr(self, 'previous_activity') and self.previous_activity.get(person_id):
            prev_activity = self.previous_activity[person_id]
            if prev_activity != activity:
                # Check for sudden changes from calm to agitated
                if prev_activity in ['standing', 'walking_casual'] and activity in ['pacing', 'running']:
                    anomalies.append({
                        'type': 'sudden_agitation',
                        'severity': 'high',
                        'description': f'Sudden change from {prev_activity} to {activity}'
                    })
        
        # Store current activity for next comparison
        if not hasattr(self, 'previous_activity'):
            self.previous_activity = {}
        self.previous_activity[person_id] = activity
        
        return anomalies
    
    def _analyze_interactions(self, tracking_results, timestamp):
        """Analyze interactions between tracked persons"""
        interactions = []
        
        person_ids = list(tracking_results.keys())
        
        # Check all pairs of persons
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                person1_id = person_ids[i]
                person2_id = person_ids[j]
                
                person1_data = tracking_results[person1_id]
                person2_data = tracking_results[person2_id]
                
                # Calculate distance between persons
                pos1 = person1_data['features']['center']
                pos2 = person2_data['features']['center']
                distance = euclidean(pos1, pos2)
                
                # Interaction detection threshold
                interaction_threshold = 100  # pixels
                
                if distance < interaction_threshold:
                    interaction = {
                        'person1_id': person1_id,
                        'person2_id': person2_id,
                        'distance': distance,
                        'timestamp': timestamp,
                        'interaction_type': self._classify_interaction(distance, person1_data, person2_data)
                    }
                    interactions.append(interaction)
        
        return interactions
    
    def _classify_interaction(self, distance, person1_data, person2_data):
        """Classify type of interaction based on distance and other factors"""
        if distance < 30:
            return 'close_interaction'
        elif distance < 60:
            return 'conversation'
        elif distance < 100:
            return 'proximity'
        else:
            return 'distant'


class CrowdAnalyzer:
    """Analyze crowd dynamics and extract persons from crowded scenes"""
    
    def __init__(self):
        self.crowd_threshold = 5
        self.density_history = []
        
    def initialize(self):
        """Initialize crowd analyzer"""
        self.density_history.clear()
        
    def analyze_crowd(self, frame, tracking_results, timestamp):
        """Analyze crowd dynamics and density"""
        crowd_data = {
            'person_count': len(tracking_results),
            'crowd_density': 0,
            'crowd_flow': {},
            'congestion_areas': [],
            'person_extraction_quality': {}
        }
        
        person_count = len(tracking_results)
        
        # Calculate crowd density
        frame_area = frame.shape[0] * frame.shape[1]
        crowd_density = person_count / (frame_area / 10000)  # Persons per 100x100 pixel area
        crowd_data['crowd_density'] = crowd_density
        
        # Update density history
        self.density_history.append({
            'timestamp': timestamp,
            'density': crowd_density,
            'count': person_count
        })
        
        # Keep only recent history (last 30 seconds)
        self.density_history = [
            h for h in self.density_history
            if timestamp - h['timestamp'] <= 30
        ]
        
        # Analyze crowd flow if it's a crowded scene
        if person_count >= self.crowd_threshold:
            crowd_data['is_crowded'] = True
            
            # Analyze crowd flow patterns
            flow_analysis = self._analyze_crowd_flow(tracking_results)
            crowd_data['crowd_flow'] = flow_analysis
            
            # Detect congestion areas
            congestion_areas = self._detect_congestion_areas(frame, tracking_results)
            crowd_data['congestion_areas'] = congestion_areas
            
            # Assess person extraction quality in crowd
            extraction_quality = self._assess_extraction_quality_in_crowd(tracking_results)
            crowd_data['person_extraction_quality'] = extraction_quality
        else:
            crowd_data['is_crowded'] = False
        
        return crowd_data
    
    def _analyze_crowd_flow(self, tracking_results):
        """Analyze overall crowd movement patterns"""
        if len(tracking_results) < 3:
            return {}
        
        positions = []
        movements = []
        
        for person_id, person_data in tracking_results.items():
            center = person_data['features']['center']
            positions.append(center)
            
            # Get movement vector if available from tracking history
            # This would require access to previous frame data
            # For now, we'll use a simplified approach
        
        # Calculate crowd center
        crowd_center = (
            np.mean([pos[0] for pos in positions]),
            np.mean([pos[1] for pos in positions])
        )
        
        # Calculate crowd spread
        distances_from_center = [euclidean(pos, crowd_center) for pos in positions]
        crowd_spread = np.mean(distances_from_center)
        
        # Analyze crowd distribution
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_spread = max(x_coords) - min(x_coords)
        y_spread = max(y_coords) - min(y_coords)
        
        return {
            'crowd_center': crowd_center,
            'crowd_spread': crowd_spread,
            'x_distribution': x_spread,
            'y_distribution': y_spread,
            'crowd_shape': 'elongated' if max(x_spread, y_spread) / min(x_spread, y_spread) > 2 else 'compact'
        }
    
    def _detect_congestion_areas(self, frame, tracking_results):
        """Detect areas of high person density (congestion)"""
        if len(tracking_results) < 4:
            return []
        
        positions = [person_data['features']['center'] for person_data in tracking_results.values()]
        
        # Use DBSCAN clustering to find dense areas
        try:
            clustering = DBSCAN(eps=50, min_samples=3).fit(positions)
            labels = clustering.labels_
            
            congestion_areas = []
            
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise points
                    continue
                
                cluster_positions = [positions[i] for i in range(len(positions)) if labels[i] == cluster_id]
                
                if len(cluster_positions) >= 3:  # Minimum for congestion
                    # Calculate cluster bounding box
                    x_coords = [pos[0] for pos in cluster_positions]
                    y_coords = [pos[1] for pos in cluster_positions]
                    
                    congestion_area = {
                        'cluster_id': cluster_id,
                        'person_count': len(cluster_positions),
                        'center': (np.mean(x_coords), np.mean(y_coords)),
                        'bbox': (min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)),
                        'density': len(cluster_positions) / ((max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords)) + 1)
                    }
                    congestion_areas.append(congestion_area)
            
            return congestion_areas
            
        except Exception as e:
            print(f"Congestion detection error: {e}")
            return []
    
    def _assess_extraction_quality_in_crowd(self, tracking_results):
        """Assess quality of person extraction in crowded scenes"""
        quality_assessment = {}
        
        for person_id, person_data in tracking_results.items():
            bbox = person_data['bbox']
            confidence = person_data['confidence']
            
            # Calculate quality factors
            
            # 1. Bounding box size (larger is generally better in crowds)
            bbox_area = bbox[2] * bbox[3]
            size_score = min(bbox_area / (100 * 150), 1.0)  # Normalize to 100x150 optimal
            
            # 2. Detection confidence
            confidence_score = confidence
            
            # 3. Overlap with other detections (less overlap is better)
            overlap_penalty = 0
            for other_id, other_data in tracking_results.items():
                if other_id != person_id:
                    overlap = self._calculate_bbox_overlap(bbox, other_data['bbox'])
                    overlap_penalty += overlap
            
            overlap_score = max(0, 1 - overlap_penalty / len(tracking_results))
            
            # 4. Position quality (center positions often better than edges)
            # This would require frame dimensions
            position_score = 0.8  # Default moderate score
            
            # Overall quality score
            overall_quality = (size_score * 0.3 + confidence_score * 0.4 + 
                             overlap_score * 0.2 + position_score * 0.1)
            
            quality_assessment[person_id] = {
                'overall_quality': overall_quality,
                'size_score': size_score,
                'confidence_score': confidence_score,
                'overlap_score': overlap_score,
                'position_score': position_score,
                'extraction_difficulty': 'high' if overall_quality < 0.5 else 'medium' if overall_quality < 0.7 else 'low'
            }
        
        return quality_assessment
    
    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection_area = (right - left) * (bottom - top)
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            union_area = bbox1_area + bbox2_area - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0
        else:
            return 0


# Global analyzer instance
intelligent_analyzer = IntelligentFootageAnalyzer()