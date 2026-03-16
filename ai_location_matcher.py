"""
AI Location Matcher - FAST face detection with smart sampling abd do fast 
"""
import cv2
import numpy as np
import face_recognition
import os
import json
from datetime import datetime
from __init__ import db
from models import Case, SurveillanceFootage, LocationMatch, PersonDetection
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

class AILocationMatcher:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.max_workers = max(2, multiprocessing.cpu_count() - 1)
        
    def find_location_matches(self, case_id):
        try:
            case = Case.query.get(case_id)
            if not case:
                return []
            
            footage_list = SurveillanceFootage.query.filter_by(is_active=True).all()
            matches = []
            
            for footage in footage_list:
                match_score = 0.0
                
                if case.last_seen_location and footage.location_name:
                    import re
                    # Remove ALL spaces for better matching (pantnagar = pant nagar)
                    case_clean = re.sub(r'[^a-z0-9]', '', case.last_seen_location.lower())
                    footage_clean = re.sub(r'[^a-z0-9]', '', footage.location_name.lower())
                    
                    # Exact match without spaces
                    if case_clean == footage_clean:
                        match_score = 1.0
                    # Substring match (pantnagar in pantnagarcol)
                    elif footage_clean in case_clean or case_clean in footage_clean:
                        match_score = 0.9
                    else:
                        # Word-based matching with spaces
                        case_words = set(re.sub(r'[^a-z0-9\s]', ' ', case.last_seen_location.lower()).split())
                        footage_words = set(re.sub(r'[^a-z0-9\s]', ' ', footage.location_name.lower()).split())
                        common = case_words.intersection(footage_words)
                        if common:
                            match_score = len(common) / max(len(case_words), len(footage_words))
                
                if match_score > 0.2:
                    matches.append({'footage': footage, 'match_score': match_score, 'distance_km': None})
            
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            return matches
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return []
    
    def process_new_case(self, case_id):
        try:
            matches = self.find_location_matches(case_id)
            for match_data in matches:
                existing = LocationMatch.query.filter_by(case_id=case_id, footage_id=match_data['footage'].id).first()
                if not existing:
                    # Realistic location match: 75-85%
                    realistic_match = 0.75 + (match_data['match_score'] * 0.10)
                    location_match = LocationMatch(
                        case_id=case_id,
                        footage_id=match_data['footage'].id,
                        match_score=realistic_match,
                        distance_km=match_data['distance_km'],
                        status='pending'
                    )
                    db.session.add(location_match)
            db.session.commit()
            return len(matches)
        except Exception as e:
            logger.error(f"Error processing case: {e}")
            db.session.rollback()
            return 0
    
    def process_new_footage(self, footage_id):
        try:
            footage = SurveillanceFootage.query.get(footage_id)
            if not footage:
                logger.error(f"Footage {footage_id} not found")
                return 0
            
            logger.info(f"Processing footage: {footage.location_name}")
            
            active_cases = Case.query.filter(Case.status.in_(['Approved', 'Active', 'Under Processing'])).all()
            logger.info(f"Found {len(active_cases)} active cases")
            matches_created = 0
            
            for case in active_cases:
                match_score = 0.0
                if case.last_seen_location and footage.location_name:
                    import re
                    # Remove ALL spaces for better matching (pantnagar = pant nagar)
                    case_clean = re.sub(r'[^a-z0-9]', '', case.last_seen_location.lower())
                    footage_clean = re.sub(r'[^a-z0-9]', '', footage.location_name.lower())
                    
                    # Exact match without spaces
                    if case_clean == footage_clean:
                        match_score = 1.0
                    # Substring match (pantnagar in pantnagarcol)
                    elif footage_clean in case_clean or case_clean in footage_clean:
                        match_score = 0.9
                    else:
                        # Word-based matching with spaces
                        case_words = set(re.sub(r'[^a-z0-9\s]', ' ', case.last_seen_location.lower()).split())
                        footage_words = set(re.sub(r'[^a-z0-9\s]', ' ', footage.location_name.lower()).split())
                        common = case_words.intersection(footage_words)
                        if common:
                            match_score = len(common) / max(len(case_words), len(footage_words))
                
                logger.info(f"Case {case.id} ({case.person_name}): location='{case.last_seen_location}', match_score={match_score}")
                
                if match_score > 0.2:
                    existing = LocationMatch.query.filter_by(case_id=case.id, footage_id=footage_id).first()
                    if not existing:
                        # Realistic location match: 75-85%
                        realistic_match = 0.75 + (match_score * 0.10)
                        location_match = LocationMatch(
                            case_id=case.id,
                            footage_id=footage_id,
                            match_score=realistic_match,
                            distance_km=None,
                            status='pending'
                        )
                        db.session.add(location_match)
                        matches_created += 1
                        logger.info(f"Created match for case {case.id}")
                    else:
                        logger.info(f"Match already exists for case {case.id}")
            
            db.session.commit()
            logger.info(f"Total matches created: {matches_created}")
            return matches_created
        except Exception as e:
            logger.error(f"Error processing footage: {e}")
            db.session.rollback()
            return 0
    
    def analyze_footage_for_person(self, match_id):
        try:
            match = LocationMatch.query.get(match_id)
            if not match:
                return False
            
            match.status = 'processing'
            match.ai_analysis_started = datetime.utcnow()
            db.session.commit()
            
            # Get target face encodings
            target_encodings = []
            for target_image in match.case.target_images:
                image_path = os.path.join('static', target_image.image_path)
                if not os.path.exists(image_path):
                    image_path = os.path.join('app', 'static', target_image.image_path)
                
                if os.path.exists(image_path):
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            target_encodings.append(encodings[0])
                    except:
                        pass
            
            if not target_encodings:
                match.status = 'failed'
                db.session.commit()
                return False
            
            # Get footage path
            footage_path = os.path.join('static', match.footage.video_path)
            if not os.path.exists(footage_path):
                footage_path = os.path.join('app', 'static', match.footage.video_path)
            if not os.path.exists(footage_path):
                match.status = 'failed'
                db.session.commit()
                return False
            
            # FAST video analysis
            detections = self._fast_analyze_video(footage_path, target_encodings, match_id)
            
            match.detection_count = len(detections)
            match.person_found = len(detections) > 0
            
            if detections:
                confidences = [d['confidence'] for d in detections]
                match.confidence_score = sum(confidences) / len(confidences)
                match.match_score = match.confidence_score
            else:
                match.confidence_score = 0.0
                match.match_score = 0.0
            
            match.status = 'completed'
            match.ai_analysis_completed = datetime.utcnow()
            db.session.commit()
            
            return True
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            if match:
                match.status = 'failed'
                db.session.commit()
            return False
    
    def _fast_analyze_video(self, video_path, target_encodings, match_id):
        """ULTRA FAST video analysis - smart sampling + batch processing"""
        detections = []
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # SMART SAMPLING: Sample frames intelligently
            # For short videos: every 0.5 sec
            # For long videos: every 2 sec
            if duration < 60:  # < 1 min
                sample_interval = 0.5
            elif duration < 300:  # < 5 min
                sample_interval = 1.0
            elif duration < 1800:  # < 30 min
                sample_interval = 2.0
            else:  # > 30 min
                sample_interval = 5.0
            
            frame_indices = [int(i * fps * sample_interval) for i in range(int(duration / sample_interval))]
            
            logger.info(f"Fast analysis: {duration:.1f}s video, sampling {len(frame_indices)} frames")
            
            # Batch process frames
            frames_to_process = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames_to_process.append((idx, frame))
            
            cap.release()
            
            # Process frames in parallel batches
            batch_size = 5
            for i in range(0, len(frames_to_process), batch_size):
                batch = frames_to_process[i:i+batch_size]
                batch_detections = self._process_frame_batch(batch, target_encodings, match_id, fps)
                detections.extend(batch_detections)
            
            db.session.commit()
            logger.info(f"Fast analysis complete: {len(detections)} detections in {len(frame_indices)} frames")
            
        except Exception as e:
            logger.error(f"Fast video analysis error: {e}")
        
        return detections
    
    def _process_frame_batch(self, frame_batch, target_encodings, match_id, fps):
        """Process multiple frames together for speed"""
        detections = []
        
        for frame_idx, frame in frame_batch:
            timestamp = frame_idx / fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Quick face detection with scaling
            all_faces = []
            
            # Try 2x scale first (fastest for distant faces)
            h, w = rgb_frame.shape[:2]
            scaled_2x = cv2.resize(rgb_frame, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
            faces_2x = face_recognition.face_locations(scaled_2x, model="cnn")
            if faces_2x:
                faces_2x = [(int(t/2), int(r/2), int(b/2), int(l/2)) for t,r,b,l in faces_2x]
                all_faces.extend(faces_2x)
            
            # If no faces, try 1.5x scale
            if not all_faces:
                scaled_15x = cv2.resize(rgb_frame, (int(w*1.5), int(h*1.5)), interpolation=cv2.INTER_LINEAR)
                faces_15x = face_recognition.face_locations(scaled_15x, model="cnn")
                if faces_15x:
                    faces_15x = [(int(t/1.5), int(r/1.5), int(b/1.5), int(l/1.5)) for t,r,b,l in faces_15x]
                    all_faces.extend(faces_15x)
            
            # If still no faces, try original
            if not all_faces:
                all_faces = face_recognition.face_locations(rgb_frame, model="cnn")
            
            if all_faces:
                encodings = face_recognition.face_encodings(rgb_frame, all_faces)
                for encoding, location in zip(encodings, all_faces):
                    distances = face_recognition.face_distance(target_encodings, encoding)
                    best_distance = float(np.min(distances))
                    
                    # MULTIPLE VERIFICATION for Asian faces
                    # 1. Distance check
                    # 2. Multiple target encoding comparison
                    # 3. Face quality check
                    
                    # Check against ALL target encodings (not just best match)
                    good_matches = sum(1 for d in distances if d <= 0.3)
                    match_ratio = good_matches / len(target_encodings)
                    
                    # Face quality check
                    top, right, bottom, left = location
                    face_height = bottom - top
                    face_width = right - left
                    face_area = face_height * face_width
                    
                    # Skip very small or distorted faces
                    if face_area < 2500 or face_width < 40 or face_height < 40:
                        continue
                    
                    # Aspect ratio check (normal face should be ~1.3 height/width)
                    aspect_ratio = face_height / face_width if face_width > 0 else 0
                    if aspect_ratio < 1.0 or aspect_ratio > 2.0:
                        continue
                    
                    # ULTRA STRICT matching for Asian faces - prevent cross-race confusion
                    # Face distance: 0.0 = perfect, 0.3 = maximum for Asian faces
                    # PLUS: Must match with majority of target encodings
                    if best_distance <= 0.3 and match_ratio >= 0.6:  # 60% of target images must match
                        if best_distance < 0.15:  # Excellent match
                            confidence_score = 0.90 + (0.05 * (1 - best_distance / 0.15))  # 90-95%
                        elif best_distance < 0.25:  # Very good match
                            confidence_score = 0.80 + (0.10 * (1 - (best_distance - 0.15) / 0.10))  # 80-90%
                        else:  # Good match
                            confidence_score = 0.70 + (0.10 * (1 - (best_distance - 0.25) / 0.05))  # 70-80%
                        
                        confidence_percent = confidence_score * 100
                        
                        # ULTRA STRICT threshold - only 80%+ matches for Asian faces
                        if confidence_score >= 0.80:
                            self._save_detection(frame, location, timestamp, match_id, confidence_score, best_distance)
                            detections.append({'timestamp': timestamp, 'confidence': confidence_score})
        
        return detections
    
    def _save_detection(self, frame, location, timestamp, match_id, confidence_score, distance):
        try:
            frame_filename = f"detection_{match_id}_{int(timestamp)}.jpg"
            frame_dir = os.path.join('static', 'detections')
            if not os.path.exists(frame_dir):
                frame_dir = os.path.join('app', 'static', 'detections')
            os.makedirs(frame_dir, exist_ok=True)
            
            top, right, bottom, left = location
            region = frame[max(0, top-20):min(frame.shape[0], bottom+20), 
                          max(0, left-20):min(frame.shape[1], right+20)]
            
            if region.size > 0:
                cv2.imwrite(os.path.join(frame_dir, frame_filename), region)
                
                detection = PersonDetection(
                    location_match_id=match_id,
                    timestamp=timestamp,
                    confidence_score=confidence_score,
                    face_match_score=confidence_score,
                    clothing_match_score=None,
                    detection_box=json.dumps({'top': int(top), 'right': int(right), 'bottom': int(bottom), 'left': int(left), 'distance': float(distance)}),
                    frame_path=f"detections/{frame_filename}",
                    analysis_method='cnn_face_recognition'
                )
                db.session.add(detection)
        except Exception as e:
            logger.error(f"Save detection error: {e}")

ai_matcher = AILocationMatcher()
