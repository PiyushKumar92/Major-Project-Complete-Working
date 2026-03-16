import logging
import os
from datetime import datetime, timezone
import cv2
import face_recognition
from flask import current_app
from werkzeug.utils import secure_filename
from __init__ import db
from models import Case, Sighting

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VisionProcessor:
    def __init__(self, case_id):
        self.case_id = case_id
        self.case = Case.query.get(case_id)
        if not self.case:
            logger.error(f"Case {case_id} not found")
            raise ValueError(f"Case {case_id} not found")
        
        self.config = current_app.config
        self.frame_skip = self.config.get('FRAME_SKIP', 15)
        self.face_confidence_threshold = self.config.get('FACE_CONFIDENCE_THRESHOLD', 0.65)
        self.face_match_tolerance = self.config.get('FACE_MATCH_TOLERANCE', 0.40)
        self.upload_folder = self.config.get('UPLOAD_FOLDER', 'app/static/uploads')
        
        self.target_encodings = self._get_target_encodings()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info(f"VisionProcessor initialized for case {self.case_id}")

    def _get_secure_path(self, filename):
        """Get secure path for file"""
        if not filename or '..' in filename:
            return None
        secure_name = secure_filename(os.path.basename(filename))
        file_path = os.path.join(self.upload_folder, secure_name)
        if not os.path.abspath(file_path).startswith(os.path.abspath(self.upload_folder)):
            return None
        return file_path
    
    def _get_target_encodings(self):
        """Load target images and return face encodings"""
        encodings = []
        for target_image in self.case.target_images:
            try:
                secure_path = self._get_secure_path(target_image.image_path)
                if not secure_path or not os.path.exists(secure_path):
                    continue
                image = face_recognition.load_image_file(secure_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings.extend(face_encodings)
            except Exception as e:
                logger.error(f"Error processing target image {target_image.id}: {e}")
        return encodings



    def _detect_people(self, frame):
        """Detect people in frame using HOG detector"""
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        if len(weights) == 0:
            return []
        import numpy as np
        confident_indices = np.where(weights > 0.5)[0]
        return [rects[i] for i in confident_indices]

    def _process_frame(self, frame, frame_number, fps, video_obj):
        """Process frame for person detection and matching"""
        try:
            timestamp = frame_number / fps
            people_boxes = self._detect_people(frame)
            
            for (x, y, w, h) in people_boxes:
                person_roi = frame[y:y + h, x:x + w]
                face_confidence = self._match_face(person_roi)
                if face_confidence > self.face_confidence_threshold:
                    self._create_sighting(timestamp, face_confidence, "face", video_obj, person_roi)
        except Exception as e:
            logger.error(f"Error processing frame {frame_number} for case {self.case_id}: {e}")

    def _match_face(self, person_roi):
        """Perfect matching - only real faces"""
        if not self.target_encodings:
            return 0.0
        try:
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            
            # Face detection validation
            face_locations = face_recognition.face_locations(rgb_roi)
            if not face_locations:
                return 0.0  # No face = No score
            
            roi_face_encodings = face_recognition.face_encodings(rgb_roi, face_locations)
            if not roi_face_encodings:
                return 0.0
            
            face_distances = face_recognition.face_distance(self.target_encodings, roi_face_encodings[0])
            if len(face_distances) > 0:
                min_distance = min(face_distances)
                
                # Only accept good matches
                if min_distance > 0.4:
                    return 0.0  # Not a match
                
                base_confidence = 1.0 - min_distance
                
                # Blur check for quality adjustment
                gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if blur_score < 30:  # Very blurry
                    return base_confidence * 0.6
                elif blur_score < 100:  # Slightly blurry
                    return base_confidence * 0.9
                else:  # Clear face
                    return base_confidence * 1.2
        except Exception as e:
            logger.error(f"Face matching error for case {self.case_id}: {e}")
        return 0.0

    def _create_sighting(self, timestamp, confidence, method, video_obj, person_roi):
        """Create sighting record and save thumbnail"""
        try:
            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            thumbnail_filename = f"sighting_{self.case_id}_{video_obj.id}_{timestamp_str}.jpg"
            
            secure_path = self._get_secure_path(thumbnail_filename)
            if not secure_path:
                logger.error(f"Invalid thumbnail path for case {self.case_id}")
                return
            
            if not cv2.imwrite(secure_path, person_roi):
                logger.error(f"Failed to save thumbnail for case {self.case_id}")
                return
            
            # Convert absolute path to web-accessible relative path
            filename = os.path.basename(secure_path)
            db_path = f"static/uploads/{filename}"
            
            sighting = Sighting(
                case_id=self.case_id,
                search_video_id=video_obj.id,
                timestamp=timestamp,
                confidence_score=confidence,
                detection_method=method,
                thumbnail_path=db_path
            )
            db.session.add(sighting)
            logger.info(f"Sighting created for case {self.case_id} at {timestamp:.2f}s")
        except Exception as e:
            logger.error(f"Failed to create sighting for case {self.case_id}: {e}")

    def run_analysis(self):
        """Analyze all search videos for the case"""
        logger.info(f"Starting analysis for case {self.case_id}")
        
        for video in self.case.search_videos:
            # Secure path validation to prevent path traversal
            if '..' in video.video_path or os.path.isabs(video.video_path):
                logger.error(f"Invalid video path detected: {video.video_path}")
                continue
                
            video_path = os.path.join('app', video.video_path)
            video_path = os.path.abspath(video_path)
            
            if not video_path.startswith(os.path.abspath('app')):
                logger.error(f"Path traversal attempt blocked: {video.video_path}")
                continue
                
            if not os.path.exists(video_path):
                logger.error(f"Video not found: {video_path}")
                continue
            
            cap = None
            try:
                video.status = "Processing"
                db.session.commit()
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video file: {video_path}")
                    video.status = "Failed"
                    db.session.commit()
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % self.frame_skip == 0:
                        self._process_frame(frame, frame_count, fps, video)
                    frame_count += 1
                
                video.status = "Completed"
                video.processed_at = datetime.now(timezone.utc)
                logger.info(f"Completed video {video.id} for case {self.case_id}")
                
            except (cv2.error, OSError, IOError) as e:
                logger.error(f"Video processing error for {video.id}: {e}")
                video.status = "Failed"
            except Exception as e:
                logger.error(f"Critical error processing video {video.id}: {e}", exc_info=True)
                video.status = "Failed"
            finally:
                if cap:
                    cap.release()
                try:
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Database commit failed: {e}")
                    try:
                        db.session.rollback()
                    except Exception as rollback_error:
                        logger.critical(f"Database rollback failed: {rollback_error}")
