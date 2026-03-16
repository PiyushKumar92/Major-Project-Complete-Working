"""
Enhanced Vision Processor with YOLO v8, FaceNet, and DeepSORT
Extends existing vision_engine.py with advanced AI capabilities
"""

import logging
import os
from datetime import datetime, timezone
import cv2
import numpy as np
from flask import current_app
from werkzeug.utils import secure_filename
from __init__ import db
from models import Case, Sighting
from enhanced_ai_engine import EnhancedAIEngine

logger = logging.getLogger(__name__)

class EnhancedVisionProcessor:
    """Enhanced Vision Processor with advanced AI capabilities"""
    
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
        
        # Initialize enhanced AI engine
        self.ai_engine = EnhancedAIEngine()
        
        # Get target encodings
        self.target_encodings = self._get_target_encodings()
        
        logger.info(f"Enhanced VisionProcessor initialized for case {self.case_id}")
        
        # Log available AI capabilities
        ai_info = self.ai_engine.get_system_info()
        logger.info(f"AI Capabilities: {ai_info}")

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
                    
                # Load image
                image = cv2.imread(secure_path)
                if image is None:
                    continue
                    
                # Use enhanced face feature extraction
                face_features = self.ai_engine.extract_face_features_facenet(image)
                if face_features is not None:
                    encodings.append(face_features)
                    
            except Exception as e:
                logger.error(f"Error processing target image {target_image.id}: {e}")
                
        return encodings

    def _process_frame_enhanced(self, frame, frame_number, fps, video_obj):
        """Enhanced frame processing with YOLO v8, FaceNet, and DeepSORT"""
        try:
            timestamp = frame_number / fps
            
            # Enhanced person detection using YOLO v8
            person_detections = self.ai_engine.detect_persons_yolo(frame)
            
            # Enhanced person tracking using DeepSORT
            tracked_persons = self.ai_engine.track_persons_deepsort(frame, person_detections)
            
            # Process each tracked person
            for person in tracked_persons:
                x, y, w, h = person['bbox']
                person_roi = frame[y:y+h, x:x+w]
                
                # Enhanced face matching
                face_confidence = self.ai_engine._match_face_enhanced(
                    person_roi, self.target_encodings
                )
                
                if face_confidence > self.face_confidence_threshold:
                    # Additional AI analysis
                    demographics = self.ai_engine.analyze_demographics(person_roi)
                    clothing = self.ai_engine.analyze_clothing(person_roi)
                    quality_score = self.ai_engine._calculate_detection_quality(person_roi)
                    
                    # Create enhanced sighting
                    self._create_enhanced_sighting(
                        timestamp, face_confidence, person, video_obj, 
                        person_roi, demographics, clothing, quality_score
                    )
                    
        except Exception as e:
            logger.error(f"Enhanced frame processing error for case {self.case_id}: {e}")

    def _create_enhanced_sighting(self, timestamp, confidence, person_data, video_obj, 
                                person_roi, demographics, clothing, quality_score):
        """Create enhanced sighting with additional AI analysis"""
        try:
            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            thumbnail_filename = f"enhanced_sighting_{self.case_id}_{video_obj.id}_{timestamp_str}.jpg"
            
            secure_path = self._get_secure_path(thumbnail_filename)
            if not secure_path:
                logger.error(f"Invalid thumbnail path for case {self.case_id}")
                return
            
            # Save thumbnail with enhanced annotations
            annotated_roi = self._annotate_detection(
                person_roi, person_data, demographics, clothing, quality_score
            )
            
            if not cv2.imwrite(secure_path, annotated_roi):
                logger.error(f"Failed to save enhanced thumbnail for case {self.case_id}")
                return
            
            # Convert to web-accessible path
            filename = os.path.basename(secure_path)
            db_path = f"static/uploads/{filename}"
            
            # Create enhanced sighting record
            sighting = Sighting(
                case_id=self.case_id,
                search_video_id=video_obj.id,
                timestamp=timestamp,
                confidence_score=confidence,
                detection_method=f"enhanced_{person_data.get('method', 'ai')}",
                thumbnail_path=db_path
            )
            
            # Add enhanced metadata (if Sighting model supports it)
            if hasattr(sighting, 'metadata'):
                sighting.metadata = {
                    'track_id': person_data.get('track_id', -1),
                    'detection_confidence': person_data.get('confidence', 0.0),
                    'demographics': demographics,
                    'clothing_analysis': clothing,
                    'quality_score': quality_score,
                    'ai_methods_used': self.ai_engine.get_system_info()['ai_methods_used']
                }
            
            db.session.add(sighting)
            logger.info(f"Enhanced sighting created for case {self.case_id} at {timestamp:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to create enhanced sighting for case {self.case_id}: {e}")

    def _annotate_detection(self, image, person_data, demographics, clothing, quality_score):
        """Annotate detection with AI analysis results"""
        try:
            annotated = image.copy()
            height, width = annotated.shape[:2]
            
            # Add bounding box
            cv2.rectangle(annotated, (5, 5), (width-5, height-5), (0, 255, 0), 2)
            
            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1
            
            y_offset = 20
            
            # Detection method
            method = person_data.get('method', 'unknown')
            cv2.putText(annotated, f"Method: {method}", (10, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += 20
            
            # Quality score
            cv2.putText(annotated, f"Quality: {quality_score:.2f}", (10, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += 20
            
            # Demographics (if available)
            if demographics.get('estimated_age') != 'Unknown':
                cv2.putText(annotated, f"Age: {demographics['estimated_age']}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += 20
            
            # Clothing colors
            if clothing.get('dominant_colors'):
                colors_str = ', '.join(clothing['dominant_colors'][:2])  # First 2 colors
                cv2.putText(annotated, f"Colors: {colors_str}", (10, y_offset), 
                           font, font_scale, color, thickness)
            
            return annotated
            
        except Exception as e:
            logger.error(f"Annotation error: {e}")
            return image

    def run_enhanced_analysis(self):
        """Run enhanced analysis with advanced AI capabilities"""
        logger.info(f"Starting enhanced analysis for case {self.case_id}")
        
        # Log AI capabilities being used
        ai_info = self.ai_engine.get_system_info()
        logger.info(f"Using AI methods: {ai_info}")
        
        analysis_results = {
            'case_id': self.case_id,
            'videos_processed': 0,
            'total_detections': 0,
            'ai_methods_used': ai_info,
            'processing_errors': [],
            'start_time': datetime.now().isoformat()
        }
        
        for video in self.case.search_videos:
            try:
                # Security validation
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
                
                # Process video with enhanced AI
                video_results = self._process_video_enhanced(video, video_path)
                
                analysis_results['videos_processed'] += 1
                analysis_results['total_detections'] += video_results.get('detections_count', 0)
                
                if video_results.get('error'):
                    analysis_results['processing_errors'].append({
                        'video_id': video.id,
                        'error': video_results['error']
                    })
                
            except Exception as e:
                logger.error(f"Error processing video {video.id}: {e}")
                analysis_results['processing_errors'].append({
                    'video_id': video.id,
                    'error': str(e)
                })
        
        analysis_results['end_time'] = datetime.now().isoformat()
        logger.info(f"Enhanced analysis completed for case {self.case_id}: {analysis_results}")
        
        return analysis_results

    def _process_video_enhanced(self, video_obj, video_path):
        """Process single video with enhanced AI"""
        results = {
            'video_id': video_obj.id,
            'detections_count': 0,
            'processing_time': 0,
            'error': None
        }
        
        cap = None
        start_time = datetime.now()
        
        try:
            video_obj.status = "Processing"
            db.session.commit()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = 0
            detections_before = len(video_obj.sightings)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frames with enhanced AI
                if frame_count % self.frame_skip == 0:
                    self._process_frame_enhanced(frame, frame_count, fps, video_obj)
                
                frame_count += 1
            
            # Calculate detections added
            detections_after = len(video_obj.sightings)
            results['detections_count'] = detections_after - detections_before
            
            video_obj.status = "Completed"
            video_obj.processed_at = datetime.now(timezone.utc)
            
            logger.info(f"Enhanced processing completed for video {video_obj.id}: "
                       f"{results['detections_count']} detections found")
            
        except Exception as e:
            logger.error(f"Enhanced video processing error for {video_obj.id}: {e}")
            video_obj.status = "Failed"
            results['error'] = str(e)
            
        finally:
            if cap:
                cap.release()
            
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            try:
                db.session.commit()
            except Exception as e:
                logger.error(f"Database commit failed: {e}")
                try:
                    db.session.rollback()
                except Exception as rollback_error:
                    logger.critical(f"Database rollback failed: {rollback_error}")
        
        return results

    def get_analysis_summary(self):
        """Get summary of enhanced analysis capabilities"""
        return {
            'case_id': self.case_id,
            'ai_capabilities': self.ai_engine.get_system_info(),
            'target_encodings_count': len(self.target_encodings),
            'configuration': {
                'frame_skip': self.frame_skip,
                'face_confidence_threshold': self.face_confidence_threshold,
                'face_match_tolerance': self.face_match_tolerance
            }
        }

# Backward compatibility function
def create_enhanced_vision_processor(case_id):
    """Create enhanced vision processor with fallback to basic processor"""
    try:
        return EnhancedVisionProcessor(case_id)
    except Exception as e:
        logger.warning(f"Enhanced processor failed, using basic processor: {e}")
        # Fallback to basic vision processor
        from vision_engine import VisionProcessor
        return VisionProcessor(case_id)