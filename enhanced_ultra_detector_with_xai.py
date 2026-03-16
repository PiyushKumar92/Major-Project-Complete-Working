"""
Enhanced Ultra-Advanced Person Detector with XAI and Evidence Integrity
Integrates all perfection improvements for absolute accuracy and legal readiness
"""

import cv2
import numpy as np
import face_recognition
import logging
import json
import hashlib
from datetime import datetime, timezone
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

# Import our new systems
from xai_feature_weighting_system import analyze_detection_with_xai, XAIDetectionResult
from evidence_integrity_system import create_evidence_frame, create_evidence_chain
from system_resilience_monitor import get_system_status
from ultra_advanced_person_detector import UltraAdvancedPersonDetector
from professional_person_detector import ProfessionalPersonDetector

logger = logging.getLogger(__name__)

class EnhancedUltraDetectorWithXAI:
    """
    Enhanced Ultra-Advanced Person Detector with XAI and Evidence Integrity
    Provides absolute perfection with transparent AI decisions and legal-grade evidence
    """
    
    def __init__(self, case_id: int):
        self.case_id = case_id
        
        # Initialize base detectors
        self.ultra_detector = UltraAdvancedPersonDetector(case_id)
        self.professional_detector = ProfessionalPersonDetector(case_id)
        
        # XAI and Evidence systems
        self.xai_results: List[XAIDetectionResult] = []
        self.evidence_chain = None
        
        # Enhanced thresholds for perfection
        self.PERFECTION_THRESHOLD = 0.95  # 95% confidence for auto-approval
        self.HIGH_CONFIDENCE_THRESHOLD = 0.85  # 85% for high confidence
        self.CONFIRMATION_THRESHOLD = 0.65  # 65% needs confirmation
        self.LOW_QUALITY_THRESHOLD = 0.40  # Below this goes to enhanced review
        
        # System status monitoring
        self.system_status = get_system_status()
        self.gpu_available = self.system_status.get('gpu_available', False)
        self.cpu_fallback_active = self.system_status.get('cpu_fallback_active', False)
        
        logger.info(f"Enhanced Ultra Detector initialized for case {case_id}")
        logger.info(f"System status: GPU={self.gpu_available}, CPU_Fallback={self.cpu_fallback_active}")
    
    def analyze_footage_with_perfection(self, footage_id: int) -> Dict:
        """
        Analyze footage with absolute perfection and legal readiness
        """
        from models import SurveillanceFootage
        
        footage = SurveillanceFootage.query.get(footage_id)
        if not footage:
            return {"error": "Footage not found"}
        
        # Create evidence chain for this analysis
        self.evidence_chain = create_evidence_chain(self.case_id)
        
        logger.info(f"Starting perfection analysis: Case {self.case_id}, Footage {footage_id}")
        
        # Step 1: Multi-method detection with fallback
        detection_results = self._perform_multi_method_detection(footage_id)
        
        # Step 2: XAI analysis for each detection
        xai_enhanced_results = self._enhance_with_xai_analysis(detection_results)
        
        # Step 3: Evidence integrity for high-confidence detections
        evidence_secured_results = self._secure_evidence_integrity(xai_enhanced_results, footage_id)
        
        # Step 4: Low-quality review queue for uncertain detections
        low_quality_queued = self._queue_low_quality_detections(evidence_secured_results)
        
        # Step 5: Generate comprehensive perfection report
        perfection_report = self._generate_perfection_report(evidence_secured_results, low_quality_queued)
        
        # Step 6: Save all results with full audit trail
        self._save_perfection_results(evidence_secured_results, footage_id)
        
        return perfection_report
    
    def _perform_multi_method_detection(self, footage_id: int) -> List[Dict]:
        """
        Perform detection using multiple methods with intelligent fallback
        """
        all_detections = []
        
        try:
            # Method 1: Ultra-Advanced Detection (GPU preferred)
            if self.gpu_available and not self.cpu_fallback_active:
                logger.info("Using GPU-accelerated ultra-advanced detection")
                ultra_results = self.ultra_detector.analyze_footage_ultra_advanced(footage_id)
                
                if not ultra_results.get('error'):
                    for appearance in ultra_results.get('appearances', []):
                        for detection in appearance.get('detections', []):
                            detection['source_method'] = 'ultra_advanced_gpu'
                            detection['footage_id'] = footage_id
                            detection['case_id'] = self.case_id
                            all_detections.append(detection)
            
            # Method 2: Professional Detection (CPU fallback or parallel)
            logger.info("Running professional detection analysis")
            professional_results = self.professional_detector.analyze_footage_professional(footage_id)
            
            if not professional_results.get('error'):
                for appearance in professional_results.get('appearances', []):
                    for detection in appearance.get('detections', []):
                        detection['source_method'] = 'professional_cpu'
                        detection['footage_id'] = footage_id
                        detection['case_id'] = self.case_id
                        all_detections.append(detection)
            
            # Method 3: Enhanced detection for difficult cases
            enhanced_detections = self._perform_enhanced_detection(footage_id)
            all_detections.extend(enhanced_detections)
            
        except Exception as e:
            logger.error(f"Error in multi-method detection: {e}")
            # Fallback to basic detection
            fallback_detections = self._perform_fallback_detection(footage_id)
            all_detections.extend(fallback_detections)
        
        # Merge and deduplicate detections
        merged_detections = self._merge_duplicate_detections(all_detections)
        
        logger.info(f"Multi-method detection complete: {len(merged_detections)} unique detections")
        return merged_detections
    
    def _perform_enhanced_detection(self, footage_id: int) -> List[Dict]:
        """
        Enhanced detection for difficult scenarios
        """
        enhanced_detections = []
        
        try:
            from models import SurveillanceFootage
            
            footage = SurveillanceFootage.query.get(footage_id)
            video_path = os.path.join('static', footage.video_path)
            
            if not os.path.exists(video_path):
                return enhanced_detections
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return enhanced_detections
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 60th frame for enhanced analysis
                if frame_count % 60 == 0:
                    timestamp = frame_count / fps
                    
                    # Enhanced preprocessing for difficult conditions
                    enhanced_frame = self._enhance_frame_quality(frame)
                    
                    # Multiple detection attempts with different parameters
                    frame_detections = self._detect_with_multiple_parameters(enhanced_frame, timestamp)
                    
                    for detection in frame_detections:
                        detection['source_method'] = 'enhanced_preprocessing'
                        detection['footage_id'] = footage_id
                        detection['case_id'] = self.case_id
                        enhanced_detections.append(detection)
                
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Enhanced detection error: {e}")
        
        return enhanced_detections
    
    def _enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality for better detection
        """
        try:
            # Histogram equalization
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Frame enhancement error: {e}")
            return frame
    
    def _detect_with_multiple_parameters(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        """
        Detect faces with multiple parameter sets for robustness
        """
        detections = []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Multiple detection attempts
            detection_params = [
                {'model': 'hog', 'number_of_times_to_upsample': 1},
                {'model': 'hog', 'number_of_times_to_upsample': 2},
                {'model': 'cnn', 'number_of_times_to_upsample': 0},
            ]
            
            for params in detection_params:
                try:
                    face_locations = face_recognition.face_locations(
                        rgb_frame, 
                        model=params['model'],
                        number_of_times_to_upsample=params.get('number_of_times_to_upsample', 1)
                    )
                    
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            # Calculate confidence against target
                            confidence = self._calculate_enhanced_confidence(face_encoding)
                            
                            if confidence > self.LOW_QUALITY_THRESHOLD:
                                detections.append({
                                    'timestamp': timestamp,
                                    'confidence': confidence,
                                    'bbox': (left, top, right-left, bottom-top),
                                    'method': f"enhanced_{params['model']}",
                                    'face_encoding': face_encoding.tolist()
                                })
                
                except Exception as e:
                    logger.error(f"Detection parameter error: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Multiple parameter detection error: {e}")
        
        return detections
    
    def _calculate_enhanced_confidence(self, face_encoding: np.ndarray) -> float:
        """
        Calculate enhanced confidence with multiple target encodings
        """
        if not self.ultra_detector.target_encodings:
            return 0.0
        
        best_confidence = 0.0
        
        for target_encoding in self.ultra_detector.target_encodings:
            try:
                # Multiple similarity measures
                distance = face_recognition.face_distance([target_encoding], face_encoding)[0]
                euclidean_confidence = max(0, 1.0 - distance)
                
                # Cosine similarity
                from scipy.spatial.distance import cosine
                cosine_sim = 1 - cosine(target_encoding, face_encoding)
                
                # Correlation coefficient
                correlation = np.corrcoef(target_encoding, face_encoding)[0, 1]
                correlation = max(0, correlation)
                
                # Combined confidence
                combined_confidence = (euclidean_confidence * 0.5) + (cosine_sim * 0.3) + (correlation * 0.2)
                best_confidence = max(best_confidence, combined_confidence)
                
            except Exception as e:
                logger.error(f"Enhanced confidence calculation error: {e}")
                continue
        
        return best_confidence
    
    def _merge_duplicate_detections(self, all_detections: List[Dict]) -> List[Dict]:
        """
        Merge duplicate detections from multiple methods
        """
        if not all_detections:
            return []
        
        # Sort by timestamp
        sorted_detections = sorted(all_detections, key=lambda x: x.get('timestamp', 0))
        
        merged = []
        current_group = []
        
        for detection in sorted_detections:
            if not current_group:
                current_group = [detection]
            else:
                # Check if this detection is close in time to current group
                time_diff = abs(detection.get('timestamp', 0) - current_group[-1].get('timestamp', 0))
                
                if time_diff <= 2.0:  # Within 2 seconds
                    current_group.append(detection)
                else:
                    # Process current group and start new one
                    merged_detection = self._merge_detection_group(current_group)
                    if merged_detection:
                        merged.append(merged_detection)
                    current_group = [detection]
        
        # Don't forget the last group
        if current_group:
            merged_detection = self._merge_detection_group(current_group)
            if merged_detection:
                merged.append(merged_detection)
        
        return merged
    
    def _merge_detection_group(self, detection_group: List[Dict]) -> Optional[Dict]:
        """
        Merge a group of similar detections
        """
        if not detection_group:
            return None
        
        if len(detection_group) == 1:
            return detection_group[0]
        
        # Find best detection (highest confidence)
        best_detection = max(detection_group, key=lambda x: x.get('confidence', 0))
        
        # Combine information from all detections
        merged = best_detection.copy()
        merged['source_methods'] = list(set([d.get('source_method', 'unknown') for d in detection_group]))
        merged['detection_count'] = len(detection_group)
        merged['confidence_range'] = {
            'min': min([d.get('confidence', 0) for d in detection_group]),
            'max': max([d.get('confidence', 0) for d in detection_group]),
            'avg': sum([d.get('confidence', 0) for d in detection_group]) / len(detection_group)
        }
        
        return merged
    
    def _perform_fallback_detection(self, footage_id: int) -> List[Dict]:
        """
        Fallback detection method when primary methods fail
        """
        fallback_detections = []
        
        try:
            logger.warning("Using fallback detection method")
            
            # Simple OpenCV cascade detection as last resort
            from models import SurveillanceFootage
            
            footage = SurveillanceFootage.query.get(footage_id)
            video_path = os.path.join('static', footage.video_path)
            
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                
                if cap.isOpened():
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_count % 90 == 0:  # Every 3 seconds
                            timestamp = frame_count / fps
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            
                            for (x, y, w, h) in faces:
                                fallback_detections.append({
                                    'timestamp': timestamp,
                                    'confidence': 0.5,  # Default fallback confidence
                                    'bbox': (x, y, w, h),
                                    'method': 'fallback_cascade',
                                    'source_method': 'fallback_opencv',
                                    'footage_id': footage_id,
                                    'case_id': self.case_id
                                })
                        
                        frame_count += 1
                    
                    cap.release()
        
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
        
        return fallback_detections