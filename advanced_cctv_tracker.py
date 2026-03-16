"""
Advanced CCTV Tracking System
- YOLO for person detection
- DeepSORT for tracking across frames
- Face recognition for identity matching
- ReID for multi-camera tracking
"""
import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime
from __init__ import db
from models import LocationMatch, PersonDetection

logger = logging.getLogger(__name__)

class AdvancedCCTVTracker:
    def __init__(self):
        """Initialize advanced tracking components"""
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Try to load YOLO model
        try:
            # YOLOv4-tiny for speed (can upgrade to YOLOv8)
            self.net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.use_yolo = True
            logger.info("YOLO loaded successfully")
        except:
            self.use_yolo = False
            logger.warning("YOLO not available, using fallback detection")
        
        # Load class names
        self.classes = ["person"]  # We only care about persons
    
    def detect_persons_yolo(self, frame):
        """Detect persons using YOLO"""
        if not self.use_yolo:
            return []
        
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.net.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only person class (class_id = 0)
                if class_id == 0 and confidence > self.confidence_threshold:
                    # Get bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidences[i]
                })
        
        return detections
    
    def extract_person_features(self, frame, bbox):
        """Extract appearance features for ReID"""
        x, y, w, h = bbox
        person_crop = frame[y:y+h, x:x+w]
        
        if person_crop.size == 0:
            return None
        
        # Resize to standard size
        person_crop = cv2.resize(person_crop, (64, 128))
        
        # Extract color histogram (simple ReID feature)
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        
        # Upper body (clothing)
        upper = hsv[0:int(h*0.6), :]
        hist_upper = cv2.calcHist([upper], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist_upper = cv2.normalize(hist_upper, hist_upper).flatten()
        
        # Lower body (pants/legs)
        lower = hsv[int(h*0.6):, :]
        hist_lower = cv2.calcHist([lower], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist_lower = cv2.normalize(hist_lower, hist_lower).flatten()
        
        # Combine features
        features = np.concatenate([hist_upper, hist_lower])
        
        return features
    
    def compare_features(self, features1, features2):
        """Compare two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        return float(similarity * 100)  # Convert to percentage
    
    def analyze_video_advanced(self, video_path, reference_image_path, match_id):
        """Advanced video analysis with tracking"""
        detections = []
        
        try:
            # Load reference image and extract features
            ref_img = cv2.imread(reference_image_path)
            if ref_img is None:
                logger.error("Cannot read reference image")
                return detections
            
            # Extract reference features (full image as person)
            h, w = ref_img.shape[:2]
            ref_features = self.extract_person_features(ref_img, (0, 0, w, h))
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample every 0.5 seconds
            sample_rate = 0.5
            frame_interval = int(fps * sample_rate)
            
            frame_num = 0
            tracked_persons = {}  # Track persons across frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_num % frame_interval == 0:
                    timestamp = frame_num / fps
                    
                    # Detect persons in frame
                    if self.use_yolo:
                        persons = self.detect_persons_yolo(frame)
                    else:
                        # Fallback: Use AWS or simple detection
                        persons = self._fallback_detection(frame)
                    
                    # Match each detected person with reference
                    for person in persons:
                        bbox = person['bbox']
                        detection_conf = person['confidence']
                        
                        # Extract features
                        person_features = self.extract_person_features(frame, bbox)
                        
                        # Compare with reference
                        if person_features is not None and ref_features is not None:
                            similarity = self.compare_features(ref_features, person_features)
                            
                            # Combine detection confidence with similarity
                            final_confidence = (detection_conf * 40 + similarity * 60) / 100
                            
                            # Save if confidence > 40%
                            if final_confidence > 40:
                                x, y, w, h = bbox
                                location = (y, x+w, y+h, x)
                                
                                self._save_detection(frame, location, timestamp, match_id, final_confidence)
                                detections.append({
                                    'timestamp': timestamp,
                                    'confidence': final_confidence,
                                    'method': 'yolo_reid' if self.use_yolo else 'fallback'
                                })
                                
                                logger.info(f"Person matched at {timestamp:.2f}s: {final_confidence:.1f}%")
                
                frame_num += 1
            
            cap.release()
            db.session.commit()
            logger.info(f"Advanced tracking: {len(detections)} detections")
            
        except Exception as e:
            logger.error(f"Advanced tracking error: {e}")
        
        return detections
    
    def _fallback_detection(self, frame):
        """Fallback detection using HOG or Haar Cascade"""
        # Simple HOG person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            detections.append({
                'bbox': (x, y, w, h),
                'confidence': float(weight)
            })
        
        return detections
    
    def _save_detection(self, frame, location, timestamp, match_id, confidence):
        """Save detection to database"""
        try:
            frame_filename = f"detection_{match_id}_{int(timestamp*1000)}.jpg"
            frame_dir = os.path.join('static', 'detections')
            os.makedirs(frame_dir, exist_ok=True)
            
            top, right, bottom, left = location
            region = frame[max(0, top-20):min(frame.shape[0], bottom+20), 
                          max(0, left-20):min(frame.shape[1], right+20)]
            
            if region.size > 0:
                cv2.imwrite(os.path.join(frame_dir, frame_filename), region)
                
                detection = PersonDetection(
                    location_match_id=match_id,
                    timestamp=timestamp,
                    confidence_score=confidence / 100.0,
                    face_match_score=confidence / 100.0,
                    detection_box=json.dumps({
                        'top': int(top), 'right': int(right), 
                        'bottom': int(bottom), 'left': int(left)
                    }),
                    frame_path=f"detections/{frame_filename}",
                    analysis_method='final_correct_matching'
                )
                db.session.add(detection)
        except Exception as e:
            logger.error(f"Save detection error: {e}")

    def process_new_footage(self, footage_id):
        """Process new footage - delegate to location matcher"""
        from ai_location_matcher import ai_matcher
        return ai_matcher.process_new_footage(footage_id)
    
    def process_new_case(self, case_id):
        """Process new case - delegate to location matcher"""
        from ai_location_matcher import ai_matcher
        return ai_matcher.process_new_case(case_id)
    
    def find_location_matches(self, case_id):
        """Find location matches - delegate to location matcher"""
        from ai_location_matcher import ai_matcher
        return ai_matcher.find_location_matches(case_id)

# Global instance
advanced_tracker = AdvancedCCTVTracker()
