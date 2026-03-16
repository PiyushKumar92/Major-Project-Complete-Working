"""
Advanced CCTV Person Matching System
Optimized for detecting persons in crowded CCTV footage with multiple faces
"""

import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
import json

class AdvancedCCTVMatcher:
    def __init__(self):
        self.min_face_size = (30, 30)  # Minimum face size in pixels
        self.max_face_size = (300, 300)  # Maximum face size for processing
        self.confidence_threshold = 0.4  # Lower threshold for distant/small faces
        self.multi_scale_factors = [1.0, 1.2, 1.5]  # Multiple scales for detection
        
    def extract_reference_encodings(self, case):
        """Extract high-quality face encodings from case photos"""
        reference_encodings = []
        encoding_metadata = []
        
        for img in case.target_images:
            img_path = os.path.join('app', 'static', img.image_path.replace('static/', ''))
            if not os.path.exists(img_path):
                continue
            
            try:
                # Load and preprocess image
                image = face_recognition.load_image_file(img_path)
                
                # Multiple detection methods for better accuracy
                face_locations = []
                
                # Method 1: HOG (faster, good for clear faces)
                hog_locations = face_recognition.face_locations(image, model="hog")
                face_locations.extend(hog_locations)
                
                # Method 2: CNN (slower, better for difficult faces)
                if not face_locations:
                    try:
                        cnn_locations = face_recognition.face_locations(image, model="cnn")
                        face_locations.extend(cnn_locations)
                    except:
                        pass
                
                # Method 3: OpenCV Haar Cascades (fallback)
                if not face_locations:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Convert OpenCV format to face_recognition format
                    for (x, y, w, h) in faces:
                        face_locations.append((y, x + w, y + h, x))
                
                if face_locations:
                    # Get encodings for all detected faces
                    encodings = face_recognition.face_encodings(image, face_locations)
                    
                    for i, encoding in enumerate(encodings):
                        if len(encoding) == 128:  # Valid encoding
                            top, right, bottom, left = face_locations[i]
                            face_area = (bottom - top) * (right - left)
                            
                            # Store encoding with metadata
                            reference_encodings.append(encoding)
                            encoding_metadata.append({
                                'source_image': img.image_path,
                                'face_area': face_area,
                                'location': face_locations[i],
                                'quality_score': self._calculate_face_quality(image, face_locations[i])
                            })
            
            except Exception as e:
                print(f"Error processing reference image {img.image_path}: {str(e)}")
                continue
        
        return reference_encodings, encoding_metadata
    
    def _calculate_face_quality(self, image, face_location):
        """Calculate face quality score for prioritizing encodings"""
        top, right, bottom, left = face_location
        face_img = image[top:bottom, left:right]
        
        if face_img.size == 0:
            return 0.0
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 1000, 1.0)
        
        # 2. Size score
        face_area = (bottom - top) * (right - left)
        size_score = min(face_area / (100 * 100), 1.0)  # Optimal at 100x100
        
        # 3. Brightness score
        brightness = np.mean(gray_face)
        brightness_score = 1.0 - abs(brightness - 127) / 127
        
        # 4. Contrast score
        contrast = gray_face.std()
        contrast_score = min(contrast / 50, 1.0)
        
        return (sharpness_score * 0.4 + size_score * 0.3 + brightness_score * 0.15 + contrast_score * 0.15)
    
    def analyze_cctv_footage(self, video_path, reference_encodings, encoding_metadata):
        """Analyze CCTV footage for person detection with crowd handling"""
        detections = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return detections
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Adaptive sampling based on video length
            if duration > 300:  # > 5 minutes
                sample_interval = max(int(fps * 2), 30)  # Every 2 seconds
            elif duration > 60:  # > 1 minute
                sample_interval = max(int(fps), 15)  # Every second
            else:
                sample_interval = max(int(fps * 0.5), 5)  # Every 0.5 seconds
            
            frame_number = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on sampling interval
                if frame_number % sample_interval != 0:
                    frame_number += 1
                    continue
                
                timestamp = frame_number / fps if fps > 0 else frame_number
                
                # Process frame for person detection
                frame_detections = self._process_frame_for_crowd(
                    frame, reference_encodings, encoding_metadata, timestamp
                )
                
                detections.extend(frame_detections)
                processed_frames += 1
                frame_number += 1
                
                # Limit processing for very long videos
                if processed_frames > 1000:  # Max 1000 frames
                    break
            
            cap.release()
            
        except Exception as e:
            print(f"Error analyzing CCTV footage: {str(e)}")
        
        # Post-process detections
        return self._post_process_detections(detections)
    
    def _process_frame_for_crowd(self, frame, reference_encodings, encoding_metadata, timestamp):
        """Process single frame optimized for crowded scenes"""
        detections = []
        
        try:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]
            
            # Multi-scale detection for better crowd handling
            all_face_locations = []
            all_face_encodings = []
            
            for scale in self.multi_scale_factors:
                if scale != 1.0:
                    # Resize frame for different scales
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    scaled_frame = cv2.resize(rgb_frame, (new_width, new_height))
                else:
                    scaled_frame = rgb_frame
                
                # Detect faces at this scale
                try:
                    # Use HOG for speed in crowded scenes
                    face_locations = face_recognition.face_locations(
                        scaled_frame, 
                        number_of_times_to_upsample=0,  # Faster processing
                        model="hog"
                    )
                    
                    if face_locations:
                        # Scale back coordinates if needed
                        if scale != 1.0:
                            face_locations = [
                                (int(top/scale), int(right/scale), int(bottom/scale), int(left/scale))
                                for top, right, bottom, left in face_locations
                            ]
                        
                        # Filter faces by size (remove too small/large faces)
                        filtered_locations = []
                        for top, right, bottom, left in face_locations:
                            face_width = right - left
                            face_height = bottom - top
                            
                            if (self.min_face_size[0] <= face_width <= self.max_face_size[0] and
                                self.min_face_size[1] <= face_height <= self.max_face_size[1]):
                                filtered_locations.append((top, right, bottom, left))
                        
                        all_face_locations.extend(filtered_locations)
                
                except Exception as e:
                    print(f"Face detection error at scale {scale}: {str(e)}")
                    continue
            
            # Remove duplicate detections (faces detected at multiple scales)
            unique_locations = self._remove_duplicate_faces(all_face_locations)
            
            if not unique_locations:
                return detections
            
            # Limit number of faces processed per frame (performance optimization)
            if len(unique_locations) > 20:  # Max 20 faces per frame
                # Sort by face size (larger faces first)
                unique_locations.sort(key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]), reverse=True)
                unique_locations = unique_locations[:20]
            
            # Get encodings for detected faces
            try:
                face_encodings = face_recognition.face_encodings(rgb_frame, unique_locations)
            except Exception as e:
                print(f"Face encoding error: {str(e)}")
                return detections
            
            # Match against reference encodings
            for i, face_encoding in enumerate(face_encodings):
                if i >= len(unique_locations):
                    break
                
                face_location = unique_locations[i]
                
                # Calculate distances to all reference encodings
                distances = face_recognition.face_distance(reference_encodings, face_encoding)
                
                if len(distances) > 0:
                    min_distance = np.min(distances)
                    best_match_idx = np.argmin(distances)
                    
                    # Convert distance to confidence (lower distance = higher confidence)
                    confidence = max(0.0, 1.0 - min_distance)
                    
                    # Adjust confidence based on reference encoding quality
                    if best_match_idx < len(encoding_metadata):
                        ref_quality = encoding_metadata[best_match_idx]['quality_score']
                        confidence = confidence * (0.7 + 0.3 * ref_quality)  # Boost for high-quality references
                    
                    # Check if confidence meets threshold
                    if confidence >= self.confidence_threshold:
                        top, right, bottom, left = face_location
                        
                        # Calculate additional metrics
                        face_area = (bottom - top) * (right - left)
                        frame_area = frame_height * frame_width
                        size_ratio = face_area / frame_area
                        
                        # Distance from center (people in center often more important)
                        center_x, center_y = frame_width // 2, frame_height // 2
                        face_center_x = (left + right) // 2
                        face_center_y = (top + bottom) // 2
                        distance_from_center = np.sqrt(
                            (face_center_x - center_x)**2 + (face_center_y - center_y)**2
                        ) / np.sqrt(center_x**2 + center_y**2)
                        
                        detection = {
                            'timestamp': timestamp,
                            'confidence': confidence,
                            'face_location': face_location,
                            'face_area': face_area,
                            'size_ratio': size_ratio,
                            'distance_from_center': distance_from_center,
                            'reference_match_idx': best_match_idx,
                            'min_distance': min_distance,
                            'frame_dimensions': (frame_width, frame_height)
                        }
                        
                        detections.append(detection)
        
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
        
        return detections
    
    def _remove_duplicate_faces(self, face_locations, overlap_threshold=0.3):
        """Remove duplicate face detections with overlap threshold"""
        if not face_locations:
            return []
        
        unique_faces = []
        
        for face in face_locations:
            top, right, bottom, left = face
            is_duplicate = False
            
            for existing_face in unique_faces:
                ex_top, ex_right, ex_bottom, ex_left = existing_face
                
                # Calculate overlap
                overlap_left = max(left, ex_left)
                overlap_right = min(right, ex_right)
                overlap_top = max(top, ex_top)
                overlap_bottom = min(bottom, ex_bottom)
                
                if overlap_left < overlap_right and overlap_top < overlap_bottom:
                    overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
                    face_area = (right - left) * (bottom - top)
                    existing_area = (ex_right - ex_left) * (ex_bottom - ex_top)
                    
                    overlap_ratio = overlap_area / min(face_area, existing_area)
                    
                    if overlap_ratio > overlap_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _post_process_detections(self, detections):
        """Post-process detections to remove false positives and improve accuracy"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Group detections by time proximity (within 2 seconds)
        grouped_detections = []
        current_group = []
        
        for detection in detections:
            if not current_group:
                current_group.append(detection)
            else:
                time_diff = abs(detection['timestamp'] - current_group[-1]['timestamp'])
                if time_diff <= 2.0:  # Within 2 seconds
                    current_group.append(detection)
                else:
                    if current_group:
                        grouped_detections.append(current_group)
                    current_group = [detection]
        
        if current_group:
            grouped_detections.append(current_group)
        
        # Process each group
        final_detections = []
        
        for group in grouped_detections:
            if len(group) == 1:
                final_detections.append(group[0])
            else:
                # Multiple detections in short time - take the best one
                best_detection = max(group, key=lambda x: x['confidence'])
                
                # Add temporal consistency score
                avg_confidence = sum(d['confidence'] for d in group) / len(group)
                best_detection['temporal_consistency'] = len(group)
                best_detection['avg_group_confidence'] = avg_confidence
                
                final_detections.append(best_detection)
        
        # Final filtering based on quality metrics
        high_quality_detections = []
        
        for detection in final_detections:
            quality_score = self._calculate_detection_quality(detection)
            detection['quality_score'] = quality_score
            
            # Keep high-quality detections
            if quality_score > 0.3:  # Minimum quality threshold
                high_quality_detections.append(detection)
        
        return high_quality_detections
    
    def _calculate_detection_quality(self, detection):
        """Calculate overall quality score for a detection"""
        confidence = detection['confidence']
        size_ratio = detection['size_ratio']
        distance_from_center = detection['distance_from_center']
        
        # Base score from confidence
        quality = confidence * 0.6
        
        # Size bonus (faces that are not too small or too large)
        if 0.001 <= size_ratio <= 0.05:  # Good size range
            quality += 0.2
        elif 0.0005 <= size_ratio <= 0.1:  # Acceptable range
            quality += 0.1
        
        # Center proximity bonus (slight preference for centered faces)
        center_bonus = (1.0 - distance_from_center) * 0.1
        quality += center_bonus
        
        # Temporal consistency bonus
        if 'temporal_consistency' in detection and detection['temporal_consistency'] > 1:
            consistency_bonus = min(detection['temporal_consistency'] * 0.05, 0.1)
            quality += consistency_bonus
        
        return min(quality, 1.0)
    
    def generate_detection_report(self, detections, case_info):
        """Generate comprehensive detection report"""
        if not detections:
            return {
                'total_detections': 0,
                'high_confidence_detections': 0,
                'detection_summary': 'No person detected in the footage',
                'recommendations': ['Upload clearer reference photos', 'Check if person appears in the footage']
            }
        
        high_confidence = [d for d in detections if d['confidence'] > 0.7]
        medium_confidence = [d for d in detections if 0.5 <= d['confidence'] <= 0.7]
        low_confidence = [d for d in detections if d['confidence'] < 0.5]
        
        # Time analysis
        timestamps = [d['timestamp'] for d in detections]
        if timestamps:
            first_detection = min(timestamps)
            last_detection = max(timestamps)
            detection_span = last_detection - first_detection
        else:
            first_detection = last_detection = detection_span = 0
        
        report = {
            'total_detections': len(detections),
            'high_confidence_detections': len(high_confidence),
            'medium_confidence_detections': len(medium_confidence),
            'low_confidence_detections': len(low_confidence),
            'detection_timespan': detection_span,
            'first_detection_time': first_detection,
            'last_detection_time': last_detection,
            'average_confidence': sum(d['confidence'] for d in detections) / len(detections),
            'best_detection': max(detections, key=lambda x: x['confidence']) if detections else None,
            'detection_summary': self._generate_summary(detections),
            'recommendations': self._generate_recommendations(detections, case_info)
        }
        
        return report
    
    def _generate_summary(self, detections):
        """Generate human-readable summary of detections"""
        if not detections:
            return "No person detected in the footage"
        
        high_conf = len([d for d in detections if d['confidence'] > 0.7])
        total = len(detections)
        
        if high_conf > 0:
            return f"Person detected {total} times with {high_conf} high-confidence matches"
        elif total > 5:
            return f"Person possibly detected {total} times with moderate confidence"
        else:
            return f"Person detected {total} times with low to moderate confidence"
    
    def _generate_recommendations(self, detections, case_info):
        """Generate recommendations based on detection results"""
        recommendations = []
        
        if not detections:
            recommendations.extend([
                "Person not found in this footage",
                "Try uploading clearer reference photos",
                "Check if the person actually appears in this video",
                "Consider uploading additional footage from the same location"
            ])
        else:
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
            
            if avg_confidence < 0.5:
                recommendations.extend([
                    "Low confidence detections - results may not be reliable",
                    "Upload higher quality reference photos for better matching",
                    "Ensure reference photos show clear frontal face views"
                ])
            
            if len(detections) < 3:
                recommendations.append("Few detections found - person may appear briefly in footage")
            
            # Check face sizes in detections
            small_faces = [d for d in detections if d['size_ratio'] < 0.001]
            if len(small_faces) > len(detections) * 0.5:
                recommendations.append("Many detections show small/distant faces - results may be less accurate")
        
        return recommendations

# Global matcher instance
cctv_matcher = AdvancedCCTVMatcher()