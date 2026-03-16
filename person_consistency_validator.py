"""
Person Consistency Validator
Validates that all uploaded photos and videos contain the same person
"""

import cv2
import numpy as np
import face_recognition
import os
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

class PersonConsistencyValidator:
    def __init__(self):
        self.face_encodings = []
        self.validation_threshold = 0.45  # More lenient for real-world scenarios
        self.min_face_confidence = 0.4  # Lower threshold for better detection
        
    def extract_face_encodings(self, image_path: str) -> List[np.ndarray]:
        """Extract face encodings from an image"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations with multiple methods
            face_locations = face_recognition.face_locations(image, model="hog")
            
            # If no faces found, try CNN model (more accurate but slower)
            if not face_locations:
                try:
                    face_locations = face_recognition.face_locations(image, model="cnn")
                except:
                    pass
            
            if not face_locations:
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            return face_encodings
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return []
    
    def extract_faces_from_video(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract face encodings from video frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
            
            all_encodings = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces with multiple attempts
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                # Try CNN if HOG fails
                if not face_locations:
                    try:
                        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
                    except:
                        pass
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    all_encodings.extend(face_encodings)
            
            cap.release()
            return all_encodings
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return []
    
    def validate_person_consistency(self, image_paths: List[str], video_paths: List[str] = None) -> Dict:
        """
        Validate that all images and videos contain the same person
        Returns validation results with detailed analysis
        """
        validation_result = {
            'is_consistent': False,
            'confidence_score': 0.0,
            'total_faces_found': 0,
            'consistent_faces': 0,
            'inconsistent_files': [],
            'primary_person_encoding': None,
            'detailed_analysis': {},
            'recommendations': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        if not image_paths:
            validation_result['recommendations'].append("At least one photo is required for validation")
            return validation_result
        
        # Step 1: Extract face encodings from all images
        all_face_data = []
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                validation_result['inconsistent_files'].append({
                    'file': img_path,
                    'issue': 'File not found'
                })
                continue
                
            encodings = self.extract_face_encodings(img_path)
            
            if not encodings:
                validation_result['inconsistent_files'].append({
                    'file': img_path,
                    'issue': 'No face detected'
                })
                validation_result['recommendations'].append(f"No face found in {os.path.basename(img_path)} - please upload a clear photo")
                continue
            
            if len(encodings) > 1:
                validation_result['recommendations'].append(f"Multiple faces detected in {os.path.basename(img_path)} - please upload photos with only the target person")
            
            # Use the first (most prominent) face
            all_face_data.append({
                'file': img_path,
                'type': 'image',
                'encoding': encodings[0],
                'face_count': len(encodings)
            })
        
        # Step 2: Extract face encodings from videos (if provided)
        if video_paths:
            for vid_path in video_paths:
                if not os.path.exists(vid_path):
                    validation_result['inconsistent_files'].append({
                        'file': vid_path,
                        'issue': 'File not found'
                    })
                    continue
                
                encodings = self.extract_faces_from_video(vid_path)
                
                if not encodings:
                    validation_result['inconsistent_files'].append({
                        'file': vid_path,
                        'issue': 'No face detected in video'
                    })
                    validation_result['recommendations'].append(f"No clear face found in {os.path.basename(vid_path)} - please upload video with clear view of the person")
                    continue
                
                # Use the best encoding (most common face)
                if encodings:
                    # Find the most consistent face across video frames
                    best_encoding = self._find_most_consistent_face(encodings)
                    all_face_data.append({
                        'file': vid_path,
                        'type': 'video',
                        'encoding': best_encoding,
                        'face_count': len(encodings)
                    })
        
        validation_result['total_faces_found'] = len(all_face_data)
        
        if len(all_face_data) < 1:
            # No faces detected - pass validation with warning
            validation_result['is_consistent'] = True
            validation_result['confidence_score'] = 0.5
            validation_result['recommendations'].append("No faces detected - manual review recommended")
            return validation_result
        
        # Step 3: Compare all faces for consistency
        if len(all_face_data) == 1:
            # Only one file with face - consider it consistent
            validation_result['is_consistent'] = True
            validation_result['confidence_score'] = 1.0
            validation_result['consistent_faces'] = 1
            validation_result['primary_person_encoding'] = all_face_data[0]['encoding'].tolist()
        else:
            # Multiple files - check consistency
            consistency_results = self._check_face_consistency(all_face_data)
            validation_result.update(consistency_results)
        
        # Step 4: Generate detailed analysis
        validation_result['detailed_analysis'] = self._generate_detailed_analysis(all_face_data, validation_result)
        
        return validation_result
    
    def _find_most_consistent_face(self, encodings: List[np.ndarray]) -> np.ndarray:
        """Find the most consistent face encoding from a list"""
        if len(encodings) == 1:
            return encodings[0]
        
        # Compare all encodings and find the most central one
        similarity_scores = []
        
        for i, encoding in enumerate(encodings):
            scores = []
            for j, other_encoding in enumerate(encodings):
                if i != j:
                    distance = face_recognition.face_distance([encoding], other_encoding)[0]
                    scores.append(1 - distance)  # Convert distance to similarity
            
            avg_similarity = np.mean(scores) if scores else 0
            similarity_scores.append(avg_similarity)
        
        # Return the encoding with highest average similarity
        best_idx = np.argmax(similarity_scores)
        return encodings[best_idx]
    
    def _check_face_consistency(self, face_data: List[Dict]) -> Dict:
        """Check consistency between multiple faces"""
        if len(face_data) < 2:
            return {
                'is_consistent': True,
                'confidence_score': 1.0,
                'consistent_faces': len(face_data),
                'primary_person_encoding': face_data[0]['encoding'].tolist() if face_data else None
            }
        
        # Use first face as reference
        reference_encoding = face_data[0]['encoding']
        consistent_count = 1  # Reference is always consistent with itself
        inconsistent_files = []
        
        similarity_scores = []
        
        for i in range(1, len(face_data)):
            current_encoding = face_data[i]['encoding']
            
            # Calculate similarity
            distance = face_recognition.face_distance([reference_encoding], current_encoding)[0]
            similarity = 1 - distance
            similarity_scores.append(similarity)
            
            if similarity >= self.validation_threshold:
                consistent_count += 1
            else:
                inconsistent_files.append({
                    'file': face_data[i]['file'],
                    'issue': f'Different person detected (similarity: {similarity:.2f})',
                    'similarity_score': similarity
                })
        
        # Calculate overall confidence
        if similarity_scores:
            avg_similarity = np.mean(similarity_scores)
            confidence_score = min(1.0, avg_similarity * 1.2)  # Boost confidence slightly
        else:
            confidence_score = 1.0
        
        is_consistent = consistent_count == len(face_data)
        
        return {
            'is_consistent': is_consistent,
            'confidence_score': confidence_score,
            'consistent_faces': consistent_count,
            'inconsistent_files': inconsistent_files,
            'primary_person_encoding': reference_encoding.tolist()
        }
    
    def _generate_detailed_analysis(self, face_data: List[Dict], validation_result: Dict) -> Dict:
        """Generate detailed analysis report"""
        analysis = {
            'total_files_processed': len(face_data),
            'images_processed': len([f for f in face_data if f['type'] == 'image']),
            'videos_processed': len([f for f in face_data if f['type'] == 'video']),
            'face_quality_assessment': [],
            'consistency_matrix': [],
            'recommendations_detailed': []
        }
        
        # Assess face quality for each file
        for face_info in face_data:
            quality_score = self._assess_face_quality(face_info)
            analysis['face_quality_assessment'].append({
                'file': os.path.basename(face_info['file']),
                'type': face_info['type'],
                'quality_score': quality_score,
                'face_count': face_info['face_count']
            })
        
        # Generate recommendations based on analysis
        if validation_result['confidence_score'] < 0.7:
            analysis['recommendations_detailed'].append("Low confidence detected - consider uploading clearer photos")
        
        if len([f for f in face_data if f['face_count'] > 1]) > 0:
            analysis['recommendations_detailed'].append("Multiple faces detected in some files - use photos with only the target person")
        
        if validation_result['total_faces_found'] < 3:
            analysis['recommendations_detailed'].append("Upload more photos from different angles for better AI analysis")
        
        return analysis
    
    def _assess_face_quality(self, face_info: Dict) -> float:
        """Assess the quality of detected face"""
        # Simple quality assessment based on face count and type
        base_score = 0.8
        
        # Penalize multiple faces
        if face_info['face_count'] > 1:
            base_score -= 0.2
        
        # Bonus for videos (more data)
        if face_info['type'] == 'video':
            base_score += 0.1
        
        return min(1.0, max(0.0, base_score))

def validate_case_person_consistency(case_id: int, image_paths: List[str], video_paths: List[str] = None) -> Dict:
    """
    Main function to validate person consistency for a case
    """
    validator = PersonConsistencyValidator()
    
    # Convert relative paths to absolute paths
    abs_image_paths = []
    for img_path in image_paths:
        if not os.path.isabs(img_path):
            abs_path = os.path.join('static', img_path)
        else:
            abs_path = img_path
        abs_image_paths.append(abs_path)
    
    abs_video_paths = []
    if video_paths:
        for vid_path in video_paths:
            if not os.path.isabs(vid_path):
                abs_path = os.path.join('static', vid_path)
            else:
                abs_path = vid_path
            abs_video_paths.append(abs_path)
    
    # Perform validation
    result = validator.validate_person_consistency(abs_image_paths, abs_video_paths)
    result['case_id'] = case_id
    
    return result

if __name__ == "__main__":
    # Test the validator
    print("Person Consistency Validator - Test Mode")
    
    # Example usage
    test_images = ["test1.jpg", "test2.jpg"]
    test_videos = ["test_video.mp4"]
    
    result = validate_case_person_consistency(1, test_images, test_videos)
    print(json.dumps(result, indent=2, default=str))