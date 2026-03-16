import cv2
import numpy as np
import os
from PIL import Image
import face_recognition
import json
from datetime import datetime

class AIProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def analyze_uploaded_photos(self, case_id, photo_paths):
        """Analyze all uploaded photos and extract facial features"""
        analysis_results = {
            'case_id': case_id,
            'photos_analyzed': [],
            'face_encodings': [],
            'quality_scores': [],
            'best_photo': None,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        best_quality = 0
        
        for i, photo_path in enumerate(photo_paths):
            try:
                # Load and analyze photo
                image = face_recognition.load_image_file(photo_path)
                
                # Get face locations and encodings
                face_locations = face_recognition.face_locations(image, model="hog")
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if face_encodings:
                    # Calculate quality score
                    quality_score = self._calculate_photo_quality(image, face_locations[0])
                    
                    photo_analysis = {
                        'photo_index': i,
                        'photo_path': photo_path,
                        'faces_detected': len(face_encodings),
                        'face_encoding': face_encodings[0].tolist(),
                        'face_location': face_locations[0],
                        'quality_score': quality_score,
                        'image_size': image.shape[:2]
                    }
                    
                    analysis_results['photos_analyzed'].append(photo_analysis)
                    analysis_results['face_encodings'].append(face_encodings[0].tolist())
                    analysis_results['quality_scores'].append(quality_score)
                    
                    # Track best quality photo
                    if quality_score > best_quality:
                        best_quality = quality_score
                        analysis_results['best_photo'] = photo_analysis
                        
            except Exception as e:
                print(f"Error analyzing photo {photo_path}: {str(e)}")
                
        return analysis_results
    
    def analyze_uploaded_videos(self, case_id, video_paths):
        """Extract key frames and analyze videos for facial features"""
        video_analysis = {
            'case_id': case_id,
            'videos_analyzed': [],
            'extracted_frames': [],
            'face_encodings_from_video': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        for video_path in video_paths:
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Extract frames at regular intervals
                frame_interval = max(1, frame_count // 10)  # Extract 10 frames max
                extracted_frames = []
                
                for frame_num in range(0, frame_count, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        if face_encodings:
                            timestamp = frame_num / fps
                            frame_analysis = {
                                'frame_number': frame_num,
                                'timestamp': timestamp,
                                'face_encoding': face_encodings[0].tolist(),
                                'face_location': face_locations[0],
                                'quality_score': self._calculate_frame_quality(rgb_frame, face_locations[0])
                            }
                            extracted_frames.append(frame_analysis)
                            video_analysis['face_encodings_from_video'].append(face_encodings[0].tolist())
                
                cap.release()
                
                video_info = {
                    'video_path': video_path,
                    'total_frames': frame_count,
                    'fps': fps,
                    'duration': frame_count / fps,
                    'frames_extracted': len(extracted_frames),
                    'faces_found': len(extracted_frames)
                }
                
                video_analysis['videos_analyzed'].append(video_info)
                video_analysis['extracted_frames'].extend(extracted_frames)
                
            except Exception as e:
                print(f"Error analyzing video {video_path}: {str(e)}")
                
        return video_analysis
    
    def create_face_profile(self, case_id, photo_analysis, video_analysis):
        """Create comprehensive face profile for the missing person"""
        face_profile = {
            'case_id': case_id,
            'created_at': datetime.now().isoformat(),
            'primary_encoding': None,
            'all_encodings': [],
            'confidence_threshold': 0.6,
            'quality_metrics': {
                'best_photo_quality': 0,
                'average_quality': 0,
                'total_samples': 0
            }
        }
        
        # Collect all face encodings
        all_encodings = []
        all_qualities = []
        
        # From photos
        if photo_analysis and photo_analysis.get('face_encodings'):
            all_encodings.extend(photo_analysis['face_encodings'])
            all_qualities.extend(photo_analysis['quality_scores'])
            
        # From videos
        if video_analysis and video_analysis.get('face_encodings_from_video'):
            all_encodings.extend(video_analysis['face_encodings_from_video'])
            # Add quality scores for video frames
            for frame in video_analysis.get('extracted_frames', []):
                all_qualities.append(frame.get('quality_score', 0.5))
        
        if all_encodings:
            # Calculate average encoding as primary
            face_profile['primary_encoding'] = np.mean(all_encodings, axis=0).tolist()
            face_profile['all_encodings'] = all_encodings
            
            # Calculate quality metrics
            face_profile['quality_metrics'] = {
                'best_photo_quality': max(all_qualities) if all_qualities else 0,
                'average_quality': np.mean(all_qualities) if all_qualities else 0,
                'total_samples': len(all_encodings)
            }
            
            # Adjust confidence threshold based on quality
            avg_quality = face_profile['quality_metrics']['average_quality']
            if avg_quality > 0.8:
                face_profile['confidence_threshold'] = 0.5  # Lower threshold for high quality
            elif avg_quality > 0.6:
                face_profile['confidence_threshold'] = 0.6  # Standard threshold
            else:
                face_profile['confidence_threshold'] = 0.7  # Higher threshold for low quality
        
        return face_profile
    
    def _calculate_photo_quality(self, image, face_location):
        """Calculate quality score for a photo"""
        try:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            
            # Convert to grayscale for analysis
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray_face)
            
            # Calculate contrast
            contrast = gray_face.std()
            
            # Face size score
            face_area = (bottom - top) * (right - left)
            size_score = min(face_area / 10000, 1.0)  # Normalize to 0-1
            
            # Combine scores
            quality_score = (
                min(sharpness / 1000, 1.0) * 0.4 +  # Sharpness weight
                min(brightness / 255, 1.0) * 0.2 +   # Brightness weight
                min(contrast / 100, 1.0) * 0.2 +     # Contrast weight
                size_score * 0.2                      # Size weight
            )
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.5  # Default quality score
    
    def _calculate_frame_quality(self, frame, face_location):
        """Calculate quality score for a video frame"""
        return self._calculate_photo_quality(frame, face_location)
    
    def save_analysis_results(self, case_id, photo_analysis, video_analysis, face_profile):
        """Save all analysis results to files"""
        try:
            # Create analysis directory
            analysis_dir = os.path.join('app', 'static', 'analysis', str(case_id))
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Save photo analysis
            with open(os.path.join(analysis_dir, 'photo_analysis.json'), 'w') as f:
                json.dump(photo_analysis, f, indent=2)
            
            # Save video analysis
            with open(os.path.join(analysis_dir, 'video_analysis.json'), 'w') as f:
                json.dump(video_analysis, f, indent=2)
            
            # Save face profile
            with open(os.path.join(analysis_dir, 'face_profile.json'), 'w') as f:
                json.dump(face_profile, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")
            return False

def process_case_media(case_id, photo_paths, video_paths):
    """Main function to process all media for a case"""
    processor = AIProcessor()
    
    # Analyze photos
    photo_analysis = processor.analyze_uploaded_photos(case_id, photo_paths)
    
    # Analyze videos
    video_analysis = processor.analyze_uploaded_videos(case_id, video_paths)
    
    # Create face profile
    face_profile = processor.create_face_profile(case_id, photo_analysis, video_analysis)
    
    # Save results
    processor.save_analysis_results(case_id, photo_analysis, video_analysis, face_profile)
    
    return {
        'photo_analysis': photo_analysis,
        'video_analysis': video_analysis,
        'face_profile': face_profile,
        'status': 'completed'
    }