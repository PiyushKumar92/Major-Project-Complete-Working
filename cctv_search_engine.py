#!/usr/bin/env python3
"""
CCTV Search Engine - Core functionality to find persons in CCTV footage
"""

import cv2
import numpy as np
import face_recognition
import json
import os
from datetime import datetime
from ai_processor import AIProcessor

class CCTVSearchEngine:
    def __init__(self):
        self.ai_processor = AIProcessor()
        self.confidence_threshold = 0.6
        
    def search_person_in_cctv(self, case_id, target_face_encodings, cctv_video_path):
        """
        Main function to search for a person in CCTV footage
        """
        search_results = {
            'case_id': case_id,
            'cctv_video': cctv_video_path,
            'search_timestamp': datetime.now().isoformat(),
            'matches_found': [],
            'total_frames_processed': 0,
            'faces_detected': 0,
            'matches_count': 0,
            'confidence_scores': [],
            'best_match': None
        }
        
        try:
            cap = cv2.VideoCapture(cctv_video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Process every 30th frame for efficiency (1 frame per second if 30fps)
            frame_interval = max(1, int(fps))
            
            best_confidence = 0
            
            for frame_num in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                    
                search_results['total_frames_processed'] += 1
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in current frame
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                if not face_locations:
                    continue
                    
                search_results['faces_detected'] += len(face_locations)
                
                # Get face encodings for detected faces
                frame_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Compare with target person's face encodings
                for i, frame_face_encoding in enumerate(frame_face_encodings):
                    # Compare with all target encodings
                    distances = face_recognition.face_distance(target_face_encodings, frame_face_encoding)
                    min_distance = np.min(distances)
                    confidence = 1 - min_distance  # Convert distance to confidence
                    
                    if confidence >= self.confidence_threshold:
                        timestamp = frame_num / fps
                        
                        match_info = {
                            'frame_number': frame_num,
                            'timestamp': timestamp,
                            'timestamp_formatted': self._format_timestamp(timestamp),
                            'confidence': float(confidence),
                            'face_location': face_locations[i],
                            'face_encoding': frame_face_encoding.tolist()
                        }
                        
                        search_results['matches_found'].append(match_info)
                        search_results['confidence_scores'].append(confidence)
                        search_results['matches_count'] += 1
                        
                        # Track best match
                        if confidence > best_confidence:
                            best_confidence = confidence
                            search_results['best_match'] = match_info
                            
                        # Save frame with detection
                        self._save_detection_frame(frame, face_locations[i], case_id, frame_num, confidence)
            
            cap.release()
            
            # Calculate summary statistics
            if search_results['confidence_scores']:
                search_results['average_confidence'] = float(np.mean(search_results['confidence_scores']))
                search_results['max_confidence'] = float(np.max(search_results['confidence_scores']))
            else:
                search_results['average_confidence'] = 0.0
                search_results['max_confidence'] = 0.0
                
        except Exception as e:
            search_results['error'] = str(e)
            print(f"Error in CCTV search: {str(e)}")
            
        return search_results
    
    def batch_search_multiple_cctv(self, case_id, target_face_encodings, cctv_folder_path):
        """
        Search person in multiple CCTV videos in a folder
        """
        batch_results = {
            'case_id': case_id,
            'total_videos_processed': 0,
            'videos_with_matches': 0,
            'total_matches': 0,
            'video_results': [],
            'batch_timestamp': datetime.now().isoformat()
        }
        
        # Get all video files in folder
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for file in os.listdir(cctv_folder_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(cctv_folder_path, file))
        
        # Process each video
        for video_path in video_files:
            print(f"Processing CCTV video: {os.path.basename(video_path)}")
            
            video_result = self.search_person_in_cctv(case_id, target_face_encodings, video_path)
            video_result['video_name'] = os.path.basename(video_path)
            
            batch_results['video_results'].append(video_result)
            batch_results['total_videos_processed'] += 1
            
            if video_result['matches_count'] > 0:
                batch_results['videos_with_matches'] += 1
                batch_results['total_matches'] += video_result['matches_count']
        
        return batch_results
    
    def _format_timestamp(self, seconds):
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _save_detection_frame(self, frame, face_location, case_id, frame_num, confidence):
        """Save frame with detected face highlighted"""
        try:
            # Create detections directory
            detections_dir = os.path.join('static', 'detections', str(case_id))
            os.makedirs(detections_dir, exist_ok=True)
            
            # Draw rectangle around detected face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add confidence text
            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                       (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save frame
            filename = f"detection_frame_{frame_num}_conf_{confidence:.2f}.jpg"
            filepath = os.path.join(detections_dir, filename)
            cv2.imwrite(filepath, frame)
            
        except Exception as e:
            print(f"Error saving detection frame: {str(e)}")
    
    def generate_search_report(self, search_results):
        """Generate detailed search report"""
        report = {
            'case_id': search_results['case_id'],
            'search_summary': {
                'total_matches': search_results['matches_count'],
                'average_confidence': search_results.get('average_confidence', 0),
                'max_confidence': search_results.get('max_confidence', 0),
                'frames_processed': search_results['total_frames_processed'],
                'faces_detected': search_results['faces_detected']
            },
            'match_timeline': [],
            'recommendations': []
        }
        
        # Create timeline of matches
        for match in search_results['matches_found']:
            report['match_timeline'].append({
                'time': match['timestamp_formatted'],
                'confidence': match['confidence'],
                'frame': match['frame_number']
            })
        
        # Generate recommendations
        if search_results['matches_count'] > 0:
            if search_results['max_confidence'] > 0.8:
                report['recommendations'].append("High confidence matches found - person likely identified")
            elif search_results['max_confidence'] > 0.6:
                report['recommendations'].append("Moderate confidence matches - manual verification recommended")
            else:
                report['recommendations'].append("Low confidence matches - consider additional reference photos")
        else:
            report['recommendations'].append("No matches found - person may not be in this footage")
        
        return report

def search_person_in_cctv_footage(case_id, reference_photos, cctv_videos):
    """
    Main entry point for CCTV search functionality
    """
    search_engine = CCTVSearchEngine()
    
    # Extract face encodings from reference photos
    target_encodings = []
    for photo_path in reference_photos:
        try:
            image = face_recognition.load_image_file(photo_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                target_encodings.extend(encodings)
        except Exception as e:
            print(f"Error processing reference photo {photo_path}: {str(e)}")
    
    if not target_encodings:
        return {"error": "No face encodings found in reference photos"}
    
    # Search in CCTV footage
    all_results = []
    for cctv_path in cctv_videos:
        result = search_engine.search_person_in_cctv(case_id, target_encodings, cctv_path)
        all_results.append(result)
    
    return {
        'case_id': case_id,
        'search_results': all_results,
        'total_cctv_processed': len(cctv_videos),
        'status': 'completed'
    }