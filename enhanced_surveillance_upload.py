"""
Enhanced Surveillance Footage Upload System
- Large file support (up to 10GB)
- Multiple file upload
- AI location analysis
- Progress tracking
- Smart case matching
"""

import os
import cv2
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app, request, jsonify, flash
from geopy.geocoders import Nominatim
import threading
import time

class EnhancedSurveillanceUploader:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024 * 1024  # 10GB
        self.supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm']
        self.upload_progress = {}
        
    def analyze_location_with_ai(self, location_data):
        """AI-powered location analysis for case matching"""
        try:
            # Extract location components
            location_name = location_data.get('location_name', '').lower()
            address = location_data.get('location_address', '').lower()
            city = location_data.get('city', '').lower()
            state = location_data.get('state', '').lower()
            landmarks = location_data.get('landmarks', '').lower()
            
            # Create comprehensive location profile
            location_profile = {
                'primary_location': location_name,
                'full_address': address,
                'city': city,
                'state': state,
                'landmarks': landmarks.split(',') if landmarks else [],
                'search_keywords': []
            }
            
            # Generate search keywords for case matching
            keywords = []
            if location_name:
                keywords.extend(location_name.split())
            if address:
                keywords.extend(address.split())
            if city:
                keywords.append(city)
            if landmarks:
                keywords.extend([l.strip() for l in landmarks.split(',')])
                
            location_profile['search_keywords'] = list(set(keywords))
            
            return location_profile
            
        except Exception as e:
            print(f"Location analysis error: {e}")
            return None
    
    def get_video_metadata(self, file_path):
        """Extract comprehensive video metadata"""
        try:
            cap = cv2.VideoCapture(file_path)
            
            metadata = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0,
                'resolution': '',
                'file_size': os.path.getsize(file_path),
                'quality_score': 0
            }
            
            # Calculate duration
            if metadata['fps'] > 0:
                metadata['duration'] = metadata['frame_count'] / metadata['fps']
            
            # Set resolution
            metadata['resolution'] = f"{metadata['width']}x{metadata['height']}"
            
            # Calculate quality score
            if metadata['width'] >= 1920:
                metadata['quality_score'] = 100  # 1080p+
            elif metadata['width'] >= 1280:
                metadata['quality_score'] = 80   # 720p
            elif metadata['width'] >= 640:
                metadata['quality_score'] = 60   # 480p
            else:
                metadata['quality_score'] = 40   # Below 480p
                
            cap.release()
            return metadata
            
        except Exception as e:
            print(f"Metadata extraction error: {e}")
            return None
    
    def process_upload_with_progress(self, upload_id, files, form_data):
        """Process upload with real-time progress tracking"""
        try:
            self.upload_progress[upload_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Starting upload...',
                'files_processed': 0,
                'total_files': len(files),
                'current_file': '',
                'estimated_time': 0
            }
            
            processed_files = []
            start_time = time.time()
            
            for i, file_data in enumerate(files):
                file = file_data['file']
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"surveillance_{timestamp}_{i+1}_{filename}"
                
                # Update progress
                self.upload_progress[upload_id].update({
                    'progress': int((i / len(files)) * 50),  # 50% for upload
                    'message': f'Uploading {filename}...',
                    'current_file': filename,
                    'files_processed': i
                })
                
                # Save file
                surveillance_dir = os.path.join('static', 'surveillance')
                os.makedirs(surveillance_dir, exist_ok=True)
                file_path = os.path.join(surveillance_dir, filename)
                file.save(file_path)
                
                # Get metadata
                metadata = self.get_video_metadata(file_path)
                if metadata:
                    processed_files.append({
                        'filename': filename,
                        'path': f"surveillance/{filename}",
                        'metadata': metadata
                    })
            
            # AI Location Analysis
            self.upload_progress[upload_id].update({
                'progress': 60,
                'message': 'Analyzing location with AI...'
            })
            
            location_profile = self.analyze_location_with_ai(form_data)
            
            # Case Matching
            self.upload_progress[upload_id].update({
                'progress': 80,
                'message': 'Finding matching cases...'
            })
            
            matching_cases = self.find_matching_cases(location_profile)
            
            # Complete
            elapsed_time = time.time() - start_time
            self.upload_progress[upload_id].update({
                'status': 'completed',
                'progress': 100,
                'message': f'Upload completed successfully in {elapsed_time:.1f}s',
                'files_processed': len(files),
                'processed_files': processed_files,
                'matching_cases': matching_cases,
                'location_profile': location_profile
            })
            
        except Exception as e:
            self.upload_progress[upload_id].update({
                'status': 'error',
                'progress': 0,
                'message': f'Upload failed: {str(e)}'
            })
    
    def find_matching_cases(self, location_profile):
        """Find cases that match the footage location"""
        try:
            from models import Case
            
            if not location_profile:
                return []
            
            matching_cases = []
            keywords = location_profile.get('search_keywords', [])
            
            # Search in active cases
            active_cases = Case.query.filter(
                Case.status.in_(['Active', 'Under Investigation', 'Pending Approval'])
            ).all()
            
            for case in active_cases:
                match_score = 0
                
                # Check last seen location
                if case.last_seen_location:
                    location_lower = case.last_seen_location.lower()
                    for keyword in keywords:
                        if keyword in location_lower:
                            match_score += 10
                
                # Check case description
                if case.description:
                    desc_lower = case.description.lower()
                    for keyword in keywords:
                        if keyword in desc_lower:
                            match_score += 5
                
                if match_score >= 10:  # Minimum threshold
                    matching_cases.append({
                        'case_id': case.id,
                        'person_name': case.person_name,
                        'match_score': match_score,
                        'last_seen_location': case.last_seen_location,
                        'status': case.status
                    })
            
            # Sort by match score
            matching_cases.sort(key=lambda x: x['match_score'], reverse=True)
            return matching_cases[:10]  # Top 10 matches
            
        except Exception as e:
            print(f"Case matching error: {e}")
            return []

# Global uploader instance
surveillance_uploader = EnhancedSurveillanceUploader()