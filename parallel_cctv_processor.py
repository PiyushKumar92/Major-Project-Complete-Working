#!/usr/bin/env python3
"""
Parallel CCTV Processing Engine - Handle massive CCTV footage efficiently
"""

import multiprocessing as mp
import concurrent.futures
import os
import json
import time
from datetime import datetime, timedelta
import face_recognition
import cv2
import numpy as np
from pathlib import Path

class ParallelCCTVProcessor:
    def __init__(self, max_workers=None):
        # Use all CPU cores by default
        self.max_workers = max_workers or mp.cpu_count()
        self.confidence_threshold = 0.6
        
    def process_massive_cctv_footage(self, case_id, reference_encodings, cctv_footage_list, location_filter=None):
        """
        Process hundreds/thousands of CCTV videos in parallel
        """
        start_time = time.time()
        
        results = {
            'case_id': case_id,
            'total_videos': len(cctv_footage_list),
            'processed_videos': 0,
            'videos_with_matches': 0,
            'total_matches': 0,
            'processing_time': 0,
            'location_summary': {},
            'video_results': [],
            'parallel_workers': self.max_workers
        }
        
        print(f"🚀 Starting parallel processing of {len(cctv_footage_list)} CCTV videos")
        print(f"💻 Using {self.max_workers} parallel workers")
        
        # Group videos by location if location filter provided
        if location_filter:
            cctv_footage_list = self._filter_by_location(cctv_footage_list, location_filter)
            print(f"📍 Filtered to {len(cctv_footage_list)} videos near location: {location_filter}")
        
        # Process videos in parallel batches
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all video processing jobs
            future_to_video = {
                executor.submit(self._process_single_video, video_info, reference_encodings, case_id): video_info
                for video_info in cctv_footage_list
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_video):
                video_info = future_to_video[future]
                try:
                    video_result = future.result()
                    results['video_results'].append(video_result)
                    results['processed_videos'] += 1
                    
                    if video_result['matches_found'] > 0:
                        results['videos_with_matches'] += 1
                        results['total_matches'] += video_result['matches_found']
                    
                    # Update location summary
                    location = video_result.get('location', 'Unknown')
                    if location not in results['location_summary']:
                        results['location_summary'][location] = {
                            'videos_processed': 0,
                            'matches_found': 0,
                            'total_duration': 0
                        }
                    
                    results['location_summary'][location]['videos_processed'] += 1
                    results['location_summary'][location]['matches_found'] += video_result['matches_found']
                    results['location_summary'][location]['total_duration'] += video_result.get('duration', 0)
                    
                    # Progress update
                    progress = (results['processed_videos'] / results['total_videos']) * 100
                    print(f"📹 Progress: {results['processed_videos']}/{results['total_videos']} ({progress:.1f}%) - Matches: {results['total_matches']}")
                    
                except Exception as e:
                    print(f"❌ Error processing {video_info.get('path', 'unknown')}: {str(e)}")
        
        results['processing_time'] = time.time() - start_time
        print(f"✅ Parallel processing completed in {results['processing_time']:.2f} seconds")
        
        return results
    
    def _process_single_video(self, video_info, reference_encodings, case_id):
        """
        Process a single CCTV video (runs in separate process)
        """
        video_path = video_info['path']
        video_result = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'location': video_info.get('location', 'Unknown'),
            'camera_id': video_info.get('camera_id', 'Unknown'),
            'timestamp_start': video_info.get('timestamp_start'),
            'timestamp_end': video_info.get('timestamp_end'),
            'matches_found': 0,
            'match_details': [],
            'duration': 0,
            'frames_processed': 0,
            'processing_time': 0
        }
        
        try:
            start_time = time.time()
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                video_result['error'] = 'Could not open video'
                return video_result
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_result['duration'] = frame_count / fps if fps > 0 else 0
            
            # Process every 30th frame (1 frame per second for 30fps video)
            frame_interval = max(1, int(fps)) if fps > 0 else 30
            
            for frame_num in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                video_result['frames_processed'] += 1
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                if not face_locations:
                    continue
                
                # Get face encodings
                frame_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Compare with reference encodings
                for i, face_encoding in enumerate(frame_encodings):
                    distances = face_recognition.face_distance(reference_encodings, face_encoding)
                    min_distance = np.min(distances)
                    confidence = 1 - min_distance
                    
                    if confidence >= self.confidence_threshold:
                        timestamp = frame_num / fps if fps > 0 else frame_num
                        
                        match_detail = {
                            'frame_number': frame_num,
                            'timestamp': timestamp,
                            'timestamp_formatted': self._format_timestamp(timestamp),
                            'confidence': float(confidence),
                            'face_location': face_locations[i]
                        }
                        
                        video_result['match_details'].append(match_detail)
                        video_result['matches_found'] += 1
            
            cap.release()
            video_result['processing_time'] = time.time() - start_time
            
        except Exception as e:
            video_result['error'] = str(e)
        
        return video_result
    
    def _filter_by_location(self, cctv_list, target_location):
        """
        Filter CCTV footage by location proximity
        """
        filtered_list = []
        target_location_lower = target_location.lower()
        
        for video_info in cctv_list:
            video_location = video_info.get('location', '').lower()
            
            # Simple location matching (can be enhanced with GPS coordinates)
            if (target_location_lower in video_location or 
                video_location in target_location_lower or
                self._calculate_location_similarity(target_location_lower, video_location) > 0.7):
                filtered_list.append(video_info)
        
        return filtered_list
    
    def _calculate_location_similarity(self, loc1, loc2):
        """
        Calculate similarity between two location strings
        """
        words1 = set(loc1.split())
        words2 = set(loc2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class CCTVFootageManager:
    """
    Manage and organize CCTV footage by location and time
    """
    
    def __init__(self, footage_base_path):
        self.footage_base_path = footage_base_path
        
    def scan_footage_directory(self, target_location=None, date_range=None):
        """
        Scan and catalog all CCTV footage
        """
        footage_catalog = {
            'total_videos': 0,
            'total_duration': 0,
            'locations': {},
            'date_range': {},
            'camera_coverage': {},
            'videos': []
        }
        
        # Scan directory structure
        for root, dirs, files in os.walk(self.footage_base_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(root, file)
                    video_info = self._extract_video_metadata(video_path, root)
                    
                    # Filter by location if specified
                    if target_location and not self._location_matches(video_info['location'], target_location):
                        continue
                    
                    # Filter by date range if specified
                    if date_range and not self._date_in_range(video_info['date'], date_range):
                        continue
                    
                    footage_catalog['videos'].append(video_info)
                    footage_catalog['total_videos'] += 1
                    footage_catalog['total_duration'] += video_info['duration']
                    
                    # Update location stats
                    location = video_info['location']
                    if location not in footage_catalog['locations']:
                        footage_catalog['locations'][location] = {
                            'video_count': 0,
                            'total_duration': 0,
                            'cameras': set()
                        }
                    
                    footage_catalog['locations'][location]['video_count'] += 1
                    footage_catalog['locations'][location]['total_duration'] += video_info['duration']
                    footage_catalog['locations'][location]['cameras'].add(video_info['camera_id'])
        
        # Convert sets to lists for JSON serialization
        for location_data in footage_catalog['locations'].values():
            location_data['cameras'] = list(location_data['cameras'])
        
        return footage_catalog
    
    def _extract_video_metadata(self, video_path, directory_path):
        """
        Extract metadata from video file and directory structure
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            # Extract location from directory structure
            path_parts = directory_path.replace(self.footage_base_path, '').strip(os.sep).split(os.sep)
            location = path_parts[0] if path_parts else 'Unknown'
            
            # Extract camera ID and date from filename
            filename = os.path.basename(video_path)
            camera_id = self._extract_camera_id(filename)
            date = self._extract_date(filename)
            
            return {
                'path': video_path,
                'filename': filename,
                'location': location,
                'camera_id': camera_id,
                'date': date,
                'duration': duration,
                'size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
        except Exception as e:
            return {
                'path': video_path,
                'filename': os.path.basename(video_path),
                'location': 'Unknown',
                'camera_id': 'Unknown',
                'date': None,
                'duration': 0,
                'size_mb': 0,
                'error': str(e)
            }
    
    def _extract_camera_id(self, filename):
        """Extract camera ID from filename"""
        # Common patterns: CAM01, Camera_1, CCTV_001, etc.
        import re
        patterns = [
            r'CAM(\d+)',
            r'Camera[_-]?(\d+)',
            r'CCTV[_-]?(\d+)',
            r'Cam(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return f"CAM{match.group(1).zfill(3)}"
        
        return 'Unknown'
    
    def _extract_date(self, filename):
        """Extract date from filename"""
        import re
        # Common date patterns
        patterns = [
            r'(\d{4}[-_]\d{2}[-_]\d{2})',
            r'(\d{2}[-_]\d{2}[-_]\d{4})',
            r'(\d{8})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return None
    
    def _location_matches(self, video_location, target_location):
        """Check if video location matches target location"""
        return target_location.lower() in video_location.lower()
    
    def _date_in_range(self, video_date, date_range):
        """Check if video date is in specified range"""
        if not video_date or not date_range:
            return True
        # Implement date range checking logic
        return True

def demo_parallel_processing():
    """
    Demo showing parallel CCTV processing capabilities
    """
    print("PARALLEL CCTV PROCESSING DEMO")
    print("=" * 50)
    
    # Example: 100 CCTV videos, 24 hours each
    example_footage = []
    locations = ['Mall_Entrance', 'Mall_Food_Court', 'Mall_Parking', 'Street_Main', 'Street_Side']
    
    for i in range(100):
        location = locations[i % len(locations)]
        example_footage.append({
            'path': f'/cctv_footage/{location}/CAM{i:03d}_2024_01_15.mp4',
            'location': location,
            'camera_id': f'CAM{i:03d}',
            'duration': 24 * 3600,  # 24 hours
            'timestamp_start': '2024-01-15 00:00:00',
            'timestamp_end': '2024-01-15 23:59:59'
        })
    
    print(f"📹 Total CCTV footage: {len(example_footage)} videos")
    print(f"⏱️  Total duration: {sum(v['duration'] for v in example_footage) / 3600:.0f} hours")
    print(f"📍 Locations covered: {len(locations)}")
    
    # Simulate processing
    processor = ParallelCCTVProcessor(max_workers=8)
    
    print(f"\n🚀 Processing with {processor.max_workers} parallel workers")
    print("📊 Expected performance:")
    print(f"   • Sequential processing: ~{len(example_footage) * 2:.0f} minutes")
    print(f"   • Parallel processing: ~{(len(example_footage) * 2) / processor.max_workers:.0f} minutes")
    print(f"   • Speed improvement: {processor.max_workers}x faster")
    
    return example_footage

if __name__ == "__main__":
    demo_parallel_processing()