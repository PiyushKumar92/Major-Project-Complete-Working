#!/usr/bin/env python3
"""
Location-based CCTV Management System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from __init__ import create_app, db
from models import Case, CCTVFootage, LocationMatch
import json
from datetime import datetime, timedelta
import multiprocessing as mp
import concurrent.futures

class LocationCCTVManager:
    def __init__(self):
        self.max_workers = mp.cpu_count()
        
    def find_nearby_cctv_footage(self, case_id, last_seen_location, radius_km=5):
        """
        Find all CCTV footage near the last seen location
        """
        app = create_app()
        with app.app_context():
            case = Case.query.get(case_id)
            if not case:
                return {"error": "Case not found"}
            
            print(f"Finding CCTV footage near: {last_seen_location}")
            
            # Search for nearby CCTV footage
            nearby_footage = self._search_nearby_footage(last_seen_location, radius_km)
            
            # Organize by location and time
            organized_footage = self._organize_footage_by_location_time(nearby_footage)
            
            # Calculate coverage statistics
            coverage_stats = self._calculate_coverage_stats(organized_footage)
            
            result = {
                'case_id': case_id,
                'target_location': last_seen_location,
                'search_radius_km': radius_km,
                'total_footage_found': len(nearby_footage),
                'locations_covered': len(organized_footage),
                'coverage_stats': coverage_stats,
                'organized_footage': organized_footage,
                'processing_estimate': self._estimate_processing_time(nearby_footage)
            }
            
            return result
    
    def _search_nearby_footage(self, target_location, radius_km):
        """
        Search for CCTV footage near target location
        """
        # Simulate CCTV footage database
        # In real implementation, this would query actual CCTV database
        
        sample_footage = [
            {
                'id': 1,
                'location': 'Mall Main Entrance',
                'camera_id': 'CAM001',
                'date': '2024-01-15',
                'time_start': '00:00:00',
                'time_end': '23:59:59',
                'duration_hours': 24,
                'file_path': '/cctv/mall/entrance/CAM001_20240115.mp4',
                'file_size_gb': 12.5,
                'resolution': '1920x1080',
                'fps': 30
            },
            {
                'id': 2,
                'location': 'Mall Food Court',
                'camera_id': 'CAM002',
                'date': '2024-01-15',
                'time_start': '06:00:00',
                'time_end': '22:00:00',
                'duration_hours': 16,
                'file_path': '/cctv/mall/foodcourt/CAM002_20240115.mp4',
                'file_size_gb': 8.2,
                'resolution': '1920x1080',
                'fps': 25
            },
            {
                'id': 3,
                'location': 'Mall Parking Area',
                'camera_id': 'CAM003',
                'date': '2024-01-15',
                'time_start': '00:00:00',
                'time_end': '23:59:59',
                'duration_hours': 24,
                'file_path': '/cctv/mall/parking/CAM003_20240115.mp4',
                'file_size_gb': 15.8,
                'resolution': '2560x1440',
                'fps': 30
            },
            {
                'id': 4,
                'location': 'Street Main Road',
                'camera_id': 'CAM004',
                'date': '2024-01-15',
                'time_start': '00:00:00',
                'time_end': '23:59:59',
                'duration_hours': 24,
                'file_path': '/cctv/street/main/CAM004_20240115.mp4',
                'file_size_gb': 18.3,
                'resolution': '1920x1080',
                'fps': 30
            },
            {
                'id': 5,
                'location': 'Street Side Road',
                'camera_id': 'CAM005',
                'date': '2024-01-15',
                'time_start': '00:00:00',
                'time_end': '23:59:59',
                'duration_hours': 24,
                'file_path': '/cctv/street/side/CAM005_20240115.mp4',
                'file_size_gb': 11.7,
                'resolution': '1280x720',
                'fps': 25
            }
        ]
        
        # Filter by location proximity (simplified)
        target_words = set(target_location.lower().split())
        nearby_footage = []
        
        for footage in sample_footage:
            location_words = set(footage['location'].lower().split())
            
            # Calculate location similarity
            common_words = target_words.intersection(location_words)
            similarity = len(common_words) / len(target_words.union(location_words))
            
            if similarity > 0.3:  # 30% similarity threshold
                footage['location_similarity'] = similarity
                footage['distance_km'] = self._calculate_mock_distance(target_location, footage['location'])
                nearby_footage.append(footage)
        
        return nearby_footage
    
    def _organize_footage_by_location_time(self, footage_list):
        """
        Organize footage by location and time periods
        """
        organized = {}
        
        for footage in footage_list:
            location = footage['location']
            
            if location not in organized:
                organized[location] = {
                    'location_name': location,
                    'total_cameras': 0,
                    'total_duration_hours': 0,
                    'total_size_gb': 0,
                    'time_coverage': {},
                    'cameras': {},
                    'footage_files': []
                }
            
            loc_data = organized[location]
            
            # Add camera info
            camera_id = footage['camera_id']
            if camera_id not in loc_data['cameras']:
                loc_data['cameras'][camera_id] = {
                    'camera_id': camera_id,
                    'resolution': footage['resolution'],
                    'fps': footage['fps'],
                    'footage_count': 0,
                    'total_hours': 0
                }
                loc_data['total_cameras'] += 1
            
            # Update statistics
            loc_data['cameras'][camera_id]['footage_count'] += 1
            loc_data['cameras'][camera_id]['total_hours'] += footage['duration_hours']
            loc_data['total_duration_hours'] += footage['duration_hours']
            loc_data['total_size_gb'] += footage['file_size_gb']
            loc_data['footage_files'].append(footage)
            
            # Time coverage
            date = footage['date']
            if date not in loc_data['time_coverage']:
                loc_data['time_coverage'][date] = {
                    'date': date,
                    'hours_covered': 0,
                    'cameras_active': set(),
                    'footage_files': []
                }
            
            loc_data['time_coverage'][date]['hours_covered'] += footage['duration_hours']
            loc_data['time_coverage'][date]['cameras_active'].add(camera_id)
            loc_data['time_coverage'][date]['footage_files'].append(footage)
        
        # Convert sets to lists for JSON serialization
        for location_data in organized.values():
            for date_data in location_data['time_coverage'].values():
                date_data['cameras_active'] = list(date_data['cameras_active'])
        
        return organized
    
    def _calculate_coverage_stats(self, organized_footage):
        """
        Calculate comprehensive coverage statistics
        """
        stats = {
            'total_locations': len(organized_footage),
            'total_cameras': 0,
            'total_duration_hours': 0,
            'total_size_gb': 0,
            'average_duration_per_location': 0,
            'location_breakdown': []
        }
        
        for location, data in organized_footage.items():
            stats['total_cameras'] += data['total_cameras']
            stats['total_duration_hours'] += data['total_duration_hours']
            stats['total_size_gb'] += data['total_size_gb']
            
            stats['location_breakdown'].append({
                'location': location,
                'cameras': data['total_cameras'],
                'duration_hours': data['total_duration_hours'],
                'size_gb': data['total_size_gb']
            })
        
        if stats['total_locations'] > 0:
            stats['average_duration_per_location'] = stats['total_duration_hours'] / stats['total_locations']
        
        return stats
    
    def _estimate_processing_time(self, footage_list):
        """
        Estimate processing time for all footage
        """
        total_duration_hours = sum(f['duration_hours'] for f in footage_list)
        
        # Estimates based on parallel processing
        estimates = {
            'sequential_processing_hours': total_duration_hours * 0.1,  # 10% of video duration
            'parallel_processing_hours': (total_duration_hours * 0.1) / self.max_workers,
            'recommended_workers': min(self.max_workers, len(footage_list)),
            'total_footage_hours': total_duration_hours,
            'processing_speed_ratio': '1:10'  # 1 hour processing for 10 hours footage
        }
        
        return estimates
    
    def _calculate_mock_distance(self, location1, location2):
        """
        Mock distance calculation (in real app, use GPS coordinates)
        """
        # Simplified distance based on location similarity
        words1 = set(location1.lower().split())
        words2 = set(location2.lower().split())
        
        common_words = words1.intersection(words2)
        similarity = len(common_words) / len(words1.union(words2))
        
        # Convert similarity to mock distance (higher similarity = lower distance)
        distance = (1 - similarity) * 10  # Max 10km distance
        return round(distance, 1)
    
    def process_location_footage_parallel(self, case_id, organized_footage, reference_encodings):
        """
        Process all footage for a location in parallel
        """
        print(f"Starting parallel processing for case {case_id}")
        print(f"Using {self.max_workers} parallel workers")
        
        all_results = []
        
        for location, location_data in organized_footage.items():
            print(f"Processing location: {location}")
            print(f"  Cameras: {location_data['total_cameras']}")
            print(f"  Duration: {location_data['total_duration_hours']} hours")
            print(f"  Files: {len(location_data['footage_files'])}")
            
            # Process all footage files for this location in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for footage in location_data['footage_files']:
                    future = executor.submit(
                        self._process_single_footage_file,
                        footage,
                        reference_encodings,
                        case_id
                    )
                    futures.append(future)
                
                # Collect results
                location_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        location_results.append(result)
                    except Exception as e:
                        print(f"Error processing footage: {str(e)}")
                
                all_results.extend(location_results)
        
        return all_results
    
    def _process_single_footage_file(self, footage, reference_encodings, case_id):
        """
        Process a single footage file (runs in separate process)
        """
        # This would contain the actual face recognition processing
        # Similar to the parallel_cctv_processor._process_single_video method
        
        result = {
            'footage_id': footage['id'],
            'location': footage['location'],
            'camera_id': footage['camera_id'],
            'file_path': footage['file_path'],
            'matches_found': 0,
            'processing_time': 0,
            'status': 'completed'
        }
        
        # Mock processing result
        import random
        result['matches_found'] = random.randint(0, 5)  # Random matches for demo
        result['processing_time'] = footage['duration_hours'] * 0.1  # 10% of video duration
        
        return result

def demo_location_cctv_system():
    """
    Demonstrate location-based CCTV management
    """
    print("LOCATION-BASED CCTV MANAGEMENT DEMO")
    print("=" * 50)
    
    manager = LocationCCTVManager()
    
    # Example case
    case_id = 2
    last_seen_location = "Mall Main Entrance"
    
    print(f"Case ID: {case_id}")
    print(f"Last seen location: {last_seen_location}")
    print(f"Search radius: 5 km")
    print()
    
    # Find nearby CCTV footage
    result = manager.find_nearby_cctv_footage(case_id, last_seen_location)
    
    print("SEARCH RESULTS:")
    print(f"Total footage found: {result['total_footage_found']}")
    print(f"Locations covered: {result['locations_covered']}")
    print(f"Total duration: {result['coverage_stats']['total_duration_hours']} hours")
    print(f"Total size: {result['coverage_stats']['total_size_gb']:.1f} GB")
    print()
    
    print("LOCATION BREAKDOWN:")
    for location_data in result['coverage_stats']['location_breakdown']:
        print(f"  {location_data['location']}:")
        print(f"    Cameras: {location_data['cameras']}")
        print(f"    Duration: {location_data['duration_hours']} hours")
        print(f"    Size: {location_data['size_gb']:.1f} GB")
    print()
    
    print("PROCESSING ESTIMATES:")
    estimates = result['processing_estimate']
    print(f"Sequential processing: {estimates['sequential_processing_hours']:.1f} hours")
    print(f"Parallel processing: {estimates['parallel_processing_hours']:.1f} hours")
    print(f"Speed improvement: {estimates['sequential_processing_hours'] / estimates['parallel_processing_hours']:.1f}x faster")
    print(f"Recommended workers: {estimates['recommended_workers']}")

if __name__ == "__main__":
    demo_location_cctv_system()