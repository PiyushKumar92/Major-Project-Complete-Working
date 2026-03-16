"""
Integration Engine - Police DB, Social Media, Traffic Cameras
Integrates with external systems while maintaining backward compatibility
"""

import requests
import json
import sqlite3
import os
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import hashlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationResult:
    """Integration result data structure"""
    source: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    status: str

class PoliceDBIntegration:
    """Police Database Integration"""
    
    def __init__(self):
        self.enabled = self._check_police_db_config()
        self.api_endpoint = os.getenv('POLICE_DB_API', 'http://localhost:8080/api/police')
        self.api_key = os.getenv('POLICE_DB_KEY', 'demo_key')
        
    def _check_police_db_config(self) -> bool:
        """Check if police DB integration is configured"""
        try:
            # Check for configuration
            return os.path.exists('config/police_db.json') or os.getenv('POLICE_DB_API')
        except:
            return False
    
    def search_missing_persons(self, person_features: Dict) -> List[IntegrationResult]:
        """Search missing persons in police database"""
        if not self.enabled:
            return []
        
        try:
            # Mock police DB search - replace with actual API
            headers = {'Authorization': f'Bearer {self.api_key}'}
            payload = {
                'face_encoding': person_features.get('face_encoding', []),
                'age_range': person_features.get('age_range', [18, 65]),
                'gender': person_features.get('gender', 'unknown'),
                'location': person_features.get('location', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            # Simulate API call (replace with actual implementation)
            response = self._mock_police_api_call(payload)
            
            results = []
            for record in response.get('matches', []):
                result = IntegrationResult(
                    source='police_db',
                    data=record,
                    confidence=record.get('confidence', 0.0),
                    timestamp=datetime.now(),
                    status='found'
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Police DB search error: {e}")
            return []
    
    def _mock_police_api_call(self, payload: Dict) -> Dict:
        """Mock police API call - replace with actual implementation"""
        # Simulate database response
        return {
            'matches': [
                {
                    'case_id': 'MP2024001',
                    'name': 'John Doe',
                    'age': 25,
                    'missing_since': '2024-01-15',
                    'last_seen_location': 'Central Park',
                    'confidence': 0.85,
                    'contact_officer': 'Officer Smith',
                    'phone': '+91-9876543210'
                }
            ],
            'total_matches': 1
        }

class SocialMediaIntegration:
    """Social Media Integration"""
    
    def __init__(self):
        self.enabled = self._check_social_media_config()
        self.platforms = ['facebook', 'twitter', 'instagram']
        self.api_keys = {
            'facebook': os.getenv('FACEBOOK_API_KEY', 'demo_fb_key'),
            'twitter': os.getenv('TWITTER_API_KEY', 'demo_twitter_key'),
            'instagram': os.getenv('INSTAGRAM_API_KEY', 'demo_ig_key')
        }
    
    def _check_social_media_config(self) -> bool:
        """Check if social media integration is configured"""
        try:
            return os.path.exists('config/social_media.json') or any([
                os.getenv('FACEBOOK_API_KEY'),
                os.getenv('TWITTER_API_KEY'),
                os.getenv('INSTAGRAM_API_KEY')
            ])
        except:
            return False
    
    def search_person_posts(self, person_features: Dict) -> List[IntegrationResult]:
        """Search for person in social media posts"""
        if not self.enabled:
            return []
        
        results = []
        
        for platform in self.platforms:
            try:
                platform_results = self._search_platform(platform, person_features)
                results.extend(platform_results)
            except Exception as e:
                logger.error(f"Social media search error on {platform}: {e}")
        
        return results
    
    def _search_platform(self, platform: str, features: Dict) -> List[IntegrationResult]:
        """Search specific social media platform"""
        # Mock social media search - replace with actual API calls
        mock_data = self._mock_social_media_data(platform, features)
        
        results = []
        for post in mock_data.get('posts', []):
            result = IntegrationResult(
                source=f'social_media_{platform}',
                data=post,
                confidence=post.get('confidence', 0.0),
                timestamp=datetime.now(),
                status='found'
            )
            results.append(result)
        
        return results
    
    def _mock_social_media_data(self, platform: str, features: Dict) -> Dict:
        """Mock social media data - replace with actual API calls"""
        return {
            'posts': [
                {
                    'platform': platform,
                    'post_id': f'{platform}_123456',
                    'user': f'user_{platform}',
                    'content': 'Found this person at mall',
                    'location': 'Shopping Mall',
                    'timestamp': '2024-01-20T10:30:00',
                    'confidence': 0.75,
                    'image_url': f'https://{platform}.com/image123.jpg'
                }
            ]
        }

class TrafficCameraIntegration:
    """Traffic Camera Integration"""
    
    def __init__(self):
        self.enabled = self._check_traffic_camera_config()
        self.camera_endpoints = self._load_camera_endpoints()
        self.active_feeds = {}
        
    def _check_traffic_camera_config(self) -> bool:
        """Check if traffic camera integration is configured"""
        try:
            return os.path.exists('config/traffic_cameras.json') or os.getenv('TRAFFIC_CAMERA_API')
        except:
            return False
    
    def _load_camera_endpoints(self) -> List[Dict]:
        """Load traffic camera endpoints"""
        try:
            if os.path.exists('config/traffic_cameras.json'):
                with open('config/traffic_cameras.json', 'r') as f:
                    return json.load(f).get('cameras', [])
            else:
                # Mock camera endpoints
                return [
                    {
                        'id': 'TC001',
                        'name': 'Main Street Junction',
                        'url': 'http://traffic-cam-1.city.gov/stream',
                        'location': {'lat': 28.6139, 'lng': 77.2090},
                        'active': True
                    },
                    {
                        'id': 'TC002', 
                        'name': 'Highway Toll Plaza',
                        'url': 'http://traffic-cam-2.city.gov/stream',
                        'location': {'lat': 28.7041, 'lng': 77.1025},
                        'active': True
                    }
                ]
        except:
            return []
    
    def search_traffic_footage(self, person_features: Dict, time_range: tuple) -> List[IntegrationResult]:
        """Search person in traffic camera footage"""
        if not self.enabled:
            return []
        
        results = []
        start_time, end_time = time_range
        
        for camera in self.camera_endpoints:
            if not camera.get('active', False):
                continue
                
            try:
                camera_results = self._search_camera_footage(camera, person_features, start_time, end_time)
                results.extend(camera_results)
            except Exception as e:
                logger.error(f"Traffic camera search error for {camera['id']}: {e}")
        
        return results
    
    def _search_camera_footage(self, camera: Dict, features: Dict, start_time: datetime, end_time: datetime) -> List[IntegrationResult]:
        """Search specific camera footage"""
        # Mock traffic camera search - replace with actual implementation
        mock_detections = self._mock_traffic_camera_data(camera, features, start_time, end_time)
        
        results = []
        for detection in mock_detections.get('detections', []):
            result = IntegrationResult(
                source=f'traffic_camera_{camera["id"]}',
                data=detection,
                confidence=detection.get('confidence', 0.0),
                timestamp=datetime.now(),
                status='detected'
            )
            results.append(result)
        
        return results
    
    def _mock_traffic_camera_data(self, camera: Dict, features: Dict, start_time: datetime, end_time: datetime) -> Dict:
        """Mock traffic camera data - replace with actual implementation"""
        return {
            'detections': [
                {
                    'camera_id': camera['id'],
                    'camera_name': camera['name'],
                    'detection_time': '2024-01-20T14:25:00',
                    'location': camera['location'],
                    'confidence': 0.82,
                    'vehicle_info': 'Blue sedan, License: DL-01-AB-1234',
                    'direction': 'North-bound',
                    'image_url': f'http://traffic-footage.city.gov/{camera["id"]}/20240120_142500.jpg'
                }
            ]
        }

class IntegrationEngine:
    """Main Integration Engine"""
    
    def __init__(self):
        self.police_db = PoliceDBIntegration()
        self.social_media = SocialMediaIntegration()
        self.traffic_cameras = TrafficCameraIntegration()
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get integration system information"""
        return {
            'police_db_enabled': self.police_db.enabled,
            'social_media_enabled': self.social_media.enabled,
            'traffic_cameras_enabled': self.traffic_cameras.enabled,
            'total_traffic_cameras': len(self.traffic_cameras.camera_endpoints),
            'social_media_platforms': len(self.social_media.platforms),
            'cache_size': len(self.cache)
        }
    
    def comprehensive_search(self, person_features: Dict, search_options: Dict = None) -> Dict[str, List[IntegrationResult]]:
        """Comprehensive search across all integrated systems"""
        if search_options is None:
            search_options = {
                'include_police_db': True,
                'include_social_media': True,
                'include_traffic_cameras': True,
                'time_range_hours': 24
            }
        
        results = {
            'police_db': [],
            'social_media': [],
            'traffic_cameras': [],
            'summary': {}
        }
        
        # Generate cache key
        cache_key = self._generate_cache_key(person_features, search_options)
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < self.cache_timeout:
                return cache_entry['data']
        
        # Police DB search
        if search_options.get('include_police_db', True):
            try:
                results['police_db'] = self.police_db.search_missing_persons(person_features)
            except Exception as e:
                logger.error(f"Police DB search failed: {e}")
        
        # Social Media search
        if search_options.get('include_social_media', True):
            try:
                results['social_media'] = self.social_media.search_person_posts(person_features)
            except Exception as e:
                logger.error(f"Social media search failed: {e}")
        
        # Traffic Camera search
        if search_options.get('include_traffic_cameras', True):
            try:
                hours_back = search_options.get('time_range_hours', 24)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours_back)
                results['traffic_cameras'] = self.traffic_cameras.search_traffic_footage(
                    person_features, (start_time, end_time)
                )
            except Exception as e:
                logger.error(f"Traffic camera search failed: {e}")
        
        # Generate summary
        results['summary'] = self._generate_search_summary(results)
        
        # Cache results
        self.cache[cache_key] = {
            'data': results,
            'timestamp': datetime.now()
        }
        
        return results
    
    def _generate_cache_key(self, features: Dict, options: Dict) -> str:
        """Generate cache key for search results"""
        key_data = {
            'features': str(sorted(features.items())),
            'options': str(sorted(options.items()))
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _generate_search_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate search summary"""
        total_matches = sum(len(results[key]) for key in ['police_db', 'social_media', 'traffic_cameras'])
        
        highest_confidence = 0.0
        best_match_source = None
        
        for source, matches in results.items():
            if source == 'summary':
                continue
            for match in matches:
                if match.confidence > highest_confidence:
                    highest_confidence = match.confidence
                    best_match_source = source
        
        return {
            'total_matches': total_matches,
            'police_db_matches': len(results['police_db']),
            'social_media_matches': len(results['social_media']),
            'traffic_camera_matches': len(results['traffic_cameras']),
            'highest_confidence': highest_confidence,
            'best_match_source': best_match_source,
            'search_timestamp': datetime.now().isoformat()
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'systems': {
                'police_db': {
                    'enabled': self.police_db.enabled,
                    'status': 'active' if self.police_db.enabled else 'disabled',
                    'endpoint': self.police_db.api_endpoint if self.police_db.enabled else None
                },
                'social_media': {
                    'enabled': self.social_media.enabled,
                    'status': 'active' if self.social_media.enabled else 'disabled',
                    'platforms': self.social_media.platforms if self.social_media.enabled else []
                },
                'traffic_cameras': {
                    'enabled': self.traffic_cameras.enabled,
                    'status': 'active' if self.traffic_cameras.enabled else 'disabled',
                    'camera_count': len(self.traffic_cameras.camera_endpoints)
                }
            },
            'cache_stats': {
                'entries': len(self.cache),
                'timeout_seconds': self.cache_timeout
            },
            'last_updated': datetime.now().isoformat()
        }

# Global integration engine instance
integration_engine = IntegrationEngine()

def get_integration_engine():
    """Get global integration engine instance"""
    return integration_engine