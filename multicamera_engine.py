"""
Multi-Camera Correlation Engine
Cross-camera tracking and route reconstruction
"""

import cv2
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
import threading
import time

try:
    from enhanced_ai_engine import EnhancedAIEngine
    ENHANCED_AI_AVAILABLE = True
except ImportError:
    from vision_engine import VisionEngine
    ENHANCED_AI_AVAILABLE = False

class MultiCameraEngine:
    def __init__(self, db_path="multicamera.db"):
        self.db_path = db_path
        self.cross_tracker = CrossCameraTracker()
        self.route_reconstructor = RouteReconstructor()
        self.camera_network = CameraNetwork()
        
        # Initialize AI engine
        if ENHANCED_AI_AVAILABLE:
            self.ai_engine = EnhancedAIEngine()
        else:
            self.ai_engine = VisionEngine()
        
        self._init_database()
        self.active_correlations = {}
        
    def _init_database(self):
        """Initialize multi-camera database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_detections (
                id INTEGER PRIMARY KEY,
                global_track_id TEXT,
                camera_id TEXT,
                local_track_id INTEGER,
                timestamp DATETIME,
                x INTEGER, y INTEGER, w INTEGER, h INTEGER,
                confidence REAL,
                face_encoding TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_transitions (
                id INTEGER PRIMARY KEY,
                global_track_id TEXT,
                from_camera TEXT,
                to_camera TEXT,
                transition_time REAL,
                confidence REAL,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reconstructed_routes (
                id INTEGER PRIMARY KEY,
                global_track_id TEXT,
                route_data TEXT,
                total_duration REAL,
                camera_count INTEGER,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_camera(self, camera_id, position, coverage_area):
        """Register camera in network"""
        self.camera_network.add_camera(camera_id, position, coverage_area)
    
    def process_detection(self, camera_id, detection, timestamp=None):
        """Process detection for cross-camera correlation"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract face encoding for correlation
        face_encoding = self._extract_face_encoding(detection)
        
        # Cross-camera tracking
        global_track_id = self.cross_tracker.correlate_detection(
            camera_id, detection, face_encoding, timestamp
        )
        
        # Store detection
        self._store_cross_detection(global_track_id, camera_id, detection, timestamp, face_encoding)
        
        # Check for camera transitions
        transition = self._detect_transition(global_track_id, camera_id, timestamp)
        
        # Update route reconstruction
        route = self.route_reconstructor.update_route(global_track_id, camera_id, detection, timestamp)
        
        return {
            'global_track_id': global_track_id,
            'camera_id': camera_id,
            'transition': transition,
            'route_segment': route,
            'correlation_confidence': self.cross_tracker.get_confidence(global_track_id)
        }
    
    def _extract_face_encoding(self, detection):
        """Extract face encoding for correlation"""
        try:
            if ENHANCED_AI_AVAILABLE:
                # Use FaceNet encoding
                return "facenet_encoding_placeholder"
            else:
                # Use face_recognition encoding
                return "face_recognition_encoding_placeholder"
        except:
            return None
    
    def _store_cross_detection(self, global_track_id, camera_id, detection, timestamp, face_encoding):
        """Store cross-camera detection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            x, y, w, h = detection['bbox']
            confidence = detection.get('confidence', 0.5)
            local_track_id = detection.get('track_id', 0)
            
            cursor.execute('''
                INSERT INTO cross_detections 
                (global_track_id, camera_id, local_track_id, timestamp, x, y, w, h, confidence, face_encoding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (global_track_id, camera_id, local_track_id, timestamp, x, y, w, h, confidence, 
                  json.dumps(face_encoding) if face_encoding else None))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cross-detection storage error: {e}")
    
    def _detect_transition(self, global_track_id, current_camera, timestamp):
        """Detect camera transitions"""
        return self.cross_tracker.detect_transition(global_track_id, current_camera, timestamp)
    
    def get_global_tracks(self, hours=24):
        """Get all global tracks"""
        return self.cross_tracker.get_global_tracks(hours)
    
    def get_route_reconstruction(self, global_track_id):
        """Get reconstructed route for track"""
        return self.route_reconstructor.get_route(global_track_id)
    
    def get_camera_correlations(self):
        """Get camera correlation statistics"""
        return self.camera_network.get_correlations()

class CrossCameraTracker:
    def __init__(self, similarity_threshold=0.7):
        self.global_tracks = defaultdict(lambda: {
            'cameras': {},
            'last_seen': {},
            'face_encodings': [],
            'confidence': 0.0
        })
        self.track_counter = 0
        self.similarity_threshold = similarity_threshold
        
    def correlate_detection(self, camera_id, detection, face_encoding, timestamp):
        """Correlate detection across cameras"""
        best_match = None
        best_similarity = 0.0
        
        # Compare with existing global tracks
        for global_id, track_data in self.global_tracks.items():
            if face_encoding and track_data['face_encodings']:
                similarity = self._calculate_similarity(face_encoding, track_data['face_encodings'])
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match = global_id
        
        # Create new global track if no match
        if best_match is None:
            best_match = f"global_{self.track_counter}"
            self.track_counter += 1
        
        # Update global track
        self._update_global_track(best_match, camera_id, detection, face_encoding, timestamp, best_similarity)
        
        return best_match
    
    def _calculate_similarity(self, encoding1, encoding_list):
        """Calculate similarity between face encodings"""
        if not encoding1 or not encoding_list:
            return 0.0
        
        # Placeholder similarity calculation
        # In real implementation, use cosine similarity or Euclidean distance
        return 0.8  # Mock similarity
    
    def _update_global_track(self, global_id, camera_id, detection, face_encoding, timestamp, confidence):
        """Update global track data"""
        track = self.global_tracks[global_id]
        
        # Update camera presence
        if camera_id not in track['cameras']:
            track['cameras'][camera_id] = []
        
        track['cameras'][camera_id].append({
            'detection': detection,
            'timestamp': timestamp,
            'confidence': confidence
        })
        
        # Update last seen
        track['last_seen'][camera_id] = timestamp
        
        # Update face encodings
        if face_encoding and len(track['face_encodings']) < 5:
            track['face_encodings'].append(face_encoding)
        
        # Update overall confidence
        track['confidence'] = max(track['confidence'], confidence)
    
    def detect_transition(self, global_track_id, current_camera, timestamp):
        """Detect camera transition"""
        if global_track_id not in self.global_tracks:
            return None
        
        track = self.global_tracks[global_track_id]
        
        # Check for recent activity in other cameras
        for camera_id, last_seen in track['last_seen'].items():
            if camera_id != current_camera:
                time_diff = (timestamp - last_seen).total_seconds()
                
                # Transition detected if seen in another camera within last 30 seconds
                if 0 < time_diff <= 30:
                    return {
                        'from_camera': camera_id,
                        'to_camera': current_camera,
                        'transition_time': time_diff,
                        'confidence': track['confidence']
                    }
        
        return None
    
    def get_confidence(self, global_track_id):
        """Get tracking confidence"""
        if global_track_id in self.global_tracks:
            return self.global_tracks[global_track_id]['confidence']
        return 0.0
    
    def get_global_tracks(self, hours=24):
        """Get global tracks from last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        active_tracks = {}
        
        for global_id, track_data in self.global_tracks.items():
            # Check if track has recent activity
            recent_activity = any(
                last_seen > cutoff_time 
                for last_seen in track_data['last_seen'].values()
            )
            
            if recent_activity:
                active_tracks[global_id] = {
                    'cameras': list(track_data['cameras'].keys()),
                    'camera_count': len(track_data['cameras']),
                    'confidence': track_data['confidence'],
                    'last_activity': max(track_data['last_seen'].values())
                }
        
        return active_tracks

class RouteReconstructor:
    def __init__(self):
        self.routes = defaultdict(list)
        
    def update_route(self, global_track_id, camera_id, detection, timestamp):
        """Update route for global track"""
        route_point = {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'position': detection['bbox'],
            'confidence': detection.get('confidence', 0.5)
        }
        
        self.routes[global_track_id].append(route_point)
        
        # Keep only last 100 points
        if len(self.routes[global_track_id]) > 100:
            self.routes[global_track_id] = self.routes[global_track_id][-100:]
        
        return route_point
    
    def get_route(self, global_track_id):
        """Get reconstructed route"""
        if global_track_id not in self.routes:
            return None
        
        route_points = self.routes[global_track_id]
        
        if len(route_points) < 2:
            return None
        
        # Calculate route statistics
        total_duration = (route_points[-1]['timestamp'] - route_points[0]['timestamp']).total_seconds()
        unique_cameras = len(set(point['camera_id'] for point in route_points))
        
        # Group by camera for visualization
        camera_segments = defaultdict(list)
        for point in route_points:
            camera_segments[point['camera_id']].append(point)
        
        return {
            'global_track_id': global_track_id,
            'total_duration': total_duration,
            'camera_count': unique_cameras,
            'route_points': route_points,
            'camera_segments': dict(camera_segments),
            'start_time': route_points[0]['timestamp'],
            'end_time': route_points[-1]['timestamp']
        }
    
    def get_route_summary(self, global_track_id):
        """Get route summary statistics"""
        route = self.get_route(global_track_id)
        if not route:
            return None
        
        return {
            'duration': route['total_duration'],
            'cameras_visited': route['camera_count'],
            'total_detections': len(route['route_points']),
            'avg_confidence': np.mean([p['confidence'] for p in route['route_points']]),
            'camera_sequence': [seg for seg in route['camera_segments'].keys()]
        }

class CameraNetwork:
    def __init__(self):
        self.cameras = {}
        self.adjacency_matrix = {}
        self.transition_counts = defaultdict(int)
        
    def add_camera(self, camera_id, position, coverage_area):
        """Add camera to network"""
        self.cameras[camera_id] = {
            'position': position,
            'coverage_area': coverage_area,
            'neighbors': []
        }
        
        # Calculate neighbors based on coverage overlap
        self._update_adjacency()
    
    def _update_adjacency(self):
        """Update camera adjacency matrix"""
        camera_ids = list(self.cameras.keys())
        
        for i, cam1 in enumerate(camera_ids):
            for j, cam2 in enumerate(camera_ids):
                if i != j:
                    # Simple distance-based adjacency
                    pos1 = self.cameras[cam1]['position']
                    pos2 = self.cameras[cam2]['position']
                    
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # Adjacent if within 100 units
                    if distance <= 100:
                        if cam2 not in self.cameras[cam1]['neighbors']:
                            self.cameras[cam1]['neighbors'].append(cam2)
    
    def record_transition(self, from_camera, to_camera):
        """Record camera transition"""
        self.transition_counts[(from_camera, to_camera)] += 1
    
    def get_correlations(self):
        """Get camera correlation statistics"""
        return {
            'total_cameras': len(self.cameras),
            'total_transitions': sum(self.transition_counts.values()),
            'most_common_transitions': sorted(
                self.transition_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'camera_network': self.cameras
        }

# Global multi-camera engine instance
multicamera_engine = MultiCameraEngine()