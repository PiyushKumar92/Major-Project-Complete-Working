"""
Advanced Analytics Engine
Movement patterns, heat maps, and predictive analytics
"""

import cv2
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
import os

try:
    from enhanced_ai_engine import EnhancedAIEngine
    ENHANCED_AI_AVAILABLE = True
except ImportError:
    from vision_engine import VisionEngine
    ENHANCED_AI_AVAILABLE = False

class AnalyticsEngine:
    def __init__(self, db_path="analytics.db"):
        self.db_path = db_path
        self.movement_tracker = MovementTracker()
        self.heatmap_generator = HeatmapGenerator()
        self.predictor = PredictiveAnalytics()
        
        # Initialize AI engine
        if ENHANCED_AI_AVAILABLE:
            self.ai_engine = EnhancedAIEngine()
        else:
            self.ai_engine = VisionEngine()
        
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                camera_id TEXT,
                person_id INTEGER,
                x INTEGER, y INTEGER, w INTEGER, h INTEGER,
                confidence REAL,
                location_name TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movement_patterns (
                id INTEGER PRIMARY KEY,
                person_id INTEGER,
                camera_id TEXT,
                path_data TEXT,
                duration INTEGER,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_frame(self, frame, camera_id, timestamp=None):
        """Process frame for analytics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Detect persons
        if ENHANCED_AI_AVAILABLE:
            detections = self.ai_engine.detect_persons_yolo(frame)
        else:
            detections = self.ai_engine.detect_persons(frame)
        
        analytics_data = {
            'detections': [],
            'movement_data': None,
            'heatmap_data': None
        }
        
        movement_data = None
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection.get('confidence', 0.5)
            
            # Store detection
            try:
                self._store_detection(timestamp, camera_id, x, y, w, h, confidence)
            except Exception as e:
                print(f"Database storage error: {e}")
            
            # Track movement
            movement_data = self.movement_tracker.update(camera_id, x + w//2, y + h//2, timestamp)
            
            # Update heatmap
            self.heatmap_generator.add_detection(camera_id, x + w//2, y + h//2)
            
            analytics_data['detections'].append({
                'bbox': [x, y, w, h],
                'confidence': confidence,
                'center': [x + w//2, y + h//2]
            })
        
        analytics_data['movement_data'] = movement_data
        analytics_data['heatmap_data'] = self.heatmap_generator.get_heatmap(camera_id)
        
        return analytics_data
    
    def _store_detection(self, timestamp, camera_id, x, y, w, h, confidence):
        """Store detection in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detections (timestamp, camera_id, x, y, w, h, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, camera_id, x, y, w, h, confidence))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            # Create database if it doesn't exist
            self._init_database()
    
    def get_movement_patterns(self, camera_id, hours=24):
        """Get movement patterns for analysis"""
        return self.movement_tracker.get_patterns(camera_id, hours)
    
    def get_heatmap(self, camera_id):
        """Get heatmap data"""
        return self.heatmap_generator.get_heatmap(camera_id)
    
    def get_predictions(self, camera_id):
        """Get predictive analytics"""
        return self.predictor.analyze(camera_id, self.db_path)

class MovementTracker:
    def __init__(self, max_history=1000):
        self.tracks = defaultdict(lambda: deque(maxlen=max_history))
        self.track_id = 0
    
    def update(self, camera_id, x, y, timestamp):
        """Update movement tracking"""
        # Simple tracking based on proximity
        best_track = None
        min_distance = float('inf')
        
        for track_id, track_data in self.tracks.items():
            if track_data and track_data[-1]['camera_id'] == camera_id:
                last_pos = track_data[-1]
                distance = np.sqrt((x - last_pos['x'])**2 + (y - last_pos['y'])**2)
                
                if distance < min_distance and distance < 100:  # 100px threshold
                    min_distance = distance
                    best_track = track_id
        
        if best_track is None:
            best_track = self.track_id
            self.track_id += 1
        
        # Add new position
        self.tracks[best_track].append({
            'x': x, 'y': y,
            'timestamp': timestamp,
            'camera_id': camera_id
        })
        
        return {
            'track_id': best_track,
            'path': list(self.tracks[best_track])[-10:],  # Last 10 positions
            'speed': self._calculate_speed(best_track)
        }
    
    def _calculate_speed(self, track_id):
        """Calculate movement speed"""
        track_data = self.tracks[track_id]
        if len(track_data) < 2:
            return 0
        
        recent = list(track_data)[-5:]  # Last 5 positions
        if len(recent) < 2:
            return 0
        
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(recent)):
            dx = recent[i]['x'] - recent[i-1]['x']
            dy = recent[i]['y'] - recent[i-1]['y']
            distance = np.sqrt(dx**2 + dy**2)
            
            time_diff = (recent[i]['timestamp'] - recent[i-1]['timestamp']).total_seconds()
            
            total_distance += distance
            total_time += time_diff
        
        return total_distance / max(total_time, 1)  # pixels per second
    
    def get_patterns(self, camera_id, hours=24):
        """Get movement patterns"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        patterns = []
        
        for track_id, track_data in self.tracks.items():
            recent_data = [p for p in track_data if p['timestamp'] > cutoff_time and p['camera_id'] == camera_id]
            
            if len(recent_data) > 5:
                patterns.append({
                    'track_id': track_id,
                    'path': [{'x': p['x'], 'y': p['y'], 'timestamp': p['timestamp'].isoformat()} for p in recent_data],
                    'duration': (recent_data[-1]['timestamp'] - recent_data[0]['timestamp']).total_seconds(),
                    'avg_speed': self._calculate_speed(track_id)
                })
        
        return patterns

class HeatmapGenerator:
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self.heatmaps = defaultdict(lambda: np.zeros(resolution[::-1], dtype=np.float32))
        self.decay_factor = 0.99
    
    def add_detection(self, camera_id, x, y):
        """Add detection to heatmap"""
        if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
            # Apply Gaussian blur around detection point
            self.heatmaps[camera_id][y, x] += 1
            
            # Apply small Gaussian kernel
            kernel_size = 15
            kernel = cv2.getGaussianKernel(kernel_size, 5)
            kernel = kernel @ kernel.T
            
            y_start = max(0, y - kernel_size//2)
            y_end = min(self.resolution[1], y + kernel_size//2 + 1)
            x_start = max(0, x - kernel_size//2)
            x_end = min(self.resolution[0], x + kernel_size//2 + 1)
            
            ky_start = max(0, kernel_size//2 - y)
            ky_end = ky_start + (y_end - y_start)
            kx_start = max(0, kernel_size//2 - x)
            kx_end = kx_start + (x_end - x_start)
            
            self.heatmaps[camera_id][y_start:y_end, x_start:x_end] += kernel[ky_start:ky_end, kx_start:kx_end] * 0.5
    
    def get_heatmap(self, camera_id):
        """Get normalized heatmap"""
        heatmap = self.heatmaps[camera_id].copy()
        
        # Apply decay
        self.heatmaps[camera_id] *= self.decay_factor
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        return heatmap
    
    def get_heatmap_overlay(self, camera_id, background_frame):
        """Get heatmap overlay on background"""
        heatmap = self.get_heatmap(camera_id)
        
        # Resize heatmap to match frame
        if background_frame.shape[:2] != heatmap.shape:
            heatmap = cv2.resize(heatmap, (background_frame.shape[1], background_frame.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with background
        overlay = cv2.addWeighted(background_frame, 0.7, heatmap_colored, 0.3, 0)
        
        return overlay

class PredictiveAnalytics:
    def analyze(self, camera_id, db_path):
        """Perform predictive analysis"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get recent detection data
            cursor.execute('''
                SELECT timestamp, x, y FROM detections 
                WHERE camera_id = ? AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp
            ''', (camera_id,))
            
            data = cursor.fetchall()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error in predictions: {e}")
            return {'status': 'database_error', 'error': str(e)}
        
        if len(data) < 10:
            return {'status': 'insufficient_data'}
        
        # Analyze patterns
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        location_frequency = defaultdict(int)
        
        for timestamp_str, x, y in data:
            timestamp = datetime.fromisoformat(timestamp_str)
            hour = timestamp.hour
            day = timestamp.strftime('%A')
            
            hourly_counts[hour] += 1
            daily_counts[day] += 1
            
            # Grid-based location frequency
            grid_x = x // 50
            grid_y = y // 50
            location_frequency[(grid_x, grid_y)] += 1
        
        # Predictions
        peak_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_days = sorted(daily_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        hotspots = sorted(location_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate trends
        recent_data = data[-100:]  # Last 100 detections
        older_data = data[-200:-100] if len(data) >= 200 else []
        
        trend = 'stable'
        if older_data:
            recent_rate = len(recent_data) / max(1, len(recent_data))
            older_rate = len(older_data) / max(1, len(older_data))
            
            if recent_rate > older_rate * 1.2:
                trend = 'increasing'
            elif recent_rate < older_rate * 0.8:
                trend = 'decreasing'
        
        return {
            'status': 'success',
            'peak_hours': [{'hour': h, 'count': c} for h, c in peak_hours],
            'peak_days': [{'day': d, 'count': c} for d, c in peak_days],
            'hotspots': [{'location': loc, 'frequency': freq} for loc, freq in hotspots],
            'trend': trend,
            'total_detections': len(data),
            'avg_daily_detections': len(data) / 7
        }

# Global analytics engine instance
analytics_engine = AnalyticsEngine()