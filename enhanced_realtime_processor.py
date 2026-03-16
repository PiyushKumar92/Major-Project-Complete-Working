"""
Enhanced Real-Time Processor with Analytics Integration
Combines real-time processing with advanced analytics
"""

import cv2
import threading
import time
import json
from datetime import datetime
from flask_socketio import SocketIO, emit
from queue import Queue
import numpy as np

try:
    from enhanced_ai_engine import EnhancedAIEngine
    ENHANCED_AI_AVAILABLE = True
except ImportError:
    from vision_engine import VisionEngine
    ENHANCED_AI_AVAILABLE = False

from analytics_engine import analytics_engine

class EnhancedRealTimeProcessor:
    def __init__(self, socketio, db):
        self.socketio = socketio
        self.db = db
        self.active_streams = {}
        self.processing_queue = Queue()
        self.alert_threshold = 0.7
        
        # Initialize AI engine
        if ENHANCED_AI_AVAILABLE:
            self.ai_engine = EnhancedAIEngine()
            print("✓ Enhanced AI Engine with Analytics loaded")
        else:
            self.ai_engine = VisionEngine()
            print("✓ Basic AI Engine with Analytics loaded")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
    
    def start_stream(self, stream_id, source, missing_persons_data):
        """Start processing a CCTV stream with analytics"""
        if stream_id in self.active_streams:
            return False
        
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return False
            
            self.active_streams[stream_id] = {
                'capture': cap,
                'missing_persons': missing_persons_data,
                'active': True,
                'last_alert': 0,
                'analytics_enabled': True
            }
            
            # Start stream thread
            stream_thread = threading.Thread(
                target=self._stream_worker, 
                args=(stream_id,), 
                daemon=True
            )
            stream_thread.start()
            
            self.socketio.emit('stream_started', {
                'stream_id': stream_id,
                'status': 'active',
                'analytics_enabled': True,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"Error starting stream {stream_id}: {e}")
            return False
    
    def stop_stream(self, stream_id):
        """Stop processing a CCTV stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['active'] = False
            self.active_streams[stream_id]['capture'].release()
            del self.active_streams[stream_id]
            
            self.socketio.emit('stream_stopped', {
                'stream_id': stream_id,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
        return False
    
    def _stream_worker(self, stream_id):
        """Worker thread for individual stream processing with analytics"""
        stream_data = self.active_streams[stream_id]
        cap = stream_data['capture']
        frame_count = 0
        
        while stream_data['active']:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for better analytics
            frame_count += 1
            if frame_count % 3 == 0:
                self.processing_queue.put({
                    'stream_id': stream_id,
                    'frame': frame,
                    'timestamp': time.time()
                })
            
            time.sleep(0.033)  # ~30 FPS
    
    def _process_frames(self):
        """Main processing thread with analytics integration"""
        while True:
            try:
                if not self.processing_queue.empty():
                    frame_data = self.processing_queue.get()
                    self._analyze_frame_with_analytics(frame_data)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Frame processing error: {e}")
    
    def _analyze_frame_with_analytics(self, frame_data):
        """Analyze frame for missing persons and analytics"""
        stream_id = frame_data['stream_id']
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        if stream_id not in self.active_streams:
            return
        
        missing_persons = self.active_streams[stream_id]['missing_persons']
        
        try:
            # Process frame through analytics engine
            analytics_data = analytics_engine.process_frame(
                frame, stream_id, datetime.fromtimestamp(timestamp)
            )
            
            # Send analytics data via WebSocket
            self.socketio.emit('analytics_update', {
                'stream_id': stream_id,
                'analytics': {
                    'detection_count': len(analytics_data['detections']),
                    'movement_data': analytics_data['movement_data'],
                    'timestamp': datetime.fromtimestamp(timestamp).isoformat()
                }
            })
            
            # Check detections against missing persons
            for detection in analytics_data['detections']:
                x, y, w, h = detection['bbox']
                confidence = detection.get('confidence', 0.5)
                
                if confidence < self.alert_threshold:
                    continue
                
                # Extract face from detection
                face_region = frame[y:y+h, x:x+w]
                
                # Compare with missing persons
                for person in missing_persons:
                    match_confidence = self._compare_faces(face_region, person)
                    
                    if match_confidence > self.alert_threshold:
                        self._send_enhanced_alert(stream_id, person, match_confidence, 
                                                timestamp, detection, analytics_data)
        
        except Exception as e:
            print(f"Enhanced frame analysis error: {e}")
    
    def _compare_faces(self, face_frame, person_data):
        """Compare detected face with missing person"""
        try:
            if ENHANCED_AI_AVAILABLE:
                return self.ai_engine.compare_faces_facenet(face_frame, person_data['photo_path'])
            else:
                return self.ai_engine.compare_faces(face_frame, person_data['photo_path'])
        except:
            return 0.0
    
    def _send_enhanced_alert(self, stream_id, person, confidence, timestamp, detection, analytics_data):
        """Send enhanced alert with analytics context"""
        current_time = time.time()
        last_alert = self.active_streams[stream_id]['last_alert']
        
        # Prevent spam alerts
        if current_time - last_alert < 10:
            return
        
        self.active_streams[stream_id]['last_alert'] = current_time
        
        # Enhanced alert with analytics context
        alert_data = {
            'type': 'person_detected_enhanced',
            'stream_id': stream_id,
            'person_id': person['id'],
            'person_name': person['name'],
            'confidence': round(confidence * 100, 2),
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'location': detection['bbox'],
            'analytics_context': {
                'movement_speed': analytics_data['movement_data']['speed'] if analytics_data['movement_data'] else 0,
                'track_id': analytics_data['movement_data']['track_id'] if analytics_data['movement_data'] else None,
                'detection_center': detection['center']
            }
        }
        
        # Send enhanced WebSocket alert
        self.socketio.emit('enhanced_real_time_alert', alert_data)
        
        # Log enhanced alert
        self._log_enhanced_alert(alert_data)
        
        print(f"🚨 ENHANCED ALERT: {person['name']} detected with {alert_data['confidence']}% confidence")
        if analytics_data['movement_data']:
            print(f"   Movement speed: {analytics_data['movement_data']['speed']:.1f} px/s")
    
    def _log_enhanced_alert(self, alert_data):
        """Log enhanced alert with analytics data"""
        try:
            # Enhanced logging with analytics context
            pass
        except Exception as e:
            print(f"Enhanced alert logging error: {e}")
    
    def get_analytics_summary(self):
        """Get analytics summary for all active streams"""
        summary = {
            'active_streams': len(self.active_streams),
            'queue_size': self.processing_queue.qsize(),
            'ai_engine': 'Enhanced' if ENHANCED_AI_AVAILABLE else 'Basic',
            'analytics_enabled': True
        }
        
        # Add per-stream analytics
        for stream_id in self.active_streams.keys():
            movement_patterns = analytics_engine.get_movement_patterns(stream_id, 1)  # Last hour
            predictions = analytics_engine.get_predictions(stream_id)
            
            summary[f'{stream_id}_patterns'] = len(movement_patterns)
            summary[f'{stream_id}_trend'] = predictions.get('trend', 'stable') if predictions.get('status') == 'success' else 'unknown'
        
        return summary
    
    def get_stream_analytics(self, stream_id):
        """Get detailed analytics for specific stream"""
        if stream_id not in self.active_streams:
            return None
        
        return {
            'movement_patterns': analytics_engine.get_movement_patterns(stream_id, 24),
            'predictions': analytics_engine.get_predictions(stream_id),
            'heatmap_available': True
        }

# Global enhanced processor instance
enhanced_realtime_processor = None