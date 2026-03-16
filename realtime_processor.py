"""
Real-Time CCTV Processing with WebSocket Support
Enhanced AI integration for live video streams
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

class RealTimeProcessor:
    def __init__(self, socketio, db):
        self.socketio = socketio
        self.db = db
        self.active_streams = {}
        self.processing_queue = Queue()
        self.alert_threshold = 0.7  # Confidence threshold for alerts
        
        # Initialize AI engine
        if ENHANCED_AI_AVAILABLE:
            self.ai_engine = EnhancedAIEngine()
            print("✓ Enhanced AI Engine loaded for real-time processing")
        else:
            self.ai_engine = VisionEngine()
            print("✓ Basic AI Engine loaded for real-time processing")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
    
    def start_stream(self, stream_id, source, missing_persons_data):
        """Start processing a CCTV stream"""
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
                'last_alert': 0
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
        """Worker thread for individual stream processing"""
        stream_data = self.active_streams[stream_id]
        cap = stream_data['capture']
        frame_count = 0
        
        while stream_data['active']:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for performance
            frame_count += 1
            if frame_count % 5 == 0:
                self.processing_queue.put({
                    'stream_id': stream_id,
                    'frame': frame,
                    'timestamp': time.time()
                })
            
            time.sleep(0.033)  # ~30 FPS
    
    def _process_frames(self):
        """Main processing thread for all frames"""
        while True:
            try:
                if not self.processing_queue.empty():
                    frame_data = self.processing_queue.get()
                    self._analyze_frame(frame_data)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Frame processing error: {e}")
    
    def _analyze_frame(self, frame_data):
        """Analyze frame for missing persons"""
        stream_id = frame_data['stream_id']
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        if stream_id not in self.active_streams:
            return
        
        missing_persons = self.active_streams[stream_id]['missing_persons']
        
        try:
            # Detect persons in frame
            if ENHANCED_AI_AVAILABLE:
                detections = self.ai_engine.detect_persons_yolo(frame)
            else:
                detections = self.ai_engine.detect_persons(frame)
            
            if not detections:
                return
            
            # Check each detection against missing persons
            for detection in detections:
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
                        self._send_alert(stream_id, person, match_confidence, timestamp, detection)
        
        except Exception as e:
            print(f"Frame analysis error: {e}")
    
    def _compare_faces(self, face_frame, person_data):
        """Compare detected face with missing person"""
        try:
            if ENHANCED_AI_AVAILABLE:
                return self.ai_engine.compare_faces_facenet(face_frame, person_data['photo_path'])
            else:
                return self.ai_engine.compare_faces(face_frame, person_data['photo_path'])
        except:
            return 0.0
    
    def _send_alert(self, stream_id, person, confidence, timestamp, detection):
        """Send real-time alert via WebSocket"""
        current_time = time.time()
        last_alert = self.active_streams[stream_id]['last_alert']
        
        # Prevent spam alerts (minimum 10 seconds between alerts for same person)
        if current_time - last_alert < 10:
            return
        
        self.active_streams[stream_id]['last_alert'] = current_time
        
        alert_data = {
            'type': 'person_detected',
            'stream_id': stream_id,
            'person_id': person['id'],
            'person_name': person['name'],
            'confidence': round(confidence * 100, 2),
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'location': detection['bbox']
        }
        
        # Send WebSocket alert
        self.socketio.emit('real_time_alert', alert_data)
        
        # Log alert to database
        self._log_alert(alert_data)
        
        print(f"🚨 ALERT: {person['name']} detected with {alert_data['confidence']}% confidence")
    
    def _log_alert(self, alert_data):
        """Log alert to database"""
        try:
            # Add to alerts table (assuming you have one)
            # This is a placeholder - adjust based on your database schema
            pass
        except Exception as e:
            print(f"Alert logging error: {e}")
    
    def get_active_streams(self):
        """Get list of active streams"""
        return list(self.active_streams.keys())
    
    def get_stream_stats(self):
        """Get processing statistics"""
        return {
            'active_streams': len(self.active_streams),
            'queue_size': self.processing_queue.qsize(),
            'ai_engine': 'Enhanced' if ENHANCED_AI_AVAILABLE else 'Basic'
        }