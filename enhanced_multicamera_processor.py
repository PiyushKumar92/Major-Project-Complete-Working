"""
Enhanced Multi-Camera Real-Time Processor
Integrates real-time processing with multi-camera correlation
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

from multicamera_engine import multicamera_engine
from analytics_engine import analytics_engine

class EnhancedMultiCameraProcessor:
    def __init__(self, socketio, db):
        self.socketio = socketio
        self.db = db
        self.active_streams = {}
        self.processing_queue = Queue()
        self.alert_threshold = 0.7
        
        # Initialize AI engine
        if ENHANCED_AI_AVAILABLE:
            self.ai_engine = EnhancedAIEngine()
            print("✓ Enhanced Multi-Camera AI Engine loaded")
        else:
            self.ai_engine = VisionEngine()
            print("✓ Basic Multi-Camera AI Engine loaded")
        
        # Initialize camera network
        self._setup_camera_network()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
    
    def _setup_camera_network(self):
        """Setup default camera network"""
        # Register default cameras
        cameras = [
            ('camera-1', [100, 100], 80),
            ('camera-2', [300, 200], 80),
            ('camera-3', [500, 150], 80)
        ]
        
        for camera_id, position, coverage in cameras:
            multicamera_engine.register_camera(camera_id, position, coverage)
    
    def start_stream(self, stream_id, source, missing_persons_data):
        """Start processing stream with multi-camera correlation"""
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
                'multicamera_enabled': True
            }
            
            # Start stream thread
            stream_thread = threading.Thread(
                target=self._stream_worker, 
                args=(stream_id,), 
                daemon=True
            )
            stream_thread.start()
            
            self.socketio.emit('multicamera_stream_started', {
                'stream_id': stream_id,
                'status': 'active',
                'multicamera_enabled': True,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"Error starting multi-camera stream {stream_id}: {e}")
            return False
    
    def stop_stream(self, stream_id):
        """Stop processing stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['active'] = False
            self.active_streams[stream_id]['capture'].release()
            del self.active_streams[stream_id]
            
            self.socketio.emit('multicamera_stream_stopped', {
                'stream_id': stream_id,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
        return False
    
    def _stream_worker(self, stream_id):
        """Worker thread for multi-camera stream processing"""
        stream_data = self.active_streams[stream_id]
        cap = stream_data['capture']
        frame_count = 0
        
        while stream_data['active']:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame
            frame_count += 1
            if frame_count % 3 == 0:
                self.processing_queue.put({
                    'stream_id': stream_id,
                    'frame': frame,
                    'timestamp': time.time()
                })
            
            time.sleep(0.033)  # ~30 FPS
    
    def _process_frames(self):
        """Main processing thread with multi-camera correlation"""
        while True:
            try:
                if not self.processing_queue.empty():
                    frame_data = self.processing_queue.get()
                    self._analyze_frame_multicamera(frame_data)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Multi-camera frame processing error: {e}")
    
    def _analyze_frame_multicamera(self, frame_data):
        """Analyze frame with multi-camera correlation"""
        stream_id = frame_data['stream_id']
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        if stream_id not in self.active_streams:
            return
        
        missing_persons = self.active_streams[stream_id]['missing_persons']
        
        try:
            # Process through analytics engine
            analytics_data = analytics_engine.process_frame(
                frame, stream_id, datetime.fromtimestamp(timestamp)
            )
            
            # Process detections through multi-camera engine
            multicamera_results = []
            
            for detection in analytics_data['detections']:
                # Multi-camera correlation
                correlation_result = multicamera_engine.process_detection(
                    stream_id, detection, datetime.fromtimestamp(timestamp)
                )
                
                multicamera_results.append(correlation_result)
                
                # Check for missing person matches
                x, y, w, h = detection['bbox']
                confidence = detection.get('confidence', 0.5)
                
                if confidence < self.alert_threshold:
                    continue
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Compare with missing persons
                for person in missing_persons:
                    match_confidence = self._compare_faces(face_region, person)
                    
                    if match_confidence > self.alert_threshold:
                        self._send_multicamera_alert(
                            stream_id, person, match_confidence, timestamp, 
                            detection, correlation_result
                        )
            
            # Send multi-camera update
            self.socketio.emit('multicamera_update', {
                'stream_id': stream_id,
                'analytics': analytics_data,
                'correlations': multicamera_results,
                'timestamp': datetime.fromtimestamp(timestamp).isoformat()
            })
        
        except Exception as e:
            print(f"Multi-camera analysis error: {e}")
    
    def _compare_faces(self, face_frame, person_data):
        """Compare detected face with missing person"""
        try:
            if ENHANCED_AI_AVAILABLE:
                return self.ai_engine.compare_faces_facenet(face_frame, person_data['photo_path'])
            else:
                return self.ai_engine.compare_faces(face_frame, person_data['photo_path'])
        except:
            return 0.0
    
    def _send_multicamera_alert(self, stream_id, person, confidence, timestamp, detection, correlation):
        """Send enhanced multi-camera alert"""
        current_time = time.time()
        last_alert = self.active_streams[stream_id]['last_alert']
        
        # Prevent spam alerts
        if current_time - last_alert < 10:
            return
        
        self.active_streams[stream_id]['last_alert'] = current_time
        
        # Enhanced alert with multi-camera context
        alert_data = {
            'type': 'multicamera_person_detected',
            'stream_id': stream_id,
            'person_id': person['id'],
            'person_name': person['name'],
            'confidence': round(confidence * 100, 2),
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'location': detection['bbox'],
            'multicamera_context': {
                'global_track_id': correlation['global_track_id'],
                'correlation_confidence': correlation['correlation_confidence'],
                'transition': correlation['transition'],
                'cross_camera_history': self._get_cross_camera_history(correlation['global_track_id'])
            }
        }
        
        # Send enhanced WebSocket alert
        self.socketio.emit('multicamera_alert', alert_data)
        
        # Log enhanced alert
        self._log_multicamera_alert(alert_data)
        
        print(f"🚨 MULTI-CAMERA ALERT: {person['name']} detected")
        print(f"   Global Track: {correlation['global_track_id']}")
        print(f"   Correlation: {correlation['correlation_confidence']:.2f}")
        
        if correlation['transition']:
            print(f"   Transition: {correlation['transition']['from_camera']} → {correlation['transition']['to_camera']}")
    
    def _get_cross_camera_history(self, global_track_id):
        """Get cross-camera history for track"""
        try:
            route = multicamera_engine.get_route_reconstruction(global_track_id)
            if route:
                return {
                    'cameras_visited': route['camera_count'],
                    'total_duration': route['total_duration'],
                    'camera_sequence': list(route['camera_segments'].keys())
                }
        except:
            pass
        
        return {'cameras_visited': 1, 'total_duration': 0, 'camera_sequence': []}
    
    def _log_multicamera_alert(self, alert_data):
        """Log multi-camera alert"""
        try:
            # Enhanced logging with multi-camera context
            pass
        except Exception as e:
            print(f"Multi-camera alert logging error: {e}")
    
    def get_multicamera_summary(self):
        """Get multi-camera system summary"""
        correlations = multicamera_engine.get_camera_correlations()
        global_tracks = multicamera_engine.get_global_tracks(1)  # Last hour
        
        return {
            'active_streams': len(self.active_streams),
            'queue_size': self.processing_queue.qsize(),
            'ai_engine': 'Enhanced' if ENHANCED_AI_AVAILABLE else 'Basic',
            'multicamera_enabled': True,
            'cameras_in_network': correlations['total_cameras'],
            'active_global_tracks': len(global_tracks),
            'total_transitions': correlations['total_transitions']
        }
    
    def get_stream_correlations(self, stream_id):
        """Get correlations for specific stream"""
        if stream_id not in self.active_streams:
            return None
        
        global_tracks = multicamera_engine.get_global_tracks(24)
        
        # Filter tracks that involve this camera
        relevant_tracks = {
            track_id: track_data 
            for track_id, track_data in global_tracks.items()
            if stream_id in track_data['cameras']
        }
        
        return {
            'stream_id': stream_id,
            'relevant_tracks': relevant_tracks,
            'track_count': len(relevant_tracks)
        }

# Global enhanced multi-camera processor instance
enhanced_multicamera_processor = None