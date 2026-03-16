# Live CCTV Stream Processor
import cv2
import time
from gpu_cnn_detector import GPUCNNDetector
import face_recognition
import numpy as np

class LiveCCTVProcessor:
    """Real-time CCTV stream processing"""
    
    def __init__(self, case_id):
        self.case_id = case_id
        self.gpu_detector = GPUCNNDetector()
        self.target_encodings = []  # Load from case
        self.alert_callback = None
        
    def process_live_stream(self, stream_url):
        """
        Process live CCTV stream
        
        stream_url examples:
        - RTSP: "rtsp://username:password@ip:port/stream"
        - HTTP: "http://ip:port/video"
        - Webcam: 0
        """
        
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print(f"Cannot open stream: {stream_url}")
            return
        
        print(f"Live stream started: {stream_url}")
        
        frame_count = 0
        fps_list = []
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or error")
                break
            
            # Process every frame (real-time)
            detections = self.gpu_detector.detect_faces_gpu_cnn(frame)
            
            # Check for target person
            for detection in detections:
                x, y, w, h = detection['bbox']
                
                # Match with target
                confidence = self._match_target(frame, (x, y, w, h))
                
                if confidence > 0.7:  # High confidence match
                    print(f"⚠ ALERT: Target detected! Confidence: {confidence:.2f}")
                    
                    # Draw box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"MATCH: {confidence:.2f}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 0, 255), 2)
                    
                    # Trigger alert
                    if self.alert_callback:
                        self.alert_callback(frame, confidence, detection)
                else:
                    # Draw normal box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            fps_list.append(fps)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Live CCTV - Press Q to quit', frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                avg_fps = sum(fps_list[-100:]) / len(fps_list[-100:])
                print(f"Frames: {frame_count}, Avg FPS: {avg_fps:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"\nFinal Stats:")
            print(f"  Total Frames: {frame_count}")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Processing: {'Real-time' if avg_fps >= 25 else 'Near real-time'}")
    
    def _match_target(self, frame, bbox):
        """Match detected face with target"""
        try:
            x, y, w, h = bbox
            face_region = frame[y:y+h, x:x+w]
            
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if face_encodings and self.target_encodings:
                distances = face_recognition.face_distance(
                    self.target_encodings, face_encodings[0]
                )
                
                if len(distances) > 0:
                    min_distance = min(distances)
                    if min_distance < 0.4:
                        return 1.0 - min_distance
            
            return 0.0
        except:
            return 0.0
    
    def set_alert_callback(self, callback):
        """Set callback function for alerts"""
        self.alert_callback = callback

# Test live stream
def test_live_stream():
    """Test with webcam"""
    
    print("="*60)
    print("LIVE CCTV STREAM TEST")
    print("="*60)
    
    processor = LiveCCTVProcessor(case_id=1)
    
    # Test with webcam (0) or RTSP stream
    stream_url = 0  # Webcam
    # stream_url = "rtsp://username:password@192.168.1.100:554/stream"
    
    print("\nStarting live stream...")
    print("Press 'Q' to quit\n")
    
    processor.process_live_stream(stream_url)
    
    print("\nStream ended")
    print("="*60)

if __name__ == "__main__":
    test_live_stream()
