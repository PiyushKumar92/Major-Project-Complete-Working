# GPU-Accelerated CNN Face Detector using PyTorch
import torch
import torch.nn as nn
import cv2
import numpy as np
import face_recognition
from facenet_pytorch import MTCNN
import logging

logger = logging.getLogger(__name__)

class GPUCNNDetector:
    """GPU-accelerated CNN face detection using PyTorch"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"GPU CNN Detector using: {self.device}")
        
        # Initialize MTCNN (Multi-task CNN) for face detection
        try:
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                min_face_size=20,  # Detect small faces
                thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
                post_process=False
            )
            print("MTCNN (GPU CNN) initialized successfully")
        except Exception as e:
            print(f"MTCNN init failed: {e}")
            self.mtcnn = None
    
    def detect_faces_gpu_cnn(self, frame):
        """Detect faces using GPU-accelerated CNN"""
        
        if self.mtcnn is None:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using MTCNN (GPU CNN)
            boxes, probs = self.mtcnn.detect(rgb_frame)
            
            detections = []
            
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob > 0.9:  # High confidence only
                        x1, y1, x2, y2 = box.astype(int)
                        w, h = x2 - x1, y2 - y1
                        
                        detections.append({
                            'bbox': (x1, y1, w, h),
                            'confidence': float(prob),
                            'method': 'gpu_cnn_mtcnn',
                            'type': 'face'
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"GPU CNN detection error: {e}")
            return []
    
    def match_face_gpu(self, frame, bbox, target_encodings):
        """Match detected face with target using GPU"""
        
        try:
            x, y, w, h = bbox
            face_region = frame[y:y+h, x:x+w]
            
            # Convert to RGB
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            face_locations = [(0, w, h, 0)]  # Full region
            face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
            
            if not face_encodings or not target_encodings:
                return 0.0
            
            # Compare with targets
            distances = face_recognition.face_distance(target_encodings, face_encodings[0])
            
            if len(distances) > 0:
                min_distance = min(distances)
                
                # Convert distance to confidence
                if min_distance < 0.4:
                    confidence = 1.0 - min_distance
                    return confidence
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Face matching error: {e}")
            return 0.0

# Install required package
def install_facenet():
    """Install facenet-pytorch for GPU CNN"""
    import subprocess
    import sys
    
    print("Installing facenet-pytorch for GPU CNN...")
    subprocess.run([sys.executable, "-m", "pip", "install", "facenet-pytorch"])
    print("Installation complete!")

# Test function
def test_gpu_cnn():
    """Test GPU CNN detector"""
    
    print("="*60)
    print("TESTING GPU CNN FACE DETECTOR")
    print("="*60)
    
    # Check GPU
    import torch
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Initialize detector
    detector = GPUCNNDetector()
    
    # Test on webcam or video
    import time
    
    cap = cv2.VideoCapture(0)  # Webcam
    
    if not cap.isOpened():
        print("Cannot open webcam, testing on sample image...")
        # Create test image
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start = time.time()
        detections = detector.detect_faces_gpu_cnn(test_frame)
        end = time.time()
        
        print(f"\nTest Results:")
        print(f"  Detections: {len(detections)}")
        print(f"  Time: {(end-start)*1000:.2f}ms")
        print(f"  FPS: {1/(end-start):.1f}")
    else:
        print("\nTesting on webcam (10 frames)...")
        fps_list = []
        
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            detections = detector.detect_faces_gpu_cnn(frame)
            end = time.time()
            
            fps = 1 / (end - start)
            fps_list.append(fps)
            
            print(f"Frame {i+1}: {len(detections)} faces, {fps:.1f} FPS")
        
        cap.release()
        
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"\nAverage FPS: {avg_fps:.1f}")
            
            if avg_fps > 25:
                print("✓ GPU CNN is FAST!")
            else:
                print("⚠ Performance could be better")
    
    print("="*60)

if __name__ == "__main__":
    # Check if facenet-pytorch is installed
    try:
        import facenet_pytorch
        print("facenet-pytorch already installed")
    except ImportError:
        print("Installing facenet-pytorch...")
        install_facenet()
    
    # Test GPU CNN
    test_gpu_cnn()
