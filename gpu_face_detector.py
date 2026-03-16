# GPU-Accelerated Face Detection using PyTorch
import torch
import cv2
import numpy as np

class GPUFaceDetector:
    """Fast GPU-based face detection using PyTorch"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained face detection model
        try:
            # Using OpenCV's DNN module with GPU
            self.net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt',
                'res10_300x300_ssd_iter_140000.caffemodel'
            )
            if torch.cuda.is_available():
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("GPU acceleration enabled for face detection")
            else:
                print("Using CPU for face detection")
        except:
            print("Model files not found, using cascade")
            self.net = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces_gpu(self, frame, confidence_threshold=0.5):
        """Detect faces using GPU acceleration"""
        
        if self.net is not None:
            # DNN-based detection (GPU accelerated)
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    faces.append((x, y, x2-x, y2-y))
            
            return faces
        else:
            # Fallback to cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces

# Download models if needed
def download_models():
    """Download required model files"""
    import urllib.request
    import os
    
    print("Downloading face detection models...")
    
    # Prototxt
    if not os.path.exists('deploy.prototxt'):
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
        urllib.request.urlretrieve(url, 'deploy.prototxt')
        print("Downloaded deploy.prototxt")
    
    # Model weights
    if not os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
        url = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
        urllib.request.urlretrieve(url, 'res10_300x300_ssd_iter_140000.caffemodel')
        print("Downloaded model weights")
    
    print("Models ready!")

if __name__ == "__main__":
    # Download models
    download_models()
    
    # Test GPU detection
    detector = GPUFaceDetector()
    
    # Test on webcam or video
    import time
    
    cap = cv2.VideoCapture(0)  # Webcam
    
    fps_list = []
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        faces = detector.detect_faces_gpu(frame)
        end = time.time()
        
        fps = 1 / (end - start)
        fps_list.append(fps)
        
        print(f"Frame {i}: {len(faces)} faces, {fps:.1f} FPS")
    
    cap.release()
    
    if fps_list:
        print(f"\nAverage FPS: {sum(fps_list)/len(fps_list):.1f}")
