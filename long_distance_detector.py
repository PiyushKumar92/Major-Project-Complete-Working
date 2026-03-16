# Long Distance Face Detection Enhancement
import cv2
import numpy as np
import face_recognition

class LongDistanceDetector:
    """Enhanced detection for long distance/small faces"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    def enhance_frame_for_distance(self, frame):
        """Enhance frame for better long distance detection"""
        
        # 1. Upscale image (2x)
        height, width = frame.shape[:2]
        upscaled = cv2.resize(frame, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # 2. Denoise
        denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
        
        # 3. Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 4. Contrast enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l,a,b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_small_faces(self, frame):
        """Detect small/distant faces"""
        
        # Enhance frame
        enhanced = self.enhance_frame_for_distance(frame)
        
        detections = []
        
        # Multiple scale detection for small faces
        scales = [1.01, 1.02, 1.03, 1.05]
        min_neighbors_list = [2, 3, 4]
        
        for scale in scales:
            for min_neighbors in min_neighbors_list:
                # Frontal faces
                faces = self.face_cascade.detectMultiScale(
                    cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY),
                    scaleFactor=scale,
                    minNeighbors=min_neighbors,
                    minSize=(10, 10),  # Very small faces
                    maxSize=(500, 500)
                )
                
                for (x, y, w, h) in faces:
                    # Scale back to original coordinates
                    x, y, w, h = x//2, y//2, w//2, h//2
                    detections.append({
                        'bbox': (x, y, w, h),
                        'method': f'enhanced_cascade_{scale}',
                        'type': 'small_face'
                    })
        
        # HOG on enhanced image
        try:
            rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_enhanced, 
                model='hog',
                number_of_times_to_upsample=2  # More upsampling for small faces
            )
            
            for (top, right, bottom, left) in face_locations:
                # Scale back
                top, right, bottom, left = top//2, right//2, bottom//2, left//2
                detections.append({
                    'bbox': (left, top, right-left, bottom-top),
                    'method': 'hog_enhanced',
                    'type': 'small_face'
                })
        except:
            pass
        
        return detections, enhanced

# Test function
def test_long_distance():
    import os
    
    print("Testing Long Distance Detection...")
    
    detector = LongDistanceDetector()
    
    # Test on surveillance video
    video_path = "static/surveillance"
    if os.path.exists(video_path):
        videos = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
        
        if videos:
            test_video = os.path.join(video_path, videos[0])
            cap = cv2.VideoCapture(test_video)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Frame size: {frame.shape}")
                    
                    # Normal detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    normal_faces = detector.face_cascade.detectMultiScale(gray)
                    print(f"Normal detection: {len(normal_faces)} faces")
                    
                    # Enhanced detection
                    enhanced_detections, enhanced_frame = detector.detect_small_faces(frame)
                    print(f"Enhanced detection: {len(enhanced_detections)} faces")
                    
                    # Save comparison
                    cv2.imwrite("static/temp/normal_frame.jpg", frame)
                    cv2.imwrite("static/temp/enhanced_frame.jpg", enhanced_frame)
                    print("Saved comparison images in static/temp/")
            
            cap.release()
    
    print("Test complete!")

if __name__ == "__main__":
    test_long_distance()
