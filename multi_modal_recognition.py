"""
Multi-Modal Person Recognition System
Enhanced recognition with facial recognition, age progression, clothing analysis, and biometric features
"""

import cv2
import numpy as np
import face_recognition
import os
import json
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import mediapipe as mp

class MultiModalRecognizer:
    """Advanced multi-modal person recognition system"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Recognition thresholds
        self.face_threshold = 0.6
        self.clothing_threshold = 0.7
        self.biometric_threshold = 0.8
        
    def create_person_profile(self, case_id, image_paths):
        """Create comprehensive person profile with all modalities"""
        profile = {
            'case_id': case_id,
            'created_at': datetime.now().isoformat(),
            'facial_features': {},
            'clothing_patterns': {},
            'biometric_features': {},
            'age_progression_data': {},
            'confidence_scores': {}
        }
        
        all_face_encodings = []
        all_clothing_features = []
        all_biometric_features = []
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # 1. Facial Recognition with Age Progression
                face_data = self._extract_facial_features(image)
                if face_data:
                    all_face_encodings.append(face_data['encoding'])
                    profile['facial_features'] = face_data
                
                # 2. Clothing Pattern Analysis
                clothing_data = self._analyze_clothing_patterns(image)
                if clothing_data:
                    all_clothing_features.append(clothing_data)
                
                # 3. Biometric Feature Extraction
                biometric_data = self._extract_biometric_features(image)
                if biometric_data:
                    all_biometric_features.append(biometric_data)
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        # Aggregate features
        if all_face_encodings:
            profile['facial_features']['primary_encoding'] = np.mean(all_face_encodings, axis=0).tolist()
            profile['facial_features']['all_encodings'] = [enc.tolist() for enc in all_face_encodings]
        
        if all_clothing_features:
            profile['clothing_patterns'] = self._aggregate_clothing_features(all_clothing_features)
        
        if all_biometric_features:
            profile['biometric_features'] = self._aggregate_biometric_features(all_biometric_features)
        
        # Calculate overall confidence
        profile['confidence_scores'] = self._calculate_profile_confidence(profile)
        
        return profile
    
    def _extract_facial_features(self, image):
        """Extract facial features with age progression analysis"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return None
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if not face_encodings:
                return None
            
            # Get the largest face
            largest_face_idx = 0
            if len(face_locations) > 1:
                areas = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
                largest_face_idx = np.argmax(areas)
            
            face_location = face_locations[largest_face_idx]
            face_encoding = face_encodings[largest_face_idx]
            
            # Extract face region for detailed analysis
            top, right, bottom, left = face_location
            face_image = rgb_image[top:bottom, left:right]
            
            # Age progression features
            age_features = self._analyze_age_features(face_image)
            
            # Face quality assessment
            quality_score = self._assess_face_quality(face_image)
            
            return {
                'encoding': face_encoding,
                'location': face_location,
                'age_features': age_features,
                'quality_score': quality_score,
                'face_landmarks': self._extract_face_landmarks(face_image)
            }
            
        except Exception as e:
            print(f"Error in facial feature extraction: {e}")
            return None
    
    def _analyze_age_features(self, face_image):
        """Analyze age-related facial features for progression"""
        try:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # Wrinkle detection using edge detection
            edges = cv2.Canny(gray_face, 50, 150)
            wrinkle_density = np.sum(edges > 0) / edges.size
            
            # Skin texture analysis
            texture_variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Face shape analysis (for age progression)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                face_area = cv2.contourArea(largest_contour)
                face_perimeter = cv2.arcLength(largest_contour, True)
                roundness = 4 * np.pi * face_area / (face_perimeter ** 2) if face_perimeter > 0 else 0
            else:
                roundness = 0
            
            return {
                'wrinkle_density': float(wrinkle_density),
                'texture_variance': float(texture_variance),
                'face_roundness': float(roundness),
                'estimated_age_group': self._estimate_age_group(wrinkle_density, texture_variance)
            }
            
        except Exception as e:
            print(f"Error in age analysis: {e}")
            return {}
    
    def _estimate_age_group(self, wrinkle_density, texture_variance):
        """Estimate age group based on facial features"""
        if wrinkle_density < 0.1 and texture_variance > 1000:
            return "young_adult"  # 18-30
        elif wrinkle_density < 0.2 and texture_variance > 500:
            return "adult"  # 30-50
        elif wrinkle_density < 0.3:
            return "middle_aged"  # 50-65
        else:
            return "senior"  # 65+
    
    def _extract_face_landmarks(self, face_image):
        """Extract facial landmarks for detailed matching"""
        try:
            # Use dlib or MediaPipe for landmark detection
            # For now, return basic geometric features
            h, w = face_image.shape[:2]
            
            # Calculate basic proportions
            return {
                'face_width': w,
                'face_height': h,
                'aspect_ratio': w / h if h > 0 else 0,
                'center_point': (w // 2, h // 2)
            }
        except:
            return {}
    
    def _analyze_clothing_patterns(self, image):
        """Advanced clothing pattern analysis with seasonal variations"""
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Detect person region (simplified - use upper body)
            h, w = image.shape[:2]
            upper_body = image[h//4:3*h//4, w//4:3*w//4]  # Focus on torso area
            
            if upper_body.size == 0:
                return None
            
            # Color analysis
            dominant_colors = self._extract_dominant_colors(upper_body)
            
            # Texture analysis
            texture_features = self._analyze_clothing_texture(upper_body)
            
            # Pattern detection
            patterns = self._detect_clothing_patterns(upper_body)
            
            return {
                'dominant_colors': dominant_colors,
                'texture_features': texture_features,
                'patterns': patterns,
                'seasonal_category': self._classify_seasonal_clothing(dominant_colors, texture_features)
            }
            
        except Exception as e:
            print(f"Error in clothing analysis: {e}")
            return None
    
    def _extract_dominant_colors(self, clothing_region, n_colors=5):
        """Extract dominant colors from clothing region"""
        try:
            # Reshape image for k-means
            data = clothing_region.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8 and get color percentages
            centers = np.uint8(centers)
            
            # Calculate color percentages
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels)
            
            colors_with_percentages = []
            for i, (color, percentage) in enumerate(zip(centers, percentages)):
                colors_with_percentages.append({
                    'color_bgr': color.tolist(),
                    'color_hex': '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0]),
                    'percentage': float(percentage)
                })
            
            # Sort by percentage
            colors_with_percentages.sort(key=lambda x: x['percentage'], reverse=True)
            
            return colors_with_percentages
            
        except Exception as e:
            print(f"Error extracting dominant colors: {e}")
            return []
    
    def _analyze_clothing_texture(self, clothing_region):
        """Analyze clothing texture patterns"""
        try:
            gray_clothing = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern for texture
            lbp = self._calculate_lbp(gray_clothing)
            
            # Gabor filter responses for texture analysis
            gabor_responses = self._apply_gabor_filters(gray_clothing)
            
            # Edge density for fabric texture
            edges = cv2.Canny(gray_clothing, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            return {
                'lbp_histogram': lbp.tolist() if lbp is not None else [],
                'gabor_energy': gabor_responses,
                'edge_density': float(edge_density),
                'texture_roughness': float(cv2.Laplacian(gray_clothing, cv2.CV_64F).var())
            }
            
        except Exception as e:
            print(f"Error in texture analysis: {e}")
            return {}
    
    def _calculate_lbp(self, gray_image):
        """Calculate Local Binary Pattern"""
        try:
            rows, cols = gray_image.shape
            lbp_image = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = gray_image[i, j]
                    code = 0
                    
                    # 8-neighborhood
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp_image[i-1, j-1] = code
            
            # Calculate histogram
            hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            return hist
            
        except Exception as e:
            print(f"Error calculating LBP: {e}")
            return None
    
    def _apply_gabor_filters(self, gray_image):
        """Apply Gabor filters for texture analysis"""
        try:
            responses = []
            
            # Different orientations and frequencies
            orientations = [0, 45, 90, 135]
            frequencies = [0.1, 0.3, 0.5]
            
            for angle in orientations:
                for freq in frequencies:
                    kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                    responses.append(float(np.mean(filtered)))
            
            return responses
            
        except Exception as e:
            print(f"Error applying Gabor filters: {e}")
            return []
    
    def _detect_clothing_patterns(self, clothing_region):
        """Detect specific clothing patterns"""
        try:
            gray_clothing = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2GRAY)
            
            # Pattern detection using template matching or frequency analysis
            patterns = {
                'stripes': self._detect_stripes(gray_clothing),
                'checks': self._detect_checks(gray_clothing),
                'solid': self._detect_solid_color(gray_clothing),
                'textured': self._detect_textured_pattern(gray_clothing)
            }
            
            # Determine dominant pattern
            dominant_pattern = max(patterns.items(), key=lambda x: x[1])
            
            return {
                'pattern_scores': patterns,
                'dominant_pattern': dominant_pattern[0],
                'pattern_confidence': dominant_pattern[1]
            }
            
        except Exception as e:
            print(f"Error detecting patterns: {e}")
            return {}
    
    def _detect_stripes(self, gray_image):
        """Detect stripe patterns using FFT"""
        try:
            # Apply FFT to detect periodic patterns
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Look for peaks indicating periodic patterns
            peaks = np.max(magnitude_spectrum) - np.mean(magnitude_spectrum)
            stripe_score = min(peaks / 10.0, 1.0)  # Normalize
            
            return float(stripe_score)
            
        except:
            return 0.0
    
    def _detect_checks(self, gray_image):
        """Detect checkered patterns"""
        try:
            # Use template matching for check patterns
            # Simplified: look for alternating intensity patterns
            rows, cols = gray_image.shape
            
            # Check horizontal and vertical variations
            h_variation = np.mean(np.abs(np.diff(gray_image, axis=1)))
            v_variation = np.mean(np.abs(np.diff(gray_image, axis=0)))
            
            check_score = min((h_variation + v_variation) / 100.0, 1.0)
            return float(check_score)
            
        except:
            return 0.0
    
    def _detect_solid_color(self, gray_image):
        """Detect solid color patterns"""
        try:
            # Calculate variance - low variance indicates solid color
            variance = np.var(gray_image)
            solid_score = max(0, 1.0 - variance / 1000.0)  # Inverse relationship
            
            return float(solid_score)
            
        except:
            return 0.0
    
    def _detect_textured_pattern(self, gray_image):
        """Detect textured patterns"""
        try:
            # Use edge density and local variance
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Local variance
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean) ** 2, -1, kernel)
            texture_score = min(np.mean(local_variance) / 1000.0, 1.0)
            
            return float(texture_score)
            
        except:
            return 0.0
    
    def _classify_seasonal_clothing(self, colors, texture_features):
        """Classify clothing as seasonal category"""
        try:
            if not colors:
                return "unknown"
            
            # Analyze dominant colors for seasonal classification
            dominant_color = colors[0]['color_bgr']
            b, g, r = dominant_color
            
            # Convert to HSV for better color analysis
            hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            hue, sat, val = hsv_color
            
            # Seasonal classification based on color and texture
            if val < 50:  # Dark colors
                if texture_features.get('texture_roughness', 0) > 500:
                    return "winter_heavy"  # Dark, thick clothing
                else:
                    return "formal"  # Dark, smooth clothing
            elif sat > 150:  # Bright colors
                return "summer_casual"  # Bright, casual clothing
            elif hue in range(20, 40):  # Earth tones
                return "autumn"  # Earth tone clothing
            else:
                return "spring_summer"  # Light, casual clothing
                
        except:
            return "unknown"
    
    def _extract_biometric_features(self, image):
        """Extract biometric features (height, build estimation)"""
        try:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect pose landmarks
            results = self.pose.process(rgb_image)
            
            if not results.pose_landmarks:
                return None
            
            # Extract key body landmarks
            landmarks = results.pose_landmarks.landmark
            h, w = image.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            key_points = {}
            for idx, landmark in enumerate(landmarks):
                key_points[idx] = {
                    'x': int(landmark.x * w),
                    'y': int(landmark.y * h),
                    'visibility': landmark.visibility
                }
            
            # Calculate biometric features
            biometric_data = self._calculate_body_measurements(key_points, h, w)
            
            return biometric_data
            
        except Exception as e:
            print(f"Error extracting biometric features: {e}")
            return None
    
    def _calculate_body_measurements(self, landmarks, image_height, image_width):
        """Calculate body measurements from pose landmarks"""
        try:
            # MediaPipe pose landmark indices
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            NOSE = 0
            
            measurements = {}
            
            # Shoulder width
            if LEFT_SHOULDER in landmarks and RIGHT_SHOULDER in landmarks:
                left_shoulder = landmarks[LEFT_SHOULDER]
                right_shoulder = landmarks[RIGHT_SHOULDER]
                shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
                measurements['shoulder_width_pixels'] = shoulder_width
                measurements['shoulder_width_ratio'] = shoulder_width / image_width
            
            # Body height (head to feet)
            if NOSE in landmarks and LEFT_ANKLE in landmarks and RIGHT_ANKLE in landmarks:
                nose = landmarks[NOSE]
                left_ankle = landmarks[LEFT_ANKLE]
                right_ankle = landmarks[RIGHT_ANKLE]
                avg_ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
                body_height = abs(avg_ankle_y - nose['y'])
                measurements['body_height_pixels'] = body_height
                measurements['body_height_ratio'] = body_height / image_height
            
            # Hip width
            if LEFT_HIP in landmarks and RIGHT_HIP in landmarks:
                left_hip = landmarks[LEFT_HIP]
                right_hip = landmarks[RIGHT_HIP]
                hip_width = abs(left_hip['x'] - right_hip['x'])
                measurements['hip_width_pixels'] = hip_width
                measurements['hip_width_ratio'] = hip_width / image_width
            
            # Body proportions
            if 'shoulder_width_ratio' in measurements and 'hip_width_ratio' in measurements:
                measurements['shoulder_hip_ratio'] = measurements['shoulder_width_ratio'] / measurements['hip_width_ratio']
            
            # Build estimation
            measurements['build_type'] = self._estimate_build_type(measurements)
            
            return measurements
            
        except Exception as e:
            print(f"Error calculating body measurements: {e}")
            return {}
    
    def _estimate_build_type(self, measurements):
        """Estimate body build type from measurements"""
        try:
            shoulder_hip_ratio = measurements.get('shoulder_hip_ratio', 1.0)
            shoulder_width_ratio = measurements.get('shoulder_width_ratio', 0.3)
            
            if shoulder_hip_ratio > 1.2 and shoulder_width_ratio > 0.35:
                return "athletic_broad"
            elif shoulder_hip_ratio > 1.1:
                return "athletic_medium"
            elif shoulder_hip_ratio < 0.9:
                return "pear_shaped"
            elif shoulder_width_ratio < 0.25:
                return "slim"
            elif shoulder_width_ratio > 0.4:
                return "broad"
            else:
                return "average"
                
        except:
            return "unknown"
    
    def _aggregate_clothing_features(self, all_clothing_features):
        """Aggregate clothing features from multiple images"""
        try:
            aggregated = {
                'dominant_colors': [],
                'common_patterns': {},
                'seasonal_categories': [],
                'texture_summary': {}
            }
            
            # Collect all colors
            all_colors = []
            for features in all_clothing_features:
                if 'dominant_colors' in features:
                    all_colors.extend(features['dominant_colors'])
            
            # Find most common colors
            if all_colors:
                # Group similar colors and find most frequent
                aggregated['dominant_colors'] = all_colors[:5]  # Top 5 colors
            
            # Aggregate patterns
            pattern_counts = {}
            for features in all_clothing_features:
                if 'patterns' in features and 'dominant_pattern' in features['patterns']:
                    pattern = features['patterns']['dominant_pattern']
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            aggregated['common_patterns'] = pattern_counts
            
            # Seasonal categories
            seasonal_counts = {}
            for features in all_clothing_features:
                if 'seasonal_category' in features:
                    category = features['seasonal_category']
                    seasonal_counts[category] = seasonal_counts.get(category, 0) + 1
            
            aggregated['seasonal_categories'] = seasonal_counts
            
            return aggregated
            
        except Exception as e:
            print(f"Error aggregating clothing features: {e}")
            return {}
    
    def _aggregate_biometric_features(self, all_biometric_features):
        """Aggregate biometric features from multiple images"""
        try:
            aggregated = {
                'average_measurements': {},
                'build_types': {},
                'confidence_score': 0.0
            }
            
            # Collect measurements
            measurements = ['shoulder_width_ratio', 'body_height_ratio', 'hip_width_ratio', 'shoulder_hip_ratio']
            
            for measurement in measurements:
                values = []
                for features in all_biometric_features:
                    if measurement in features:
                        values.append(features[measurement])
                
                if values:
                    aggregated['average_measurements'][measurement] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'samples': len(values)
                    }
            
            # Build type consensus
            build_counts = {}
            for features in all_biometric_features:
                if 'build_type' in features:
                    build_type = features['build_type']
                    build_counts[build_type] = build_counts.get(build_type, 0) + 1
            
            aggregated['build_types'] = build_counts
            
            # Calculate confidence based on consistency
            if build_counts:
                max_count = max(build_counts.values())
                total_count = sum(build_counts.values())
                aggregated['confidence_score'] = max_count / total_count
            
            return aggregated
            
        except Exception as e:
            print(f"Error aggregating biometric features: {e}")
            return {}
    
    def _calculate_profile_confidence(self, profile):
        """Calculate overall confidence scores for the profile"""
        try:
            confidence_scores = {
                'facial_confidence': 0.0,
                'clothing_confidence': 0.0,
                'biometric_confidence': 0.0,
                'overall_confidence': 0.0
            }
            
            # Facial confidence
            if 'facial_features' in profile and 'quality_score' in profile['facial_features']:
                confidence_scores['facial_confidence'] = profile['facial_features']['quality_score']
            
            # Clothing confidence
            if 'clothing_patterns' in profile and 'common_patterns' in profile['clothing_patterns']:
                patterns = profile['clothing_patterns']['common_patterns']
                if patterns:
                    max_pattern_count = max(patterns.values())
                    total_patterns = sum(patterns.values())
                    confidence_scores['clothing_confidence'] = max_pattern_count / total_patterns
            
            # Biometric confidence
            if 'biometric_features' in profile and 'confidence_score' in profile['biometric_features']:
                confidence_scores['biometric_confidence'] = profile['biometric_features']['confidence_score']
            
            # Overall confidence (weighted average)
            weights = [0.5, 0.3, 0.2]  # Face, clothing, biometric
            confidences = [
                confidence_scores['facial_confidence'],
                confidence_scores['clothing_confidence'],
                confidence_scores['biometric_confidence']
            ]
            
            confidence_scores['overall_confidence'] = sum(w * c for w, c in zip(weights, confidences))
            
            return confidence_scores
            
        except Exception as e:
            print(f"Error calculating confidence scores: {e}")
            return {}
    
    def _assess_face_quality(self, face_image):
        """Assess quality of face image for recognition"""
        try:
            if face_image.size == 0:
                return 0.0
            
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000, 1.0)
            
            # Brightness
            brightness = np.mean(gray_face)
            brightness_score = 1.0 - abs(brightness - 127) / 127
            
            # Contrast
            contrast = gray_face.std()
            contrast_score = min(contrast / 50, 1.0)
            
            # Size adequacy
            height, width = face_image.shape[:2]
            size_score = min((width * height) / (100 * 100), 1.0)
            
            # Combined quality score
            quality_score = (sharpness_score * 0.4 + brightness_score * 0.2 + 
                           contrast_score * 0.2 + size_score * 0.2)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            print(f"Error assessing face quality: {e}")
            return 0.0

# Global recognizer instance
multi_modal_recognizer = MultiModalRecognizer()