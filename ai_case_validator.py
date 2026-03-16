"""
AI-Powered Case Validation System
Automatically validates and approves/rejects cases based on multiple parameters
"""

import cv2
import numpy as np
import os
import re
from datetime import datetime, timedelta
import face_recognition
from PIL import Image, ImageStat
import hashlib

# Optional imports with fallbacks
try:
    import requests
except ImportError:
    requests = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

class AIValidator:
    def __init__(self):
        self.validation_threshold = 0.75  # 75% confidence needed for auto-approval (higher for CCTV accuracy)
        self.rejection_reasons = []
        
    def validate_case(self, case):
        """Main validation function - returns (decision, confidence, reasons)"""
        self.rejection_reasons = []
        scores = {}
        
        # 1. Photo Validation
        photo_score = self._validate_photos(case)
        scores['photos'] = photo_score
        
        # 2. Form Data Validation
        form_score = self._validate_form_data(case)
        scores['form_data'] = form_score
        
        # 3. Text Content Analysis
        text_score = self._validate_text_content(case)
        scores['text_quality'] = text_score
        
        # 4. Video Validation (if provided)
        video_score = self._validate_videos(case)
        scores['videos'] = video_score
        
        # 5. Consistency Check
        consistency_score = self._check_consistency(case)
        scores['consistency'] = consistency_score
        
        # 6. Spam/Fraud Detection
        fraud_score = self._detect_fraud(case)
        scores['fraud_check'] = fraud_score
        
        # Additional validation for CCTV optimization
        cctv_readiness_score = self._assess_cctv_readiness(case)
        scores['cctv_readiness'] = cctv_readiness_score
        
        # Optimized weights for advanced CCTV matching
        weights = {
            'photos': 0.40,        # Maximum priority - critical for CCTV matching
            'form_data': 0.20,     # Essential case information
            'text_quality': 0.08,  # Minimal - focus on visual data
            'videos': 0.12,        # Person verification in videos
            'consistency': 0.15,   # Cross-validation between media
            'fraud_check': 0.05    # Basic security check
        }
        
        # Calculate base confidence
        overall_confidence = sum(scores[key] * weights[key] for key in weights if key in scores)
        
        # CCTV readiness bonus/penalty
        if cctv_readiness_score > 0.8:
            overall_confidence += 0.05  # Bonus for excellent CCTV readiness
        elif cctv_readiness_score < 0.4:
            overall_confidence -= 0.1   # Penalty for poor CCTV readiness
        
        # Decision logic
        if overall_confidence >= self.validation_threshold:
            decision = 'APPROVE'
        else:
            decision = 'REJECT'
            
        # Generate smart feedback for rejected cases
        smart_feedback = None
        if decision == 'REJECT':
            smart_feedback = self.generate_smart_feedback(case, scores)
        
        return decision, overall_confidence, scores, self.rejection_reasons, smart_feedback
    
    def _assess_cctv_readiness(self, case):
        """Assess how ready the case is for CCTV matching"""
        readiness_score = 0.0
        
        if not case.target_images:
            return 0.0
        
        try:
            # Check photo quality for CCTV matching
            photo_readiness_scores = []
            
            for image in case.target_images:
                image_path = os.path.join('app', 'static', image.image_path.replace('static/', ''))
                
                if not os.path.exists(image_path):
                    continue
                
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    # 1. Face encoding success (critical for CCTV)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_img)
                    
                    if not face_locations:
                        continue  # Skip images without faces
                    
                    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                    if not face_encodings:
                        continue  # Skip if encoding fails
                    
                    encoding_score = 1.0  # Successfully encoded
                    
                    # 2. Face size adequacy for distant CCTV detection
                    best_face_location = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
                    top, right, bottom, left = best_face_location
                    face_width = right - left
                    face_height = bottom - top
                    
                    # Minimum size for CCTV detection (at least 60x60 pixels)
                    min_size_score = 1.0 if min(face_width, face_height) >= 60 else min(face_width, face_height) / 60
                    
                    # 3. Image resolution for detail preservation
                    img_height, img_width = img.shape[:2]
                    resolution_score = min((img_width * img_height) / (800 * 600), 1.0)  # At least 800x600
                    
                    # 4. Face clarity for feature matching
                    face_img = img[top:bottom, left:right]
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    clarity_score = min(cv2.Laplacian(gray_face, cv2.CV_64F).var() / 1000, 1.0)
                    
                    # 5. Lighting conditions
                    brightness = np.mean(gray_face)
                    lighting_score = 1.0 - abs(brightness - 127) / 127
                    
                    # 6. Face angle (frontal faces work better in CCTV)
                    try:
                        landmarks = face_recognition.face_landmarks(rgb_img, [best_face_location])
                        if landmarks and landmarks[0]:
                            # Simple frontal face check using eye positions
                            left_eye = landmarks[0].get('left_eye', [])
                            right_eye = landmarks[0].get('right_eye', [])
                            
                            if left_eye and right_eye:
                                left_eye_center = np.mean(left_eye, axis=0)
                                right_eye_center = np.mean(right_eye, axis=0)
                                eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
                                
                                # Good eye distance indicates frontal face
                                angle_score = min(eye_distance / 50, 1.0)  # Normalize
                            else:
                                angle_score = 0.7  # Default if landmarks incomplete
                        else:
                            angle_score = 0.6  # Default if no landmarks
                    except:
                        angle_score = 0.7  # Default on error
                    
                    # Calculate composite readiness score for this photo
                    photo_readiness = (
                        encoding_score * 0.25 +      # Must be encodable
                        min_size_score * 0.20 +      # Must be large enough
                        resolution_score * 0.15 +    # Good resolution
                        clarity_score * 0.20 +       # Clear features
                        lighting_score * 0.10 +      # Good lighting
                        angle_score * 0.10           # Good angle
                    )
                    
                    photo_readiness_scores.append(photo_readiness)
                    
                except Exception as e:
                    print(f"Error assessing photo readiness: {str(e)}")
                    continue
            
            if photo_readiness_scores:
                # Use the best photo's readiness score
                readiness_score = max(photo_readiness_scores)
                
                # Bonus for multiple good photos
                good_photos = len([score for score in photo_readiness_scores if score > 0.7])
                if good_photos > 1:
                    readiness_score += min(good_photos * 0.05, 0.15)  # Up to 15% bonus
            
            # Additional checks for CCTV matching
            
            # Multiple angles bonus (helps with varied CCTV angles)
            if len(case.target_images) >= 3:
                readiness_score += 0.05
            
            # Video consistency bonus (person appears in videos)
            if case.search_videos:
                try:
                    from advanced_cctv_matcher import cctv_matcher
                    ref_encodings, _ = cctv_matcher.extract_reference_encodings(case)
                    if ref_encodings:
                        readiness_score += 0.1  # Bonus for successful encoding extraction
                except:
                    pass
            
        except Exception as e:
            print(f"CCTV readiness assessment error: {str(e)}")
            readiness_score = 0.5  # Default moderate score on error
        
        return min(readiness_score, 1.0)
    
    def _validate_photos(self, case):
        """Validate uploaded photos"""
        if not case.target_images:
            self.rejection_reasons.append("[ERROR] No photos provided - At least one clear photo is required")
            return 0.0
        
        photo_scores = []
        
        for image in case.target_images:
            score = 0.0
            image_path = os.path.join('app', 'static', image.image_path.replace('static/', ''))
            
            if not os.path.exists(image_path):
                continue
                
            try:
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # 1. Image Quality Check
                quality_score = self._check_image_quality(img)
                score += quality_score * 0.3
                
                # 2. Face Detection
                face_score = self._detect_faces(img)
                score += face_score * 0.4
                
                # 3. AI Generated Detection
                ai_generated_score = self._detect_ai_generated(img)
                score += ai_generated_score * 0.2
                
                # 4. Image Authenticity
                authenticity_score = self._check_authenticity(img)
                score += authenticity_score * 0.1
                
                photo_scores.append(score)
                
            except Exception as e:
                print(f"Error validating photo: {str(e)}")
                continue
        
        if not photo_scores:
            self.rejection_reasons.append("[ERROR] No valid photos found - Please upload clear, readable images")
            return 0.0
        
        avg_score = sum(photo_scores) / len(photo_scores)
        
        if avg_score < 0.5:
            self.rejection_reasons.append("[ERROR] Photo quality too low - Please upload clearer, high-resolution photos")
        
        return avg_score
    
    def _check_image_quality(self, img):
        """Check image quality metrics"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_normalized = min(blur_score / 1000, 1.0)  # Normalize
        
        # 2. Brightness check
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 127) / 127  # Optimal around 127
        
        # 3. Contrast check
        contrast = gray.std()
        contrast_score = min(contrast / 64, 1.0)  # Normalize
        
        # 4. Resolution check
        height, width = img.shape[:2]
        resolution_score = min((width * height) / (640 * 480), 1.0)  # At least VGA
        
        return (blur_normalized + brightness_score + contrast_score + resolution_score) / 4
    
    def _detect_faces(self, img):
        """Advanced face detection and quality validation for CCTV matching"""
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Multiple face detection methods for better accuracy
            face_locations = face_recognition.face_locations(rgb_img, model="hog")
            
            # Fallback to CNN model if no faces found
            if not face_locations:
                try:
                    face_locations = face_recognition.face_locations(rgb_img, model="cnn")
                except:
                    pass
            
            if not face_locations:
                self.rejection_reasons.append("[ERROR] No clear faces detected in photos - Required for CCTV matching")
                return 0.0
            
            # Analyze each face for CCTV matching suitability
            face_scores = []
            best_face_info = None
            
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_img = rgb_img[top:bottom, left:right]
                face_height = bottom - top
                face_width = right - left
                
                # 1. Face size analysis (critical for CCTV detection)
                face_area = face_height * face_width
                img_area = img.shape[0] * img.shape[1]
                size_ratio = face_area / img_area
                
                # Minimum face size for CCTV matching (at least 8% of image)
                if size_ratio < 0.08:
                    continue  # Skip too small faces
                
                size_score = min(size_ratio / 0.15, 1.0)  # Optimal at 15%
                
                # 2. Face resolution check (minimum pixels for features)
                min_face_pixels = 80 * 80  # Minimum 80x80 pixels
                if face_area < min_face_pixels:
                    continue  # Skip low resolution faces
                
                resolution_score = min(face_area / (120 * 120), 1.0)  # Optimal at 120x120
                
                # 3. Face clarity and sharpness (critical for matching)
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                clarity_score = min(laplacian_var / 800, 1.0)  # Higher threshold for clarity
                
                if clarity_score < 0.3:
                    continue  # Skip blurry faces
                
                # 4. Face angle and pose analysis
                try:
                    # Check if face is frontal (better for matching)
                    face_landmarks = face_recognition.face_landmarks(rgb_img, [face_location])
                    if face_landmarks:
                        landmarks = face_landmarks[0]
                        
                        # Calculate face angle using nose and eye positions
                        nose_tip = landmarks['nose_tip'][2] if 'nose_tip' in landmarks else None
                        left_eye = landmarks['left_eye'][0] if 'left_eye' in landmarks else None
                        right_eye = landmarks['right_eye'][3] if 'right_eye' in landmarks else None
                        
                        pose_score = 1.0  # Default good pose
                        
                        if nose_tip and left_eye and right_eye:
                            # Check if face is roughly frontal
                            eye_center_x = (left_eye[0] + right_eye[0]) / 2
                            nose_offset = abs(nose_tip[0] - eye_center_x)
                            face_width_center = (left + right) / 2
                            
                            # Penalize extreme angles
                            if nose_offset > face_width * 0.15:
                                pose_score = 0.6  # Side angle penalty
                    else:
                        pose_score = 0.7  # No landmarks detected
                except:
                    pose_score = 0.8  # Default if landmark detection fails
                
                # 5. Lighting and contrast analysis
                brightness = np.mean(face_gray)
                contrast = face_gray.std()
                
                # Optimal brightness range (not too dark/bright)
                brightness_score = 1.0 - abs(brightness - 127) / 127
                brightness_score = max(brightness_score, 0.3)  # Minimum score
                
                # Good contrast for feature detection
                contrast_score = min(contrast / 50, 1.0)
                
                # 6. Face encoding quality (for actual matching)
                try:
                    face_encodings = face_recognition.face_encodings(rgb_img, [face_location])
                    if face_encodings:
                        encoding_score = 1.0  # Successfully encoded
                    else:
                        encoding_score = 0.0  # Failed to encode
                except:
                    encoding_score = 0.5  # Partial success
                
                # Calculate composite face score
                face_score = (
                    size_score * 0.25 +           # Face size importance
                    resolution_score * 0.20 +     # Resolution importance
                    clarity_score * 0.25 +        # Clarity critical for matching
                    pose_score * 0.15 +           # Pose affects matching
                    brightness_score * 0.10 +     # Lighting conditions
                    contrast_score * 0.05 +       # Feature visibility
                    encoding_score * 0.20         # Encoding success critical
                )
                
                face_scores.append(face_score)
                
                # Track best face for detailed feedback
                if not best_face_info or face_score > best_face_info['score']:
                    best_face_info = {
                        'score': face_score,
                        'size_ratio': size_ratio,
                        'clarity': clarity_score,
                        'resolution': face_area,
                        'pose': pose_score
                    }
            
            if not face_scores:
                self.rejection_reasons.append("[ERROR] No suitable faces found for CCTV matching - faces too small, blurry, or poorly lit")
                return 0.0
            
            best_score = max(face_scores)
            
            # Provide specific feedback for improvement
            if best_score < 0.6 and best_face_info:
                issues = []
                if best_face_info['size_ratio'] < 0.1:
                    issues.append("face too small in image")
                if best_face_info['clarity'] < 0.4:
                    issues.append("face not clear/sharp enough")
                if best_face_info['pose'] < 0.7:
                    issues.append("face angle not optimal (prefer frontal view)")
                
                if issues:
                    self.rejection_reasons.append(f"[WARNING] Face quality issues: {', '.join(issues)} - May affect CCTV detection accuracy")
            
            return best_score
            
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            self.rejection_reasons.append("[ERROR] Face detection failed - Please ensure clear face photos")
            return 0.0
    
    def _detect_ai_generated(self, img):
        """Advanced AI-generated image detection"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            ai_indicators = 0
            total_checks = 0
            
            # 1. Perfect symmetry check (AI faces often too symmetric)
            if width > 100:
                total_checks += 1
                left_half = gray[:, :width//2]
                right_half = cv2.flip(gray[:, width//2:], 1)
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                
                diff = cv2.absdiff(left_half, right_half)
                symmetry_score = 1.0 - (np.mean(diff) / 255.0)
                
                if symmetry_score > 0.92:  # Unnaturally symmetric
                    ai_indicators += 1
                    self.rejection_reasons.append("[WARNING] Photo appears artificially symmetric - possible AI generated")
            
            # 2. Unnatural smoothness (AI skin too perfect)
            total_checks += 1
            blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_variance < 30:  # Too smooth
                ai_indicators += 1
                self.rejection_reasons.append("[WARNING] Photo appears unnaturally smooth - possible AI generated")
            
            # 3. Pixel pattern analysis (AI artifacts)
            total_checks += 1
            # Check for repetitive patterns common in AI
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (width * height)
            
            if edge_density < 0.02:  # Too few natural edges
                ai_indicators += 1
                self.rejection_reasons.append("[WARNING] Photo lacks natural texture patterns - possible AI generated")
            
            # 4. Color distribution analysis
            total_checks += 1
            color_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_variance = np.var(color_hist)
            
            if color_variance < 1000:  # Too uniform color distribution
                ai_indicators += 1
                self.rejection_reasons.append("[WARNING] Photo has unnatural color distribution - possible AI generated")
            
            # 5. EXIF data check (AI images often lack proper EXIF)
            total_checks += 1
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                exif_data = pil_img._getexif()
                
                if not exif_data or len(exif_data) < 3:
                    ai_indicators += 0.5  # Partial indicator
                    self.rejection_reasons.append("[WARNING] Photo lacks camera metadata - possible AI generated or heavily processed")
            except:
                pass
            
            # Calculate AI probability
            ai_probability = ai_indicators / total_checks if total_checks > 0 else 0
            
            if ai_probability > 0.6:  # High AI probability
                return 0.1  # Very suspicious
            elif ai_probability > 0.4:  # Medium AI probability
                return 0.3  # Suspicious
            elif ai_probability > 0.2:  # Low AI probability
                return 0.6  # Questionable
            else:
                return 1.0  # Likely real
            
        except Exception as e:
            print(f"AI detection error: {str(e)}")
            return 0.7  # Default to likely real with some caution
    
    def _check_authenticity(self, img):
        """Basic authenticity checks"""
        try:
            # Check for common manipulation artifacts
            # 1. JPEG compression artifacts
            # 2. Inconsistent lighting
            # 3. Edge artifacts
            
            # For now, return high score (can be enhanced with more sophisticated detection)
            return 0.9
            
        except Exception:
            return 0.8
    
    def _validate_form_data(self, case):
        """Validate form data completeness and quality"""
        score = 0.0
        required_fields = ['person_name', 'last_seen_location', 'date_missing']
        
        # Check required fields
        missing_fields = []
        for field in required_fields:
            value = getattr(case, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field)
        
        if missing_fields:
            self.rejection_reasons.append(f"[ERROR] Missing required information: {', '.join(missing_fields)}")
            return 0.0
        
        # Name validation
        name_score = self._validate_name(case.person_name)
        score += name_score * 0.3
        
        # Location validation
        location_score = self._validate_location(case.last_seen_location)
        score += location_score * 0.3
        
        # Date validation
        date_score = self._validate_date(case.date_missing)
        score += date_score * 0.2
        
        # Age validation
        age_score = self._validate_age(case.age)
        score += age_score * 0.1
        
        # Contact info validation
        contact_score = self._validate_contact_info(case)
        score += contact_score * 0.1
        
        return score
    
    def _validate_name(self, name):
        """Validate person name"""
        if not name or len(name.strip()) < 2:
            self.rejection_reasons.append("[ERROR] Person name too short or invalid")
            return 0.0
        
        # Check for suspicious patterns
        if re.search(r'[0-9]{3,}', name):  # Too many numbers
            self.rejection_reasons.append("[ERROR] Person name contains suspicious number patterns")
            return 0.3
        
        if len(name) > 100:
            self.rejection_reasons.append("[ERROR] Person name too long")
            return 0.5
        
        # Check for reasonable character set
        if not re.match(r'^[a-zA-Z\s\.\-\']+$', name) and not re.match(r'^[\u0900-\u097F\s\.\-\']+$', name):
            # Allow English and Hindi characters
            self.rejection_reasons.append("[ERROR] Person name contains invalid characters")
            return 0.4
        
        return 1.0
    
    def _validate_location(self, location):
        """Validate location information"""
        if not location or len(location.strip()) < 5:
            self.rejection_reasons.append("[ERROR] Location information too brief - Please provide detailed location")
            return 0.0
        
        # Check for meaningful location info
        location_keywords = ['street', 'road', 'area', 'city', 'town', 'village', 'district', 'state', 'near', 'opposite', 'behind']
        hindi_keywords = ['गली', 'सड़क', 'इलाका', 'शहर', 'गांव', 'जिला', 'राज्य', 'के पास', 'के सामने']
        
        location_lower = location.lower()
        has_keywords = any(keyword in location_lower for keyword in location_keywords + hindi_keywords)
        
        if not has_keywords and len(location) < 20:
            self.rejection_reasons.append("[ERROR] Location information not detailed enough - Please provide specific landmarks, area names, or addresses")
            return 0.4
        
        return 1.0
    
    def _validate_date(self, date_missing):
        """Validate missing date"""
        if not date_missing:
            return 0.5  # Optional field
        
        try:
            # Check if date is not in future
            if date_missing > datetime.now().date():
                self.rejection_reasons.append("[ERROR] Missing date cannot be in the future")
                return 0.0
            
            # Check if date is not too old (more than 10 years)
            ten_years_ago = datetime.now().date() - timedelta(days=3650)
            if date_missing < ten_years_ago:
                self.rejection_reasons.append("[ERROR] Missing date is too old (more than 10 years)")
                return 0.2
            
            return 1.0
            
        except Exception:
            self.rejection_reasons.append("[ERROR] Invalid date format")
            return 0.0
    
    def _validate_age(self, age):
        """Validate age information"""
        if age is None:
            return 0.8  # Optional field
        
        if age < 0 or age > 120:
            self.rejection_reasons.append("[ERROR] Age must be between 0 and 120 years")
            return 0.0
        
        return 1.0
    
    def _validate_contact_info(self, case):
        """Validate contact information"""
        score = 0.0
        
        # Check if contact details are provided in case details
        if hasattr(case, 'details') and case.details:
            details_lower = case.details.lower()
            
            # Look for phone number patterns
            phone_pattern = r'[\+]?[1-9]?[\d\s\-\(\)]{9,15}'
            has_phone = bool(re.search(phone_pattern, case.details))
            
            # Look for email patterns
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            has_email = bool(re.search(email_pattern, case.details))
            
            if has_phone or has_email:
                score = 1.0
            else:
                score = 0.5
        else:
            score = 0.3
        
        return score
    
    def _validate_text_content(self, case):
        """Validate text content quality"""
        if not case.details:
            self.rejection_reasons.append("[ERROR] No case details provided - Please describe the circumstances")
            return 0.0
        
        details = case.details.strip()
        
        # Length check
        if len(details) < 50:
            self.rejection_reasons.append("[ERROR] Case details too brief - Please provide more detailed information")
            return 0.2
        
        # Check for meaningful content
        words = details.split()
        if len(words) < 10:
            self.rejection_reasons.append("[ERROR] Case details not detailed enough")
            return 0.3
        
        # Check for spam patterns
        spam_indicators = ['click here', 'visit website', 'call now', 'limited time', 'act fast']
        spam_count = sum(1 for indicator in spam_indicators if indicator.lower() in details.lower())
        
        if spam_count > 2:
            self.rejection_reasons.append("[ERROR] Content appears to be spam")
            return 0.1
        
        # Language quality check (basic)
        if TextBlob:
            try:
                blob = TextBlob(details)
                # Check if text makes sense (very basic check)
                if len(blob.sentences) == 0:
                    return 0.4
            except:
                pass
        
        return 1.0
    
    def _validate_videos(self, case):
        """Advanced video validation for person identification"""
        if not case.search_videos:
            return 0.8  # Videos are optional but helpful
        
        video_scores = []
        person_consistency_scores = []
        
        for video in case.search_videos:
            video_path = os.path.join('app', 'static', video.video_path.replace('static/', ''))
            
            if not os.path.exists(video_path):
                continue
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                score = 0.0
                
                # 1. Duration and technical quality
                if 2 <= duration <= 180:  # 2 seconds to 3 minutes optimal
                    score += 0.2
                elif 1 <= duration <= 300:  # Acceptable range
                    score += 0.15
                
                # 2. Resolution check (important for person detection)
                if width >= 640 and height >= 480:  # Minimum VGA
                    score += 0.15
                elif width >= 320 and height >= 240:  # Low but usable
                    score += 0.1
                
                # 3. Frame rate check
                if fps >= 15:  # Good frame rate
                    score += 0.1
                elif fps >= 10:  # Acceptable
                    score += 0.05
                
                # 4. Advanced frame analysis for person detection
                sample_frames = min(10, max(5, frame_count // 10))  # More samples
                frame_scores = []
                person_detected_frames = 0
                face_detected_frames = 0
                
                # Get reference face encodings from case photos
                reference_encodings = []
                if case.target_images:
                    for img in case.target_images:
                        img_path = os.path.join('app', 'static', img.image_path.replace('static/', ''))
                        if os.path.exists(img_path):
                            try:
                                ref_img = face_recognition.load_image_file(img_path)
                                encodings = face_recognition.face_encodings(ref_img)
                                if encodings:
                                    reference_encodings.extend(encodings)
                            except:
                                continue
                
                for i in range(sample_frames):
                    frame_pos = i * frame_count // sample_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Basic frame quality
                    frame_quality = self._check_image_quality(frame)
                    frame_scores.append(frame_quality)
                    
                    # Person detection in frame
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces in video frame
                        face_locations = face_recognition.face_locations(rgb_frame)
                        
                        if face_locations:
                            face_detected_frames += 1
                            
                            # Check if any face matches reference photos
                            if reference_encodings:
                                frame_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                
                                for frame_encoding in frame_encodings:
                                    matches = face_recognition.compare_faces(reference_encodings, frame_encoding, tolerance=0.6)
                                    if any(matches):
                                        person_detected_frames += 1
                                        break
                        
                        # Additional person detection using body detection
                        # (Simple motion/contour detection as fallback)
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Detect significant contours (potential persons)
                        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Count significant contours (potential persons)
                        significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
                        if significant_contours:
                            person_detected_frames += 0.5  # Partial credit for body detection
                    
                    except Exception as e:
                        continue
                
                cap.release()
                
                # Calculate frame quality score
                if frame_scores:
                    avg_frame_quality = sum(frame_scores) / len(frame_scores)
                    score += avg_frame_quality * 0.25
                
                # Person detection scoring
                if sample_frames > 0:
                    face_detection_ratio = face_detected_frames / sample_frames
                    person_detection_ratio = person_detected_frames / sample_frames
                    
                    # Bonus for person visibility
                    score += face_detection_ratio * 0.15  # Faces detected
                    score += person_detection_ratio * 0.15  # Target person detected
                    
                    # Store consistency score
                    if reference_encodings:
                        consistency_score = person_detection_ratio
                        person_consistency_scores.append(consistency_score)
                
                video_scores.append(score)
                
            except Exception as e:
                print(f"Video validation error: {str(e)}")
                continue
        
        if not video_scores:
            return 0.8  # No valid videos but not critical
        
        avg_video_score = sum(video_scores) / len(video_scores)
        
        # Bonus for person consistency across videos
        if person_consistency_scores:
            avg_consistency = sum(person_consistency_scores) / len(person_consistency_scores)
            if avg_consistency > 0.3:  # Person appears in videos
                avg_video_score += 0.1  # Bonus for consistency
            elif avg_consistency < 0.1:  # Person rarely appears
                self.rejection_reasons.append("[WARNING] Target person not clearly visible in uploaded videos")
        
        return min(avg_video_score, 1.0)
    
    def _check_consistency(self, case):
        """Advanced consistency check between photos, videos, and form data"""
        score = 1.0
        consistency_issues = []
        
        # 1. Cross-reference person name in details
        if case.details and case.person_name:
            name_parts = case.person_name.lower().split()
            details_lower = case.details.lower()
            
            name_mentioned = any(part in details_lower for part in name_parts if len(part) > 2)
            if not name_mentioned and len(case.person_name) > 5:
                score -= 0.15
                consistency_issues.append("person name not mentioned in details")
        
        # 2. Location consistency check
        if case.details and case.last_seen_location:
            location_parts = case.last_seen_location.lower().split()
            details_lower = case.details.lower()
            
            location_mentioned = any(part in details_lower for part in location_parts if len(part) > 3)
            if not location_mentioned:
                score -= 0.15
                consistency_issues.append("location not referenced in case details")
        
        # 3. Age consistency with description
        if case.age and case.details:
            details_lower = case.details.lower()
            age_descriptors = {
                'infant': (0, 2, ['baby', 'infant', 'newborn']),
                'child': (2, 13, ['child', 'kid', 'boy', 'girl', 'student']),
                'teenager': (13, 20, ['teenager', 'teen', 'adolescent', 'high school', 'college']),
                'young_adult': (20, 35, ['young', 'adult', 'man', 'woman']),
                'adult': (35, 60, ['adult', 'man', 'woman', 'father', 'mother']),
                'elderly': (60, 120, ['elderly', 'old', 'senior', 'grandfather', 'grandmother'])
            }
            
            expected_category = None
            for category, (min_age, max_age, keywords) in age_descriptors.items():
                if min_age <= case.age < max_age:
                    expected_category = category
                    break
            
            if expected_category:
                category_keywords = age_descriptors[expected_category][2]
                age_mentioned = any(keyword in details_lower for keyword in category_keywords)
                
                # Check for conflicting age descriptors
                conflicting_found = False
                for other_cat, (_, _, other_keywords) in age_descriptors.items():
                    if other_cat != expected_category:
                        if any(keyword in details_lower for keyword in other_keywords):
                            conflicting_found = True
                            break
                
                if conflicting_found:
                    score -= 0.2
                    consistency_issues.append(f"age ({case.age}) conflicts with description")
        
        # 4. Photo-Video person consistency
        if case.target_images and case.search_videos:
            try:
                # Get face encodings from photos
                photo_encodings = []
                for img in case.target_images:
                    img_path = os.path.join('app', 'static', img.image_path.replace('static/', ''))
                    if os.path.exists(img_path):
                        try:
                            photo = face_recognition.load_image_file(img_path)
                            encodings = face_recognition.face_encodings(photo)
                            photo_encodings.extend(encodings)
                        except:
                            continue
                
                if photo_encodings:
                    # Check if same person appears in videos
                    video_matches = 0
                    total_video_checks = 0
                    
                    for video in case.search_videos:
                        video_path = os.path.join('app', 'static', video.video_path.replace('static/', ''))
                        if os.path.exists(video_path):
                            try:
                                cap = cv2.VideoCapture(video_path)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                
                                # Sample 3 frames from video
                                for i in [0.2, 0.5, 0.8]:  # 20%, 50%, 80% through video
                                    frame_pos = int(frame_count * i)
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                                    ret, frame = cap.read()
                                    
                                    if ret:
                                        total_video_checks += 1
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        face_locations = face_recognition.face_locations(rgb_frame)
                                        
                                        if face_locations:
                                            frame_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                            
                                            for frame_encoding in frame_encodings:
                                                matches = face_recognition.compare_faces(photo_encodings, frame_encoding, tolerance=0.6)
                                                if any(matches):
                                                    video_matches += 1
                                                    break
                                
                                cap.release()
                            except:
                                continue
                    
                    # Evaluate photo-video consistency
                    if total_video_checks > 0:
                        match_ratio = video_matches / total_video_checks
                        if match_ratio < 0.2:  # Person rarely appears in videos
                            score -= 0.25
                            consistency_issues.append("person in photos doesn't match person in videos")
                        elif match_ratio > 0.6:  # Good consistency
                            score += 0.1  # Bonus for good consistency
            
            except Exception as e:
                print(f"Photo-video consistency check error: {str(e)}")
        
        # 5. Multiple photo consistency (same person in all photos)
        if len(case.target_images) > 1:
            try:
                all_encodings = []
                for img in case.target_images:
                    img_path = os.path.join('app', 'static', img.image_path.replace('static/', ''))
                    if os.path.exists(img_path):
                        try:
                            photo = face_recognition.load_image_file(img_path)
                            encodings = face_recognition.face_encodings(photo)
                            if encodings:
                                all_encodings.append((encodings[0], img.id))
                        except:
                            continue
                
                if len(all_encodings) > 1:
                    # Check if all photos show the same person
                    inconsistent_photos = 0
                    total_comparisons = 0
                    
                    for i in range(len(all_encodings)):
                        for j in range(i + 1, len(all_encodings)):
                            total_comparisons += 1
                            match = face_recognition.compare_faces([all_encodings[i][0]], all_encodings[j][0], tolerance=0.6)[0]
                            if not match:
                                inconsistent_photos += 1
                    
                    if total_comparisons > 0:
                        inconsistency_ratio = inconsistent_photos / total_comparisons
                        if inconsistency_ratio > 0.3:  # More than 30% inconsistent
                            score -= 0.3
                            consistency_issues.append("uploaded photos show different people")
            
            except Exception as e:
                print(f"Multi-photo consistency check error: {str(e)}")
        
        # Add specific feedback for consistency issues
        if consistency_issues:
            self.rejection_reasons.append(f"[WARNING] Consistency issues detected: {', '.join(consistency_issues)}")
        
        return max(score, 0.0)
    
    def _detect_fraud(self, case):
        """Detect potential fraud or spam cases"""
        fraud_indicators = 0
        
        # Check for duplicate content (basic hash check)
        if case.details:
            content_hash = hashlib.md5(case.details.encode()).hexdigest()
            # In production, you'd check this against a database of known spam
        
        # Check for suspicious patterns
        if case.details:
            details_lower = case.details.lower()
            
            # Spam keywords
            spam_keywords = ['money', 'reward', 'prize', 'lottery', 'winner', 'congratulations']
            fraud_indicators += sum(1 for keyword in spam_keywords if keyword in details_lower)
            
            # Suspicious contact patterns
            if 'whatsapp' in details_lower or 'telegram' in details_lower:
                fraud_indicators += 1
            
            # Too many URLs
            url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', case.details))
            if url_count > 2:
                fraud_indicators += 2
        
        # Check for suspicious timing (too many cases from same user in short time)
        # This would require database access to check user's recent submissions
        
        if fraud_indicators > 3:
            self.rejection_reasons.append("[ERROR] Content appears suspicious - possible spam or fraud")
            return 0.0
        elif fraud_indicators > 1:
            return 0.5
        
        return 1.0
    
    def generate_rejection_message(self, reasons):
        """Generate comprehensive rejection message with CCTV-specific guidance"""
        if not reasons:
            return "Case requires review before approval for CCTV analysis."
        
        message = "Your case submission needs the following improvements for accurate CCTV detection:\n\n"
        
        # Categorize reasons
        photo_issues = [r for r in reasons if any(word in r.lower() for word in ['photo', 'face', 'image', 'ai generated', 'clear'])]
        form_issues = [r for r in reasons if any(word in r.lower() for word in ['name', 'location', 'details', 'information'])]
        consistency_issues = [r for r in reasons if 'consistency' in r.lower()]
        other_issues = [r for r in reasons if r not in photo_issues + form_issues + consistency_issues]
        
        issue_count = 1
        
        if photo_issues:
            message += "[PHOTO] **Photo Quality Issues (Critical for CCTV Matching):**\n"
            for reason in photo_issues:
                message += f"{issue_count}. {reason}\n"
                issue_count += 1
            message += "\n"
        
        if form_issues:
            message += "[FORM] **Form Information Issues:**\n"
            for reason in form_issues:
                message += f"{issue_count}. {reason}\n"
                issue_count += 1
            message += "\n"
        
        if consistency_issues:
            message += "[CONSISTENCY] **Data Consistency Issues:**\n"
            for reason in consistency_issues:
                message += f"{issue_count}. {reason}\n"
                issue_count += 1
            message += "\n"
        
        if other_issues:
            message += "[OTHER] **Other Issues:**\n"
            for reason in other_issues:
                message += f"{issue_count}. {reason}\n"
                issue_count += 1
            message += "\n"
        
        message += "[ACTION] **Please address these issues and resubmit your case.**\n\n"
        
        message += "[TIPS] **CCTV Detection Tips:**\n"
        message += "• Upload clear, frontal face photos (minimum 60x60 pixels face size)\n"
        message += "• Ensure good lighting and sharp focus in photos\n"
        message += "• Avoid AI-generated or heavily filtered images\n"
        message += "• Include multiple angles if available\n"
        message += "• Provide detailed location and timing information\n\n"
        
        message += "[INFO] **Why This Matters:** Our AI needs high-quality reference photos to accurately detect the person in CCTV footage, even when they appear small or distant in crowded scenes.\n\n"
        
        message += "[SUPPORT] Contact support if you need assistance with photo requirements."
        
        return message
    
    def generate_smart_feedback(self, case, scores):
        """Generate smart feedback using the Smart Rejection System"""
        try:
            from smart_rejection_system import smart_rejection_system
            return smart_rejection_system.generate_smart_feedback(case, scores, self.rejection_reasons)
        except ImportError:
            # Fallback to basic feedback if smart system not available
            return {
                'overall_assessment': {'grade': 'C', 'status': 'Needs improvement'},
                'basic_feedback': self.generate_rejection_message(self.rejection_reasons)
            }
    
    def generate_approval_message(self, confidence, scores):
        """Generate approval message with CCTV-optimized confidence details"""
        message = f"[APPROVED] Your case has been automatically approved (Confidence: {confidence:.1%})\n\n"
        message += "[ANALYSIS] AI Analysis Results:\n"
        
        score_labels = {
            'photos': '[PHOTO] Photo Quality (CCTV Ready)',
            'form_data': '[FORM] Form Completeness',
            'text_quality': '[TEXT] Content Quality',
            'videos': '[VIDEO] Video Quality',
            'consistency': '[CONSISTENCY] Data Consistency',
            'fraud_check': '[SECURITY] Security Check',
            'cctv_readiness': '[CCTV] CCTV Matching Readiness'
        }
        
        for key, score in scores.items():
            label = score_labels.get(key, key.replace('_', ' ').title())
            percentage = score * 100
            
            if key == 'photos' or key == 'cctv_readiness':
                emoji = "[EXCELLENT]" if score > 0.8 else "[GOOD]" if score > 0.6 else "[WARNING]" if score > 0.4 else "[ERROR]"
            else:
                emoji = "[GOOD]" if score > 0.7 else "[WARNING]" if score > 0.4 else "[ERROR]"
            
            message += f"{emoji} {label}: {percentage:.0f}%\n"
        
        # CCTV-specific guidance
        cctv_score = scores.get('cctv_readiness', 0)
        if cctv_score > 0.8:
            message += "\n[EXCELLENT] Your photos are optimized for CCTV detection in crowded scenes."
        elif cctv_score > 0.6:
            message += "\n[GOOD] Good photo quality for CCTV matching. Results should be reliable."
        else:
            message += "\n[WARNING] Photos may work for CCTV detection but consider uploading clearer frontal face photos for better results."
        
        message += "\n\n[PROCESSING] Your case will now proceed to advanced AI investigation with CCTV analysis."
        
        return message

# Global validator instance
ai_validator = AIValidator()

# Integrate with continuous learning system
try:
    from learning_integration import learning_integration
    ai_validator = learning_integration.integrate_with_ai_validator(ai_validator)
except ImportError:
    pass  # Learning system not available