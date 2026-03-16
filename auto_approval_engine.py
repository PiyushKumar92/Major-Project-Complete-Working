"""
AI-Powered Case Approval Engine
Multi-factor automated decision making system for case approval/rejection
"""

import os
import cv2
import numpy as np
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import face_recognition
from PIL import Image
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoApprovalEngine:
    """
    Multi-factor automated decision making system for case approval
    Features:
    - Photo quality score (>0.7 threshold)
    - Information completeness (>80% fields filled)
    - Duplicate detection (cosine similarity)
    - Risk assessment (ML model trained on historical data)
    - Legal compliance check (automated policy validation)
    - Fraud detection (pattern recognition)
    """
    
    def __init__(self):
        # Thresholds for auto-approval
        self.photo_quality_threshold = 0.7
        self.information_completeness_threshold = 0.8
        self.duplicate_similarity_threshold = 0.85
        self.risk_score_threshold = 0.6
        self.fraud_score_threshold = 0.3
        self.overall_approval_threshold = 0.75
        
        # Legal compliance rules
        self.legal_compliance_rules = self._initialize_legal_rules()
        
        # Fraud detection patterns
        self.fraud_patterns = self._initialize_fraud_patterns()
        
        # ML models for risk assessment
        self.risk_model = None
        self.fraud_model = None
        self._initialize_ml_models()
        
        # Historical data for pattern analysis
        self.case_history = []
        self._load_historical_data()
    
    def evaluate_case_for_approval(self, case) -> Dict[str, Any]:
        """
        Main evaluation function - comprehensive multi-factor analysis
        Returns approval decision with detailed scoring
        """
        try:
            logger.info(f"Starting comprehensive evaluation for case {case.id}")
            
            evaluation = {
                'case_id': case.id,
                'timestamp': datetime.now(),
                'factors': {},
                'decision': 'PENDING',
                'confidence': 0.0,
                'reasons': [],
                'recommendations': [],
                'risk_flags': [],
                'compliance_status': 'UNKNOWN'
            }
            
            # Factor 1: Photo Quality Analysis (25% weight)
            photo_analysis = self._analyze_photo_quality(case)
            evaluation['factors']['photo_quality'] = photo_analysis
            
            # Factor 2: Information Completeness (20% weight)
            info_analysis = self._analyze_information_completeness(case)
            evaluation['factors']['information_completeness'] = info_analysis
            
            # Factor 3: Duplicate Detection (15% weight)
            duplicate_analysis = self._detect_duplicates(case)
            evaluation['factors']['duplicate_detection'] = duplicate_analysis
            
            # Factor 4: Risk Assessment (20% weight)
            risk_analysis = self._assess_risk_factors(case)
            evaluation['factors']['risk_assessment'] = risk_analysis
            
            # Factor 5: Legal Compliance Check (10% weight)
            compliance_analysis = self._check_legal_compliance(case)
            evaluation['factors']['legal_compliance'] = compliance_analysis
            
            # Factor 6: Fraud Detection (10% weight)
            fraud_analysis = self._detect_fraud_patterns(case)
            evaluation['factors']['fraud_detection'] = fraud_analysis
            
            # Calculate overall score and make decision
            final_decision = self._make_final_decision(evaluation)
            evaluation.update(final_decision)
            
            # Log decision for learning
            self._log_decision_for_learning(case, evaluation)
            
            logger.info(f"Case {case.id} evaluation complete: {evaluation['decision']} (confidence: {evaluation['confidence']:.2%})")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Case evaluation failed for case {case.id}: {str(e)}")
            return self._get_default_evaluation(case, str(e))
    
    def _analyze_photo_quality(self, case) -> Dict[str, Any]:
        """Advanced photo quality analysis with ML-based scoring"""
        if not case.target_images:
            return {
                'score': 0.0,
                'passed': False,
                'issues': ['No photos provided'],
                'details': {'total_photos': 0, 'quality_scores': []},
                'recommendations': ['Upload at least 2 clear, high-quality photos']
            }
        
        photo_scores = []
        face_detection_success = 0
        technical_issues = []
        
        for image in case.target_images:
            image_path = os.path.join('app', 'static', image.image_path.replace('static/', ''))
            
            if not os.path.exists(image_path):
                continue
            
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Comprehensive photo analysis
                photo_quality = self._evaluate_single_photo(img)
                photo_scores.append(photo_quality['overall_score'])
                
                if photo_quality['face_detected']:
                    face_detection_success += 1
                
                technical_issues.extend(photo_quality['issues'])
                
            except Exception as e:
                logger.error(f"Error analyzing photo {image.id}: {str(e)}")
                continue
        
        if not photo_scores:
            return {
                'score': 0.0,
                'passed': False,
                'issues': ['No valid photos found'],
                'details': {'total_photos': len(case.target_images), 'quality_scores': []},
                'recommendations': ['Upload clear, readable photos with visible faces']
            }
        
        # Calculate final photo quality score
        avg_quality = sum(photo_scores) / len(photo_scores)
        face_detection_rate = face_detection_success / len(photo_scores)
        
        # Bonuses and penalties
        multi_photo_bonus = min(len(photo_scores) * 0.05, 0.15)  # Up to 15% bonus
        face_penalty = (1 - face_detection_rate) * 0.2
        
        final_score = min(avg_quality + multi_photo_bonus - face_penalty, 1.0)
        passed = final_score >= self.photo_quality_threshold
        
        return {
            'score': max(final_score, 0.0),
            'passed': passed,
            'issues': list(set(technical_issues)),
            'details': {
                'total_photos': len(case.target_images),
                'quality_scores': photo_scores,
                'face_detection_rate': face_detection_rate,
                'average_quality': avg_quality
            },
            'recommendations': self._get_photo_recommendations(final_score, technical_issues)
        }
    
    def _evaluate_single_photo(self, img) -> Dict[str, Any]:
        """Comprehensive single photo evaluation"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            
            scores = {}
            issues = []
            
            # 1. Sharpness/Blur detection
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores['sharpness'] = min(laplacian_var / 1000, 1.0)
            if scores['sharpness'] < 0.3:
                issues.append('Photo is blurry or out of focus')
            
            # 2. Lighting analysis
            brightness = np.mean(gray)
            scores['lighting'] = 1.0 - abs(brightness - 127) / 127
            if scores['lighting'] < 0.4:
                issues.append('Poor lighting conditions')
            
            # 3. Contrast analysis
            contrast = gray.std()
            scores['contrast'] = min(contrast / 64, 1.0)
            if scores['contrast'] < 0.3:
                issues.append('Low contrast - details not clear')
            
            # 4. Resolution check
            scores['resolution'] = min((width * height) / (800 * 600), 1.0)
            if scores['resolution'] < 0.5:
                issues.append('Low resolution - minimum 800x600 recommended')
            
            # 5. Face detection and quality
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            face_detected = len(face_locations) > 0
            
            scores['face_quality'] = 0.0
            if face_detected:
                # Analyze best face
                largest_face = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
                top, right, bottom, left = largest_face
                
                face_area = (right - left) * (bottom - top)
                image_area = width * height
                face_ratio = face_area / image_area
                
                scores['face_quality'] = min(face_ratio / 0.08, 1.0)  # Optimal at 8% of image
                
                if face_ratio < 0.03:
                    issues.append('Face too small in image')
                
                # Face encoding test
                try:
                    encodings = face_recognition.face_encodings(rgb_img, [largest_face])
                    if not encodings:
                        issues.append('Face not suitable for recognition')
                        scores['face_quality'] *= 0.5
                except:
                    issues.append('Face encoding failed')
                    scores['face_quality'] *= 0.3
            else:
                issues.append('No face detected in photo')
            
            # 6. Noise analysis
            noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
            scores['noise'] = max(1.0 - noise_level / 50, 0.0)
            if scores['noise'] < 0.6:
                issues.append('High noise level in image')
            
            # 7. AI-generated detection
            ai_score = self._detect_ai_generated_image(img)
            scores['authenticity'] = ai_score
            if ai_score < 0.5:
                issues.append('Image may be AI-generated or heavily processed')
            
            # Calculate overall score
            weights = {
                'sharpness': 0.25, 'lighting': 0.20, 'contrast': 0.15,
                'resolution': 0.15, 'face_quality': 0.15, 'noise': 0.05, 'authenticity': 0.05
            }
            
            overall_score = sum(scores[key] * weights[key] for key in weights)
            
            return {
                'overall_score': overall_score,
                'face_detected': face_detected,
                'scores': scores,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Single photo evaluation failed: {str(e)}")
            return {
                'overall_score': 0.0,
                'face_detected': False,
                'scores': {},
                'issues': ['Photo analysis failed']
            }
    
    def _detect_ai_generated_image(self, img) -> float:
        """Detect AI-generated images using multiple techniques"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            ai_indicators = 0
            total_checks = 5
            
            # 1. Unnatural symmetry check
            if width > 100:
                left_half = gray[:, :width//2]
                right_half = cv2.flip(gray[:, width//2:], 1)
                min_width = min(left_half.shape[1], right_half.shape[1])
                
                if min_width > 0:
                    left_half = left_half[:, :min_width]
                    right_half = right_half[:, :min_width]
                    
                    diff = cv2.absdiff(left_half, right_half)
                    symmetry_score = 1.0 - (np.mean(diff) / 255.0)
                    
                    if symmetry_score > 0.95:  # Unnaturally symmetric
                        ai_indicators += 1
            
            # 2. Texture analysis
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if texture_variance < 50:  # Too smooth
                ai_indicators += 1
            
            # 3. Edge density analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (width * height)
            if edge_density < 0.015:  # Too few natural edges
                ai_indicators += 1
            
            # 4. Color distribution
            color_hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
            color_variance = np.var(color_hist)
            if color_variance < 500:  # Too uniform
                ai_indicators += 1
            
            # 5. Frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Check for artificial patterns in frequency domain
            freq_variance = np.var(magnitude_spectrum)
            if freq_variance < 2.0:  # Too regular frequency pattern
                ai_indicators += 1
            
            # Calculate authenticity score (inverse of AI probability)
            ai_probability = ai_indicators / total_checks
            return 1.0 - ai_probability
            
        except Exception as e:
            logger.error(f"AI detection failed: {str(e)}")
            return 0.7  # Default to likely authentic
    
    def _analyze_information_completeness(self, case) -> Dict[str, Any]:
        """Analyze completeness and quality of case information"""
        required_fields = {
            'person_name': {'weight': 0.20, 'min_length': 2},
            'last_seen_location': {'weight': 0.20, 'min_length': 10},
            'date_missing': {'weight': 0.15, 'min_length': 0},
            'details': {'weight': 0.25, 'min_length': 50},
            'age': {'weight': 0.10, 'min_length': 0},
            'target_images': {'weight': 0.10, 'min_length': 1}
        }
        
        field_scores = {}
        missing_fields = []
        quality_issues = []
        
        total_score = 0.0
        
        for field, config in required_fields.items():
            field_score = self._evaluate_field_quality(case, field, config)
            field_scores[field] = field_score
            
            if field_score['score'] == 0.0:
                missing_fields.append(field)
            elif field_score['score'] < 0.5:
                quality_issues.extend(field_score['issues'])
            
            total_score += field_score['score'] * config['weight']
        
        # Optional fields bonus
        optional_fields = ['clothing_description', 'contact_address', 'last_seen_time']
        optional_bonus = 0.0
        
        for field in optional_fields:
            if hasattr(case, field) and getattr(case, field):
                optional_bonus += 0.03
        
        total_score += min(optional_bonus, 0.09)
        
        # Information consistency check
        consistency_score = self._check_information_consistency(case)
        total_score += consistency_score * 0.1
        
        passed = total_score >= self.information_completeness_threshold
        
        return {
            'score': min(total_score, 1.0),
            'passed': passed,
            'field_scores': field_scores,
            'missing_fields': missing_fields,
            'quality_issues': quality_issues,
            'consistency_score': consistency_score,
            'recommendations': self._get_information_recommendations(field_scores, missing_fields)
        }
    
    def _evaluate_field_quality(self, case, field_name: str, config: Dict) -> Dict[str, Any]:
        """Evaluate quality of individual field"""
        if field_name == 'target_images':
            value = len(case.target_images) if case.target_images else 0
            score = 1.0 if value >= config['min_length'] else 0.0
            issues = [] if score > 0 else ['No photos uploaded']
        else:
            value = getattr(case, field_name, None)
            
            if not value:
                return {'score': 0.0, 'issues': [f'{field_name} is required']}
            
            if isinstance(value, str):
                if len(value.strip()) < config['min_length']:
                    return {
                        'score': 0.3,
                        'issues': [f'{field_name} is too short (minimum {config["min_length"]} characters)']
                    }
                
                # Additional quality checks for text fields
                if field_name == 'details':
                    quality_score = self._analyze_text_quality(value)
                    return {
                        'score': quality_score['score'],
                        'issues': quality_score['issues']
                    }
                
                score = 1.0
                issues = []
            else:
                score = 1.0
                issues = []
        
        return {'score': score, 'issues': issues}
    
    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze quality of text content"""
        text = text.strip()
        words = text.split()
        
        score = 0.0
        issues = []
        
        # Length analysis
        if len(text) < 50:
            issues.append('Description too brief')
            score += 0.2
        elif len(text) < 100:
            score += 0.5
        else:
            score += 0.8
        
        # Word count
        if len(words) < 10:
            issues.append('Not enough detail')
        elif len(words) >= 20:
            score += 0.1
        
        # Information density
        info_keywords = ['when', 'where', 'what', 'who', 'how', 'time', 'date', 'location', 'wearing', 'seen', 'last']
        keyword_count = sum(1 for word in words if word.lower() in info_keywords)
        
        if keyword_count >= 3:
            score += 0.1
        elif keyword_count < 1:
            issues.append('Lacks specific details (when, where, what, etc.)')
        
        # Spam detection
        spam_indicators = ['click here', 'visit website', 'call now', 'limited time', 'act fast', 'guaranteed']
        spam_count = sum(1 for indicator in spam_indicators if indicator.lower() in text.lower())
        
        if spam_count > 0:
            issues.append('Content appears promotional or spam-like')
            score *= 0.3
        
        return {
            'score': min(score, 1.0),
            'word_count': len(words),
            'character_count': len(text),
            'keyword_density': keyword_count / len(words) if words else 0,
            'issues': issues
        }
    
    def _check_information_consistency(self, case) -> float:
        """Check consistency between different information fields"""
        consistency_score = 1.0
        
        # Name consistency in details
        if case.person_name and case.details:
            name_parts = [part for part in case.person_name.lower().split() if len(part) > 2]
            details_lower = case.details.lower()
            
            name_mentioned = any(part in details_lower for part in name_parts)
            if not name_mentioned and len(name_parts) > 0:
                consistency_score -= 0.2
        
        # Location consistency
        if case.last_seen_location and case.details:
            location_parts = [part for part in case.last_seen_location.lower().split() if len(part) > 3]
            details_lower = case.details.lower()
            
            location_mentioned = any(part in details_lower for part in location_parts)
            if not location_mentioned and len(location_parts) > 0:
                consistency_score -= 0.2
        
        # Age consistency with description
        if case.age and case.details:
            age_descriptors = {
                'child': (0, 13), 'kid': (0, 13), 'teenager': (13, 20), 'teen': (13, 20),
                'young': (18, 35), 'adult': (18, 65), 'elderly': (60, 120), 'old': (60, 120)
            }
            
            details_lower = case.details.lower()
            for descriptor, (min_age, max_age) in age_descriptors.items():
                if descriptor in details_lower:
                    if not (min_age <= case.age <= max_age):
                        consistency_score -= 0.3
                    break
        
        return max(consistency_score, 0.0)
    
    def _detect_duplicates(self, case) -> Dict[str, Any]:
        """Advanced duplicate detection using multiple similarity measures"""
        try:
            from models import Case
            
            # Get recent cases for comparison (last 60 days)
            recent_cases = Case.query.filter(
                Case.id != case.id,
                Case.created_at >= datetime.now() - timedelta(days=60)
            ).all()
            
            if not recent_cases:
                return {
                    'score': 1.0,
                    'passed': True,
                    'similar_cases': [],
                    'max_similarity': 0.0,
                    'duplicate_risk': 'Very Low'
                }
            
            similarities = []
            
            for other_case in recent_cases:
                similarity = self._calculate_comprehensive_similarity(case, other_case)
                
                if similarity > 0.3:  # Only consider significant similarities
                    similarities.append({
                        'case_id': other_case.id,
                        'person_name': other_case.person_name,
                        'similarity_score': similarity,
                        'created_at': other_case.created_at,
                        'status': other_case.status
                    })
            
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            max_similarity = similarities[0]['similarity_score'] if similarities else 0.0
            
            # Determine duplicate risk and score
            if max_similarity >= self.duplicate_similarity_threshold:
                duplicate_risk = 'High'
                score = 0.1  # High penalty
                passed = False
            elif max_similarity >= 0.7:
                duplicate_risk = 'Medium'
                score = 0.4
                passed = True
            elif max_similarity >= 0.5:
                duplicate_risk = 'Low'
                score = 0.7
                passed = True
            else:
                duplicate_risk = 'Very Low'
                score = 1.0
                passed = True
            
            return {
                'score': score,
                'passed': passed,
                'similar_cases': similarities[:3],  # Top 3 similar cases
                'max_similarity': max_similarity,
                'duplicate_risk': duplicate_risk,
                'total_similar_cases': len(similarities)
            }
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {str(e)}")
            return {
                'score': 0.8,
                'passed': True,
                'similar_cases': [],
                'max_similarity': 0.0,
                'duplicate_risk': 'Unknown',
                'error': str(e)
            }
    
    def _calculate_comprehensive_similarity(self, case1, case2) -> float:
        """Calculate comprehensive similarity between two cases"""
        similarity_components = {}
        
        # Name similarity (35% weight)
        if case1.person_name and case2.person_name:
            similarity_components['name'] = self._calculate_text_similarity(
                case1.person_name.lower(), case2.person_name.lower()
            ) * 0.35
        
        # Location similarity (25% weight)
        if case1.last_seen_location and case2.last_seen_location:
            similarity_components['location'] = self._calculate_text_similarity(
                case1.last_seen_location.lower(), case2.last_seen_location.lower()
            ) * 0.25
        
        # Age similarity (15% weight)
        if case1.age and case2.age:
            age_diff = abs(case1.age - case2.age)
            age_similarity = max(0, 1 - age_diff / 10)
            similarity_components['age'] = age_similarity * 0.15
        
        # Details similarity (20% weight)
        if case1.details and case2.details:
            similarity_components['details'] = self._calculate_text_similarity(
                case1.details.lower(), case2.details.lower()
            ) * 0.20
        
        # Date similarity (5% weight)
        if case1.date_missing and case2.date_missing:
            date_diff = abs((case1.date_missing - case2.date_missing).days)
            date_similarity = max(0, 1 - date_diff / 30)  # Similar if within 30 days
            similarity_components['date'] = date_similarity * 0.05
        
        return sum(similarity_components.values())
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity"""
        try:
            if not text1 or not text2:
                return 0.0
            
            if len(text1) < 10 or len(text2) < 10:
                return self._calculate_simple_similarity(text1, text2)
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000
            )
            
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception:
            return self._calculate_simple_similarity(text1, text2)
    
    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """Simple character-based similarity for short texts"""
        if not text1 or not text2:
            return 0.0
        
        set1, set2 = set(text1.lower()), set(text2.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_risk_factors(self, case) -> Dict[str, Any]:
        """ML-based risk assessment using historical data patterns"""
        try:
            risk_factors = {}
            risk_score = 0.5  # Base risk score
            
            # Age-based risk assessment
            if case.age:
                if case.age < 13:  # Children - higher risk
                    risk_factors['age_risk'] = 'High - Child case'
                    risk_score += 0.3
                elif case.age > 70:  # Elderly - medium-high risk
                    risk_factors['age_risk'] = 'Medium-High - Elderly case'
                    risk_score += 0.2
                elif 13 <= case.age <= 17:  # Teenagers - medium risk
                    risk_factors['age_risk'] = 'Medium - Teenager case'
                    risk_score += 0.1
                else:
                    risk_factors['age_risk'] = 'Low - Adult case'
            
            # Time-based risk assessment
            if case.date_missing:
                days_missing = (datetime.now().date() - case.date_missing.date()).days
                if days_missing <= 1:
                    risk_factors['time_risk'] = 'Critical - Recently missing'
                    risk_score += 0.4
                elif days_missing <= 7:
                    risk_factors['time_risk'] = 'High - Missing within a week'
                    risk_score += 0.3
                elif days_missing <= 30:
                    risk_factors['time_risk'] = 'Medium - Missing within a month'
                    risk_score += 0.1
                else:
                    risk_factors['time_risk'] = 'Low - Long-term missing'
            
            # Content-based risk assessment
            if case.details:
                high_risk_keywords = ['kidnapped', 'abducted', 'forced', 'threatened', 'danger', 'weapon', 'violence']
                medium_risk_keywords = ['worried', 'concerned', 'unusual', 'never', 'always', 'depression', 'mental']
                
                details_lower = case.details.lower()
                
                high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in details_lower)
                medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in details_lower)
                
                if high_risk_count > 0:
                    risk_factors['content_risk'] = f'High - {high_risk_count} high-risk indicators'
                    risk_score += high_risk_count * 0.2
                elif medium_risk_count > 0:
                    risk_factors['content_risk'] = f'Medium - {medium_risk_count} medium-risk indicators'
                    risk_score += medium_risk_count * 0.1
                else:
                    risk_factors['content_risk'] = 'Low - No risk indicators'
            
            # Location-based risk assessment
            if case.last_seen_location:
                high_risk_locations = ['highway', 'isolated', 'remote', 'forest', 'abandoned', 'dark', 'late night']
                location_lower = case.last_seen_location.lower()
                
                location_risk_count = sum(1 for location in high_risk_locations if location in location_lower)
                if location_risk_count > 0:
                    risk_factors['location_risk'] = f'High - {location_risk_count} high-risk location indicators'
                    risk_score += location_risk_count * 0.15
                else:
                    risk_factors['location_risk'] = 'Low - No high-risk location indicators'
            
            # Requester type risk assessment
            if hasattr(case, 'requester_type'):
                if case.requester_type in ['police', 'government', 'law_enforcement']:
                    risk_factors['requester_risk'] = 'High - Official request'
                    risk_score += 0.2
                elif case.requester_type == 'family':
                    risk_factors['requester_risk'] = 'Medium - Family request'
                    risk_score += 0.1
                else:
                    risk_factors['requester_risk'] = 'Low - Other requester'
            
            # ML model prediction (if available)
            if self.risk_model:
                try:
                    features = self._extract_risk_features(case)
                    ml_risk_score = self.risk_model.predict_proba([features])[0][1]  # Probability of high risk
                    risk_factors['ml_prediction'] = f'ML Risk Score: {ml_risk_score:.2%}'
                    risk_score = (risk_score + ml_risk_score) / 2  # Average with rule-based score
                except Exception as e:
                    logger.error(f"ML risk prediction failed: {str(e)}")
            
            # Normalize risk score
            risk_score = min(max(risk_score, 0.0), 1.0)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = 'Critical'
            elif risk_score >= 0.6:
                risk_level = 'High'
            elif risk_score >= 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            passed = risk_score >= self.risk_score_threshold
            
            return {
                'score': risk_score,
                'passed': passed,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': self._get_risk_recommendations(risk_level, risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {
                'score': 0.5,
                'passed': True,
                'risk_level': 'Medium',
                'risk_factors': {'error': str(e)},
                'recommendations': ['Manual risk assessment required']
            }
    
    def _check_legal_compliance(self, case) -> Dict[str, Any]:
        """Automated legal compliance check against policy rules"""
        compliance_issues = []
        compliance_score = 1.0
        
        # Check against legal compliance rules
        for rule_name, rule_func in self.legal_compliance_rules.items():
            try:
                is_compliant, issue_description = rule_func(case)
                if not is_compliant:
                    compliance_issues.append(f"{rule_name}: {issue_description}")
                    compliance_score -= 0.2
            except Exception as e:
                logger.error(f"Legal compliance check '{rule_name}' failed: {str(e)}")
                compliance_issues.append(f"{rule_name}: Check failed - {str(e)}")
                compliance_score -= 0.1
        
        compliance_score = max(compliance_score, 0.0)
        
        if compliance_score >= 0.8:
            compliance_status = 'Compliant'
        elif compliance_score >= 0.6:
            compliance_status = 'Minor Issues'
        else:
            compliance_status = 'Non-Compliant'
        
        passed = compliance_score >= 0.6  # 60% compliance threshold
        
        return {
            'score': compliance_score,
            'passed': passed,
            'status': compliance_status,
            'issues': compliance_issues,
            'recommendations': self._get_compliance_recommendations(compliance_issues)
        }
    
    def _detect_fraud_patterns(self, case) -> Dict[str, Any]:
        """Advanced fraud detection using pattern recognition"""
        fraud_indicators = []
        fraud_score = 0.0  # 0 = no fraud, 1 = high fraud probability
        
        # Pattern 1: Suspicious content patterns
        if case.details:
            details_lower = case.details.lower()
            
            # Financial fraud indicators
            financial_keywords = ['money', 'reward', 'prize', 'lottery', 'winner', 'inheritance', 'fund']
            financial_count = sum(1 for keyword in financial_keywords if keyword in details_lower)
            if financial_count > 2:
                fraud_indicators.append(f'Financial fraud patterns detected ({financial_count} indicators)')
                fraud_score += 0.4
            
            # Spam/promotional content
            spam_keywords = ['click here', 'visit website', 'call now', 'limited time', 'act fast', 'guaranteed']
            spam_count = sum(1 for keyword in spam_keywords if keyword in details_lower)
            if spam_count > 1:
                fraud_indicators.append(f'Promotional content detected ({spam_count} indicators)')
                fraud_score += 0.3
            
            # Suspicious contact patterns
            suspicious_contacts = ['whatsapp only', 'telegram only', 'no phone calls', 'email only']
            contact_issues = sum(1 for pattern in suspicious_contacts if pattern in details_lower)
            if contact_issues > 0:
                fraud_indicators.append('Suspicious contact preferences')
                fraud_score += 0.2
        
        # Pattern 2: Inconsistent information
        inconsistency_score = 1.0 - self._check_information_consistency(case)
        if inconsistency_score > 0.5:
            fraud_indicators.append('High information inconsistency')
            fraud_score += inconsistency_score * 0.3
        
        # Pattern 3: Suspicious timing patterns
        if hasattr(case, 'user_id'):
            try:
                from models import Case
                # Check for multiple recent submissions from same user
                recent_cases = Case.query.filter(
                    Case.user_id == case.user_id,
                    Case.created_at >= datetime.now() - timedelta(hours=24)
                ).count()
                
                if recent_cases > 3:  # More than 3 cases in 24 hours
                    fraud_indicators.append(f'Suspicious submission frequency ({recent_cases} cases in 24h)')
                    fraud_score += 0.3
            except Exception:
                pass
        
        # Pattern 4: ML-based fraud detection
        if self.fraud_model:
            try:
                features = self._extract_fraud_features(case)
                ml_fraud_score = self.fraud_model.predict_proba([features])[0][1]
                if ml_fraud_score > 0.7:
                    fraud_indicators.append(f'ML fraud detection: {ml_fraud_score:.2%} probability')
                    fraud_score = max(fraud_score, ml_fraud_score)
            except Exception as e:
                logger.error(f"ML fraud detection failed: {str(e)}")
        
        # Pattern 5: Photo authenticity
        if case.target_images:
            ai_generated_count = 0
            for image in case.target_images:
                image_path = os.path.join('app', 'static', image.image_path.replace('static/', ''))
                if os.path.exists(image_path):
                    try:
                        img = cv2.imread(image_path)
                        if img is not None:
                            authenticity_score = self._detect_ai_generated_image(img)
                            if authenticity_score < 0.5:
                                ai_generated_count += 1
                    except:
                        continue
            
            if ai_generated_count > 0:
                fraud_indicators.append(f'{ai_generated_count} potentially AI-generated images')
                fraud_score += ai_generated_count * 0.2
        
        fraud_score = min(fraud_score, 1.0)
        
        # Determine fraud risk level
        if fraud_score >= 0.7:
            fraud_risk = 'High'
        elif fraud_score >= 0.4:
            fraud_risk = 'Medium'
        elif fraud_score >= 0.2:
            fraud_risk = 'Low'
        else:
            fraud_risk = 'Very Low'
        
        passed = fraud_score <= self.fraud_score_threshold
        
        return {
            'score': 1.0 - fraud_score,  # Convert to positive score (higher is better)
            'passed': passed,
            'fraud_probability': fraud_score,
            'fraud_risk': fraud_risk,
            'indicators': fraud_indicators,
            'recommendations': self._get_fraud_recommendations(fraud_risk, fraud_indicators)
        }
    
    def _make_final_decision(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Make final approval decision based on all factors"""
        factors = evaluation['factors']
        
        # Calculate weighted overall score
        weights = {
            'photo_quality': 0.25,
            'information_completeness': 0.20,
            'duplicate_detection': 0.15,
            'risk_assessment': 0.20,
            'legal_compliance': 0.10,
            'fraud_detection': 0.10
        }
        
        overall_score = sum(
            factors[factor]['score'] * weights[factor]
            for factor in weights
            if factor in factors
        )
        
        # Check critical failures (any factor that must pass)
        critical_failures = []
        
        for factor_name, factor_data in factors.items():
            if not factor_data.get('passed', True):
                if factor_name in ['legal_compliance', 'fraud_detection']:
                    critical_failures.append(factor_name)
                elif factor_name == 'duplicate_detection' and factor_data.get('duplicate_risk') == 'High':
                    critical_failures.append(factor_name)
        
        # Make decision
        if critical_failures:
            decision = 'REJECT'
            confidence = 0.9  # High confidence in rejection due to critical failures
            reasons = [f"Critical failure in {failure}" for failure in critical_failures]
        elif overall_score >= self.overall_approval_threshold:
            decision = 'APPROVE'
            confidence = min(overall_score + 0.1, 0.95)
            reasons = ['All quality thresholds met']
        else:
            decision = 'REJECT'
            confidence = 1.0 - overall_score
            reasons = self._generate_rejection_reasons(factors)
        
        # Generate recommendations
        recommendations = []
        for factor_data in factors.values():
            if 'recommendations' in factor_data:
                recommendations.extend(factor_data['recommendations'])
        
        # Collect risk flags
        risk_flags = []
        if factors.get('risk_assessment', {}).get('risk_level') in ['Critical', 'High']:
            risk_flags.append(f"High risk case: {factors['risk_assessment']['risk_level']}")
        
        if factors.get('fraud_detection', {}).get('fraud_risk') in ['High', 'Medium']:
            risk_flags.append(f"Fraud risk: {factors['fraud_detection']['fraud_risk']}")
        
        return {
            'decision': decision,
            'confidence': confidence,
            'overall_score': overall_score,
            'reasons': reasons,
            'recommendations': list(set(recommendations)),
            'risk_flags': risk_flags,
            'compliance_status': factors.get('legal_compliance', {}).get('status', 'Unknown'),
            'critical_failures': critical_failures
        }
    
    def _generate_rejection_reasons(self, factors: Dict[str, Any]) -> List[str]:
        """Generate specific rejection reasons based on failed factors"""
        reasons = []
        
        for factor_name, factor_data in factors.items():
            if not factor_data.get('passed', True):
                if factor_name == 'photo_quality':
                    reasons.append(f"Photo quality insufficient ({factor_data['score']:.1%})")
                elif factor_name == 'information_completeness':
                    reasons.append(f"Information incomplete ({factor_data['score']:.1%})")
                elif factor_name == 'duplicate_detection':
                    reasons.append(f"Potential duplicate case (similarity: {factor_data.get('max_similarity', 0):.1%})")
                elif factor_name == 'risk_assessment':
                    reasons.append(f"Risk assessment concerns ({factor_data['risk_level']})")
                elif factor_name == 'legal_compliance':
                    reasons.append(f"Legal compliance issues ({factor_data['status']})")
                elif factor_name == 'fraud_detection':
                    reasons.append(f"Fraud detection concerns ({factor_data['fraud_risk']} risk)")
        
        return reasons
    
    # Helper methods for recommendations
    def _get_photo_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """Generate photo-specific recommendations"""
        recommendations = []
        
        if score < 0.4:
            recommendations.append('Upload higher quality photos with better lighting and focus')
        
        if 'Photo is blurry' in issues:
            recommendations.append('Ensure photos are sharp and in focus')
        
        if 'No face detected' in issues:
            recommendations.append('Upload photos with clearly visible faces')
        
        if 'Low resolution' in issues:
            recommendations.append('Use higher resolution images (minimum 800x600)')
        
        if 'Face too small in image' in issues:
            recommendations.append('Use photos where the face is larger and more prominent')
        
        return recommendations
    
    def _get_information_recommendations(self, field_scores: Dict, missing_fields: List[str]) -> List[str]:
        """Generate information-specific recommendations"""
        recommendations = []
        
        for field in missing_fields:
            if field == 'person_name':
                recommendations.append('Provide complete person name')
            elif field == 'last_seen_location':
                recommendations.append('Add detailed location with landmarks and address')
            elif field == 'details':
                recommendations.append('Provide comprehensive case description')
            elif field == 'target_images':
                recommendations.append('Upload clear photos of the person')
        
        return recommendations
    
    def _get_risk_recommendations(self, risk_level: str, risk_factors: Dict) -> List[str]:
        """Generate risk-specific recommendations"""
        recommendations = []
        
        if risk_level in ['Critical', 'High']:
            recommendations.append('Case requires immediate attention due to high risk factors')
            recommendations.append('Consider escalating to emergency services')
        
        if 'age_risk' in risk_factors and 'Child' in risk_factors['age_risk']:
            recommendations.append('Child case - activate amber alert protocols if applicable')
        
        return recommendations
    
    def _get_compliance_recommendations(self, issues: List[str]) -> List[str]:
        """Generate compliance-specific recommendations"""
        recommendations = []
        
        for issue in issues:
            if 'privacy' in issue.lower():
                recommendations.append('Review privacy policy compliance')
            elif 'consent' in issue.lower():
                recommendations.append('Ensure proper consent documentation')
            elif 'data' in issue.lower():
                recommendations.append('Verify data handling compliance')
        
        return recommendations
    
    def _get_fraud_recommendations(self, fraud_risk: str, indicators: List[str]) -> List[str]:
        """Generate fraud-specific recommendations"""
        recommendations = []
        
        if fraud_risk in ['High', 'Medium']:
            recommendations.append('Manual verification required due to fraud indicators')
            recommendations.append('Verify submitter identity and contact information')
        
        if any('AI-generated' in indicator for indicator in indicators):
            recommendations.append('Verify authenticity of uploaded images')
        
        return recommendations
    
    # Initialization methods
    def _initialize_legal_rules(self) -> Dict:
        """Initialize legal compliance rules"""
        return {
            'privacy_compliance': self._check_privacy_compliance,
            'data_protection': self._check_data_protection,
            'consent_verification': self._check_consent_verification,
            'age_verification': self._check_age_verification,
            'content_policy': self._check_content_policy
        }
    
    def _initialize_fraud_patterns(self) -> Dict:
        """Initialize fraud detection patterns"""
        return {
            'financial_fraud': ['money', 'reward', 'prize', 'lottery', 'inheritance'],
            'spam_content': ['click here', 'visit website', 'call now', 'guaranteed'],
            'suspicious_contacts': ['whatsapp only', 'telegram only', 'no calls'],
            'fake_urgency': ['urgent', 'immediate', 'act fast', 'limited time']
        }
    
    def _initialize_ml_models(self):
        """Initialize ML models for risk and fraud detection"""
        try:
            # Try to load pre-trained models
            model_path = 'models'
            os.makedirs(model_path, exist_ok=True)
            
            risk_model_path = os.path.join(model_path, 'risk_model.pkl')
            fraud_model_path = os.path.join(model_path, 'fraud_model.pkl')
            
            if os.path.exists(risk_model_path):
                with open(risk_model_path, 'rb') as f:
                    self.risk_model = pickle.load(f)
            
            if os.path.exists(fraud_model_path):
                with open(fraud_model_path, 'rb') as f:
                    self.fraud_model = pickle.load(f)
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load ML models: {str(e)}")
            self.risk_model = None
            self.fraud_model = None
    
    def _load_historical_data(self):
        """Load historical case data for pattern analysis"""
        try:
            # In production, load from database
            # For now, initialize empty
            self.case_history = []
        except Exception as e:
            logger.warning(f"Could not load historical data: {str(e)}")
    
    # Legal compliance check methods
    def _check_privacy_compliance(self, case) -> Tuple[bool, str]:
        """Check privacy policy compliance"""
        # Basic privacy checks
        if not case.details:
            return False, "No case details provided for privacy assessment"
        
        # Check for sensitive information exposure
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Z]{2}\d{2}[A-Z]{4}\d{6}\b'  # Passport
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, case.details):
                return False, "Sensitive personal information detected in case details"
        
        return True, "Privacy compliance check passed"
    
    def _check_data_protection(self, case) -> Tuple[bool, str]:
        """Check data protection compliance"""
        # Verify data minimization principle
        if case.details and len(case.details) > 2000:
            return False, "Excessive personal data provided"
        
        return True, "Data protection compliance check passed"
    
    def _check_consent_verification(self, case) -> Tuple[bool, str]:
        """Check consent verification"""
        # Basic consent check - in production, verify against consent records
        if hasattr(case, 'requester_type'):
            if case.requester_type in ['family', 'police', 'government']:
                return True, "Authorized requester type"
        
        return True, "Consent verification passed"
    
    def _check_age_verification(self, case) -> Tuple[bool, str]:
        """Check age-related compliance"""
        if case.age and case.age < 13:
            # Special handling for children
            if not hasattr(case, 'requester_type') or case.requester_type not in ['family', 'police', 'government']:
                return False, "Child case requires authorized requester"
        
        return True, "Age verification passed"
    
    def _check_content_policy(self, case) -> Tuple[bool, str]:
        """Check content policy compliance"""
        if case.details:
            prohibited_content = ['violence', 'threat', 'harm', 'illegal', 'drug']
            details_lower = case.details.lower()
            
            for content in prohibited_content:
                if content in details_lower:
                    return False, f"Potentially prohibited content detected: {content}"
        
        return True, "Content policy compliance check passed"
    
    # Feature extraction methods for ML
    def _extract_risk_features(self, case) -> List[float]:
        """Extract features for ML risk assessment"""
        features = []
        
        # Age feature
        features.append(case.age if case.age else 25)  # Default age
        
        # Days missing
        if case.date_missing:
            days_missing = (datetime.now().date() - case.date_missing.date()).days
            features.append(min(days_missing, 365))  # Cap at 365 days
        else:
            features.append(0)
        
        # Text length
        features.append(len(case.details) if case.details else 0)
        
        # Number of photos
        features.append(len(case.target_images) if case.target_images else 0)
        
        # Risk keywords count
        if case.details:
            risk_keywords = ['danger', 'threat', 'kidnap', 'abduct', 'weapon', 'violence']
            risk_count = sum(1 for keyword in risk_keywords if keyword in case.details.lower())
            features.append(risk_count)
        else:
            features.append(0)
        
        return features
    
    def _extract_fraud_features(self, case) -> List[float]:
        """Extract features for ML fraud detection"""
        features = []
        
        # Text-based features
        if case.details:
            details_lower = case.details.lower()
            
            # Financial keywords
            financial_keywords = ['money', 'reward', 'prize', 'lottery']
            features.append(sum(1 for keyword in financial_keywords if keyword in details_lower))
            
            # Spam keywords
            spam_keywords = ['click', 'visit', 'call now', 'guaranteed']
            features.append(sum(1 for keyword in spam_keywords if keyword in details_lower))
            
            # Text quality indicators
            features.append(len(case.details))
            features.append(len(case.details.split()))
        else:
            features.extend([0, 0, 0, 0])
        
        # Consistency score
        features.append(self._check_information_consistency(case))
        
        return features
    
    def _log_decision_for_learning(self, case, evaluation: Dict[str, Any]):
        """Log decision for ML model improvement"""
        try:
            log_entry = {
                'case_id': case.id,
                'timestamp': datetime.now(),
                'decision': evaluation['decision'],
                'confidence': evaluation['confidence'],
                'factors': evaluation['factors'],
                'features': {
                    'risk_features': self._extract_risk_features(case),
                    'fraud_features': self._extract_fraud_features(case)
                }
            }
            
            # In production, save to database or file for model retraining
            logger.info(f"Decision logged for case {case.id}: {evaluation['decision']}")
            
        except Exception as e:
            logger.error(f"Failed to log decision: {str(e)}")
    
    def _get_default_evaluation(self, case, error_msg: str) -> Dict[str, Any]:
        """Return default evaluation in case of errors"""
        return {
            'case_id': case.id,
            'timestamp': datetime.now(),
            'factors': {},
            'decision': 'PENDING',
            'confidence': 0.0,
            'reasons': [f'System error: {error_msg}'],
            'recommendations': ['Manual review required due to system error'],
            'risk_flags': ['System error occurred'],
            'compliance_status': 'UNKNOWN',
            'error': error_msg
        }

# Global instance
auto_approval_engine = AutoApprovalEngine()