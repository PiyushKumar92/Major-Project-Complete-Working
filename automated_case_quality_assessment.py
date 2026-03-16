"""
Automated Case Quality Assessment System
ML-based comprehensive case scoring and quality evaluation
"""

import os
import cv2
import numpy as np
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import face_recognition
from PIL import Image
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaseQualityAssessment:
    """ML-based case quality assessment and scoring system"""
    
    def __init__(self):
        self.photo_quality_threshold = 0.6
        self.information_completeness_threshold = 0.7
        self.duplicate_similarity_threshold = 0.85
        self.urgency_keywords = {
            'critical': ['kidnapped', 'abducted', 'danger', 'threat', 'emergency', 'urgent', 'critical'],
            'high': ['missing', 'disappeared', 'lost', 'help', 'worried', 'concerned', 'police'],
            'medium': ['looking for', 'find', 'search', 'locate', 'contact'],
            'low': ['reunion', 'reconnect', 'old friend', 'relative']
        }
        
    def assess_case_quality(self, case) -> Dict:
        """Comprehensive case quality assessment"""
        try:
            assessment = {
                'overall_score': 0.0,
                'photo_quality': self._assess_photo_quality(case),
                'information_completeness': self._assess_information_completeness(case),
                'urgency_classification': self._classify_urgency(case),
                'duplicate_risk': self._assess_duplicate_risk(case),
                'recommendations': [],
                'quality_grade': 'F',
                'processing_priority': 'Low',
                'estimated_success_rate': 0.0
            }
            
            # Calculate weighted overall score
            weights = {
                'photo_quality': 0.35,
                'information_completeness': 0.30,
                'urgency_classification': 0.20,
                'duplicate_risk': 0.15
            }
            
            assessment['overall_score'] = (
                assessment['photo_quality']['score'] * weights['photo_quality'] +
                assessment['information_completeness']['score'] * weights['information_completeness'] +
                assessment['urgency_classification']['score'] * weights['urgency_classification'] +
                assessment['duplicate_risk']['score'] * weights['duplicate_risk']
            )
            
            # Determine quality grade
            assessment['quality_grade'] = self._calculate_quality_grade(assessment['overall_score'])
            
            # Set processing priority
            assessment['processing_priority'] = self._determine_processing_priority(assessment)
            
            # Estimate success rate
            assessment['estimated_success_rate'] = self._estimate_success_rate(assessment)
            
            # Generate recommendations
            assessment['recommendations'] = self._generate_recommendations(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Case quality assessment failed: {str(e)}")
            return self._get_default_assessment()
    
    def _assess_photo_quality(self, case) -> Dict:
        """Advanced photo quality analysis with ML-based scoring"""
        if not case.target_images:
            return {
                'score': 0.0,
                'details': {
                    'total_photos': 0,
                    'quality_scores': [],
                    'face_detection_success': 0,
                    'blur_analysis': [],
                    'lighting_analysis': [],
                    'resolution_analysis': []
                },
                'issues': ['No photos provided'],
                'recommendations': ['Upload at least 2 clear photos for better results']
            }
        
        photo_scores = []
        face_detection_count = 0
        blur_scores = []
        lighting_scores = []
        resolution_scores = []
        issues = []
        
        for image in case.target_images:
            image_path = os.path.join('app', 'static', image.image_path.replace('static/', ''))
            
            if not os.path.exists(image_path):
                continue
            
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Individual photo quality metrics
                photo_quality = self._analyze_single_photo_quality(img)
                photo_scores.append(photo_quality['overall_score'])
                
                # Face detection
                if photo_quality['face_detected']:
                    face_detection_count += 1
                
                # Collect detailed metrics
                blur_scores.append(photo_quality['blur_score'])
                lighting_scores.append(photo_quality['lighting_score'])
                resolution_scores.append(photo_quality['resolution_score'])
                
                # Collect issues
                issues.extend(photo_quality['issues'])
                
            except Exception as e:
                logger.error(f"Error analyzing photo {image.id}: {str(e)}")
                continue
        
        if not photo_scores:
            return {
                'score': 0.0,
                'details': {
                    'total_photos': len(case.target_images),
                    'quality_scores': [],
                    'face_detection_success': 0,
                    'blur_analysis': [],
                    'lighting_analysis': [],
                    'resolution_analysis': []
                },
                'issues': ['No valid photos found'],
                'recommendations': ['Upload clear, high-quality photos']
            }
        
        # Calculate overall photo quality score
        avg_quality = sum(photo_scores) / len(photo_scores)
        face_detection_rate = face_detection_count / len(photo_scores)
        
        # Bonus for multiple good photos
        photo_count_bonus = min(len(photo_scores) * 0.1, 0.3)
        
        # Penalty for poor face detection
        face_penalty = (1 - face_detection_rate) * 0.2
        
        final_score = min(avg_quality + photo_count_bonus - face_penalty, 1.0)
        
        return {
            'score': max(final_score, 0.0),
            'details': {
                'total_photos': len(case.target_images),
                'quality_scores': photo_scores,
                'face_detection_success': face_detection_count,
                'face_detection_rate': face_detection_rate,
                'average_blur_score': sum(blur_scores) / len(blur_scores) if blur_scores else 0,
                'average_lighting_score': sum(lighting_scores) / len(lighting_scores) if lighting_scores else 0,
                'average_resolution_score': sum(resolution_scores) / len(resolution_scores) if resolution_scores else 0
            },
            'issues': list(set(issues)),
            'recommendations': self._get_photo_recommendations(final_score, face_detection_rate, issues)
        }
    
    def _analyze_single_photo_quality(self, img) -> Dict:
        """Analyze quality of a single photo"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            
            # 1. Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_normalized = min(blur_score / 1000, 1.0)
            
            # 2. Lighting analysis
            brightness = np.mean(gray)
            lighting_score = 1.0 - abs(brightness - 127) / 127
            
            # 3. Contrast analysis
            contrast = gray.std()
            contrast_score = min(contrast / 64, 1.0)
            
            # 4. Resolution analysis
            resolution_score = min((width * height) / (640 * 480), 1.0)
            
            # 5. Face detection
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            face_detected = len(face_locations) > 0
            
            # Face quality analysis if detected
            face_quality_score = 0.0
            if face_detected:
                # Analyze largest face
                largest_face = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
                top, right, bottom, left = largest_face
                face_width = right - left
                face_height = bottom - top
                
                # Face size relative to image
                face_area = face_width * face_height
                image_area = width * height
                face_size_ratio = face_area / image_area
                
                face_quality_score = min(face_size_ratio / 0.1, 1.0)  # Optimal at 10% of image
            
            # 6. Noise analysis
            noise_score = 1.0 - min(np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray) / 50, 1.0)
            
            # Calculate overall score
            overall_score = (
                blur_normalized * 0.25 +
                lighting_score * 0.20 +
                contrast_score * 0.15 +
                resolution_score * 0.15 +
                face_quality_score * 0.15 +
                noise_score * 0.10
            )
            
            # Identify issues
            issues = []
            if blur_normalized < 0.4:
                issues.append('Photo is blurry')
            if lighting_score < 0.5:
                issues.append('Poor lighting conditions')
            if resolution_score < 0.5:
                issues.append('Low resolution')
            if not face_detected:
                issues.append('No face detected')
            elif face_quality_score < 0.3:
                issues.append('Face too small in image')
            
            return {
                'overall_score': overall_score,
                'blur_score': blur_normalized,
                'lighting_score': lighting_score,
                'contrast_score': contrast_score,
                'resolution_score': resolution_score,
                'face_detected': face_detected,
                'face_quality_score': face_quality_score,
                'noise_score': noise_score,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Single photo analysis failed: {str(e)}")
            return {
                'overall_score': 0.0,
                'blur_score': 0.0,
                'lighting_score': 0.0,
                'contrast_score': 0.0,
                'resolution_score': 0.0,
                'face_detected': False,
                'face_quality_score': 0.0,
                'noise_score': 0.0,
                'issues': ['Analysis failed']
            }
    
    def _assess_information_completeness(self, case) -> Dict:
        """Assess completeness and quality of case information"""
        score = 0.0
        details = {}
        issues = []
        
        # Required fields scoring
        required_fields = {
            'person_name': 0.20,
            'last_seen_location': 0.20,
            'date_missing': 0.15,
            'details': 0.25,
            'age': 0.10,
            'target_images': 0.10
        }
        
        field_scores = {}
        
        # Check each required field
        for field, weight in required_fields.items():
            field_score = self._score_field_completeness(case, field)
            field_scores[field] = field_score
            score += field_score * weight
        
        # Optional fields bonus
        optional_fields = ['clothing_description', 'contact_address', 'last_seen_time']
        optional_bonus = 0.0
        
        for field in optional_fields:
            if hasattr(case, field) and getattr(case, field):
                optional_bonus += 0.05
        
        score += min(optional_bonus, 0.15)
        
        # Text quality analysis
        if case.details:
            text_quality = self._analyze_text_quality(case.details)
            score += text_quality['score'] * 0.1
            details['text_quality'] = text_quality
            if text_quality['issues']:
                issues.extend(text_quality['issues'])
        
        # Information consistency check
        consistency_score = self._check_information_consistency(case)
        score += consistency_score * 0.1
        details['consistency_score'] = consistency_score
        
        details['field_scores'] = field_scores
        details['optional_bonus'] = optional_bonus
        
        return {
            'score': min(score, 1.0),
            'details': details,
            'issues': issues,
            'recommendations': self._get_information_recommendations(field_scores, issues)
        }
    
    def _score_field_completeness(self, case, field_name) -> float:
        """Score individual field completeness"""
        if field_name == 'target_images':
            return 1.0 if case.target_images else 0.0
        
        value = getattr(case, field_name, None)
        
        if not value:
            return 0.0
        
        if isinstance(value, str):
            if len(value.strip()) < 2:
                return 0.0
            elif len(value.strip()) < 10:
                return 0.5
            else:
                return 1.0
        
        return 1.0 if value else 0.0
    
    def _analyze_text_quality(self, text: str) -> Dict:
        """Analyze quality of text content"""
        if not text:
            return {'score': 0.0, 'issues': ['No text provided']}
        
        text = text.strip()
        words = text.split()
        sentences = text.split('.')
        
        score = 0.0
        issues = []
        
        # Length analysis
        if len(text) < 50:
            issues.append('Description too short')
            score += 0.2
        elif len(text) < 100:
            score += 0.5
        else:
            score += 1.0
        
        # Word count analysis
        if len(words) < 10:
            issues.append('Not enough detail provided')
        elif len(words) >= 20:
            score += 0.2
        
        # Sentence structure
        if len(sentences) > 1:
            score += 0.1
        
        # Information density (keywords)
        info_keywords = ['when', 'where', 'what', 'who', 'how', 'why', 'time', 'date', 'location', 'wearing', 'seen']
        keyword_count = sum(1 for word in words if word.lower() in info_keywords)
        if keyword_count >= 3:
            score += 0.2
        
        # Spam/quality indicators
        spam_indicators = ['click here', 'visit', 'website', 'call now']
        if any(indicator in text.lower() for indicator in spam_indicators):
            issues.append('Content appears promotional')
            score *= 0.5
        
        return {
            'score': min(score / 1.5, 1.0),  # Normalize to 0-1
            'word_count': len(words),
            'character_count': len(text),
            'sentence_count': len(sentences),
            'keyword_density': keyword_count / len(words) if words else 0,
            'issues': issues
        }
    
    def _check_information_consistency(self, case) -> float:
        """Check consistency between different information fields"""
        consistency_score = 1.0
        
        # Check name consistency in details
        if case.person_name and case.details:
            name_parts = case.person_name.lower().split()
            details_lower = case.details.lower()
            
            name_mentioned = any(part in details_lower for part in name_parts if len(part) > 2)
            if not name_mentioned:
                consistency_score -= 0.2
        
        # Check location consistency
        if case.last_seen_location and case.details:
            location_parts = case.last_seen_location.lower().split()
            details_lower = case.details.lower()
            
            location_mentioned = any(part in details_lower for part in location_parts if len(part) > 3)
            if not location_mentioned:
                consistency_score -= 0.2
        
        # Check age consistency with description
        if case.age and case.details:
            age_descriptors = {
                'child': (0, 13), 'teenager': (13, 20), 'young': (20, 35),
                'adult': (35, 60), 'elderly': (60, 120)
            }
            
            details_lower = case.details.lower()
            for descriptor, (min_age, max_age) in age_descriptors.items():
                if descriptor in details_lower:
                    if not (min_age <= case.age < max_age):
                        consistency_score -= 0.3
                    break
        
        return max(consistency_score, 0.0)
    
    def _classify_urgency(self, case) -> Dict:
        """ML-based urgency classification"""
        urgency_score = 0.0
        urgency_level = 'Low'
        factors = []
        
        # Analyze text content for urgency indicators
        text_content = (case.details or '') + ' ' + (case.person_name or '')
        text_lower = text_content.lower()
        
        # Keyword-based urgency scoring
        for level, keywords in self.urgency_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_matches > 0:
                if level == 'critical':
                    urgency_score += keyword_matches * 0.4
                    factors.append(f"Critical keywords found: {keyword_matches}")
                elif level == 'high':
                    urgency_score += keyword_matches * 0.3
                    factors.append(f"High priority keywords found: {keyword_matches}")
                elif level == 'medium':
                    urgency_score += keyword_matches * 0.2
                elif level == 'low':
                    urgency_score += keyword_matches * 0.1
        
        # Age-based urgency (children and elderly are higher priority)
        if case.age:
            if case.age < 13:
                urgency_score += 0.3
                factors.append("Child case - high priority")
            elif case.age > 70:
                urgency_score += 0.2
                factors.append("Elderly case - increased priority")
        
        # Time-based urgency (recent cases are more urgent)
        if case.date_missing:
            days_missing = (datetime.now().date() - case.date_missing.date()).days
            if days_missing <= 1:
                urgency_score += 0.3
                factors.append("Recently missing - urgent")
            elif days_missing <= 7:
                urgency_score += 0.2
                factors.append("Missing within a week")
            elif days_missing <= 30:
                urgency_score += 0.1
        
        # Requester type urgency
        if hasattr(case, 'requester_type'):
            if case.requester_type in ['police', 'government']:
                urgency_score += 0.2
                factors.append("Official request - high priority")
        
        # Determine urgency level
        if urgency_score >= 0.8:
            urgency_level = 'Critical'
        elif urgency_score >= 0.6:
            urgency_level = 'High'
        elif urgency_score >= 0.4:
            urgency_level = 'Medium'
        else:
            urgency_level = 'Low'
        
        return {
            'score': min(urgency_score, 1.0),
            'level': urgency_level,
            'factors': factors,
            'days_missing': (datetime.now().date() - case.date_missing.date()).days if case.date_missing else None
        }
    
    def _assess_duplicate_risk(self, case) -> Dict:
        """Assess risk of duplicate case using similarity algorithms"""
        try:
            from models import Case
            
            # Get recent cases for comparison (last 30 days)
            recent_cases = Case.query.filter(
                Case.id != case.id,
                Case.created_at >= datetime.now() - timedelta(days=30)
            ).all()
            
            if not recent_cases:
                return {
                    'score': 1.0,  # No duplicates found
                    'risk_level': 'Low',
                    'similar_cases': [],
                    'max_similarity': 0.0
                }
            
            similarities = []
            
            for other_case in recent_cases:
                similarity = self._calculate_case_similarity(case, other_case)
                if similarity > 0.3:  # Only consider significant similarities
                    similarities.append({
                        'case_id': other_case.id,
                        'person_name': other_case.person_name,
                        'similarity_score': similarity,
                        'created_at': other_case.created_at
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            max_similarity = similarities[0]['similarity_score'] if similarities else 0.0
            
            # Determine risk level
            if max_similarity >= self.duplicate_similarity_threshold:
                risk_level = 'High'
                score = 0.2  # High penalty for likely duplicates
            elif max_similarity >= 0.7:
                risk_level = 'Medium'
                score = 0.5
            elif max_similarity >= 0.5:
                risk_level = 'Low'
                score = 0.8
            else:
                risk_level = 'Very Low'
                score = 1.0
            
            return {
                'score': score,
                'risk_level': risk_level,
                'similar_cases': similarities[:5],  # Top 5 similar cases
                'max_similarity': max_similarity,
                'total_similar_cases': len(similarities)
            }
            
        except Exception as e:
            logger.error(f"Duplicate risk assessment failed: {str(e)}")
            return {
                'score': 0.8,  # Default moderate score
                'risk_level': 'Unknown',
                'similar_cases': [],
                'max_similarity': 0.0,
                'error': str(e)
            }
    
    def _calculate_case_similarity(self, case1, case2) -> float:
        """Calculate similarity between two cases using multiple factors"""
        similarity_score = 0.0
        
        # Name similarity (40% weight)
        if case1.person_name and case2.person_name:
            name_similarity = self._calculate_text_similarity(
                case1.person_name.lower(), 
                case2.person_name.lower()
            )
            similarity_score += name_similarity * 0.4
        
        # Location similarity (30% weight)
        if case1.last_seen_location and case2.last_seen_location:
            location_similarity = self._calculate_text_similarity(
                case1.last_seen_location.lower(),
                case2.last_seen_location.lower()
            )
            similarity_score += location_similarity * 0.3
        
        # Age similarity (10% weight)
        if case1.age and case2.age:
            age_diff = abs(case1.age - case2.age)
            age_similarity = max(0, 1 - age_diff / 10)  # Similar if within 10 years
            similarity_score += age_similarity * 0.1
        
        # Details similarity (20% weight)
        if case1.details and case2.details:
            details_similarity = self._calculate_text_similarity(
                case1.details.lower(),
                case2.details.lower()
            )
            similarity_score += details_similarity * 0.2
        
        return similarity_score
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Simple character-based similarity for short texts
            if len(text1) < 20 or len(text2) < 20:
                return self._calculate_character_similarity(text1, text2)
            
            # TF-IDF based similarity for longer texts
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception as e:
            logger.error(f"Text similarity calculation failed: {str(e)}")
            return self._calculate_character_similarity(text1, text2)
    
    def _calculate_character_similarity(self, text1: str, text2: str) -> float:
        """Simple character-based similarity calculation"""
        if not text1 or not text2:
            return 0.0
        
        # Levenshtein distance approximation
        len1, len2 = len(text1), len(text2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0
        
        # Simple character overlap
        set1, set2 = set(text1.lower()), set(text2.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        elif score >= 0.5:
            return 'D'
        else:
            return 'F'
    
    def _determine_processing_priority(self, assessment: Dict) -> str:
        """Determine processing priority based on assessment"""
        urgency_level = assessment['urgency_classification']['level']
        quality_score = assessment['overall_score']
        duplicate_risk = assessment['duplicate_risk']['risk_level']
        
        # High urgency cases get priority regardless of quality
        if urgency_level == 'Critical':
            return 'Critical'
        elif urgency_level == 'High':
            return 'High'
        
        # For medium/low urgency, consider quality
        if quality_score >= 0.8:
            return 'High' if urgency_level == 'Medium' else 'Medium'
        elif quality_score >= 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    def _estimate_success_rate(self, assessment: Dict) -> float:
        """Estimate investigation success rate based on case quality"""
        base_rate = 0.5  # 50% base success rate
        
        # Photo quality impact (30% influence)
        photo_impact = assessment['photo_quality']['score'] * 0.3
        
        # Information completeness impact (25% influence)
        info_impact = assessment['information_completeness']['score'] * 0.25
        
        # Urgency impact (20% influence) - urgent cases get more resources
        urgency_score = assessment['urgency_classification']['score']
        urgency_impact = urgency_score * 0.2
        
        # Duplicate risk impact (negative influence)
        duplicate_penalty = (1 - assessment['duplicate_risk']['score']) * 0.15
        
        # Time factor (recent cases have higher success rate)
        time_bonus = 0.1  # Default bonus for recent cases
        
        estimated_rate = base_rate + photo_impact + info_impact + urgency_impact + time_bonus - duplicate_penalty
        
        return min(max(estimated_rate, 0.1), 0.95)  # Clamp between 10% and 95%
    
    def _generate_recommendations(self, assessment: Dict) -> List[str]:
        """Generate actionable recommendations based on assessment"""
        recommendations = []
        
        # Photo quality recommendations
        if assessment['photo_quality']['score'] < 0.6:
            recommendations.extend(assessment['photo_quality']['recommendations'])
        
        # Information completeness recommendations
        if assessment['information_completeness']['score'] < 0.7:
            recommendations.extend(assessment['information_completeness']['recommendations'])
        
        # Duplicate risk recommendations
        if assessment['duplicate_risk']['risk_level'] in ['High', 'Medium']:
            recommendations.append('Review similar cases to avoid duplicates')
        
        # Urgency-based recommendations
        urgency_level = assessment['urgency_classification']['level']
        if urgency_level == 'Critical':
            recommendations.append('Immediate processing required - critical case')
        elif urgency_level == 'High':
            recommendations.append('Prioritize this case for faster processing')
        
        # Success rate recommendations
        if assessment['estimated_success_rate'] < 0.4:
            recommendations.append('Consider requesting additional information to improve success rate')
        
        return recommendations
    
    def _get_photo_recommendations(self, score: float, face_detection_rate: float, issues: List[str]) -> List[str]:
        """Generate photo-specific recommendations"""
        recommendations = []
        
        if score < 0.4:
            recommendations.append('Upload higher quality photos with better lighting')
        
        if face_detection_rate < 0.5:
            recommendations.append('Ensure faces are clearly visible and well-lit')
        
        if 'Photo is blurry' in issues:
            recommendations.append('Use sharper, non-blurry photos')
        
        if 'Low resolution' in issues:
            recommendations.append('Upload higher resolution images (minimum 640x480)')
        
        if 'Face too small in image' in issues:
            recommendations.append('Use photos where the face takes up more of the image')
        
        return recommendations
    
    def _get_information_recommendations(self, field_scores: Dict, issues: List[str]) -> List[str]:
        """Generate information-specific recommendations"""
        recommendations = []
        
        for field, score in field_scores.items():
            if score < 0.5:
                if field == 'person_name':
                    recommendations.append('Provide complete person name')
                elif field == 'last_seen_location':
                    recommendations.append('Add detailed location information with landmarks')
                elif field == 'details':
                    recommendations.append('Provide more detailed case description')
                elif field == 'age':
                    recommendations.append('Include age information if known')
        
        if 'Description too short' in issues:
            recommendations.append('Expand case description with more details')
        
        if 'Not enough detail provided' in issues:
            recommendations.append('Include circumstances, timeline, and relevant details')
        
        return recommendations
    
    def _get_default_assessment(self) -> Dict:
        """Return default assessment in case of errors"""
        return {
            'overall_score': 0.5,
            'photo_quality': {'score': 0.5, 'details': {}, 'issues': [], 'recommendations': []},
            'information_completeness': {'score': 0.5, 'details': {}, 'issues': [], 'recommendations': []},
            'urgency_classification': {'score': 0.5, 'level': 'Medium', 'factors': []},
            'duplicate_risk': {'score': 0.8, 'risk_level': 'Low', 'similar_cases': []},
            'recommendations': ['System error - manual review required'],
            'quality_grade': 'C',
            'processing_priority': 'Medium',
            'estimated_success_rate': 0.5
        }

# Global instance
case_quality_assessor = CaseQualityAssessment()