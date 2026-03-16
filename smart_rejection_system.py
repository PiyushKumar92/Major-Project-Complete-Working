"""
Smart Rejection with Auto-Feedback System
Provides detailed, actionable feedback for case improvements
"""

import cv2
import numpy as np
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

class SmartRejectionSystem:
    """
    Advanced rejection system with specific improvement suggestions,
    examples, and auto-generated improvement templates
    """
    
    def __init__(self):
        self.improvement_templates = self._load_improvement_templates()
        self.example_database = self._load_examples()
        
    def generate_smart_feedback(self, case, validation_scores: Dict, rejection_reasons: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive smart feedback with specific improvements
        """
        feedback = {
            'overall_assessment': self._assess_overall_quality(validation_scores),
            'photo_feedback': self._analyze_photo_issues(case, validation_scores.get('photos', 0)),
            'form_feedback': self._analyze_form_issues(case, validation_scores.get('form_data', 0)),
            'content_feedback': self._analyze_content_issues(case, validation_scores.get('text_quality', 0)),
            'improvement_plan': self._create_improvement_plan(case, validation_scores),
            'examples': self._get_relevant_examples(case, rejection_reasons),
            'templates': self._get_improvement_templates(case, rejection_reasons),
            'priority_actions': self._identify_priority_actions(validation_scores),
            'estimated_approval_chance': self._estimate_approval_chance(validation_scores)
        }
        
        return feedback
    
    def _assess_overall_quality(self, scores: Dict) -> Dict[str, Any]:
        """Assess overall case quality with specific metrics"""
        total_score = sum(scores.values()) / len(scores) if scores else 0
        
        if total_score >= 0.8:
            grade = "A"
            status = "Excellent - Minor improvements needed"
        elif total_score >= 0.6:
            grade = "B"
            status = "Good - Some improvements required"
        elif total_score >= 0.4:
            grade = "C"
            status = "Fair - Significant improvements needed"
        else:
            grade = "D"
            status = "Poor - Major improvements required"
        
        return {
            'grade': grade,
            'score': total_score,
            'status': status,
            'breakdown': scores
        }
    
    def _analyze_photo_issues(self, case, photo_score: float) -> Dict[str, Any]:
        """Detailed photo analysis with specific improvement suggestions"""
        issues = []
        suggestions = []
        examples = []
        
        if not case.target_images:
            issues.append("No photos uploaded")
            suggestions.extend([
                "Upload at least 2 clear photos of the person",
                "Include one frontal face photo and one full body photo",
                "Ensure photos are recent (within last 2 years)"
            ])
            examples.append({
                'type': 'required_photos',
                'description': 'Minimum photo requirements',
                'items': ['Clear frontal face photo', 'Full body photo', 'Side profile (optional)']
            })
        else:
            # Analyze existing photos
            photo_analysis = self._detailed_photo_analysis(case.target_images)
            
            if photo_analysis['blur_issues']:
                issues.append("Photos are blurry or out of focus")
                suggestions.extend([
                    "Use your phone's camera in good lighting",
                    "Hold the phone steady while taking photos",
                    "Tap on the person's face to focus before taking the photo"
                ])
                examples.append({
                    'type': 'photo_quality',
                    'description': 'How to take clear photos',
                    'tips': [
                        'Use natural daylight when possible',
                        'Avoid using digital zoom',
                        'Clean your camera lens before taking photos'
                    ]
                })
            
            if photo_analysis['lighting_issues']:
                issues.append("Poor lighting in photos")
                suggestions.extend([
                    "Take photos in natural daylight or well-lit areas",
                    "Avoid backlighting (don't take photos with light behind the person)",
                    "Use your phone's flash if indoors"
                ])
                examples.append({
                    'type': 'lighting',
                    'description': 'Good lighting examples',
                    'good': ['Outdoor daylight', 'Indoor with good room lighting', 'Even lighting on face'],
                    'bad': ['Backlit photos', 'Very dark photos', 'Harsh shadows on face']
                })
            
            if photo_analysis['face_size_issues']:
                issues.append("Face too small in photos")
                suggestions.extend([
                    "Take photos closer to the person (within 3-6 feet)",
                    "Ensure the face takes up at least 1/4 of the photo",
                    "Crop existing photos to focus on the face area"
                ])
                examples.append({
                    'type': 'face_size',
                    'description': 'Proper face size in photos',
                    'guideline': 'Face should be clearly visible and take up significant portion of image'
                })
            
            if photo_analysis['angle_issues']:
                issues.append("Photos not showing face clearly")
                suggestions.extend([
                    "Include at least one frontal face photo",
                    "Avoid photos where face is turned away",
                    "Remove sunglasses or masks if possible"
                ])
        
        return {
            'score': photo_score,
            'issues': issues,
            'suggestions': suggestions,
            'examples': examples,
            'priority': 'HIGH' if photo_score < 0.4 else 'MEDIUM' if photo_score < 0.7 else 'LOW'
        }
    
    def _analyze_form_issues(self, case, form_score: float) -> Dict[str, Any]:
        """Analyze form completion issues with specific guidance"""
        issues = []
        suggestions = []
        examples = []
        
        # Check each required field
        if not case.person_name or len(case.person_name.strip()) < 2:
            issues.append("Person name missing or too short")
            suggestions.append("Provide the full name of the missing person")
            examples.append({
                'type': 'name_format',
                'good_examples': ['John Smith', 'Priya Sharma', 'Mohammad Ali Khan'],
                'bad_examples': ['J', 'John', 'Person']
            })
        
        if not case.last_seen_location or len(case.last_seen_location.strip()) < 10:
            issues.append("Location information insufficient")
            suggestions.extend([
                "Provide detailed location with landmarks",
                "Include area name, city, and state",
                "Add nearby landmarks or well-known places"
            ])
            examples.append({
                'type': 'location_format',
                'good_examples': [
                    'Near City Mall, Sector 18, Noida, UP',
                    'Bus Stop opposite SBI Bank, MG Road, Bangalore',
                    'Railway Station Platform 2, New Delhi'
                ],
                'bad_examples': ['Delhi', 'Near mall', 'City center']
            })
        
        if not case.age:
            issues.append("Age not provided")
            suggestions.append("Provide approximate age if exact age unknown")
            examples.append({
                'type': 'age_format',
                'note': 'Age helps in identification and search prioritization'
            })
        
        if not case.date_missing:
            issues.append("Missing date not provided")
            suggestions.append("Provide the date when person was last seen")
        
        # Check contact information in details
        if case.details:
            has_contact = self._check_contact_info(case.details)
            if not has_contact:
                issues.append("No contact information provided")
                suggestions.extend([
                    "Include your phone number for updates",
                    "Provide email address for communication",
                    "Add alternative contact person details"
                ])
                examples.append({
                    'type': 'contact_format',
                    'examples': [
                        'Contact: +91-9876543210',
                        'Email: family@example.com',
                        'Alt Contact: Relative Name - 9876543210'
                    ]
                })
        
        return {
            'score': form_score,
            'issues': issues,
            'suggestions': suggestions,
            'examples': examples,
            'priority': 'HIGH' if form_score < 0.5 else 'MEDIUM'
        }
    
    def _analyze_content_issues(self, case, content_score: float) -> Dict[str, Any]:
        """Analyze case description content quality"""
        issues = []
        suggestions = []
        examples = []
        
        if not case.details or len(case.details.strip()) < 50:
            issues.append("Case description too brief")
            suggestions.extend([
                "Provide detailed description of circumstances",
                "Include what the person was wearing",
                "Mention any distinguishing features",
                "Describe the situation when last seen"
            ])
            examples.append({
                'type': 'detailed_description',
                'template': self._get_description_template(),
                'sample': "John was last seen on March 15th at 6 PM near City Mall. He was wearing a blue shirt and black jeans. He is 5'8\" tall with short black hair. He was going to meet a friend but never returned home. He usually carries a black backpack and wears glasses."
            })
        
        if case.details:
            word_count = len(case.details.split())
            if word_count < 20:
                issues.append("Description lacks sufficient detail")
                suggestions.extend([
                    "Expand description to at least 50 words",
                    "Include timeline of events",
                    "Add physical description details",
                    "Mention any relevant circumstances"
                ])
        
        # Check for important keywords
        if case.details:
            important_keywords = ['wearing', 'clothes', 'height', 'hair', 'last seen', 'time', 'going to']
            details_lower = case.details.lower()
            missing_info = []
            
            if not any(word in details_lower for word in ['wearing', 'clothes', 'shirt', 'dress']):
                missing_info.append("clothing description")
            
            if not any(word in details_lower for word in ['height', 'tall', 'short', 'feet']):
                missing_info.append("height information")
            
            if not any(word in details_lower for word in ['hair', 'black hair', 'brown hair']):
                missing_info.append("hair description")
            
            if missing_info:
                issues.append(f"Missing key information: {', '.join(missing_info)}")
                suggestions.append("Include physical characteristics and clothing details")
        
        return {
            'score': content_score,
            'issues': issues,
            'suggestions': suggestions,
            'examples': examples,
            'priority': 'MEDIUM' if content_score < 0.6 else 'LOW'
        }
    
    def _create_improvement_plan(self, case, scores: Dict) -> Dict[str, Any]:
        """Create step-by-step improvement plan"""
        plan_steps = []
        estimated_time = 0
        
        # Priority 1: Critical issues (photos)
        if scores.get('photos', 0) < 0.5:
            plan_steps.append({
                'step': 1,
                'title': 'Fix Photo Issues (Critical)',
                'actions': [
                    'Take new clear photos in good lighting',
                    'Ensure face is clearly visible and large enough',
                    'Upload at least 2 different photos'
                ],
                'time_estimate': '10-15 minutes',
                'priority': 'CRITICAL'
            })
            estimated_time += 15
        
        # Priority 2: Form completion
        if scores.get('form_data', 0) < 0.7:
            plan_steps.append({
                'step': len(plan_steps) + 1,
                'title': 'Complete Required Information',
                'actions': [
                    'Fill in all required fields completely',
                    'Provide detailed location information',
                    'Add contact information'
                ],
                'time_estimate': '5-10 minutes',
                'priority': 'HIGH'
            })
            estimated_time += 10
        
        # Priority 3: Improve description
        if scores.get('text_quality', 0) < 0.6:
            plan_steps.append({
                'step': len(plan_steps) + 1,
                'title': 'Enhance Case Description',
                'actions': [
                    'Expand description with more details',
                    'Include physical characteristics',
                    'Add timeline and circumstances'
                ],
                'time_estimate': '5-10 minutes',
                'priority': 'MEDIUM'
            })
            estimated_time += 8
        
        return {
            'steps': plan_steps,
            'total_estimated_time': f"{estimated_time} minutes",
            'completion_order': 'Follow steps in order for best results'
        }
    
    def _identify_priority_actions(self, scores: Dict) -> List[Dict[str, Any]]:
        """Identify top 3 priority actions for improvement"""
        actions = []
        
        # Sort issues by impact and score
        score_items = [(k, v) for k, v in scores.items()]
        score_items.sort(key=lambda x: x[1])  # Lowest scores first
        
        action_map = {
            'photos': {
                'action': 'Upload clear, well-lit photos with visible faces',
                'impact': 'HIGH',
                'reason': 'Photos are critical for AI identification'
            },
            'form_data': {
                'action': 'Complete all required form fields with detailed information',
                'impact': 'HIGH',
                'reason': 'Complete information enables effective search'
            },
            'text_quality': {
                'action': 'Expand case description with specific details',
                'impact': 'MEDIUM',
                'reason': 'Detailed description helps in manual verification'
            },
            'consistency': {
                'action': 'Ensure information matches across photos and description',
                'impact': 'MEDIUM',
                'reason': 'Consistency improves case credibility'
            }
        }
        
        for score_key, score_value in score_items[:3]:  # Top 3 lowest scores
            if score_value < 0.7 and score_key in action_map:
                action_info = action_map[score_key].copy()
                action_info['current_score'] = f"{score_value:.1%}"
                action_info['target_score'] = "80%+"
                actions.append(action_info)
        
        return actions
    
    def _estimate_approval_chance(self, scores: Dict) -> Dict[str, Any]:
        """Estimate approval chance after improvements"""
        current_avg = sum(scores.values()) / len(scores) if scores else 0
        
        # Calculate potential improvement
        improvement_potential = {}
        for key, score in scores.items():
            if score < 0.8:
                potential_gain = min(0.8 - score, 0.3)  # Max 30% improvement per category
                improvement_potential[key] = score + potential_gain
            else:
                improvement_potential[key] = score
        
        potential_avg = sum(improvement_potential.values()) / len(improvement_potential)
        
        if potential_avg >= 0.75:
            approval_chance = "Very High (85-95%)"
            message = "Following the improvement plan should result in automatic approval"
        elif potential_avg >= 0.6:
            approval_chance = "High (70-85%)"
            message = "Good chance of approval with recommended improvements"
        elif potential_avg >= 0.45:
            approval_chance = "Moderate (50-70%)"
            message = "Approval possible with significant improvements"
        else:
            approval_chance = "Low (30-50%)"
            message = "Major improvements needed for approval"
        
        return {
            'current_score': f"{current_avg:.1%}",
            'potential_score': f"{potential_avg:.1%}",
            'approval_chance': approval_chance,
            'message': message
        }
    
    def _detailed_photo_analysis(self, target_images) -> Dict[str, bool]:
        """Perform detailed analysis of uploaded photos"""
        analysis = {
            'blur_issues': False,
            'lighting_issues': False,
            'face_size_issues': False,
            'angle_issues': False,
            'resolution_issues': False
        }
        
        for image in target_images:
            image_path = os.path.join('app', 'static', image.image_path.replace('static/', ''))
            
            if not os.path.exists(image_path):
                continue
            
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Check blur
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < 500:  # Threshold for blur
                    analysis['blur_issues'] = True
                
                # Check lighting
                brightness = np.mean(gray)
                if brightness < 50 or brightness > 200:
                    analysis['lighting_issues'] = True
                
                # Check resolution
                height, width = img.shape[:2]
                if width < 400 or height < 400:
                    analysis['resolution_issues'] = True
                
                # Check face detection and size
                try:
                    import face_recognition
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_img)
                    
                    if not face_locations:
                        analysis['angle_issues'] = True
                    else:
                        # Check face size
                        for face_location in face_locations:
                            top, right, bottom, left = face_location
                            face_area = (right - left) * (bottom - top)
                            img_area = width * height
                            face_ratio = face_area / img_area
                            
                            if face_ratio < 0.05:  # Face too small
                                analysis['face_size_issues'] = True
                except:
                    pass
                
            except Exception as e:
                continue
        
        return analysis
    
    def _check_contact_info(self, details: str) -> bool:
        """Check if contact information is provided"""
        if not details:
            return False
        
        # Check for phone number patterns
        phone_pattern = r'[\+]?[1-9]?[\d\s\-\(\)]{9,15}'
        has_phone = bool(re.search(phone_pattern, details))
        
        # Check for email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        has_email = bool(re.search(email_pattern, details))
        
        return has_phone or has_email
    
    def _get_relevant_examples(self, case, rejection_reasons: List[str]) -> List[Dict[str, Any]]:
        """Get relevant examples based on specific issues"""
        examples = []
        
        # Photo examples
        if any('photo' in reason.lower() or 'face' in reason.lower() for reason in rejection_reasons):
            examples.append({
                'category': 'Photo Quality',
                'good_example': {
                    'description': 'Clear frontal face photo in good lighting',
                    'characteristics': ['Face clearly visible', 'Good lighting', 'Sharp focus', 'Appropriate size']
                },
                'bad_example': {
                    'description': 'Blurry, dark, or distant photo',
                    'issues': ['Face too small', 'Poor lighting', 'Blurry image', 'Face not visible']
                }
            })
        
        # Location examples
        if any('location' in reason.lower() for reason in rejection_reasons):
            examples.append({
                'category': 'Location Information',
                'good_example': {
                    'description': 'Detailed location with landmarks',
                    'format': 'Near [Landmark], [Area Name], [City], [State]',
                    'sample': 'Near Metro Station Gate 2, Connaught Place, New Delhi, Delhi'
                },
                'bad_example': {
                    'description': 'Vague location information',
                    'issues': ['Too general', 'No landmarks', 'Missing city/state']
                }
            })
        
        return examples
    
    def _get_improvement_templates(self, case, rejection_reasons: List[str]) -> Dict[str, str]:
        """Get improvement templates based on issues"""
        templates = {}
        
        # Description template
        if any('detail' in reason.lower() or 'description' in reason.lower() for reason in rejection_reasons):
            templates['case_description'] = self._get_description_template()
        
        # Contact template
        if any('contact' in reason.lower() for reason in rejection_reasons):
            templates['contact_information'] = self._get_contact_template()
        
        return templates
    
    def _get_description_template(self) -> str:
        """Get case description template"""
        return """
**Case Description Template:**

**Person Details:**
- Full Name: [Person's complete name]
- Age: [Age in years]
- Height: [Height in feet/inches or cm]
- Hair: [Color and style]
- Build: [Slim/Medium/Heavy build]

**Last Seen Information:**
- Date: [DD/MM/YYYY]
- Time: [HH:MM AM/PM]
- Location: [Detailed location with landmarks]
- Circumstances: [What was the person doing, where were they going]

**Clothing Description:**
- Upper: [Shirt/T-shirt color and type]
- Lower: [Pants/jeans color and type]
- Footwear: [Shoes/sandals description]
- Accessories: [Watch, bag, glasses, etc.]

**Additional Information:**
- Any distinguishing marks or features
- Mental/physical health conditions (if relevant)
- Possible destinations or contacts
- Any other relevant details

**Contact Information:**
- Primary Contact: [Name and phone number]
- Alternative Contact: [Name and phone number]
- Email: [Email address]
"""
    
    def _get_contact_template(self) -> str:
        """Get contact information template"""
        return """
**Contact Information Template:**

Primary Contact: [Your Name] - [Phone Number]
Alternative Contact: [Relative/Friend Name] - [Phone Number]
Email: [Your Email Address]
Address: [Your Address for official communication]

**Emergency Contact:**
Police Station: [Local police station name and number]
Family Contact: [Close family member details]
"""
    
    def _load_improvement_templates(self) -> Dict[str, str]:
        """Load pre-defined improvement templates"""
        return {
            'photo_improvement': """
**Photo Improvement Checklist:**
□ Take photos in natural daylight or good indoor lighting
□ Ensure face takes up at least 1/4 of the photo
□ Use clear, sharp focus (tap on face before taking photo)
□ Include at least one frontal face photo
□ Avoid backlighting (light behind person)
□ Remove sunglasses/masks if possible
□ Take multiple angles if available
""",
            'form_completion': """
**Form Completion Checklist:**
□ Full name (first and last name)
□ Accurate age or approximate age
□ Detailed last seen location with landmarks
□ Date when person was last seen
□ Contact phone number
□ Contact email address
□ Case category selection
□ Relationship to missing person
""",
            'description_enhancement': """
**Description Enhancement Guide:**
□ Physical appearance (height, build, hair, distinguishing marks)
□ Clothing worn when last seen
□ Circumstances of disappearance
□ Timeline of events
□ Possible destinations or contacts
□ Relevant background information
□ Any health conditions or concerns
□ Contact information for updates
"""
        }
    
    def _load_examples(self) -> Dict[str, Any]:
        """Load example database for reference"""
        return {
            'good_cases': [
                {
                    'type': 'complete_case',
                    'description': 'Well-documented case with clear photos and detailed information',
                    'features': ['Clear frontal photo', 'Detailed location', 'Complete timeline', 'Contact info']
                }
            ],
            'common_issues': [
                {
                    'issue': 'Blurry photos',
                    'solution': 'Use phone camera in good lighting, hold steady',
                    'prevention': 'Take multiple photos and select the clearest one'
                }
            ]
        }

# Global instance
smart_rejection_system = SmartRejectionSystem()