"""
Intelligent Case Categorization System
NLP + ML classification for automatic case type detection, risk assessment, and priority scoring
"""

import re
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using rule-based classification")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available, using basic NLP")

class IntelligentCaseCategorizer:
    """Advanced NLP + ML case categorization system"""
    
    def __init__(self):
        self.case_type_keywords = {
            'missing_person': {
                'primary': ['missing', 'disappeared', 'vanished', 'lost', 'gone', 'absent', 'whereabouts', 'runaway'],
                'secondary': ['family', 'child', 'teenager', 'elderly', 'person', 'individual', 'last seen', 'contact'],
                'urgency_indicators': ['child', 'elderly', 'medical', 'danger', 'threat', 'kidnap', 'abduct'],
                'risk_factors': ['mental health', 'suicidal', 'depression', 'medication', 'vulnerable', 'disability']
            },
            'criminal_tracking': {
                'primary': ['criminal', 'suspect', 'perpetrator', 'offender', 'fugitive', 'wanted', 'escape'],
                'secondary': ['crime', 'theft', 'robbery', 'assault', 'murder', 'fraud', 'violence', 'illegal'],
                'urgency_indicators': ['armed', 'dangerous', 'violent', 'weapon', 'threat', 'public safety'],
                'risk_factors': ['repeat offender', 'gang', 'organized crime', 'weapon', 'violence history']
            },
            'suspect_identification': {
                'primary': ['identify', 'unknown', 'suspect', 'perpetrator', 'who is', 'recognition'],
                'secondary': ['cctv', 'footage', 'video', 'photo', 'image', 'surveillance', 'witness'],
                'urgency_indicators': ['ongoing', 'active', 'recent', 'immediate', 'current'],
                'risk_factors': ['multiple incidents', 'pattern', 'escalating', 'public place']
            },
            'witness_location': {
                'primary': ['witness', 'saw', 'observed', 'testimony', 'statement', 'account'],
                'secondary': ['incident', 'crime', 'accident', 'event', 'occurrence', 'situation'],
                'urgency_indicators': ['court', 'trial', 'legal', 'deadline', 'time sensitive'],
                'risk_factors': ['intimidation', 'threat', 'safety', 'protection needed']
            },
            'surveillance_analysis': {
                'primary': ['surveillance', 'cctv', 'camera', 'footage', 'video', 'recording'],
                'secondary': ['analysis', 'review', 'examine', 'study', 'investigate', 'monitor'],
                'urgency_indicators': ['real-time', 'live', 'ongoing', 'immediate', 'active'],
                'risk_factors': ['security breach', 'unauthorized', 'intrusion', 'threat']
            },
            'fraud_investigation': {
                'primary': ['fraud', 'scam', 'deception', 'fake', 'counterfeit', 'forgery'],
                'secondary': ['money', 'financial', 'identity', 'document', 'credit', 'bank', 'payment'],
                'urgency_indicators': ['ongoing', 'active', 'large amount', 'multiple victims'],
                'risk_factors': ['organized', 'sophisticated', 'international', 'cyber']
            },
            'security_investigation': {
                'primary': ['security', 'breach', 'unauthorized', 'intrusion', 'violation'],
                'secondary': ['access', 'system', 'facility', 'perimeter', 'protocol', 'clearance'],
                'urgency_indicators': ['active', 'ongoing', 'critical', 'classified', 'sensitive'],
                'risk_factors': ['insider threat', 'espionage', 'sabotage', 'terrorism']
            },
            'vehicle_tracking': {
                'primary': ['vehicle', 'car', 'truck', 'motorcycle', 'license', 'plate'],
                'secondary': ['stolen', 'hit and run', 'accident', 'traffic', 'parking', 'violation'],
                'urgency_indicators': ['accident', 'injury', 'hit and run', 'emergency'],
                'risk_factors': ['reckless driving', 'chase', 'high speed', 'public danger']
            }
        }
        
        self.risk_level_indicators = {
            'critical': {
                'keywords': ['kidnap', 'abduct', 'weapon', 'armed', 'dangerous', 'threat', 'violence', 'murder', 'terrorism'],
                'age_factors': ['child', 'infant', 'baby', 'toddler', 'elderly'],
                'time_factors': ['immediate', 'urgent', 'emergency', 'critical', 'asap'],
                'vulnerability': ['medical condition', 'disability', 'mental health', 'suicidal', 'medication']
            },
            'high': {
                'keywords': ['missing', 'suspect', 'criminal', 'fraud', 'theft', 'assault', 'robbery'],
                'age_factors': ['teenager', 'minor', 'senior'],
                'time_factors': ['recent', 'today', 'yesterday', 'this week'],
                'vulnerability': ['vulnerable', 'at risk', 'homeless', 'runaway']
            },
            'medium': {
                'keywords': ['witness', 'identify', 'locate', 'find', 'search', 'investigation'],
                'age_factors': ['adult', 'young adult'],
                'time_factors': ['last month', 'few weeks', 'recently'],
                'vulnerability': ['concerned', 'worried', 'unusual']
            },
            'low': {
                'keywords': ['verification', 'background', 'check', 'confirm', 'routine'],
                'age_factors': [],
                'time_factors': ['old case', 'historical', 'archive'],
                'vulnerability': []
            }
        }
        
        self.priority_scoring_factors = {
            'urgency_multiplier': {
                'critical': 2.0,
                'high': 1.5,
                'medium': 1.0,
                'low': 0.7
            },
            'case_type_priority': {
                'missing_person': 1.8,
                'criminal_tracking': 1.9,
                'suspect_identification': 1.6,
                'witness_location': 1.4,
                'surveillance_analysis': 1.2,
                'fraud_investigation': 1.3,
                'security_investigation': 1.7,
                'vehicle_tracking': 1.1
            },
            'requester_authority': {
                'police_officer': 1.8,
                'detective': 1.9,
                'law_enforcement': 2.0,
                'government_official': 1.7,
                'security_agency': 1.6,
                'family_member': 1.4,
                'private_investigator': 1.2,
                'concerned_citizen': 1.0
            }
        }
        
        # Initialize ML models if available
        self.tfidf_vectorizer = None
        self.case_classifier = None
        self.risk_classifier = None
        
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models with training data"""
        try:
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            # Initialize classifiers
            self.case_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.risk_classifier = LogisticRegression(random_state=42)
            
            # Train with synthetic data (in production, use real historical data)
            self._train_models_with_synthetic_data()
            
        except Exception as e:
            logger.error(f"ML model initialization failed: {str(e)}")
            self.tfidf_vectorizer = None
            self.case_classifier = None
            self.risk_classifier = None
    
    def _train_models_with_synthetic_data(self):
        """Train models with synthetic training data"""
        try:
            # Synthetic training data (in production, use real case data)
            training_texts = []
            training_labels = []
            risk_labels = []
            
            # Generate synthetic training examples
            for case_type, keywords in self.case_type_keywords.items():
                for _ in range(20):  # 20 examples per case type
                    # Create synthetic case description
                    primary_words = np.random.choice(keywords['primary'], size=2, replace=False)
                    secondary_words = np.random.choice(keywords['secondary'], size=2, replace=False)
                    
                    text = f"Case involving {' '.join(primary_words)} with {' '.join(secondary_words)}"
                    
                    # Add urgency indicators randomly
                    if np.random.random() > 0.7 and keywords['urgency_indicators']:
                        urgency_word = np.random.choice(keywords['urgency_indicators'])
                        text += f" {urgency_word} situation"
                        risk_labels.append('high')
                    else:
                        risk_labels.append('medium')
                    
                    training_texts.append(text)
                    training_labels.append(case_type)
            
            # Train TF-IDF vectorizer
            X = self.tfidf_vectorizer.fit_transform(training_texts)
            
            # Train case type classifier
            self.case_classifier.fit(X, training_labels)
            
            # Train risk classifier
            self.risk_classifier.fit(X, risk_labels)
            
            logger.info("ML models trained successfully with synthetic data")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
    
    def categorize_case(self, case) -> Dict:
        """Main categorization function - returns comprehensive analysis"""
        try:
            # Extract text content for analysis
            text_content = self._extract_case_text(case)
            
            # Perform categorization
            categorization_result = {
                'case_type_detection': self._detect_case_type(text_content, case),
                'risk_assessment': self._assess_risk_level(text_content, case),
                'priority_scoring': self._calculate_priority_score(text_content, case),
                'tag_generation': self._generate_tags(text_content, case),
                'confidence_scores': {},
                'recommendations': [],
                'processing_notes': []
            }
            
            # Add confidence scores
            categorization_result['confidence_scores'] = {
                'case_type_confidence': categorization_result['case_type_detection']['confidence'],
                'risk_confidence': categorization_result['risk_assessment']['confidence'],
                'priority_confidence': categorization_result['priority_scoring']['confidence'],
                'overall_confidence': self._calculate_overall_confidence(categorization_result)
            }
            
            # Generate recommendations
            categorization_result['recommendations'] = self._generate_recommendations(categorization_result, case)
            
            # Add processing notes
            categorization_result['processing_notes'] = self._generate_processing_notes(categorization_result)
            
            return categorization_result
            
        except Exception as e:
            logger.error(f"Case categorization failed: {str(e)}")
            return self._get_default_categorization()
    
    def _extract_case_text(self, case) -> str:
        """Extract all relevant text from case for analysis"""
        text_parts = []
        
        # Basic case information
        if hasattr(case, 'person_name') and case.person_name:
            text_parts.append(case.person_name)
        
        if hasattr(case, 'details') and case.details:
            text_parts.append(case.details)
        
        if hasattr(case, 'last_seen_location') and case.last_seen_location:
            text_parts.append(case.last_seen_location)
        
        # Form-specific fields
        if hasattr(case, 'case_type') and case.case_type:
            text_parts.append(case.case_type.replace('_', ' '))
        
        if hasattr(case, 'case_category') and case.case_category:
            text_parts.append(case.case_category.replace('_', ' '))
        
        if hasattr(case, 'requester_type') and case.requester_type:
            text_parts.append(case.requester_type.replace('_', ' '))
        
        # Additional context
        if hasattr(case, 'clothing_description') and case.clothing_description:
            text_parts.append(case.clothing_description)
        
        return ' '.join(text_parts).lower()
    
    def _detect_case_type(self, text_content: str, case) -> Dict:
        """Detect case type using NLP + ML"""
        # Try ML-based detection first
        if self.case_classifier and self.tfidf_vectorizer:
            try:
                X = self.tfidf_vectorizer.transform([text_content])
                predicted_type = self.case_classifier.predict(X)[0]
                probabilities = self.case_classifier.predict_proba(X)[0]
                confidence = max(probabilities)
                
                return {
                    'detected_type': predicted_type,
                    'confidence': float(confidence),
                    'method': 'ml_classification',
                    'alternative_types': self._get_alternative_types(probabilities),
                    'manual_override': getattr(case, 'case_type', None)
                }
            except Exception as e:
                logger.warning(f"ML classification failed: {str(e)}, falling back to rule-based")
        
        # Fallback to rule-based detection
        return self._rule_based_case_type_detection(text_content, case)
    
    def _rule_based_case_type_detection(self, text_content: str, case) -> Dict:
        """Rule-based case type detection using keyword matching"""
        type_scores = {}
        
        for case_type, keywords in self.case_type_keywords.items():
            score = 0.0
            
            # Primary keywords (high weight)
            for keyword in keywords['primary']:
                if keyword in text_content:
                    score += 2.0
            
            # Secondary keywords (medium weight)
            for keyword in keywords['secondary']:
                if keyword in text_content:
                    score += 1.0
            
            # Urgency indicators (bonus)
            for keyword in keywords['urgency_indicators']:
                if keyword in text_content:
                    score += 0.5
            
            type_scores[case_type] = score
        
        # Get best match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            max_score = type_scores[best_type]
            
            # Calculate confidence based on score distribution
            total_score = sum(type_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
            
            return {
                'detected_type': best_type,
                'confidence': min(confidence, 0.95),  # Cap at 95%
                'method': 'rule_based',
                'scores': type_scores,
                'manual_override': getattr(case, 'case_type', None)
            }
        
        return {
            'detected_type': 'other_investigation',
            'confidence': 0.3,
            'method': 'default',
            'scores': {},
            'manual_override': getattr(case, 'case_type', None)
        }
    
    def _assess_risk_level(self, text_content: str, case) -> Dict:
        """Assess risk level using multiple factors"""
        risk_scores = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        risk_factors = []
        
        # Keyword-based risk assessment
        for risk_level, indicators in self.risk_level_indicators.items():
            level_score = 0
            
            # Check keywords
            for keyword in indicators['keywords']:
                if keyword in text_content:
                    level_score += 2.0
                    risk_factors.append(f"Keyword: {keyword}")
            
            # Check age factors
            for age_factor in indicators['age_factors']:
                if age_factor in text_content:
                    level_score += 1.5
                    risk_factors.append(f"Age factor: {age_factor}")
            
            # Check time factors
            for time_factor in indicators['time_factors']:
                if time_factor in text_content:
                    level_score += 1.0
                    risk_factors.append(f"Time factor: {time_factor}")
            
            # Check vulnerability factors
            for vuln_factor in indicators['vulnerability']:
                if vuln_factor in text_content:
                    level_score += 2.5
                    risk_factors.append(f"Vulnerability: {vuln_factor}")
            
            risk_scores[risk_level] = level_score
        
        # Additional contextual risk assessment
        
        # Age-based risk (if age is provided)
        if hasattr(case, 'age') and case.age is not None:
            if case.age < 13:  # Child
                risk_scores['critical'] += 3.0
                risk_factors.append("Child case - high risk")
            elif case.age > 70:  # Elderly
                risk_scores['high'] += 2.0
                risk_factors.append("Elderly case - increased risk")
        
        # Time-based risk (recent cases are higher risk)
        if hasattr(case, 'date_missing') and case.date_missing:
            days_missing = (datetime.now().date() - case.date_missing.date()).days
            if days_missing <= 1:
                risk_scores['critical'] += 2.0
                risk_factors.append("Recent disappearance - critical")
            elif days_missing <= 7:
                risk_scores['high'] += 1.5
                risk_factors.append("Recent case - high priority")
        
        # Requester type risk assessment
        if hasattr(case, 'requester_type') and case.requester_type:
            if case.requester_type in ['police_officer', 'law_enforcement', 'government_official']:
                risk_scores['high'] += 1.0
                risk_factors.append("Official request - elevated priority")
        
        # Determine final risk level
        max_risk_level = max(risk_scores, key=risk_scores.get)
        max_score = risk_scores[max_risk_level]
        
        # Calculate confidence
        total_score = sum(risk_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        
        return {
            'risk_level': max_risk_level,
            'confidence': min(confidence, 0.95),
            'risk_factors': risk_factors,
            'risk_scores': risk_scores,
            'assessment_method': 'multi_factor_analysis'
        }
    
    def _calculate_priority_score(self, text_content: str, case) -> Dict:
        """Calculate priority score using historical data patterns"""
        base_score = 50.0  # Base priority score (0-100 scale)
        scoring_factors = []
        
        # Case type priority multiplier
        case_type = getattr(case, 'case_type', 'other_investigation')
        type_multiplier = self.priority_scoring_factors['case_type_priority'].get(case_type, 1.0)
        base_score *= type_multiplier
        scoring_factors.append(f"Case type multiplier: {type_multiplier:.1f}")
        
        # Requester authority multiplier
        requester_type = getattr(case, 'requester_type', 'concerned_citizen')
        authority_multiplier = self.priority_scoring_factors['requester_authority'].get(requester_type, 1.0)
        base_score *= authority_multiplier
        scoring_factors.append(f"Requester authority: {authority_multiplier:.1f}")
        
        # Urgency level multiplier
        urgency_level = getattr(case, 'urgency_level', 'medium')
        urgency_multiplier = self.priority_scoring_factors['urgency_multiplier'].get(urgency_level, 1.0)
        base_score *= urgency_multiplier
        scoring_factors.append(f"Urgency multiplier: {urgency_multiplier:.1f}")
        
        # Time sensitivity bonus
        if hasattr(case, 'date_missing') and case.date_missing:
            days_missing = (datetime.now().date() - case.date_missing.date()).days
            if days_missing <= 1:
                base_score += 20  # Recent cases get priority
                scoring_factors.append("Recent case bonus: +20")
            elif days_missing <= 7:
                base_score += 10
                scoring_factors.append("Weekly case bonus: +10")
        
        # Information quality bonus
        info_quality_score = self._assess_information_quality(case)
        base_score += info_quality_score * 10  # Up to 10 points for quality
        scoring_factors.append(f"Information quality: +{info_quality_score * 10:.1f}")
        
        # Media availability bonus
        media_bonus = 0
        if hasattr(case, 'target_images') and case.target_images:
            media_bonus += 5
        if hasattr(case, 'search_videos') and case.search_videos:
            media_bonus += 5
        base_score += media_bonus
        if media_bonus > 0:
            scoring_factors.append(f"Media availability: +{media_bonus}")
        
        # Cap the score at 100
        final_score = min(base_score, 100.0)
        
        # Determine priority category
        if final_score >= 80:
            priority_category = 'Critical'
        elif final_score >= 65:
            priority_category = 'High'
        elif final_score >= 40:
            priority_category = 'Medium'
        else:
            priority_category = 'Low'
        
        return {
            'priority_score': final_score,
            'priority_category': priority_category,
            'scoring_factors': scoring_factors,
            'confidence': 0.85,  # High confidence in rule-based scoring
            'recommended_sla': self._get_recommended_sla(priority_category)
        }
    
    def _assess_information_quality(self, case) -> float:
        """Assess the quality and completeness of case information"""
        quality_score = 0.0
        
        # Required fields completeness
        required_fields = ['person_name', 'last_seen_location', 'details']
        completed_required = sum(1 for field in required_fields if getattr(case, field, None))
        quality_score += (completed_required / len(required_fields)) * 0.4
        
        # Optional fields bonus
        optional_fields = ['age', 'clothing_description', 'contact_address']
        completed_optional = sum(1 for field in optional_fields if getattr(case, field, None))
        quality_score += (completed_optional / len(optional_fields)) * 0.2
        
        # Text quality assessment
        if hasattr(case, 'details') and case.details:
            text_length = len(case.details)
            if text_length >= 100:
                quality_score += 0.2
            elif text_length >= 50:
                quality_score += 0.1
            
            # Word count
            word_count = len(case.details.split())
            if word_count >= 20:
                quality_score += 0.1
        
        # Media availability
        if hasattr(case, 'target_images') and case.target_images:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _generate_tags(self, text_content: str, case) -> Dict:
        """Generate searchable tags using NLP analysis"""
        tags = {
            'automatic_tags': [],
            'entity_tags': [],
            'location_tags': [],
            'temporal_tags': [],
            'risk_tags': [],
            'category_tags': []
        }
        
        # Automatic keyword extraction
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text_content)
                # Extract noun phrases as potential tags
                noun_phrases = [phrase.lower() for phrase in blob.noun_phrases if len(phrase) > 3]
                tags['automatic_tags'] = list(set(noun_phrases))[:10]  # Top 10 unique phrases
            except:
                pass
        
        # Rule-based tag extraction
        
        # Entity tags (person-related)
        entity_patterns = {
            'age_group': r'\b(child|teenager|adult|elderly|infant|toddler|senior)\b',
            'gender': r'\b(male|female|man|woman|boy|girl)\b',
            'relationship': r'\b(father|mother|son|daughter|brother|sister|friend|colleague)\b'
        }
        
        for tag_type, pattern in entity_patterns.items():
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            if matches:
                tags['entity_tags'].extend([f"{tag_type}:{match.lower()}" for match in set(matches)])
        
        # Location tags
        location_keywords = ['street', 'road', 'area', 'city', 'town', 'village', 'district', 'state', 'near', 'park', 'school', 'hospital']
        for keyword in location_keywords:
            if keyword in text_content:
                tags['location_tags'].append(f"location:{keyword}")
        
        # Temporal tags
        temporal_keywords = ['today', 'yesterday', 'last week', 'last month', 'recent', 'morning', 'evening', 'night']
        for keyword in temporal_keywords:
            if keyword in text_content:
                tags['temporal_tags'].append(f"time:{keyword}")
        
        # Risk tags
        risk_keywords = ['danger', 'threat', 'weapon', 'violence', 'medical', 'emergency', 'urgent']
        for keyword in risk_keywords:
            if keyword in text_content:
                tags['risk_tags'].append(f"risk:{keyword}")
        
        # Category tags based on case type
        if hasattr(case, 'case_type') and case.case_type:
            tags['category_tags'].append(f"type:{case.case_type}")
        
        if hasattr(case, 'case_category') and case.case_category:
            tags['category_tags'].append(f"category:{case.case_category}")
        
        # Clean and deduplicate tags
        for tag_type in tags:
            tags[tag_type] = list(set(tags[tag_type]))[:5]  # Limit to 5 per type
        
        return tags
    
    def _get_alternative_types(self, probabilities) -> List[Dict]:
        """Get alternative case types with probabilities"""
        if not hasattr(self, 'case_classifier') or not self.case_classifier:
            return []
        
        try:
            classes = self.case_classifier.classes_
            alternatives = []
            
            # Get top 3 alternatives
            top_indices = np.argsort(probabilities)[-3:][::-1]
            
            for idx in top_indices:
                alternatives.append({
                    'case_type': classes[idx],
                    'probability': float(probabilities[idx])
                })
            
            return alternatives
        except:
            return []
    
    def _calculate_overall_confidence(self, categorization_result) -> float:
        """Calculate overall confidence score"""
        confidences = [
            categorization_result['case_type_detection']['confidence'],
            categorization_result['risk_assessment']['confidence'],
            categorization_result['priority_scoring']['confidence']
        ]
        
        return sum(confidences) / len(confidences)
    
    def _generate_recommendations(self, categorization_result, case) -> List[str]:
        """Generate actionable recommendations based on categorization"""
        recommendations = []
        
        # Case type specific recommendations
        detected_type = categorization_result['case_type_detection']['detected_type']
        
        if detected_type == 'missing_person':
            recommendations.append("Contact local authorities and hospitals")
            recommendations.append("Gather recent photos and detailed physical description")
            recommendations.append("Check social media and digital footprints")
        
        elif detected_type == 'criminal_tracking':
            recommendations.append("Coordinate with law enforcement agencies")
            recommendations.append("Analyze surveillance footage from crime scene")
            recommendations.append("Interview witnesses and gather evidence")
        
        elif detected_type == 'fraud_investigation':
            recommendations.append("Preserve digital evidence and transaction records")
            recommendations.append("Contact financial institutions")
            recommendations.append("Document all communication with suspects")
        
        # Risk level recommendations
        risk_level = categorization_result['risk_assessment']['risk_level']
        
        if risk_level == 'critical':
            recommendations.append("IMMEDIATE ACTION REQUIRED - Escalate to emergency response")
            recommendations.append("Activate all available resources")
        
        elif risk_level == 'high':
            recommendations.append("Prioritize case processing and resource allocation")
            recommendations.append("Consider media alerts if appropriate")
        
        # Priority score recommendations
        priority_score = categorization_result['priority_scoring']['priority_score']
        
        if priority_score >= 80:
            recommendations.append("Fast-track case processing")
            recommendations.append("Assign senior investigators")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def _generate_processing_notes(self, categorization_result) -> List[str]:
        """Generate internal processing notes"""
        notes = []
        
        # Confidence notes
        overall_confidence = categorization_result['confidence_scores']['overall_confidence']
        if overall_confidence < 0.6:
            notes.append("Low confidence categorization - manual review recommended")
        
        # Method notes
        method = categorization_result['case_type_detection']['method']
        notes.append(f"Case type detected using: {method}")
        
        # Risk assessment notes
        risk_factors = categorization_result['risk_assessment']['risk_factors']
        if risk_factors:
            notes.append(f"Risk factors identified: {len(risk_factors)} factors")
        
        return notes
    
    def _get_recommended_sla(self, priority_category: str) -> str:
        """Get recommended Service Level Agreement based on priority"""
        sla_mapping = {
            'Critical': '2 hours',
            'High': '8 hours',
            'Medium': '24 hours',
            'Low': '72 hours'
        }
        return sla_mapping.get(priority_category, '24 hours')
    
    def _get_default_categorization(self) -> Dict:
        """Return default categorization in case of errors"""
        return {
            'case_type_detection': {
                'detected_type': 'other_investigation',
                'confidence': 0.3,
                'method': 'default'
            },
            'risk_assessment': {
                'risk_level': 'medium',
                'confidence': 0.5,
                'risk_factors': []
            },
            'priority_scoring': {
                'priority_score': 50.0,
                'priority_category': 'Medium',
                'confidence': 0.5
            },
            'tag_generation': {
                'automatic_tags': [],
                'entity_tags': [],
                'location_tags': [],
                'temporal_tags': [],
                'risk_tags': [],
                'category_tags': []
            },
            'confidence_scores': {
                'overall_confidence': 0.4
            },
            'recommendations': ['Manual review required'],
            'processing_notes': ['System error - default categorization applied']
        }

# Global categorizer instance
intelligent_categorizer = IntelligentCaseCategorizer()