"""
Outcome Prediction Engine
ML-based success probability and case resolution likelihood scoring
"""

import os
import json
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from pathlib import Path

@dataclass
class PredictionResult:
    case_id: int
    success_probability: float
    resolution_likelihood: float
    estimated_timeline: int  # days
    resource_requirements: Dict
    success_factors: List[str]
    risk_factors: List[str]
    confidence_level: float

@dataclass
class SimilarCase:
    case_id: int
    similarity_score: float
    outcome: str
    timeline_days: int
    success_factors: List[str]

class OutcomePredictionEngine:
    def __init__(self):
        self.prediction_models = self._initialize_models()
        self.historical_patterns = self._load_historical_patterns()
        self.success_factors = self._load_success_factors()
        self.setup_logging()
        self.setup_database()
    
    def setup_logging(self):
        """Setup prediction logging"""
        log_dir = Path('logs/prediction')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('OutcomePrediction')
        handler = logging.FileHandler('logs/prediction/outcome_prediction.log')
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def setup_database(self):
        """Setup prediction tracking database"""
        db_path = 'instance/outcome_prediction.db'
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Prediction results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcome_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                success_probability REAL NOT NULL,
                resolution_likelihood REAL NOT NULL,
                estimated_timeline INTEGER NOT NULL,
                resource_requirements TEXT,
                success_factors TEXT,
                risk_factors TEXT,
                confidence_level REAL NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                actual_outcome TEXT,
                actual_timeline INTEGER,
                prediction_accuracy REAL
            )
        ''')
        
        # Similar cases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS similar_cases_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                similar_case_id INTEGER NOT NULL,
                similarity_score REAL NOT NULL,
                outcome TEXT,
                timeline_days INTEGER,
                factors_matched TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _initialize_models(self) -> Dict:
        """Initialize ML prediction models"""
        return {
            'success_probability': {
                'weights': {
                    'photo_quality': 0.25,
                    'information_completeness': 0.20,
                    'case_type': 0.15,
                    'time_since_missing': 0.15,
                    'location_specificity': 0.10,
                    'evidence_count': 0.10,
                    'surveillance_availability': 0.05
                },
                'baseline_success_rates': {
                    'missing_person': 0.65,
                    'criminal_investigation': 0.45,
                    'surveillance_request': 0.80,
                    'person_tracking': 0.70,
                    'evidence_analysis': 0.55
                }
            },
            'timeline_prediction': {
                'base_timelines': {
                    'missing_person': 14,  # days
                    'criminal_investigation': 30,
                    'surveillance_request': 7,
                    'person_tracking': 10,
                    'evidence_analysis': 21
                },
                'complexity_multipliers': {
                    'low': 0.7,
                    'medium': 1.0,
                    'high': 1.5,
                    'critical': 2.0
                }
            }
        }
    
    def _load_historical_patterns(self) -> Dict:
        """Load historical case patterns for prediction"""
        return {
            'success_patterns': {
                'high_quality_photos': {'success_rate': 0.85, 'avg_timeline': 12},
                'multiple_sightings': {'success_rate': 0.78, 'avg_timeline': 8},
                'recent_disappearance': {'success_rate': 0.72, 'avg_timeline': 5},
                'urban_location': {'success_rate': 0.68, 'avg_timeline': 10},
                'family_cooperation': {'success_rate': 0.75, 'avg_timeline': 14}
            },
            'risk_patterns': {
                'poor_photo_quality': {'success_impact': -0.25, 'timeline_impact': 1.4},
                'rural_location': {'success_impact': -0.15, 'timeline_impact': 1.3},
                'old_case': {'success_impact': -0.30, 'timeline_impact': 1.6},
                'limited_information': {'success_impact': -0.20, 'timeline_impact': 1.2}
            }
        }
    
    def _load_success_factors(self) -> Dict:
        """Load factors that contribute to case success"""
        return {
            'critical_factors': [
                'high_quality_reference_photos',
                'recent_timeline',
                'specific_location_information',
                'multiple_evidence_sources',
                'active_surveillance_coverage'
            ],
            'supporting_factors': [
                'family_cooperation',
                'witness_availability',
                'social_media_presence',
                'distinctive_appearance',
                'regular_location_patterns'
            ],
            'risk_factors': [
                'poor_photo_quality',
                'vague_location_information',
                'old_timeline',
                'limited_cooperation',
                'no_surveillance_coverage'
            ]
        }
    
    def predict_case_outcome(self, case_id: int) -> PredictionResult:
        """Predict case outcome using ML analysis"""
        try:
            from models import Case
            from __init__ import create_app, db
            
            app = create_app()
            with app.app_context():
                case = Case.query.get(case_id)
                if not case:
                    raise ValueError(f"Case {case_id} not found")
                
                # Extract case features
                features = self._extract_case_features(case)
                
                # Calculate success probability
                success_prob = self._calculate_success_probability(case, features)
                
                # Calculate resolution likelihood
                resolution_likelihood = self._calculate_resolution_likelihood(case, features)
                
                # Estimate timeline
                estimated_timeline = self._estimate_timeline(case, features)
                
                # Predict resource requirements
                resource_requirements = self._predict_resource_requirements(case, features)
                
                # Identify success and risk factors
                success_factors, risk_factors = self._identify_factors(case, features)
                
                # Calculate confidence level
                confidence_level = self._calculate_confidence(features)
                
                # Find similar cases
                similar_cases = self._find_similar_cases(case, features)
                
                result = PredictionResult(
                    case_id=case_id,
                    success_probability=success_prob,
                    resolution_likelihood=resolution_likelihood,
                    estimated_timeline=estimated_timeline,
                    resource_requirements=resource_requirements,
                    success_factors=success_factors,
                    risk_factors=risk_factors,
                    confidence_level=confidence_level
                )
                
                # Store prediction
                self._store_prediction(result, similar_cases)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error predicting outcome for case {case_id}: {e}")
            return PredictionResult(
                case_id=case_id,
                success_probability=0.5,
                resolution_likelihood=0.5,
                estimated_timeline=30,
                resource_requirements={},
                success_factors=[],
                risk_factors=['prediction_error'],
                confidence_level=0.0
            )
    
    def _extract_case_features(self, case) -> Dict:
        """Extract features from case for ML prediction"""
        features = {}
        
        # Basic case information
        features['case_type'] = case.case_type or 'missing_person'
        features['priority'] = case.priority or 'Medium'
        features['requester_type'] = case.requester_type or 'family'
        
        # Time-based features
        if case.date_missing:
            days_missing = (datetime.utcnow() - case.date_missing).days
            features['days_missing'] = min(days_missing, 365)  # Cap at 1 year
        else:
            features['days_missing'] = 0
        
        # Photo quality assessment
        photo_count = len(case.target_images)
        features['photo_count'] = photo_count
        features['has_photos'] = photo_count > 0
        
        # Try to get quality assessment
        try:
            quality_assessment = case.quality_assessments[-1] if case.quality_assessments else None
            if quality_assessment:
                features['photo_quality_score'] = quality_assessment.photo_quality_score
                features['information_completeness'] = quality_assessment.information_completeness_score
                features['overall_quality'] = quality_assessment.overall_score
            else:
                features['photo_quality_score'] = 0.5 if photo_count > 0 else 0.0
                features['information_completeness'] = 0.6
                features['overall_quality'] = 0.5
        except:
            features['photo_quality_score'] = 0.5 if photo_count > 0 else 0.0
            features['information_completeness'] = 0.6
            features['overall_quality'] = 0.5
        
        # Location features
        features['has_location'] = bool(case.last_seen_location)
        if case.last_seen_location:
            location_specificity = len(case.last_seen_location.split(','))
            features['location_specificity'] = min(location_specificity / 5.0, 1.0)
        else:
            features['location_specificity'] = 0.0
        
        # Evidence and analysis features
        features['sightings_count'] = len(case.sightings)
        features['high_conf_sightings'] = len([s for s in case.sightings if s.confidence_score > 0.8])
        features['verified_sightings'] = len([s for s in case.sightings if s.verified])
        
        # Surveillance coverage
        try:
            location_matches_count = len(case.location_matches)
            features['surveillance_coverage'] = min(location_matches_count / 5.0, 1.0)
        except:
            features['surveillance_coverage'] = 0.0
        
        # Case complexity
        complexity_score = 0
        if features['days_missing'] > 30:
            complexity_score += 1
        if features['photo_quality_score'] < 0.5:
            complexity_score += 1
        if features['location_specificity'] < 0.5:
            complexity_score += 1
        if features['information_completeness'] < 0.6:
            complexity_score += 1
        
        features['complexity'] = ['low', 'medium', 'high', 'critical'][min(complexity_score, 3)]
        
        return features
    
    def _calculate_success_probability(self, case, features: Dict) -> float:
        """Calculate probability of successful case resolution"""
        model = self.prediction_models['success_probability']
        
        # Start with baseline success rate for case type
        base_rate = model['baseline_success_rates'].get(features['case_type'], 0.6)
        
        # Apply weighted factors
        weighted_score = 0.0
        weights = model['weights']
        
        # Photo quality impact
        weighted_score += weights['photo_quality'] * features['photo_quality_score']
        
        # Information completeness impact
        weighted_score += weights['information_completeness'] * features['information_completeness']
        
        # Case type impact (normalized)
        case_type_scores = {'missing_person': 0.8, 'criminal_investigation': 0.6, 'surveillance_request': 0.9}
        case_type_score = case_type_scores.get(features['case_type'], 0.7)
        weighted_score += weights['case_type'] * case_type_score
        
        # Time impact (recent cases have higher success rate)
        time_score = max(0.1, 1.0 - (features['days_missing'] / 365.0))
        weighted_score += weights['time_since_missing'] * time_score
        
        # Location specificity impact
        weighted_score += weights['location_specificity'] * features['location_specificity']
        
        # Evidence count impact
        evidence_score = min(1.0, (features['sightings_count'] + features['verified_sightings']) / 10.0)
        weighted_score += weights['evidence_count'] * evidence_score
        
        # Surveillance availability impact
        weighted_score += weights['surveillance_availability'] * features['surveillance_coverage']
        
        # Combine base rate with weighted factors
        final_probability = (base_rate * 0.4) + (weighted_score * 0.6)
        
        # Apply historical pattern adjustments
        for pattern, data in self.historical_patterns['success_patterns'].items():
            if self._pattern_matches(pattern, features):
                final_probability *= (1.0 + (data['success_rate'] - 0.6) * 0.3)
        
        for pattern, data in self.historical_patterns['risk_patterns'].items():
            if self._pattern_matches(pattern, features):
                final_probability += data['success_impact'] * 0.5
        
        return max(0.05, min(0.95, final_probability))
    
    def _calculate_resolution_likelihood(self, case, features: Dict) -> float:
        """Calculate likelihood of case resolution in near term"""
        # Base likelihood on case activity and evidence
        base_likelihood = 0.5
        
        # Recent activity increases likelihood
        if case.updated_at and (datetime.utcnow() - case.updated_at).days < 7:
            base_likelihood += 0.2
        
        # Evidence quality impact
        if features['verified_sightings'] > 0:
            base_likelihood += 0.3
        elif features['high_conf_sightings'] > 0:
            base_likelihood += 0.2
        elif features['sightings_count'] > 0:
            base_likelihood += 0.1
        
        # Surveillance coverage impact
        base_likelihood += features['surveillance_coverage'] * 0.2
        
        # Time decay (older cases less likely to resolve quickly)
        time_factor = max(0.3, 1.0 - (features['days_missing'] / 180.0))
        base_likelihood *= time_factor
        
        return max(0.1, min(0.9, base_likelihood))
    
    def _estimate_timeline(self, case, features: Dict) -> int:
        """Estimate timeline for case resolution"""
        model = self.prediction_models['timeline_prediction']
        
        # Base timeline for case type
        base_timeline = model['base_timelines'].get(features['case_type'], 21)
        
        # Apply complexity multiplier
        complexity_multiplier = model['complexity_multipliers'].get(features['complexity'], 1.0)
        estimated_timeline = base_timeline * complexity_multiplier
        
        # Adjust based on current evidence
        if features['verified_sightings'] > 0:
            estimated_timeline *= 0.7  # Faster with verified evidence
        elif features['high_conf_sightings'] > 0:
            estimated_timeline *= 0.8
        
        # Adjust based on photo quality
        if features['photo_quality_score'] > 0.8:
            estimated_timeline *= 0.8
        elif features['photo_quality_score'] < 0.4:
            estimated_timeline *= 1.3
        
        # Adjust based on time already passed
        if features['days_missing'] > 30:
            estimated_timeline *= 1.2
        
        return max(3, min(90, int(estimated_timeline)))
    
    def _predict_resource_requirements(self, case, features: Dict) -> Dict:
        """Predict resource requirements for case resolution"""
        requirements = {
            'ai_analysis_hours': 0,
            'manual_review_hours': 0,
            'surveillance_footage_needed': 0,
            'admin_involvement_level': 'low',
            'estimated_cost_category': 'standard'
        }
        
        # AI analysis requirements
        base_ai_hours = 2
        if features['surveillance_coverage'] > 0.5:
            base_ai_hours += 4
        if features['sightings_count'] > 5:
            base_ai_hours += 2
        requirements['ai_analysis_hours'] = base_ai_hours
        
        # Manual review requirements
        manual_hours = 1
        if features['complexity'] in ['high', 'critical']:
            manual_hours += 3
        if features['case_type'] == 'criminal_investigation':
            manual_hours += 2
        requirements['manual_review_hours'] = manual_hours
        
        # Surveillance footage needs
        if features['surveillance_coverage'] < 0.3:
            requirements['surveillance_footage_needed'] = 5
        elif features['surveillance_coverage'] < 0.6:
            requirements['surveillance_footage_needed'] = 2
        
        # Admin involvement
        if features['case_type'] == 'criminal_investigation' or features['complexity'] == 'critical':
            requirements['admin_involvement_level'] = 'high'
        elif features['complexity'] == 'high':
            requirements['admin_involvement_level'] = 'medium'
        
        # Cost category
        total_effort = requirements['ai_analysis_hours'] + requirements['manual_review_hours']
        if total_effort > 8:
            requirements['estimated_cost_category'] = 'premium'
        elif total_effort > 4:
            requirements['estimated_cost_category'] = 'enhanced'
        
        return requirements
    
    def _identify_factors(self, case, features: Dict) -> Tuple[List[str], List[str]]:
        """Identify success and risk factors for the case"""
        success_factors = []
        risk_factors = []
        
        # Check critical success factors
        if features['photo_quality_score'] > 0.8:
            success_factors.append('High quality reference photos')
        if features['days_missing'] < 7:
            success_factors.append('Recent timeline - higher success probability')
        if features['location_specificity'] > 0.7:
            success_factors.append('Specific location information available')
        if features['verified_sightings'] > 0:
            success_factors.append('Verified sightings provide strong leads')
        if features['surveillance_coverage'] > 0.5:
            success_factors.append('Good surveillance coverage in area')
        
        # Check supporting factors
        if features['information_completeness'] > 0.8:
            success_factors.append('Complete case information provided')
        if features['requester_type'] in ['police', 'government']:
            success_factors.append('Official investigation support')
        if features['sightings_count'] > 2:
            success_factors.append('Multiple sightings increase success chance')
        
        # Check risk factors
        if features['photo_quality_score'] < 0.4:
            risk_factors.append('Poor photo quality may limit AI analysis')
        if features['days_missing'] > 30:
            risk_factors.append('Extended timeline reduces success probability')
        if features['location_specificity'] < 0.3:
            risk_factors.append('Vague location information')
        if features['surveillance_coverage'] < 0.2:
            risk_factors.append('Limited surveillance coverage in area')
        if features['information_completeness'] < 0.5:
            risk_factors.append('Incomplete case information')
        
        return success_factors, risk_factors
    
    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate confidence level in prediction"""
        confidence_factors = []
        
        # Data quality factors
        if features['has_photos']:
            confidence_factors.append(0.2)
        if features['photo_quality_score'] > 0.6:
            confidence_factors.append(0.15)
        if features['information_completeness'] > 0.7:
            confidence_factors.append(0.15)
        if features['has_location']:
            confidence_factors.append(0.1)
        
        # Evidence factors
        if features['sightings_count'] > 0:
            confidence_factors.append(0.1)
        if features['verified_sightings'] > 0:
            confidence_factors.append(0.15)
        if features['surveillance_coverage'] > 0.3:
            confidence_factors.append(0.1)
        
        # Time factor
        if features['days_missing'] < 30:
            confidence_factors.append(0.05)
        
        return min(0.95, sum(confidence_factors))
    
    def _pattern_matches(self, pattern: str, features: Dict) -> bool:
        """Check if case features match a historical pattern"""
        pattern_checks = {
            'high_quality_photos': features['photo_quality_score'] > 0.8,
            'multiple_sightings': features['sightings_count'] > 2,
            'recent_disappearance': features['days_missing'] < 7,
            'urban_location': features['location_specificity'] > 0.6,
            'family_cooperation': features['requester_type'] == 'family',
            'poor_photo_quality': features['photo_quality_score'] < 0.4,
            'rural_location': features['location_specificity'] < 0.4,
            'old_case': features['days_missing'] > 60,
            'limited_information': features['information_completeness'] < 0.5
        }
        
        return pattern_checks.get(pattern, False)
    
    def _find_similar_cases(self, case, features: Dict) -> List[SimilarCase]:
        """Find similar historical cases for comparison"""
        try:
            from models import Case
            from __init__ import db
            
            # Get completed cases of same type
            similar_cases = Case.query.filter(
                Case.case_type == case.case_type,
                Case.status.in_(['Case Solved', 'Case Over']),
                Case.id != case.id
            ).limit(10).all()
            
            results = []
            for similar_case in similar_cases:
                similarity_score = self._calculate_similarity(features, similar_case)
                if similarity_score > 0.5:
                    timeline_days = 30  # Default
                    if similar_case.completed_at and similar_case.created_at:
                        timeline_days = (similar_case.completed_at - similar_case.created_at).days
                    
                    results.append(SimilarCase(
                        case_id=similar_case.id,
                        similarity_score=similarity_score,
                        outcome=similar_case.investigation_outcome or 'Resolved',
                        timeline_days=timeline_days,
                        success_factors=['similar_case_pattern']
                    ))
            
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)[:5]
            
        except Exception as e:
            self.logger.error(f"Error finding similar cases: {e}")
            return []
    
    def _calculate_similarity(self, features: Dict, other_case) -> float:
        """Calculate similarity score between cases"""
        similarity_score = 0.0
        
        # Case type match
        if other_case.case_type == features['case_type']:
            similarity_score += 0.3
        
        # Priority match
        if other_case.priority == features['priority']:
            similarity_score += 0.1
        
        # Photo availability match
        other_has_photos = len(other_case.target_images) > 0
        if other_has_photos == features['has_photos']:
            similarity_score += 0.2
        
        # Location availability match
        other_has_location = bool(other_case.last_seen_location)
        if other_has_location == features['has_location']:
            similarity_score += 0.1
        
        # Requester type match
        if other_case.requester_type == features['requester_type']:
            similarity_score += 0.1
        
        # Age similarity (if available)
        if other_case.age and features.get('age'):
            age_diff = abs(other_case.age - features['age'])
            if age_diff < 10:
                similarity_score += 0.1
        
        # Sightings similarity
        other_sightings = len(other_case.sightings)
        if abs(other_sightings - features['sightings_count']) < 3:
            similarity_score += 0.1
        
        return similarity_score
    
    def _store_prediction(self, result: PredictionResult, similar_cases: List[SimilarCase]):
        """Store prediction results in database"""
        try:
            conn = sqlite3.connect('instance/outcome_prediction.db')
            cursor = conn.cursor()
            
            # Store main prediction
            cursor.execute('''
                INSERT INTO outcome_predictions 
                (case_id, success_probability, resolution_likelihood, estimated_timeline,
                 resource_requirements, success_factors, risk_factors, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.case_id,
                result.success_probability,
                result.resolution_likelihood,
                result.estimated_timeline,
                json.dumps(result.resource_requirements),
                json.dumps(result.success_factors),
                json.dumps(result.risk_factors),
                result.confidence_level
            ))
            
            # Store similar cases
            for similar_case in similar_cases:
                cursor.execute('''
                    INSERT INTO similar_cases_analysis
                    (case_id, similar_case_id, similarity_score, outcome, timeline_days, factors_matched)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.case_id,
                    similar_case.case_id,
                    similar_case.similarity_score,
                    similar_case.outcome,
                    similar_case.timeline_days,
                    json.dumps(similar_case.success_factors)
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Prediction stored for case {result.case_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
    
    def get_prediction_summary(self, case_id: int) -> Dict:
        """Get prediction summary for admin dashboard"""
        try:
            conn = sqlite3.connect('instance/outcome_prediction.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT success_probability, resolution_likelihood, estimated_timeline,
                       resource_requirements, success_factors, risk_factors, confidence_level,
                       prediction_date
                FROM outcome_predictions 
                WHERE case_id = ? 
                ORDER BY prediction_date DESC 
                LIMIT 1
            ''', (case_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'success_probability': result[0],
                    'resolution_likelihood': result[1],
                    'estimated_timeline': result[2],
                    'resource_requirements': json.loads(result[3]) if result[3] else {},
                    'success_factors': json.loads(result[4]) if result[4] else [],
                    'risk_factors': json.loads(result[5]) if result[5] else [],
                    'confidence_level': result[6],
                    'prediction_date': result[7]
                }
            
            conn.close()
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting prediction summary: {e}")
            return {}

# Global instance
outcome_predictor = OutcomePredictionEngine()

def predict_case_outcome(case_id: int) -> Dict:
    """Predict case outcome and return summary"""
    try:
        result = outcome_predictor.predict_case_outcome(case_id)
        return {
            'success': True,
            'prediction': {
                'success_probability': result.success_probability,
                'resolution_likelihood': result.resolution_likelihood,
                'estimated_timeline': result.estimated_timeline,
                'resource_requirements': result.resource_requirements,
                'success_factors': result.success_factors,
                'risk_factors': result.risk_factors,
                'confidence_level': result.confidence_level
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_prediction_summary(case_id: int) -> Dict:
    """Get existing prediction summary"""
    return outcome_predictor.get_prediction_summary(case_id)