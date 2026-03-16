"""
Learning Integration Module
Integrates continuous learning system with existing AI components
"""

from continuous_learning_system import continuous_learning_system
from datetime import datetime
import json
from typing import Dict, Any, Optional

class LearningIntegration:
    def __init__(self):
        self.learning_system = continuous_learning_system
    
    def integrate_with_ai_validator(self, ai_validator):
        """Integrate learning system with AI validator"""
        original_validate_case = ai_validator.validate_case
        
        def enhanced_validate_case(case):
            # Get adaptive thresholds
            ai_validator.validation_threshold = self.learning_system.get_adaptive_threshold('case_validation')
            
            # Apply learned patterns
            case_data = self._extract_case_features(case)
            pattern_recommendations = self.learning_system.apply_learned_patterns(case_data, 'case_validation')
            
            # Run original validation
            decision, confidence, scores, reasons, smart_feedback = original_validate_case(case)
            
            # Apply pattern-based confidence boost
            if pattern_recommendations['confidence_boost'] > 0:
                confidence += pattern_recommendations['confidence_boost']
                confidence = min(confidence, 1.0)
            
            # Store prediction for learning
            self._store_validation_prediction(case, decision, confidence, scores, pattern_recommendations)
            
            return decision, confidence, scores, reasons, smart_feedback
        
        ai_validator.validate_case = enhanced_validate_case
        return ai_validator
    
    def integrate_with_cctv_matcher(self, cctv_matcher):
        """Integrate learning system with CCTV matcher"""
        if not hasattr(cctv_matcher, 'match_person_in_footage'):
            return cctv_matcher
        
        original_match_person = cctv_matcher.match_person_in_footage
        
        def enhanced_match_person(case, footage_path, confidence_threshold=None):
            # Use adaptive threshold if not specified
            if confidence_threshold is None:
                confidence_threshold = self.learning_system.get_adaptive_threshold('cctv_matching')
            
            # Apply learned patterns
            case_data = self._extract_case_features(case)
            pattern_recommendations = self.learning_system.apply_learned_patterns(case_data, 'cctv_matching')
            
            # Run original matching
            results = original_match_person(case, footage_path, confidence_threshold)
            
            # Apply pattern-based adjustments
            if pattern_recommendations['confidence_boost'] > 0:
                for result in results:
                    if 'confidence' in result:
                        result['confidence'] += pattern_recommendations['confidence_boost']
                        result['confidence'] = min(result['confidence'], 1.0)
            
            # Store prediction for learning
            self._store_cctv_prediction(case, footage_path, results, pattern_recommendations)
            
            return results
        
        cctv_matcher.match_person_in_footage = enhanced_match_person
        return cctv_matcher
    
    def integrate_with_case_quality_assessment(self, quality_assessor):
        """Integrate learning system with case quality assessment"""
        if not hasattr(quality_assessor, 'assess_case_quality'):
            return quality_assessor
        
        original_assess_quality = quality_assessor.assess_case_quality
        
        def enhanced_assess_quality(case):
            # Get adaptive thresholds
            photo_threshold = self.learning_system.get_adaptive_threshold('photo_quality')
            form_threshold = self.learning_system.get_adaptive_threshold('form_completeness')
            
            # Apply learned patterns
            case_data = self._extract_case_features(case)
            pattern_recommendations = self.learning_system.apply_learned_patterns(case_data, 'quality_assessment')
            
            # Run original assessment
            assessment = original_assess_quality(case)
            
            # Apply adaptive thresholds
            if 'photo_quality_score' in assessment:
                assessment['photo_quality_threshold'] = photo_threshold
            if 'form_completeness_score' in assessment:
                assessment['form_completeness_threshold'] = form_threshold
            
            # Apply pattern-based adjustments
            if pattern_recommendations['confidence_boost'] > 0:
                assessment['overall_score'] = min(
                    assessment.get('overall_score', 0) + pattern_recommendations['confidence_boost'],
                    1.0
                )
            
            # Store prediction for learning
            self._store_quality_prediction(case, assessment, pattern_recommendations)
            
            return assessment
        
        quality_assessor.assess_case_quality = enhanced_assess_quality
        return quality_assessor
    
    def integrate_with_case_approval_engine(self, approval_engine):
        """Integrate learning system with case approval engine"""
        if not hasattr(approval_engine, 'evaluate_case_for_approval'):
            return approval_engine
        
        original_evaluate_case = approval_engine.evaluate_case_for_approval
        
        def enhanced_evaluate_case(case):
            # Get adaptive threshold
            approval_threshold = self.learning_system.get_adaptive_threshold('case_approval')
            
            # Apply learned patterns
            case_data = self._extract_case_features(case)
            pattern_recommendations = self.learning_system.apply_learned_patterns(case_data, 'case_approval')
            
            # Run original evaluation
            result = original_evaluate_case(case)
            
            # Apply adaptive threshold
            if 'confidence_score' in result:
                # Adjust decision based on adaptive threshold
                if result['confidence_score'] >= approval_threshold:
                    result['decision'] = 'APPROVE'
                else:
                    result['decision'] = 'REJECT'
            
            # Apply pattern-based confidence boost
            if pattern_recommendations['confidence_boost'] > 0:
                result['confidence_score'] = min(
                    result.get('confidence_score', 0) + pattern_recommendations['confidence_boost'],
                    1.0
                )
            
            # Store prediction for learning
            self._store_approval_prediction(case, result, pattern_recommendations)
            
            return result
        
        approval_engine.evaluate_case_for_approval = enhanced_evaluate_case
        return approval_engine
    
    def _extract_case_features(self, case) -> Dict[str, Any]:
        """Extract features from case for pattern matching"""
        features = {}
        
        # Basic case features
        if hasattr(case, 'age') and case.age:
            if case.age < 13:
                features['age_group'] = 'child'
            elif case.age < 20:
                features['age_group'] = 'teenager'
            elif case.age < 35:
                features['age_group'] = 'young_adult'
            elif case.age < 60:
                features['age_group'] = 'adult'
            else:
                features['age_group'] = 'elderly'
        
        # Location type
        if hasattr(case, 'last_seen_location') and case.last_seen_location:
            location = case.last_seen_location.lower()
            if any(word in location for word in ['market', 'mall', 'shop']):
                features['location_type'] = 'commercial'
            elif any(word in location for word in ['school', 'college', 'university']):
                features['location_type'] = 'educational'
            elif any(word in location for word in ['park', 'garden', 'playground']):
                features['location_type'] = 'recreational'
            elif any(word in location for word in ['station', 'bus', 'train', 'airport']):
                features['location_type'] = 'transport'
            else:
                features['location_type'] = 'residential'
        
        # Case urgency based on missing duration
        if hasattr(case, 'date_missing') and case.date_missing:
            try:
                # Handle both date and datetime objects
                if hasattr(case.date_missing, 'date'):
                    # It's a datetime object
                    missing_date = case.date_missing.date()
                else:
                    # It's already a date object
                    missing_date = case.date_missing
                
                days_missing = (datetime.now().date() - missing_date).days
                
                if days_missing <= 1:
                    features['case_urgency'] = 'critical'
                elif days_missing <= 7:
                    features['case_urgency'] = 'high'
                elif days_missing <= 30:
                    features['case_urgency'] = 'medium'
                else:
                    features['case_urgency'] = 'low'
            except Exception as e:
                # Fallback if date calculation fails
                features['case_urgency'] = 'medium'
        
        # Photo count
        if hasattr(case, 'target_images'):
            features['photo_count'] = len(case.target_images) if case.target_images else 0
        
        # Video availability
        if hasattr(case, 'search_videos'):
            features['has_videos'] = bool(case.search_videos)
        
        # Details length (information richness)
        if hasattr(case, 'details') and case.details:
            details_length = len(case.details.strip())
            if details_length < 100:
                features['details_richness'] = 'low'
            elif details_length < 300:
                features['details_richness'] = 'medium'
            else:
                features['details_richness'] = 'high'
        
        return features
    
    def _store_validation_prediction(self, case, decision, confidence, scores, pattern_recommendations):
        """Store validation prediction for future learning"""
        context_data = self._extract_case_features(case)
        context_data.update({
            'scores': scores,
            'pattern_matches': len(pattern_recommendations.get('pattern_matches', [])),
            'confidence_boost': pattern_recommendations.get('confidence_boost', 0)
        })
        
        # Store prediction (actual outcome will be recorded later via feedback)
        predicted_value = 1.0 if decision == 'APPROVE' else 0.0
        
        # For now, we store the prediction without outcome (will be updated via feedback)
        # This is a placeholder - actual outcome recording happens through admin feedback
        pass
    
    def _store_cctv_prediction(self, case, footage_path, results, pattern_recommendations):
        """Store CCTV matching prediction for future learning"""
        context_data = self._extract_case_features(case)
        context_data.update({
            'footage_path': footage_path,
            'matches_found': len(results),
            'pattern_matches': len(pattern_recommendations.get('pattern_matches', [])),
            'confidence_boost': pattern_recommendations.get('confidence_boost', 0)
        })
        
        # Store highest confidence match as prediction
        if results:
            max_confidence = max(result.get('confidence', 0) for result in results)
            # Prediction storage logic here
        
    def _store_quality_prediction(self, case, assessment, pattern_recommendations):
        """Store quality assessment prediction for future learning"""
        context_data = self._extract_case_features(case)
        context_data.update({
            'assessment': assessment,
            'pattern_matches': len(pattern_recommendations.get('pattern_matches', [])),
            'confidence_boost': pattern_recommendations.get('confidence_boost', 0)
        })
        
        # Store overall quality score as prediction
        overall_score = assessment.get('overall_score', 0)
        # Prediction storage logic here
    
    def _store_approval_prediction(self, case, result, pattern_recommendations):
        """Store approval prediction for future learning"""
        context_data = self._extract_case_features(case)
        context_data.update({
            'approval_result': result,
            'pattern_matches': len(pattern_recommendations.get('pattern_matches', [])),
            'confidence_boost': pattern_recommendations.get('confidence_boost', 0)
        })
        
        # Store approval decision as prediction
        predicted_value = 1.0 if result.get('decision') == 'APPROVE' else 0.0
        # Prediction storage logic here
    
    def record_admin_feedback(self, case_id: int, feedback_type: str, actual_outcome: str, 
                            admin_notes: str = None):
        """Record admin feedback for learning"""
        context_data = {'admin_notes': admin_notes} if admin_notes else {}
        
        # This would typically get the original prediction from storage
        # For now, we'll record the feedback directly
        self.learning_system.record_feedback(
            case_id=case_id,
            prediction_type=feedback_type,
            predicted_value=0.5,  # Placeholder - should be actual prediction
            actual_outcome=actual_outcome,
            confidence_score=0.5,  # Placeholder - should be actual confidence
            feedback_source='admin',
            context_data=context_data
        )
    
    def record_user_feedback(self, case_id: int, feedback_type: str, user_satisfaction: str,
                           user_comments: str = None):
        """Record user feedback for learning"""
        # Convert user satisfaction to learning outcome
        outcome_mapping = {
            'very_satisfied': 'correct',
            'satisfied': 'correct',
            'neutral': 'partial_correct',
            'dissatisfied': 'false_positive',
            'very_dissatisfied': 'false_positive'
        }
        
        actual_outcome = outcome_mapping.get(user_satisfaction, 'unknown')
        
        context_data = {
            'user_satisfaction': user_satisfaction,
            'user_comments': user_comments
        }
        
        self.learning_system.record_feedback(
            case_id=case_id,
            prediction_type=feedback_type,
            predicted_value=0.5,  # Placeholder
            actual_outcome=actual_outcome,
            confidence_score=0.5,  # Placeholder
            feedback_source='user',
            context_data=context_data
        )
    
    def record_system_feedback(self, case_id: int, prediction_type: str, predicted_value: float,
                             confidence_score: float, actual_outcome: str, context_data: Dict = None):
        """Record system-generated feedback for learning"""
        self.learning_system.record_feedback(
            case_id=case_id,
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            actual_outcome=actual_outcome,
            confidence_score=confidence_score,
            feedback_source='system',
            context_data=context_data or {}
        )
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning system insights for admin dashboard"""
        stats = self.learning_system.get_learning_stats()
        
        insights = {
            'system_performance': {
                'overall_accuracy': stats['system_accuracy'],
                'total_feedback_samples': stats['total_feedback'],
                'learning_active': stats['total_feedback'] > 50
            },
            'adaptive_thresholds': stats['thresholds'],
            'component_performance': {},
            'recent_improvements': [],
            'recommendations': []
        }
        
        # Analyze component performance
        for metric in stats['recent_metrics']:
            component = metric['component']
            if component not in insights['component_performance']:
                insights['component_performance'][component] = {}
            insights['component_performance'][component][metric['metric']] = metric['value']
        
        # Generate recommendations
        if stats['system_accuracy'] < 0.8:
            insights['recommendations'].append({
                'type': 'accuracy_improvement',
                'message': 'System accuracy below 80%. Consider reviewing recent false positives.',
                'priority': 'high'
            })
        
        for threshold_info in stats['thresholds']:
            if threshold_info['adjustments'] > 10:
                insights['recommendations'].append({
                    'type': 'threshold_stability',
                    'message': f"{threshold_info['component']} threshold frequently adjusted. Review feedback quality.",
                    'priority': 'medium'
                })
        
        return insights

# Global learning integration instance
learning_integration = LearningIntegration()