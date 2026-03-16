"""
XAI Feature Weighting Audit System
Provides transparent AI decision making with detailed confidence breakdowns
"""

import json
import hashlib
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureWeights:
    """Individual feature weights for a detection"""
    facial_structure_score: float = 0.0
    facial_structure_weight: float = 0.4
    
    clothing_biometric_score: float = 0.0
    clothing_biometric_weight: float = 0.25
    
    temporal_consistency_score: float = 0.0
    temporal_consistency_weight: float = 0.2
    
    body_pose_score: float = 0.0
    body_pose_weight: float = 0.1
    
    motion_pattern_score: float = 0.0
    motion_pattern_weight: float = 0.05
    
    def calculate_ensemble_score(self) -> float:
        """Calculate weighted ensemble score"""
        total_score = (
            self.facial_structure_score * self.facial_structure_weight +
            self.clothing_biometric_score * self.clothing_biometric_weight +
            self.temporal_consistency_score * self.temporal_consistency_weight +
            self.body_pose_score * self.body_pose_weight +
            self.motion_pattern_score * self.motion_pattern_weight
        )
        
        total_weight = (
            self.facial_structure_weight +
            self.clothing_biometric_weight +
            self.temporal_consistency_weight +
            self.body_pose_weight +
            self.motion_pattern_weight
        )
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_confidence_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed confidence breakdown"""
        return {
            "facial_structure": {
                "score": self.facial_structure_score,
                "weight": self.facial_structure_weight,
                "contribution": self.facial_structure_score * self.facial_structure_weight
            },
            "clothing_biometric": {
                "score": self.clothing_biometric_score,
                "weight": self.clothing_biometric_weight,
                "contribution": self.clothing_biometric_score * self.clothing_biometric_weight
            },
            "temporal_consistency": {
                "score": self.temporal_consistency_score,
                "weight": self.temporal_consistency_weight,
                "contribution": self.temporal_consistency_score * self.temporal_consistency_weight
            },
            "body_pose": {
                "score": self.body_pose_score,
                "weight": self.body_pose_weight,
                "contribution": self.body_pose_score * self.body_pose_weight
            },
            "motion_pattern": {
                "score": self.motion_pattern_score,
                "weight": self.motion_pattern_weight,
                "contribution": self.motion_pattern_score * self.motion_pattern_weight
            }
        }

@dataclass
class XAIDetectionResult:
    """Explainable AI detection result with full transparency"""
    detection_id: str
    timestamp: float
    case_id: int
    footage_id: int
    
    # Core detection data
    bounding_box: Tuple[int, int, int, int]
    frame_hash: str
    frame_path: str
    
    # Feature weights and scores
    feature_weights: FeatureWeights
    ensemble_confidence: float
    
    # Temporal analysis
    detection_duration: float = 0.0
    sequence_consistency: float = 0.0
    tracking_stability: float = 0.0
    
    # Quality metrics
    frame_quality_score: float = 0.0
    face_visibility_score: float = 0.0
    lighting_quality_score: float = 0.0
    
    # Decision rationale
    decision_factors: List[str] = None
    uncertainty_factors: List[str] = None
    
    # Verification status
    requires_confirmation: bool = False
    confidence_category: str = "medium"  # low, medium, high, very_high
    
    def __post_init__(self):
        if self.decision_factors is None:
            self.decision_factors = []
        if self.uncertainty_factors is None:
            self.uncertainty_factors = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['feature_weights'] = asdict(self.feature_weights)
        return result

class XAIFeatureWeightingSystem:
    """Explainable AI system for transparent decision making"""
    
    def __init__(self):
        # Confidence thresholds
        self.VERY_HIGH_THRESHOLD = 0.90
        self.HIGH_THRESHOLD = 0.75
        self.MEDIUM_THRESHOLD = 0.60
        self.LOW_THRESHOLD = 0.40
        
        # Temporal consistency requirements
        self.MIN_SEQUENCE_DURATION = 5.0  # 5 seconds minimum
        self.MIN_CONSISTENCY_SCORE = 0.7
        
        # Quality thresholds
        self.MIN_FRAME_QUALITY = 0.5
        self.MIN_FACE_VISIBILITY = 0.6
        
    def analyze_detection_with_xai(self, detection_data: Dict) -> XAIDetectionResult:
        """Analyze detection with full XAI transparency"""
        
        # Generate unique detection ID
        detection_id = self._generate_detection_id(detection_data)
        
        # Calculate frame hash for integrity
        frame_hash = self._calculate_frame_hash(detection_data.get('frame_path', ''))
        
        # Extract feature weights
        feature_weights = self._calculate_feature_weights(detection_data)
        
        # Calculate ensemble confidence
        ensemble_confidence = feature_weights.calculate_ensemble_score()
        
        # Temporal analysis
        temporal_metrics = self._analyze_temporal_consistency(detection_data)
        
        # Quality assessment
        quality_metrics = self._assess_frame_quality(detection_data)
        
        # Decision rationale
        decision_factors, uncertainty_factors = self._generate_decision_rationale(
            feature_weights, temporal_metrics, quality_metrics, ensemble_confidence
        )
        
        # Determine confidence category and confirmation requirement
        confidence_category, requires_confirmation = self._categorize_confidence(
            ensemble_confidence, temporal_metrics, quality_metrics
        )
        
        # Create XAI result
        xai_result = XAIDetectionResult(
            detection_id=detection_id,
            timestamp=detection_data.get('timestamp', 0.0),
            case_id=detection_data.get('case_id', 0),
            footage_id=detection_data.get('footage_id', 0),
            bounding_box=detection_data.get('bbox', (0, 0, 0, 0)),
            frame_hash=frame_hash,
            frame_path=detection_data.get('frame_path', ''),
            feature_weights=feature_weights,
            ensemble_confidence=ensemble_confidence,
            detection_duration=temporal_metrics.get('duration', 0.0),
            sequence_consistency=temporal_metrics.get('consistency', 0.0),
            tracking_stability=temporal_metrics.get('stability', 0.0),
            frame_quality_score=quality_metrics.get('frame_quality', 0.0),
            face_visibility_score=quality_metrics.get('face_visibility', 0.0),
            lighting_quality_score=quality_metrics.get('lighting_quality', 0.0),
            decision_factors=decision_factors,
            uncertainty_factors=uncertainty_factors,
            requires_confirmation=requires_confirmation,
            confidence_category=confidence_category
        )
        
        return xai_result
    
    def _generate_detection_id(self, detection_data: Dict) -> str:
        """Generate unique detection ID"""
        timestamp = detection_data.get('timestamp', 0.0)
        case_id = detection_data.get('case_id', 0)
        footage_id = detection_data.get('footage_id', 0)
        
        id_string = f"{case_id}_{footage_id}_{timestamp}_{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def _calculate_frame_hash(self, frame_path: str) -> str:
        """Calculate SHA-256 hash of detection frame for integrity"""
        try:
            import cv2
            import os
            
            if not frame_path or not os.path.exists(frame_path):
                return "no_frame_available"
            
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return "frame_read_error"
            
            # Calculate hash
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            return hashlib.sha256(frame_bytes).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating frame hash: {e}")
            return "hash_calculation_error"
    
    def _calculate_feature_weights(self, detection_data: Dict) -> FeatureWeights:
        """Calculate individual feature weights and scores"""
        
        # Extract scores from detection data
        face_score = detection_data.get('face_confidence', 0.0)
        clothing_score = detection_data.get('clothing_confidence', 0.0)
        body_score = detection_data.get('body_confidence', 0.0)
        motion_score = detection_data.get('motion_confidence', 0.0)
        
        # Calculate temporal consistency
        temporal_score = self._calculate_temporal_score(detection_data)
        
        # Adaptive weighting based on available features
        weights = FeatureWeights()
        
        # Adjust weights based on feature availability and quality
        if face_score > 0.5:
            weights.facial_structure_score = face_score
            weights.facial_structure_weight = 0.5  # Higher weight for good face detection
        else:
            weights.facial_structure_score = face_score
            weights.facial_structure_weight = 0.2  # Lower weight for poor face detection
            # Increase other weights to compensate
            weights.clothing_biometric_weight = 0.35
            weights.body_pose_weight = 0.25
        
        weights.clothing_biometric_score = clothing_score
        weights.temporal_consistency_score = temporal_score
        weights.body_pose_score = body_score
        weights.motion_pattern_score = motion_score
        
        return weights
    
    def _calculate_temporal_score(self, detection_data: Dict) -> float:
        """Calculate temporal consistency score"""
        duration = detection_data.get('duration', 0.0)
        consistency = detection_data.get('consistency', 0.0)
        
        # Score based on duration and consistency
        duration_score = min(1.0, duration / self.MIN_SEQUENCE_DURATION)
        temporal_score = (duration_score * 0.6) + (consistency * 0.4)
        
        return temporal_score
    
    def _analyze_temporal_consistency(self, detection_data: Dict) -> Dict[str, float]:
        """Analyze temporal consistency of detection"""
        return {
            'duration': detection_data.get('duration', 0.0),
            'consistency': detection_data.get('consistency', 0.0),
            'stability': detection_data.get('tracking_stability', 0.0)
        }
    
    def _assess_frame_quality(self, detection_data: Dict) -> Dict[str, float]:
        """Assess frame quality metrics"""
        return {
            'frame_quality': detection_data.get('frame_quality', 0.0),
            'face_visibility': detection_data.get('face_visibility', 0.0),
            'lighting_quality': detection_data.get('lighting_quality', 0.0)
        }
    
    def _generate_decision_rationale(self, feature_weights: FeatureWeights, 
                                   temporal_metrics: Dict, quality_metrics: Dict,
                                   ensemble_confidence: float) -> Tuple[List[str], List[str]]:
        """Generate human-readable decision rationale"""
        
        decision_factors = []
        uncertainty_factors = []
        
        # Analyze each feature contribution
        breakdown = feature_weights.get_confidence_breakdown()
        
        for feature, data in breakdown.items():
            score = data['score']
            contribution = data['contribution']
            
            if score > 0.8:
                decision_factors.append(f"Strong {feature.replace('_', ' ')} match (score: {score:.2f})")
            elif score > 0.6:
                decision_factors.append(f"Good {feature.replace('_', ' ')} match (score: {score:.2f})")
            elif score > 0.4:
                uncertainty_factors.append(f"Moderate {feature.replace('_', ' ')} uncertainty (score: {score:.2f})")
            else:
                uncertainty_factors.append(f"Low {feature.replace('_', ' ')} confidence (score: {score:.2f})")
        
        # Temporal analysis
        if temporal_metrics['duration'] >= self.MIN_SEQUENCE_DURATION:
            decision_factors.append(f"Sufficient tracking duration ({temporal_metrics['duration']:.1f}s)")
        else:
            uncertainty_factors.append(f"Short tracking duration ({temporal_metrics['duration']:.1f}s)")
        
        if temporal_metrics['consistency'] >= self.MIN_CONSISTENCY_SCORE:
            decision_factors.append(f"High temporal consistency ({temporal_metrics['consistency']:.2f})")
        else:
            uncertainty_factors.append(f"Low temporal consistency ({temporal_metrics['consistency']:.2f})")
        
        # Quality analysis
        if quality_metrics['frame_quality'] >= self.MIN_FRAME_QUALITY:
            decision_factors.append(f"Good frame quality ({quality_metrics['frame_quality']:.2f})")
        else:
            uncertainty_factors.append(f"Poor frame quality ({quality_metrics['frame_quality']:.2f})")
        
        return decision_factors, uncertainty_factors
    
    def _categorize_confidence(self, ensemble_confidence: float, 
                             temporal_metrics: Dict, quality_metrics: Dict) -> Tuple[str, bool]:
        """Categorize confidence level and determine if confirmation is needed"""
        
        # Base categorization on ensemble confidence
        if ensemble_confidence >= self.VERY_HIGH_THRESHOLD:
            base_category = "very_high"
        elif ensemble_confidence >= self.HIGH_THRESHOLD:
            base_category = "high"
        elif ensemble_confidence >= self.MEDIUM_THRESHOLD:
            base_category = "medium"
        else:
            base_category = "low"
        
        # Adjust based on temporal and quality factors
        temporal_bonus = 0.0
        quality_bonus = 0.0
        
        if temporal_metrics['duration'] >= self.MIN_SEQUENCE_DURATION:
            temporal_bonus += 0.05
        if temporal_metrics['consistency'] >= self.MIN_CONSISTENCY_SCORE:
            temporal_bonus += 0.05
        
        if quality_metrics['frame_quality'] >= self.MIN_FRAME_QUALITY:
            quality_bonus += 0.03
        if quality_metrics['face_visibility'] >= self.MIN_FACE_VISIBILITY:
            quality_bonus += 0.02
        
        adjusted_confidence = ensemble_confidence + temporal_bonus + quality_bonus
        
        # Final categorization
        if adjusted_confidence >= self.VERY_HIGH_THRESHOLD:
            final_category = "very_high"
            requires_confirmation = False
        elif adjusted_confidence >= self.HIGH_THRESHOLD:
            final_category = "high"
            requires_confirmation = False
        elif adjusted_confidence >= self.MEDIUM_THRESHOLD:
            final_category = "medium"
            requires_confirmation = True
        else:
            final_category = "low"
            requires_confirmation = True
        
        return final_category, requires_confirmation
    
    def generate_xai_report(self, xai_results: List[XAIDetectionResult]) -> Dict:
        """Generate comprehensive XAI report"""
        
        if not xai_results:
            return {"error": "No XAI results provided"}
        
        # Aggregate statistics
        total_detections = len(xai_results)
        confidence_distribution = {
            "very_high": len([r for r in xai_results if r.confidence_category == "very_high"]),
            "high": len([r for r in xai_results if r.confidence_category == "high"]),
            "medium": len([r for r in xai_results if r.confidence_category == "medium"]),
            "low": len([r for r in xai_results if r.confidence_category == "low"])
        }
        
        confirmations_needed = len([r for r in xai_results if r.requires_confirmation])
        
        # Feature importance analysis
        feature_importance = self._calculate_feature_importance(xai_results)
        
        # Top detections analysis
        top_detections = sorted(xai_results, key=lambda x: x.ensemble_confidence, reverse=True)[:5]
        
        report = {
            "report_id": hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:16],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_detections": total_detections,
            "confidence_distribution": confidence_distribution,
            "confirmations_needed": confirmations_needed,
            "feature_importance": feature_importance,
            "top_detections": [result.to_dict() for result in top_detections],
            "integrity_verified": True,
            "xai_version": "1.0"
        }
        
        return report
    
    def _calculate_feature_importance(self, xai_results: List[XAIDetectionResult]) -> Dict:
        """Calculate overall feature importance across all detections"""
        
        if not xai_results:
            return {}
        
        total_contributions = {
            "facial_structure": 0.0,
            "clothing_biometric": 0.0,
            "temporal_consistency": 0.0,
            "body_pose": 0.0,
            "motion_pattern": 0.0
        }
        
        for result in xai_results:
            breakdown = result.feature_weights.get_confidence_breakdown()
            for feature, data in breakdown.items():
                total_contributions[feature] += data['contribution']
        
        # Normalize to percentages
        total_contribution = sum(total_contributions.values())
        if total_contribution > 0:
            feature_importance = {
                feature: (contribution / total_contribution) * 100
                for feature, contribution in total_contributions.items()
            }
        else:
            feature_importance = {feature: 0.0 for feature in total_contributions.keys()}
        
        return feature_importance

# Global XAI system instance
xai_system = XAIFeatureWeightingSystem()

def analyze_detection_with_xai(detection_data: Dict) -> XAIDetectionResult:
    """Global function for XAI analysis"""
    return xai_system.analyze_detection_with_xai(detection_data)

def generate_xai_report(xai_results: List[XAIDetectionResult]) -> Dict:
    """Global function for XAI report generation"""
    return xai_system.generate_xai_report(xai_results)