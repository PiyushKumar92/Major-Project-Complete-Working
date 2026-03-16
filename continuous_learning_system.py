"""
Continuous Learning System - Self-Improving AI
Implements feedback loops, false positive reduction, adaptive thresholds, and cross-case pattern learning
"""

import json
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import os
from collections import defaultdict
import pickle
import logging

@dataclass
class FeedbackData:
    case_id: int
    prediction_type: str  # 'approval', 'rejection', 'cctv_match', 'quality_score'
    predicted_value: float
    actual_outcome: str  # 'correct', 'false_positive', 'false_negative'
    confidence_score: float
    timestamp: datetime
    feedback_source: str  # 'admin', 'user', 'system'
    context_data: Dict[str, Any]

@dataclass
class LearningMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    confidence_calibration: float

@dataclass
class AdaptiveThreshold:
    component: str  # 'photo_quality', 'form_validation', 'cctv_matching'
    current_threshold: float
    optimal_threshold: float
    performance_history: List[float]
    last_updated: datetime
    adjustment_count: int

class ContinuousLearningSystem:
    def __init__(self, db_path='instance/app.db'):
        self.db_path = db_path
        self.learning_data_path = 'learning_data'
        self.model_cache = {}
        self.thresholds = {}
        self.performance_history = defaultdict(list)
        self.pattern_cache = {}
        
        # Initialize learning database
        self._init_learning_db()
        self._load_thresholds()
        
        # Learning parameters
        self.min_feedback_samples = 10
        self.learning_rate = 0.1
        self.threshold_adjustment_sensitivity = 0.05
        self.pattern_similarity_threshold = 0.8
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_learning_db(self):
        """Initialize learning database tables"""
        os.makedirs(self.learning_data_path, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                prediction_type TEXT,
                predicted_value REAL,
                actual_outcome TEXT,
                confidence_score REAL,
                timestamp DATETIME,
                feedback_source TEXT,
                context_data TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT,
                metric_type TEXT,
                metric_value REAL,
                timestamp DATETIME,
                sample_size INTEGER
            )
        ''')
        
        # Adaptive thresholds table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_thresholds (
                component TEXT PRIMARY KEY,
                current_threshold REAL,
                optimal_threshold REAL,
                performance_history TEXT,
                last_updated DATETIME,
                adjustment_count INTEGER
            )
        ''')
        
        # Pattern learning table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                success_rate REAL,
                usage_count INTEGER,
                last_used DATETIME,
                created_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_feedback(self, case_id: int, prediction_type: str, predicted_value: float, 
                       actual_outcome: str, confidence_score: float, feedback_source: str = 'system',
                       context_data: Dict[str, Any] = None):
        """Record feedback for continuous learning"""
        feedback = FeedbackData(
            case_id=case_id,
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            actual_outcome=actual_outcome,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            feedback_source=feedback_source,
            context_data=context_data or {}
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_feedback 
            (case_id, prediction_type, predicted_value, actual_outcome, confidence_score, 
             timestamp, feedback_source, context_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.case_id, feedback.prediction_type, feedback.predicted_value,
            feedback.actual_outcome, feedback.confidence_score, feedback.timestamp,
            feedback.feedback_source, json.dumps(feedback.context_data)
        ))
        
        conn.commit()
        conn.close()
        
        # Trigger learning if enough samples
        self._trigger_learning_if_ready(prediction_type)
        
        self.logger.info(f"Recorded feedback for case {case_id}: {prediction_type} -> {actual_outcome}")
    
    def _trigger_learning_if_ready(self, prediction_type: str):
        """Trigger learning process if enough feedback samples available"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM learning_feedback 
            WHERE prediction_type = ? AND processed = FALSE
        ''', (prediction_type,))
        
        unprocessed_count = cursor.fetchone()[0]
        conn.close()
        
        if unprocessed_count >= self.min_feedback_samples:
            self.learn_from_feedback(prediction_type)
    
    def learn_from_feedback(self, prediction_type: str = None):
        """Main learning function - updates models and thresholds based on feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unprocessed feedback
        if prediction_type:
            cursor.execute('''
                SELECT * FROM learning_feedback 
                WHERE prediction_type = ? AND processed = FALSE
                ORDER BY timestamp DESC
            ''', (prediction_type,))
        else:
            cursor.execute('''
                SELECT * FROM learning_feedback 
                WHERE processed = FALSE
                ORDER BY timestamp DESC
            ''')
        
        feedback_data = cursor.fetchall()
        
        if not feedback_data:
            return
        
        # Group feedback by prediction type
        feedback_by_type = defaultdict(list)
        for row in feedback_data:
            feedback_by_type[row[2]].append(row)  # row[2] is prediction_type
        
        # Process each prediction type
        for pred_type, feedbacks in feedback_by_type.items():
            self._process_feedback_batch(pred_type, feedbacks)
        
        # Mark feedback as processed
        feedback_ids = [row[0] for row in feedback_data]
        if feedback_ids:
            placeholders = ','.join(['?' for _ in feedback_ids])
            cursor.execute(f'''
                UPDATE learning_feedback 
                SET processed = TRUE 
                WHERE id IN ({placeholders})
            ''', feedback_ids)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Processed {len(feedback_data)} feedback samples")
    
    def _process_feedback_batch(self, prediction_type: str, feedbacks: List[Tuple]):
        """Process a batch of feedback for specific prediction type"""
        # Calculate performance metrics
        correct_predictions = 0
        false_positives = 0
        false_negatives = 0
        total_predictions = len(feedbacks)
        
        predicted_values = []
        actual_outcomes = []
        confidence_scores = []
        
        for feedback in feedbacks:
            predicted_val = feedback[3]  # predicted_value
            actual_outcome = feedback[4]  # actual_outcome
            confidence = feedback[5]  # confidence_score
            
            predicted_values.append(predicted_val)
            confidence_scores.append(confidence)
            
            if actual_outcome == 'correct':
                correct_predictions += 1
                actual_outcomes.append(1)
            elif actual_outcome == 'false_positive':
                false_positives += 1
                actual_outcomes.append(0)
            elif actual_outcome == 'false_negative':
                false_negatives += 1
                actual_outcomes.append(1)
            else:
                actual_outcomes.append(0)
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        precision = correct_predictions / (correct_predictions + false_positives) if (correct_predictions + false_positives) > 0 else 0
        recall = correct_predictions / (correct_predictions + false_negatives) if (correct_predictions + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = false_positives / total_predictions if total_predictions > 0 else 0
        fn_rate = false_negatives / total_predictions if total_predictions > 0 else 0
        
        # Confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(predicted_values, actual_outcomes, confidence_scores)
        
        metrics = LearningMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            confidence_calibration=confidence_calibration
        )
        
        # Store metrics
        self._store_metrics(prediction_type, metrics, total_predictions)
        
        # Update adaptive thresholds
        self._update_adaptive_threshold(prediction_type, metrics, feedbacks)
        
        # Learn patterns
        self._learn_patterns(prediction_type, feedbacks)
        
        self.logger.info(f"Updated {prediction_type}: Accuracy={accuracy:.3f}, F1={f1_score:.3f}, FP_Rate={fp_rate:.3f}")
    
    def _calculate_confidence_calibration(self, predicted_values: List[float], 
                                        actual_outcomes: List[int], 
                                        confidence_scores: List[float]) -> float:
        """Calculate how well confidence scores match actual performance"""
        if not confidence_scores:
            return 0.5
        
        # Bin predictions by confidence level
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(bins) - 1):
            bin_mask = (np.array(confidence_scores) >= bins[i]) & (np.array(confidence_scores) < bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(np.array(actual_outcomes)[bin_mask])
                bin_confidence = np.mean(np.array(confidence_scores)[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
        
        if not bin_accuracies:
            return 0.5
        
        # Calculate calibration error (lower is better)
        calibration_error = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
        return 1.0 - calibration_error  # Convert to calibration score (higher is better)
    
    def _store_metrics(self, component: str, metrics: LearningMetrics, sample_size: int):
        """Store performance metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now()
        
        for metric_name, metric_value in asdict(metrics).items():
            cursor.execute('''
                INSERT INTO learning_metrics 
                (component, metric_type, metric_value, timestamp, sample_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (component, metric_name, metric_value, timestamp, sample_size))
        
        conn.commit()
        conn.close()
    
    def _update_adaptive_threshold(self, component: str, metrics: LearningMetrics, feedbacks: List[Tuple]):
        """Update adaptive thresholds based on performance"""
        current_threshold = self.thresholds.get(component, 0.75)
        
        # Calculate optimal threshold based on F1 score and false positive rate
        target_fp_rate = 0.1  # Target 10% false positive rate
        target_f1_score = 0.8  # Target 80% F1 score
        
        # Threshold adjustment logic
        threshold_adjustment = 0.0
        
        if metrics.false_positive_rate > target_fp_rate:
            # Too many false positives - increase threshold
            threshold_adjustment = self.threshold_adjustment_sensitivity
        elif metrics.false_positive_rate < target_fp_rate * 0.5 and metrics.f1_score < target_f1_score:
            # Very low false positives but poor F1 - decrease threshold
            threshold_adjustment = -self.threshold_adjustment_sensitivity
        
        # Apply learning rate
        threshold_adjustment *= self.learning_rate
        
        new_threshold = np.clip(current_threshold + threshold_adjustment, 0.1, 0.95)
        
        # Update threshold if significant change
        if abs(new_threshold - current_threshold) > 0.01:
            self.thresholds[component] = new_threshold
            self._save_threshold(component, new_threshold, metrics)
            
            self.logger.info(f"Adjusted {component} threshold: {current_threshold:.3f} -> {new_threshold:.3f}")
    
    def _save_threshold(self, component: str, new_threshold: float, metrics: LearningMetrics):
        """Save updated threshold to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current threshold data
        cursor.execute('SELECT * FROM adaptive_thresholds WHERE component = ?', (component,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            performance_history = json.loads(existing[3]) if existing[3] else []
            performance_history.append(metrics.f1_score)
            
            # Keep only last 50 performance scores
            if len(performance_history) > 50:
                performance_history = performance_history[-50:]
            
            cursor.execute('''
                UPDATE adaptive_thresholds 
                SET current_threshold = ?, optimal_threshold = ?, performance_history = ?, 
                    last_updated = ?, adjustment_count = adjustment_count + 1
                WHERE component = ?
            ''', (new_threshold, new_threshold, json.dumps(performance_history), 
                  datetime.now(), component))
        else:
            # Insert new
            cursor.execute('''
                INSERT INTO adaptive_thresholds 
                (component, current_threshold, optimal_threshold, performance_history, 
                 last_updated, adjustment_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (component, new_threshold, new_threshold, json.dumps([metrics.f1_score]), 
                  datetime.now(), 1))
        
        conn.commit()
        conn.close()
    
    def _learn_patterns(self, prediction_type: str, feedbacks: List[Tuple]):
        """Learn successful patterns from feedback data"""
        successful_cases = []
        failed_cases = []
        
        for feedback in feedbacks:
            context_data = json.loads(feedback[8]) if feedback[8] else {}  # context_data
            actual_outcome = feedback[4]  # actual_outcome
            
            if actual_outcome == 'correct':
                successful_cases.append(context_data)
            else:
                failed_cases.append(context_data)
        
        # Extract patterns from successful cases
        if len(successful_cases) >= 3:  # Minimum cases for pattern
            patterns = self._extract_patterns(successful_cases, prediction_type)
            
            for pattern in patterns:
                self._store_pattern(prediction_type, pattern, successful_cases)
    
    def _extract_patterns(self, cases: List[Dict], prediction_type: str) -> List[Dict]:
        """Extract common patterns from successful cases"""
        patterns = []
        
        if not cases:
            return patterns
        
        # Extract common features
        common_features = {}
        
        # Analyze numerical features
        numerical_features = ['photo_quality', 'form_completeness', 'consistency_score', 'cctv_readiness']
        for feature in numerical_features:
            values = [case.get(feature) for case in cases if case.get(feature) is not None]
            if len(values) >= len(cases) * 0.7:  # Feature present in 70% of cases
                common_features[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Analyze categorical features
        categorical_features = ['age_group', 'location_type', 'case_urgency']
        for feature in categorical_features:
            values = [case.get(feature) for case in cases if case.get(feature) is not None]
            if values:
                from collections import Counter
                value_counts = Counter(values)
                most_common = value_counts.most_common(3)  # Top 3 values
                if most_common[0][1] >= len(cases) * 0.4:  # Appears in 40% of cases
                    common_features[feature] = {
                        'common_values': most_common,
                        'dominant_value': most_common[0][0]
                    }
        
        if common_features:
            patterns.append({
                'type': 'feature_combination',
                'features': common_features,
                'sample_size': len(cases)
            })
        
        return patterns
    
    def _store_pattern(self, pattern_type: str, pattern: Dict, cases: List[Dict]):
        """Store learned pattern in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        success_rate = 1.0  # Initially assume 100% success rate
        pattern_data = json.dumps(pattern)
        
        cursor.execute('''
            INSERT INTO learned_patterns 
            (pattern_type, pattern_data, success_rate, usage_count, last_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pattern_type, pattern_data, success_rate, 0, datetime.now(), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_adaptive_threshold(self, component: str) -> float:
        """Get current adaptive threshold for component"""
        return self.thresholds.get(component, 0.75)
    
    def apply_learned_patterns(self, case_data: Dict, prediction_type: str) -> Dict[str, Any]:
        """Apply learned patterns to improve predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pattern_data, success_rate, usage_count 
            FROM learned_patterns 
            WHERE pattern_type = ? AND success_rate > 0.7
            ORDER BY success_rate DESC, usage_count DESC
        ''', (prediction_type,))
        
        patterns = cursor.fetchall()
        conn.close()
        
        recommendations = {
            'confidence_boost': 0.0,
            'risk_factors': [],
            'success_indicators': [],
            'pattern_matches': []
        }
        
        for pattern_data, success_rate, usage_count in patterns:
            pattern = json.loads(pattern_data)
            match_score = self._calculate_pattern_match(case_data, pattern)
            
            if match_score > self.pattern_similarity_threshold:
                recommendations['pattern_matches'].append({
                    'pattern': pattern,
                    'match_score': match_score,
                    'success_rate': success_rate
                })
                
                # Boost confidence based on pattern match
                confidence_boost = (match_score * success_rate) * 0.1
                recommendations['confidence_boost'] += confidence_boost
                
                # Update pattern usage
                self._update_pattern_usage(pattern_data)
        
        return recommendations
    
    def _calculate_pattern_match(self, case_data: Dict, pattern: Dict) -> float:
        """Calculate how well case data matches learned pattern"""
        if pattern.get('type') != 'feature_combination':
            return 0.0
        
        features = pattern.get('features', {})
        if not features:
            return 0.0
        
        match_scores = []
        
        for feature_name, feature_pattern in features.items():
            case_value = case_data.get(feature_name)
            
            if case_value is None:
                continue
            
            if isinstance(feature_pattern, dict) and 'mean' in feature_pattern:
                # Numerical feature
                mean = feature_pattern['mean']
                std = feature_pattern['std']
                
                if std > 0:
                    # Calculate z-score similarity
                    z_score = abs(case_value - mean) / std
                    similarity = max(0, 1 - z_score / 3)  # 3-sigma rule
                else:
                    similarity = 1.0 if case_value == mean else 0.0
                
                match_scores.append(similarity)
            
            elif isinstance(feature_pattern, dict) and 'dominant_value' in feature_pattern:
                # Categorical feature
                if case_value == feature_pattern['dominant_value']:
                    match_scores.append(1.0)
                else:
                    # Check if in common values
                    common_values = [item[0] for item in feature_pattern.get('common_values', [])]
                    if case_value in common_values:
                        match_scores.append(0.7)
                    else:
                        match_scores.append(0.0)
        
        return np.mean(match_scores) if match_scores else 0.0
    
    def _update_pattern_usage(self, pattern_data: str):
        """Update pattern usage count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE learned_patterns 
            SET usage_count = usage_count + 1, last_used = ?
            WHERE pattern_data = ?
        ''', (datetime.now(), pattern_data))
        
        conn.commit()
        conn.close()
    
    def _load_thresholds(self):
        """Load adaptive thresholds from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT component, current_threshold FROM adaptive_thresholds')
        thresholds = cursor.fetchall()
        
        for component, threshold in thresholds:
            self.thresholds[component] = threshold
        
        conn.close()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get feedback counts
        cursor.execute('''
            SELECT prediction_type, COUNT(*) as count, 
                   AVG(CASE WHEN actual_outcome = 'correct' THEN 1.0 ELSE 0.0 END) as accuracy
            FROM learning_feedback 
            GROUP BY prediction_type
        ''')
        feedback_stats = cursor.fetchall()
        
        # Get recent metrics
        cursor.execute('''
            SELECT component, metric_type, AVG(metric_value) as avg_value
            FROM learning_metrics 
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY component, metric_type
        ''')
        recent_metrics = cursor.fetchall()
        
        # Get threshold adjustments
        cursor.execute('''
            SELECT component, current_threshold, adjustment_count, last_updated
            FROM adaptive_thresholds
        ''')
        threshold_stats = cursor.fetchall()
        
        # Get pattern counts
        cursor.execute('''
            SELECT pattern_type, COUNT(*) as count, AVG(success_rate) as avg_success_rate
            FROM learned_patterns
            GROUP BY pattern_type
        ''')
        pattern_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'feedback_stats': [
                {'type': row[0], 'count': row[1], 'accuracy': row[2]} 
                for row in feedback_stats
            ],
            'recent_metrics': [
                {'component': row[0], 'metric': row[1], 'value': row[2]} 
                for row in recent_metrics
            ],
            'thresholds': [
                {'component': row[0], 'threshold': row[1], 'adjustments': row[2], 'last_updated': row[3]} 
                for row in threshold_stats
            ],
            'patterns': [
                {'type': row[0], 'count': row[1], 'success_rate': row[2]} 
                for row in pattern_stats
            ],
            'total_feedback': sum(row[1] for row in feedback_stats),
            'system_accuracy': np.mean([row[2] for row in feedback_stats]) if feedback_stats else 0.0
        }
    
    def reduce_false_positives(self, prediction_type: str, target_fp_rate: float = 0.05):
        """Specifically focus on reducing false positives through reinforcement learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent false positive cases
        cursor.execute('''
            SELECT * FROM learning_feedback 
            WHERE prediction_type = ? AND actual_outcome = 'false_positive'
            AND timestamp > datetime('now', '-30 days')
            ORDER BY timestamp DESC
        ''', (prediction_type,))
        
        fp_cases = cursor.fetchall()
        
        if len(fp_cases) < 5:
            return  # Need minimum cases for analysis
        
        # Analyze false positive patterns
        fp_contexts = []
        for case in fp_cases:
            context = json.loads(case[8]) if case[8] else {}
            fp_contexts.append(context)
        
        # Extract common features in false positives
        fp_patterns = self._extract_patterns(fp_contexts, f"{prediction_type}_false_positive")
        
        # Adjust thresholds to reduce false positives
        current_threshold = self.get_adaptive_threshold(prediction_type)
        
        # Calculate false positive rate
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN actual_outcome = 'false_positive' THEN 1 ELSE 0 END) as fp_count
            FROM learning_feedback 
            WHERE prediction_type = ? AND timestamp > datetime('now', '-30 days')
        ''', (prediction_type,))
        
        result = cursor.fetchone()
        current_fp_rate = result[1] / result[0] if result[0] > 0 else 0
        
        if current_fp_rate > target_fp_rate:
            # Increase threshold to reduce false positives
            adjustment = min(0.1, (current_fp_rate - target_fp_rate) * 2)
            new_threshold = min(0.95, current_threshold + adjustment)
            
            self.thresholds[prediction_type] = new_threshold
            
            # Store false positive patterns for future reference
            for pattern in fp_patterns:
                pattern['type'] = 'false_positive_indicator'
                self._store_pattern(f"{prediction_type}_fp", pattern, fp_contexts)
            
            self.logger.info(f"Reduced FP threshold for {prediction_type}: {current_threshold:.3f} -> {new_threshold:.3f}")
        
        conn.close()

# Global learning system instance
continuous_learning_system = ContinuousLearningSystem()