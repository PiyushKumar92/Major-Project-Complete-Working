"""
Confidence Distribution and Breakdown Analysis System
Provides detailed confidence analysis for AI detections
"""

import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from models import PersonDetection, LocationMatch, Case
from __init__ import db

class ConfidenceAnalyzer:
    """Comprehensive confidence analysis and distribution system"""
    
    def __init__(self):
        self.confidence_ranges = {
            'very_high': (0.9, 1.0),
            'high': (0.8, 0.9),
            'good': (0.7, 0.8),
            'medium': (0.6, 0.7),
            'low': (0.5, 0.6),
            'very_low': (0.0, 0.5)
        }
        
        self.confidence_labels = {
            'very_high': 'Very High (90-100%)',
            'high': 'High (80-90%)',
            'good': 'Good (70-80%)',
            'medium': 'Medium (60-70%)',
            'low': 'Low (50-60%)',
            'very_low': 'Very Low (0-50%)'
        }
        
        self.confidence_colors = {
            'very_high': '#28a745',  # Green
            'high': '#20c997',       # Teal
            'good': '#ffc107',       # Yellow
            'medium': '#fd7e14',     # Orange
            'low': '#dc3545',        # Red
            'very_low': '#6c757d'    # Gray
        }
    
    def get_confidence_distribution(self, case_id=None, match_id=None, location_id=None):
        """Get comprehensive confidence distribution analysis"""
        
        # Build query based on filters
        query = PersonDetection.query
        
        if case_id:
            query = query.join(LocationMatch).filter(LocationMatch.case_id == case_id)
        elif match_id:
            query = query.filter(PersonDetection.location_match_id == match_id)
        elif location_id:
            query = query.join(LocationMatch).filter(LocationMatch.footage_id == location_id)
        
        detections = query.all()
        
        if not detections:
            return self._empty_distribution()
        
        # Calculate distribution
        distribution = self._calculate_distribution(detections)
        
        # Calculate statistics
        statistics = self._calculate_statistics(detections)
        
        # Generate insights
        insights = self._generate_insights(detections, distribution, statistics)
        
        # Time-based analysis
        time_analysis = self._analyze_confidence_over_time(detections)
        
        return {
            'distribution': distribution,
            'statistics': statistics,
            'insights': insights,
            'time_analysis': time_analysis,
            'total_detections': len(detections),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_distribution(self, detections):
        """Calculate confidence distribution across ranges"""
        distribution = {}
        total_count = len(detections)
        
        for range_name, (min_conf, max_conf) in self.confidence_ranges.items():
            count = sum(1 for d in detections 
                       if min_conf <= d.confidence_score < max_conf or 
                       (range_name == 'very_high' and d.confidence_score == 1.0))
            
            percentage = (count / total_count * 100) if total_count > 0 else 0
            
            distribution[range_name] = {
                'count': count,
                'percentage': round(percentage, 1),
                'label': self.confidence_labels[range_name],
                'color': self.confidence_colors[range_name],
                'range': f"{int(min_conf*100)}-{int(max_conf*100)}%"
            }
        
        return distribution
    
    def _calculate_statistics(self, detections):
        """Calculate comprehensive confidence statistics"""
        if not detections:
            return {}
        
        confidence_scores = [d.confidence_score for d in detections]
        
        # Basic statistics
        mean_confidence = np.mean(confidence_scores)
        median_confidence = np.median(confidence_scores)
        std_confidence = np.std(confidence_scores)
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)
        
        # Percentiles
        percentiles = {
            '25th': np.percentile(confidence_scores, 25),
            '50th': np.percentile(confidence_scores, 50),
            '75th': np.percentile(confidence_scores, 75),
            '90th': np.percentile(confidence_scores, 90),
            '95th': np.percentile(confidence_scores, 95)
        }
        
        # Quality metrics
        high_quality_count = sum(1 for score in confidence_scores if score >= 0.8)
        medium_quality_count = sum(1 for score in confidence_scores if 0.6 <= score < 0.8)
        low_quality_count = sum(1 for score in confidence_scores if score < 0.6)
        
        # Reliability score (weighted average based on confidence distribution)
        reliability_score = self._calculate_reliability_score(confidence_scores)
        
        return {
            'mean': round(mean_confidence, 3),
            'median': round(median_confidence, 3),
            'std_deviation': round(std_confidence, 3),
            'min': round(min_confidence, 3),
            'max': round(max_confidence, 3),
            'percentiles': {k: round(v, 3) for k, v in percentiles.items()},
            'quality_breakdown': {
                'high_quality': {'count': high_quality_count, 'percentage': round(high_quality_count/len(confidence_scores)*100, 1)},
                'medium_quality': {'count': medium_quality_count, 'percentage': round(medium_quality_count/len(confidence_scores)*100, 1)},
                'low_quality': {'count': low_quality_count, 'percentage': round(low_quality_count/len(confidence_scores)*100, 1)}
            },
            'reliability_score': round(reliability_score, 2),
            'total_samples': len(confidence_scores)
        }
    
    def _calculate_reliability_score(self, confidence_scores):
        """Calculate overall reliability score (0-100)"""
        if not confidence_scores:
            return 0
        
        # Weight scores based on confidence levels
        weighted_sum = 0
        total_weight = 0
        
        for score in confidence_scores:
            if score >= 0.9:
                weight = 1.0
            elif score >= 0.8:
                weight = 0.8
            elif score >= 0.7:
                weight = 0.6
            elif score >= 0.6:
                weight = 0.4
            else:
                weight = 0.2
            
            weighted_sum += score * weight
            total_weight += weight
        
        return (weighted_sum / total_weight * 100) if total_weight > 0 else 0
    
    def _generate_insights(self, detections, distribution, statistics):
        """Generate actionable insights from confidence analysis"""
        insights = []
        
        total_detections = len(detections)
        high_conf_percentage = distribution['very_high']['percentage'] + distribution['high']['percentage']
        low_conf_percentage = distribution['low']['percentage'] + distribution['very_low']['percentage']
        
        # Overall quality assessment
        if high_conf_percentage >= 70:
            insights.append({
                'type': 'success',
                'title': 'Excellent Detection Quality',
                'message': f'{high_conf_percentage:.1f}% of detections have high confidence (80%+)',
                'icon': 'fas fa-check-circle'
            })
        elif high_conf_percentage >= 50:
            insights.append({
                'type': 'info',
                'title': 'Good Detection Quality',
                'message': f'{high_conf_percentage:.1f}% of detections have high confidence',
                'icon': 'fas fa-info-circle'
            })
        else:
            insights.append({
                'type': 'warning',
                'title': 'Mixed Detection Quality',
                'message': f'Only {high_conf_percentage:.1f}% of detections have high confidence',
                'icon': 'fas fa-exclamation-triangle'
            })
        
        # Low confidence warning
        if low_conf_percentage > 30:
            insights.append({
                'type': 'warning',
                'title': 'High Low-Confidence Rate',
                'message': f'{low_conf_percentage:.1f}% of detections have low confidence (<60%)',
                'icon': 'fas fa-exclamation-triangle'
            })
        
        # Consistency analysis
        if statistics['std_deviation'] < 0.1:
            insights.append({
                'type': 'success',
                'title': 'Consistent Results',
                'message': f'Low variance in confidence scores (σ = {statistics["std_deviation"]:.3f})',
                'icon': 'fas fa-chart-line'
            })
        elif statistics['std_deviation'] > 0.25:
            insights.append({
                'type': 'warning',
                'title': 'Inconsistent Results',
                'message': f'High variance in confidence scores (σ = {statistics["std_deviation"]:.3f})',
                'icon': 'fas fa-chart-line'
            })
        
        # Reliability assessment
        reliability = statistics['reliability_score']
        if reliability >= 80:
            insights.append({
                'type': 'success',
                'title': 'High Reliability',
                'message': f'Overall reliability score: {reliability:.1f}/100',
                'icon': 'fas fa-shield-alt'
            })
        elif reliability >= 60:
            insights.append({
                'type': 'info',
                'title': 'Moderate Reliability',
                'message': f'Overall reliability score: {reliability:.1f}/100',
                'icon': 'fas fa-shield-alt'
            })
        else:
            insights.append({
                'type': 'danger',
                'title': 'Low Reliability',
                'message': f'Overall reliability score: {reliability:.1f}/100 - Review needed',
                'icon': 'fas fa-shield-alt'
            })
        
        # Recommendations
        if low_conf_percentage > 20:
            insights.append({
                'type': 'info',
                'title': 'Recommendation',
                'message': 'Consider manual review of low-confidence detections',
                'icon': 'fas fa-lightbulb'
            })
        
        return insights
    
    def _analyze_confidence_over_time(self, detections):
        """Analyze how confidence varies over time in video"""
        if not detections:
            return {}
        
        # Sort by timestamp
        sorted_detections = sorted(detections, key=lambda d: d.timestamp)
        
        # Group by time intervals (every 10 seconds)
        time_groups = defaultdict(list)
        for detection in sorted_detections:
            time_bucket = int(detection.timestamp // 10) * 10  # 10-second buckets
            time_groups[time_bucket].append(detection.confidence_score)
        
        # Calculate average confidence per time bucket
        time_analysis = []
        for time_bucket in sorted(time_groups.keys()):
            scores = time_groups[time_bucket]
            avg_confidence = np.mean(scores)
            max_confidence = np.max(scores)
            min_confidence = np.min(scores)
            count = len(scores)
            
            time_analysis.append({
                'timestamp': time_bucket,
                'avg_confidence': round(avg_confidence, 3),
                'max_confidence': round(max_confidence, 3),
                'min_confidence': round(min_confidence, 3),
                'detection_count': count,
                'time_label': f"{time_bucket//60:02d}:{time_bucket%60:02d}"
            })
        
        return {
            'timeline': time_analysis,
            'total_duration': max(time_groups.keys()) if time_groups else 0,
            'peak_confidence_time': max(time_analysis, key=lambda x: x['avg_confidence'])['timestamp'] if time_analysis else 0,
            'lowest_confidence_time': min(time_analysis, key=lambda x: x['avg_confidence'])['timestamp'] if time_analysis else 0
        }
    
    def _empty_distribution(self):
        """Return empty distribution structure"""
        return {
            'distribution': {range_name: {
                'count': 0,
                'percentage': 0,
                'label': self.confidence_labels[range_name],
                'color': self.confidence_colors[range_name],
                'range': f"{int(min_conf*100)}-{int(max_conf*100)}%"
            } for range_name, (min_conf, max_conf) in self.confidence_ranges.items()},
            'statistics': {},
            'insights': [{
                'type': 'info',
                'title': 'No Data Available',
                'message': 'No detections found for analysis',
                'icon': 'fas fa-info-circle'
            }],
            'time_analysis': {},
            'total_detections': 0,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_case_confidence_summary(self, case_id):
        """Get confidence summary for a specific case across all locations"""
        matches = LocationMatch.query.filter_by(case_id=case_id).all()
        
        if not matches:
            return self._empty_distribution()
        
        all_detections = []
        location_summaries = []
        
        for match in matches:
            detections = PersonDetection.query.filter_by(location_match_id=match.id).all()
            all_detections.extend(detections)
            
            if detections:
                location_analysis = self.get_confidence_distribution(match_id=match.id)
                location_summaries.append({
                    'location_name': match.footage.location_name,
                    'match_id': match.id,
                    'detection_count': len(detections),
                    'avg_confidence': location_analysis['statistics'].get('mean', 0),
                    'max_confidence': location_analysis['statistics'].get('max', 0),
                    'reliability_score': location_analysis['statistics'].get('reliability_score', 0)
                })
        
        # Overall analysis
        overall_analysis = self.get_confidence_distribution()
        overall_analysis['location_summaries'] = location_summaries
        
        return overall_analysis
    
    def export_confidence_report(self, case_id=None, match_id=None):
        """Export detailed confidence analysis report"""
        analysis = self.get_confidence_distribution(case_id=case_id, match_id=match_id)
        
        report = {
            'report_type': 'Confidence Distribution Analysis',
            'generated_at': datetime.now().isoformat(),
            'filters': {
                'case_id': case_id,
                'match_id': match_id
            },
            'summary': {
                'total_detections': analysis['total_detections'],
                'mean_confidence': analysis['statistics'].get('mean', 0),
                'reliability_score': analysis['statistics'].get('reliability_score', 0)
            },
            'detailed_analysis': analysis
        }
        
        return report

# Global instance
confidence_analyzer = ConfidenceAnalyzer()