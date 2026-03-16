"""
Analytics Routes
API endpoints for advanced analytics dashboard
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from analytics_engine import analytics_engine
import json
import base64
import cv2
import numpy as np

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics-dashboard')
def analytics_dashboard():
    """Analytics dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('main.login'))
    
    return render_template('analytics_dashboard.html')

@analytics_bp.route('/api/analytics/movement-patterns/<camera_id>')
def get_movement_patterns(camera_id):
    """Get movement patterns for camera"""
    try:
        hours = request.args.get('hours', 24, type=int)
        patterns = analytics_engine.get_movement_patterns(camera_id, hours)
        
        return jsonify({
            'success': True,
            'patterns': patterns,
            'camera_id': camera_id
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/heatmap/<camera_id>')
def get_heatmap(camera_id):
    """Get heatmap data"""
    try:
        heatmap = analytics_engine.get_heatmap(camera_id)
        
        # Convert to base64 for web display
        _, buffer = cv2.imencode('.png', heatmap)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'heatmap': heatmap_b64,
            'camera_id': camera_id
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/predictions/<camera_id>')
def get_predictions(camera_id):
    """Get predictive analytics"""
    try:
        predictions = analytics_engine.get_predictions(camera_id)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'camera_id': camera_id
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/process-frame', methods=['POST'])
def process_frame():
    """Process frame for analytics"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        
        # This would typically receive frame data
        # For demo, return mock analytics
        mock_analytics = {
            'detections': [
                {'bbox': [100, 100, 50, 100], 'confidence': 0.85, 'center': [125, 150]}
            ],
            'movement_data': {
                'track_id': 1,
                'path': [{'x': 125, 'y': 150, 'timestamp': '2025-12-16T17:45:00'}],
                'speed': 2.5
            }
        }
        
        return jsonify({
            'success': True,
            'analytics': mock_analytics
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/summary')
def get_analytics_summary():
    """Get overall analytics summary"""
    try:
        # Mock summary data - in real implementation, aggregate from database
        summary = {
            'total_cameras': 3,
            'active_tracks': 5,
            'avg_detections_per_hour': 12.5,
            'peak_activity_hour': 14,
            'trend': 'increasing'
        }
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})