"""
Advanced UI/UX Routes
Interactive timeline, 3D visualization, and mobile interfaces
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import json
from datetime import datetime, timedelta

advanced_ui_bp = Blueprint('advanced_ui', __name__)

@advanced_ui_bp.route('/timeline-dashboard')
def timeline_dashboard():
    """Interactive timeline dashboard"""
    # Remove session check - allow all logged in users
    return render_template('timeline_dashboard.html')

@advanced_ui_bp.route('/mobile-app')
def mobile_app():
    """Mobile app interface"""
    return render_template('mobile_app.html')

@advanced_ui_bp.route('/api/timeline/events')
def get_timeline_events():
    """Get timeline events data"""
    try:
        # Mock timeline data - replace with actual database queries
        events = [
            {
                'id': 1,
                'title': 'Case Reported',
                'description': 'Missing person case filed',
                'date': '2025-12-10T10:00:00',
                'type': 'report',
                'status': 'completed',
                'location': 'Police Station',
                'person_id': 1
            },
            {
                'id': 2,
                'title': 'CCTV Analysis Started',
                'description': 'AI analysis of surveillance footage',
                'date': '2025-12-10T14:30:00',
                'type': 'analysis',
                'status': 'completed',
                'location': 'Tech Center',
                'person_id': 1
            },
            {
                'id': 3,
                'title': 'Lead Found',
                'description': 'Potential match detected in camera network',
                'date': '2025-12-11T09:15:00',
                'type': 'lead',
                'status': 'active',
                'location': 'Shopping Mall',
                'person_id': 1
            },
            {
                'id': 4,
                'title': 'Investigation Update',
                'description': 'Field team dispatched to location',
                'date': '2025-12-11T16:45:00',
                'type': 'investigation',
                'status': 'in_progress',
                'location': 'Field Location',
                'person_id': 1
            }
        ]
        
        return jsonify({
            'success': True,
            'events': events
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@advanced_ui_bp.route('/api/mobile/cases')
def get_mobile_cases():
    """Get cases data for mobile app"""
    try:
        # Mock mobile cases data
        cases = [
            {
                'id': 1,
                'name': 'John Doe',
                'age': 25,
                'status': 'investigating',
                'last_seen': '2025-12-10',
                'location': 'Downtown Area',
                'photo': '/static/uploads/case_1_photo.jpg',
                'priority': 'high'
            },
            {
                'id': 2,
                'name': 'Jane Smith',
                'age': 17,
                'status': 'leads_found',
                'last_seen': '2025-12-08',
                'location': 'School District',
                'photo': '/static/uploads/case_2_photo.jpg',
                'priority': 'urgent'
            }
        ]
        
        return jsonify({
            'success': True,
            'cases': cases
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@advanced_ui_bp.route('/api/mobile/submit-tip', methods=['POST'])
def submit_mobile_tip():
    """Submit tip from mobile app"""
    try:
        data = request.get_json()
        
        tip_data = {
            'case_id': data.get('case_id'),
            'tip_text': data.get('tip_text'),
            'location': data.get('location'),
            'contact_info': data.get('contact_info'),
            'timestamp': datetime.now().isoformat()
        }
        
        # In real implementation, save to database
        
        return jsonify({
            'success': True,
            'message': 'Tip submitted successfully',
            'tip_id': f"tip_{tip_data['case_id']}_{int(datetime.now().timestamp())}"
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@advanced_ui_bp.route('/api/ui/heatmap-data')
def get_heatmap_data():
    """Get heatmap data for visualization"""
    try:
        # Mock heatmap data
        heatmap_data = {
            'points': [
                {'lat': 40.7128, 'lng': -74.0060, 'intensity': 0.8},
                {'lat': 40.7589, 'lng': -73.9851, 'intensity': 0.6},
                {'lat': 40.7505, 'lng': -73.9934, 'intensity': 0.9},
                {'lat': 40.7282, 'lng': -73.7949, 'intensity': 0.4}
            ],
            'center': {'lat': 40.7128, 'lng': -74.0060},
            'zoom': 12
        }
        
        return jsonify({
            'success': True,
            'heatmap': heatmap_data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})