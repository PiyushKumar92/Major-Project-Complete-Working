"""
Integration Routes - Flask API endpoints for external system integration
"""

from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from datetime import datetime, timedelta
import json
from integration_engine import integration_engine
from models import db, Case, Person, Detection

integration_bp = Blueprint('integration', __name__, url_prefix='/integration')

@integration_bp.route('/dashboard')
@login_required
def integration_dashboard():
    """Integration dashboard"""
    system_info = integration_engine.get_system_info()
    integration_status = integration_engine.get_integration_status()
    
    return render_template('integration_dashboard.html',
                         system_info=system_info,
                         integration_status=integration_status)

@integration_bp.route('/api/search', methods=['POST'])
@login_required
def comprehensive_search():
    """Comprehensive search across all integrated systems"""
    try:
        data = request.get_json()
        
        # Extract person features
        person_features = {
            'face_encoding': data.get('face_encoding', []),
            'age_range': data.get('age_range', [18, 65]),
            'gender': data.get('gender', 'unknown'),
            'location': data.get('location', ''),
            'description': data.get('description', '')
        }
        
        # Search options
        search_options = {
            'include_police_db': data.get('include_police_db', True),
            'include_social_media': data.get('include_social_media', True),
            'include_traffic_cameras': data.get('include_traffic_cameras', True),
            'time_range_hours': data.get('time_range_hours', 24)
        }
        
        # Perform comprehensive search
        results = integration_engine.comprehensive_search(person_features, search_options)
        
        return jsonify({
            'success': True,
            'results': {
                'police_db': [
                    {
                        'source': r.source,
                        'data': r.data,
                        'confidence': r.confidence,
                        'timestamp': r.timestamp.isoformat(),
                        'status': r.status
                    } for r in results['police_db']
                ],
                'social_media': [
                    {
                        'source': r.source,
                        'data': r.data,
                        'confidence': r.confidence,
                        'timestamp': r.timestamp.isoformat(),
                        'status': r.status
                    } for r in results['social_media']
                ],
                'traffic_cameras': [
                    {
                        'source': r.source,
                        'data': r.data,
                        'confidence': r.confidence,
                        'timestamp': r.timestamp.isoformat(),
                        'status': r.status
                    } for r in results['traffic_cameras']
                ],
                'summary': results['summary']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@integration_bp.route('/api/police-db/search', methods=['POST'])
@login_required
def police_db_search():
    """Search police database"""
    try:
        data = request.get_json()
        person_features = {
            'face_encoding': data.get('face_encoding', []),
            'age_range': data.get('age_range', [18, 65]),
            'gender': data.get('gender', 'unknown'),
            'location': data.get('location', '')
        }
        
        results = integration_engine.police_db.search_missing_persons(person_features)
        
        return jsonify({
            'success': True,
            'results': [
                {
                    'source': r.source,
                    'data': r.data,
                    'confidence': r.confidence,
                    'timestamp': r.timestamp.isoformat(),
                    'status': r.status
                } for r in results
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@integration_bp.route('/api/social-media/search', methods=['POST'])
@login_required
def social_media_search():
    """Search social media platforms"""
    try:
        data = request.get_json()
        person_features = {
            'face_encoding': data.get('face_encoding', []),
            'description': data.get('description', ''),
            'location': data.get('location', '')
        }
        
        results = integration_engine.social_media.search_person_posts(person_features)
        
        return jsonify({
            'success': True,
            'results': [
                {
                    'source': r.source,
                    'data': r.data,
                    'confidence': r.confidence,
                    'timestamp': r.timestamp.isoformat(),
                    'status': r.status
                } for r in results
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@integration_bp.route('/api/traffic-cameras/search', methods=['POST'])
@login_required
def traffic_cameras_search():
    """Search traffic camera footage"""
    try:
        data = request.get_json()
        person_features = {
            'face_encoding': data.get('face_encoding', []),
            'location': data.get('location', '')
        }
        
        # Time range
        hours_back = data.get('time_range_hours', 24)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        results = integration_engine.traffic_cameras.search_traffic_footage(
            person_features, (start_time, end_time)
        )
        
        return jsonify({
            'success': True,
            'results': [
                {
                    'source': r.source,
                    'data': r.data,
                    'confidence': r.confidence,
                    'timestamp': r.timestamp.isoformat(),
                    'status': r.status
                } for r in results
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@integration_bp.route('/api/status')
@login_required
def integration_status():
    """Get integration system status"""
    try:
        status = integration_engine.get_integration_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@integration_bp.route('/api/system-info')
@login_required
def system_info():
    """Get integration system information"""
    try:
        info = integration_engine.get_system_info()
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@integration_bp.route('/api/traffic-cameras/list')
@login_required
def list_traffic_cameras():
    """List available traffic cameras"""
    try:
        cameras = integration_engine.traffic_cameras.camera_endpoints
        return jsonify({
            'success': True,
            'cameras': cameras
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@integration_bp.route('/api/case/<int:case_id>/integrate', methods=['POST'])
@login_required
def integrate_case_data(case_id):
    """Integrate external data for a specific case"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get person features from case
        person_features = {
            'age_range': [case.age - 5, case.age + 5] if case.age else [18, 65],
            'gender': case.gender or 'unknown',
            'location': case.last_seen_location or '',
            'description': case.description or ''
        }
        
        # Perform comprehensive search
        results = integration_engine.comprehensive_search(person_features)
        
        # Store integration results in database (optional)
        # You can create an IntegrationResult model to store these
        
        return jsonify({
            'success': True,
            'case_id': case_id,
            'integration_results': {
                'police_db_matches': len(results['police_db']),
                'social_media_matches': len(results['social_media']),
                'traffic_camera_matches': len(results['traffic_cameras']),
                'summary': results['summary']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500