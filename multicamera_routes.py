"""
Multi-Camera Routes
API endpoints for cross-camera tracking and route reconstruction
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from multicamera_engine import multicamera_engine
import json
from datetime import datetime

multicamera_bp = Blueprint('multicamera', __name__)

@multicamera_bp.route('/multicamera-dashboard')
def multicamera_dashboard():
    """Multi-camera correlation dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('main.login'))
    
    return render_template('multicamera_dashboard.html')

@multicamera_bp.route('/api/multicamera/register-camera', methods=['POST'])
def register_camera():
    """Register camera in network"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        position = data.get('position', [0, 0])
        coverage_area = data.get('coverage_area', 50)
        
        multicamera_engine.register_camera(camera_id, position, coverage_area)
        
        return jsonify({
            'success': True,
            'message': f'Camera {camera_id} registered successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@multicamera_bp.route('/api/multicamera/global-tracks')
def get_global_tracks():
    """Get global tracking data"""
    try:
        hours = request.args.get('hours', 24, type=int)
        tracks = multicamera_engine.get_global_tracks(hours)
        
        return jsonify({
            'success': True,
            'tracks': tracks,
            'total_tracks': len(tracks)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@multicamera_bp.route('/api/multicamera/route/<global_track_id>')
def get_route_reconstruction(global_track_id):
    """Get reconstructed route for track"""
    try:
        route = multicamera_engine.get_route_reconstruction(global_track_id)
        
        if route:
            # Convert datetime objects to ISO format for JSON
            route_data = route.copy()
            route_data['start_time'] = route['start_time'].isoformat()
            route_data['end_time'] = route['end_time'].isoformat()
            
            for point in route_data['route_points']:
                point['timestamp'] = point['timestamp'].isoformat()
            
            return jsonify({
                'success': True,
                'route': route_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Route not found'
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@multicamera_bp.route('/api/multicamera/correlations')
def get_camera_correlations():
    """Get camera correlation statistics"""
    try:
        correlations = multicamera_engine.get_camera_correlations()
        
        return jsonify({
            'success': True,
            'correlations': correlations
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@multicamera_bp.route('/api/multicamera/process-detection', methods=['POST'])
def process_detection():
    """Process detection for cross-camera correlation"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        detection = data.get('detection')
        
        # Mock detection processing
        result = multicamera_engine.process_detection(camera_id, detection)
        
        return jsonify({
            'success': True,
            'correlation': result
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@multicamera_bp.route('/api/multicamera/network-status')
def get_network_status():
    """Get multi-camera network status"""
    try:
        correlations = multicamera_engine.get_camera_correlations()
        tracks = multicamera_engine.get_global_tracks(1)  # Last hour
        
        status = {
            'cameras_online': correlations['total_cameras'],
            'active_tracks': len(tracks),
            'total_transitions': correlations['total_transitions'],
            'network_health': 'good' if correlations['total_cameras'] > 0 else 'offline'
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})