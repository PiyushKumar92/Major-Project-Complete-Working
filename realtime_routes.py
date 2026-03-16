"""
Real-Time Processing Routes
WebSocket endpoints for live CCTV monitoring
"""

from flask import Blueprint, render_template, request, jsonify, session
from flask_socketio import emit
from realtime_processor import RealTimeProcessor
from flask import redirect, url_for
import json

realtime_bp = Blueprint('realtime', __name__)

# Global processor instance (will be initialized in app)
realtime_processor = None

def init_realtime_processor(socketio, db):
    """Initialize real-time processor"""
    global realtime_processor
    realtime_processor = RealTimeProcessor(socketio, db)
    return realtime_processor

@realtime_bp.route('/live-monitoring')
def live_monitoring():
    """Live CCTV monitoring dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('live_monitoring.html')

@realtime_bp.route('/api/start-stream', methods=['POST'])
def start_stream():
    """Start CCTV stream processing"""
    try:
        data = request.get_json()
        stream_id = data.get('stream_id')
        source = data.get('source')  # Camera URL or device index
        
        # Get missing persons data from database
        missing_persons = get_missing_persons_for_monitoring()
        
        success = realtime_processor.start_stream(stream_id, source, missing_persons)
        
        return jsonify({
            'success': success,
            'message': 'Stream started successfully' if success else 'Failed to start stream'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@realtime_bp.route('/api/stop-stream', methods=['POST'])
def stop_stream():
    """Stop CCTV stream processing"""
    try:
        data = request.get_json()
        stream_id = data.get('stream_id')
        
        success = realtime_processor.stop_stream(stream_id)
        
        return jsonify({
            'success': success,
            'message': 'Stream stopped successfully' if success else 'Stream not found'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@realtime_bp.route('/api/stream-status')
def stream_status():
    """Get status of all streams"""
    try:
        stats = realtime_processor.get_stream_stats()
        active_streams = realtime_processor.get_active_streams()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'active_streams': active_streams
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def get_missing_persons_for_monitoring():
    """Get missing persons data for real-time monitoring"""
    try:
        from models import MissingPerson
        from sqlalchemy import and_
        
        # Get active missing person cases
        missing_persons = MissingPerson.query.filter(
            and_(
                MissingPerson.status.in_(['reported', 'investigating', 'leads_found']),
                MissingPerson.primary_photo.isnot(None)
            )
        ).all()
        
        return [{
            'id': person.id,
            'name': person.name,
            'photo_path': person.primary_photo
        } for person in missing_persons]
        
    except Exception as e:
        print(f"Error fetching missing persons: {e}")
        return []

# WebSocket Events
def register_socketio_events(socketio):
    """Register WebSocket events"""
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected to real-time monitoring')
        emit('connection_status', {'status': 'connected'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected from real-time monitoring')
    
    @socketio.on('request_stream_status')
    def handle_stream_status():
        """Send current stream status to client"""
        if realtime_processor:
            stats = realtime_processor.get_stream_stats()
            emit('stream_status_update', stats)