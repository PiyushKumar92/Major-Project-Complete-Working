"""
Cloud & Scalability Routes
API endpoints for cloud management and scaling
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from cloud_engine import cloud_engine
import json
from datetime import datetime

cloud_bp = Blueprint('cloud', __name__)

@cloud_bp.route('/cloud-dashboard')
def cloud_dashboard():
    """Cloud management dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('main.login'))
    
    return render_template('cloud_dashboard.html')

@cloud_bp.route('/api/cloud/status')
def get_cloud_status():
    """Get cloud system status"""
    try:
        metrics = cloud_engine.get_cloud_metrics()
        instances = cloud_engine.auto_scaler.get_instances()
        edge_status = cloud_engine.edge_manager.get_edge_status()
        
        status = {
            'aws_enabled': cloud_engine.aws_enabled,
            'instance_count': len(instances),
            'edge_nodes': edge_status['total_nodes'],
            'online_nodes': edge_status['online_nodes'],
            'metrics': metrics
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/upload', methods=['POST'])
def upload_to_cloud():
    """Upload file to cloud storage"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save file temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        # Upload to cloud
        result = cloud_engine.upload_to_cloud(temp_path)
        
        # Clean up temp file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/scale', methods=['POST'])
def scale_resources():
    """Scale cloud resources"""
    try:
        data = request.get_json()
        target_capacity = data.get('target_capacity', 1)
        
        result = cloud_engine.scale_resources(target_capacity)
        
        return jsonify({
            'success': True,
            'scaling_result': result
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/metrics')
def get_cloud_metrics():
    """Get detailed cloud metrics"""
    try:
        metrics = cloud_engine.get_cloud_metrics()
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/edge/register', methods=['POST'])
def register_edge_node():
    """Register edge computing node"""
    try:
        data = request.get_json()
        node_id = data.get('node_id')
        node_config = data.get('config', {})
        
        cloud_engine.edge_manager.register_edge_node(node_id, node_config)
        
        return jsonify({
            'success': True,
            'message': f'Edge node {node_id} registered successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/edge/deploy', methods=['POST'])
def deploy_to_edge():
    """Deploy to edge computing node"""
    try:
        data = request.get_json()
        node_id = data.get('node_id')
        deployment_config = data.get('deployment_config', {})
        
        result = cloud_engine.deploy_to_edge(node_id, deployment_config)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/edge/status')
def get_edge_status():
    """Get edge computing status"""
    try:
        status = cloud_engine.edge_manager.get_edge_status()
        
        return jsonify({
            'success': True,
            'edge_status': status
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/process', methods=['POST'])
def process_on_cloud():
    """Process task on cloud compute"""
    try:
        data = request.get_json()
        task_data = data.get('task_data', {})
        
        result = cloud_engine.process_on_cloud(task_data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/load-balancer/add-server', methods=['POST'])
def add_server_to_lb():
    """Add server to load balancer"""
    try:
        data = request.get_json()
        server_url = data.get('server_url')
        weight = data.get('weight', 1)
        
        cloud_engine.load_balancer.add_server(server_url, weight)
        
        return jsonify({
            'success': True,
            'message': f'Server {server_url} added to load balancer'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@cloud_bp.route('/api/cloud/auto-scale/config', methods=['POST'])
def configure_auto_scaling():
    """Configure auto-scaling parameters"""
    try:
        data = request.get_json()
        
        if 'min_instances' in data:
            cloud_engine.auto_scaler.min_instances = data['min_instances']
        if 'max_instances' in data:
            cloud_engine.auto_scaler.max_instances = data['max_instances']
        if 'target_cpu' in data:
            cloud_engine.auto_scaler.target_cpu = data['target_cpu']
        
        return jsonify({
            'success': True,
            'message': 'Auto-scaling configuration updated'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})