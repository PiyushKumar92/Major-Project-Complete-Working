"""
Continuous Learning System Routes
Admin routes for managing the continuous learning system
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime, timedelta
import csv
import io
from continuous_learning_system import continuous_learning_system
from learning_integration import learning_integration

learning_bp = Blueprint("learning", __name__, url_prefix="/admin")

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            from flask import abort
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@learning_bp.route("/continuous-learning")
@login_required
@admin_required
def continuous_learning_dashboard():
    """Continuous Learning System Dashboard"""
    try:
        # Get learning system statistics
        learning_stats = continuous_learning_system.get_learning_stats()
        
        # Get learning insights
        learning_insights = learning_integration.get_learning_insights()
        
        # Get adaptive thresholds
        adaptive_thresholds = {
            'case_validation': continuous_learning_system.get_adaptive_threshold('case_validation'),
            'photo_quality': continuous_learning_system.get_adaptive_threshold('photo_quality'),
            'form_completeness': continuous_learning_system.get_adaptive_threshold('form_completeness'),
            'cctv_matching': continuous_learning_system.get_adaptive_threshold('cctv_matching'),
            'case_approval': continuous_learning_system.get_adaptive_threshold('case_approval')
        }
        
        return render_template(
            "admin/continuous_learning.html",
            learning_stats=learning_stats,
            learning_insights=learning_insights,
            adaptive_thresholds=adaptive_thresholds
        )
        
    except Exception as e:
        flash(f'Error loading continuous learning dashboard: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))

@learning_bp.route("/record-feedback", methods=["POST"])
@login_required
@admin_required
def record_admin_feedback():
    """Record admin feedback for learning system"""
    try:
        data = request.get_json()
        case_id = data.get('case_id')
        feedback_type = data.get('feedback_type')
        actual_outcome = data.get('actual_outcome')
        admin_notes = data.get('admin_notes', '')
        
        if not all([case_id, feedback_type, actual_outcome]):
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        # Record feedback
        learning_integration.record_admin_feedback(
            case_id=case_id,
            feedback_type=feedback_type,
            actual_outcome=actual_outcome,
            admin_notes=admin_notes
        )
        
        return jsonify({
            'success': True,
            'message': 'Admin feedback recorded successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@learning_bp.route("/trigger-learning", methods=["POST"])
@login_required
@admin_required
def trigger_learning():
    """Manually trigger learning process"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()
        
        prediction_type = data.get('prediction_type')

        
        if prediction_type:
            continuous_learning_system.learn_from_feedback(prediction_type)
            message = f'Learning triggered for {prediction_type}'
        else:
            continuous_learning_system.learn_from_feedback()
            message = 'Learning triggered for all prediction types'
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@learning_bp.route("/reduce-false-positives", methods=["POST"])
@login_required
@admin_required
def reduce_false_positives():
    """Trigger false positive reduction"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()
        
        prediction_type = data.get('prediction_type', 'case_validation')
        target_fp_rate = float(data.get('target_fp_rate', 0.05))
        
        continuous_learning_system.reduce_false_positives(prediction_type, target_fp_rate)
        
        return jsonify({
            'success': True,
            'message': f'False positive reduction initiated for {prediction_type}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@learning_bp.route("/learning-performance")
@login_required
@admin_required
def learning_performance():
    """Get learning system performance metrics"""
    try:
        stats = continuous_learning_system.get_learning_stats()
        
        # Calculate performance trends
        performance_data = {
            'accuracy_trend': [],
            'threshold_adjustments': [],
            'pattern_usage': []
        }
        
        # Mock trend data (in production, this would come from historical data)
        import random
        for i in range(30):  # Last 30 days
            performance_data['accuracy_trend'].append({
                'date': (datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d'),
                'accuracy': min(1.0, stats['system_accuracy'] + random.uniform(-0.1, 0.1))
            })
        
        return jsonify({
            'success': True,
            'performance': performance_data,
            'current_stats': stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@learning_bp.route("/update-threshold", methods=["POST"])
@login_required
@admin_required
def update_threshold():
    """Manually update adaptive threshold"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()
        
        component = data.get('component')
        new_threshold = float(data.get('threshold', 0))
        
        if not component or new_threshold < 0 or new_threshold > 1:
            return jsonify({'success': False, 'error': 'Invalid component or threshold value'})
        
        # Update threshold
        continuous_learning_system.thresholds[component] = new_threshold
        
        # Save to database
        import json
        import sqlite3
        db_conn = sqlite3.connect(continuous_learning_system.db_path)
        cursor = db_conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO adaptive_thresholds 
            (component, current_threshold, optimal_threshold, performance_history, 
             last_updated, adjustment_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (component, new_threshold, new_threshold, json.dumps([]), 
              datetime.now(), 1))
        
        db_conn.commit()
        db_conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Threshold for {component} updated to {new_threshold}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@learning_bp.route("/export-learning-data")
@login_required
@admin_required
def export_learning_data():
    """Export learning system data"""
    try:
        stats = continuous_learning_system.get_learning_stats()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Component', 'Metric', 'Value', 'Export Date'])
        
        # Write feedback stats
        for feedback in stats['feedback_stats']:
            writer.writerow([feedback['type'], 'Total Feedback', feedback['count'], datetime.now().strftime('%Y-%m-%d')])
            writer.writerow([feedback['type'], 'Accuracy', f"{feedback['accuracy']:.3f}", datetime.now().strftime('%Y-%m-%d')])
        
        # Write threshold data
        for threshold in stats['thresholds']:
            writer.writerow([threshold['component'], 'Current Threshold', f"{threshold['threshold']:.3f}", datetime.now().strftime('%Y-%m-%d')])
            writer.writerow([threshold['component'], 'Adjustments Made', threshold['adjustments'], datetime.now().strftime('%Y-%m-%d')])
        
        # Write pattern data
        for pattern in stats['patterns']:
            writer.writerow([pattern['type'], 'Pattern Count', pattern['count'], datetime.now().strftime('%Y-%m-%d')])
            writer.writerow([pattern['type'], 'Success Rate', f"{pattern['success_rate']:.3f}", datetime.now().strftime('%Y-%m-%d')])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'learning_system_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        flash(f'Error exporting learning data: {str(e)}', 'error')
        return redirect(url_for('learning.continuous_learning_dashboard'))

@learning_bp.route("/learning-insights")
@login_required
@admin_required
def get_learning_insights():
    """Get detailed learning insights"""
    try:
        insights = learning_integration.get_learning_insights()
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@learning_bp.route("/pattern-analysis")
@login_required
@admin_required
def pattern_analysis():
    """Get pattern analysis data"""
    try:
        import sqlite3
        conn = sqlite3.connect(continuous_learning_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pattern_type, pattern_data, success_rate, usage_count, last_used
            FROM learned_patterns
            ORDER BY success_rate DESC, usage_count DESC
            LIMIT 20
        ''')
        
        patterns = []
        for row in cursor.fetchall():
            import json
            pattern_data = json.loads(row[1]) if row[1] else {}
            patterns.append({
                'type': row[0],
                'data': pattern_data,
                'success_rate': row[2],
                'usage_count': row[3],
                'last_used': row[4]
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'patterns': patterns
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@learning_bp.route("/system-learning-status")
@login_required
@admin_required
def system_learning_status():
    """Get current learning system status"""
    try:
        stats = continuous_learning_system.get_learning_stats()
        
        status = {
            'learning_active': stats['total_feedback'] > 10,
            'accuracy_good': stats['system_accuracy'] > 0.8,
            'patterns_learned': len(stats['patterns']) > 0,
            'thresholds_optimized': len(stats['thresholds']) > 0,
            'recent_activity': stats['total_feedback'] > 0
        }
        
        overall_health = sum(status.values()) / len(status)
        
        return jsonify({
            'success': True,
            'status': status,
            'overall_health': overall_health,
            'health_percentage': int(overall_health * 100)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})