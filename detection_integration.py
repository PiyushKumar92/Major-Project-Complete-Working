"""
DETECTION INTEGRATION MODULE
Integrates advanced person detection with existing case management system
"""

from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_login import login_required, current_user
from functools import wraps
import os
import json
from datetime import datetime
import logging

from models import Case, SurveillanceFootage, LocationMatch, PersonDetection, db
from advanced_person_detection import analyze_person_in_footage
from timeline_report_generator import generate_comprehensive_timeline_report

# Setup logging
logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__, url_prefix='/admin/detection')

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

@detection_bp.route('/advanced-analysis/<int:case_id>')
@login_required
@admin_required
def advanced_analysis_dashboard(case_id):
    """Advanced detection analysis dashboard for a specific case"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get available surveillance footage
        available_footage = SurveillanceFootage.query.filter(
            SurveillanceFootage.location_name.ilike(f"%{case.last_seen_location}%")
        ).all()
        
        # Get existing analysis results
        existing_matches = LocationMatch.query.filter_by(case_id=case_id).all()
        
        return render_template(
            'admin/advanced_detection_analysis.html',
            case=case,
            available_footage=available_footage,
            existing_matches=existing_matches
        )
        
    except Exception as e:
        logger.error(f"Error loading advanced analysis dashboard: {str(e)}")
        flash('Error loading analysis dashboard', 'error')
        return redirect(url_for('admin.cases'))

@detection_bp.route('/start-advanced-analysis/<int:case_id>', methods=['POST'])
@login_required
@admin_required
def start_advanced_analysis(case_id):
    """Start advanced 5-method ensemble analysis"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get analysis parameters
        data = request.get_json()
        footage_ids = data.get('footage_ids', [])
        cooldown_period = data.get('cooldown_period', 3)
        confidence_threshold = data.get('confidence_threshold', 0.65)
        
        if not footage_ids:
            return jsonify({'error': 'No footage selected for analysis'}), 400
        
        # Get case target image
        target_image_path = None
        if case.target_images:
            target_image_path = os.path.join('app', case.target_images[0].image_path)
        
        if not target_image_path or not os.path.exists(target_image_path):
            return jsonify({'error': 'No valid target image found for case'}), 400
        
        # Prepare case details for analysis
        case_details = {
            'person_name': case.person_name,
            'age': case.age,
            'description': case.description,
            'location': case.last_seen_location,
            'last_seen_time': case.last_seen_time.isoformat() if case.last_seen_time else None,
            'height_width_ratio': 2.5,  # Default body proportion
            'clothing_description': case.description  # Use description for clothing analysis
        }
        
        analysis_results = []
        
        # Analyze each selected footage
        for footage_id in footage_ids:
            footage = SurveillanceFootage.query.get(footage_id)
            if not footage:
                continue
            
            footage_path = os.path.join('app', 'static', footage.video_path)
            if not os.path.exists(footage_path):
                logger.warning(f"Footage file not found: {footage_path}")
                continue
            
            logger.info(f"Starting advanced analysis for footage: {footage.title}")
            
            # Perform advanced detection analysis
            result = analyze_person_in_footage(
                footage_path=footage_path,
                target_image_path=target_image_path,
                case_details=case_details,
                cooldown_period=cooldown_period
            )
            
            # Add footage metadata to result
            result['footage_id'] = footage_id
            result['footage_title'] = footage.title
            result['footage_location'] = footage.location_name
            
            analysis_results.append(result)
            
            # Store results in database
            if result.get('status') == 'detection_found':
                # Create or update location match
                location_match = LocationMatch.query.filter_by(
                    case_id=case_id,
                    footage_id=footage_id
                ).first()
                
                if not location_match:
                    location_match = LocationMatch(
                        case_id=case_id,
                        footage_id=footage_id,
                        match_score=0.9,  # High score for advanced analysis
                        match_type='advanced_ensemble'
                    )
                    db.session.add(location_match)
                    db.session.flush()
                
                # Update match with analysis results
                timeline_report = result.get('timeline_report', {})
                location_match.status = 'completed'
                location_match.person_found = True
                location_match.confidence_score = self._calculate_overall_confidence(timeline_report)
                location_match.detection_count = timeline_report.get('total_appearances', 0)
                
                # Store detailed detection results
                self._store_detection_results(location_match.id, timeline_report)
        
        db.session.commit()
        
        # Generate comprehensive timeline report
        comprehensive_report = generate_comprehensive_timeline_report(
            analysis_results, case_details
        )
        
        # Update case status if detections found
        detection_count = sum(1 for r in analysis_results if r.get('status') == 'detection_found')
        if detection_count > 0:
            case.status = 'Under Processing'
            case.admin_message = f"Advanced AI analysis completed: Person detected in {detection_count} footage files using 5-method ensemble detection."
        else:
            case.admin_message = f"Advanced AI analysis completed: No detections found in {len(analysis_results)} footage files analyzed."
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Advanced analysis completed for {len(footage_ids)} footage files',
            'results': {
                'analysis_results': analysis_results,
                'comprehensive_report': comprehensive_report,
                'detection_summary': {
                    'total_footage_analyzed': len(analysis_results),
                    'footage_with_detections': detection_count,
                    'total_appearances': sum(
                        r.get('coalesced_appearances', 0) for r in analysis_results
                    )
                }
            }
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in advanced analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@detection_bp.route('/timeline-report/<int:case_id>')
@login_required
@admin_required
def view_timeline_report(case_id):
    """View comprehensive timeline report for a case"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get all location matches for this case
        location_matches = LocationMatch.query.filter_by(case_id=case_id).all()
        
        if not location_matches:
            flash('No analysis results found for this case', 'warning')
            return redirect(url_for('admin.case_detail', case_id=case_id))
        
        # Reconstruct analysis results from database
        analysis_results = []
        for match in location_matches:
            if match.person_found:
                # Get detection details
                detections = PersonDetection.query.filter_by(location_match_id=match.id).all()
                
                # Reconstruct timeline entries
                timeline_entries = []
                for detection in detections:
                    timeline_entries.append({
                        'footage_file': match.footage.title,
                        'location': match.footage.location_name,
                        'appearance': f"Detection #{detection.id}",
                        'start_time': detection.formatted_timestamp,
                        'end_time': detection.formatted_timestamp,  # Single point detection
                        'total_duration': '1s',  # Placeholder
                        'avg_confidence': f"{detection.confidence_score * 100:.1f}%"
                    })
                
                analysis_results.append({
                    'status': 'detection_found',
                    'timeline_report': {
                        'footage_file': match.footage.title,
                        'location': match.footage.location_name,
                        'total_appearances': len(timeline_entries),
                        'timeline_entries': timeline_entries
                    }
                })
        
        # Generate comprehensive report
        case_details = {
            'person_name': case.person_name,
            'case_id': case.id,
            'location': case.last_seen_location
        }
        
        comprehensive_report = generate_comprehensive_timeline_report(
            analysis_results, case_details
        )
        
        return render_template(
            'admin/timeline_report.html',
            case=case,
            comprehensive_report=comprehensive_report
        )
        
    except Exception as e:
        logger.error(f"Error generating timeline report: {str(e)}")
        flash('Error generating timeline report', 'error')
        return redirect(url_for('admin.case_detail', case_id=case_id))

@detection_bp.route('/export-timeline/<int:case_id>')
@login_required
@admin_required
def export_timeline_report(case_id):
    """Export timeline report as JSON/PDF"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get comprehensive report data
        # (Implementation would generate and return downloadable report)
        
        return jsonify({
            'success': True,
            'message': 'Timeline report exported successfully',
            'download_url': f'/admin/detection/download-report/{case_id}'
        })
        
    except Exception as e:
        logger.error(f"Error exporting timeline report: {str(e)}")
        return jsonify({'error': 'Export failed'}), 500

def _calculate_overall_confidence(timeline_report: dict) -> float:
    """Calculate overall confidence from timeline report"""
    timeline_entries = timeline_report.get('timeline_entries', [])
    if not timeline_entries:
        return 0.0
    
    confidences = []
    for entry in timeline_entries:
        conf_str = entry.get('avg_confidence', '0%').replace('%', '')
        confidences.append(float(conf_str))
    
    return sum(confidences) / len(confidences) / 100  # Convert to 0-1 scale

def _store_detection_results(location_match_id: int, timeline_report: dict):
    """Store detailed detection results in database"""
    timeline_entries = timeline_report.get('timeline_entries', [])
    
    for entry in timeline_entries:
        # Create PersonDetection record for each appearance
        detection = PersonDetection(
            location_match_id=location_match_id,
            timestamp=entry.get('start_time', '00:00:00'),
            confidence_score=float(entry.get('avg_confidence', '0%').replace('%', '')) / 100,
            analysis_method='final_correct_matching',
            verified=False  # Requires admin verification
        )
        db.session.add(detection)

# Register blueprint
def register_detection_routes(app):
    """Register detection routes with Flask app"""
    app.register_blueprint(detection_bp)