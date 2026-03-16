"""
Routes for Location-based CCTV Footage Matching
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, abort
from flask_login import login_required, current_user
from functools import wraps
from intelligent_location_matcher import location_matcher
from models import Case, SurveillanceFootage, Sighting
from advanced_person_detector import analyze_footage_for_case
from ultra_advanced_person_detector import analyze_footage_ultra_advanced

# Local admin_required decorator to avoid circular import
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

location_bp = Blueprint('location', __name__, url_prefix='/admin/location')

@location_bp.route('/match-footage/<int:case_id>')
@login_required
@admin_required
def match_footage_for_case(case_id):
    """
    Show CCTV footage matches for a specific case
    """
    try:
        matches = location_matcher.find_matching_footage_for_case(case_id)
        
        if 'error' in matches:
            flash(matches['error'], 'error')
            return redirect(url_for('admin.cases'))
        
        # Get suggestions with admin approval settings
        suggestions = location_matcher.get_location_based_suggestions(
            case_id, 
            admin_approval_required=True
        )
        
        return render_template(
            'admin/location_footage_matches.html',
            matches=matches,
            suggestions=suggestions,
            case_id=case_id
        )
        
    except Exception as e:
        flash(f"Error finding footage matches: {str(e)}", 'error')
        return redirect(url_for('admin.cases'))

@location_bp.route('/match-cases/<int:footage_id>')
@login_required
@admin_required
def match_cases_for_footage(footage_id):
    """
    Show case matches for a specific CCTV footage
    """
    try:
        matches = location_matcher.find_matching_cases_for_footage(footage_id)
        
        if 'error' in matches:
            flash(matches['error'], 'error')
            return redirect(url_for('admin.surveillance_footage'))
        
        return render_template(
            'admin/location_case_matches.html',
            matches=matches,
            footage_id=footage_id
        )
        
    except Exception as e:
        flash(f"Error finding case matches: {str(e)}", 'error')
        return redirect(url_for('admin.surveillance_footage'))

@location_bp.route('/api/quick-match/<int:case_id>')
@login_required
@admin_required
def api_quick_match(case_id):
    """
    API endpoint for quick location matching
    """
    try:
        matches = location_matcher.find_matching_footage_for_case(case_id)
        
        if 'error' in matches:
            return jsonify({'success': False, 'error': matches['error']})
        
        # Simplified response for API
        simplified_matches = []
        for match in matches['matches'][:5]:  # Top 5 matches
            simplified_matches.append({
                'footage_id': match['footage_id'],
                'title': match['footage_title'],
                'location': match['location_name'],
                'match_score': round(match['match_score'] * 100, 1),
                'confidence': 'High' if match['match_score'] >= 0.8 else 'Medium' if match['match_score'] >= 0.6 else 'Low'
            })
        
        return jsonify({
            'success': True,
            'case_id': case_id,
            'total_matches': matches['matching_footage_count'],
            'top_matches': simplified_matches
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@location_bp.route('/criteria')
@login_required
@admin_required
def matching_criteria():
    """
    Show location matching criteria and settings
    """
    criteria = location_matcher._get_match_criteria_explanation()
    
    return render_template(
        'admin/location_matching_criteria.html',
        criteria=criteria
    )

@location_bp.route('/analyze-footage/<int:case_id>/<int:footage_id>')
@login_required
@admin_required
def analyze_specific_footage(case_id, footage_id):
    """
    Analyze specific CCTV footage for person detection
    """
    try:
        result = analyze_footage_for_case(case_id, footage_id)
        
        if 'error' in result:
            flash(result['error'], 'error')
            return redirect(url_for('location.match_footage_for_case', case_id=case_id))
        
        flash(f"Analysis complete! Found {result['total_detections']} potential matches.", 'success')
        return redirect(url_for('location.view_analysis_results', case_id=case_id, footage_id=footage_id))
        
    except Exception as e:
        flash(f"Analysis failed: {str(e)}", 'error')
        return redirect(url_for('location.match_footage_for_case', case_id=case_id))

@location_bp.route('/results/<int:case_id>/<int:footage_id>')
@login_required
@admin_required
def view_analysis_results(case_id, footage_id):
    """
    View analysis results for specific footage
    """
    try:
        case = Case.query.get_or_404(case_id)
        footage = SurveillanceFootage.query.get_or_404(footage_id)
        
        sightings = Sighting.query.filter_by(
            case_id=case_id,
            search_video_id=footage_id
        ).order_by(Sighting.timestamp.asc()).all()
        
        grouped_sightings = []
        current_group = []
        last_timestamp = 0
        
        for sighting in sightings:
            if sighting.timestamp - last_timestamp > 5:
                if current_group:
                    grouped_sightings.append(current_group)
                current_group = [sighting]
            else:
                current_group.append(sighting)
            last_timestamp = sighting.timestamp
        
        if current_group:
            grouped_sightings.append(current_group)
        
        return render_template(
            'admin/footage_analysis_results.html',
            case=case,
            footage=footage,
            sightings=sightings,
            grouped_sightings=grouped_sightings,
            total_detections=len(sightings)
        )
        
    except Exception as e:
        flash(f"Error loading results: {str(e)}", 'error')
        return redirect(url_for('admin.cases'))

@location_bp.route('/api/analyze-all/<int:case_id>', methods=['POST'])
@login_required
@admin_required
def api_analyze_all_footage(case_id):
    """
    API to analyze all matching footage for a case
    """
    try:
        matches = location_matcher.find_matching_footage_for_case(case_id)
        
        if 'error' in matches:
            return jsonify({'success': False, 'error': matches['error']})
        
        results = []
        total_detections = 0
        
        for match in matches['matches'][:5]:
            footage_id = match['footage_id']
            analysis_result = analyze_footage_for_case(case_id, footage_id)
            
            if 'error' not in analysis_result:
                results.append({
                    'footage_id': footage_id,
                    'footage_title': match['footage_title'],
                    'detections': analysis_result['total_detections'],
                    'match_score': match['match_score']
                })
                total_detections += analysis_result['total_detections']
        
        return jsonify({
            'success': True,
            'case_id': case_id,
            'analyzed_footage': len(results),
            'total_detections': total_detections,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})