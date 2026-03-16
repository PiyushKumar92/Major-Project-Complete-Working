# Add this route to admin.py

@admin_bp.route("/charts-analytics")
@login_required
@admin_required
def charts_analytics():
    """Dedicated charts and analytics page"""
    try:
        from comprehensive_status_system import get_dashboard_status_counts
        
        all_cases = Case.query.all()
        admin_status_counts = get_dashboard_status_counts(all_cases, 'admin')
        
        status_counts = db.session.query(Case.status, func.count(Case.id)).group_by(Case.status).all()
        
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        daily_cases_raw = (
            db.session.query(
                func.date(Case.created_at).label("date"), 
                func.count(Case.id).label("count")
            )
            .filter(Case.created_at >= thirty_days_ago)
            .group_by(func.date(Case.created_at))
            .all()
        )
        
        daily_cases = []
        for date_obj, count in daily_cases_raw:
            if date_obj:
                formatted_date = date_obj.strftime("%m/%d") if hasattr(date_obj, 'strftime') else str(date_obj)
            else:
                formatted_date = ""
            daily_cases.append((formatted_date, count))
        
        try:
            from models import LocationMatch, PersonDetection
            ai_metrics = {
                'total_matches': LocationMatch.query.count(),
                'successful_matches': LocationMatch.query.filter_by(person_found=True).count(),
                'high_confidence': PersonDetection.query.filter(PersonDetection.confidence_score > 0.8).count(),
                'processing': LocationMatch.query.filter_by(status='processing').count()
            }
        except Exception:
            ai_metrics = {'total_matches': 0, 'successful_matches': 0, 'high_confidence': 0, 'processing': 0}
        
        return render_template(
            "admin/charts_analytics.html",
            status_counts=status_counts,
            daily_cases=daily_cases,
            admin_status_counts=admin_status_counts,
            ai_metrics=ai_metrics
        )
        
    except Exception as e:
        flash(f'Error loading charts analytics: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))