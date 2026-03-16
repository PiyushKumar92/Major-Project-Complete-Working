from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, make_response, send_file, abort
from flask_login import login_required, current_user
from functools import wraps
from __init__ import db
from models import User, Case, SystemLog, AdminMessage, Announcement, BlogPost, FAQ, AISettings, Sighting, ContactMessage, ChatRoom, ChatMessage, SurveillanceFootage, LocationMatch, PersonDetection, Notification, IntelligentFootageAnalysis, PersonTrackingResult, BehavioralEvent, AppearanceChangeEvent, CrowdAnalysisResult, PersonProfile, RecognitionMatch
from system_monitor import system_monitor, get_system_status
from security_automation import security_automation, get_security_status
from werkzeug.utils import secure_filename
import os
import cv2
import io
import logging
from sqlalchemy import func, desc, and_, or_, case
from datetime import datetime, timedelta, date
import csv
import io
import json
from models import get_ist_now

# Setup logger
logger = logging.getLogger(__name__)

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)

    return decorated_function


@admin_bp.route("/dashboard")
@login_required
@admin_required
def dashboard():
    # Comprehensive status statistics using status system
    from comprehensive_status_system import get_dashboard_status_counts, ALL_CASE_STATUSES
    
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    
    # Get all cases for comprehensive analysis
    all_cases = Case.query.all()
    total_cases = len(all_cases)
    
    # Get comprehensive admin dashboard counts
    admin_status_counts = get_dashboard_status_counts(all_cases, 'admin')
    
    # Individual status counts for admin dashboard
    pending_approvals = admin_status_counts['pending_approval']
    approved_cases = admin_status_counts['approved']
    rejected_cases = admin_status_counts['rejected']
    under_processing = admin_status_counts['under_processing']
    case_solved = admin_status_counts.get('completed', 0)  # Use 'completed' instead of 'case_solved'
    case_over = admin_status_counts.get('completed', 0)  # Use 'completed' for case_over as well
    withdrawn_cases = admin_status_counts['withdrawn']
    active_cases = admin_status_counts['active']
    final_cases = admin_status_counts.get('completed', 0)  # Use 'completed' for final_cases
    
    total_sightings = Sighting.query.count()
    
    # Announcement statistics
    try:
        active_announcements = Announcement.query.filter_by(is_active=True).count()
    except Exception:
        active_announcements = 0
    
    # Status distribution
    status_counts = db.session.query(Case.status, func.count(Case.id)).group_by(Case.status).all()
    
    # Time-based analytics
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
    
    # Convert date objects to formatted strings
    daily_cases = []
    for date_obj, count in daily_cases_raw:
        if date_obj:
            if isinstance(date_obj, str):
                # If it's already a string, parse it first
                try:
                    parsed_date = datetime.strptime(str(date_obj), "%Y-%m-%d")
                    formatted_date = parsed_date.strftime("%m/%d")
                except:
                    formatted_date = str(date_obj)
            else:
                # If it's a date object, format it directly
                formatted_date = date_obj.strftime("%m/%d")
        else:
            formatted_date = ""
        daily_cases.append((formatted_date, count))
    
    # AI Performance metrics
    avg_processing_time = db.session.query(
        func.avg(func.extract('epoch', Case.updated_at - Case.created_at))
    ).filter(Case.status.in_(['Completed', 'Active'])).scalar() or 0
    
    high_confidence_matches = Sighting.query.filter(Sighting.confidence_score > 0.8).count()
    
    # Geographic data (top locations)
    location_stats = (
        db.session.query(
            Case.last_seen_location,
            func.count(Case.id).label('case_count')
        )
        .filter(Case.last_seen_location.isnot(None))
        .group_by(Case.last_seen_location)
        .order_by(desc('case_count'))
        .limit(10)
        .all()
    )
    
    # Recent activity
    recent_logs = SystemLog.query.order_by(desc(SystemLog.timestamp)).limit(10).all()
    
    # Contact messages (with error handling for missing table)
    try:
        unread_messages = ContactMessage.query.filter_by(is_read=False).count()
        recent_contact_messages = ContactMessage.query.order_by(desc(ContactMessage.created_at)).limit(5).all()
    except Exception:
        unread_messages = 0
        recent_contact_messages = []
    
    # Recent cases with error handling
    try:
        recent_cases = Case.query.order_by(desc(Case.created_at)).limit(10).all()
    except Exception:
        recent_cases = []
    
    # Chat statistics
    try:
        active_chats = ChatRoom.query.filter_by(is_active=True).count()
        total_chat_messages = ChatMessage.query.count()
    except Exception:
        active_chats = 0
        total_chat_messages = 0
    
    # AI Analysis Statistics with error handling
    try:
        from models import LocationMatch, PersonDetection, SurveillanceFootage
        # Only count real footage (not test data)
        real_footage_count = SurveillanceFootage.query.filter(
            and_(
                ~SurveillanceFootage.video_path.like('%test%'),
                ~SurveillanceFootage.title.like('%Test%')
            )
        ).count()
        total_location_matches = LocationMatch.query.join(SurveillanceFootage).filter(
            and_(
                ~SurveillanceFootage.video_path.like('%test%'),
                ~SurveillanceFootage.title.like('%Test%')
            )
        ).count()
        successful_detections = LocationMatch.query.join(SurveillanceFootage).filter(
            LocationMatch.person_found == True,
            ~SurveillanceFootage.video_path.like('%test%'),
            ~SurveillanceFootage.title.like('%Test%')
        ).count()
        pending_analysis = LocationMatch.query.join(SurveillanceFootage).filter(
            LocationMatch.status == 'pending',
            ~SurveillanceFootage.video_path.like('%test%'),
            ~SurveillanceFootage.title.like('%Test%')
        ).count()
        processing_analysis = LocationMatch.query.join(SurveillanceFootage).filter(
            LocationMatch.status == 'processing',
            ~SurveillanceFootage.video_path.like('%test%'),
            ~SurveillanceFootage.title.like('%Test%')
        ).count()
    except Exception:
        real_footage_count = 0
        total_location_matches = 0
        successful_detections = 0
        pending_analysis = 0
        processing_analysis = 0
    
    # Mark dashboard notifications as read when admin visits dashboard
    from flask import session
    session['admin_last_check'] = datetime.now().isoformat()
    
    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        active_users=active_users,
        total_cases=total_cases,
        total_sightings=total_sightings,
        status_counts=status_counts,
        daily_cases=daily_cases,
        avg_processing_time=avg_processing_time,
        high_confidence_matches=high_confidence_matches,
        location_stats=location_stats,
        recent_logs=recent_logs,
        unread_messages=unread_messages,
        recent_contact_messages=recent_contact_messages,
        active_chats=active_chats,
        total_chat_messages=total_chat_messages,
        recent_cases=recent_cases,
        real_footage_count=real_footage_count,
        total_location_matches=total_location_matches,
        successful_detections=successful_detections,
        pending_analysis=pending_analysis,
        processing_analysis=processing_analysis,
        active_announcements=active_announcements,
        pending_approvals=pending_approvals,
        pending_cases=pending_approvals,
        # Comprehensive status counts for admin
        admin_status_counts=admin_status_counts,
        approved_cases=approved_cases,
        rejected_cases=rejected_cases,
        under_processing=under_processing,
        case_solved=case_solved,
        case_over=case_over,
        withdrawn_cases=withdrawn_cases,
        active_cases=active_cases,
        final_cases=final_cases,
        system_health=get_system_status(),
        security_status=get_security_status()
    )


@admin_bp.route("/users")
@login_required
@admin_required
def users():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    status_filter = request.args.get('status', '')
    role_filter = request.args.get('role', '')
    sort_by = request.args.get('sort', 'created_at')
    sort_order = request.args.get('order', 'desc')
    
    query = User.query
    
    # Apply search filter with enhanced matching
    if search:
        search_term = f"%{search.strip().lower()}%"
        # Split search term for better matching
        search_words = search.strip().lower().split()
        
        if len(search_words) == 1:
            # Single word search - use ilike for partial matching
            query = query.filter(or_(
                User.username.ilike(search_term),
                User.email.ilike(search_term)
            ))
        else:
            # Multiple words - search each word
            conditions = []
            for word in search_words:
                word_term = f"%{word}%"
                conditions.extend([
                    User.username.ilike(word_term),
                    User.email.ilike(word_term)
                ])
            query = query.filter(or_(*conditions))
    
    # Apply status filter
    if status_filter == 'active':
        query = query.filter(User.is_active == True)
    elif status_filter == 'inactive':
        query = query.filter(User.is_active == False)
    
    # Apply role filter
    if role_filter == 'admin':
        query = query.filter(User.is_admin == True)
    elif role_filter == 'user':
        query = query.filter(User.is_admin == False)
    
    # Apply sorting
    if sort_by == 'username':
        sort_column = User.username
    elif sort_by == 'email':
        sort_column = User.email
    elif sort_by == 'last_login':
        sort_column = User.last_login
    else:
        sort_column = User.created_at
    
    if sort_order == 'asc':
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())
    
    users = query.paginate(page=page, per_page=20, error_out=False)
    
    # Calculate statistics
    total_cases_on_page = sum(len(user.cases) for user in users.items)
    active_users = User.query.filter(User.last_login.isnot(None)).count()
    admin_users = User.query.filter(User.is_admin == True).count()
    
    # Debug info for search
    debug_info = None
    if search:
        debug_info = {
            'search_term': search,
            'total_found': users.total if users else 0,
            'available_users': [u.username for u in User.query.all()]
        }
    
    return render_template(
        "admin/users.html", 
        users=users, 
        search=search,
        status_filter=status_filter,
        role_filter=role_filter,
        sort_by=sort_by,
        sort_order=sort_order,
        total_cases=total_cases_on_page,
        active_users=active_users,
        admin_users=admin_users,
        moment=datetime,
        debug_info=debug_info
    )


@admin_bp.route("/users/<int:user_id>/toggle_admin", methods=["POST"])
@login_required
@admin_required
def toggle_admin(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("❌ Cannot modify your own admin status", "error")
        return redirect(url_for("admin.users"))

    old_status = user.is_admin
    user.is_admin = not user.is_admin
    
    try:
        db.session.commit()
        status_text = "granted" if user.is_admin else "revoked"
        flash(f"✅ Admin privileges {status_text} for {user.username}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ Failed to update admin status: {str(e)}", "error")
    
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    # Check if trying to delete self
    if user_id == current_user.id:
        flash("❌ Cannot delete your own account", "error")
        return redirect(url_for("admin.users"))
    
    try:
        from sqlalchemy import text, create_engine
        
        # CRITICAL: Use no_autoflush to prevent premature flushes
        with db.session.no_autoflush:
            # Get username and case count BEFORE any operations
            result = db.session.execute(
                text("SELECT username FROM `user` WHERE id = :user_id"),
                {'user_id': user_id}
            ).fetchone()
            
            if not result:
                flash("❌ User not found", "error")
                return redirect(url_for("admin.users"))
            
            username = result[0]
            
            # Get case count
            case_count_result = db.session.execute(
                text("SELECT COUNT(*) FROM `cases` WHERE user_id = :user_id"),
                {'user_id': user_id}
            ).fetchone()
            case_count = case_count_result[0] if case_count_result else 0
        
        # CRITICAL: Close current session and create new raw connection
        db.session.close()
        
        # Create direct database connection bypassing SQLAlchemy ORM
        import sqlite3
        conn = sqlite3.connect('instance/app.db')
        cursor = conn.cursor()
        
        print(f"Deleting user {username} (ID: {user_id}) with {case_count} cases...")
        
        # Turn off foreign key constraints
        cursor.execute("PRAGMA foreign_keys = OFF")
        conn.commit()
        
        # Get all case IDs for this user
        cursor.execute("SELECT id FROM `cases` WHERE user_id = ?", (user_id,))
        case_ids = [row[0] for row in cursor.fetchall()]
        
        # Delete ALL case-related data for each case
        for case_id in case_ids:
            print(f"  Deleting case {case_id}...")
            
            # Get location_match IDs
            cursor.execute("SELECT id FROM location_match WHERE case_id = ?", (case_id,))
            match_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete person_detection for these matches
            for match_id in match_ids:
                cursor.execute("DELETE FROM person_detection WHERE location_match_id = ?", (match_id,))
            
            # Get validation IDs
            cursor.execute("SELECT id FROM person_consistency_validation WHERE case_id = ?", (case_id,))
            validation_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete person_consistency_issue
            for validation_id in validation_ids:
                cursor.execute("DELETE FROM person_consistency_issue WHERE validation_id = ?", (validation_id,))
            
            # Delete all case-related tables
            case_tables = [
                'location_match',
                'case_categorization',
                'case_quality_assessment',
                'person_consistency_validation',
                'person_profile',
                'recognition_match',
                'intelligent_footage_analysis',
                'person_tracking_result',
                'behavioral_event',
                'appearance_change_event',
                'crowd_analysis_result',
                'target_image',
                'search_video',
                'sighting',
            ]
            
            for table in case_tables:
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE case_id = ?", (case_id,))
                except Exception as e:
                    print(f"    {table}: {e}")
            
            # Delete the case itself (use backticks for reserved keyword)
            cursor.execute("DELETE FROM `cases` WHERE id = ?", (case_id,))
        
        print(f"  Deleting user-related data...")
        
        # Delete chat_message where user is sender
        cursor.execute("DELETE FROM chat_message WHERE sender_id = ?", (user_id,))
        
        # Delete chat_room where user is participant or admin
        cursor.execute("DELETE FROM chat_room WHERE user_id = ? OR admin_id = ?", (user_id, user_id))
        
        # Delete notifications
        cursor.execute("DELETE FROM notification WHERE user_id = ? OR sender_id = ?", (user_id, user_id))
        
        # Delete admin messages
        cursor.execute("DELETE FROM admin_message WHERE sender_id = ? OR recipient_id = ?", (user_id, user_id))
        
        # Delete system logs for this user
        cursor.execute("DELETE FROM system_log WHERE user_id = ?", (user_id,))
        
        # Log the deletion action
        cursor.execute(
            "INSERT INTO system_log (user_id, action, details, ip_address, timestamp) VALUES (?, 'user_deleted', ?, ?, datetime('now'))",
            (current_user.id, f'Admin {current_user.username} permanently deleted user {username} ({case_count} cases removed)', request.remote_addr or '0.0.0.0')
        )
        
        # FINALLY delete the user (use backticks for reserved keyword)
        cursor.execute("DELETE FROM `user` WHERE id = ?", (user_id,))
        
        # Commit everything
        conn.commit()
        
        print(f"✅ User {username} deleted successfully!")
        
        # Turn foreign keys back ON
        cursor.execute("PRAGMA foreign_keys = ON")
        conn.commit()
        
        # Close connection
        conn.close()
        
        # Recreate SQLAlchemy session
        db.session.remove()
        
        flash(f"✅ User '{username}' and all associated data permanently deleted ({case_count} cases removed)", "success")
        
    except Exception as e:
        # Close connection on error
        try:
            conn.close()
        except:
            pass
        
        # Recreate session
        try:
            db.session.remove()
        except:
            pass
        
        import traceback
        print("\n" + "="*50)
        print("DELETE USER ERROR:")
        traceback.print_exc()
        print("="*50 + "\n")
        
        flash(f"❌ Failed to delete user: {str(e)}", "error")
    
    return redirect(url_for("admin.users"))


# USER MANAGEMENT - Both deactivate and delete options available
@admin_bp.route("/users/<int:user_id>/deactivate", methods=["POST"])
@login_required
@admin_required
def deactivate_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("❌ Cannot deactivate your own account", "error")
        return redirect(url_for("admin.users"))

    username = user.username
    case_count = len(user.cases)
    
    try:
        # Deactivate user instead of deleting
        user.is_active = False
        
        # Log the deactivation
        log = SystemLog(
            user_id=current_user.id,
            action='user_deactivated',
            details=f'Admin {current_user.username} deactivated user {username} ({case_count} cases preserved)',
            ip_address=request.remote_addr
        )
        db.session.add(log)
        db.session.commit()
        
        flash(f"✅ User '{username}' deactivated successfully ({case_count} cases preserved)", "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"❌ Failed to deactivate user: {str(e)}", "error")
    
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>")
@login_required
@admin_required
def user_detail(user_id):
    user = User.query.get_or_404(user_id)
    
    # Calculate total sightings across all user's cases
    total_sightings = sum(len(case.sightings) for case in user.cases)
    
    # Get recent activity logs for this user
    activity_logs = SystemLog.query.filter_by(user_id=user_id).order_by(desc(SystemLog.timestamp)).limit(10).all()
    
    return render_template(
        "admin/user_detail.html", 
        user=user, 
        total_sightings=total_sightings,
        activity_logs=activity_logs
    )


@admin_bp.route("/impersonate/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def impersonate_user(user_id):
    from flask_login import logout_user, login_user
    from flask import session
    
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot impersonate yourself'}), 400
    
    # Store admin user ID for later restoration
    session['impersonating_admin_id'] = current_user.id
    session['is_impersonating'] = True
    
    # Log the impersonation
    log = SystemLog(
        user_id=user_id,
        action='admin_impersonation',
        details=f'Admin {current_user.username} logged in as {user.username}',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()
    
    logout_user()
    login_user(user)
    
    return jsonify({'success': True})


@admin_bp.route("/stop_impersonation", methods=["POST"])
@login_required
def stop_impersonation():
    from flask_login import logout_user, login_user
    from flask import session
    
    if not session.get('is_impersonating'):
        return jsonify({'error': 'Not currently impersonating'}), 400
    
    admin_id = session.get('impersonating_admin_id')
    if not admin_id:
        return jsonify({'error': 'Admin ID not found'}), 400
    
    admin_user = User.query.get(admin_id)
    if not admin_user:
        return jsonify({'error': 'Admin user not found'}), 400
    
    # Log the end of impersonation
    log = SystemLog(
        user_id=admin_id,
        action='admin_impersonation_end',
        details=f'Admin {admin_user.username} stopped impersonating {current_user.username}',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()
    
    logout_user()
    login_user(admin_user)
    
    session.pop('impersonating_admin_id', None)
    session.pop('is_impersonating', None)
    
    return jsonify({'success': True})


@admin_bp.route("/cases")
@login_required
@admin_required
def cases():
    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status', '')
    search = request.args.get('search', '')
    
    query = Case.query
    
    # Apply filters
    if status_filter:
        query = query.filter_by(status=status_filter)
    if search:
        query = query.filter(or_(
            Case.person_name.contains(search),
            Case.last_seen_location.contains(search)
        ))
    
    cases = query.order_by(desc(Case.created_at)).paginate(
        page=page, per_page=12, error_out=False
    )
    
    # Statistics
    total_cases = Case.query.count()
    active_cases = Case.query.filter(Case.status.in_(['Pending Approval', 'Approved', 'Under Processing'])).count()
    completed_cases = Case.query.filter(Case.status.in_(['Case Solved', 'Case Over'])).count()
    
    return render_template(
        "admin/cases.html", 
        cases=cases,
        total_cases=total_cases,
        active_cases=active_cases,
        completed_cases=completed_cases,
        status_filter=status_filter,
        search=search
    )


@admin_bp.route("/cases/<int:case_id>")
@login_required
@admin_required
def case_detail(case_id):
    """Detailed admin view of a specific case with full AI analysis"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get system logs
        logs = (
            SystemLog.query.filter_by(case_id=case_id)
            .order_by(SystemLog.timestamp.desc())
            .all()
        )
        
        # Get AI analysis results with error handling
        location_matches = []
        all_detections = []
        try:
            from models import LocationMatch, PersonDetection
            location_matches = LocationMatch.query.filter_by(case_id=case_id).all()
            
            # Get all detections with match info
            for match in location_matches:
                detections = PersonDetection.query.filter_by(location_match_id=match.id).all()
                for detection in detections:
                    detection.match = match
                    all_detections.append(detection)
            
            # Sort by confidence and timestamp
            all_detections.sort(key=lambda x: (x.confidence_score, x.timestamp), reverse=True)
        except Exception as ai_error:
            logger.warning(f"AI analysis data unavailable for case {case_id}: {str(ai_error)}")
        
        # Calculate analysis statistics
        analysis_stats = {
            'total_matches': len(location_matches),
            'successful_matches': len([m for m in location_matches if m.person_found]),
            'total_detections': len(all_detections),
            'high_confidence_detections': len([d for d in all_detections if d.confidence_score > 0.7]),
            'processing_matches': len([m for m in location_matches if m.status == 'processing']),
            'pending_matches': len([m for m in location_matches if m.status == 'pending'])
        }
        
        return render_template(
            "admin/case_detail.html", 
            case=case, 
            logs=logs,
            location_matches=location_matches,
            detections=all_detections,
            analysis_stats=analysis_stats
        )
        
    except Exception as e:
        logger.error(f"Error loading admin case detail {case_id}: {str(e)}")
        flash("Error loading case details. Please try again.", "error")
        return redirect(url_for("admin.cases"))


# CASE DELETION PERMANENTLY DISABLED
# Admin cannot delete cases - only change status to preserve all data
# Cases can only be withdrawn by users or status changed by admin


@admin_bp.route("/cases/<int:case_id>/approve", methods=["POST"])
@login_required
@admin_required
def approve_case(case_id):
    """Approve case with AI override tracking if needed"""
    case = Case.query.get_or_404(case_id)
    
    # Check if this is overriding an AI rejection
    is_ai_override = False
    if case.status == "Rejected" and case.admin_message and ('AI' in case.admin_message or 'automatically' in case.admin_message.lower()):
        is_ai_override = True
    
    # Get override reason if provided
    override_reason = request.form.get('override_reason', 'Admin manual approval after review')
    
    # Track override if this was an AI decision
    if is_ai_override:
        if not case.admin_notes:
            case.admin_notes = ''
        case.admin_notes += f"\n\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}] AI REJECTION OVERRIDDEN by {current_user.username}:\n"
        case.admin_notes += f"Previous: 'Rejected' (AI Decision)\n"
        case.admin_notes += f"New: 'Approved' (Admin Override)\n"
        case.admin_notes += f"Override Reason: {override_reason}\n"
        
        # Log the override
        log = SystemLog(
            case_id=case_id,
            user_id=current_user.id,
            action='ai_rejection_override',
            details=f'Admin {current_user.username} overrode AI rejection and approved case. Reason: {override_reason}',
            ip_address=request.remote_addr
        )
        db.session.add(log)
    
    # Try to use AI matcher for location matching
    nearby_footage = []
    matches_created = 0
    
    try:
        from ai_location_matcher import ai_matcher
        nearby_footage = ai_matcher.find_nearby_footage(case.last_seen_location)
        
        if not nearby_footage:
            flash(f"⚠️ No CCTV footage available for location '{case.last_seen_location}' or nearby areas. Case approved but AI analysis cannot start until footage is uploaded.", "warning")
        
        matches_created = ai_matcher.process_new_case(case_id)
        
    except Exception as e:
        try:
            from simple_ai_fallback import simple_ai
            matches_created = simple_ai.process_new_case(case_id)
        except Exception as fallback_error:
            print(f"AI processing failed: {str(e)}, Fallback failed: {str(fallback_error)}")
            matches_created = 0
    
    # Update case status
    case.status = "Approved"
    
    # Clear AI message and set admin message
    if is_ai_override:
        case.admin_message = f"Case approved by admin after review. {override_reason}"
    else:
        case.admin_message = "Case approved by admin team. Investigation will begin shortly."
    
    db.session.commit()
    
    # Auto-trigger advanced location matching
    try:
        from advanced_location_matcher import advanced_matcher
        matches_created = advanced_matcher.auto_process_approved_case(case.id)
        
        if matches_created > 0:
            # START AI ANALYSIS FOR ALL PENDING MATCHES
            from models import LocationMatch
            pending_matches = LocationMatch.query.filter_by(
                case_id=case.id,
                status='pending'
            ).all()
            
            analyses_started = 0
            for match in pending_matches:
                try:
                    ai_matcher.analyze_footage_for_person(match.id)
                    analyses_started += 1
                    logger.info(f"✅ Started analysis for match {match.id} (Case {case.id})")
                except Exception as e:
                    logger.error(f"❌ Error analyzing match {match.id}: {e}")
            
            case.status = "Under Processing"
            db.session.commit()
            
            if is_ai_override:
                flash(f"✅ AI rejection overridden! Case approved, {matches_created} matches found, and {analyses_started} analyses started.", "success")
            else:
                flash(f"Case approved! Advanced AI location matching activated - {matches_created} matches found and {analyses_started} analyses started.", "success")
        else:
            if is_ai_override:
                flash("✅ AI rejection overridden! Case approved. Please upload relevant CCTV footage to enable AI analysis.", "success")
            else:
                flash("Case approved! No matching surveillance footage found. Please upload relevant CCTV footage to enable AI analysis.", "warning")
    except Exception as e:
        logger.error(f"Advanced location matching error: {str(e)}")
        if matches_created > 0:
            case.status = "Under Processing"
            db.session.commit()
            flash(f"Case approved! AI analysis started with {matches_created} footage matches.", "success")
        else:
            flash(f"Case approved! Upload relevant CCTV footage to start AI analysis.", "success")
    
    # Notify user about approval
    from models import Notification
    
    notification_message = f"Your {case.case_type.replace('_', ' ').title() if case.case_type else 'Missing Person'} case for {case.person_name} has been approved. Advanced AI analysis will begin shortly."
    
    if is_ai_override:
        notification_message += f"\n\n🤖➡️👤 Note: Our admin team has reviewed and approved your case after careful consideration."
    
    notification = Notification(
        user_id=case.user_id,
        sender_id=current_user.id,
        title=f"✅ Case Approved: {case.person_name}",
        message=notification_message,
        type="success",
        created_at=get_ist_now()
    )
    db.session.add(notification)
    db.session.commit()
    
    return redirect(url_for("admin.case_detail", case_id=case_id))


@admin_bp.route("/cases/<int:case_id>/reject", methods=["POST"])
@login_required
@admin_required
def reject_case(case_id):
    """Reject case with AI override tracking if needed"""
    case = Case.query.get_or_404(case_id)
    
    # Check if this is overriding an AI approval
    is_ai_override = False
    if case.status == "Approved" and case.admin_message and ('AI' in case.admin_message or 'automatically' in case.admin_message.lower()):
        is_ai_override = True
    
    rejection_reason = request.form.get('rejection_reason', 'Case requires revision before processing')
    override_reason = request.form.get('override_reason', rejection_reason)
    
    # Track override if this was an AI decision
    if is_ai_override:
        if not case.admin_notes:
            case.admin_notes = ''
        case.admin_notes += f"\n\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}] AI APPROVAL OVERRIDDEN by {current_user.username}:\n"
        case.admin_notes += f"Previous: 'Approved' (AI Decision)\n"
        case.admin_notes += f"New: 'Rejected' (Admin Override)\n"
        case.admin_notes += f"Override Reason: {override_reason}\n"
        
        # Log the override
        log = SystemLog(
            case_id=case_id,
            user_id=current_user.id,
            action='ai_approval_override',
            details=f'Admin {current_user.username} overrode AI approval and rejected case. Reason: {override_reason}',
            ip_address=request.remote_addr
        )
        db.session.add(log)
    
    # Update case
    case.status = "Rejected"
    case.admin_message = rejection_reason
    db.session.commit()
    
    # Notify user about rejection
    from models import Notification
    
    notification_message = f"Your case for {case.person_name} requires revision before processing.\n\n📝 Admin Feedback:\n{rejection_reason}\n\nPlease review the feedback and resubmit with the necessary corrections."
    
    if is_ai_override:
        notification_message += f"\n\n🤖➡️👤 Note: Our admin team has reviewed this case and determined it needs revision despite initial AI approval."
    
    notification = Notification(
        user_id=case.user_id,
        sender_id=current_user.id,
        title=f"❌ Case Rejected: {case.person_name}",
        message=notification_message,
        type="danger",
        created_at=get_ist_now()
    )
    db.session.add(notification)
    db.session.commit()
    
    if is_ai_override:
        flash(f"✅ AI approval overridden! Case for {case.person_name} has been rejected with admin authority.", "success")
    else:
        flash(f"Case for {case.person_name} has been rejected", "success")
    
    return redirect(url_for("admin.case_detail", case_id=case_id))


@admin_bp.route("/cases/<int:case_id>/update-status", methods=["POST"])
@login_required
@admin_required
def update_case_status(case_id):
    """Update case status and investigation outcome with comprehensive AI override tracking"""
    case = Case.query.get_or_404(case_id)
    
    new_status = request.form.get('status')
    investigation_outcome = request.form.get('investigation_outcome')
    investigation_notes = request.form.get('investigation_notes')
    admin_message = request.form.get('admin_message')
    priority = request.form.get('priority')
    ai_override = request.form.get('ai_override') == 'true'
    override_reason = request.form.get('override_reason', '')
    
    old_status = case.status
    
    # Check if this is an AI decision override
    is_ai_decision = False
    if case.admin_message and ('AI' in case.admin_message or 'automatically' in case.admin_message.lower()):
        is_ai_decision = True
    
    # Track AI override for any status change from AI decisions
    if ai_override or (is_ai_decision and new_status != old_status):
        if not override_reason:
            override_reason = 'Admin manual review and decision'
        
        # Initialize admin notes if empty
        if not case.admin_notes:
            case.admin_notes = ''
        
        # Add override tracking
        case.admin_notes += f"\n\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}] ADMIN OVERRIDE by {current_user.username}:\n"
        case.admin_notes += f"Previous Status: '{old_status}' {'(AI Decision)' if is_ai_decision else '(Manual)'}\n"
        case.admin_notes += f"New Status: '{new_status}' (Admin Decision)\n"
        case.admin_notes += f"Override Reason: {override_reason}\n"
        
        # Log the override in system logs
        log = SystemLog(
            case_id=case_id,
            user_id=current_user.id,
            action='ai_decision_override' if is_ai_decision else 'admin_status_change',
            details=f'Admin {current_user.username} changed case status: {old_status} → {new_status}. Reason: {override_reason}',
            ip_address=request.remote_addr
        )
        db.session.add(log)
    
    # Update case fields
    case.status = new_status
    case.investigation_outcome = investigation_outcome if investigation_outcome else None
    case.investigation_notes = investigation_notes if investigation_notes else None
    case.priority = priority
    case.updated_at = datetime.utcnow()
    
    # Handle admin message - clear AI message if admin is overriding
    if admin_message:
        case.admin_message = admin_message
    elif ai_override or is_ai_decision:
        # Clear AI message and replace with admin message
        case.admin_message = f"Status updated by admin after review. {override_reason if override_reason else 'Manual admin decision.'}"
    
    # Set completion time for final status cases
    if new_status in ['Case Solved', 'Case Over']:
        case.completed_at = datetime.utcnow()
        # Ensure admin message is provided for final statuses
        if not case.admin_message or 'AI' in case.admin_message:
            if new_status == 'Case Solved':
                case.admin_message = "Your case has been successfully resolved by our investigation team. Thank you for using our platform."
            else:
                case.admin_message = "Your case investigation has been completed by our team. Thank you for your patience."
    else:
        case.completed_at = None
    
    db.session.commit()
    
    # Notify user about status change
    from models import Notification
    
    # Create comprehensive user notification
    base_messages = {
        'Pending Approval': f'⏳ Your case is now pending admin approval.',
        'Approved': f'✅ Your {case.case_type.replace("_", " ").title() if case.case_type else "Missing Person"} case has been approved and investigation will begin.',
        'Rejected': f'❌ Your case requires revision before it can be processed.',
        'Under Processing': f'🔄 Your case is now under active investigation. Our team is working on it.',
        'Case Solved': f'🎉 Great news! Your case has been resolved successfully.',
        'Case Over': f'🔒 Your case investigation has been completed.',
        'Withdrawn': f'📋 Your case has been withdrawn as requested.'
    }
    
    notification_message = base_messages.get(new_status, f'Your case status has been updated to {new_status}.')
    
    # Add admin message if provided
    if case.admin_message:
        notification_message += f'\n\n📝 Admin Message:\n{case.admin_message}'
    
    # Add override information to user notification
    if ai_override or is_ai_decision:
        if old_status in ['Approved', 'Rejected'] and is_ai_decision:
            notification_message += f"\n\n🤖➡️👤 Note: Our admin team has reviewed and updated the AI decision after careful consideration."
        else:
            notification_message += f"\n\n👤 Note: This decision was made by our admin team after thorough review."
    
    # Only send notification if status actually changed
    if new_status != old_status:
        notification_type = 'success' if new_status in ['Approved', 'Case Solved'] else 'danger' if new_status == 'Rejected' else 'info'
        
        notification = Notification(
            user_id=case.user_id,
            sender_id=current_user.id,
            title=f'📋 Case Status Update: {case.person_name}',
            message=notification_message,
            type=notification_type,
            created_at=get_ist_now()
        )
        db.session.add(notification)
        db.session.commit()
    
    # Success message for admin
    if ai_override or is_ai_decision:
        flash(f'✅ AI decision overridden! Case status changed to "{new_status}" with admin authority.', 'success')
    else:
        flash(f'✅ Case status updated to "{new_status}" successfully!', 'success')
    
    return redirect(url_for('admin.case_detail', case_id=case_id))


@admin_bp.route("/cases/<int:case_id>/requeue", methods=["POST"])
@login_required
@admin_required
def requeue_case(case_id):
    case = Case.query.get_or_404(case_id)
    case.status = "Under Processing"
    case.completed_at = None
    db.session.commit()

    try:
        from tasks import process_case
        process_case.delay(case_id)
    except:
        pass  # Continue even if task queue fails

    flash(f"Case for {case.person_name} re-queued for processing")
    return redirect(url_for("admin.case_detail", case_id=case_id))


# ===== ADVANCED ADMIN FEATURES =====

# Data Export Routes
@admin_bp.route("/export/users")
@login_required
@admin_required
def export_users():
    users = User.query.all()
    output = io.StringIO()
    writer = csv.writer(output)
    
    # CSV headers
    writer.writerow(['ID', 'Username', 'Email', 'Is Admin', 'Active', 'Cases Count', 'Created At', 'Last Login', 'Location'])
    
    for user in users:
        writer.writerow([
            user.id,
            user.username,
            user.email,
            user.is_admin,
            user.is_active,
            len(user.cases),
            user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never',
            user.location or 'Not specified'
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'users_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@admin_bp.route("/export/cases")
@login_required
@admin_required
def export_cases():
    cases = Case.query.all()
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Person Name', 'Age', 'Status', 'Priority', 'Creator', 'Location', 'Sightings', 'Created At'])
    
    for case in cases:
        writer.writerow([
            case.id,
            case.person_name,
            case.age or 'Unknown',
            case.status,
            case.priority,
            case.creator.username,
            case.last_seen_location or 'Not specified',
            case.total_sightings,
            case.created_at.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'cases_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


# Analytics Routes
@admin_bp.route("/analytics")
@login_required
@admin_required
def analytics():
    try:
        # Basic statistics
        total_cases = Case.query.count()
        total_users = User.query.count()
        total_sightings = Sighting.query.count()
        
        # Case status distribution - convert to tuples
        processing_results = db.session.query(
            Case.status,
            func.count(Case.id).label('count')
        ).group_by(Case.status).all()
        
        # Convert Row objects to tuples
        processing_stats = [(row[0], row[1]) for row in processing_results]
        
        # Simple confidence distribution without complex CASE statements
        confidence_distribution = []
        try:
            very_high = Sighting.query.filter(Sighting.confidence_score >= 0.90).count()
            high = Sighting.query.filter(and_(Sighting.confidence_score >= 0.80, Sighting.confidence_score < 0.90)).count()
            medium = Sighting.query.filter(and_(Sighting.confidence_score >= 0.60, Sighting.confidence_score < 0.80)).count()
            low = Sighting.query.filter(and_(Sighting.confidence_score >= 0.40, Sighting.confidence_score < 0.60)).count()
            very_low = Sighting.query.filter(Sighting.confidence_score < 0.40).count()
            
            confidence_distribution = [
                ('Very High (90%+)', very_high),
                ('High (80-89%)', high),
                ('Medium (60-79%)', medium),
                ('Low (40-59%)', low),
                ('Very Low (<40%)', very_low)
            ]
        except Exception:
            confidence_distribution = [('No Data', 0)]
        
        # Geographic data - convert to tuples for template compatibility
        location_data = []
        try:
            location_results = db.session.query(
                Case.last_seen_location,
                func.count(Case.id).label('case_count')
            ).filter(Case.last_seen_location.isnot(None)).group_by(Case.last_seen_location).all()
            
            # Convert Row objects to tuples
            location_data = [(row[0], row[1]) for row in location_results]
        except Exception:
            location_data = []
        
        # AI Performance metrics
        ai_stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'pending_analysis': 0,
            'avg_confidence': 0.0
        }
        
        try:
            from models import LocationMatch
            ai_stats['total_matches'] = LocationMatch.query.count()
            ai_stats['successful_matches'] = LocationMatch.query.filter_by(person_found=True).count()
            ai_stats['pending_analysis'] = LocationMatch.query.filter_by(status='pending').count()
            
            avg_conf = db.session.query(func.avg(Sighting.confidence_score)).scalar()
            ai_stats['avg_confidence'] = round(avg_conf or 0.0, 2)
        except Exception:
            pass
        
        return render_template(
            "admin/analytics.html",
            total_cases=total_cases,
            total_users=total_users,
            total_sightings=total_sightings,
            processing_stats=processing_stats,
            confidence_distribution=confidence_distribution,
            location_data=location_data,
            ai_stats=ai_stats
        )
        
    except Exception as e:
        flash(f"Error loading analytics: {str(e)}", "error")
        return redirect(url_for("admin.dashboard"))


# User Messaging System
@admin_bp.route("/messages")
@login_required
@admin_required
def messages():
    sent_messages = AdminMessage.query.filter_by(sender_id=current_user.id).order_by(desc(AdminMessage.created_at)).all()
    return render_template("admin/messages.html", sent_messages=sent_messages)




@admin_bp.route("/send-message/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def send_message(user_id):
    from models import Notification
    user = User.query.get_or_404(user_id)
    
    subject = request.form.get('subject', 'Admin Message')
    message_content = request.form.get('message', '')
    
    if not message_content.strip():
        flash("Message cannot be empty.", "error")
        return redirect(url_for("admin.users"))
    
    try:
        # Create admin message record
        admin_message = AdminMessage(
            sender_id=current_user.id,
            recipient_id=user_id,
            subject=subject,
            content=message_content
        )
        db.session.add(admin_message)
        
        # Create notification for user
        notification = Notification(
            user_id=user_id,
            sender_id=current_user.id,
            title=f"Message from Admin: {subject}",
            message=message_content,
            type="info",
            created_at=get_ist_now()
        )
        db.session.add(notification)
        db.session.commit()
        
        flash(f"✅ Message sent to {user.username} successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to send message: {str(e)}", "error")
    
    return redirect(url_for("admin.users"))


# Announcement Management
@admin_bp.route("/announcements")
@login_required
@admin_required
def announcements():
    announcements = Announcement.query.order_by(desc(Announcement.created_at)).all()
    return render_template("admin/announcements.html", announcements=announcements)


@admin_bp.route("/announcements/create", methods=["GET", "POST"])
@login_required
@admin_required
def create_announcement():
    # Calculate tomorrow's date
    tomorrow = date.today() + timedelta(days=1)
    
    if request.method == "POST":
        title = request.form.get('title')
        content = request.form.get('content')
        type = request.form.get('type', 'info')
        expires_at = request.form.get('expires_at')
        
        announcement = Announcement(
            title=title,
            content=content,
            type=type,
            created_by=current_user.id,
            created_at=get_ist_now(),
            expires_at=datetime.strptime(expires_at, '%Y-%m-%d') if expires_at else None
        )
        db.session.add(announcement)
        db.session.flush()  # Get announcement ID
        
        # Create notifications for all non-admin users
        from models import Notification
        regular_users = User.query.filter_by(is_admin=False).all()
        
        for user in regular_users:
            notification = Notification(
                user_id=user.id,
                sender_id=current_user.id,
                title=f"📢 New Announcement: {title}",
                message=content,
                type=type,
                created_at=get_ist_now()
            )
            db.session.add(notification)
        
        db.session.commit()
        
        flash(f"Announcement created and sent to {len(regular_users)} users!", "success")
        return redirect(url_for("admin.announcements"))
    
    return render_template("admin/create_announcement.html", tomorrow=tomorrow)


@admin_bp.route("/announcements/<int:announcement_id>/toggle", methods=["POST"])
@login_required
@admin_required
def toggle_announcement(announcement_id):
    announcement = Announcement.query.get_or_404(announcement_id)
    announcement.is_active = not announcement.is_active
    db.session.commit()
    
    status = "activated" if announcement.is_active else "deactivated"
    flash(f"Announcement {status} successfully!", "success")
    return redirect(url_for("admin.announcements"))


# ANNOUNCEMENT DELETION DISABLED - Only deactivate
@admin_bp.route("/announcements/<int:announcement_id>/archive", methods=["POST"])
@login_required
@admin_required
def archive_announcement(announcement_id):
    """Archive an announcement instead of deleting"""
    try:
        announcement = Announcement.query.get_or_404(announcement_id)
        title = announcement.title
        
        # Deactivate instead of delete
        announcement.is_active = False
        db.session.commit()
        
        flash(f"Announcement '{title}' has been archived (data preserved).", "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"Error archiving announcement: {str(e)}", "error")
    
    return redirect(url_for("admin.announcements"))


# AI Settings Management
@admin_bp.route("/ai-settings")
@login_required
@admin_required
def ai_settings():
    settings = AISettings.query.all()
    
    # Initialize default settings if none exist
    if not settings:
        default_settings = [
            ('confidence_threshold', '0.7', 'Minimum confidence score for matches'),
            ('max_processing_time', '300', 'Maximum processing time per video (seconds)'),
            ('face_detection_model', 'hog', 'Face detection model (hog/cnn)'),
            ('enable_clothing_analysis', 'true', 'Enable clothing-based matching')
        ]
        
        for name, value, desc in default_settings:
            setting = AISettings(setting_name=name, setting_value=value, description=desc, updated_by=current_user.id)
            db.session.add(setting)
        
        db.session.commit()
        settings = AISettings.query.all()
    
    return render_template("admin/ai_settings.html", settings=settings)


@admin_bp.route("/ai-settings", methods=["POST"])
@login_required
@admin_required
def update_ai_settings():
    """Handle AI settings form submission"""
    try:
        updated_count = 0
        
        for setting_id, value in request.form.items():
            if setting_id.startswith('setting_'):
                setting_id = setting_id.replace('setting_', '')
                setting = AISettings.query.get(setting_id)
                if setting:
                    # Handle checkbox values (they only appear in form if checked)
                    if setting.setting_name.startswith('enable_'):
                        setting.setting_value = 'true'
                    else:
                        setting.setting_value = value
                    
                    setting.updated_by = current_user.id
                    setting.updated_at = datetime.utcnow()
                    updated_count += 1
        
        # Handle unchecked checkboxes (they don't appear in form data)
        all_settings = AISettings.query.all()
        for setting in all_settings:
            if setting.setting_name.startswith('enable_'):
                checkbox_name = f'setting_{setting.id}'
                if checkbox_name not in request.form:
                    setting.setting_value = 'false'
                    setting.updated_by = current_user.id
                    setting.updated_at = datetime.utcnow()
                    updated_count += 1
        
        db.session.commit()
        flash(f"✅ AI settings updated successfully! {updated_count} settings modified.", "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"❌ Error updating AI settings: {str(e)}", "error")
    
    return redirect(url_for("admin.ai_settings"))


# Content Management
@admin_bp.route("/content")
@login_required
@admin_required
def content_management():
    blog_posts = BlogPost.query.order_by(desc(BlogPost.created_at)).limit(5).all()
    faqs = FAQ.query.order_by(FAQ.order, FAQ.id).all()
    return render_template("admin/content_management.html", blog_posts=blog_posts, faqs=faqs)


@admin_bp.route("/content/faq/create", methods=["GET", "POST"])
@login_required
@admin_required
def create_faq():
    if request.method == "POST":
        question = request.form.get('question')
        answer = request.form.get('answer')
        category = request.form.get('category', 'General')
        order = int(request.form.get('order', 0))
        
        faq = FAQ(
            question=question,
            answer=answer,
            category=category,
            order=order,
            created_by=current_user.id
        )
        db.session.add(faq)
        db.session.commit()
        
        flash("FAQ created successfully!", "success")
        return redirect(url_for("admin.content_management"))
    
    return render_template("admin/create_faq.html")


@admin_bp.route("/contact-messages")
@login_required
@admin_required
def contact_messages():
    page = request.args.get('page', 1, type=int)
    messages = ContactMessage.query.order_by(desc(ContactMessage.created_at)).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template("admin/contact_messages.html", messages=messages)


@admin_bp.route("/contact-messages/<int:message_id>/mark-read", methods=["POST"])
@login_required
@admin_required
def mark_message_read(message_id):
    try:
        message = ContactMessage.query.get_or_404(message_id)
        message.is_read = True
        db.session.commit()
        flash("Message marked as read successfully!", "success")
        return redirect(url_for('admin.contact_messages'))
    except Exception as e:
        db.session.rollback()
        flash("Failed to mark message as read.", "error")
        return redirect(url_for('admin.contact_messages'))


@admin_bp.route("/contact-messages/<int:message_id>/view", methods=["GET"])
@login_required
@admin_required
def view_full_message(message_id):
    message = ContactMessage.query.get_or_404(message_id)
    return jsonify({
        'success': True,
        'message': {
            'id': message.id,
            'name': message.name,
            'email': message.email,
            'subject': message.subject,
            'content': message.message,
            'created_at': message.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'is_read': message.is_read
        }
    })


# CONTACT MESSAGE DELETION DISABLED - Only mark as read
@admin_bp.route("/contact-messages/<int:message_id>/archive", methods=["POST"])
@login_required
@admin_required
def archive_contact_message(message_id):
    message = ContactMessage.query.get_or_404(message_id)
    message.is_read = True  # Mark as read instead of deleting
    db.session.commit()
    flash("Message archived successfully (data preserved)!", "success")
    return redirect(url_for("admin.contact_messages"))


@admin_bp.route("/chats")
@login_required
@admin_required
def admin_chats():
    """Admin chat management"""
    from models import ChatRoom, ChatMessage
    
    # Get all chat rooms with recent activity
    chat_rooms = ChatRoom.query.order_by(desc(ChatRoom.last_message_at)).all()
    
    # Get chat statistics
    total_chats = ChatRoom.query.count()
    active_chats = ChatRoom.query.filter_by(is_active=True).count()
    total_messages = ChatMessage.query.count()
    
    return render_template(
        "admin/chats.html",
        chat_rooms=chat_rooms,
        total_chats=total_chats,
        active_chats=active_chats,
        total_messages=total_messages
    )


@admin_bp.route("/chats/<int:room_id>/close", methods=["POST"])
@login_required
@admin_required
def close_chat(room_id):
    """Close a chat room"""
    from models import ChatRoom
    
    room = ChatRoom.query.get_or_404(room_id)
    room.is_active = False
    db.session.commit()
    
    flash(f"Chat with {room.user.username} has been closed.", "success")
    return redirect(url_for("admin.admin_chats"))


@admin_bp.route("/surveillance-footage")
@login_required
@admin_required
def surveillance_footage():
    """Surveillance footage management"""
    from models import SurveillanceFootage
    
    page = request.args.get('page', 1, type=int)
    # Show all footage including test data
    footage_list = SurveillanceFootage.query.order_by(
        desc(SurveillanceFootage.created_at)
    ).paginate(page=page, per_page=12, error_out=False)
    
    # AI Analysis Statistics - show all footage
    from models import LocationMatch
    total_matches = LocationMatch.query.count()
    successful_detections = LocationMatch.query.filter(
        LocationMatch.person_found == True
    ).count()
    
    return render_template(
        "admin/surveillance_footage.html", 
        footage_list=footage_list,
        total_matches=total_matches,
        successful_detections=successful_detections
    )


@admin_bp.route("/surveillance-footage/upload", methods=["GET", "POST"])
@login_required
@admin_required
def upload_surveillance_footage():
    """Upload new surveillance footage with automatic case matching"""
    from models import SurveillanceFootage, Case, Notification
    from ai_location_matcher import ai_matcher
    import os
    import cv2
    from werkzeug.utils import secure_filename
    
    if request.method == "POST":
        try:
            # Get form data
            title = request.form.get('title')
            description = request.form.get('description', '')
            location_name = request.form.get('location_name')
            location_address = request.form.get('location_address')
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')
            date_recorded = request.form.get('date_recorded')
            camera_type = request.form.get('camera_type')
            quality = request.form.get('quality')
            
            # Handle multiple file uploads (check both field names)
            files = request.files.getlist('video_files') or request.files.getlist('video_file')
            if not files or all(f.filename == '' for f in files):
                flash('No video files selected', 'error')
                return redirect(request.url)
            
            processed_files = []
            for i, file in enumerate(files):
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"surveillance_{timestamp}_{i+1}_{filename}"
                    
                    # Create surveillance directory
                    surveillance_dir = os.path.join('static', 'surveillance')
                    os.makedirs(surveillance_dir, exist_ok=True)
                    
                    file_path = os.path.join(surveillance_dir, filename)
                    file.save(file_path)
                    
                    # Get video metadata
                    cap = cv2.VideoCapture(file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    resolution = f"{width}x{height}"
                    file_size = os.path.getsize(file_path)
                    cap.release()
                    
                    # Create database entry
                    footage_title = f"{title} - Part {i+1}" if len(files) > 1 else title
                    footage = SurveillanceFootage(
                        title=footage_title,
                        description=description,
                        location_name=location_name,
                        location_address=location_address,
                        latitude=float(latitude) if latitude else None,
                        longitude=float(longitude) if longitude else None,
                        video_path=f"surveillance/{filename}",
                        file_size=file_size,
                        duration=duration,
                        fps=fps,
                        resolution=resolution,
                        quality=quality,
                        date_recorded=datetime.strptime(date_recorded, '%Y-%m-%dT%H:%M') if date_recorded else None,
                        camera_type=camera_type,
                        uploaded_by=current_user.id
                    )
                    
                    db.session.add(footage)
                    db.session.commit()
                    processed_files.append(footage)
            
            # AI: Find matching cases
            newly_processable_cases = []
            pending_cases = Case.query.filter_by(status='Pending Approval').all()
            
            for case in pending_cases:
                if case.last_seen_location and location_name:
                    if (location_name.lower() in case.last_seen_location.lower() or 
                        case.last_seen_location.lower() in location_name.lower()):
                        newly_processable_cases.append(case)
            
            # Notify case owners
            for case in newly_processable_cases:
                notification = Notification(
                    user_id=case.user_id,
                    sender_id=current_user.id,
                    title=f"New CCTV Footage Available: {case.person_name}",
                    message=f"New surveillance footage uploaded for {location_name}",
                    type="info",
                    created_at=get_ist_now()
                )
                db.session.add(notification)
            
            db.session.commit()
            
            # Automatically start AI analysis for uploaded footage
            from ai_location_matcher import ai_matcher
            total_analyses = 0
            
            for footage in processed_files:
                try:
                    # Find and create matches
                    matches_created = ai_matcher.process_new_footage(footage.id)
                    
                    if matches_created > 0:
                        # Start analysis for all pending matches
                        pending_matches = LocationMatch.query.filter_by(
                            footage_id=footage.id,
                            status='pending'
                        ).all()
                        
                        for match in pending_matches:
                            try:
                                # Try background task first, fallback to direct call
                                try:
                                    from tasks import analyze_footage_match
                                    analyze_footage_match.delay(match.id)
                                except:
                                    # Celery not available, run directly
                                    ai_matcher.analyze_footage_for_person(match.id)
                                total_analyses += 1
                            except Exception as e:
                                logger.error(f"Error analyzing match {match.id}: {e}")
                except Exception as e:
                    logger.error(f"Error processing footage {footage.id}: {e}")
            
            if total_analyses > 0:
                flash(f"{len(processed_files)} files uploaded! AI analysis started for {total_analyses} matching cases. Results will be available shortly.", "success")
            else:
                flash(f"{len(processed_files)} files uploaded! No matching cases found for automatic analysis.", "warning")
            
            return redirect(url_for('admin.surveillance_footage'))
                
        except Exception as e:
            flash(f'Error uploading footage: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template("admin/upload_surveillance_footage.html")


@admin_bp.route("/surveillance-footage/<int:footage_id>/analyze", methods=["POST"])
@login_required
@admin_required
def analyze_surveillance_footage(footage_id):
    """Trigger AI analysis for uploaded footage"""
    from flask import request
    
    # Allow both form and JSON requests
    if not request.is_json and not request.form:
        pass  # Allow empty POST
    try:
        from models import SurveillanceFootage, Case, LocationMatch
        from aws_rekognition_matcher import aws_matcher as ai_matcher
        
        footage = SurveillanceFootage.query.get_or_404(footage_id)
        
        # Find matching cases (creates new matches if needed)
        matches_created = ai_matcher.process_new_footage(footage_id)
        
        # Get all matches for this footage (including existing ones)
        all_matches = LocationMatch.query.filter_by(footage_id=footage_id).all()
        
        if not all_matches:
            return jsonify({
                'success': False,
                'error': 'No matching cases found for this location. Make sure there are approved cases with matching location.'
            })
        
        # Reset completed/failed matches to pending for re-analysis
        for match in all_matches:
            if match.status in ['completed', 'failed']:
                match.status = 'pending'
                match.person_found = False
                match.confidence_score = None
                match.detection_count = 0
        db.session.commit()
        
        # Start analysis for all pending matches
        pending_matches = LocationMatch.query.filter_by(
            footage_id=footage_id,
            status='pending'
        ).all()
        
        analysis_started = 0
        for match in pending_matches:
            try:
                # Start analysis in background
                ai_matcher.analyze_footage_for_person(match.id)
                analysis_started += 1
            except Exception as e:
                logger.error(f"Error analyzing match {match.id}: {e}")
        
        return jsonify({
            'success': True,
            'message': f'AI analysis started for {analysis_started} matching cases. Results will be available shortly.'
        })
        
    except Exception as e:
        logger.error(f"Error starting analysis for footage {footage_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@admin_bp.route("/surveillance-footage/<int:footage_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_surveillance_footage(footage_id):
    """Delete surveillance footage and related data"""
    from models import SurveillanceFootage, LocationMatch, PersonDetection
    import os
    
    footage = SurveillanceFootage.query.get_or_404(footage_id)
    
    try:
        # Delete related detections first
        matches = LocationMatch.query.filter_by(footage_id=footage_id).all()
        for match in matches:
            PersonDetection.query.filter_by(location_match_id=match.id).delete()
            db.session.delete(match)
        
        # Delete video file
        video_path = os.path.join('static', footage.video_path)
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Delete footage record
        db.session.delete(footage)
        db.session.commit()
        
        flash('Surveillance footage deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting footage: {str(e)}', 'error')
    
    return redirect(url_for('admin.surveillance_footage'))


@admin_bp.route("/ai-analysis")
@login_required
@admin_required
def ai_analysis():
    """AI Analysis Results Dashboard"""
    from models import LocationMatch, PersonDetection, SurveillanceFootage
    
    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status', '')
    
    # Get location matches with filters
    query = LocationMatch.query
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    matches = query.order_by(desc(LocationMatch.created_at)).paginate(
        page=page, per_page=20, error_out=False
    )
    
    # Statistics
    total_matches = LocationMatch.query.count()
    successful_matches = LocationMatch.query.filter_by(person_found=True).count()
    pending_matches = LocationMatch.query.filter_by(status='pending').count()
    processing_matches = LocationMatch.query.filter_by(status='processing').count()
    
    return render_template(
        "admin/ai_analysis.html",
        matches=matches,
        total_matches=total_matches,
        successful_matches=successful_matches,
        pending_matches=pending_matches,
        processing_matches=processing_matches,
        status_filter=status_filter
    )


@admin_bp.route("/ai-analysis/<int:match_id>")
@login_required
@admin_required
def ai_analysis_detail(match_id):
    """Detailed AI analysis results for a specific match"""
    from models import LocationMatch, PersonDetection
    
    match = LocationMatch.query.get_or_404(match_id)
    detections = PersonDetection.query.filter_by(location_match_id=match_id).order_by(PersonDetection.timestamp).all()
    
    return render_template(
        "admin/ai_analysis_detail.html",
        match=match,
        detections=detections
    )


@admin_bp.route("/ai-analysis/<int:match_id>/reprocess", methods=["POST"])
@login_required
@admin_required
def reprocess_ai_analysis(match_id):
    """Reprocess AI analysis for a specific match"""
    from models import LocationMatch
    from ai_location_matcher import ai_matcher
    
    match = LocationMatch.query.get_or_404(match_id)
    
    # Reset status and start reprocessing
    match.status = 'pending'
    match.person_found = False
    match.confidence_score = None
    match.detection_count = 0
    db.session.commit()
    
    # Start AI analysis
    success = ai_matcher.analyze_footage_for_person(match_id)
    
    if success:
        flash('AI analysis restarted successfully!', 'success')
    else:
        flash('Failed to restart AI analysis.', 'error')
    
    return redirect(url_for('admin.ai_analysis_detail', match_id=match_id))


@admin_bp.route("/location-insights")
@login_required
@admin_required
def location_insights():
    """Location intelligence dashboard for CCTV deployment"""
    from models import SurveillanceFootage, LocationMatch
    
    # Get location statistics
    location_stats = db.session.query(
        Case.last_seen_location,
        func.count(Case.id).label('case_count')
    ).filter(Case.last_seen_location.isnot(None)).group_by(Case.last_seen_location).order_by(desc('case_count')).all()
    
    # Get CCTV coverage data (exclude test data) - FIXED
    cctv_locations = SurveillanceFootage.query.filter(
        and_(
            ~SurveillanceFootage.video_path.like('%test%'),
            ~SurveillanceFootage.title.like('%Test%'),
            SurveillanceFootage.location_name.isnot(None)
        )
    ).with_entities(
        SurveillanceFootage.location_name,
        func.count(SurveillanceFootage.id).label('camera_count')
    ).group_by(SurveillanceFootage.location_name).all()
    
    # Debug print to verify correct count
    print(f"DEBUG: CCTV locations count = {len(cctv_locations)}")
    print(f"DEBUG: CCTV locations data = {cctv_locations}")
    
    # Calculate coverage gaps
    case_locations = set([stat[0].lower() for stat in location_stats])
    cctv_coverage = set([loc[0].lower() for loc in cctv_locations])
    coverage_gaps = case_locations - cctv_coverage
    
    return render_template(
        "admin/location_insights.html",
        location_stats=location_stats,
        cctv_locations=cctv_locations,
        coverage_gaps=coverage_gaps
    )


@admin_bp.route("/surveillance-footage/<int:footage_id>/details")
@login_required
@admin_required
def footage_details(footage_id):
    """Get footage details for modal display"""
    from models import SurveillanceFootage, LocationMatch
    
    footage = SurveillanceFootage.query.get_or_404(footage_id)
    matches = LocationMatch.query.filter_by(footage_id=footage_id).all()
    
    html_content = f"""
    <div class="row">
        <div class="col-md-6">
            <h6>Footage Information</h6>
            <table class="table table-sm">
                <tr><td><strong>Title:</strong></td><td>{footage.title}</td></tr>
                <tr><td><strong>Location:</strong></td><td>{footage.location_name}</td></tr>
                <tr><td><strong>Duration:</strong></td><td>{footage.formatted_duration}</td></tr>
                <tr><td><strong>Quality:</strong></td><td>{footage.quality}</td></tr>
                <tr><td><strong>File Size:</strong></td><td>{footage.formatted_file_size}</td></tr>
                <tr><td><strong>Uploaded:</strong></td><td>{footage.created_at.strftime('%d %b %Y %H:%M')}</td></tr>
            </table>
        </div>
        <div class="col-md-6">
            <h6>AI Analysis Results</h6>
            <p><strong>Total Matches:</strong> {len(matches)}</p>
            <p><strong>Successful Detections:</strong> {len([m for m in matches if m.person_found])}</p>
            <p><strong>Processing Status:</strong> {'Processed' if footage.is_processed else 'Pending'}</p>
        </div>
    </div>
    """
    
    return jsonify({'success': True, 'html': html_content})


@admin_bp.route("/ai-analysis/bulk-start", methods=["POST"])
@login_required
@admin_required
def bulk_start_analysis():
    """Start bulk AI analysis for pending matches"""
    try:
        from models import LocationMatch
        from ai_location_matcher import ai_matcher
        
        pending_matches = LocationMatch.query.filter_by(status='pending').all()
        
        for match in pending_matches:
            # Start analysis in background
            try:
                ai_matcher.analyze_footage_for_person(match.id)
            except Exception as e:
                logger.error(f"Error starting analysis for match {match.id}: {str(e)}")
        
        return jsonify({
            'success': True, 
            'count': len(pending_matches),
            'message': f'Started analysis for {len(pending_matches)} matches'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/trigger-automatic-analysis", methods=["POST"])
@login_required
@admin_required
def trigger_automatic_analysis():
    """Trigger automatic AI analysis for all approved cases with available footage"""
    try:
        # Use AWS Rekognition - fast and accurate
        from aws_rekognition_matcher import aws_matcher as ai_matcher
        logger.info("Using AWS Rekognition for analysis")
        
        # Get all approved cases
        approved_cases = Case.query.filter_by(status='Approved').all()
        
        if not approved_cases:
            return jsonify({
                'success': False,
                'error': 'No approved cases found'
            })
        
        total_matches = 0
        total_analyses = 0
        
        # Process each approved case
        for case in approved_cases:
            try:
                # Create location matches for this case
                matches_created = ai_matcher.process_new_case(case.id)
                total_matches += matches_created
                
                if matches_created > 0:
                    # Start analysis for all pending matches of this case
                    pending_matches = LocationMatch.query.filter_by(
                        case_id=case.id,
                        status='pending'
                    ).all()
                    
                    for match in pending_matches:
                        try:
                            ai_matcher.analyze_footage_for_person(match.id)
                            total_analyses += 1
                        except Exception as e:
                            logger.error(f"Error analyzing match {match.id}: {e}")
                    
                    # Update case status
                    case.status = 'Under Processing'
                    
            except Exception as e:
                logger.error(f"Error processing case {case.id}: {e}")
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Created {total_matches} matches and started {total_analyses} parallel AI analyses',
            'matches': total_matches,
            'analyses': total_analyses
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/ai-analysis/<int:match_id>/reset", methods=["POST"])
@login_required
@admin_required
def reset_analysis_match(match_id):
    """Reset an AI analysis match"""
    try:
        from models import LocationMatch
        
        match = LocationMatch.query.get_or_404(match_id)
        match.status = 'pending'
        match.person_found = False
        match.confidence_score = None
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Match reset successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/ai-analysis/<int:match_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_analysis_match(match_id):
    """Delete an AI analysis match and related detections"""
    try:
        from models import LocationMatch, PersonDetection
        
        match = LocationMatch.query.get_or_404(match_id)
        
        # Delete related detections
        PersonDetection.query.filter_by(location_match_id=match_id).delete()
        
        # Delete match
        db.session.delete(match)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Match deleted successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/detection/<int:detection_id>/<action>", methods=["POST"])
@login_required
@admin_required
def verify_detection(detection_id, action):
    """Verify or reject a detection"""
    try:
        from models import PersonDetection
        
        detection = PersonDetection.query.get_or_404(detection_id)
        
        if action == 'verify':
            detection.verified = True
            detection.verified_by = current_user.id
            message = 'Detection verified successfully'
        elif action == 'reject':
            detection.verified = False
            detection.verified_by = current_user.id
            message = 'Detection rejected successfully'
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
        
        db.session.commit()
        return jsonify({'success': True, 'message': message})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/detection/<int:detection_id>/details", methods=["GET"])
@login_required
@admin_required
def get_detection_details(detection_id):
    """Get detailed information about a detection"""
    try:
        from models import PersonDetection
        
        detection = PersonDetection.query.get_or_404(detection_id)
        
        return jsonify({
            'success': True,
            'detection': {
                'id': detection.id,
                'timestamp': detection.timestamp,
                'formatted_timestamp': detection.formatted_timestamp,
                'confidence_score': detection.confidence_score,
                'face_match_score': detection.face_match_score,
                'clothing_match_score': detection.clothing_match_score,
                'frame_path': detection.frame_path,
                'analysis_method': detection.analysis_method,
                'verified': detection.verified,
                'notes': detection.notes
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@admin_bp.route("/cases/<int:case_id>/review")
@login_required
@admin_required
def case_review(case_id):
    """Detailed case review with nearby footage and approval controls"""
    case = Case.query.get_or_404(case_id)
    
    # Try to find nearby surveillance footage
    nearby_footage = []
    try:
        from ai_location_matcher import ai_matcher
        nearby_footage = ai_matcher.find_nearby_footage(case.last_seen_location)
    except Exception as e:
        print(f"AI matcher error: {str(e)}")
        # Fallback: get all footage
        try:
            nearby_footage = SurveillanceFootage.query.filter_by(is_active=True).all()
        except:
            nearby_footage = []
    
    # Get existing location matches
    location_matches = LocationMatch.query.filter_by(case_id=case_id).all()
    
    # Get intelligent analyses for this case
    intelligent_analyses = IntelligentFootageAnalysis.query.filter_by(case_id=case_id).all()
    
    # Check if case can be approved (always allow approval)
    can_approve = True
    
    return render_template(
        "admin/case_review.html",
        case=case,
        nearby_footage=nearby_footage,
        location_matches=location_matches,
        intelligent_analyses=intelligent_analyses,
        can_approve=can_approve
    )


@admin_bp.route("/analyze-footage/<int:case_id>/<int:footage_id>", methods=["POST"])
@login_required
@admin_required
def analyze_footage(case_id, footage_id):
    """Manually assign case to specific footage for AI analysis"""
    try:
        from ai_location_matcher import ai_matcher
        
        case = Case.query.get_or_404(case_id)
        footage = SurveillanceFootage.query.get_or_404(footage_id)
        
        # Check if match already exists
        existing_match = LocationMatch.query.filter_by(
            case_id=case_id,
            footage_id=footage_id
        ).first()
        
        if not existing_match:
            # Create new location match with manual assignment
            location_match = LocationMatch(
                case_id=case_id,
                footage_id=footage_id,
                match_score=0.9,  # High score for admin-selected footage
                match_type='manual',
                status='pending'
            )
            db.session.add(location_match)
            db.session.commit()
            
            # Log the manual assignment
            log = SystemLog(
                case_id=case_id,
                user_id=current_user.id,
                action='manual_footage_assignment',
                details=f'Admin manually assigned case {case.person_name} to footage {footage.title}',
                ip_address=request.remote_addr
            )
            db.session.add(log)
            db.session.commit()
            
            # Start AI analysis
            success = ai_matcher.analyze_footage_for_person(location_match.id)
            
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'Manual analysis started for {footage.title}'
                })
            else:
                return jsonify({
                    'success': False, 
                    'error': 'Failed to start AI analysis'
                })
        else:
            # Restart analysis for existing match
            existing_match.status = 'pending'
            existing_match.match_type = 'manual'
            db.session.commit()
            
            success = ai_matcher.analyze_footage_for_person(existing_match.id)
            
            if success:
                return jsonify({
                    'success': True, 
                    'message': 'Analysis restarted for existing match'
                })
            else:
                return jsonify({
                    'success': False, 
                    'error': 'Failed to restart analysis'
                })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/cases/<int:case_id>/start-analysis", methods=["POST"])
@login_required
@admin_required
def start_case_analysis(case_id):
    """Start AI analysis for all nearby footage after approval"""
    try:
        from ai_location_matcher import ai_matcher
        
        case = Case.query.get_or_404(case_id)
        
        # Check if case is approved
        if case.status not in ['Approved', 'Under Processing']:
            return jsonify({
                'success': False, 
                'error': 'Case must be approved before starting analysis'
            })
        
        # Find or create location matches for all nearby footage
        matches_created = ai_matcher.process_new_case(case_id)
        
        if matches_created == 0:
            return jsonify({
                'success': False, 
                'error': 'No suitable footage found for this location. Please upload relevant CCTV footage.'
            })
        
        # Update case status
        case.status = 'Under Processing'
        db.session.commit()
        
        # Start analysis for all pending matches
        pending_matches = LocationMatch.query.filter_by(
            case_id=case_id, 
            status='pending'
        ).all()
        
        analysis_started = 0
        for match in pending_matches:
            if ai_matcher.analyze_footage_for_person(match.id):
                analysis_started += 1
        
        # Notify user about analysis start
        from models import Notification
        notification = Notification(
            user_id=case.user_id,
            sender_id=current_user.id,
            title=f"🔍 AI Analysis Started: {case.person_name}",
            message=f"AI analysis has begun for your case. We're analyzing {analysis_started} CCTV footage files from the area. You'll be notified when results are available.",
            type="info",
            created_at=get_ist_now()
        )
        db.session.add(notification)
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': f'Started analysis for {analysis_started} footage matches'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/surveillance-footage/bulk-upload", methods=["GET", "POST"])
@login_required
@admin_required
def bulk_upload_footage():
    """Bulk upload multiple CCTV footage files"""
    if request.method == "POST":
        try:
            uploaded_files = request.files.getlist('video_files')
            location_name = request.form.get('location_name')
            camera_type = request.form.get('camera_type', 'CCTV')
            quality = request.form.get('quality', 'HD')
            
            uploaded_count = 0
            
            for file in uploaded_files:
                if file and file.filename:
                    # Process each file
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"bulk_{timestamp}_{filename}"
                    
                    # Save file
                    surveillance_dir = os.path.join('app', 'static', 'surveillance')
                    os.makedirs(surveillance_dir, exist_ok=True)
                    file_path = os.path.join(surveillance_dir, filename)
                    file.save(file_path)
                    
                    # Get video metadata
                    import cv2
                    cap = cv2.VideoCapture(file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    file_size = os.path.getsize(file_path)
                    cap.release()
                    
                    # Create database entry
                    footage = SurveillanceFootage(
                        title=f"{location_name} - {file.filename}",
                        location_name=location_name,
                        video_path=f"surveillance/{filename}",
                        file_size=file_size,
                        duration=duration,
                        fps=fps,
                        quality=quality,
                        camera_type=camera_type,
                        uploaded_by=current_user.id
                    )
                    
                    db.session.add(footage)
                    uploaded_count += 1
            
            db.session.commit()
            
            # Auto-match with existing cases AND start AI analysis automatically
            from ai_location_matcher import ai_matcher
            total_matches = 0
            total_analyses = 0
            
            for footage in SurveillanceFootage.query.filter_by(location_name=location_name).all():
                matches = ai_matcher.process_new_footage(footage.id)
                total_matches += matches
                
                # Automatically start AI analysis for all matches
                if matches > 0:
                    pending_matches = LocationMatch.query.filter_by(
                        footage_id=footage.id,
                        status='pending'
                    ).all()
                    
                    for match in pending_matches:
                        try:
                            ai_matcher.analyze_footage_for_person(match.id)
                            total_analyses += 1
                        except Exception as e:
                            logger.error(f"Error analyzing match {match.id}: {e}")
            
            flash(f'Successfully uploaded {uploaded_count} files, created {total_matches} AI matches, and started {total_analyses} automatic analyses!', 'success')
            return jsonify({'success': True, 'message': f'Uploaded {uploaded_count} files, AI analysis started for {total_analyses} matches', 'matches': total_matches, 'analyses': total_analyses})
            
        except Exception as e:
            flash(f'Error during bulk upload: {str(e)}', 'error')
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template("admin/bulk_upload_footage.html")


@admin_bp.route("/footage-analysis-results/<int:case_id>")
@login_required
@admin_required
def footage_analysis_results(case_id):
    """View detailed analysis results for a case"""
    case = Case.query.get_or_404(case_id)
    
    # Get all matches and detections
    matches_with_detections = []
    location_matches = LocationMatch.query.filter_by(case_id=case_id).all()
    
    for match in location_matches:
        detections = PersonDetection.query.filter_by(location_match_id=match.id).all()
        matches_with_detections.append({
            'match': match,
            'detections': detections,
            'footage': match.footage
        })
    
    return render_template(
        "admin/footage_analysis_results.html",
        case=case,
        matches_with_detections=matches_with_detections
    )


@admin_bp.route("/detection/<int:detection_id>/note", methods=["POST"])
@login_required
@admin_required
def add_detection_note(detection_id):
    """Add note to a detection"""
    try:
        from models import PersonDetection
        
        detection = PersonDetection.query.get_or_404(detection_id)
        data = request.get_json()
        note = data.get('note', '').strip()
        
        if note:
            detection.notes = note
            db.session.commit()
            return jsonify({'success': True, 'message': 'Note added successfully'})
        else:
            return jsonify({'success': False, 'error': 'Note cannot be empty'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/cases/<int:case_id>/export-results")
@login_required
@admin_required
def export_case_results(case_id):
    """Export case analysis results"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get all matches and detections
        location_matches = LocationMatch.query.filter_by(case_id=case_id).all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # CSV headers
        writer.writerow([
            'Case ID', 'Person Name', 'Footage Title', 'Location', 
            'Match Score', 'Detection Time', 'Confidence Score', 
            'Face Score', 'Clothing Score', 'Method', 'Verified', 'Notes'
        ])
        
        for match in location_matches:
            detections = PersonDetection.query.filter_by(location_match_id=match.id).all()
            
            if detections:
                for detection in detections:
                    writer.writerow([
                        case.id,
                        case.person_name,
                        match.footage.title,
                        match.footage.location_name,
                        f"{match.match_score:.3f}",
                        detection.formatted_timestamp,
                        f"{detection.confidence_score:.3f}",
                        f"{detection.face_match_score:.3f}" if detection.face_match_score else '',
                        f"{detection.clothing_match_score:.3f}" if detection.clothing_match_score else '',
                        detection.analysis_method or '',
                        'Yes' if detection.verified else 'No',
                        detection.notes or ''
                    ])
            else:
                writer.writerow([
                    case.id,
                    case.person_name,
                    match.footage.title,
                    match.footage.location_name,
                    f"{match.match_score:.3f}",
                    'No detections',
                    '', '', '', '', '', ''
                ])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'case_{case_id}_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        flash(f'Error exporting results: {str(e)}', 'error')
        return redirect(url_for('admin.case_review', case_id=case_id))


@admin_bp.route("/system-status")
@login_required
@admin_required
def system_status():
    """Enhanced system status with self-management monitoring"""
    try:
        # Get system health from monitoring
        system_health = get_system_status()
        
        # Get security status
        security_status = get_security_status()
        
        # Database status
        db_status = 'Connected'
        try:
            from sqlalchemy import text
            result = db.session.execute(text('SELECT 1'))
            user_count = User.query.count()
            if user_count >= 0:
                db_status = 'Connected'
            else:
                db_status = 'Error'
        except Exception as e:
            print(f"Database error: {str(e)}")
            db_status = 'Error'
            try:
                db.session.rollback()
            except:
                pass
        
        # Redis status
        redis_status = 'Not Available'
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            redis_status = 'Connected'
        except:
            pass
        
        # AI system status
        ai_status = 'Available'
        try:
            from ai_location_matcher import ai_matcher
            if ai_matcher.face_cascade.empty():
                ai_status = 'Error - Face cascade not loaded'
        except Exception as e:
            ai_status = f'Error - {str(e)}'
        
        # System statistics (exclude test data)
        stats = {
            'total_cases': Case.query.count(),
            'pending_cases': Case.query.filter_by(status='Pending Approval').count(),
            'active_cases': Case.query.filter(Case.status.in_(['Queued', 'Processing', 'Active'])).count(),
            'total_footage': SurveillanceFootage.query.filter(
                ~SurveillanceFootage.video_path.like('%test%')
            ).count(),
            'total_matches': LocationMatch.query.join(SurveillanceFootage).filter(
                ~SurveillanceFootage.video_path.like('%test%')
            ).count(),
            'processing_matches': LocationMatch.query.join(SurveillanceFootage).filter(
                LocationMatch.status == 'processing',
                ~SurveillanceFootage.video_path.like('%test%')
            ).count(),
            'total_detections': PersonDetection.query.join(LocationMatch).join(SurveillanceFootage).filter(
                ~SurveillanceFootage.video_path.like('%test%')
            ).count(),
            'verified_detections': PersonDetection.query.join(LocationMatch).join(SurveillanceFootage).filter(
                PersonDetection.verified == True,
                ~SurveillanceFootage.video_path.like('%test%')
            ).count()
        }
        
        return render_template(
            "admin/system_status.html",
            db_status=db_status,
            redis_status=redis_status,
            ai_status=ai_status,
            stats=stats,
            system_health=system_health,
            security_status=security_status
        )
        
    except Exception as e:
        flash(f'Error loading system status: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route("/ai-validation-dashboard")
@login_required
@admin_required
def ai_validation_dashboard():
    """AI Validation Dashboard for monitoring AI decisions"""
    try:
        # Calculate AI statistics
        total_cases = Case.query.count()
        auto_approved = Case.query.filter_by(status='Approved').filter(
            Case.admin_message.like('%AI%')
        ).count()
        auto_rejected = Case.query.filter_by(status='Rejected').filter(
            Case.admin_message.like('%AI%')
        ).count()
        
        # Count admin overrides from system logs
        overrides = SystemLog.query.filter_by(action='ai_decision_override').count()
        
        # Calculate accuracy (simplified)
        ai_decisions = auto_approved + auto_rejected
        accuracy = round((ai_decisions - overrides) / ai_decisions * 100) if ai_decisions > 0 else 0
        
        ai_stats = {
            'auto_approved': auto_approved,
            'auto_rejected': auto_rejected,
            'overrides': overrides,
            'accuracy': accuracy
        }
        
        # Get recent AI decisions (mock data for now)
        recent_decisions = []
        recent_cases = Case.query.filter(
            Case.admin_message.like('%AI%')
        ).order_by(Case.created_at.desc()).limit(10).all()
        
        for case in recent_cases:
            decision_data = {
                'case': case,
                'ai_decision': 'APPROVE' if case.status == 'Approved' else 'REJECT',
                'confidence': 0.85 if case.status == 'Approved' else 0.45,  # Mock confidence
                'key_issues': 'Photo quality, form completeness' if case.status == 'Rejected' else None,
                'admin_override': 'ai_decision_override' in [log.action for log in SystemLog.query.filter_by(case_id=case.id).all()],
                'created_at': case.created_at
            }
            recent_decisions.append(decision_data)
        
        # AI Settings (mock for now)
        ai_settings = {
            'approval_threshold': 0.7,
            'photo_weight': 0.25,
            'enable_auto_reject': True
        }
        
        return render_template(
            "admin/ai_validation_dashboard.html",
            ai_stats=ai_stats,
            recent_decisions=recent_decisions,
            ai_settings=ai_settings
        )
        
    except Exception as e:
        flash(f'Error loading AI dashboard: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route("/system-report")
@login_required
@admin_required
def system_report():
    """Generate comprehensive system report"""
    try:
        # Generate detailed system report
        report_data = {
            'generated_at': datetime.now(),
            'total_users': User.query.count(),
            'active_users': User.query.filter_by(is_active=True).count(),
            'admin_users': User.query.filter_by(is_admin=True).count(),
            'total_cases': Case.query.count(),
            'pending_cases': Case.query.filter_by(status='Pending Approval').count(),
            'completed_cases': Case.query.filter_by(status='Completed').count(),
            'total_footage': SurveillanceFootage.query.count(),
            'total_matches': LocationMatch.query.count(),
            'successful_matches': LocationMatch.query.filter_by(person_found=True).count(),
            'total_detections': PersonDetection.query.count(),
            'verified_detections': PersonDetection.query.filter_by(verified=True).count()
        }
        
        return render_template("admin/system_report.html", report=report_data)
        
    except Exception as e:
        flash(f'Error generating system report: {str(e)}', 'error')
        return redirect(url_for('admin.system_status'))


@admin_bp.route("/optimize-database", methods=["POST"])
@login_required
@admin_required
def optimize_database():
    """Optimize database performance with automated cleanup"""
    try:
        # Use system monitor for automated cleanup
        system_monitor._cleanup_temporary_files()
        
        # Perform database optimization tasks
        optimizations_performed = []
        
        # Clean up old logs
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        old_logs = SystemLog.query.filter(SystemLog.timestamp < thirty_days_ago).count()
        SystemLog.query.filter(SystemLog.timestamp < thirty_days_ago).delete()
        optimizations_performed.append(f"Cleaned {old_logs} old system logs")
        
        # Clean up old notifications
        old_notifications = db.session.query(Notification).filter(
            Notification.created_at < thirty_days_ago,
            Notification.is_read == True
        ).count()
        db.session.query(Notification).filter(
            Notification.created_at < thirty_days_ago,
            Notification.is_read == True
        ).delete()
        optimizations_performed.append(f"Cleaned {old_notifications} old notifications")
        
        db.session.commit()
        
        # Log optimization in system monitor
        system_monitor.logger.info(f"Database optimization completed: {len(optimizations_performed)} tasks")
        
        return jsonify({
            'success': True,
            'message': 'Database optimization completed with system monitoring',
            'optimizations': optimizations_performed
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/test-system-health", methods=["POST"])
@login_required
@admin_required
def test_system_health():
    """Test complete system health"""
    try:
        test_results = []
        
        # Test system monitoring
        try:
            health = get_system_status()
            if health.get('status') != 'unknown':
                test_results.append("✅ System monitoring active")
            else:
                test_results.append("❌ System monitoring not active")
        except Exception as e:
            test_results.append(f"❌ System monitoring error: {str(e)}")
        
        # Test security automation
        try:
            security = get_security_status()
            if security.get('status') != 'unknown':
                test_results.append("✅ Security automation active")
            else:
                test_results.append("❌ Security automation not active")
        except Exception as e:
            test_results.append(f"❌ Security automation error: {str(e)}")
        
        # Test face recognition
        try:
            import face_recognition
            import numpy as np
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            face_locations = face_recognition.face_locations(test_image)
            test_results.append("✅ Face recognition library working")
        except Exception as e:
            test_results.append(f"❌ Face recognition error: {str(e)}")
        
        # Test OpenCV
        try:
            import cv2
            test_results.append(f"✅ OpenCV version: {cv2.__version__}")
        except Exception as e:
            test_results.append(f"❌ OpenCV error: {str(e)}")
        
        # Test AI matcher
        try:
            from ai_location_matcher import ai_matcher
            test_results.append("✅ AI location matcher loaded")
        except Exception as e:
            test_results.append(f"❌ AI matcher error: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': 'System health test completed',
            'results': test_results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/system-recovery", methods=["POST"])
@login_required
@admin_required
def system_recovery():
    """Execute system recovery actions"""
    try:
        recovery_type = request.json.get('type', 'cache')
        
        if recovery_type == 'cache':
            system_monitor._clear_system_cache()
            message = 'System cache cleared successfully'
        elif recovery_type == 'files':
            system_monitor._cleanup_temporary_files()
            message = 'Temporary files cleaned up successfully'
        elif recovery_type == 'services':
            system_monitor._restart_background_services()
            message = 'Background services restarted successfully'
        else:
            return jsonify({'success': False, 'error': 'Invalid recovery type'})
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/clear-cache", methods=["POST"])
@login_required
@admin_required
def clear_cache():
    """Clear system cache"""
    try:
        # Use system monitor's cache clearing
        system_monitor._clear_system_cache()
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully using system monitor'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/system-self-management")
@login_required
@admin_required
def system_self_management():
    """System Self-Management Dashboard"""
    try:
        # Get system monitoring status
        system_health = get_system_status()
        
        # Get security status
        security_status = get_security_status()
        
        # Get performance trends
        performance_trends = system_monitor.get_performance_trends(24)
        
        return render_template(
            "admin/system_self_management.html",
            system_health=system_health,
            security_status=security_status,
            performance_trends=performance_trends
        )
        
    except Exception as e:
        flash(f'Error loading system management: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route("/start-system-monitoring", methods=["POST"])
@login_required
@admin_required
def start_system_monitoring():
    """Start automated system monitoring"""
    try:
        from system_monitor import start_system_monitoring
        start_system_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'System monitoring started successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/start-security-automation", methods=["POST"])
@login_required
@admin_required
def start_security_automation():
    """Start automated security monitoring"""
    try:
        from security_automation import start_security_automation
        start_security_automation()
        
        return jsonify({
            'success': True,
            'message': 'Security automation started successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/trigger-backup", methods=["POST"])
@login_required
@admin_required
def trigger_backup():
    """Trigger manual backup"""
    try:
        backup_type = request.json.get('type', 'all')
        results = security_automation.perform_automated_backup(backup_type)
        
        return jsonify({
            'success': True,
            'message': f'Backup completed: {backup_type}',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/compliance-check", methods=["POST"])
@login_required
@admin_required
def compliance_check():
    """Run compliance check"""
    try:
        results = security_automation.perform_compliance_check()
        
        return jsonify({
            'success': True,
            'message': 'Compliance check completed',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/generate-system-report", methods=["POST"])
@login_required
@admin_required
def generate_system_report():
    """Generate comprehensive system report"""
    try:
        # Get system health
        system_health = get_system_status()
        
        # Get security status
        security_status = get_security_status()
        
        # Generate report data
        report_data = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_health': system_health,
            'security_status': security_status,
            'system_stats': {
                'total_users': User.query.count(),
                'active_users': User.query.filter_by(is_active=True).count(),
                'admin_users': User.query.filter_by(is_admin=True).count(),
                'total_cases': Case.query.count(),
                'pending_cases': Case.query.filter_by(status='Pending Approval').count(),
                'approved_cases': Case.query.filter_by(status='Approved').count(),
                'processing_cases': Case.query.filter_by(status='Processing').count(),
                'completed_cases': Case.query.filter_by(status='Completed').count(),
                'real_footage': SurveillanceFootage.query.filter(
                    ~SurveillanceFootage.video_path.like('%test%')
                ).count(),
                'location_matches': LocationMatch.query.count(),
                'successful_detections': LocationMatch.query.filter_by(person_found=True).count(),
                'total_detections': PersonDetection.query.count(),
                'verified_detections': PersonDetection.query.filter_by(verified=True).count()
            },
            'performance_metrics': {
                'avg_processing_time': 0,
                'success_rate': 0,
                'verification_rate': 0
            }
        }
        
        # Calculate performance metrics
        total_matches = report_data['system_stats']['location_matches']
        successful = report_data['system_stats']['successful_detections']
        total_detections = report_data['system_stats']['total_detections']
        verified = report_data['system_stats']['verified_detections']
        
        if total_matches > 0:
            report_data['performance_metrics']['success_rate'] = round((successful / total_matches) * 100, 2)
        
        if total_detections > 0:
            report_data['performance_metrics']['verification_rate'] = round((verified / total_detections) * 100, 2)
        
        return jsonify({
            'success': True,
            'message': 'System report generated successfully',
            'report': report_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ===== INTELLIGENT FOOTAGE ANALYSIS ROUTES =====

@admin_bp.route("/cases/<int:case_id>/intelligent-footage-analysis")
@login_required
@admin_required
def intelligent_footage_analysis(case_id):
    """Intelligent footage analysis dashboard for a specific case"""
    case = Case.query.get_or_404(case_id)
    
    # Get all intelligent analyses for this case
    analyses = IntelligentFootageAnalysis.query.filter_by(case_id=case_id).order_by(
        IntelligentFootageAnalysis.analysis_started.desc()
    ).all()
    
    return render_template(
        "admin/intelligent_footage_analysis.html",
        case=case,
        analyses=analyses
    )


@admin_bp.route("/cases/<int:case_id>/start-intelligent-analysis", methods=["POST"])
@login_required
@admin_required
def start_intelligent_analysis(case_id):
    """Start intelligent footage analysis for a case"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get available surveillance footage for this case location
        available_footage = SurveillanceFootage.query.filter(
            SurveillanceFootage.location_name.ilike(f"%{case.last_seen_location}%")
        ).all()
        
        if not available_footage:
            return jsonify({
                'success': False,
                'error': 'No surveillance footage available for this location'
            })
        
        # Start analysis for each footage
        from intelligent_footage_analyzer import intelligent_analyzer
        analyses_started = 0
        
        for footage in available_footage:
            # Check if analysis already exists
            existing_analysis = IntelligentFootageAnalysis.query.filter_by(
                case_id=case_id,
                footage_id=footage.id
            ).first()
            
            if not existing_analysis:
                # Create new analysis record
                analysis = IntelligentFootageAnalysis(
                    case_id=case_id,
                    footage_id=footage.id,
                    analysis_type="comprehensive",
                    status="processing"
                )
                db.session.add(analysis)
                db.session.flush()  # Get analysis ID
                
                # Prepare reference data for analysis
                reference_data = {
                    'face_encodings': [],
                    'visual_features': {},
                    'case_info': {
                        'person_name': case.person_name,
                        'age': case.age,
                        'description': case.description
                    }
                }
                
                # Add face encodings from case images if available
                for target_image in case.target_images:
                    try:
                        import face_recognition
                        import os
                        image_path = os.path.join('app', target_image.image_path)
                        if os.path.exists(image_path):
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            if encodings:
                                reference_data['face_encodings'].extend([enc.tolist() for enc in encodings])
                    except Exception as e:
                        print(f"Error processing reference image: {e}")
                
                # Start comprehensive analysis
                try:
                    footage_path = os.path.join('app', 'static', footage.video_path)
                    results = intelligent_analyzer.analyze_footage_comprehensive(footage_path, reference_data)
                    
                    if 'error' not in results:
                        # Update analysis with results
                        analysis.persons_tracked = results['summary'].get('persons_tracked', 0)
                        analysis.tracking_duration = results['summary'].get('total_duration', 0)
                        analysis.tracking_confidence = results['summary'].get('analysis_confidence', 0)
                        analysis.target_person_found = any(
                            person_data.get('is_target', False) 
                            for frame_data in results['multi_person_tracking'].values()
                            for person_data in frame_data.values()
                        )
                        analysis.appearance_changes_detected = len(results['appearance_changes'])
                        analysis.unusual_behaviors_count = results['summary'].get('behavioral_anomalies', 0)
                        analysis.crowd_scenes_detected = results['summary'].get('crowd_scenes_detected', 0)
                        analysis.max_crowd_density = max(
                            [frame_data.get('crowd_density', 0) for frame_data in results['crowd_analysis'].values()],
                            default=0
                        )
                        analysis.analysis_confidence = results['summary'].get('analysis_confidence', 0)
                        analysis.status = "completed"
                        analysis.analysis_completed = datetime.utcnow()
                        
                        analyses_started += 1
                    else:
                        analysis.status = "failed"
                        analysis.error_message = results.get('error', 'Unknown error')
                        analysis.analysis_completed = datetime.utcnow()
                        
                except Exception as e:
                    analysis.status = "failed"
                    analysis.error_message = str(e)
                    analysis.analysis_completed = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Started intelligent analysis for {analyses_started} footage files'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/intelligent-analysis/<int:analysis_id>/retry", methods=["POST"])
@login_required
@admin_required
def retry_intelligent_analysis(analysis_id):
    """Retry failed intelligent analysis"""
    try:
        analysis = IntelligentFootageAnalysis.query.get_or_404(analysis_id)
        
        # Reset analysis status
        analysis.status = "processing"
        analysis.error_message = None
        analysis.analysis_started = datetime.utcnow()
        analysis.analysis_completed = None
        
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Analysis restarted successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/behavioral-event/<int:event_id>/verify", methods=["POST"])
@login_required
@admin_required
def verify_behavioral_event(event_id):
    """Verify a behavioral event"""
    try:
        event = BehavioralEvent.query.get_or_404(event_id)
        event.verified = True
        event.verified_by = current_user.id
        event.verification_notes = request.form.get('notes', '')
        
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Event verified successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


# ===== AUTONOMOUS CASE RESOLUTION AJAX ROUTES =====

@admin_bp.route("/resolution-decision/<int:case_id>")
@login_required
@admin_required
def get_resolution_decision(case_id):
    """Get resolution decision details for a case"""
    try:
        import sqlite3
        conn = sqlite3.connect('instance/case_resolution.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT resolution_type, confidence_score, evidence_count,
                   auto_closure_eligible, legal_compliance_met, closure_reason,
                   detected_at, status
            FROM case_resolutions 
            WHERE case_id = ? 
            ORDER BY detected_at DESC 
            LIMIT 1
        ''', (case_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                'success': True,
                'decision': {
                    'resolution_type': result[0],
                    'confidence_score': result[1],
                    'evidence_count': result[2],
                    'auto_closure_eligible': result[3],
                    'legal_compliance_met': result[4],
                    'closure_reason': result[5],
                    'detected_at': result[6],
                    'status': result[7]
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No resolution decision found for this case'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/resolution-statistics")
@login_required
@admin_required
def get_resolution_statistics():
    """Get autonomous resolution statistics"""
    try:
        import sqlite3
        conn = sqlite3.connect('instance/case_resolution.db')
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute('SELECT COUNT(*) FROM case_resolutions')
        total_analyses = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM case_resolutions WHERE auto_closure_eligible = 1')
        auto_eligible = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM case_resolutions WHERE legal_compliance_met = 1')
        legal_compliant = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(confidence_score) FROM case_resolutions')
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Get resolution type distribution
        cursor.execute('''
            SELECT resolution_type, COUNT(*) 
            FROM case_resolutions 
            GROUP BY resolution_type
        ''')
        resolution_types = dict(cursor.fetchall())
        
        conn.close()
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_analyses': total_analyses,
                'auto_eligible': auto_eligible,
                'legal_compliant': legal_compliant,
                'avg_confidence': round(avg_confidence, 3),
                'resolution_types': resolution_types,
                'automation_rate': round((auto_eligible / total_analyses * 100), 1) if total_analyses > 0 else 0
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ===== MULTI-MODAL PERSON RECOGNITION ROUTES =====

@admin_bp.route("/cases/<int:case_id>/create-person-profile", methods=["POST"])
@login_required
@admin_required
def create_person_profile(case_id):
    """Create comprehensive person profile with multi-modal recognition"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get case images
        image_paths = []
        for target_image in case.target_images:
            image_path = os.path.join('app', target_image.image_path)
            if os.path.exists(image_path):
                image_paths.append(image_path)
        
        if not image_paths:
            return jsonify({'success': False, 'error': 'No valid images found'})
        
        # Create multi-modal profile
        from multi_modal_recognition import multi_modal_recognizer
        profile_data = multi_modal_recognizer.create_person_profile(case_id, image_paths)
        
        # Check if profile already exists
        existing_profile = PersonProfile.query.filter_by(case_id=case_id).first()
        if existing_profile:
            profile = existing_profile
        else:
            profile = PersonProfile(case_id=case_id)
            db.session.add(profile)
        
        # Update profile data
        import json
        
        # Facial features
        if 'facial_features' in profile_data:
            facial_data = profile_data['facial_features']
            if 'primary_encoding' in facial_data:
                profile.primary_face_encoding = json.dumps(facial_data['primary_encoding'])
            profile.face_quality_score = facial_data.get('quality_score', 0.0)
        
        # Clothing features
        if 'clothing_patterns' in profile_data:
            clothing_data = profile_data['clothing_patterns']
            profile.dominant_colors = json.dumps(clothing_data.get('dominant_colors', []))
            seasonal_cats = clothing_data.get('seasonal_categories', {})
            profile.seasonal_category = list(seasonal_cats.keys())[0] if seasonal_cats else 'unknown'
        
        # Biometric features
        if 'biometric_features' in profile_data:
            biometric_data = profile_data['biometric_features']
            profile.body_measurements = json.dumps(biometric_data.get('average_measurements', {}))
            build_types = biometric_data.get('build_types', {})
            profile.build_type = max(build_types.keys(), key=build_types.get) if build_types else 'unknown'
            profile.biometric_confidence = biometric_data.get('confidence_score', 0.0)
        
        # Overall confidence
        confidence_scores = profile_data.get('confidence_scores', {})
        profile.profile_confidence = confidence_scores.get('overall_confidence', 0.0)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Multi-modal person profile created successfully',
            'profile_id': profile.id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/cases/<int:case_id>/person-profile")
@login_required
@admin_required
def view_person_profile(case_id):
    """View comprehensive person profile"""
    case = Case.query.get_or_404(case_id)
    profile = PersonProfile.query.filter_by(case_id=case_id).first()
    
    if not profile:
        flash('No person profile found. Create one first.', 'warning')
        return redirect(url_for('admin.case_review', case_id=case_id))
    
    return render_template(
        "admin/person_profile.html",
        case=case,
        profile=profile
    )


# ===== AUTONOMOUS CASE RESOLUTION ROUTES =====

@admin_bp.route("/autonomous-case-resolution")
@admin_bp.route("/autonomous-case-resolution")
@login_required
@admin_required
def autonomous_case_resolution():
    """Autonomous Case Resolution Dashboard - Smart case closure system"""
    try:
        # Get resolution candidates with fallback
        candidates = []
        try:
            candidates = get_resolution_candidates()
        except Exception as e:
            logger.warning(f"Could not get resolution candidates: {e}")
            # Fallback: get cases older than 7 days
            from datetime import timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            old_cases = Case.query.filter(
                Case.status.in_(['Under Processing', 'Approved']),
                Case.updated_at < week_ago
            ).limit(10).all()
            candidates = [case.id for case in old_cases]
        
        # Get recent resolution decisions
        recent_decisions = []
        try:
            import sqlite3
            import os
            
            db_path = 'instance/case_resolution.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT case_id, resolution_type, confidence_score, evidence_count,
                           auto_closure_eligible, legal_compliance_met, closure_reason,
                           detected_at, status
                    FROM case_resolutions 
                    ORDER BY detected_at DESC 
                    LIMIT 20
                ''')
                
                for row in cursor.fetchall():
                    case = Case.query.get(row[0])
                    if case:
                        recent_decisions.append({
                            'case': case,
                            'resolution_type': row[1] or 'unknown',
                            'confidence_score': float(row[2]) if row[2] else 0.0,
                            'evidence_count': int(row[3]) if row[3] else 0,
                            'auto_closure_eligible': bool(row[4]),
                            'legal_compliance_met': bool(row[5]),
                            'closure_reason': row[6] or 'No reason provided',
                            'detected_at': row[7] or 'Unknown',
                            'status': row[8] or 'pending'
                        })
                
                conn.close()
                
        except Exception as e:
            logger.warning(f"Error loading resolution decisions: {e}")
        
        # Calculate statistics
        stats = {
            'resolution_candidates': len(candidates),
            'auto_closures': len([d for d in recent_decisions if d.get('auto_closure_eligible', False)]),
            'manual_reviews': len([d for d in recent_decisions if not d.get('auto_closure_eligible', False)]),
            'legal_compliant': len([d for d in recent_decisions if d.get('legal_compliance_met', False)]),
            'avg_confidence': (
                sum([d.get('confidence_score', 0) for d in recent_decisions]) / len(recent_decisions) 
                if recent_decisions else 0.0
            )
        }
        
        # Get case objects for candidates
        candidate_cases = []
        for case_id in candidates[:20]:
            try:
                case = Case.query.get(case_id)
                if case:
                    candidate_cases.append(case)
            except:
                continue
        
        return render_template(
            "admin/autonomous_case_resolution.html",
            candidates=candidates,
            candidate_cases=candidate_cases,
            recent_decisions=recent_decisions,
            stats=stats
        )
        
    except Exception as e:
        logger.error(f'Error in autonomous resolution: {str(e)}')
        # Return basic dashboard with empty data instead of error
        return render_template(
            "admin/autonomous_case_resolution.html",
            candidates=[],
            candidate_cases=[],
            recent_decisions=[],
            stats={'resolution_candidates': 0, 'auto_closures': 0, 'manual_reviews': 0, 'legal_compliant': 0, 'avg_confidence': 0.0}
        )


@admin_bp.route("/analyze-case-resolution/<int:case_id>", methods=["POST"])
@login_required
@admin_required
def analyze_case_resolution_route(case_id):
    """Analyze specific case for autonomous resolution"""
    try:
        result = analyze_case_resolution(case_id)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': f"Resolution analysis completed: {result.get('action', 'analyzed')}",
                'result': result
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Analysis failed')
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/predict-case-outcome/<int:case_id>", methods=["POST"])
@login_required
@admin_required
def predict_case_outcome_route(case_id):
    """Predict outcome for specific case"""
    try:
        result = predict_case_outcome(case_id)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Outcome prediction completed successfully',
                'prediction': result.get('prediction')
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Prediction failed')
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route("/case-outcome-prediction/<int:case_id>")
@login_required
@admin_required
def case_outcome_prediction(case_id):
    """View detailed outcome prediction for a case"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get existing prediction summary
        prediction_summary = get_prediction_summary(case_id)
        
        # If no prediction exists, create one
        if not prediction_summary:
            prediction_result = predict_case_outcome(case_id)
            if prediction_result.get('success'):
                prediction_summary = prediction_result.get('prediction', {})
        
        return render_template(
            "admin/case_outcome_prediction.html",
            case=case,
            prediction=prediction_summary
        )
        
    except Exception as e:
        flash(f'Error loading outcome prediction: {str(e)}', 'error')
        return redirect(url_for('admin.case_detail', case_id=case_id))


@admin_bp.route("/bulk-analyze-resolutions", methods=["POST"])
@login_required
@admin_required
def bulk_analyze_resolutions():
    """Bulk analyze multiple cases for autonomous resolution"""
    try:
        # Get all resolution candidates
        candidates = get_resolution_candidates()
        
        if not candidates:
            return jsonify({
                'success': False,
                'error': 'No cases eligible for resolution analysis'
            })
        
        results = {
            'analyzed': 0,
            'auto_closed': 0,
            'manual_review': 0,
            'continue_investigation': 0,
            'errors': 0
        }
        
        for case_id in candidates[:10]:  # Limit to 10 cases at once
            try:
                result = analyze_case_resolution(case_id)
                results['analyzed'] += 1
                
                if result.get('success'):
                    action = result.get('action', 'unknown')
                    if action == 'auto_closed':
                        results['auto_closed'] += 1
                    elif action == 'queued_for_review':
                        results['manual_review'] += 1
                    elif action == 'continue_investigation':
                        results['continue_investigation'] += 1
                else:
                    results['errors'] += 1
                    
            except Exception as e:
                results['errors'] += 1
                print(f"Error analyzing case {case_id}: {e}")
        
        return jsonify({
            'success': True,
            'message': f"Bulk analysis completed: {results['analyzed']} cases analyzed",
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})




@admin_bp.route('/api/3d/camera-network')
@login_required
@admin_required
def get_3d_camera_network():
    """Get real camera network data for 3D visualization"""
    try:
        from models import SurveillanceFootage, LocationMatch, PersonDetection, Case
        
        # Get all real footage (cameras)
        cameras_data = []
        footage_list = SurveillanceFootage.query.all()
        
        for idx, footage in enumerate(footage_list):
            cameras_data.append({
                'id': f'Camera {idx + 1}',
                'footage_id': footage.id,
                'location': footage.location_name or 'Unknown',
                'status': 'active',
                'position': [idx * 30 - 50, idx * 20, 10],
                'rotation': [0, idx * 45, 0],
                'coverage': 50
            })
        
        # Get all detections
        detections_data = []
        all_detections = PersonDetection.query.order_by(PersonDetection.timestamp.desc()).limit(50).all()
        
        for idx, detection in enumerate(all_detections):
            # Get case_id from location_match
            match = LocationMatch.query.get(detection.location_match_id) if detection.location_match_id else None
            
            detections_data.append({
                'id': detection.id,
                'position': [idx * 10 - 50, idx * 5, 2],
                'confidence': float(detection.confidence_score or 0.5),
                'timestamp': detection.created_at.isoformat() if detection.created_at else datetime.now().isoformat(),
                'case_id': match.case_id if match else None
            })
        
        # Get routes
        routes_data = []
        cases_with_detections = db.session.query(Case.id).join(LocationMatch).join(PersonDetection).distinct().limit(5).all()
        
        for case_id_tuple in cases_with_detections:
            case_id = case_id_tuple[0]
            case_detections = PersonDetection.query.join(LocationMatch).filter(
                LocationMatch.case_id == case_id
            ).order_by(PersonDetection.timestamp).limit(10).all()
            
            if len(case_detections) > 1:
                path = [[idx * 15 - 40, idx * 10, 1] for idx in range(len(case_detections))]
                avg_conf = sum(float(d.confidence_score or 0) for d in case_detections) / len(case_detections)
                routes_data.append({
                    'id': f'Route {case_id}',
                    'case_id': case_id,
                    'path': path,
                    'confidence': avg_conf
                })
        
        return jsonify({
            'success': True,
            'network': {
                'cameras': cameras_data,
                'detections': detections_data,
                'routes': routes_data
            }
        })
        
    except Exception as e:
        logger.error(f'Error getting 3D network data: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@login_required
@admin_required
def get_admin_notifications():
    """Get admin notification count and data"""
    try:
        from flask import session
        
        # Get last check time from session
        last_check = session.get('admin_last_check')
        if last_check:
            last_check = datetime.fromisoformat(last_check)
        else:
            last_check = datetime.min
        
        # Count new cases since last check
        new_cases = Case.query.filter(Case.created_at > last_check).count()
        
        # Count pending approvals
        pending_approvals = Case.query.filter_by(status='Pending Approval').count()
        
        # Count recent rejections that might need review
        recent_rejections = Case.query.filter(
            Case.status == 'Rejected',
            Case.created_at > last_check
        ).count()
        
        # Count unread contact messages
        try:
            unread_messages = ContactMessage.query.filter_by(is_read=False).count()
        except:
            unread_messages = 0
        
        # Count active surveillance footage needing review
        try:
            new_footage = SurveillanceFootage.query.filter(
                SurveillanceFootage.created_at > last_check
            ).count()
        except:
            new_footage = 0
        
        # Calculate total notifications
        total_notifications = (
            new_cases + pending_approvals + recent_rejections + 
            unread_messages + new_footage
        )
        
        return jsonify({
            'count': total_notifications,
            'new_cases': new_cases,
            'pending_approvals': pending_approvals,
            'recent_rejections': recent_rejections,
            'unread_messages': unread_messages,
            'new_footage': new_footage
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/mark_notifications_read', methods=['POST'])
@login_required
@admin_required
def mark_admin_notifications_read():
    """Mark admin notifications as read"""
    try:
        from flask import session
        session['admin_last_check'] = datetime.now().isoformat()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route("/charts-analytics")
@login_required
@admin_required
def charts_analytics():
    """Charts and Analytics page"""
    try:
        # Get comprehensive status statistics
        from comprehensive_status_system import get_dashboard_status_counts, ALL_CASE_STATUSES
        
        all_cases = Case.query.all()
        admin_status_counts = get_dashboard_status_counts(all_cases, 'admin')
        
        # Time-based analytics
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
        
        # Convert date objects to formatted strings
        daily_cases = []
        for date_obj, count in daily_cases_raw:
            if date_obj:
                if isinstance(date_obj, str):
                    try:
                        parsed_date = datetime.strptime(str(date_obj), "%Y-%m-%d")
                        formatted_date = parsed_date.strftime("%m/%d")
                    except:
                        formatted_date = str(date_obj)
                else:
                    formatted_date = date_obj.strftime("%m/%d")
            else:
                formatted_date = ""
            daily_cases.append((formatted_date, count))
        
        # Status distribution
        status_counts = db.session.query(Case.status, func.count(Case.id)).group_by(Case.status).all()
        
        # AI Performance metrics
        try:
            from models import LocationMatch
            ai_stats = {
                'total_matches': LocationMatch.query.count(),
                'successful_matches': LocationMatch.query.filter_by(person_found=True).count(),
                'pending_analysis': LocationMatch.query.filter_by(status='pending').count(),
            }
            avg_conf = db.session.query(func.avg(Sighting.confidence_score)).scalar()
            ai_stats['avg_confidence'] = round(avg_conf or 0.0, 2)
        except Exception:
            ai_stats = {'total_matches': 0, 'successful_matches': 0, 'pending_analysis': 0, 'avg_confidence': 0.0}
        
        return render_template(
            "admin/charts_analytics.html",
            admin_status_counts=admin_status_counts,
            daily_cases=daily_cases,
            status_counts=status_counts,
            ai_stats=ai_stats
        )
        
    except Exception as e:
        flash(f"Error loading charts analytics: {str(e)}", "error")
        return redirect(url_for("admin.dashboard"))


# ===== CONFIDENCE ANALYSIS ROUTES =====

@admin_bp.route("/confidence-analysis")
@login_required
@admin_required
def confidence_analysis():
    """Confidence Distribution and Breakdown Analysis Dashboard"""
    try:
        from confidence_analyzer import confidence_analyzer
        
        # Get filters from request
        case_id = request.args.get('case_id', type=int)
        match_id = request.args.get('match_id', type=int)
        location_id = request.args.get('location_id', type=int)
        
        # Get comprehensive confidence analysis
        analysis = confidence_analyzer.get_confidence_distribution(
            case_id=case_id,
            match_id=match_id,
            location_id=location_id
        )
        
        # Get available filters data
        cases = Case.query.filter_by(status='Approved').all()
        locations = SurveillanceFootage.query.filter(
            ~SurveillanceFootage.video_path.like('%test%')
        ).all()
        
        return render_template(
            "admin/confidence_analysis.html",
            analysis=analysis,
            cases=cases,
            locations=locations,
            selected_case=case_id,
            selected_match=match_id,
            selected_location=location_id
        )
        
    except Exception as e:
        flash(f"Error loading confidence analysis: {str(e)}", "error")
        return redirect(url_for("admin.ai_analysis"))


@admin_bp.route("/api/confidence-analysis")
@login_required
@admin_required
def api_confidence_analysis():
    """API endpoint for confidence analysis data"""
    try:
        from confidence_analyzer import confidence_analyzer
        
        # Get filters from request
        case_id = request.args.get('case_id', type=int)
        match_id = request.args.get('match_id', type=int)
        location_id = request.args.get('location_id', type=int)
        
        # Get analysis
        analysis = confidence_analyzer.get_confidence_distribution(
            case_id=case_id,
            match_id=match_id,
            location_id=location_id
        )
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@admin_bp.route("/confidence-analysis/case/<int:case_id>")
@login_required
@admin_required
def case_confidence_analysis(case_id):
    """Detailed confidence analysis for a specific case"""
    try:
        from confidence_analyzer import confidence_analyzer
        
        case = Case.query.get_or_404(case_id)
        
        # Get case-specific confidence analysis
        analysis = confidence_analyzer.get_case_confidence_summary(case_id)
        
        # Get location matches for this case
        location_matches = LocationMatch.query.filter_by(case_id=case_id).all()
        
        return render_template(
            "admin/case_confidence_analysis.html",
            case=case,
            analysis=analysis,
            location_matches=location_matches
        )
        
    except Exception as e:
        flash(f"Error loading case confidence analysis: {str(e)}", "error")
        return redirect(url_for("admin.case_detail", case_id=case_id))


@admin_bp.route("/confidence-analysis/match/<int:match_id>")
@login_required
@admin_required
def match_confidence_analysis(match_id):
    """Detailed confidence analysis for a specific match"""
    try:
        from confidence_analyzer import confidence_analyzer
        
        match = LocationMatch.query.get_or_404(match_id)
        
        # Get match-specific confidence analysis
        analysis = confidence_analyzer.get_confidence_distribution(match_id=match_id)
        
        # Get detections for this match
        detections = PersonDetection.query.filter_by(location_match_id=match_id).all()
        
        return render_template(
            "admin/match_confidence_analysis.html",
            match=match,
            analysis=analysis,
            detections=detections
        )
        
    except Exception as e:
        flash(f"Error loading match confidence analysis: {str(e)}", "error")
        return redirect(url_for("admin.ai_analysis_detail", match_id=match_id))


@admin_bp.route("/confidence-analysis/export")
@login_required
@admin_required
def export_confidence_analysis():
    """Export confidence analysis report"""
    try:
        from confidence_analyzer import confidence_analyzer
        
        # Get filters
        case_id = request.args.get('case_id', type=int)
        match_id = request.args.get('match_id', type=int)
        
        # Generate report
        report = confidence_analyzer.export_confidence_report(
            case_id=case_id,
            match_id=match_id
        )
        
        # Create CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write report header
        writer.writerow(['Confidence Distribution Analysis Report'])
        writer.writerow(['Generated At', report['generated_at']])
        writer.writerow(['Total Detections', report['summary']['total_detections']])
        writer.writerow(['Mean Confidence', f"{report['summary']['mean_confidence']:.3f}"])
        writer.writerow(['Reliability Score', f"{report['summary']['reliability_score']:.2f}"])
        writer.writerow([])
        
        # Write distribution data
        writer.writerow(['Confidence Range', 'Count', 'Percentage'])
        for range_name, data in report['detailed_analysis']['distribution'].items():
            writer.writerow([data['label'], data['count'], f"{data['percentage']:.1f}%"])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'confidence_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        flash(f"Error exporting confidence analysis: {str(e)}", "error")
        return redirect(url_for("admin.confidence_analysis"))