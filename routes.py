import os
from datetime import datetime, timedelta
import pytz
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    abort,
)
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from functools import wraps

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def utc_to_ist(utc_dt):
    """Convert UTC datetime to IST"""
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    return utc_dt.astimezone(IST)

from __init__ import db
from models import User, Case, TargetImage, SearchVideo, Sighting, Announcement, AnnouncementRead
from forms import (
    RegistrationForm,
    LoginForm,
    ForgotPasswordForm,
    ResetPasswordForm,
    NewCaseForm,
    ContactForm,
)

from system_monitor import start_system_monitoring
from security_automation import start_security_automation, process_security_event
from auto_ai_processor import AutoAIProcessor

# Lazy import for enhanced AI - only load when needed
def get_enhanced_ai_processor():
    from enhanced_ai_engine import EnhancedAIProcessor
    return EnhancedAIProcessor()


# File validation helper functions
def _is_allowed_image_file(filename):
    """Check if uploaded file is an allowed image type"""
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def _is_allowed_video_file(filename):
    """Check if uploaded file is an allowed video type"""
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"}
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


# Helper function to normalize optional descriptive fields
def normalize_optional_field(value):
    """Normalize optional descriptive fields to handle NA, Not Known, etc."""
    if not value or not value.strip():
        return "Not provided"
    
    # Common variations of "not available" or "not known"
    na_variations = [
        'na', 'n/a', 'not available', 'not known', 'nahi pata', 'no', 'none', 
        'nil', 'unknown', 'not specified', 'not provided', 'nhi', 'nai', 
        'pata nahi', 'malum nahi', 'not sure', 'dont know', "don't know",
        'nope', 'nothing', 'blank', 'empty', '-', '--', '---', 'null'
    ]
    
    cleaned_value = value.strip().lower()
    
    # Check if it's a variation of "not available"
    if cleaned_value in na_variations or len(cleaned_value) <= 2:
        return "Not provided"
    
    # Return original value if it contains meaningful information
    return value.strip()

# Authorization helper functions
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)

    return decorated_function


def case_owner_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        case_id = kwargs.get("case_id")
        if case_id:
            case = Case.query.get_or_404(case_id)
            if case.user_id != current_user.id and not current_user.is_admin:
                abort(403)
        return f(*args, **kwargs)

    return decorated_function


# The 'process_case' import is moved inside the function to prevent a circular import.

# This 'bp' variable is what the error is looking for.
bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    """Public landing page for all visitors"""
    # Don't show cases on homepage - users can go to All Cases for that
    return render_template("index.html")


@bp.route("/features")
def features_dashboard():
    """Comprehensive features dashboard showing all implemented features"""
    # Redirect to login if not authenticated
    if not current_user.is_authenticated:
        return redirect(url_for('main.login'))
    return render_template("features_dashboard.html", title="All Features")


@bp.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    """Handle Chrome DevTools request to prevent 404 errors"""
    return jsonify({}), 200


@bp.route("/favicon.ico")
def favicon():
    """Handle favicon request to prevent 404 errors"""
    return "", 204


@bp.route("/dashboard")
@login_required
def dashboard():
    """Secure dashboard for authenticated users"""
    # Check if user is admin and redirect to admin dashboard
    if current_user.is_admin:
        return redirect(url_for("admin.dashboard"))

    # Regular user dashboard with comprehensive status tracking
    from comprehensive_status_system import get_dashboard_status_counts, ALL_CASE_STATUSES
    
    user_cases = Case.query.filter_by(user_id=current_user.id).all()
    
    # Get comprehensive status counts
    status_counts = get_dashboard_status_counts(user_cases, 'user')
    
    total_cases = status_counts['total']
    active_cases = status_counts['active']
    pending_approval = status_counts['pending_approval']
    approved_cases = status_counts['approved']
    rejected_cases = status_counts['rejected']
    under_processing = status_counts['under_processing']
    completed_cases = status_counts['completed']
    withdrawn_cases = status_counts['withdrawn']
    total_sightings = sum(len(c.sightings) for c in user_cases)
    
    # Individual status counts for detailed display
    case_solved_count = len([c for c in user_cases if c.status == "Case Solved"])
    case_over_count = len([c for c in user_cases if c.status == "Case Over"])
    
    # Case type breakdown
    case_types = {}
    for case in user_cases:
        case_type = case.case_type or 'missing_person'
        case_types[case_type] = case_types.get(case_type, 0) + 1
    
    # Priority breakdown
    priority_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    for case in user_cases:
        priority_counts[case.priority] = priority_counts.get(case.priority, 0) + 1

    # Don't show recent cases on dashboard - user can go to All Cases for that

    user_stats = {
        "total_cases": total_cases,
        "active_cases": active_cases,
        "pending_approval": pending_approval,
        "approved_cases": approved_cases,
        "rejected_cases": rejected_cases,
        "under_processing": under_processing,
        "completed_cases": completed_cases,
        "case_solved_count": case_solved_count,
        "case_over_count": case_over_count,
        "withdrawn_cases": withdrawn_cases,
        "total_sightings": total_sightings,
        "case_types": case_types,
        "priority_counts": priority_counts,
    }
    
    # Get recent unread announcements for user dashboard
    try:
        all_active = Announcement.query.filter_by(is_active=True).order_by(Announcement.created_at.desc()).all()
        read_announcement_ids = db.session.query(AnnouncementRead.announcement_id).filter_by(user_id=current_user.id).all()
        read_ids = [r[0] for r in read_announcement_ids]
        recent_announcements = [a for a in all_active if a.id not in read_ids][:3]
    except Exception:
        # If table doesn't exist, show all announcements
        recent_announcements = Announcement.query.filter_by(is_active=True).order_by(Announcement.created_at.desc()).limit(3).all()

    return render_template(
        "user_dashboard.html", 
        user_stats=user_stats, 
        recent_announcements=recent_announcements,
        case_types=case_types,
        priority_counts=priority_counts
    )


@bp.route("/register_case", methods=["GET", "POST"])
@login_required
def register_case():
    # Prevent admin users from accessing this route
    if current_user.is_admin:
        flash("Admins cannot register new cases. Please use a regular user account.", "warning")
        return redirect(url_for("admin.dashboard"))
    form = NewCaseForm()
    if request.method == 'POST' and form.validate_on_submit():
        # Check for recent duplicate submissions (within last 5 minutes)
        from datetime import datetime, timedelta
        recent_time = datetime.utcnow() - timedelta(minutes=5)
        existing_case = Case.query.filter(
            Case.user_id == current_user.id,
            Case.person_name == form.full_name.data,
            Case.created_at > recent_time
        ).first()
        
        if existing_case:
            flash(f"A case for {form.full_name.data} was already submitted recently. Please check your cases.", "warning")
            return redirect(url_for("main.profile"))
        # Get additional form data
        contact_address = request.form.get('contact_address', '')
        last_seen_time = request.form.get('last_seen_time')
        
        # Get detailed location data
        last_seen_area = request.form.get('last_seen_area', '')
        last_seen_city = request.form.get('last_seen_city', '')
        last_seen_state = request.form.get('last_seen_state', '')
        last_seen_pincode = request.form.get('last_seen_pincode', '')
        last_seen_landmarks = request.form.get('last_seen_landmarks', '')
        
        # Create comprehensive location string
        location_parts = [form.last_seen_location.data, last_seen_area, last_seen_city, last_seen_state, last_seen_pincode]
        comprehensive_location = ', '.join([part for part in location_parts if part])
        if last_seen_landmarks:
            comprehensive_location += f' (Landmarks: {last_seen_landmarks})'
        
        # Parse time if provided
        parsed_time = None
        if last_seen_time:
            try:
                from datetime import time
                hour, minute = map(int, last_seen_time.split(':'))
                parsed_time = time(hour, minute)
            except:
                parsed_time = None
        
        # Normalize optional descriptive fields for new case
        normalized_nickname = normalize_optional_field(form.nickname.data)
        normalized_physical_desc = normalize_optional_field(form.distinguishing_marks.data)
        normalized_landmarks = normalize_optional_field(last_seen_landmarks)
        normalized_case_details = normalize_optional_field(form.additional_info.data)
        normalized_contact_address = normalize_optional_field(contact_address)
        
        # Create new case with comprehensive data  
        new_case = Case(
            case_type=form.case_type.data,
            requester_type=form.requester_type.data,
            case_category=form.case_category.data,
            person_name=form.full_name.data,
            age=form.age.data,
            details=f"Case Type: {form.case_type.data}\n"
            f"Requester Type: {form.requester_type.data}\n"
            f"Case Category: {form.case_category.data}\n"
            f"Nickname/Alias: {normalized_nickname}\n"
            f"Gender: {form.gender.data or 'Unknown'}\n"
            f"Height: {form.height_cm.data}cm\n" if form.height_cm.data else "Height: Unknown\n"
            f"Physical Description: {normalized_physical_desc}\n"
            f"Contact Person: {form.contact_person_name.data}\n"
            f"Contact Phone: {form.contact_person_phone.data}\n"
            f"Contact Email: {form.contact_person_email.data}\n"
            f"Contact Address: {normalized_contact_address}\n"
            f"Location Area: {normalize_optional_field(last_seen_area)}\n"
            f"Location City: {normalize_optional_field(last_seen_city)}\n"
            f"Location State: {normalize_optional_field(last_seen_state)}\n"
            f"Location PIN: {normalize_optional_field(last_seen_pincode)}\n"
            f"Landmarks: {normalized_landmarks}\n"
            f"Case Details: {normalized_case_details}",
            last_seen_location=comprehensive_location,
            last_seen_time=parsed_time,
            contact_address=contact_address,
            date_missing=form.last_seen_date.data,
            priority="High" if form.requester_type.data in ['police', 'government'] else "Medium",
            status="Pending Approval",  # Wait for admin approval
            user_id=current_user.id,
        )
        db.session.add(new_case)
        db.session.commit()

        # Handle multiple photo uploads with professional file management
        from file_lifecycle_manager import file_manager
        from storage_policies import storage_policy_manager
        
        photo_files = form.photos.data
        primary_photo_index = int(request.form.get('primary_photo_index', 0))
        
        # Validate upload limits based on case type
        current_file_count = len(photo_files) if photo_files else 0
        
        # Store uploaded file paths for person consistency validation
        uploaded_image_paths = []
        uploaded_video_paths = []
        
        for index, photo_file in enumerate(photo_files):
            if photo_file and photo_file.filename != "":
                # Validate file type
                if not _is_allowed_image_file(photo_file.filename):
                    flash(f"Invalid image file type: {photo_file.filename}", "error")
                    continue

                # Check file size and policy limits
                photo_file.seek(0, 2)
                file_size_mb = photo_file.tell() / (1024 * 1024)
                photo_file.seek(0)
                
                is_valid, message = storage_policy_manager.validate_file_upload(
                    new_case.case_type or 'missing_person', 
                    file_size_mb, 
                    current_file_count
                )
                
                if not is_valid:
                    flash(f"Upload rejected: {message}", "error")
                    continue

                # Store file using professional file manager
                try:
                    stored_path = file_manager.store_file(photo_file, new_case.id, "images")
                    
                    # Mark photo as primary based on user selection
                    is_primary = (index == primary_photo_index)
                    target_image = TargetImage(case_id=new_case.id, image_path=stored_path, is_primary=is_primary)
                    db.session.add(target_image)
                    
                    # Add to validation list
                    uploaded_image_paths.append(stored_path)
                    
                except Exception as e:
                    flash(f"Error storing file {photo_file.filename}: {str(e)}", "error")
                    continue

        # Handle multiple video uploads with enhanced security
        video_files = form.video.data if isinstance(form.video.data, list) else [form.video.data] if form.video.data else []
        for video_file in video_files:
            if video_file and video_file.filename != "":
                # Validate file type
                if not _is_allowed_video_file(video_file.filename):
                    flash(f"Invalid video file type: {video_file.filename}", "error")
                    continue
                
                # Create secure unique filename
                try:
                    from utils import sanitize_filename, create_safe_filename
                    original_filename = sanitize_filename(video_file.filename)
                except ImportError:
                    original_filename = secure_filename(video_file.filename)
                
                if not original_filename:
                    flash("Invalid video filename", "error")
                    continue
                
                # Generate unique filename to prevent conflicts
                file_ext = (
                    original_filename.rsplit(".", 1)[1].lower()
                    if "." in original_filename
                    else "mp4"
                )
                
                try:
                    unique_filename = create_safe_filename(f"case_{new_case.id}_video", file_ext)
                except (ImportError, NameError):
                    import uuid
                    unique_filename = f"case_{new_case.id}_video_{uuid.uuid4().hex[:8]}.{file_ext}"

                upload_dir = os.path.join("static", "uploads")
                os.makedirs(upload_dir, exist_ok=True)

                save_path = os.path.join(upload_dir, unique_filename)

                # Validate file size
                video_file.seek(0, 2)
                file_size = video_file.tell()
                video_file.seek(0)

                if file_size > 100 * 1024 * 1024:  # 100MB limit for videos
                    flash(f"Video file too large: {original_filename}", "error")
                    continue
                
                video_file.save(save_path)

                # Validate file content after upload (optional)
                try:
                    from utils import validate_file_content
                    if not validate_file_content(save_path, "video"):
                        os.remove(save_path)  # Remove invalid file
                        flash(f"Invalid video file content: {original_filename}", "error")
                        continue
                except ImportError:
                    # Skip validation if utils module doesn't exist
                    pass
                
                db_path = os.path.join(
                    "uploads", unique_filename
                ).replace("\\", "/")
                search_video = SearchVideo(
                    case_id=new_case.id,
                    video_path=db_path,
                    video_name=original_filename,
                )
                db.session.add(search_video)
                
                # Add to validation list
                uploaded_video_paths.append(save_path)

        try:
            db.session.commit()
            
            # Validate that at least one photo was uploaded successfully
            if not new_case.target_images:
                flash("Warning: No valid photos were uploaded. Please add photos for better investigation results.", "warning")
            
            # Person Consistency Validation - Ensure all photos/videos contain same person
            person_consistency_passed = True
            consistency_warnings = []
            
            if uploaded_image_paths or uploaded_video_paths:
                try:
                    from person_consistency_validator import validate_case_person_consistency
                    
                    print(f"🔍 Validating person consistency for case #{new_case.id}")
                    print(f"Images: {len(uploaded_image_paths)}, Videos: {len(uploaded_video_paths)}")
                    
                    # Perform person consistency validation
                    consistency_result = validate_case_person_consistency(
                        new_case.id, 
                        uploaded_image_paths, 
                        uploaded_video_paths
                    )
                    
                    print(f"Consistency validation result: {consistency_result['is_consistent']} (confidence: {consistency_result['confidence_score']:.2f})")
                    
                    # Check validation results
                    if not consistency_result['is_consistent']:
                        person_consistency_passed = False
                        
                        # Generate detailed feedback for user
                        consistency_message = f"""⚠️ SAME PERSON VALIDATION FAILED
                        
🚫 PROBLEM: Multiple different people detected in your uploaded files!
                        
🔍 ANALYSIS RESULTS:
• Total faces found: {consistency_result['total_faces_found']}
• Same person faces: {consistency_result['consistent_faces']}
• Confidence score: {consistency_result['confidence_score']:.1%}
• Files checked: {consistency_result['detailed_analysis']['total_files_processed']}
                        
❌ FILES WITH DIFFERENT PEOPLE:
{chr(10).join([f'• {file["file"].split("/")[-1]}: {file["issue"]}' for file in consistency_result['inconsistent_files'][:5]])}
                        
🎯 REQUIRED ACTION:
• केवल उसी व्यक्ति की photos/videos upload करें जिसे आप ढूंढना चाहते हैं
• सभी files में same person होना जरूरी है
• Group photos या multiple लोगों वाली images avoid करें
• Clear, different angles से same person की photos लें
                        
💡 WHY THIS MATTERS:
• AI को सही person identify करने के लिए consistent data चाहिए
• गलत photos से CCTV में wrong person detect हो सकता है
• Accuracy बढ़ाने के लिए same person validation जरूरी है
                        
🔄 Please upload only photos/videos of the SAME PERSON and resubmit your case."""
                        
                        new_case.status = 'Rejected'
                        new_case.admin_message = consistency_message
                        
                        # Create user notification
                        from models import Notification
                        notification = Notification(
                            user_id=current_user.id,
                            title=f"🚫 Same Person Validation Failed: {new_case.person_name}",
                            message=f"आपकी uploaded files में अलग-अलग लोग detect हुए हैं! सभी photos/videos में केवल उसी व्यक्ति को होना चाहिए जिसे आप ढूंढना चाहते हैं। Consistency score: {consistency_result['confidence_score']:.1%}। कृपया सही files के साथ फिर से submit करें। यह CCTV analysis की accuracy के लिए जरूरी है।",
                            type="warning",
                            created_at=get_ist_now()
                        )
                        db.session.add(notification)
                        
                        print(f"❌ Case #{new_case.id} rejected due to person inconsistency")
                        
                    elif consistency_result['confidence_score'] < 0.8:
                        # Low confidence warning but don't reject
                        consistency_warnings.append(f"Person consistency confidence is moderate ({consistency_result['confidence_score']:.1%}). Consider uploading clearer photos for better AI analysis.")
                        print(f"⚠️ Case #{new_case.id} has moderate consistency confidence: {consistency_result['confidence_score']:.1%}")
                    else:
                        print(f"✅ Case #{new_case.id} passed person consistency validation with {consistency_result['confidence_score']:.1%} confidence")
                        
                        # Store person profile for future AI analysis
                        if consistency_result.get('primary_person_encoding'):
                            try:
                                from models import PersonProfile
                                import json
                                
                                person_profile = PersonProfile(
                                    case_id=new_case.id,
                                    primary_face_encoding=json.dumps(consistency_result['primary_person_encoding']),
                                    face_quality_score=consistency_result['confidence_score'],
                                    profile_confidence=consistency_result['confidence_score']
                                )
                                db.session.add(person_profile)
                                print(f"✅ Person profile created for case #{new_case.id}")
                            except Exception as e:
                                print(f"⚠️ Failed to create person profile: {str(e)}")
                    
                    # Store consistency validation results in database
                    try:
                        from person_consistency_models import PersonConsistencyValidation, PersonConsistencyIssue
                        import json
                        
                        validation_record = PersonConsistencyValidation(
                            case_id=new_case.id,
                            is_consistent=consistency_result['is_consistent'],
                            confidence_score=consistency_result['confidence_score'],
                            total_faces_found=consistency_result['total_faces_found'],
                            consistent_faces=consistency_result['consistent_faces'],
                            total_files_processed=consistency_result['detailed_analysis']['total_files_processed'],
                            images_processed=consistency_result['detailed_analysis']['images_processed'],
                            videos_processed=consistency_result['detailed_analysis']['videos_processed'],
                            inconsistent_files=json.dumps(consistency_result['inconsistent_files']),
                            recommendations=json.dumps(consistency_result['recommendations']),
                            detailed_analysis=json.dumps(consistency_result['detailed_analysis']),
                            primary_person_encoding=json.dumps(consistency_result.get('primary_person_encoding', [])),
                            validation_threshold=0.65  # Updated threshold
                        )
                        db.session.add(validation_record)
                        db.session.flush()  # Get validation ID
                        
                        # Store individual issues
                        for issue in consistency_result['inconsistent_files']:
                            issue_record = PersonConsistencyIssue(
                                validation_id=validation_record.id,
                                file_path=issue['file'],
                                file_type='image' if any(ext in issue['file'].lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']) else 'video',
                                issue_type='different_person' if 'Different person' in issue['issue'] else 'no_face' if 'No face' in issue['issue'] else 'other',
                                issue_description=issue['issue'],
                                similarity_score=issue.get('similarity_score', 0.0)
                            )
                            db.session.add(issue_record)
                        
                        print(f"✅ Person consistency validation results stored for case #{new_case.id}")
                        
                    except Exception as e:
                        print(f"⚠️ Failed to store consistency validation results: {str(e)}")
                        # Don't fail the case if storage fails
                    
                except Exception as e:
                    print(f"⚠️ Person consistency validation failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Don't reject case if validation fails - continue with normal flow
                    consistency_warnings.append("Person consistency validation failed - manual review recommended.")
            
            # AI-Powered Case Validation with Quality Assessment and Intelligent Categorization
            try:
                from ai_case_validator import ai_validator
                from automated_case_quality_assessment import case_quality_assessor
                from intelligent_case_categorizer import intelligent_categorizer
                from models import CaseQualityAssessment, CaseCategorization
                import json
                
                print(f"🤖 Starting comprehensive AI analysis for case #{new_case.id}")
                
                # Step 1: Intelligent Case Categorization (NEW)
                categorization_result = intelligent_categorizer.categorize_case(new_case)
                
                # Store categorization results in database
                categorization_record = CaseCategorization(
                    case_id=new_case.id,
                    detected_case_type=categorization_result['case_type_detection']['detected_type'],
                    case_type_confidence=categorization_result['case_type_detection']['confidence'],
                    alternative_types=json.dumps(categorization_result['case_type_detection'].get('alternative_types', [])),
                    detection_method=categorization_result['case_type_detection']['method'],
                    risk_level=categorization_result['risk_assessment']['risk_level'],
                    risk_confidence=categorization_result['risk_assessment']['confidence'],
                    risk_factors=json.dumps(categorization_result['risk_assessment']['risk_factors']),
                    risk_scores=json.dumps(categorization_result['risk_assessment']['risk_scores']),
                    priority_score=categorization_result['priority_scoring']['priority_score'],
                    priority_category=categorization_result['priority_scoring']['priority_category'],
                    priority_confidence=categorization_result['priority_scoring']['confidence'],
                    scoring_factors=json.dumps(categorization_result['priority_scoring']['scoring_factors']),
                    recommended_sla=categorization_result['priority_scoring']['recommended_sla'],
                    automatic_tags=json.dumps(categorization_result['tag_generation']['automatic_tags']),
                    entity_tags=json.dumps(categorization_result['tag_generation']['entity_tags']),
                    location_tags=json.dumps(categorization_result['tag_generation']['location_tags']),
                    temporal_tags=json.dumps(categorization_result['tag_generation']['temporal_tags']),
                    risk_tags=json.dumps(categorization_result['tag_generation']['risk_tags']),
                    category_tags=json.dumps(categorization_result['tag_generation']['category_tags']),
                    overall_confidence=categorization_result['confidence_scores']['overall_confidence'],
                    recommendations=json.dumps(categorization_result['recommendations']),
                    processing_notes=json.dumps(categorization_result['processing_notes'])
                )
                db.session.add(categorization_record)
                
                # Update case with AI-detected information
                if categorization_result['case_type_detection']['confidence'] > 0.7:
                    # Override case type if AI is confident
                    ai_detected_type = categorization_result['case_type_detection']['detected_type']
                    if ai_detected_type != new_case.case_type:
                        print(f"🔄 AI detected different case type: {ai_detected_type} (confidence: {categorization_result['case_type_detection']['confidence']:.1%})")
                        # Keep original but note the AI suggestion
                
                # Update priority based on AI analysis
                ai_priority = categorization_result['priority_scoring']['priority_category']
                if ai_priority in ['Critical', 'High'] and new_case.priority in ['Medium', 'Low']:
                    new_case.priority = ai_priority
                    print(f"📈 Priority elevated to {ai_priority} based on AI analysis")
                
                # Step 2: Comprehensive Quality Assessment
                quality_assessment = case_quality_assessor.assess_case_quality(new_case)
                
                # Store quality assessment in database
                quality_record = CaseQualityAssessment(
                    case_id=new_case.id,
                    overall_score=quality_assessment['overall_score'],
                    photo_quality_score=quality_assessment['photo_quality']['score'],
                    information_completeness_score=quality_assessment['information_completeness']['score'],
                    urgency_score=quality_assessment['urgency_classification']['score'],
                    duplicate_risk_score=quality_assessment['duplicate_risk']['score'],
                    quality_grade=quality_assessment['quality_grade'],
                    processing_priority=quality_assessment['processing_priority'],
                    urgency_level=quality_assessment['urgency_classification']['level'],
                    duplicate_risk_level=quality_assessment['duplicate_risk']['risk_level'],
                    estimated_success_rate=quality_assessment['estimated_success_rate'],
                    assessment_details=json.dumps(quality_assessment, default=str),
                    recommendations=json.dumps(quality_assessment['recommendations']),
                    similar_cases=json.dumps(quality_assessment['duplicate_risk'].get('similar_cases', []))
                )
                db.session.add(quality_record)
                
                # Step 3: Traditional AI Validation with Smart Feedback
                validation_result = ai_validator.validate_case(new_case)
                if len(validation_result) == 5:
                    decision, confidence, scores, reasons, smart_feedback = validation_result
                else:
                    decision, confidence, scores, reasons = validation_result
                    smart_feedback = None
                
                # Step 4: Enhanced Decision Making with Categorization, Quality Assessment, and Person Consistency
                final_decision = decision
                
                # Override decision based on person consistency (highest priority)
                if not person_consistency_passed:
                    final_decision = 'REJECT'
                    reasons = ['Person consistency validation failed - multiple different people detected']
                elif categorization_result['risk_assessment']['risk_level'] == 'critical':
                    final_decision = 'APPROVE'  # Always approve critical risk cases
                    reasons = ['Critical risk case - immediate processing required']
                elif categorization_result['priority_scoring']['priority_score'] >= 85:
                    final_decision = 'APPROVE'  # High priority cases
                    reasons = ['High priority case based on AI analysis']
                
                # Override decision based on quality assessment
                if quality_assessment['duplicate_risk']['risk_level'] == 'High':
                    final_decision = 'REJECT'
                    reasons.append("High duplicate risk detected")
                elif quality_assessment['overall_score'] < 0.3:
                    final_decision = 'REJECT'
                    reasons.append("Case quality too low for processing")
                elif quality_assessment['urgency_classification']['level'] == 'Critical':
                    final_decision = 'APPROVE'  # Always approve critical cases
                
                if final_decision == 'APPROVE':
                    new_case.status = 'Approved'
                    new_case.priority = max(quality_assessment['processing_priority'], categorization_result['priority_scoring']['priority_category'], key=lambda x: ['Low', 'Medium', 'High', 'Critical'].index(x))
                    
                    # Enhanced approval message with categorization, quality, and person consistency insights
                    consistency_info = ""
                    if uploaded_image_paths or uploaded_video_paths:
                        try:
                            consistency_info = f"""
👤 PERSON CONSISTENCY:
• Validation Status: ✅ PASSED
• Confidence Score: {consistency_result['confidence_score']:.1%}
• Faces Analyzed: {consistency_result['total_faces_found']}
• Consistent Faces: {consistency_result['consistent_faces']}
• Files Processed: {consistency_result['detailed_analysis']['total_files_processed']}
                            """
                            if consistency_warnings:
                                consistency_info += f"\n• Warnings: {'; '.join(consistency_warnings[:2])}"
                        except:
                            consistency_info = "\n👤 PERSON CONSISTENCY: ✅ Basic validation passed"
                    
                    approval_message = f"""🤖 AI AUTO-APPROVED with Comprehensive Analysis{consistency_info}
                    
🔍 CASE CATEGORIZATION:
• Detected Type: {categorization_result['case_type_detection']['detected_type'].replace('_', ' ').title()}
• Risk Level: {categorization_result['risk_assessment']['risk_level'].title()}
• Priority Score: {categorization_result['priority_scoring']['priority_score']:.0f}/100
• Recommended SLA: {categorization_result['priority_scoring']['recommended_sla']}
• AI Confidence: {categorization_result['confidence_scores']['overall_confidence']:.1%}
                    
📊 QUALITY ASSESSMENT:
• Quality Grade: {quality_assessment['quality_grade']}
• Photo Quality: {quality_assessment['photo_quality']['score']:.1%}
• Information Completeness: {quality_assessment['information_completeness']['score']:.1%}
• Success Rate: {quality_assessment['estimated_success_rate']:.1%}
• Duplicate Risk: {quality_assessment['duplicate_risk']['risk_level']}
                    
🏷️ AUTO-GENERATED TAGS:
{', '.join(categorization_result['tag_generation']['automatic_tags'][:5]) if categorization_result['tag_generation']['automatic_tags'] else 'None'}
                    
💡 AI RECOMMENDATIONS:
{chr(10).join([f'• {rec}' for rec in categorization_result['recommendations'][:3]])}
                    
✅ Case optimized for {categorization_result['priority_scoring']['recommended_sla']} processing with {quality_assessment['estimated_success_rate']:.0%} success probability."""
                    
                    new_case.admin_message = approval_message
                    
                    # Create user notification for auto-approval
                    from models import Notification
                    consistency_note = ""
                    if uploaded_image_paths or uploaded_video_paths:
                        try:
                            consistency_note = f" Person consistency: {consistency_result['confidence_score']:.0%}."
                        except:
                            consistency_note = " Person validation: Passed."
                    
                    notification = Notification(
                        user_id=current_user.id,
                        title=f"🤖 AI Auto-Approved: {new_case.person_name} ({categorization_result['case_type_detection']['detected_type'].replace('_', ' ').title()})",
                        message=f"Excellent! Your {categorization_result['case_type_detection']['detected_type'].replace('_', ' ')} case has been automatically approved with {quality_assessment['quality_grade']} quality grade, {categorization_result['risk_assessment']['risk_level']} risk level, and {quality_assessment['estimated_success_rate']:.0%} success rate.{consistency_note} Processing priority: {new_case.priority}. SLA: {categorization_result['priority_scoring']['recommended_sla']}.",
                        type="success",
                        created_at=get_ist_now()
                    )
                    db.session.add(notification)
                    
                    print(f"✅ Case #{new_case.id} auto-approved - Type: {categorization_result['case_type_detection']['detected_type']}, Risk: {categorization_result['risk_assessment']['risk_level']}, Priority: {new_case.priority}")
                    
                else:  # REJECT
                    new_case.status = 'Rejected'
                    
                    # Generate smart rejection message with detailed feedback
                    if smart_feedback:
                        # Use Smart Rejection System for detailed feedback
                        rejection_message = f"""🤖 AI ANALYSIS - Smart Improvement Guide
                        
📊 OVERALL ASSESSMENT:
• Quality Grade: {smart_feedback['overall_assessment']['grade']}
• Current Score: {smart_feedback['overall_assessment']['score']:.1%}
• Status: {smart_feedback['overall_assessment']['status']}
• Approval Chance: {smart_feedback['estimated_approval_chance']['approval_chance']}

🔍 CASE ANALYSIS:
• Detected Type: {categorization_result['case_type_detection']['detected_type'].replace('_', ' ').title()}
• Risk Level: {categorization_result['risk_assessment']['risk_level'].title()}
• Priority Score: {categorization_result['priority_scoring']['priority_score']:.0f}/100

📸 PHOTO FEEDBACK:
• Priority: {smart_feedback['photo_feedback']['priority']}
• Issues Found: {len(smart_feedback['photo_feedback']['issues'])}
• Key Issues: {', '.join(smart_feedback['photo_feedback']['issues'][:3])}

📝 FORM FEEDBACK:
• Priority: {smart_feedback['form_feedback']['priority']}
• Issues Found: {len(smart_feedback['form_feedback']['issues'])}
• Key Issues: {', '.join(smart_feedback['form_feedback']['issues'][:3])}

🎯 TOP PRIORITY ACTIONS:
{chr(10).join([f'• {action["action"]} (Impact: {action["impact"]})' for action in smart_feedback['priority_actions'][:3]])}

📋 IMPROVEMENT PLAN:
{chr(10).join([f'{step["step"]}. {step["title"]} - {step["time_estimate"]}' for step in smart_feedback['improvement_plan']['steps']])}

📈 AFTER IMPROVEMENTS:
• Expected Score: {smart_feedback['estimated_approval_chance']['potential_score']}
• {smart_feedback['estimated_approval_chance']['message']}

💡 QUICK TIPS:
• Focus on photo quality first (highest impact)
• Use the improvement templates provided
• Follow the step-by-step plan above
• Estimated total time: {smart_feedback['improvement_plan']['total_estimated_time']}

🔄 Resubmit after addressing priority issues for best results."""
                    else:
                        # Fallback to basic rejection message
                        rejection_message = f"""🤖 AI ANALYSIS - Case Needs Improvement
                        
🔍 CASE ANALYSIS:
• Detected Type: {categorization_result['case_type_detection']['detected_type'].replace('_', ' ').title()}
• Risk Level: {categorization_result['risk_assessment']['risk_level'].title()}
• Priority Score: {categorization_result['priority_scoring']['priority_score']:.0f}/100
• AI Confidence: {categorization_result['confidence_scores']['overall_confidence']:.1%}
                        
📊 QUALITY ISSUES:
• Quality Grade: {quality_assessment['quality_grade']}
• Overall Score: {quality_assessment['overall_score']:.1%}
• Success Rate: {quality_assessment['estimated_success_rate']:.1%}
• Photo Quality: {quality_assessment['photo_quality']['score']:.1%} - {len(quality_assessment['photo_quality']['issues'])} issues
• Information Completeness: {quality_assessment['information_completeness']['score']:.1%}
• Duplicate Risk: {quality_assessment['duplicate_risk']['risk_level']}
                        
❌ REJECTION REASONS:
{chr(10).join([f'• {reason}' for reason in reasons[:5]])}
                        
🎯 IMPROVEMENT RECOMMENDATIONS:
{chr(10).join([f'• {rec}' for rec in (quality_assessment['recommendations'] + categorization_result['recommendations'])[:6]])}
                        
📈 Resubmit after addressing these issues. Expected improvement: +{(1-quality_assessment['overall_score'])*100:.0f}% success rate."""
                    
                    new_case.admin_message = rejection_message
                    
                    # Create user notification for auto-rejection
                    from models import Notification
                    notification = Notification(
                        user_id=current_user.id,
                        title=f"🤖 Case Analysis Complete: {new_case.person_name} - Improvements Needed",
                        message=f"Your {categorization_result['case_type_detection']['detected_type'].replace('_', ' ')} case (Risk: {categorization_result['risk_assessment']['risk_level']}, Grade: {quality_assessment['quality_grade']}) needs improvements. Current success rate: {quality_assessment['estimated_success_rate']:.0%}. Priority score: {categorization_result['priority_scoring']['priority_score']:.0f}/100. Please review the {len(quality_assessment['recommendations'] + categorization_result['recommendations'])} AI recommendations and resubmit.",
                        type="warning",
                        created_at=get_ist_now()
                    )
                    db.session.add(notification)
                    
                    print(f"❌ Case #{new_case.id} auto-rejected - Type: {categorization_result['case_type_detection']['detected_type']}, Risk: {categorization_result['risk_assessment']['risk_level']}, Grade: {quality_assessment['quality_grade']}")
                    print(f"Analysis: Priority {categorization_result['priority_scoring']['priority_score']:.0f}/100, {len(categorization_result['recommendations'])} categorization + {len(quality_assessment['recommendations'])} quality recommendations")
                
                db.session.commit()
                
            except Exception as ai_error:
                print(f"⚠️ AI analysis (categorization/quality assessment) failed: {str(ai_error)}")
                import traceback
                traceback.print_exc()
                # Fallback to manual approval if AI fails
                new_case.status = 'Pending Approval'
                db.session.commit()
            
            # Enhanced notification system for all outcomes
            from models import Notification, User
            admins = User.query.filter_by(is_admin=True).all()
            
            # Get case type display name
            case_type_display = dict(form.case_type.choices).get(new_case.case_type, 'Investigation')
            
            # Create comprehensive user notification based on outcome
            if new_case.status == 'Approved':
                user_notification = Notification(
                    user_id=current_user.id,
                    title=f"🎉 Case Auto-Approved: {new_case.person_name}",
                    message=f"Great news! Your {case_type_display.lower()} case has been automatically approved by AI and is now active for processing. You'll receive updates as we make progress.",
                    type="success",
                    related_url=f"/case/{new_case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(user_notification)
                
            elif new_case.status == 'Rejected':
                # Extract actual rejection reasons (not optional field issues)
                actual_reasons = []
                if hasattr(new_case, 'admin_message') and new_case.admin_message:
                    if 'person consistency' in new_case.admin_message.lower():
                        actual_reasons.append("Multiple different people detected in photos/videos")
                    if 'quality' in new_case.admin_message.lower() and 'low' in new_case.admin_message.lower():
                        actual_reasons.append("Photo quality needs improvement")
                    if 'duplicate' in new_case.admin_message.lower():
                        actual_reasons.append("Similar case already exists")
                    if 'information' in new_case.admin_message.lower() and 'incomplete' in new_case.admin_message.lower():
                        actual_reasons.append("Critical information missing")
                
                reason_text = "; ".join(actual_reasons[:3]) if actual_reasons else "Case quality below minimum standards"
                
                user_notification = Notification(
                    user_id=current_user.id,
                    title=f"📋 Case Needs Improvement: {new_case.person_name}",
                    message=f"Your {case_type_display.lower()} case requires improvements. Main issues: {reason_text}. Please review the detailed feedback and edit your case to resubmit.",
                    type="warning",
                    related_url=f"/case/{new_case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(user_notification)
                
            else:  # Pending Approval
                user_notification = Notification(
                    user_id=current_user.id,
                    title=f"⏳ Case Under Review: {new_case.person_name}",
                    message=f"Your {case_type_display.lower()} case is being reviewed by our admin team. This usually takes 24-48 hours. You'll be notified once a decision is made.",
                    type="info",
                    related_url=f"/case/{new_case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(user_notification)
            
            # Create admin notifications for all outcomes with enhanced override tracking
            for admin in admins:
                if new_case.status == 'Approved':
                    admin_title = f"🤖 AI Auto-Approved: {new_case.person_name}"
                    admin_message = f"AI automatically approved this case with {confidence:.1%} confidence.\n\nType: {case_type_display}\nRequester: {dict(form.requester_type.choices).get(new_case.requester_type)}\nLocation: {new_case.last_seen_location}\nAge: {new_case.age or 'Unknown'}\nRegistered by: {current_user.username}\n\n✅ Case is ready for processing. You can override this decision if needed.\n\n🔧 Override Options: Approve → Reject, Change Status, Add Custom Message"
                    notification_type = "success"
                elif new_case.status == 'Rejected':
                    admin_title = f"🤖 AI Auto-Rejected: {new_case.person_name}"
                    admin_message = f"AI automatically rejected this case with {confidence:.1%} confidence.\n\nType: {case_type_display}\nRequester: {dict(form.requester_type.choices).get(new_case.requester_type)}\nLocation: {new_case.last_seen_location}\nAge: {new_case.age or 'Unknown'}\nRegistered by: {current_user.username}\n\n❌ User has been notified. You can override this decision if needed.\n\n🔧 Override Options: Reject → Approve, Change Status, Add Custom Message"
                    notification_type = "warning"
                else:  # Pending Approval
                    admin_title = f"⏳ Manual Review Required: {new_case.person_name}"
                    admin_message = f"AI analysis inconclusive - manual review needed.\n\nType: {case_type_display}\nRequester: {dict(form.requester_type.choices).get(new_case.requester_type)}\nLocation: {new_case.last_seen_location}\nAge: {new_case.age or 'Unknown'}\nRegistered by: {current_user.username}\n\n🔍 Please review and make a decision on this case.\n\n⚡ Priority: Manual approval required for processing"
                    notification_type = "info"
                
                notification = Notification(
                    user_id=admin.id,
                    sender_id=current_user.id,
                    title=admin_title,
                    message=admin_message,
                    type=notification_type,
                    related_url=f"/admin/cases/{new_case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(notification)
            
            # Start AI analysis of uploaded media
            try:
                from ai_processor import process_case_media
                
                # Collect photo and video paths
                photo_paths = [os.path.join("app", img.image_path) for img in new_case.target_images]
                video_paths = [os.path.join("app", vid.video_path) for vid in new_case.search_videos]
                
                # Process media in background (simplified for now)
                if photo_paths or video_paths:
                    analysis_results = process_case_media(new_case.id, photo_paths, video_paths)
                    print(f"AI Analysis completed for case {new_case.id}")
                    
            except Exception as e:
                print(f"AI Analysis failed for case {new_case.id}: {str(e)}")
            
            # Initialize advanced location matching for approved cases
            try:
                from advanced_location_matcher import advanced_matcher
                if new_case.status == 'Approved':
                    # Auto-process location matching
                    matches_created = advanced_matcher.auto_process_approved_case(new_case.id)
                    print(f"Advanced location matching: {matches_created} matches created for case {new_case.id}")
            except Exception as e:
                print(f"Advanced location matching failed for case {new_case.id}: {str(e)}")
            
            # Don't start AI processing - wait for admin approval
            success_msg = f"{case_type_display} request for {new_case.person_name} has been submitted successfully! Your case is now pending admin approval and will be processed accordingly."
            flash(success_msg, "success")
            return redirect(url_for("main.profile"))
            
        except Exception as e:
            db.session.rollback()
            flash(f"Error submitting case: {str(e)}. Please try again or contact support.", "error")

    return render_template(
        "register_case_new.html", title="Submit Investigation Request", form=form
    )


# ... (The rest of the file is the same, including all other routes)


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully!")
        return redirect(url_for("main.login"))
    return render_template("register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            # Process security event for failed login
            try:
                request_data = {
                    'ip_address': request.remote_addr,
                    'user_id': None,
                    'user_agent': request.headers.get('User-Agent', ''),
                    'path': request.path
                }
                process_security_event(request_data)
            except:
                pass
            
            flash("Invalid username or password")
            return redirect(url_for("main.login"))
        
        # Process security event for successful login
        try:
            request_data = {
                'ip_address': request.remote_addr,
                'user_id': user.id,
                'user_agent': request.headers.get('User-Agent', ''),
                'path': request.path
            }
            process_security_event(request_data)
        except:
            pass
        
        # Update login tracking and online status
        ist_now = get_ist_now()
        user.last_login = ist_now
        user.login_count = (user.login_count or 0) + 1
        user.is_online = True
        user.last_seen = ist_now
        db.session.commit()
        
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for("main.index"))
    return render_template("login.html", form=form)


@bp.route("/logout")
@login_required
def logout():
    # Update offline status
    current_user.is_online = False
    current_user.last_seen = get_ist_now()
    db.session.commit()
    logout_user()
    return redirect(url_for("main.index"))


@bp.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            flash(
                "Password reset link would be sent to your email (email sending not implemented)."
            )
        else:
            flash("Email not found")
        return redirect(url_for("main.login"))
    return render_template("forgot_password.html", form=form)


@bp.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))
    user = User.verify_reset_token(token)
    if not user:
        flash("Invalid or expired token")
        return redirect(url_for("main.forgot_password"))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash("Password has been reset successfully")
        return redirect(url_for("main.login"))
    return render_template("reset_password.html", form=form)


@bp.route("/profile")
@login_required
def profile():
    cases = Case.query.filter_by(user_id=current_user.id).order_by(Case.id.desc()).all()
    return render_template("profile.html", cases=cases)


@bp.route("/case/<int:case_id>")
@login_required
@case_owner_required
def case_details(case_id):
    """View detailed information about a specific case with AI analysis results"""
    try:
        case = Case.query.get_or_404(case_id)
        
        # Get AI analysis results
        from models import LocationMatch, PersonDetection
        location_matches = LocationMatch.query.filter_by(case_id=case_id).order_by(LocationMatch.created_at.desc()).all()
        
        # Get all detections for this case with proper formatting
        all_detections = []
        for match in location_matches:
            detections = PersonDetection.query.filter_by(location_match_id=match.id).order_by(PersonDetection.confidence_score.desc()).all()
            for detection in detections:
                detection.match = match  # Add match info to detection
                # Format timestamp for display
                minutes = int(detection.timestamp // 60)
                seconds = int(detection.timestamp % 60)
                detection.formatted_timestamp = f"{minutes:02d}:{seconds:02d}"
                all_detections.append(detection)
        
        # Sort detections by confidence score (highest first)
        all_detections.sort(key=lambda x: x.confidence_score, reverse=True)
        
        print(f"✅ Case {case_id}: Found {len(location_matches)} location matches and {len(all_detections)} detections")
        
        return render_template(
            "case_details.html", 
            case=case,
            location_matches=location_matches,
            detections=all_detections
        )
        
    except Exception as e:
        print(f"❌ Error loading case details {case_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        flash("Error loading case details. Please try again.", "error")
        return redirect(url_for("main.profile"))


@bp.route("/case/<int:case_id>/edit", methods=["GET", "POST"])
@login_required
@case_owner_required
def edit_case(case_id):
    """Edit case details with comprehensive form like new case registration"""
    case = Case.query.get_or_404(case_id)
    
    # Check if case can be edited using comprehensive status system
    from models import FINAL_STATUSES
    if case.status in FINAL_STATUSES:
        flash("This case has been finalized by admin and cannot be edited.", "error")
        return redirect(url_for('main.case_details', case_id=case_id))
    
    form = NewCaseForm()
    
    if request.method == 'GET':
        # Pre-populate form with existing case data
        form.case_type.data = case.case_type or 'missing_person'
        form.requester_type.data = case.requester_type or 'family'
        form.case_category.data = case.case_category
        form.full_name.data = case.person_name
        form.age.data = case.age
        form.last_seen_location.data = case.last_seen_location
        form.last_seen_date.data = case.date_missing
        
        # Parse details back to form fields
        if case.details:
            details_lines = case.details.split('\n')
            for line in details_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'Nickname/Alias':
                        form.nickname.data = value if value != 'N/A' else ''
                    elif key == 'Gender':
                        form.gender.data = value if value != 'Unknown' else ''
                    elif key == 'Height':
                        form.height_cm.data = int(value.replace('cm', '')) if value.replace('cm', '').isdigit() else None
                    elif key == 'Physical Description':
                        form.distinguishing_marks.data = value if value != 'Not provided' else ''
                    elif key == 'Contact Person':
                        form.contact_person_name.data = value
                    elif key == 'Contact Phone':
                        form.contact_person_phone.data = value
                    elif key == 'Contact Email':
                        form.contact_person_email.data = value
                    elif key == 'Case Details':
                        form.additional_info.data = value if value != 'None' else ''
    
    if request.method == 'POST':
        # Track changes for admin notification
        changes_made = []
        
        # Custom validation for comprehensive error handling
        validation_errors = []
        
        # Check required fields
        if not form.full_name.data or not form.full_name.data.strip():
            validation_errors.append("Full Name is required")
        if not form.last_seen_location.data or not form.last_seen_location.data.strip():
            validation_errors.append("Last Seen Location is required")
        if not form.last_seen_date.data:
            validation_errors.append("Date Missing is required")
        if not form.contact_person_name.data or not form.contact_person_name.data.strip():
            validation_errors.append("Contact Person Name is required")
        if not form.contact_person_phone.data or not form.contact_person_phone.data.strip():
            validation_errors.append("Contact Phone is required")
        
        # Check photo requirements
        photos_to_remove = request.form.get('photos_to_remove', '').split(',') if request.form.get('photos_to_remove') else []
        remaining_photos = [img for img in case.target_images if str(img.id) not in photos_to_remove]
        new_photos = form.photos.data if form.photos.data else []
        total_photos = len(remaining_photos) + len([p for p in new_photos if p and p.filename])
        
        if total_photos == 0:
            validation_errors.append("At least one photo is required")
        
        # Check primary photo selection
        primary_photo_selection = request.form.get('primary_photo_selection')
        if total_photos > 0 and not primary_photo_selection:
            validation_errors.append("Please select one primary photo")
        
        # Show validation errors if any
        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
            return render_template("edit_case.html", title="Edit Case", form=form, case=case)
        
        # Proceed with form processing if validation passes
        if form.validate_on_submit():
            # Check for changes and update case
            if case.case_type != form.case_type.data:
                changes_made.append(f"Case Type: {case.case_type} → {form.case_type.data}")
                case.case_type = form.case_type.data
            
            if case.requester_type != form.requester_type.data:
                changes_made.append(f"Requester Type: {case.requester_type} → {form.requester_type.data}")
                case.requester_type = form.requester_type.data
            
            if case.case_category != form.case_category.data:
                changes_made.append(f"Case Category: {case.case_category} → {form.case_category.data}")
                case.case_category = form.case_category.data
            
            if case.person_name != form.full_name.data:
                changes_made.append(f"Person Name: {case.person_name} → {form.full_name.data}")
                case.person_name = form.full_name.data
            
            if case.age != form.age.data:
                changes_made.append(f"Age: {case.age or 'Unknown'} → {form.age.data or 'Unknown'}")
                case.age = form.age.data
            
            if case.last_seen_location != form.last_seen_location.data:
                changes_made.append(f"Location: {case.last_seen_location} → {form.last_seen_location.data}")
                case.last_seen_location = form.last_seen_location.data
            
            if case.date_missing != form.last_seen_date.data:
                old_date = case.date_missing.strftime('%Y-%m-%d') if case.date_missing else 'Unknown'
                new_date = form.last_seen_date.data.strftime('%Y-%m-%d') if form.last_seen_date.data else 'Unknown'
                changes_made.append(f"Missing Date: {old_date} → {new_date}")
                case.date_missing = form.last_seen_date.data
            
            # Get additional form data
            contact_address = request.form.get('contact_address', '')
            last_seen_time = request.form.get('last_seen_time')
            last_seen_area = request.form.get('last_seen_area', '')
            last_seen_city = request.form.get('last_seen_city', '')
            last_seen_state = request.form.get('last_seen_state', '')
            last_seen_pincode = request.form.get('last_seen_pincode', '')
            last_seen_landmarks = request.form.get('last_seen_landmarks', '')
            
            # Parse time if provided
            parsed_time = None
            if last_seen_time:
                try:
                    from datetime import time
                    hour, minute = map(int, last_seen_time.split(':'))
                    parsed_time = time(hour, minute)
                except:
                    parsed_time = None
            
            case.last_seen_time = parsed_time
            case.contact_address = contact_address
            
            # Create comprehensive location string
            location_parts = [form.last_seen_location.data, last_seen_area, last_seen_city, last_seen_state, last_seen_pincode]
            comprehensive_location = ', '.join([part for part in location_parts if part])
            if last_seen_landmarks:
                comprehensive_location += f' (Landmarks: {last_seen_landmarks})'
            case.last_seen_location = comprehensive_location
            
            # Normalize optional descriptive fields
            normalized_nickname = normalize_optional_field(form.nickname.data)
            normalized_physical_desc = normalize_optional_field(form.distinguishing_marks.data)
            normalized_landmarks = normalize_optional_field(last_seen_landmarks)
            normalized_case_details = normalize_optional_field(form.additional_info.data)
            normalized_contact_address = normalize_optional_field(contact_address)
            
            # Update details with normalized values
            case.details = f"Case Type: {form.case_type.data}\n" \
                          f"Requester Type: {form.requester_type.data}\n" \
                          f"Case Category: {form.case_category.data}\n" \
                          f"Nickname/Alias: {normalized_nickname}\n" \
                          f"Gender: {form.gender.data or 'Unknown'}\n" \
                          f"Height: {form.height_cm.data}cm\n" if form.height_cm.data else "Height: Unknown\n" \
                          f"Physical Description: {normalized_physical_desc}\n" \
                          f"Contact Person: {form.contact_person_name.data}\n" \
                          f"Contact Phone: {form.contact_person_phone.data}\n" \
                          f"Contact Email: {form.contact_person_email.data}\n" \
                          f"Contact Address: {normalized_contact_address}\n" \
                          f"Location Area: {normalize_optional_field(last_seen_area)}\n" \
                          f"Location City: {normalize_optional_field(last_seen_city)}\n" \
                          f"Location State: {normalize_optional_field(last_seen_state)}\n" \
                          f"Location PIN: {normalize_optional_field(last_seen_pincode)}\n" \
                          f"Landmarks: {normalized_landmarks}\n" \
                          f"Case Details: {normalized_case_details}"
            
            # Handle photo/video removal
        photos_to_remove = request.form.get('photos_to_remove', '').split(',') if request.form.get('photos_to_remove') else []
        videos_to_remove = request.form.get('videos_to_remove', '').split(',') if request.form.get('videos_to_remove') else []
        
        # Remove selected photos
        for photo_id in photos_to_remove:
            if photo_id:
                photo = TargetImage.query.filter_by(id=int(photo_id), case_id=case.id).first()
                if photo:
                    changes_made.append(f"Removed photo: {photo.image_path}")
                    # Delete file
                    try:
                        file_path = os.path.join("static", photo.image_path)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except:
                        pass
                    db.session.delete(photo)
        
        # Remove selected videos
        for video_id in videos_to_remove:
            if video_id:
                video = SearchVideo.query.filter_by(id=int(video_id), case_id=case.id).first()
                if video:
                    changes_made.append(f"Removed video: {video.video_name}")
                    # Delete file
                    try:
                        file_path = os.path.join("static", video.video_path)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except:
                        pass
                    db.session.delete(video)
        
        # Handle new photo uploads with enhanced security
        photo_files = form.photos.data
        new_photos_added = []
        
        if photo_files:
            for index, photo_file in enumerate(photo_files):
                if photo_file and photo_file.filename != "":
                    # Validate file type
                    if not _is_allowed_image_file(photo_file.filename):
                        flash(f"Invalid image file type: {photo_file.filename}", "error")
                        continue
                    
                    try:
                        from utils import sanitize_filename, create_safe_filename
                        original_filename = sanitize_filename(photo_file.filename)
                    except ImportError:
                        original_filename = secure_filename(photo_file.filename)
                    
                    if not original_filename:
                        flash("Invalid filename", "error")
                        continue
                    
                    file_ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "jpg"
                    
                    try:
                        unique_filename = create_safe_filename(f"case_{case.id}_photo", file_ext)
                    except (ImportError, NameError):
                        import uuid
                        unique_filename = f"case_{case.id}_photo_{uuid.uuid4().hex[:8]}.{file_ext}"
                    
                    upload_dir = os.path.join("static", "uploads")
                    os.makedirs(upload_dir, exist_ok=True)
                    save_path = os.path.join(upload_dir, unique_filename)
                    
                    photo_file.save(save_path)
                    
                    db_path = os.path.join("uploads", unique_filename).replace("\\", "/")
                    target_image = TargetImage(case_id=case.id, image_path=db_path, is_primary=False)
                    db.session.add(target_image)
                    db.session.flush()  # Get image ID
                    
                    new_photos_added.append((target_image.id, index))
                    changes_made.append(f"Added new photo: {original_filename}")
        
        # Handle primary photo selection (unified system)
        primary_photo_selection = request.form.get('primary_photo_selection')
        if primary_photo_selection:
            # Reset all photos to non-primary
            for img in case.target_images:
                img.is_primary = False
            
            if primary_photo_selection.startswith('existing_'):
                # Existing photo selected
                photo_id = int(primary_photo_selection.replace('existing_', ''))
                primary_photo = TargetImage.query.filter_by(id=photo_id, case_id=case.id).first()
                if primary_photo:
                    primary_photo.is_primary = True
                    changes_made.append(f"Changed primary photo to existing photo")
            elif primary_photo_selection.startswith('new_'):
                # New photo selected
                new_photo_index = int(primary_photo_selection.replace('new_', ''))
                for photo_id, index in new_photos_added:
                    if index == new_photo_index:
                        new_primary = TargetImage.query.get(photo_id)
                        if new_primary:
                            new_primary.is_primary = True
                            changes_made.append(f"Set new uploaded photo as primary")
                        break
        
        # Ensure at least one photo is primary if photos exist
        if case.target_images and not any(img.is_primary for img in case.target_images):
            case.target_images[0].is_primary = True
        
        # Store original status for notification
        original_status = case.status
        
        # Reset status for resubmission
        if case.status in ['Approved', 'Under Processing', 'Rejected', 'Withdrawn']:
            case.status = 'Pending Approval'
            if changes_made:
                flash(f"Case updated and resubmitted! Your case is now pending admin review.", "info")
            else:
                flash(f"Case resubmitted successfully! Your case is now pending admin review.", "info")
        
        case.updated_at = get_ist_now()
        

        
        # Handle new video uploads (optional)
        video_files = form.video.data if isinstance(form.video.data, list) else [form.video.data] if form.video.data else []
        for video_file in video_files:
            if video_file and video_file.filename != "":
                if not _is_allowed_video_file(video_file.filename):
                    flash(f"Invalid video file type: {video_file.filename}", "error")
                    continue
                
                try:
                    from utils import sanitize_filename, create_safe_filename
                    original_filename = sanitize_filename(video_file.filename)
                except ImportError:
                    original_filename = secure_filename(video_file.filename)
                
                if original_filename:
                    file_ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "mp4"
                    
                    try:
                        unique_filename = create_safe_filename(f"case_{case.id}_video", file_ext)
                    except (ImportError, NameError):
                        import uuid
                        unique_filename = f"case_{case.id}_video_{uuid.uuid4().hex[:8]}.{file_ext}"
                    
                    upload_dir = os.path.join("static", "uploads")
                    os.makedirs(upload_dir, exist_ok=True)
                    save_path = os.path.join(upload_dir, unique_filename)
                    
                    video_file.save(save_path)
                    
                    db_path = os.path.join("uploads", unique_filename).replace("\\", "/")
                    search_video = SearchVideo(
                        case_id=case.id,
                        video_path=db_path,
                        video_name=original_filename,
                    )
                    db.session.add(search_video)
        
        try:
            db.session.commit()
            
            # Enhanced admin notification system for edited cases
            from models import Notification, User
            admins = User.query.filter_by(is_admin=True).all()
            
            # Create user notification first
            if case.status == 'Approved':
                user_notification = Notification(
                    user_id=current_user.id,
                    title=f"🎉 Edit Auto-Approved: {case.person_name}",
                    message=f"Great! Your edited case has been automatically approved by AI and is now active for processing. Your changes have been successfully applied.",
                    type="success",
                    related_url=f"/case/{case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(user_notification)
            elif case.status == 'Rejected':
                user_notification = Notification(
                    user_id=current_user.id,
                    title=f"📋 Edit Needs Improvement: {case.person_name}",
                    message=f"Your edited case requires further improvements. Please review the AI feedback and make necessary corrections before resubmitting.",
                    type="warning",
                    related_url=f"/case/{case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(user_notification)
            else:  # Pending Approval
                user_notification = Notification(
                    user_id=current_user.id,
                    title=f"⏳ Edit Under Review: {case.person_name}",
                    message=f"Your edited case is being reviewed by our admin team. You'll be notified once a decision is made on your changes.",
                    type="info",
                    related_url=f"/case/{case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(user_notification)
            
            # Create admin notifications
            for admin in admins:
                if original_status == 'Rejected':
                    title = f"🔄 Case Resubmitted: {case.person_name}"
                    action_type = "resubmitted"
                else:
                    title = f"✏️ Case Updated: {case.person_name}"
                    action_type = "updated"
                
                # Create detailed change summary
                change_summary = "\n\n📝 CHANGES MADE:\n" + "\n".join([f"• {change}" for change in changes_made[:10]])
                if len(changes_made) > 10:
                    change_summary += f"\n• ... and {len(changes_made) - 10} more changes"
                
                if not changes_made:
                    change_summary = "\n\n📝 CHANGES: New photos/videos added or existing media updated"
                
                case_type_display = case.case_type.replace('_', ' ')
                contact_phone = case.details.split('Contact Phone: ')[1].split('\n')[0] if 'Contact Phone: ' in case.details else 'N/A'
                review_action = 'resubmitted' if original_status == 'Rejected' else 'updated'
                
                # Add AI decision info if available
                ai_decision_info = ""
                if case.status == 'Approved':
                    ai_decision_info = "\n\n🤖 AI RE-ANALYSIS: Auto-approved after edit\n🔧 Override Options: Approve → Reject, Change Status, Add Custom Message"
                elif case.status == 'Rejected':
                    ai_decision_info = "\n\n🤖 AI RE-ANALYSIS: Auto-rejected after edit\n🔧 Override Options: Reject → Approve, Change Status, Add Custom Message"
                else:
                    ai_decision_info = "\n\n⏳ MANUAL REVIEW: AI analysis inconclusive - admin decision required"
                
                message = f"User {current_user.username} has {action_type} {case_type_display} case #{case.id} for {case.person_name} (was {original_status}).\n\n" \
                         f"📍 Location: {case.last_seen_location}\n" \
                         f"👤 Age: {case.age or 'Unknown'}\n" \
                         f"📞 Contact: {contact_phone}" \
                         f"{change_summary}\n\n" \
                         f"⏰ Updated: {case.updated_at.strftime('%d %b %Y at %I:%M %p')}" \
                         f"{ai_decision_info}\n\n" \
                         f"Please review the {review_action} case."
                
                notification_type = "success" if case.status == 'Approved' else "warning" if case.status == 'Rejected' else "info"
                
                notification = Notification(
                    user_id=admin.id,
                    sender_id=current_user.id,
                    title=title,
                    message=message,
                    type=notification_type,
                    related_url=f"/admin/cases/{case.id}",
                    created_at=get_ist_now()
                )
                db.session.add(notification)
            
            db.session.commit()
            
            # Trigger comprehensive AI analysis for edited case (same as new case)
            try:
                # Person Consistency Validation - Re-validate all photos/videos
                uploaded_image_paths = [os.path.join("static", img.image_path) for img in case.target_images]
                uploaded_video_paths = [os.path.join("static", vid.video_path) for vid in case.search_videos]
                
                person_consistency_passed = True
                if uploaded_image_paths or uploaded_video_paths:
                    try:
                        from person_consistency_validator import validate_case_person_consistency
                        
                        print(f"🔍 Re-validating person consistency for edited case #{case.id}")
                        consistency_result = validate_case_person_consistency(
                            case.id, 
                            uploaded_image_paths, 
                            uploaded_video_paths
                        )
                        
                        if not consistency_result['is_consistent']:
                            person_consistency_passed = False
                            case.status = 'Rejected'
                            case.admin_message = f"⚠️ PERSON CONSISTENCY FAILED AFTER EDIT\n\n" \
                                               f"Multiple different people detected in updated files!\n" \
                                               f"Confidence: {consistency_result['confidence_score']:.1%}\n" \
                                               f"Please ensure all photos/videos contain the same person."
                            
                            # Create user notification
                            from models import Notification
                            notification = Notification(
                                user_id=current_user.id,
                                title=f"🚫 Edit Rejected: Person Consistency Failed",
                                message=f"Your edited case for {case.person_name} was rejected due to person consistency issues. Multiple different people detected in uploaded files.",
                                type="warning",
                                created_at=get_ist_now()
                            )
                            db.session.add(notification)
                            db.session.commit()
                            
                            flash(f"Case edit rejected! Person consistency validation failed. Please upload photos/videos of the same person only.", "error")
                            return redirect(url_for('main.case_details', case_id=case.id))
                    except Exception as e:
                        print(f"⚠️ Person consistency validation failed: {str(e)}")
                
                # AI-Powered Case Validation with Enhanced Analysis
                try:
                    from ai_case_validator import ai_validator
                    from automated_case_quality_assessment import case_quality_assessor
                    from intelligent_case_categorizer import intelligent_categorizer
                    
                    print(f"🤖 Starting comprehensive AI re-analysis for edited case #{case.id}")
                    
                    # Step 1: Re-categorize case with new data
                    categorization_result = intelligent_categorizer.categorize_case(case)
                    
                    # Step 2: Re-assess quality with enhanced AI insights
                    quality_assessment = case_quality_assessor.assess_case_quality(case)
                    
                    # Enhanced AI validation with advanced features
                    validation_result = ai_validator.validate_case(case)
                    
                    # Add enhanced AI insights to quality assessment
                    if uploaded_image_paths:
                        enhanced_processor = get_enhanced_ai_processor()
                        ai_capabilities = enhanced_processor.engine.get_system_info()
                        quality_assessment['enhanced_ai_used'] = ai_capabilities
                        print(f"Enhanced AI capabilities applied: {ai_capabilities['capabilities']}")
                    if len(validation_result) == 5:
                        decision, confidence, scores, reasons, smart_feedback = validation_result
                    else:
                        decision, confidence, scores, reasons = validation_result
                        smart_feedback = None
                    
                    # Step 4: Enhanced decision making
                    final_decision = decision
                    
                    # Override based on analysis results
                    if categorization_result['risk_assessment']['risk_level'] == 'critical':
                        final_decision = 'APPROVE'
                    elif quality_assessment['duplicate_risk']['risk_level'] == 'High':
                        final_decision = 'REJECT'
                    elif quality_assessment['overall_score'] < 0.3:
                        final_decision = 'REJECT'
                    
                    if final_decision == 'APPROVE':
                        case.status = 'Approved'
                        case.priority = max(quality_assessment['processing_priority'], categorization_result['priority_scoring']['priority_category'], key=lambda x: ['Low', 'Medium', 'High', 'Critical'].index(x))
                        
                        approval_message = f"🤖 AI AUTO-APPROVED AFTER EDIT\n\n" \
                                         f"✅ Updated case passed comprehensive AI analysis\n" \
                                         f"Quality Grade: {quality_assessment['quality_grade']}\n" \
                                         f"Risk Level: {categorization_result['risk_assessment']['risk_level'].title()}\n" \
                                         f"Success Rate: {quality_assessment['estimated_success_rate']:.1%}\n" \
                                         f"Priority: {case.priority}\n\n" \
                                         f"Changes made: {len(changes_made)} modifications detected\n" \
                                         f"AI Confidence: {confidence:.1%}"
                        
                        case.admin_message = approval_message
                        
                        # Create enhanced user notification for auto-approval
                        from models import Notification
                        try:
                            enhanced_processor = get_enhanced_ai_processor()
                            ai_methods_used = enhanced_processor.engine.get_system_info().get('ai_methods_used', [])
                        except:
                            ai_methods_used = []
                        ai_methods_text = ', '.join(ai_methods_used) if ai_methods_used else 'Basic AI'
                        
                        notification = Notification(
                            user_id=current_user.id,
                            title=f"🤖 Enhanced AI Auto-Approved: {case.person_name}",
                            message=f"Excellent! Your edited case has been automatically approved using {ai_methods_text} with {quality_assessment['quality_grade']} quality grade and {quality_assessment['estimated_success_rate']:.0%} success rate. Advanced AI analysis applied.",
                            type="success",
                            created_at=get_ist_now()
                        )
                        db.session.add(notification)
                        
                        print(f"✅ Enhanced AI: Edited case #{case.id} auto-approved - Quality: {quality_assessment['quality_grade']}, Risk: {categorization_result['risk_assessment']['risk_level']}")
                        print(f"AI Methods Used: {ai_methods_used}")
                        
                    else:  # REJECT
                        case.status = 'Rejected'
                        
                        if smart_feedback:
                            rejection_message = f"🤖 AI ANALYSIS - Edit Needs Further Improvement\n\n" \
                                              f"📊 Updated Assessment:\n" \
                                              f"Quality Grade: {smart_feedback['overall_assessment']['grade']}\n" \
                                              f"Current Score: {smart_feedback['overall_assessment']['score']:.1%}\n" \
                                              f"Approval Chance: {smart_feedback['estimated_approval_chance']['approval_chance']}\n\n" \
                                              f"🎯 Priority Actions:\n" + "\n".join([f"• {action['action']}" for action in smart_feedback['priority_actions'][:3]]) + "\n\n" \
                                              f"📈 Expected Score After Improvements: {smart_feedback['estimated_approval_chance']['potential_score']}\n" \
                                              f"Changes made: {len(changes_made)} modifications detected"
                        else:
                            rejection_message = f"🤖 AI ANALYSIS - Edit Rejected\n\n" \
                                              f"Quality Grade: {quality_assessment['quality_grade']}\n" \
                                              f"Overall Score: {quality_assessment['overall_score']:.1%}\n" \
                                              f"Success Rate: {quality_assessment['estimated_success_rate']:.1%}\n\n" \
                                              f"Issues Found: {len(reasons)} problems detected\n" \
                                              f"Changes made: {len(changes_made)} modifications"
                        
                        case.admin_message = rejection_message
                        
                        # Create enhanced user notification for rejection
                        from models import Notification
                        try:
                            enhanced_processor = get_enhanced_ai_processor()
                            ai_methods_used = enhanced_processor.engine.get_system_info().get('ai_methods_used', [])
                        except:
                            ai_methods_used = []
                        ai_methods_text = ', '.join(ai_methods_used) if ai_methods_used else 'Basic AI'
                        
                        notification = Notification(
                            user_id=current_user.id,
                            title=f"🤖 Enhanced AI Analysis: {case.person_name} - Improvements Needed",
                            message=f"Your edited case was analyzed using {ai_methods_text} and needs improvements. Quality grade: {quality_assessment['quality_grade']}, Success rate: {quality_assessment['estimated_success_rate']:.0%}. Please review detailed AI feedback.",
                            type="warning",
                            created_at=get_ist_now()
                        )
                        db.session.add(notification)
                        
                        print(f"❌ Enhanced AI: Edited case #{case.id} rejected - Quality: {quality_assessment['quality_grade']}, Score: {quality_assessment['overall_score']:.1%}")
                        print(f"AI Methods Used: {ai_methods_used}")
                    
                    db.session.commit()
                    
                except Exception as ai_error:
                    print(f"⚠️ AI analysis failed for edited case: {str(ai_error)}")
                    # Fallback to manual approval if AI fails
                    case.status = 'Pending Approval'
                    db.session.commit()
                
            except Exception as e:
                print(f"⚠️ Comprehensive analysis failed: {str(e)}")
                db.session.rollback()
                flash(f"Case updated but AI analysis failed. Admin will review manually.", "warning")
                return redirect(url_for('main.case_details', case_id=case.id))
            
            # Enhanced status messages with AI method information
            try:
                enhanced_processor = get_enhanced_ai_processor()
                ai_methods_used = enhanced_processor.engine.get_system_info().get('ai_methods_used', [])
                ai_methods_text = ', '.join(ai_methods_used) if ai_methods_used else 'Basic AI'
            except:
                ai_methods_text = 'AI'
                
            if case.status == 'Approved':
                flash(f"🎉 Case for {case.person_name} has been updated and automatically approved using {ai_methods_text}!", "success")
            elif case.status == 'Rejected':
                flash(f"⚠️ Case for {case.person_name} was analyzed using {ai_methods_text} but needs improvements. Please review detailed feedback.", "warning")
            elif original_status == 'Rejected':
                flash(f"Case for {case.person_name} has been updated and resubmitted for admin review!", "success")
            else:
                flash(f"Case for {case.person_name} has been updated and analyzed using {ai_methods_text}! Admin has been notified.", "success")
            return redirect(url_for('main.case_details', case_id=case.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating case: {str(e)}. Please try again.", "error")
    
    return render_template("edit_case.html", title="Edit Case", form=form, case=case)


@bp.route("/case/<int:case_id>/withdraw", methods=["POST"])
@login_required
@case_owner_required
def withdraw_case(case_id):
    """Withdraw a case - change status to Withdrawn (NO DELETION)"""
    case = Case.query.get_or_404(case_id)
    person_name = case.person_name
    
    # Only change status to Withdrawn - DO NOT DELETE
    case.status = "Withdrawn"
    case.admin_message = f"Case withdrawn by user {current_user.username} on {get_ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST"
    case.updated_at = get_ist_now()
    
    # Create notification for user
    from models import Notification
    user_notification = Notification(
        user_id=current_user.id,
        title=f"Case Withdrawn: {person_name}",
        message=f"You have successfully withdrawn the case for {person_name}. The case is preserved in your records and can be resubmitted if needed.",
        type="info",
        created_at=get_ist_now()
    )
    db.session.add(user_notification)
    
    # Notify admins about withdrawal
    from models import User
    admins = User.query.filter_by(is_admin=True).all()
    for admin in admins:
        admin_notification = Notification(
            user_id=admin.id,
            sender_id=current_user.id,
            title=f"Case Withdrawn by User: {person_name}",
            message=f"User {current_user.username} has withdrawn case #{case.id} for {person_name}. Case data is preserved for records.",
            type="info",
            related_url=f"/admin/cases/{case.id}",
            created_at=get_ist_now()
        )
        db.session.add(admin_notification)
    
    db.session.commit()
    
    flash(f"Case for {person_name} has been withdrawn. You can resubmit it later if needed.", "success")
    return redirect(url_for("main.profile"))


@bp.route("/case_status/<int:case_id>")
@login_required
@case_owner_required
def case_status(case_id):
    case = Case.query.get_or_404(case_id)
    sightings = []
    for s in case.sightings:
        video_name = (
            s.search_video.video_path.split("/")[-1] if s.search_video else "N/A"
        )
        sightings.append(
            {
                "video_name": video_name,
                "timestamp": s.timestamp,
                "confidence_score": round(s.confidence_score, 2),
                "thumbnail_path": url_for(
                    "static", filename=s.thumbnail_path.replace("static\\", "/")
                ),
            }
        )
    response_data = {"status": case.status, "sightings": sightings}
    return jsonify(response_data)


@bp.route("/notifications")
@login_required
def notifications():
    """User notifications page"""
    from models import Notification

    # Get all notifications for current user, ordered by newest first
    user_notifications = (
        Notification.query.filter_by(user_id=current_user.id)
        .order_by(Notification.created_at.desc())
        .all()
    )

    # Don't automatically mark as read - let user choose

    return render_template(
        "notifications.html", title="Notifications", notifications=user_notifications
    )


@bp.route("/public-cases")
def public_cases():
    """All cases directory - shows different cases based on user type"""
    if current_user.is_authenticated and current_user.is_admin:
        # Admin sees ALL cases with all statuses
        cases = (
            Case.query.order_by(Case.created_at.desc()).all()
        )
        template_name = "admin_all_cases.html"
        title = "All Cases - Admin View"
    elif current_user.is_authenticated:
        # Regular users see only public visible cases
        from models import PUBLIC_VISIBLE_STATUSES
        cases = (
            Case.query.filter(Case.status.in_(PUBLIC_VISIBLE_STATUSES))
            .order_by(Case.created_at.desc())
            .all()
        )
        template_name = "user_all_cases.html"
        title = "All Cases - Public Cases"
    else:
        # Unregistered users see only public visible cases
        from models import PUBLIC_VISIBLE_STATUSES
        cases = (
            Case.query.filter(Case.status.in_(PUBLIC_VISIBLE_STATUSES))
            .order_by(Case.created_at.desc())
            .all()
        )
        template_name = "public_cases.html"
        title = "Public Cases Directory"
    
    return render_template(
        template_name, cases=cases, title=title
    )

@bp.route("/public-case/<int:case_id>")
def public_case_details(case_id):
    """Public case details - accessible to everyone"""
    case = Case.query.get_or_404(case_id)
    
    # Only show public visible cases
    from models import PUBLIC_VISIBLE_STATUSES
    if case.status not in PUBLIC_VISIBLE_STATUSES:
        abort(404)
    
    return render_template(
        "public_case_details.html", case=case, title=f"Case Details - {case.person_name}"
    )

@bp.route("/missing_persons")
@login_required
def missing_persons():
    """Public directory of missing persons cases"""
    # Show only active cases to public
    from models import ACTIVE_STATUSES
    cases = (
        Case.query.filter(Case.status.in_(ACTIVE_STATUSES))
        .order_by(Case.created_at.desc())
        .all()
    )
    return render_template(
        "missing_persons.html", cases=cases, title="Missing Persons Directory"
    )


@bp.route("/about")
def about():
    """About page - detailed platform information"""
    return render_template("about.html", title="About the Platform")


@bp.route("/contact", methods=["GET", "POST"])
@login_required
def contact():
    """Contact page - available to logged-in users only"""
    form = ContactForm()

    # Pre-populate form with user data if logged in
    if current_user.is_authenticated and request.method == "GET":
        form.name.data = current_user.username
        form.email.data = current_user.email

    if request.method == "POST":
        # Handle both form validation and direct form data
        if form.validate_on_submit():
            name = form.name.data
            email = form.email.data
            subject = form.subject.data
            message = form.message.data
        else:
            # Fallback to request.form for hardcoded HTML form
            name = request.form.get('name')
            email = request.form.get('email')
            subject = request.form.get('subject')
            message = request.form.get('message')
        
        if name and email and subject and message:
            # Save message to database
            from models import ContactMessage
            contact_message = ContactMessage(
                name=name,
                email=email,
                subject=subject,
                message=message
            )
            db.session.add(contact_message)
            db.session.commit()
            
            flash("Thank you for your message! We will get back to you shortly.", "success")
            return redirect(url_for("main.contact"))
        else:
            flash("Please fill in all required fields.", "error")

    return render_template("contact.html", title="Contact & Help", form=form)


@bp.route("/privacy")
def privacy():
    """Privacy & Security page"""
    return render_template("privacy.html", title="Privacy & Security")


@bp.route("/faq")
def faq():
    """FAQ page"""
    return render_template("faq.html", title="Frequently Asked Questions")


@bp.route("/chat")
@login_required
def chat_list():
    """List all chat rooms for current user"""
    from models import ChatRoom, User
    
    if current_user.is_admin:
        # Admin sees all chat rooms where they are the admin
        chat_rooms = ChatRoom.query.filter_by(admin_id=current_user.id).order_by(ChatRoom.last_message_at.desc()).all()
    else:
        # Regular user sees their chat rooms
        chat_rooms = ChatRoom.query.filter_by(user_id=current_user.id).order_by(ChatRoom.last_message_at.desc()).all()
        
        # If no chat room exists, create one with first available admin
        if not chat_rooms:
            admin = User.query.filter_by(is_admin=True).first()
            if admin:
                new_room = ChatRoom(user_id=current_user.id, admin_id=admin.id)
                db.session.add(new_room)
                db.session.commit()
                chat_rooms = [new_room]
            else:
                # No admin available, show message
                flash("No admin available for chat at the moment. Please try again later.", "warning")
    
    return render_template("chat/chat_list.html", chat_rooms=chat_rooms)


@bp.route("/chat/<int:room_id>")
@login_required
def chat_room(room_id):
    """Individual chat room"""
    from models import ChatRoom, ChatMessage
    
    room = ChatRoom.query.get_or_404(room_id)
    
    # Check access permissions
    if not current_user.is_admin and room.user_id != current_user.id:
        abort(403)
    if current_user.is_admin and room.admin_id != current_user.id:
        abort(403)
    
    # Get messages (exclude hidden ones for current user)
    try:
        messages_query = ChatMessage.query.filter_by(chat_room_id=room_id)
        
        if current_user.is_admin:
            messages_query = messages_query.filter((ChatMessage.hidden_for_admin == False) | (ChatMessage.hidden_for_admin == None))
        else:
            messages_query = messages_query.filter((ChatMessage.hidden_for_user == False) | (ChatMessage.hidden_for_user == None))
        
        messages = messages_query.order_by(ChatMessage.created_at.asc()).all()
    except Exception as e:
        print(f"Error loading messages: {e}")
        # Fallback to all messages if column doesn't exist
        messages = ChatMessage.query.filter_by(chat_room_id=room_id).order_by(ChatMessage.created_at.asc()).all()
    
    # Mark messages as seen (not just read)
    unread_messages = ChatMessage.query.filter_by(chat_room_id=room_id, is_read=False).filter(ChatMessage.sender_id != current_user.id).all()
    for msg in unread_messages:
        msg.mark_seen()
    
    return render_template("chat/chat_room.html", room=room, messages=messages, timedelta=timedelta)


@bp.route("/chat/<int:room_id>/send", methods=["POST"])
@login_required
def send_message(room_id):
    """Send a message in chat room"""
    try:
        from models import ChatRoom, ChatMessage, Notification, User
        
        print(f"Send message request for room {room_id} from user {current_user.id}")
        
        room = ChatRoom.query.get_or_404(room_id)
        
        # Check access permissions
        if not current_user.is_admin and room.user_id != current_user.id:
            print(f"Access denied: User {current_user.id} not authorized for room {room_id}")
            return jsonify({'error': 'Access denied'}), 403
        if current_user.is_admin and room.admin_id != current_user.id:
            print(f"Access denied: Admin {current_user.id} not authorized for room {room_id}")
            return jsonify({'error': 'Access denied'}), 403
        
        message_content = request.form.get('message', '').strip()
        file = request.files.get('file')
        
        print(f"Message content: '{message_content}', File: {file}")
        
        if not message_content and not file:
            print("No message content or file provided")
            return jsonify({'error': 'Message or file required'}), 400
        
        # Create message with current timestamp and initial status 'sent'
        message = ChatMessage(
            chat_room_id=room_id,
            sender_id=current_user.id,
            content=message_content if message_content else None,
            message_type='text',
            status='sent',
            created_at=get_ist_now()
        )
        
        # Handle file upload
        if file and file.filename:
            import os
            from werkzeug.utils import secure_filename
            
            filename = secure_filename(file.filename)
            if not filename:
                return jsonify({'error': 'Invalid filename'}), 400
                
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            # Determine message type
            if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
                message.message_type = 'image'
            elif file_ext in ['mp4', 'avi', 'mov', 'webm', 'mkv']:
                message.message_type = 'video'
            else:
                message.message_type = 'file'
            
            # Save file
            upload_dir = os.path.join('static', 'chat_uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            unique_filename = f"chat_{room_id}_{get_ist_now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            try:
                file.save(file_path)
                message.file_path = f"chat_uploads/{unique_filename}"
                message.file_name = filename
            except Exception as e:
                return jsonify({'error': f'File upload failed: {str(e)}'}), 500
        
        db.session.add(message)
        db.session.flush()  # Get message ID
        
        # Mark as delivered immediately (simulating instant delivery)
        message.mark_delivered()
        
        # Update room last message time
        room.last_message_at = get_ist_now()
        
        # Create notification for recipient
        recipient_id = room.admin_id if current_user.id == room.user_id else room.user_id
        notification = Notification(
            user_id=recipient_id,
            sender_id=current_user.id,
            title="New Chat Message",
            message=f"{current_user.username}: {message_content[:50]}..." if message_content else f"{current_user.username} sent a file",
            type="chat",
            related_url=f"/chat/{room_id}",
            created_at=get_ist_now()
        )
        db.session.add(notification)
        
        db.session.commit()
        
        print(f"Message {message.id} created successfully with status {message.status}")
        
        return jsonify({
            'success': True, 
            'message_id': message.id,
            'status': message.status,
            'message': 'Message sent successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error in send_message: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@bp.route("/chat/<int:room_id>/messages")
@login_required
def get_messages(room_id):
    """Get messages for chat room (AJAX)"""
    from models import ChatRoom, ChatMessage
    
    room = ChatRoom.query.get_or_404(room_id)
    
    # Check access permissions
    if not current_user.is_admin and room.user_id != current_user.id:
        abort(403)
    if current_user.is_admin and room.admin_id != current_user.id:
        abort(403)
    
    since = request.args.get('since', type=int, default=0)
    
    try:
        messages_query = ChatMessage.query.filter_by(chat_room_id=room_id).filter(ChatMessage.id > since)
        
        if current_user.is_admin:
            messages_query = messages_query.filter((ChatMessage.hidden_for_admin == False) | (ChatMessage.hidden_for_admin == None))
        else:
            messages_query = messages_query.filter((ChatMessage.hidden_for_user == False) | (ChatMessage.hidden_for_user == None))
        
        messages = messages_query.order_by(ChatMessage.created_at.asc()).all()
    except Exception as e:
        print(f"Error loading messages: {e}")
        # Fallback to all messages if column doesn't exist
        messages = ChatMessage.query.filter_by(chat_room_id=room_id).filter(ChatMessage.id > since).order_by(ChatMessage.created_at.asc()).all()
    
    messages_data = []
    for msg in messages:
        messages_data.append({
            'id': msg.id,
            'sender_id': msg.sender_id,
            'sender_name': msg.sender.username,
            'content': msg.content,
            'message_type': msg.message_type,
            'file_path': msg.file_path,
            'file_name': msg.file_name,
            'created_at': msg.created_at.isoformat(),
            'created_at_ist': utc_to_ist(msg.created_at).strftime('%I:%M %p IST'),
            'is_own': msg.sender_id == current_user.id,
            'status': msg.status,
            'delivered_at': utc_to_ist(msg.delivered_at).strftime('%I:%M %p IST') if msg.delivered_at else None,
            'seen_at': utc_to_ist(msg.seen_at).strftime('%I:%M %p IST') if msg.seen_at else None
        })
    
    return jsonify({'messages': messages_data})


@bp.route("/api/chat-notifications")
@login_required
def chat_notifications():
    """Get unread chat count for current user"""
    from models import ChatRoom, ChatMessage
    
    if current_user.is_admin:
        # Count unread messages from users to admin
        unread_count = db.session.query(ChatMessage).join(ChatRoom).filter(
            ChatRoom.admin_id == current_user.id,
            ChatMessage.is_read == False,
            ChatMessage.sender_id != current_user.id
        ).count()
    else:
        # Count unread messages from admin to user
        unread_count = db.session.query(ChatMessage).join(ChatRoom).filter(
            ChatRoom.user_id == current_user.id,
            ChatMessage.is_read == False,
            ChatMessage.sender_id != current_user.id
        ).count()
    
    return jsonify({'unread_count': unread_count})


@bp.route("/chat/start")
@login_required
def start_chat():
    """Start a new chat with admin (for users) or redirect to chat list"""
    from models import ChatRoom, User
    
    if current_user.is_admin:
        return redirect(url_for('main.chat_list'))
    
    # Check if user already has a chat room
    existing_room = ChatRoom.query.filter_by(user_id=current_user.id).first()
    if existing_room:
        return redirect(url_for('main.chat_room', room_id=existing_room.id))
    
    # Create new chat room with first available admin
    admin = User.query.filter_by(is_admin=True).first()
    if not admin:
        flash("No admin available for chat at the moment. Please try again later.", "warning")
        return redirect(url_for('main.contact'))
    
    new_room = ChatRoom(user_id=current_user.id, admin_id=admin.id)
    db.session.add(new_room)
    db.session.commit()
    
    return redirect(url_for('main.chat_room', room_id=new_room.id))


@bp.route("/api/chat/<int:room_id>/mark-seen", methods=["POST"])
@login_required
def mark_messages_seen(room_id):
    """Mark all messages in room as seen by current user"""
    from models import ChatRoom, ChatMessage
    
    room = ChatRoom.query.get_or_404(room_id)
    
    # Check access permissions
    if not current_user.is_admin and room.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    if current_user.is_admin and room.admin_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Mark all unread messages from other user as seen
    unread_messages = ChatMessage.query.filter_by(
        chat_room_id=room_id, 
        is_read=False
    ).filter(ChatMessage.sender_id != current_user.id).all()
    
    for msg in unread_messages:
        msg.mark_seen()
    
    return jsonify({'success': True, 'marked_count': len(unread_messages)})


@bp.route("/api/announcement/<int:announcement_id>/mark-read", methods=["POST"])
@login_required
def mark_announcement_read(announcement_id):
    """Mark announcement as read for current user"""
    try:
        # Check if already marked as read
        existing = AnnouncementRead.query.filter_by(
            user_id=current_user.id,
            announcement_id=announcement_id
        ).first()
        
        if not existing:
            # Mark as read
            read_record = AnnouncementRead(
                user_id=current_user.id,
                announcement_id=announcement_id,
                read_at=get_ist_now()
            )
            db.session.add(read_record)
            db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route("/api/chat/message-status/<int:message_id>")
@login_required
def get_message_status(message_id):
    """Get status of a specific message"""
    from models import ChatMessage
    
    message = ChatMessage.query.get_or_404(message_id)
    
    # Only sender can check message status
    if message.sender_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    return jsonify({
        'message_id': message.id,
        'status': message.status,
        'delivered_at': message.delivered_at.isoformat() if message.delivered_at else None,
        'seen_at': message.seen_at.isoformat() if message.seen_at else None
    })


@bp.route("/api/chat/<int:room_id>/clear", methods=["POST"])
@login_required
def clear_chat_history(room_id):
    """Hide messages for current user only"""
    from models import ChatRoom, ChatMessage
    
    room = ChatRoom.query.get_or_404(room_id)
    
    # Check access permissions
    if not current_user.is_admin and room.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    if current_user.is_admin and room.admin_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Hide messages for current user only
    try:
        messages = ChatMessage.query.filter_by(chat_room_id=room_id).all()
        
        for message in messages:
            if current_user.is_admin:
                message.hidden_for_admin = True
            else:
                message.hidden_for_user = True
        
        db.session.commit()
    except Exception as e:
        print(f"Error clearing messages: {e}")
        return jsonify({'error': 'Failed to clear messages'}), 500
    
    return jsonify({'success': True, 'message': 'Chat history cleared for you only'})


@bp.route("/api/user/<int:user_id>/status")
@login_required
def get_user_status(user_id):
    """Get online status and last seen for a user"""
    user = User.query.get_or_404(user_id)
    
    # Calculate if user is considered online (active within last 5 minutes)
    now = get_ist_now()
    online_threshold = now - timedelta(minutes=5)
    
    # Ensure both datetimes have timezone info for comparison
    user_last_seen_tz = utc_to_ist(user.last_seen) if user.last_seen and user.last_seen.tzinfo is None else user.last_seen
    is_online = user.is_online and (user_last_seen_tz and user_last_seen_tz > online_threshold)
    
    # Format last seen time
    last_seen_text = "Never"
    if user.last_seen:
        # Convert to IST if needed
        user_last_seen = utc_to_ist(user.last_seen) if user.last_seen.tzinfo is None else user.last_seen
        time_diff = now - user_last_seen
        if time_diff.total_seconds() < 60:
            last_seen_text = "Just now"
        elif time_diff.total_seconds() < 3600:
            minutes = int(time_diff.total_seconds() / 60)
            last_seen_text = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif time_diff.total_seconds() < 86400:
            hours = int(time_diff.total_seconds() / 3600)
            last_seen_text = f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            last_seen_text = user_last_seen.strftime('%d %b %Y at %I:%M %p IST')
    
    return jsonify({
        'user_id': user.id,
        'username': user.username,
        'is_online': is_online,
        'last_seen': last_seen_text,
        'last_seen_timestamp': utc_to_ist(user.last_seen).isoformat() if user.last_seen else None
    })


@bp.route("/api/user/update-activity", methods=["POST"])
@login_required
def update_user_activity():
    """Update user's last seen timestamp"""
    current_user.last_seen = get_ist_now()
    current_user.is_online = True
    db.session.commit()
    return jsonify({'success': True})


@bp.route("/api/notification/<int:notification_id>/mark-read", methods=["POST"])
@login_required
def mark_notification_read(notification_id):
    """Mark a specific notification as read"""
    from models import Notification
    
    notification = Notification.query.get_or_404(notification_id)
    
    # Check if user owns this notification
    if notification.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    try:
        notification.is_read = True
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route("/api/notification/<int:notification_id>/delete", methods=["POST"])
@login_required
def delete_notification(notification_id):
    """Delete a specific notification"""
    from models import Notification
    
    notification = Notification.query.get_or_404(notification_id)
    
    # Check if user owns this notification
    if notification.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    try:
        db.session.delete(notification)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route("/api/notifications/clear-all", methods=["POST"])
@login_required
def clear_all_notifications():
    """Delete all notifications for current user"""
    from models import Notification
    
    try:
        Notification.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route("/api/notifications/count")
@login_required
def get_notification_count():
    """Get unread notification count for current user"""
    from models import Notification
    
    try:
        unread_count = Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
        return jsonify({'unread_count': unread_count})
    except Exception as e:
        return jsonify({'unread_count': 0, 'error': str(e)}), 500


@bp.route("/api/check-new-messages", methods=["POST"])
@login_required
def check_new_messages():
    """Check for new messages since last check"""
    try:
        from models import ChatMessage, ChatRoom
        import json
        
        data = request.get_json() or {}
        last_check = data.get('last_check', 0)
        
        # Convert timestamp to datetime
        from datetime import datetime
        last_check_time = datetime.fromtimestamp(last_check / 1000) if last_check else datetime.min
        
        # Get chat rooms for current user
        if current_user.is_admin:
            room_ids = [r.id for r in ChatRoom.query.filter_by(admin_id=current_user.id).all()]
        else:
            room_ids = [r.id for r in ChatRoom.query.filter_by(user_id=current_user.id).all()]
        
        if not room_ids:
            return jsonify({'new_messages': []})
        
        # Get new messages from other users
        new_messages = ChatMessage.query.filter(
            ChatMessage.chat_room_id.in_(room_ids),
            ChatMessage.sender_id != current_user.id,
            ChatMessage.created_at > last_check_time
        ).order_by(ChatMessage.created_at.desc()).limit(5).all()
        
        messages_data = []
        for msg in new_messages:
            messages_data.append({
                'id': msg.id,
                'chat_id': msg.chat_room_id,
                'sender_name': msg.sender.username,
                'content': msg.content or 'Sent a file',
                'created_at': msg.created_at.isoformat()
            })
        
        return jsonify({'new_messages': messages_data})
        
    except Exception as e:
        return jsonify({'new_messages': [], 'error': str(e)})


@bp.route("/api/trigger-auto-analysis", methods=["POST"])
@login_required
def trigger_auto_analysis():
    """Trigger automatic AI analysis for all footage"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        processor = AutoAIProcessor()
        processor.process_all_pending_footage()
        return jsonify({'success': True, 'message': 'Strict automatic analysis completed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route("/api/case/<int:case_id>/detections-debug")
@login_required
def debug_case_detections(case_id):
    """Debug endpoint to check detections for a case"""
    try:
        from models import LocationMatch, PersonDetection
        
        case = Case.query.get_or_404(case_id)
        
        # Check if user has access
        if case.user_id != current_user.id and not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        # Get location matches
        location_matches = LocationMatch.query.filter_by(case_id=case_id).all()
        
        debug_info = {
            'case_id': case_id,
            'case_name': case.person_name,
            'case_status': case.status,
            'location_matches_count': len(location_matches),
            'location_matches': [],
            'total_detections': 0
        }
        
        for match in location_matches:
            detections = PersonDetection.query.filter_by(location_match_id=match.id).all()
            debug_info['total_detections'] += len(detections)
            
            match_info = {
                'match_id': match.id,
                'footage_location': match.footage.location_name if match.footage else 'Unknown',
                'match_score': match.match_score,
                'status': match.status,
                'person_found': match.person_found,
                'detection_count': len(detections),
                'detections': []
            }
            
            for det in detections:
                match_info['detections'].append({
                    'id': det.id,
                    'timestamp': det.timestamp,
                    'confidence_score': det.confidence_score,
                    'frame_path': det.frame_path,
                    'analysis_method': det.analysis_method
                })
            
            debug_info['location_matches'].append(match_info)
        
        return jsonify(debug_info)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


def trigger_auto_analysis_for_footage(footage_id):
    """Auto-trigger analysis when new footage is uploaded"""
    try:
        processor = AutoAIProcessor()
        
        # Get all approved cases
        from models import Case
        approved_cases = Case.query.filter_by(status='Approved').all()
        
        for case in approved_cases:
            # Check if analysis already exists
            from models import LocationMatch
            existing = LocationMatch.query.filter_by(case_id=case.id, footage_id=footage_id).first()
            if not existing:
                print(f"Auto-analyzing Case {case.id} against new footage {footage_id}")
                processor.process_single_case_against_footage(case.id, footage_id)
        
        print(f"Auto-analysis completed for footage {footage_id}")
        
    except Exception as e:
        print(f"Auto-analysis failed for footage {footage_id}: {str(e)}")



