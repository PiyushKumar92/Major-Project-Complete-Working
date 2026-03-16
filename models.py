from datetime import datetime, timedelta
import pytz
from flask_login import UserMixin
from flask_bcrypt import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from flask import current_app
from __init__ import db
from utils import sanitize_input

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def utc_to_ist(utc_dt):
    """Convert UTC datetime to IST for display"""
    if utc_dt is None:
        return None
    if utc_dt.tzinfo is None:
        # Assume UTC if no timezone info
        utc_dt = pytz.UTC.localize(utc_dt)
    return utc_dt.astimezone(IST)


class Case(db.Model):
    __tablename__ = 'cases'
    id = db.Column(db.Integer, primary_key=True)
    case_type = db.Column(db.String(30), default="missing_person")  # missing_person, criminal_investigation, surveillance_request, person_tracking, evidence_analysis
    person_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    details = db.Column(db.Text)
    clothing_description = db.Column(db.Text)
    last_seen_location = db.Column(db.String(200))
    last_seen_time = db.Column(db.Time)  # Optional time field
    contact_address = db.Column(db.Text)  # Contact person address
    date_missing = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(
        db.String(30), default="Pending Approval"
    )  # Pending Approval, Approved, Rejected, Under Processing, Case Solved, Case Over, Withdrawn
    investigation_outcome = db.Column(db.String(50))  # Found Safe, Found Deceased, Still Missing, Criminal Identified, Evidence Collected, Case Closed, etc.
    admin_notes = db.Column(db.Text)  # Admin-only notes for case resolution
    admin_message = db.Column(db.Text)  # Admin message to user for status changes
    priority = db.Column(db.String(10), default="Medium")  # Low, Medium, High, Critical
    requester_type = db.Column(db.String(20), default="family")  # family, police, organization, private_investigator, government
    case_category = db.Column(db.String(30))  # voluntary_missing, abduction, criminal_activity, witness_identification, etc.
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    assigned_to = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    completed_at = db.Column(db.DateTime)
    investigation_notes = db.Column(db.Text)  # Admin investigation notes
    urgency_level = db.Column(db.String(20), default="medium")  # low, medium, high, critical

    # Relationships
    target_images = db.relationship(
        "TargetImage", backref="case", lazy=True, cascade="all, delete-orphan"
    )
    search_videos = db.relationship(
        "SearchVideo", backref="case", lazy=True, cascade="all, delete-orphan"
    )
    sightings = db.relationship(
        "Sighting", backref="case", lazy=True, cascade="all, delete-orphan"
    )
    case_notes = db.relationship(
        "CaseNote", backref="case", lazy=True, cascade="all, delete-orphan"
    )
    location_matches = db.relationship(
        "LocationMatch", back_populates="case", lazy=True, cascade="all, delete-orphan"
    )
    
    @property
    def latest_categorization(self):
        """Get the most recent categorization for this case"""
        return CaseCategorization.query.filter_by(case_id=self.id).order_by(CaseCategorization.categorized_at.desc()).first()
    
    @property
    def ai_detected_type(self):
        """Get AI-detected case type"""
        categorization = self.latest_categorization
        return categorization.detected_case_type if categorization else self.case_type
    
    @property
    def ai_risk_level(self):
        """Get AI-assessed risk level"""
        categorization = self.latest_categorization
        return categorization.risk_level if categorization else 'Medium'
    
    @property
    def ai_priority_score(self):
        """Get AI-calculated priority score"""
        categorization = self.latest_categorization
        return categorization.priority_score if categorization else 50.0
    
    @property
    def searchable_tags(self):
        """Get all searchable tags for this case"""
        categorization = self.latest_categorization
        return categorization.all_tags if categorization else []

    def __repr__(self):
        safe_name = sanitize_input(self.person_name) if self.person_name else 'Unknown'
        case_type_display = self.case_type.replace('_', ' ').title() if self.case_type else 'Investigation'
        return f"<{case_type_display} Case: {safe_name} - {self.status} ({self.priority})>"

    @property
    def total_sightings(self):
        return len(self.sightings)

    @property
    def high_confidence_sightings(self):
        return len([s for s in self.sightings if s.confidence_score > 0.8])
    
    @property
    def primary_photo(self):
        """Get the primary photo for this case (first uploaded photo)"""
        primary = next((img for img in self.target_images if img.is_primary), None)
        return primary or (self.target_images[0] if self.target_images else None)
    
    @property
    def user_visible_status(self):
        """Get user-friendly status for display to case owners"""
        status_mapping = {
            'Pending Approval': '[PENDING] Pending Approval',
            'Approved': '[APPROVED] Approved',
            'Rejected': '[REJECTED] Rejected', 
            'Under Processing': '[PROCESSING] Under Investigation',
            'Case Solved': '[SOLVED] Investigation Complete',
            'Case Over': '[CLOSED] Case Closed',
            'Withdrawn': '[WITHDRAWN] Withdrawn'
        }
        return status_mapping.get(self.status, self.status)
    
    @property
    def is_final_status(self):
        """Check if case is in final status (admin-only control)"""
        return self.status in FINAL_STATUSES
    
    @property
    def can_be_edited(self):
        """Check if case can be edited by user - allow editing withdrawn cases for resubmission"""
        return self.status not in FINAL_STATUSES
    
    @property
    def is_active_case(self):
        """Check if case is in active investigation"""
        return self.status in ACTIVE_STATUSES
    
    @property
    def is_public_visible(self):
        """Check if case should be visible to public"""
        return self.status in PUBLIC_VISIBLE_STATUSES
    
    @property
    def can_be_resubmitted(self):
        """Check if case can be resubmitted after editing"""
        return self.status in RESUBMITTABLE_STATUSES
    
    @property
    def status_color(self):
        """Get Bootstrap color class for status"""
        color_mapping = {
            'Pending Approval': 'warning',
            'Approved': 'info',
            'Rejected': 'danger',
            'Under Processing': 'primary',
            'Case Solved': 'success',
            'Case Over': 'secondary',
            'Withdrawn': 'dark'
        }
        return color_mapping.get(self.status, 'secondary')
    
    @property
    def status_icon(self):
        """Get Font Awesome icon for status"""
        icon_mapping = {
            'Pending Approval': 'fas fa-clock',
            'Approved': 'fas fa-check-circle',
            'Rejected': 'fas fa-times-circle',
            'Under Processing': 'fas fa-spinner fa-spin',
            'Case Solved': 'fas fa-trophy',
            'Case Over': 'fas fa-lock',
            'Withdrawn': 'fas fa-ban'
        }
        return icon_mapping.get(self.status, 'fas fa-question-circle')
    
    @property
    def status_emoji(self):
        """Get emoji for status"""
        emoji_mapping = {
            'Pending Approval': '⏳',
            'Approved': '✅',
            'Rejected': '❌',
            'Under Processing': '🔄',
            'Case Solved': '🎉',
            'Case Over': '🔒',
            'Withdrawn': '🚫'
        }
        return emoji_mapping.get(self.status, '❓')
    
    @property
    def status_priority_level(self):
        """Get numeric priority level for status sorting"""
        priority_mapping = {
            'Pending Approval': 1,
            'Approved': 2,
            'Rejected': 0,
            'Under Processing': 3,
            'Case Solved': 5,
            'Case Over': 4,
            'Withdrawn': 0
        }
        return priority_mapping.get(self.status, 0)


# Complete Status System - All possible case statuses
ALL_CASE_STATUSES = [
    'Pending Approval',
    'Approved', 
    'Rejected',
    'Under Processing',
    'Case Solved',
    'Case Over',
    'Withdrawn'
]

# Status constants for better management
CASE_STATUS_CHOICES = [
    ('Pending Approval', 'Pending Approval'),
    ('Approved', 'Approved'),
    ('Rejected', 'Rejected'),
    ('Under Processing', 'Under Processing'),
    ('Case Solved', 'Case Solved'),
    ('Case Over', 'Case Over'),
    ('Withdrawn', 'Withdrawn')
]

# Status categories for different access levels
PUBLIC_VISIBLE_STATUSES = ['Approved', 'Under Processing', 'Case Solved', 'Case Over']
USER_VISIBLE_STATUSES = ['Pending Approval', 'Approved', 'Rejected', 'Under Processing', 'Case Solved', 'Case Over', 'Withdrawn']
ADMIN_ONLY_STATUSES = ['Case Solved', 'Case Over']  # Only admin can set these
FINAL_STATUSES = ['Case Solved', 'Case Over']  # Cannot be edited by users
RESUBMITTABLE_STATUSES = ['Rejected', 'Withdrawn']  # Can be edited for resubmission
ACTIVE_STATUSES = ['Approved', 'Under Processing']  # Cases in active investigation

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Enhanced user fields
    last_login = db.Column(db.DateTime)
    login_count = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    location = db.Column(db.String(200))
    is_online = db.Column(db.Boolean, default=False)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships with cascade delete
    cases = db.relationship(
        "Case", foreign_keys="Case.user_id", backref="creator", lazy=True, cascade="all, delete-orphan"
    )
    assigned_cases = db.relationship(
        "Case", foreign_keys="Case.assigned_to", backref="assignee", lazy=True
    )
    
    @property
    def unread_notifications_count(self):
        """Get count of unread notifications for this user"""
        return Notification.query.filter_by(user_id=self.id, is_read=False).count()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_reset_token(self, expires_sec=1800):
        s = URLSafeTimedSerializer(current_app.config["SECRET_KEY"])
        return s.dumps({"user_id": self.id})

    @staticmethod
    def verify_reset_token(token, expires_sec=1800):
        s = URLSafeTimedSerializer(current_app.config["SECRET_KEY"])
        try:
            user_id = s.loads(token, max_age=expires_sec)["user_id"]
        except:
            return None
        return User.query.get(user_id)


class TargetImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    image_type = db.Column(
        db.String(20), default="front"
    )  # front, side, back, full_body
    description = db.Column(db.String(200))
    is_primary = db.Column(db.Boolean, default=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TargetImage {self.image_type} for Case {self.case_id}>"


class SearchVideo(db.Model):
    """Reference videos of the missing person for AI analysis (not surveillance footage)"""
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    video_path = db.Column(db.String(200), nullable=False)
    video_name = db.Column(db.String(100), nullable=False)
    video_type = db.Column(db.String(30), default="reference")  # reference, family, social_media, personal
    description = db.Column(db.Text)  # Description of video content
    duration = db.Column(db.Float)  # in seconds
    fps = db.Column(db.Float)
    resolution = db.Column(db.String(20))
    file_size = db.Column(db.BigInteger)  # in bytes
    status = db.Column(
        db.String(20), default="Pending"
    )  # Pending, Processing, Completed, Failed
    processed_at = db.Column(db.DateTime)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    sightings = db.relationship("Sighting", backref="search_video", lazy=True)

    def __repr__(self):
        safe_name = sanitize_input(self.video_name) if self.video_name else 'Unknown'
        return f"<ReferenceVideo {safe_name} for Case {self.case_id}>"


class Sighting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    search_video_id = db.Column(
        db.Integer, db.ForeignKey("search_video.id"), nullable=False
    )
    video_name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.Float, nullable=False)  # timestamp in video (seconds)
    confidence_score = db.Column(db.Float, nullable=False)  # combined confidence
    face_score = db.Column(db.Float)
    clothing_score = db.Column(db.Float)
    detection_method = db.Column(
        db.String(20), nullable=False
    )  # face, clothing, multi_modal
    thumbnail_path = db.Column(db.String(200))
    bounding_box = db.Column(db.Text)  # JSON string of coordinates
    verified = db.Column(db.Boolean, default=False)
    verified_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Sighting Case {self.case_id} at {self.timestamp}s - {self.confidence_score:.2f}>"

    @property
    def formatted_timestamp(self):
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"


class CaseNote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    note_type = db.Column(
        db.String(20), default="General"
    )  # General, Update, Evidence, Contact
    content = db.Column(db.Text, nullable=False)
    is_important = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    author = db.relationship("User", backref="case_notes")

    def __repr__(self):
        safe_type = sanitize_input(self.note_type) if self.note_type else 'Unknown'
        return f"<CaseNote {safe_type} for Case {self.case_id}>"


class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    action = db.Column(
        db.String(50), nullable=False
    )  # case_created, video_uploaded, sighting_found, etc.
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        safe_action = sanitize_input(self.action) if self.action else 'Unknown'
        return f"<SystemLog {safe_action} at {self.timestamp}>"


class AdminMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    recipient_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    sender = db.relationship("User", foreign_keys=[sender_id])
    recipient = db.relationship("User", foreign_keys=[recipient_id])


class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(20), default="info")  # info, warning, success, danger
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=get_ist_now)
    expires_at = db.Column(db.DateTime)
    
    author = db.relationship("User", backref="announcements")
    read_by = db.relationship("AnnouncementRead", backref="announcement", lazy=True, cascade="all, delete-orphan")
    
    @property
    def ist_created_at(self):
        """Get created_at in IST timezone for display"""
        if self.created_at.tzinfo is None:
            # If stored as naive datetime, assume it's already IST
            return self.created_at
        return utc_to_ist(self.created_at)


class AnnouncementRead(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    announcement_id = db.Column(db.Integer, db.ForeignKey("announcement.id"), nullable=False)
    read_at = db.Column(db.DateTime, default=get_ist_now)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'announcement_id', name='unique_user_announcement'),)
    
    @property
    def ist_read_at(self):
        """Get read_at in IST timezone for display"""
        if self.read_at.tzinfo is None:
            # If stored as naive datetime, assume it's already IST
            return self.read_at
        return utc_to_ist(self.read_at)


class BlogPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    slug = db.Column(db.String(200), unique=True, nullable=False)
    content = db.Column(db.Text, nullable=False)
    excerpt = db.Column(db.Text)
    is_published = db.Column(db.Boolean, default=False)
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author = db.relationship("User", backref="blog_posts")


class FAQ(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500), nullable=False)
    answer = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), default="General")
    order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    author = db.relationship("User", backref="faqs")


class AISettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    setting_name = db.Column(db.String(100), unique=True, nullable=False)
    setting_value = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    updated_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    updater = db.relationship("User", backref="ai_settings_updates")


class ContactMessage(db.Model):
    """Contact form messages from users"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        safe_name = sanitize_input(self.name) if self.name else 'Unknown'
        return f"<ContactMessage from {safe_name}>"


class ChatRoom(db.Model):
    """Chat room between user and admin"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_message_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    user = db.relationship("User", foreign_keys=[user_id], backref="user_chat_rooms")
    admin = db.relationship("User", foreign_keys=[admin_id], backref="admin_chat_rooms")
    messages = db.relationship("ChatMessage", backref="chat_room", lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatRoom {self.id} - User {self.user_id} & Admin {self.admin_id}>"
    
    @property
    def unread_count_for_user(self):
        return ChatMessage.query.filter_by(chat_room_id=self.id, is_read=False).filter(ChatMessage.sender_id != self.user_id).count()
    
    @property
    def unread_count_for_admin(self):
        return ChatMessage.query.filter_by(chat_room_id=self.id, is_read=False).filter(ChatMessage.sender_id != self.admin_id).count()


class ChatMessage(db.Model):
    """Individual chat messages"""
    id = db.Column(db.Integer, primary_key=True)
    chat_room_id = db.Column(db.Integer, db.ForeignKey("chat_room.id"), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    message_type = db.Column(db.String(20), default="text")  # text, image, video, file
    content = db.Column(db.Text)  # Text content or file path
    file_path = db.Column(db.String(500))  # For media files
    file_name = db.Column(db.String(200))  # Original filename
    is_read = db.Column(db.Boolean, default=False)
    status = db.Column(db.String(20), default="sent")  # sent, delivered, seen
    delivered_at = db.Column(db.DateTime)
    seen_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    hidden_for_user = db.Column(db.Boolean, default=False)  # Hidden for regular user
    hidden_for_admin = db.Column(db.Boolean, default=False)  # Hidden for admin
    
    # Relationships
    sender = db.relationship("User", backref="sent_chat_messages")
    
    def __repr__(self):
        return f"<ChatMessage {self.id} from User {self.sender_id}>"
    
    def mark_delivered(self):
        """Mark message as delivered"""
        if self.status == 'sent':
            self.status = 'delivered'
            self.delivered_at = get_ist_now()
            db.session.commit()
    
    def mark_seen(self):
        """Mark message as seen"""
        if self.status in ['sent', 'delivered']:
            self.status = 'seen'
            self.seen_at = get_ist_now()
            self.is_read = True
            db.session.commit()


class Notification(db.Model):
    """User notification system for admin messages and system alerts"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey("user.id"))  # Null for system messages
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(20), default="info")  # info, warning, success, danger, chat
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=get_ist_now)
    related_url = db.Column(db.String(200))  # For chat notifications
    
    # Relationships
    recipient = db.relationship("User", foreign_keys=[user_id])
    sender = db.relationship("User", foreign_keys=[sender_id])
    
    def __repr__(self):
        safe_title = sanitize_input(self.title) if self.title else 'Unknown'
        return f"<Notification {safe_title} for User {self.user_id}>"
    
    @property
    def ist_created_at(self):
        """Get created_at in IST timezone for display"""
        if self.created_at.tzinfo is None:
            # If stored as naive datetime, assume it's already IST
            return self.created_at
        return utc_to_ist(self.created_at)


class SurveillanceFootage(db.Model):
    """Admin uploaded surveillance footage for location-based searches"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    location_name = db.Column(db.String(200), nullable=False)
    location_address = db.Column(db.String(500))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    video_path = db.Column(db.String(500), nullable=False)
    thumbnail_path = db.Column(db.String(500))
    file_size = db.Column(db.BigInteger)  # in bytes
    duration = db.Column(db.Float)  # in seconds
    fps = db.Column(db.Float)
    resolution = db.Column(db.String(20))
    quality = db.Column(db.String(20), default="HD")  # SD, HD, FHD, 4K
    date_recorded = db.Column(db.DateTime)
    camera_type = db.Column(db.String(50))  # CCTV, Security, Traffic, etc.
    is_active = db.Column(db.Boolean, default=True)
    is_processed = db.Column(db.Boolean, default=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    uploader = db.relationship("User", backref="surveillance_footage")
    matches = db.relationship("LocationMatch", backref="footage", lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        safe_title = sanitize_input(self.title) if self.title else 'Unknown'
        return f"<SurveillanceFootage {safe_title} at {self.location_name}>"
    
    @property
    def formatted_duration(self):
        if not self.duration:
            return "Unknown"
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def formatted_file_size(self):
        if not self.file_size:
            return "Unknown"
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        elif self.file_size < 1024 * 1024 * 1024:
            return f"{self.file_size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.file_size / (1024 * 1024 * 1024):.1f} GB"


class LocationMatch(db.Model):
    """AI-powered location matches between cases and surveillance footage"""
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    footage_id = db.Column(db.Integer, db.ForeignKey("surveillance_footage.id"), nullable=False)
    match_score = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    distance_km = db.Column(db.Float)  # Distance between locations in km
    match_type = db.Column(db.String(20), default="location")  # location, proximity, exact
    status = db.Column(db.String(20), default="pending")  # pending, processing, completed, failed
    ai_analysis_started = db.Column(db.DateTime)
    ai_analysis_completed = db.Column(db.DateTime)
    person_found = db.Column(db.Boolean, default=False)
    confidence_score = db.Column(db.Float)  # AI confidence if person found
    detection_count = db.Column(db.Integer, default=0)  # Number of detections
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    case = db.relationship("Case", back_populates="location_matches")
    detections = db.relationship("PersonDetection", backref="location_match", lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<LocationMatch Case {self.case_id} - Footage {self.footage_id} ({self.match_score:.2f})>"


class PersonDetection(db.Model):
    """AI detection results from surveillance footage analysis with XAI and evidence integrity"""
    id = db.Column(db.Integer, primary_key=True)
    location_match_id = db.Column(db.Integer, db.ForeignKey("location_match.id"), nullable=False)
    timestamp = db.Column(db.Float, nullable=False)  # Video timestamp in seconds
    confidence_score = db.Column(db.Float, nullable=False)  # AI confidence 0.0-1.0
    
    # Individual feature scores for XAI
    face_match_score = db.Column(db.Float)  # Face recognition score
    clothing_match_score = db.Column(db.Float)  # Clothing analysis score
    body_pose_score = db.Column(db.Float)  # Body pose similarity
    temporal_consistency_score = db.Column(db.Float)  # Temporal consistency
    motion_pattern_score = db.Column(db.Float)  # Motion pattern analysis
    
    # XAI Feature Weighting Data
    feature_weights = db.Column(db.Text)  # JSON with feature weights breakdown
    decision_factors = db.Column(db.Text)  # JSON array of decision factors
    uncertainty_factors = db.Column(db.Text)  # JSON array of uncertainty factors
    confidence_category = db.Column(db.String(20), default="medium")  # low, medium, high, very_high
    
    # Evidence Integrity Fields
    detection_id = db.Column(db.String(32), unique=True)  # Unique detection identifier
    frame_hash = db.Column(db.String(64))  # SHA-256 hash of detection frame
    evidence_number = db.Column(db.String(20))  # Legal evidence number
    chain_hash = db.Column(db.String(64))  # Evidence chain hash
    
    # Quality Metrics
    frame_quality_score = db.Column(db.Float, default=0.0)
    face_visibility_score = db.Column(db.Float, default=0.0)
    lighting_quality_score = db.Column(db.Float, default=0.0)
    
    # Temporal Analysis
    detection_duration = db.Column(db.Float, default=0.0)  # Duration of detection sequence
    sequence_consistency = db.Column(db.Float, default=0.0)  # Consistency across sequence
    tracking_stability = db.Column(db.Float, default=0.0)  # Tracking stability score
    
    # Original fields
    detection_box = db.Column(db.Text)  # JSON bounding box coordinates
    frame_path = db.Column(db.String(500))  # Extracted frame image path
    analysis_method = db.Column(db.String(50))  # face_recognition, clothing_analysis, multi_modal
    verified = db.Column(db.Boolean, default=False)
    verified_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Legal and verification fields
    legal_status = db.Column(db.String(20), default="pending")  # pending, verified, court_ready
    verification_timestamp = db.Column(db.DateTime)
    requires_confirmation = db.Column(db.Boolean, default=False)
    
    # Relationships
    verifier = db.relationship("User", backref="verified_detections")
    
    def __repr__(self):
        return f"<PersonDetection {self.detection_id} - Match {self.location_match_id} at {self.timestamp}s ({self.confidence_score:.2f})>"
    
    @property
    def formatted_timestamp(self):
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def feature_weights_dict(self):
        """Get feature weights as dictionary"""
        try:
            import json
            return json.loads(self.feature_weights) if self.feature_weights else {}
        except:
            return {}
    
    @property
    def decision_factors_list(self):
        """Get decision factors as list"""
        try:
            import json
            return json.loads(self.decision_factors) if self.decision_factors else []
        except:
            return []
    
    @property
    def uncertainty_factors_list(self):
        """Get uncertainty factors as list"""
        try:
            import json
            return json.loads(self.uncertainty_factors) if self.uncertainty_factors else []
        except:
            return []
    
    @property
    def is_high_confidence(self):
        """Check if detection is high confidence"""
        return self.confidence_category in ['high', 'very_high']
    
    @property
    def evidence_integrity_verified(self):
        """Check if evidence integrity is verified"""
        return bool(self.frame_hash and self.evidence_number)
    
    def verify_frame_integrity(self):
        """Verify frame integrity using stored hash"""
        if not self.frame_path or not self.frame_hash:
            return False
        
        try:
            import cv2
            import hashlib
            import os
            
            if not os.path.exists(self.frame_path):
                return False
            
            # Recalculate hash
            frame = cv2.imread(self.frame_path)
            if frame is None:
                return False
            
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            current_hash = hashlib.sha256(frame_bytes).hexdigest()
            
            return current_hash == self.frame_hash
            
        except Exception:
            return False


class CaseQualityAssessment(db.Model):
    """ML-based case quality assessment results"""
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    overall_score = db.Column(db.Float, nullable=False)  # 0.0-1.0
    photo_quality_score = db.Column(db.Float, nullable=False)
    information_completeness_score = db.Column(db.Float, nullable=False)
    urgency_score = db.Column(db.Float, nullable=False)
    duplicate_risk_score = db.Column(db.Float, nullable=False)
    quality_grade = db.Column(db.String(5), nullable=False)  # A+, A, B, C, D, F
    processing_priority = db.Column(db.String(20), nullable=False)  # Critical, High, Medium, Low
    urgency_level = db.Column(db.String(20), nullable=False)  # Critical, High, Medium, Low
    duplicate_risk_level = db.Column(db.String(20), nullable=False)  # High, Medium, Low, Very Low
    estimated_success_rate = db.Column(db.Float, nullable=False)  # 0.0-1.0
    assessment_details = db.Column(db.Text)  # JSON string with detailed analysis
    recommendations = db.Column(db.Text)  # JSON array of recommendations
    similar_cases = db.Column(db.Text)  # JSON array of similar case IDs
    assessed_at = db.Column(db.DateTime, default=datetime.utcnow)
    assessed_by = db.Column(db.String(50), default="AI_System")
    
    # Relationships
    case = db.relationship("Case", backref="quality_assessments")
    
    def __repr__(self):
        return f"<CaseQualityAssessment Case {self.case_id} - Grade {self.quality_grade} ({self.overall_score:.2f})>"
    
    @property
    def recommendations_list(self):
        """Get recommendations as a list"""
        try:
            import json
            return json.loads(self.recommendations) if self.recommendations else []
        except:
            return []
    
    @property
    def similar_cases_list(self):
        """Get similar cases as a list"""
        try:
            import json
            return json.loads(self.similar_cases) if self.similar_cases else []
        except:
            return []
    
    @property
    def assessment_details_dict(self):
        """Get assessment details as a dictionary"""
        try:
            import json
            return json.loads(self.assessment_details) if self.assessment_details else {}
        except:
            return {}


class CaseCategorization(db.Model):
    """Intelligent case categorization results using NLP + ML"""
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    
    # Case Type Detection
    detected_case_type = db.Column(db.String(50), nullable=False)
    case_type_confidence = db.Column(db.Float, nullable=False)  # 0.0-1.0
    alternative_types = db.Column(db.Text)  # JSON array of alternative types with probabilities
    detection_method = db.Column(db.String(30), default="ml_classification")  # ml_classification, rule_based, hybrid
    
    # Risk Assessment
    risk_level = db.Column(db.String(20), nullable=False)  # Critical, High, Medium, Low
    risk_confidence = db.Column(db.Float, nullable=False)  # 0.0-1.0
    risk_factors = db.Column(db.Text)  # JSON array of identified risk factors
    risk_scores = db.Column(db.Text)  # JSON object with risk level scores
    
    # Priority Scoring
    priority_score = db.Column(db.Float, nullable=False)  # 0-100 scale
    priority_category = db.Column(db.String(20), nullable=False)  # Critical, High, Medium, Low
    priority_confidence = db.Column(db.Float, nullable=False)  # 0.0-1.0
    scoring_factors = db.Column(db.Text)  # JSON array of factors affecting priority
    recommended_sla = db.Column(db.String(20))  # Recommended response time
    
    # Tag Generation
    automatic_tags = db.Column(db.Text)  # JSON array of automatically generated tags
    entity_tags = db.Column(db.Text)  # JSON array of entity-based tags
    location_tags = db.Column(db.Text)  # JSON array of location-based tags
    temporal_tags = db.Column(db.Text)  # JSON array of time-based tags
    risk_tags = db.Column(db.Text)  # JSON array of risk-based tags
    category_tags = db.Column(db.Text)  # JSON array of category-based tags
    
    # Overall Assessment
    overall_confidence = db.Column(db.Float, nullable=False)  # 0.0-1.0
    recommendations = db.Column(db.Text)  # JSON array of actionable recommendations
    processing_notes = db.Column(db.Text)  # JSON array of internal processing notes
    
    # Metadata
    categorized_at = db.Column(db.DateTime, default=datetime.utcnow)
    categorized_by = db.Column(db.String(50), default="AI_Categorizer")
    model_version = db.Column(db.String(20), default="1.0")
    
    # Relationships
    case = db.relationship("Case", backref="categorizations")
    
    def __repr__(self):
        return f"<CaseCategorization Case {self.case_id} - Type: {self.detected_case_type} Risk: {self.risk_level} Priority: {self.priority_category}>"
    
    @property
    def alternative_types_list(self):
        """Get alternative case types as a list"""
        try:
            import json
            return json.loads(self.alternative_types) if self.alternative_types else []
        except:
            return []
    
    @property
    def risk_factors_list(self):
        """Get risk factors as a list"""
        try:
            import json
            return json.loads(self.risk_factors) if self.risk_factors else []
        except:
            return []
    
    @property
    def all_tags(self):
        """Get all tags combined"""
        all_tags = []
        tag_fields = ['automatic_tags', 'entity_tags', 'location_tags', 'temporal_tags', 'risk_tags', 'category_tags']
        
        for field in tag_fields:
            try:
                import json
                field_value = getattr(self, field)
                if field_value:
                    tags = json.loads(field_value)
                    all_tags.extend(tags)
            except:
                continue
        
        return list(set(all_tags))  # Remove duplicates
    
    @property
    def recommendations_list(self):
        """Get recommendations as a list"""
        try:
            import json
            return json.loads(self.recommendations) if self.recommendations else []
        except:
            return []
    
    @property
    def processing_notes_list(self):
        """Get processing notes as a list"""
        try:
            import json
            return json.loads(self.processing_notes) if self.processing_notes else []
        except:
            return []
    
    @property
    def scoring_factors_list(self):
        """Get scoring factors as a list"""
        try:
            import json
            return json.loads(self.scoring_factors) if self.scoring_factors else []
        except:
            return []


# ===== INTELLIGENT FOOTAGE ANALYSIS MODELS =====

class IntelligentFootageAnalysis(db.Model):
    """Intelligent footage analysis results with advanced computer vision"""
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    footage_id = db.Column(db.Integer, db.ForeignKey("surveillance_footage.id"), nullable=False)
    analysis_type = db.Column(db.String(30), default="comprehensive")
    
    # Multi-person tracking results
    persons_tracked = db.Column(db.Integer, default=0)
    tracking_duration = db.Column(db.Float)
    tracking_confidence = db.Column(db.Float)
    target_person_found = db.Column(db.Boolean, default=False)
    target_tracking_confidence = db.Column(db.Float)
    
    # Temporal analysis results
    movement_patterns_detected = db.Column(db.Text)
    unusual_behaviors_count = db.Column(db.Integer, default=0)
    loitering_detected = db.Column(db.Boolean, default=False)
    erratic_movement_detected = db.Column(db.Boolean, default=False)
    
    # Appearance change detection
    appearance_changes_detected = db.Column(db.Integer, default=0)
    clothing_changes = db.Column(db.Text)
    
    # Behavioral analysis
    behavioral_anomalies = db.Column(db.Text)
    interaction_events = db.Column(db.Integer, default=0)
    activity_classification = db.Column(db.Text)
    
    # Crowd analysis
    crowd_scenes_detected = db.Column(db.Integer, default=0)
    max_crowd_density = db.Column(db.Float, default=0.0)
    congestion_areas = db.Column(db.Text)
    person_extraction_quality = db.Column(db.Float)
    
    # Overall results
    analysis_confidence = db.Column(db.Float, nullable=False)
    processing_time = db.Column(db.Float)
    frames_processed = db.Column(db.Integer, default=0)
    detections_count = db.Column(db.Integer, default=0)
    high_confidence_detections = db.Column(db.Integer, default=0)
    
    # Analysis metadata
    analysis_started = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_completed = db.Column(db.DateTime)
    status = db.Column(db.String(20), default="pending")
    error_message = db.Column(db.Text)
    
    # Relationships
    case = db.relationship("Case", backref="intelligent_analyses")
    footage = db.relationship("SurveillanceFootage", backref="intelligent_analyses")
    tracking_results = db.relationship("PersonTrackingResult", backref="analysis", lazy=True, cascade="all, delete-orphan")
    behavioral_events = db.relationship("BehavioralEvent", backref="analysis", lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<IntelligentFootageAnalysis Case {self.case_id} - Footage {self.footage_id} ({self.analysis_confidence:.2f})>"


class PersonTrackingResult(db.Model):
    """Individual person tracking results across footage"""
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("intelligent_footage_analysis.id"), nullable=False)
    person_id = db.Column(db.String(50), nullable=False)
    
    # Tracking metrics
    first_appearance = db.Column(db.Float)
    last_appearance = db.Column(db.Float)
    total_duration = db.Column(db.Float)
    tracking_confidence = db.Column(db.Float, nullable=False)
    
    # Movement analysis
    trajectory_points = db.Column(db.Text)
    total_distance = db.Column(db.Float)
    average_speed = db.Column(db.Float)
    direction_changes = db.Column(db.Integer, default=0)
    
    # Appearance features
    appearance_features = db.Column(db.Text)
    clothing_colors = db.Column(db.Text)
    size_measurements = db.Column(db.Text)
    
    # Target person matching
    is_target_person = db.Column(db.Boolean, default=False)
    target_match_confidence = db.Column(db.Float)
    face_match_score = db.Column(db.Float)
    appearance_match_score = db.Column(db.Float)
    
    # Quality metrics
    visibility_score = db.Column(db.Float)
    occlusion_percentage = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PersonTrackingResult {self.person_id} - Analysis {self.analysis_id} ({self.tracking_confidence:.2f})>"


class BehavioralEvent(db.Model):
    """Detected behavioral events and anomalies"""
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("intelligent_footage_analysis.id"), nullable=False)
    person_id = db.Column(db.String(50))
    
    # Event details
    event_type = db.Column(db.String(30), nullable=False)
    event_description = db.Column(db.Text)
    timestamp = db.Column(db.Float, nullable=False)
    duration = db.Column(db.Float)
    
    # Event metrics
    confidence_score = db.Column(db.Float, nullable=False)
    severity_level = db.Column(db.String(20), default="medium")
    anomaly_score = db.Column(db.Float)
    
    # Location and context
    location_in_frame = db.Column(db.Text)
    context_data = db.Column(db.Text)
    
    # Verification
    verified = db.Column(db.Boolean, default=False)
    verified_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    verification_notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    verifier = db.relationship("User", backref="verified_behavioral_events")
    
    def __repr__(self):
        return f"<BehavioralEvent {self.event_type} at {self.timestamp}s - Analysis {self.analysis_id}>"
    
    @property
    def formatted_timestamp(self):
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"


class AppearanceChangeEvent(db.Model):
    """Detected appearance/clothing changes during tracking"""
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("intelligent_footage_analysis.id"), nullable=False)
    person_id = db.Column(db.String(50), nullable=False)
    
    # Change details
    change_type = db.Column(db.String(30), nullable=False)
    change_description = db.Column(db.Text)
    timestamp = db.Column(db.Float, nullable=False)
    
    # Before/after comparison
    before_features = db.Column(db.Text)
    after_features = db.Column(db.Text)
    similarity_score = db.Column(db.Float)
    
    # Change metrics
    confidence_score = db.Column(db.Float, nullable=False)
    change_magnitude = db.Column(db.Float)
    
    # Verification
    verified = db.Column(db.Boolean, default=False)
    verified_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    verification_notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = db.relationship("IntelligentFootageAnalysis", backref="appearance_changes")
    verifier = db.relationship("User", backref="verified_appearance_changes")
    
    def __repr__(self):
        return f"<AppearanceChangeEvent {self.change_type} - Person {self.person_id} at {self.timestamp}s>"
    
    @property
    def formatted_timestamp(self):
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"


class CrowdAnalysisResult(db.Model):
    """Crowd analysis results for specific time periods"""
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("intelligent_footage_analysis.id"), nullable=False)
    
    # Time period
    start_timestamp = db.Column(db.Float, nullable=False)
    end_timestamp = db.Column(db.Float, nullable=False)
    
    # Crowd metrics
    person_count = db.Column(db.Integer, nullable=False)
    crowd_density = db.Column(db.Float, nullable=False)
    crowd_flow_direction = db.Column(db.String(20))
    congestion_level = db.Column(db.String(20))
    
    # Crowd dynamics
    movement_coherence = db.Column(db.Float)
    average_speed = db.Column(db.Float)
    density_variance = db.Column(db.Float)
    
    # Extraction quality
    extraction_difficulty = db.Column(db.String(20))
    person_visibility_avg = db.Column(db.Float)
    occlusion_percentage = db.Column(db.Float)
    
    # Analysis results
    target_person_extractable = db.Column(db.Boolean, default=False)
    extraction_confidence = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = db.relationship("IntelligentFootageAnalysis", backref="crowd_results")
    
    def __repr__(self):
        return f"<CrowdAnalysisResult {self.person_count} persons - Analysis {self.analysis_id}>"
    
    @property
    def formatted_time_period(self):
        start_min = int(self.start_timestamp // 60)
        start_sec = int(self.start_timestamp % 60)
        end_min = int(self.end_timestamp // 60)
        end_sec = int(self.end_timestamp % 60)
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"


# ===== MULTI-MODAL PERSON RECOGNITION MODELS =====

class PersonProfile(db.Model):
    """Comprehensive person profile with multi-modal recognition data"""
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    
    # Facial Recognition Data
    primary_face_encoding = db.Column(db.Text)  # JSON array of face encoding
    all_face_encodings = db.Column(db.Text)  # JSON array of all encodings
    face_quality_score = db.Column(db.Float, default=0.0)
    age_progression_data = db.Column(db.Text)  # JSON with age features
    
    # Clothing Analysis Data
    dominant_colors = db.Column(db.Text)  # JSON array of dominant colors
    clothing_patterns = db.Column(db.Text)  # JSON with pattern analysis
    seasonal_category = db.Column(db.String(30))  # Seasonal clothing type
    texture_features = db.Column(db.Text)  # JSON with texture analysis
    
    # Biometric Features
    body_measurements = db.Column(db.Text)  # JSON with body measurements
    build_type = db.Column(db.String(30))  # Body build classification
    height_estimation = db.Column(db.Float)  # Estimated height in cm
    biometric_confidence = db.Column(db.Float, default=0.0)
    
    # Overall Profile Data
    profile_confidence = db.Column(db.Float, nullable=False)
    recognition_threshold = db.Column(db.Float, default=0.6)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    case = db.relationship("Case", backref="person_profiles")
    recognition_matches = db.relationship("RecognitionMatch", backref="profile", lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<PersonProfile Case {self.case_id} - Confidence {self.profile_confidence:.2f}>"
    
    @property
    def face_encodings_list(self):
        """Get face encodings as list"""
        try:
            import json
            return json.loads(self.all_face_encodings) if self.all_face_encodings else []
        except:
            return []
    
    @property
    def dominant_colors_list(self):
        """Get dominant colors as list"""
        try:
            import json
            return json.loads(self.dominant_colors) if self.dominant_colors else []
        except:
            return []
    
    @property
    def body_measurements_dict(self):
        """Get body measurements as dictionary"""
        try:
            import json
            return json.loads(self.body_measurements) if self.body_measurements else {}
        except:
            return {}


class RecognitionMatch(db.Model):
    """Multi-modal recognition matches from surveillance footage"""
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey("person_profile.id"), nullable=False)
    footage_id = db.Column(db.Integer, db.ForeignKey("surveillance_footage.id"), nullable=False)
    
    # Match Details
    timestamp = db.Column(db.Float, nullable=False)  # Video timestamp
    overall_confidence = db.Column(db.Float, nullable=False)
    match_type = db.Column(db.String(30), nullable=False)  # facial, clothing, biometric, multi_modal
    
    # Individual Confidence Scores
    face_match_confidence = db.Column(db.Float, default=0.0)
    clothing_match_confidence = db.Column(db.Float, default=0.0)
    biometric_match_confidence = db.Column(db.Float, default=0.0)
    age_progression_confidence = db.Column(db.Float, default=0.0)
    
    # Detection Data
    bounding_box = db.Column(db.Text)  # JSON with detection coordinates
    extracted_features = db.Column(db.Text)  # JSON with extracted features
    frame_path = db.Column(db.String(500))  # Path to extracted frame
    
    # Verification
    verified = db.Column(db.Boolean, default=False)
    verified_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    verification_notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    footage = db.relationship("SurveillanceFootage", backref="recognition_matches")
    verifier = db.relationship("User", backref="verified_recognition_matches")
    
    def __repr__(self):
        return f"<RecognitionMatch Profile {self.profile_id} - {self.match_type} ({self.overall_confidence:.2f})>"
    
    @property
    def formatted_timestamp(self):
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def bounding_box_dict(self):
        """Get bounding box as dictionary"""
        try:
            import json
            return json.loads(self.bounding_box) if self.bounding_box else {}
        except:
            return {}
    
    @property
    def extracted_features_dict(self):
        """Get extracted features as dictionary"""
        try:
            import json
            return json.loads(self.extracted_features) if self.extracted_features else {}
        except:
            return {}
