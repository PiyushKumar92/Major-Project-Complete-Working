"""
Database models for Intelligent Footage Analysis System
"""

from datetime import datetime
from __init__ import db


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