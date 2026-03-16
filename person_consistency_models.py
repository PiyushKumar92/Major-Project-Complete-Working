"""
Database models for Person Consistency Validation
"""

from datetime import datetime
from __init__ import db

class PersonConsistencyValidation(db.Model):
    """Store person consistency validation results"""
    __tablename__ = 'person_consistency_validation'
    
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    
    # Validation Results
    is_consistent = db.Column(db.Boolean, nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    total_faces_found = db.Column(db.Integer, default=0)
    consistent_faces = db.Column(db.Integer, default=0)
    
    # File Analysis
    total_files_processed = db.Column(db.Integer, default=0)
    images_processed = db.Column(db.Integer, default=0)
    videos_processed = db.Column(db.Integer, default=0)
    
    # Issues and Recommendations
    inconsistent_files = db.Column(db.Text)  # JSON array of inconsistent files
    recommendations = db.Column(db.Text)  # JSON array of recommendations
    detailed_analysis = db.Column(db.Text)  # JSON object with detailed analysis
    
    # Primary Person Data
    primary_person_encoding = db.Column(db.Text)  # JSON array of face encoding
    
    # Validation Metadata
    validation_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    validation_method = db.Column(db.String(50), default="face_recognition")
    validation_threshold = db.Column(db.Float, default=0.6)
    
    # Relationships
    case = db.relationship("Case", backref="person_consistency_validations")
    
    def __repr__(self):
        return f"<PersonConsistencyValidation Case {self.case_id} - {'Consistent' if self.is_consistent else 'Inconsistent'} ({self.confidence_score:.2f})>"
    
    @property
    def inconsistent_files_list(self):
        """Get inconsistent files as list"""
        try:
            import json
            return json.loads(self.inconsistent_files) if self.inconsistent_files else []
        except:
            return []
    
    @property
    def recommendations_list(self):
        """Get recommendations as list"""
        try:
            import json
            return json.loads(self.recommendations) if self.recommendations else []
        except:
            return []
    
    @property
    def detailed_analysis_dict(self):
        """Get detailed analysis as dictionary"""
        try:
            import json
            return json.loads(self.detailed_analysis) if self.detailed_analysis else {}
        except:
            return {}
    
    @property
    def primary_person_encoding_list(self):
        """Get primary person encoding as list"""
        try:
            import json
            return json.loads(self.primary_person_encoding) if self.primary_person_encoding else []
        except:
            return []

class PersonConsistencyIssue(db.Model):
    """Store specific person consistency issues found"""
    __tablename__ = 'person_consistency_issue'
    
    id = db.Column(db.Integer, primary_key=True)
    validation_id = db.Column(db.Integer, db.ForeignKey("person_consistency_validation.id"), nullable=False)
    
    # Issue Details
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # image, video
    issue_type = db.Column(db.String(50), nullable=False)  # different_person, no_face, multiple_faces
    issue_description = db.Column(db.Text)
    
    # Similarity Metrics
    similarity_score = db.Column(db.Float)  # Similarity to primary person
    face_count = db.Column(db.Integer, default=0)
    
    # Resolution Status
    resolved = db.Column(db.Boolean, default=False)
    resolution_notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    validation = db.relationship("PersonConsistencyValidation", backref="issues")
    
    def __repr__(self):
        return f"<PersonConsistencyIssue {self.issue_type} - {self.file_path.split('/')[-1]}>"