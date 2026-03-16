from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import (
    StringField, 
    IntegerField, 
    TextAreaField, 
    DateTimeField, 
    SubmitField, 
    PasswordField, 
    BooleanField, 
    SelectField, 
    DateField, 
    EmailField,
    MultipleFileField,
    HiddenField
)
from wtforms.validators import DataRequired, Optional, Email, EqualTo, Length, ValidationError, NumberRange, Regexp
from models import User

# Custom file validators
def validate_image_files(form, field):
    """Custom validator for multiple image files"""
    # Photo requirements based on case type
    photo_required_types = [
        'missing_person', 'criminal_tracking', 'suspect_identification', 
        'witness_location', 'person_tracking', 'facial_recognition',
        'background_verification', 'fraud_investigation'
    ]
    
    if form.case_type and form.case_type.data in photo_required_types:
        if not field.data:
            raise ValidationError('At least one clear photo is required for this investigation type')
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    for file in field.data:
        if file and hasattr(file, 'filename') and file.filename:
            try:
                if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                    raise ValidationError(f'Invalid file type for "{file.filename}". Only PNG, JPG, JPEG, GIF, BMP, and WEBP files are allowed.')
            except (AttributeError, IndexError):
                raise ValidationError('Invalid file format detected.')

def validate_video_files(form, field):
    """Custom validator for multiple video files"""
    if field.data:
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
        
        for file in field.data:
            if file and hasattr(file, 'filename') and file.filename:
                try:
                    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                        raise ValidationError(f'Invalid file type for "{file.filename}". Only MP4, AVI, MOV, MKV, WMV, FLV, and WEBM files are allowed.')
                except (AttributeError, IndexError):
                    raise ValidationError('Invalid video file format detected.')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already taken')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

class NewCaseForm(FlaskForm):
    # Section 1: Case Type & Purpose
    case_type = SelectField('Investigation Type', choices=[
        ('missing_person', '👤 Missing Person Investigation'),
        ('criminal_tracking', '🔍 Criminal Tracking & Identification'),
        ('suspect_identification', '🎯 Suspect Identification'),
        ('witness_location', '👁️ Witness Location & Identification'),
        ('surveillance_analysis', '📹 Surveillance Footage Analysis'),
        ('person_tracking', '🗺️ Person Movement Tracking'),
        ('evidence_analysis', '🔬 Evidence & Photo Analysis'),
        ('location_verification', '📍 Location Verification'),
        ('crowd_analysis', '👥 Crowd & Event Analysis'),
        ('vehicle_tracking', '🚗 Vehicle & License Plate Tracking'),
        ('facial_recognition', '🤖 Facial Recognition Service'),
        ('security_investigation', '🛡️ Security Breach Investigation'),
        ('fraud_investigation', '💳 Fraud & Identity Investigation'),
        ('background_verification', '📋 Background Verification'),
        ('other_investigation', '❓ Other Investigation Type')
    ], validators=[DataRequired()])
    
    requester_type = SelectField('You are requesting as:', choices=[
        ('family_member', '👨‍👩‍👧‍👦 Family Member'),
        ('police_officer', '👮 Police Officer'),
        ('detective', '🕵️ Detective/Investigator'),
        ('security_agency', '🏢 Security Agency'),
        ('law_enforcement', '⚖️ Law Enforcement Agency'),
        ('government_official', '🏛️ Government Official'),
        ('private_investigator', '🔍 Private Investigator'),
        ('legal_firm', '⚖️ Legal Firm/Lawyer'),
        ('insurance_company', '🏢 Insurance Company'),
        ('corporate_security', '🏢 Corporate Security'),
        ('ngo_organization', '🤝 NGO/Humanitarian Organization'),
        ('media_journalist', '📰 Media/Journalist'),
        ('academic_researcher', '🎓 Academic/Researcher'),
        ('concerned_citizen', '👤 Concerned Citizen'),
        ('other_requester', '❓ Other')
    ], validators=[DataRequired()])
    
    case_category = SelectField('Specific Case Category', choices=[
        ('', 'Select Specific Category'),
        ('missing_child', '👶 Missing Child'),
        ('missing_adult', '👤 Missing Adult'),
        ('missing_elderly', '👴 Missing Elderly Person'),
        ('runaway_teen', '🏃 Runaway Teenager'),
        ('kidnapping_abduction', '⚠️ Kidnapping/Abduction'),
        ('human_trafficking', '🚨 Human Trafficking'),
        ('criminal_suspect', '🎯 Criminal Suspect'),
        ('wanted_fugitive', '🚨 Wanted Fugitive'),
        ('theft_robbery', '💰 Theft/Robbery Investigation'),
        ('assault_violence', '⚡ Assault/Violence Case'),
        ('fraud_scam', '💳 Fraud/Scam Investigation'),
        ('cybercrime', '💻 Cybercrime Investigation'),
        ('drug_trafficking', '💊 Drug Trafficking'),
        ('terrorism_threat', '💣 Terrorism/Security Threat'),
        ('witness_protection', '👁️ Witness Protection Case'),
        ('stalking_harassment', '📱 Stalking/Harassment'),
        ('domestic_violence', '🏠 Domestic Violence'),
        ('missing_evidence', '🔍 Missing Evidence'),
        ('identity_theft', '🆔 Identity Theft'),
        ('insurance_fraud', '📋 Insurance Fraud'),
        ('corporate_espionage', '🏢 Corporate Espionage'),
        ('background_check', '✅ Background Verification'),
        ('location_tracking', '📍 Location Tracking'),
        ('surveillance_breach', '📹 Surveillance Breach'),
        ('crowd_incident', '👥 Crowd Incident Analysis'),
        ('vehicle_accident', '🚗 Vehicle Accident Investigation'),
        ('property_crime', '🏠 Property Crime'),
        ('white_collar_crime', '💼 White Collar Crime'),
        ('organized_crime', '🕴️ Organized Crime'),
        ('cold_case', '❄️ Cold Case Investigation'),
        ('other_category', '❓ Other Category')
    ], validators=[DataRequired()])
    
    # Section 2: Personal Details
    full_name = StringField('Person Name/Subject', validators=[DataRequired(), Length(min=2, max=100)])
    nickname = StringField('Nickname/Alias (Optional)', validators=[Optional(), Length(max=50)])
    age = IntegerField('Age (if known)', validators=[Optional(), NumberRange(min=0, max=120)])
    gender = SelectField('Gender', choices=[('', 'Unknown/Not Specified'), ('male', 'Male'), ('female', 'Female'), ('other', 'Other')], validators=[Optional()])
    
    # Section 3: Physical Characteristics
    height_cm = IntegerField('Height (cm)', validators=[Optional(), NumberRange(min=30, max=250)])
    distinguishing_marks = TextAreaField('Physical Description & Distinguishing Marks', validators=[Optional(), Length(max=500)], render_kw={'placeholder': 'e.g., Height, build, hair color, scars, tattoos, clothing, etc.'})
    
    # Section 4: Contact Information (of the reporter)
    contact_person_name = StringField('Your Full Name', validators=[DataRequired(), Length(min=2, max=100)])
    contact_person_phone = StringField('Your Phone Number', validators=[DataRequired(), Regexp(r'^[\+]?[1-9]?\d{9,15}$', message='Please enter a valid phone number')])
    contact_person_email = EmailField('Your Email Address', validators=[DataRequired(), Email()])
    
    # Section 5: Investigation Details
    last_seen_date = DateField('Date of Incident/Last Seen', validators=[DataRequired()])
    last_seen_location = StringField('Location of Interest', validators=[DataRequired(), Length(min=5, max=200)])
    additional_info = TextAreaField('Case Details & Additional Information', validators=[Optional(), Length(max=1000)], render_kw={'placeholder': 'Provide detailed information about the case, circumstances, timeline, etc.'})
    
    # Section 6: Media Uploads
    photos = MultipleFileField('Upload Photos (Clear photos of subject/evidence/documents)', validators=[validate_image_files], render_kw={'accept': '.jpg,.jpeg,.png,.gif,.bmp,.webp'})
    video = MultipleFileField('Upload Reference Videos (Optional - Videos of the person for better AI analysis)', validators=[Optional(), validate_video_files], render_kw={'accept': '.mp4,.avi,.mov,.mkv,.wmv,.flv,.webm'})
    
    # Additional evidence fields
    urgency_level = SelectField('Urgency Level', choices=[
        ('low', '🟢 Low - Standard Processing'),
        ('medium', '🟡 Medium - Priority Processing'),
        ('high', '🟠 High - Urgent Processing'),
        ('critical', '🔴 Critical - Emergency Processing')
    ], default='medium', validators=[DataRequired()])
    
    case_description = TextAreaField('Detailed Case Description', validators=[DataRequired(), Length(min=20, max=2000)], render_kw={'placeholder': 'Provide comprehensive details about the case, timeline, circumstances, and any relevant information that can help with the investigation...'})
    
    submit = SubmitField('🚀 Submit Investigation Request')

class ContactForm(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField('Email Address', validators=[DataRequired(), Email()])
    subject = SelectField('Subject', choices=[
        ('technical_issue', 'Technical Issue'),
        ('case_inquiry', 'Case Status Inquiry'),
        ('investigation_help', 'Investigation Assistance'),
        ('evidence_submission', 'Evidence Submission'),
        ('law_enforcement', 'Law Enforcement Inquiry'),
        ('general_inquiry', 'General Inquiry'),
        ('feature_request', 'Feature Request'),
        ('bug_report', 'Bug Report')
    ], validators=[DataRequired()])
    message = TextAreaField('Message', validators=[DataRequired(), Length(min=10, max=1000)])
    submit = SubmitField('Send Message')

# Admin-only status update form
class AdminCaseStatusForm(FlaskForm):
    status = SelectField('Case Status', choices=[
        ('Pending Approval', 'Pending Approval'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
        ('Under Processing', 'Under Processing'),
        ('Case Solved', 'Case Solved ✅'),
        ('Case Over', 'Case Over 🔒'),
        ('Withdrawn', 'Withdrawn')
    ], validators=[DataRequired()])
    
    priority = SelectField('Priority Level', choices=[
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High'),
        ('Critical', 'Critical')
    ], validators=[DataRequired()])
    
    investigation_outcome = SelectField('Investigation Outcome', choices=[
        ('', 'Select outcome...'),
        ('Found Safe', 'Found Safe'),
        ('Found Deceased', 'Found Deceased'),
        ('Still Missing', 'Still Missing'),
        ('Criminal Identified', 'Criminal Identified'),
        ('Evidence Collected', 'Evidence Collected'),
        ('Witness Located', 'Witness Located'),
        ('Case Closed', 'Case Closed'),
        ('No Evidence Found', 'No Evidence Found'),
        ('Investigation Complete', 'Investigation Complete')
    ], validators=[Optional()])
    
    investigation_notes = TextAreaField('Admin Notes (Internal)', validators=[Optional(), Length(max=2000)], 
                                      render_kw={'placeholder': 'Internal notes for admin reference...'})
    
    admin_message = TextAreaField('Message to User', validators=[Optional(), Length(max=1000)], 
                                render_kw={'placeholder': 'Write a message for the user explaining the status change, findings, or next steps...'})
    
    submit = SubmitField('Update Case Status')

# Legacy form for backward compatibility
class CaseStatusForm(AdminCaseStatusForm):
    pass

# Legacy form for backward compatibility
class RegistrationCaseForm(NewCaseForm):
    pass
