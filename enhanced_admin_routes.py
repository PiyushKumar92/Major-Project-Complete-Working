"""
Enhanced Admin Routes for Surveillance Upload
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db, SurveillanceFootage, Case, Notification, get_ist_now
from admin import admin_required
import threading
import uuid
import os
from datetime import datetime

enhanced_admin_bp = Blueprint('enhanced_admin', __name__, url_prefix='/admin')

@enhanced_admin_bp.route("/enhanced-surveillance-upload", methods=["GET", "POST"])
@login_required
@admin_required
def enhanced_surveillance_upload():
    """Enhanced surveillance footage upload with large file support and AI analysis"""
    from enhanced_surveillance_upload import surveillance_uploader
    
    if request.method == "GET":
        return render_template("admin/enhanced_upload_surveillance.html")
    
    if request.method == "POST":
        try:
            # Get form data - Description and Landmarks are optional
            form_data = {
                'title': request.form.get('title'),
                'description': request.form.get('description', ''),  # Optional
                'location_name': request.form.get('location_name'),
                'location_address': request.form.get('location_address'),
                'city': request.form.get('city'),
                'state': request.form.get('state'),
                'pincode': request.form.get('pincode'),
                'landmarks': request.form.get('landmarks', ''),  # Optional
                'latitude': request.form.get('latitude'),  # Optional
                'longitude': request.form.get('longitude'),  # Optional
                'date_recorded': request.form.get('date_recorded'),
                'camera_type': request.form.get('camera_type'),
                'quality': request.form.get('quality', 'HD')
            }
            
            # Handle multiple file uploads
            files = request.files.getlist('video_files')
            if not files or all(f.filename == '' for f in files):
                return jsonify({'success': False, 'error': 'No video files selected'})
            
            # Validate file sizes - Support up to 10GB per file, 50GB total
            total_size = 0
            for f in files:
                if f.filename != '':
                    file_size = f.content_length or 0
                    if file_size > 10 * 1024 * 1024 * 1024:  # 10GB per file
                        return jsonify({
                            'success': False, 
                            'error': f'File {f.filename} exceeds 10GB limit'
                        })
                    total_size += file_size
            
            if total_size > 50 * 1024 * 1024 * 1024:  # 50GB total
                return jsonify({
                    'success': False, 
                    'error': f'Total file size ({total_size/1024/1024/1024:.1f}GB) exceeds 50GB limit'
                })
            
            # Generate upload ID
            upload_id = request.form.get('upload_id') or str(uuid.uuid4())
            
            # Prepare file data
            file_data = []
            for f in files:
                if f.filename != '':
                    file_data.append({'file': f, 'size': f.content_length or 0})
            
            # Start background processing with progress tracking
            thread = threading.Thread(
                target=process_enhanced_upload,
                args=(upload_id, file_data, form_data, current_user.id)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'upload_id': upload_id,
                'message': 'Upload started successfully',
                'files_count': len(file_data),
                'total_size_gb': round(total_size / (1024**3), 2)
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@enhanced_admin_bp.route("/upload-progress/<upload_id>", methods=["GET"])
@login_required
@admin_required
def get_upload_progress(upload_id):
    """Get real-time upload progress"""
    from enhanced_surveillance_upload import surveillance_uploader
    
    progress = surveillance_uploader.upload_progress.get(upload_id, {
        'status': 'not_found',
        'progress': 0,
        'message': 'Upload not found'
    })
    
    return jsonify(progress)

def process_enhanced_upload(upload_id, file_data, form_data, user_id):
    """Process upload with progress tracking and AI analysis"""
    from enhanced_surveillance_upload import surveillance_uploader
    from werkzeug.utils import secure_filename
    import cv2
    import time
    
    try:
        # Initialize progress
        surveillance_uploader.upload_progress[upload_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting upload...',
            'files_processed': 0,
            'total_files': len(file_data),
            'current_file': '',
            'estimated_time': 0
        }
        
        processed_files = []
        start_time = time.time()
        
        # Process each file
        for i, file_info in enumerate(file_data):
            file = file_info['file']
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"surveillance_{timestamp}_{i+1}_{filename}"
            
            # Update progress
            surveillance_uploader.upload_progress[upload_id].update({
                'progress': int((i / len(file_data)) * 70),  # 70% for upload
                'message': f'Uploading {filename}...',
                'current_file': filename,
                'files_processed': i
            })
            
            # Save file
            surveillance_dir = os.path.join('static', 'surveillance')
            os.makedirs(surveillance_dir, exist_ok=True)
            file_path = os.path.join(surveillance_dir, filename)
            file.save(file_path)
            
            # Get video metadata
            try:
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
                footage = SurveillanceFootage(
                    title=f"{form_data['title']} - Part {i+1}" if len(file_data) > 1 else form_data['title'],
                    description=form_data.get('description', ''),
                    location_name=form_data['location_name'],
                    location_address=form_data['location_address'],
                    latitude=float(form_data['latitude']) if form_data.get('latitude') else None,
                    longitude=float(form_data['longitude']) if form_data.get('longitude') else None,
                    video_path=f"surveillance/{filename}",
                    file_size=file_size,
                    duration=duration,
                    fps=fps,
                    resolution=resolution,
                    quality=form_data.get('quality', 'HD'),
                    date_recorded=datetime.strptime(form_data['date_recorded'], '%Y-%m-%dT%H:%M') if form_data.get('date_recorded') else None,
                    camera_type=form_data['camera_type'],
                    uploaded_by=user_id
                )
                
                db.session.add(footage)
                db.session.commit()
                
                processed_files.append({
                    'filename': filename,
                    'path': f"surveillance/{filename}",
                    'footage_id': footage.id,
                    'duration': duration,
                    'size_gb': round(file_size / (1024**3), 2)
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # AI Location Analysis
        surveillance_uploader.upload_progress[upload_id].update({
            'progress': 80,
            'message': 'AI analyzing location for case matching...'
        })
        
        # Find matching cases based on location
        matching_cases = find_location_matching_cases(form_data)
        
        # Send notifications for matching cases
        if matching_cases:
            for case in matching_cases:
                notification = Notification(
                    user_id=case['user_id'],
                    sender_id=user_id,
                    title=f"📹 New CCTV Footage: {case['person_name']}",
                    message=f"New surveillance footage uploaded for {form_data['location_name']} matches your case location. AI analysis will begin automatically.",
                    type="info",
                    created_at=get_ist_now()
                )
                db.session.add(notification)
            db.session.commit()
        
        # Complete
        elapsed_time = time.time() - start_time
        surveillance_uploader.upload_progress[upload_id].update({
            'status': 'completed',
            'progress': 100,
            'message': f'Upload completed successfully! {len(processed_files)} files processed in {elapsed_time:.1f}s',
            'files_processed': len(file_data),
            'processed_files': processed_files,
            'matching_cases': matching_cases,
            'total_duration_hours': round(sum(f.get('duration', 0) for f in processed_files) / 3600, 2)
        })
        
    except Exception as e:
        surveillance_uploader.upload_progress[upload_id].update({
            'status': 'error',
            'progress': 0,
            'message': f'Upload failed: {str(e)}'
        })

def find_location_matching_cases(form_data):
    """Find cases that match the footage location using AI"""
    try:
        # Extract location keywords
        location_keywords = []
        
        if form_data.get('location_name'):
            location_keywords.extend(form_data['location_name'].lower().split())
        if form_data.get('location_address'):
            location_keywords.extend(form_data['location_address'].lower().split())
        if form_data.get('city'):
            location_keywords.append(form_data['city'].lower())
        if form_data.get('landmarks'):
            location_keywords.extend([l.strip().lower() for l in form_data['landmarks'].split(',')])
        
        # Remove common words
        common_words = ['the', 'and', 'or', 'in', 'at', 'on', 'near', 'by', 'road', 'street']
        location_keywords = [k for k in location_keywords if k not in common_words and len(k) > 2]
        
        if not location_keywords:
            return []
        
        # Search in active cases
        active_cases = Case.query.filter(
            Case.status.in_(['Active', 'Under Investigation', 'Pending Approval'])
        ).all()
        
        matching_cases = []
        
        for case in active_cases:
            match_score = 0
            
            # Check last seen location
            if case.last_seen_location:
                location_lower = case.last_seen_location.lower()
                for keyword in location_keywords:
                    if keyword in location_lower:
                        match_score += 15
            
            # Check case description
            if case.description:
                desc_lower = case.description.lower()
                for keyword in location_keywords:
                    if keyword in desc_lower:
                        match_score += 8
            
            # Minimum threshold for matching
            if match_score >= 15:
                matching_cases.append({
                    'case_id': case.id,
                    'person_name': case.person_name,
                    'user_id': case.user_id,
                    'match_score': match_score,
                    'last_seen_location': case.last_seen_location,
                    'status': case.status
                })
        
        # Sort by match score
        matching_cases.sort(key=lambda x: x['match_score'], reverse=True)
        return matching_cases[:5]  # Top 5 matches
        
    except Exception as e:
        print(f"Case matching error: {e}")
        return []