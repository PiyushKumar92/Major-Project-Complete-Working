"""
File management utilities for secure file operations
"""
import os
import shutil
from datetime import datetime, timedelta
from flask import current_app
from __init__ import db
from models import Case, TargetImage, SearchVideo


def cleanup_orphaned_files():
    """
    Clean up files that are no longer referenced in the database
    """
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'app/static/uploads')
    
    if not os.path.exists(upload_folder):
        return
    
    # Get all files referenced in database
    referenced_files = set()
    
    # Add target image files
    for image in TargetImage.query.all():
        if image.image_path:
            filename = os.path.basename(image.image_path)
            referenced_files.add(filename)
    
    # Add search video files
    for video in SearchVideo.query.all():
        if video.video_path:
            filename = os.path.basename(video.video_path)
            referenced_files.add(filename)
    
    # Check for orphaned files
    orphaned_files = []
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path) and filename not in referenced_files:
            # Check if file is older than 24 hours (grace period)
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getctime(file_path))
            if file_age > timedelta(hours=24):
                orphaned_files.append(file_path)
    
    # Remove orphaned files
    for file_path in orphaned_files:
        try:
            os.remove(file_path)
        except OSError:
            pass
    
    return len(orphaned_files)


def get_upload_directory_size():
    """
    Get the total size of the upload directory in bytes
    """
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'app/static/uploads')
    
    if not os.path.exists(upload_folder):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(upload_folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass
    
    return total_size


def enforce_storage_limits():
    """
    Enforce storage limits by removing oldest files if necessary
    """
    max_storage = 1024 * 1024 * 1024 * 5  # 5GB limit
    current_size = get_upload_directory_size()
    
    if current_size <= max_storage:
        return 0
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'app/static/uploads')
    
    # Get all files with their creation times
    files_with_times = []
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path):
            try:
                ctime = os.path.getctime(file_path)
                size = os.path.getsize(file_path)
                files_with_times.append((file_path, ctime, size))
            except OSError:
                pass
    
    # Sort by creation time (oldest first)
    files_with_times.sort(key=lambda x: x[1])
    
    # Remove oldest files until under limit
    removed_count = 0
    for file_path, ctime, size in files_with_times:
        if current_size <= max_storage:
            break
        
        try:
            os.remove(file_path)
            current_size -= size
            removed_count += 1
        except OSError:
            pass
    
    return removed_count


def secure_file_operation(operation, *args, **kwargs):
    """
    Wrapper for file operations with error handling
    """
    try:
        return operation(*args, **kwargs)
    except (OSError, IOError) as e:
        current_app.logger.error(f"File operation failed: {e}")
        return None