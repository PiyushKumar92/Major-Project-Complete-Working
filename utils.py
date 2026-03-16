"""
Utility functions for security and data processing
"""
import html
import os
import re
from markupsafe import Markup
from flask import current_app
from werkzeug.utils import secure_filename


def sanitize_input(text):
    """
    Sanitize user input to prevent XSS attacks
    """
    if not text:
        return text
    
    # HTML escape the input
    sanitized = html.escape(str(text))
    
    # Remove script tags and javascript
    patterns = [
        (r'<script[^>]*>.*?</script>', ''),
        (r'javascript:', ''),
        (r'on\w+\s*=', '')
    ]
    
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    return sanitized


def sanitize_log_input(text):
    """
    Sanitize input for logging to prevent log injection
    """
    if not text:
        return text
    
    # Remove newlines and carriage returns to prevent log injection
    sanitized = str(text).replace('\n', ' ').replace('\r', ' ')
    
    # Limit length to prevent log flooding
    if len(sanitized) > 500:
        sanitized = sanitized[:500] + '...'
    
    return sanitized


def validate_file_path(file_path, allowed_directory):
    """
    Validate file path to prevent path traversal attacks
    """
    import os
    from werkzeug.utils import secure_filename
    
    if not file_path:
        return None
    
    # Remove any path traversal attempts
    if '..' in file_path or file_path.startswith('/'):
        return None
    
    # Get just the filename
    filename = os.path.basename(file_path)
    secure_name = secure_filename(filename)
    
    if not secure_name:
        return None
    
    # Construct safe path
    safe_path = os.path.join(allowed_directory, secure_name)
    
    # Ensure the path is within the allowed directory
    if not os.path.abspath(safe_path).startswith(os.path.abspath(allowed_directory)):
        return None
    
    return safe_path


def create_safe_filename(base_name, extension=None):
    """
    Create a safe filename with timestamp
    """
    from datetime import datetime
    from werkzeug.utils import secure_filename
    import uuid
    
    # Use UUID for additional uniqueness
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    if extension:
        # Validate extension
        extension = extension.lower().strip('.')
        if not extension.isalnum() or len(extension) > 10:
            extension = 'bin'  # Default safe extension
        filename = f"{base_name}_{timestamp}_{unique_id}.{extension}"
    else:
        filename = f"{base_name}_{timestamp}_{unique_id}"
    
    return secure_filename(filename)


def validate_file_content(file_path, expected_type='image'):
    """
    Validate file content matches expected type (basic MIME type checking)
    """
    import mimetypes
    
    if not os.path.exists(file_path):
        return False
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if expected_type == 'image':
        return mime_type and mime_type.startswith('image/')
    elif expected_type == 'video':
        return mime_type and mime_type.startswith('video/')
    
    return False


def sanitize_filename(filename):
    """
    Additional filename sanitization beyond secure_filename
    """
    if not filename:
        return None
    
    # Remove any null bytes
    filename = filename.replace('\x00', '')
    
    # Limit filename length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    # Use secure_filename
    filename = secure_filename(filename)
    
    # Ensure we have a filename after sanitization
    if not filename:
        return None
    
    return filename
