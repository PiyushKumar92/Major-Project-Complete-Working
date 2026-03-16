"""
Template helper functions for consistent image handling
"""

def get_image_url(image_path):
    """
    Get proper image URL for templates
    Handles all path formats including organized folder structure
    """
    if not image_path:
        return None
    
    # Convert backslashes to forward slashes for URLs
    clean_path = str(image_path).replace('\\', '/')
    
    # Remove leading slash if present
    if clean_path.startswith('/'):
        clean_path = clean_path[1:]
    
    # Remove 'static/' prefix if present (for old format compatibility)
    if clean_path.startswith('static/'):
        clean_path = clean_path[7:]
    
    # Handle both old and new path formats:
    # Old: uploads/case_X_photo_....jpg
    # New: uploads/YYYY/MM/case_X/images/photo_....jpg
    if clean_path.startswith('uploads/'):
        # Path already starts with uploads/ - use as is
        return clean_path
    elif 'uploads' in clean_path:
        # Find uploads and use everything from there
        uploads_index = clean_path.find('uploads')
        return clean_path[uploads_index:]
    else:
        # Legacy format - prepend uploads/
        return f"uploads/{clean_path}"
    
    return clean_path

def get_primary_photo_url(case):
    """
    Get primary photo URL for a case
    Returns primary photo if marked, otherwise first photo, or None
    """
    if not hasattr(case, 'target_images') or not case.target_images:
        return None
    
    # Look for primary photo first
    for image in case.target_images:
        if hasattr(image, 'is_primary') and image.is_primary:
            return image.image_path
    
    # Fallback to first photo
    return case.target_images[0].image_path

def get_video_url(video_path):
    """
    Get proper video URL for templates
    Handles all path formats consistently
    """
    if not video_path:
        return None
    
    # Use same logic as images
    return get_image_url(video_path)

def verify_file_exists(file_path):
    """
    Check if file exists in the expected location
    Handles both old and new organized folder structures
    """
    import os
    
    if not file_path:
        return False
    
    # Normalize path for file system check
    normalized_path = file_path.replace('/', os.sep)
    
    # Try multiple possible locations for both old and new formats
    possible_paths = [
        f"static{os.sep}{normalized_path}",  # New organized format
        f"app{os.sep}static{os.sep}{normalized_path}",  # Alternative base
        normalized_path,  # Direct path
        f"static{os.sep}uploads{os.sep}{os.path.basename(normalized_path)}"  # Old flat format fallback
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return True
    
    return False