"""
Clear old detections and re-run analysis with new strict settings
"""
from __init__ import db
from models import PersonDetection, LocationMatch
import os
import logging

logger = logging.getLogger(__name__)

def clear_detections_for_match(match_id):
    """Clear all detections for a specific match"""
    try:
        # Get all detections for this match
        detections = PersonDetection.query.filter_by(location_match_id=match_id).all()
        
        # Delete detection images
        for detection in detections:
            if detection.frame_path:
                frame_path = os.path.join('static', detection.frame_path)
                if not os.path.exists(frame_path):
                    frame_path = os.path.join('app', 'static', detection.frame_path)
                
                if os.path.exists(frame_path):
                    try:
                        os.remove(frame_path)
                        logger.info(f"Deleted frame: {frame_path}")
                    except:
                        pass
        
        # Delete detection records
        PersonDetection.query.filter_by(location_match_id=match_id).delete()
        
        # Reset match status
        match = LocationMatch.query.get(match_id)
        if match:
            match.status = 'pending'
            match.person_found = False
            match.confidence_score = 0.0
            match.detection_count = 0
            match.ai_analysis_started = None
            match.ai_analysis_completed = None
        
        db.session.commit()
        logger.info(f"Cleared {len(detections)} detections for match {match_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing detections: {e}")
        db.session.rollback()
        return False

def clear_all_detections():
    """Clear ALL detections from database"""
    try:
        # Get all detections
        detections = PersonDetection.query.all()
        
        # Delete detection images
        for detection in detections:
            if detection.frame_path:
                frame_path = os.path.join('static', detection.frame_path)
                if not os.path.exists(frame_path):
                    frame_path = os.path.join('app', 'static', detection.frame_path)
                
                if os.path.exists(frame_path):
                    try:
                        os.remove(frame_path)
                    except:
                        pass
        
        # Delete all detection records
        PersonDetection.query.delete()
        
        # Reset all matches
        matches = LocationMatch.query.all()
        for match in matches:
            match.status = 'pending'
            match.person_found = False
            match.confidence_score = 0.0
            match.detection_count = 0
            match.ai_analysis_started = None
            match.ai_analysis_completed = None
        
        db.session.commit()
        logger.info(f"Cleared {len(detections)} total detections")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing all detections: {e}")
        db.session.rollback()
        return False

if __name__ == "__main__":
    # Clear all detections
    print("Clearing all detections...")
    if clear_all_detections():
        print("✅ All detections cleared successfully!")
        print("Now you can re-run analysis with new strict settings.")
    else:
        print("❌ Failed to clear detections.")