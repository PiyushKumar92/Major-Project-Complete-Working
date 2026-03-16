import logging
import traceback
from datetime import datetime, timezone

from celery import Celery

# Configure logging for tasks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Flask app and create Celery instance
from celery_app import celery, app

# Import models within app context
with app.app_context():
    from models import Case, SystemLog
    from vision_engine import VisionProcessor
    from __init__ import db


@celery.task
def process_case(case_id):
    with app.app_context():
        case = Case.query.get(case_id)
        if not case:
            from utils import sanitize_log_input
            safe_case_id = sanitize_log_input(str(case_id))
            logger.error(f"Task failed: Case with ID {safe_case_id} not found.")
            return

        try:
            # Update case status to 'Processing'
            case.status = "Processing"
            db.session.commit()

            # Log the start of processing
            log_start = SystemLog(
                case_id=case_id,
                action="case_processing_started",
                details=f"Started processing case for {case.person_name}",
            )
            db.session.add(log_start)
            db.session.commit()

            # Run the main AI analysis
            processor = VisionProcessor(case_id)
            processor.run_analysis()

            # Update case status to 'Completed'
            case.status = "Completed"
            case.completed_at = datetime.now(timezone.utc)
            db.session.commit()

            # Log successful completion
            log_complete = SystemLog(
                case_id=case_id,
                action="case_processing_completed",
                details=f"Successfully completed processing. Found {len(case.sightings)} sightings.",
            )
            db.session.add(log_complete)
            db.session.commit()

        except Exception as e:
            # FIX 1: Roll back the failed transaction first. This is critical.
            db.session.rollback()
            
            # Now, safely update the database with the error status
            try:
                case.status = "Error"
                
                error_details = f"Error processing case {case_id}: {str(e)}\n{traceback.format_exc()}"
                log_error = SystemLog(
                    case_id=case_id,
                    action="case_processing_failed",
                    details=error_details,
                )
                db.session.add(log_error)
                db.session.commit()

            except Exception as db_error:
                # FIX 2: Use proper logging instead of print().
                from utils import sanitize_log_input
                safe_case_id = sanitize_log_input(str(case_id))
                safe_error = sanitize_log_input(str(e))
                safe_db_error = sanitize_log_input(str(db_error))
                logger.critical(f"CRITICAL: Failed to update database with error status for case {safe_case_id}.")
                logger.critical(f"Original Error: {safe_error}")
                logger.critical(f"DB Error: {safe_db_error}")

            # Re-raise the original exception so Celery knows the task failed
            raise


@celery.task
def analyze_footage_match(match_id):
    """Background task to analyze footage for a specific match"""
    with app.app_context():
        try:
            from aws_rekognition_matcher import aws_matcher
            from models import LocationMatch
            
            match = db.session.get(LocationMatch, match_id)
            if not match:
                logger.error(f"Match {match_id} not found")
                return False
            
            logger.info(f"Starting analysis for match {match_id}")
            result = aws_matcher.analyze_footage_for_person(match_id)
            
            if result:
                logger.info(f"Analysis completed for match {match_id}")
            else:
                logger.error(f"Analysis failed for match {match_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing match {match_id}: {e}", exc_info=True)
            return False


@celery.task
def cleanup_files():
    """Periodic task to clean up orphaned files and enforce storage limits"""
    with app.app_context():
        try:
            from file_manager import cleanup_orphaned_files, enforce_storage_limits
            
            # Clean up orphaned files
            orphaned_count = cleanup_orphaned_files()
            logger.info(f"Cleaned up {orphaned_count} orphaned files")
            
            # Enforce storage limits
            removed_count = enforce_storage_limits()
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old files to enforce storage limits")
            
            return f"Cleanup completed: {orphaned_count} orphaned, {removed_count} for storage"
            
        except ImportError as e:
            logger.error(f"File cleanup module import failed: {e}")
            return "Cleanup failed: Module not available"
        except Exception as e:
            logger.error(f"File cleanup failed: {e}", exc_info=True)
            raise e
