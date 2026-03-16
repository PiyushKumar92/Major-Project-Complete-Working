"""
Automatic Location Service - Background service for continuous monitoring
Runs independently to process cases and footage automatically
"""
import time
import threading
import logging
from datetime import datetime
from __init__ import create_app, db
from models import Case, SurveillanceFootage, LocationMatch
from advanced_location_matcher import advanced_matcher

logger = logging.getLogger(__name__)

class AutoLocationService:
    def __init__(self):
        self.app = create_app()
        self.running = False
        self.service_thread = None
        
    def start_service(self):
        """Start the automatic location service"""
        if self.running:
            return
        
        self.running = True
        self.service_thread = threading.Thread(target=self._service_loop, daemon=True)
        self.service_thread.start()
        logger.info("Auto Location Service started")
        
    def stop_service(self):
        """Stop the automatic location service"""
        self.running = False
        if self.service_thread:
            self.service_thread.join()
        logger.info("Auto Location Service stopped")
        
    def _service_loop(self):
        """Main service loop"""
        while self.running:
            try:
                with self.app.app_context():
                    # Process approved cases without matches
                    self._process_unmatched_cases()
                    
                    # Process new surveillance footage
                    self._process_new_footage()
                    
                # Sleep for 5 minutes before next cycle
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in service loop: {str(e)}")
                time.sleep(600)  # Wait 10 minutes on error
    
    def _process_unmatched_cases(self):
        """Process approved cases that don't have location matches yet"""
        try:
            # Get approved cases without location matches
            approved_cases = db.session.query(Case).filter(
                Case.status == 'Approved',
                ~Case.id.in_(
                    db.session.query(LocationMatch.case_id).distinct()
                )
            ).all()
            
            for case in approved_cases:
                try:
                    matches_created = advanced_matcher.auto_process_approved_case(case.id)
                    if matches_created > 0:
                        logger.info(f"Auto-processed case {case.id}: {matches_created} matches created")
                except Exception as case_error:
                    logger.error(f"Error processing case {case.id}: {str(case_error)}")
                    
        except Exception as e:
            print(f"Error processing unmatched cases: {str(e)}")
            # Continue running even if there are database issues
    
    def _process_new_footage(self):
        """Process newly uploaded surveillance footage"""
        try:
            # Get footage uploaded in last 10 minutes without matches
            from datetime import timedelta
            recent_time = datetime.utcnow() - timedelta(minutes=10)
            
            new_footage = db.session.query(SurveillanceFootage).filter(
                SurveillanceFootage.created_at >= recent_time,
                SurveillanceFootage.is_active == True,
                ~SurveillanceFootage.id.in_(
                    db.session.query(LocationMatch.footage_id).distinct()
                )
            ).all()
            
            for footage in new_footage:
                matches_created = advanced_matcher._process_new_footage()
                if matches_created > 0:
                    logger.info(f"Auto-processed footage {footage.id}: {matches_created} matches created")
                    
        except Exception as e:
            logger.error(f"Error processing new footage: {str(e)}")

# Global service instance
auto_service = AutoLocationService()

def start_auto_service():
    """Start the automatic location service"""
    auto_service.start_service()

def stop_auto_service():
    """Stop the automatic location service"""
    auto_service.stop_service()

if __name__ == "__main__":
    # Run as standalone service
    logging.basicConfig(level=logging.INFO)
    start_auto_service()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_auto_service()