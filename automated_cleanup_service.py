"""
Automated Cleanup Service
Runs scheduled cleanup tasks based on storage policies
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from file_lifecycle_manager import file_manager
from storage_policies import storage_policy_manager, CaseCategory
from models import Case, TargetImage, SearchVideo
from __init__ import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedCleanupService:
    def __init__(self):
        self.is_running = False
        self.stats = {
            'files_archived': 0,
            'files_deleted': 0,
            'space_freed_mb': 0,
            'last_run': None
        }
    
    def start_service(self):
        """Start the automated cleanup service"""
        logger.info("Starting Automated Cleanup Service")
        
        # Schedule different cleanup tasks
        schedule.every().day.at("02:00").do(self.daily_cleanup)
        schedule.every().week.at("03:00").do(self.weekly_cleanup)
        schedule.every().month.at("04:00").do(self.monthly_cleanup)
        
        # Emergency cleanup when storage is high
        schedule.every(6).hours.do(self.check_storage_limits)
        
        self.is_running = True
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_service(self):
        """Stop the cleanup service"""
        logger.info("Stopping Automated Cleanup Service")
        self.is_running = False
    
    def daily_cleanup(self):
        """Daily cleanup tasks"""
        logger.info("Starting daily cleanup")
        
        try:
            # Clean up rejected cases (immediate deletion policy)
            rejected_count = self._cleanup_rejected_cases(days_old=1)
            
            # Clean up temporary files
            temp_count = self._cleanup_temp_files()
            
            # Execute scheduled deletions
            deleted_count = file_manager.execute_scheduled_deletions()
            
            logger.info(f"Daily cleanup completed: {rejected_count} rejected cases, {temp_count} temp files, {deleted_count} scheduled deletions")
            
        except Exception as e:
            logger.error(f"Daily cleanup failed: {e}")
    
    def weekly_cleanup(self):
        """Weekly cleanup tasks"""
        logger.info("Starting weekly cleanup")
        
        try:
            # Archive old files based on policies
            archived_count = 0
            
            for category in CaseCategory:
                policy = storage_policy_manager.get_policy(category.value)
                if policy.auto_archive_days > 0:
                    count = self._archive_old_cases(category.value, policy.auto_archive_days)
                    archived_count += count
            
            # Clean up orphaned files
            orphaned_count = self._cleanup_orphaned_files()
            
            self.stats['files_archived'] += archived_count
            logger.info(f"Weekly cleanup completed: {archived_count} files archived, {orphaned_count} orphaned files cleaned")
            
        except Exception as e:
            logger.error(f"Weekly cleanup failed: {e}")
    
    def monthly_cleanup(self):
        """Monthly cleanup tasks"""
        logger.info("Starting monthly cleanup")
        
        try:
            # Deep cleanup based on retention policies
            deleted_count = 0
            
            for category in CaseCategory:
                policy = storage_policy_manager.get_policy(category.value)
                if policy.auto_delete_days > 0:
                    count = self._delete_old_cases(category.value, policy.auto_delete_days)
                    deleted_count += count
            
            # Generate cleanup report
            self._generate_cleanup_report()
            
            self.stats['files_deleted'] += deleted_count
            logger.info(f"Monthly cleanup completed: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"Monthly cleanup failed: {e}")
    
    def check_storage_limits(self):
        """Check storage limits and trigger emergency cleanup if needed"""
        try:
            stats = file_manager.get_storage_stats()
            total_size_gb = (stats['active_size'] + stats['archived_size']) / (1024**3)
            
            # Emergency cleanup if storage exceeds 80% of 100GB limit
            if total_size_gb > 80:
                logger.warning(f"Storage limit approaching: {total_size_gb:.2f}GB")
                self._emergency_cleanup()
            
        except Exception as e:
            logger.error(f"Storage check failed: {e}")
    
    def _cleanup_rejected_cases(self, days_old=30):
        """Clean up files from rejected cases"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        rejected_cases = Case.query.filter(
            Case.status == 'Rejected',
            Case.updated_at < cutoff_date
        ).all()
        
        cleaned_count = 0
        for case in rejected_cases:
            # Check if case category allows immediate deletion
            policy = storage_policy_manager.get_policy(case.case_type or 'missing_person')
            retention_days = storage_policy_manager.get_retention_days(case.case_type or 'missing_person', 'Rejected')
            
            if retention_days > 0 and retention_days <= days_old:
                if file_manager.schedule_deletion(case.id):
                    cleaned_count += 1
        
        return cleaned_count
    
    def _cleanup_temp_files(self):
        """Clean up temporary files older than 24 hours"""
        temp_path = file_manager.temp_path
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        cleaned_count = 0
        for file_path in temp_path.rglob('*'):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        return cleaned_count
    
    def _archive_old_cases(self, case_category, days_old):
        """Archive cases older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        old_cases = Case.query.filter(
            Case.case_type == case_category,
            Case.status.in_(['Approved', 'Completed']),
            Case.updated_at < cutoff_date
        ).all()
        
        archived_count = 0
        for case in old_cases:
            archived_count += file_manager._archive_case_files(case.id)
        
        return archived_count
    
    def _delete_old_cases(self, case_category, days_old):
        """Delete cases older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        old_cases = Case.query.filter(
            Case.case_type == case_category,
            Case.status.in_(['Completed', 'Withdrawn']),
            Case.updated_at < cutoff_date
        ).all()
        
        deleted_count = 0
        for case in old_cases:
            if file_manager.schedule_deletion(case.id):
                deleted_count += 1
        
        return deleted_count
    
    def _cleanup_orphaned_files(self):
        """Clean up files that don't have corresponding database records"""
        orphaned_count = 0
        
        # Check all files in uploads directory
        for file_path in file_manager.active_path.rglob('*'):
            if file_path.is_file() and not file_path.suffix == '.json':
                # Extract case_id from filename
                try:
                    filename = file_path.name
                    if filename.startswith('case_'):
                        case_id = int(filename.split('_')[1])
                        
                        # Check if case exists
                        case = Case.query.get(case_id)
                        if not case:
                            # Check if file is referenced in database
                            relative_path = str(file_path.relative_to(file_manager.base_path))
                            
                            image_exists = TargetImage.query.filter_by(image_path=relative_path).first()
                            video_exists = SearchVideo.query.filter_by(video_path=relative_path).first()
                            
                            if not image_exists and not video_exists:
                                file_path.unlink()
                                # Also remove metadata if exists
                                metadata_path = file_path.with_suffix('.json')
                                if metadata_path.exists():
                                    metadata_path.unlink()
                                orphaned_count += 1
                
                except (ValueError, IndexError):
                    continue  # Skip files that don't match expected pattern
        
        return orphaned_count
    
    def _emergency_cleanup(self):
        """Emergency cleanup when storage is critically high"""
        logger.warning("Initiating emergency cleanup")
        
        # Aggressively clean up rejected cases
        self._cleanup_rejected_cases(days_old=7)
        
        # Archive completed cases immediately
        for category in CaseCategory:
            self._archive_old_cases(category.value, days_old=30)
        
        # Clean up all temp files
        self._cleanup_temp_files()
        
        logger.info("Emergency cleanup completed")
    
    def _generate_cleanup_report(self):
        """Generate monthly cleanup report"""
        stats = file_manager.get_storage_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'storage_stats': stats,
            'cleanup_stats': self.stats,
            'policies_applied': storage_policy_manager.export_policies()
        }
        
        # Save report
        report_path = file_manager.base_path / 'reports' / f"cleanup_report_{datetime.now().strftime('%Y%m')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cleanup report generated: {report_path}")
    
    def get_service_status(self):
        """Get current service status and statistics"""
        return {
            'is_running': self.is_running,
            'stats': self.stats,
            'storage_stats': file_manager.get_storage_stats(),
            'next_scheduled_runs': {
                'daily': schedule.next_run(),
                'weekly': schedule.next_run(),
                'monthly': schedule.next_run()
            }
        }

# Global service instance
cleanup_service = AutomatedCleanupService()

if __name__ == "__main__":
    # Run as standalone service
    cleanup_service.start_service()