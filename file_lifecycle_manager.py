"""
Professional File Lifecycle Management System
Handles file storage, cleanup, archival, and compliance for large-scale operations
"""

import os
import shutil
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import json
from __init__ import db
from models import Case, TargetImage, SearchVideo

class FileStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived" 
    SCHEDULED_DELETE = "scheduled_delete"
    DELETED = "deleted"

class FileLifecycleManager:
    def __init__(self):
        self.base_path = Path("static")
        self.active_path = self.base_path / "uploads"
        self.archive_path = self.base_path / "archive"
        self.temp_path = self.base_path / "temp"
        
        # Create directories
        for path in [self.active_path, self.archive_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def organize_file_path(self, case_id, file_type, filename):
        """Simple flat structure in uploads folder"""
        return self.active_path / filename
    
    def store_file(self, file_obj, case_id, file_type="images"):
        """Store file with proper organization and metadata"""
        # Generate secure filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = Path(file_obj.filename).suffix.lower()
        filename = f"case_{case_id}_{file_type}_{timestamp}_{hashlib.md5(file_obj.filename.encode()).hexdigest()[:8]}{file_ext}"
        
        # Simple flat path
        file_path = self.organize_file_path(case_id, file_type, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_obj.save(str(file_path))
        
        # Create metadata
        metadata = {
            "original_name": file_obj.filename,
            "stored_name": filename,
            "case_id": case_id,
            "file_type": file_type,
            "size": file_path.stat().st_size,
            "created_at": datetime.now().isoformat(),
            "status": FileStatus.ACTIVE.value,
            "checksum": self._calculate_checksum(file_path)
        }
        
        # Save metadata
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(file_path.relative_to(self.base_path))
    
    def _calculate_checksum(self, file_path):
        """Calculate file checksum for integrity verification"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def cleanup_rejected_cases(self, days_old=30):
        """Clean up files from cases rejected more than X days ago"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        rejected_cases = Case.query.filter(
            Case.status == 'Rejected',
            Case.updated_at < cutoff_date
        ).all()
        
        cleaned_files = 0
        for case in rejected_cases:
            cleaned_files += self._archive_case_files(case.id)
        
        return cleaned_files
    
    def _archive_case_files(self, case_id):
        """Archive files for a specific case"""
        case = Case.query.get(case_id)
        if not case:
            return 0
        
        archived_count = 0
        
        # Archive images
        for image in case.target_images:
            if self._move_to_archive(image.image_path, case_id):
                archived_count += 1
        
        # Archive videos  
        for video in case.search_videos:
            if self._move_to_archive(video.video_path, case_id):
                archived_count += 1
        
        return archived_count
    
    def _move_to_archive(self, file_path, case_id):
        """Move file from active to archive storage"""
        try:
            source = self.base_path / file_path
            if not source.exists():
                return False
            
            # Create archive path
            archive_dest = self.archive_path / f"case_{case_id}" / source.name
            archive_dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file and metadata
            shutil.move(str(source), str(archive_dest))
            
            metadata_source = source.with_suffix('.json')
            if metadata_source.exists():
                metadata_dest = archive_dest.with_suffix('.json')
                shutil.move(str(metadata_source), str(metadata_dest))
                
                # Update metadata status
                with open(metadata_dest, 'r+') as f:
                    metadata = json.load(f)
                    metadata['status'] = FileStatus.ARCHIVED.value
                    metadata['archived_at'] = datetime.now().isoformat()
                    f.seek(0)
                    json.dump(metadata, f, indent=2)
                    f.truncate()
            
            return True
        except Exception as e:
            print(f"Error archiving file {file_path}: {e}")
            return False
    
    def get_storage_stats(self):
        """Get comprehensive storage statistics"""
        stats = {
            'active_files': 0,
            'active_size': 0,
            'archived_files': 0,
            'archived_size': 0,
            'by_case_type': {},
            'by_month': {}
        }
        
        # Scan active files
        for file_path in self.active_path.rglob('*'):
            if file_path.is_file() and not file_path.suffix == '.json':
                stats['active_files'] += 1
                stats['active_size'] += file_path.stat().st_size
        
        # Scan archived files
        for file_path in self.archive_path.rglob('*'):
            if file_path.is_file() and not file_path.suffix == '.json':
                stats['archived_files'] += 1
                stats['archived_size'] += file_path.stat().st_size
        
        return stats
    
    def schedule_deletion(self, case_id, deletion_date=None):
        """Schedule case files for deletion"""
        if not deletion_date:
            deletion_date = datetime.now() + timedelta(days=90)  # 90 days default
        
        case = Case.query.get(case_id)
        if not case:
            return False
        
        # Mark files for scheduled deletion
        for image in case.target_images:
            self._mark_for_deletion(image.image_path, deletion_date)
        
        for video in case.search_videos:
            self._mark_for_deletion(video.video_path, deletion_date)
        
        return True
    
    def _mark_for_deletion(self, file_path, deletion_date):
        """Mark individual file for scheduled deletion"""
        try:
            file_full_path = self.base_path / file_path
            metadata_path = file_full_path.with_suffix('.json')
            
            if metadata_path.exists():
                with open(metadata_path, 'r+') as f:
                    metadata = json.load(f)
                    metadata['status'] = FileStatus.SCHEDULED_DELETE.value
                    metadata['scheduled_deletion'] = deletion_date.isoformat()
                    f.seek(0)
                    json.dump(metadata, f, indent=2)
                    f.truncate()
            
            return True
        except Exception as e:
            print(f"Error marking file for deletion {file_path}: {e}")
            return False
    
    def execute_scheduled_deletions(self):
        """Execute all scheduled deletions that are due"""
        now = datetime.now()
        deleted_count = 0
        
        # Find all metadata files with scheduled deletions
        for metadata_path in self.base_path.rglob('*.json'):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if (metadata.get('status') == FileStatus.SCHEDULED_DELETE.value and
                    'scheduled_deletion' in metadata):
                    
                    deletion_time = datetime.fromisoformat(metadata['scheduled_deletion'])
                    if deletion_time <= now:
                        # Delete the actual file
                        file_path = metadata_path.with_suffix('')
                        if file_path.exists():
                            file_path.unlink()
                        
                        # Update metadata
                        metadata['status'] = FileStatus.DELETED.value
                        metadata['deleted_at'] = now.isoformat()
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        deleted_count += 1
            
            except Exception as e:
                print(f"Error processing scheduled deletion {metadata_path}: {e}")
        
        return deleted_count

# Global instance
file_manager = FileLifecycleManager()