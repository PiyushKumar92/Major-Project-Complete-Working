"""
Storage Policies and Compliance Management
Defines retention policies, cleanup schedules, and compliance rules
"""

from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import json

class RetentionPolicy(Enum):
    IMMEDIATE = "immediate"           # Delete immediately after rejection
    SHORT_TERM = "short_term"        # 30 days retention
    MEDIUM_TERM = "medium_term"      # 90 days retention  
    LONG_TERM = "long_term"          # 1 year retention
    PERMANENT = "permanent"          # Keep indefinitely
    LEGAL_HOLD = "legal_hold"        # Cannot be deleted (legal cases)

class CaseCategory(Enum):
    MISSING_PERSON = "missing_person"
    CRIMINAL_INVESTIGATION = "criminal_investigation"
    FRAUD_CASE = "fraud_case"
    SURVEILLANCE_REQUEST = "surveillance_request"
    BACKGROUND_CHECK = "background_check"

@dataclass
class StoragePolicy:
    case_category: CaseCategory
    retention_policy: RetentionPolicy
    max_file_size_mb: int
    max_files_per_case: int
    auto_archive_days: int
    auto_delete_days: int
    requires_approval: bool
    compliance_notes: str

class StoragePolicyManager:
    def __init__(self):
        self.policies = self._initialize_policies()
    
    def _initialize_policies(self) -> Dict[CaseCategory, StoragePolicy]:
        """Initialize storage policies for different case types"""
        return {
            CaseCategory.MISSING_PERSON: StoragePolicy(
                case_category=CaseCategory.MISSING_PERSON,
                retention_policy=RetentionPolicy.LONG_TERM,
                max_file_size_mb=50,
                max_files_per_case=20,
                auto_archive_days=180,
                auto_delete_days=365,
                requires_approval=True,
                compliance_notes="Missing person cases require extended retention for ongoing investigations"
            ),
            
            CaseCategory.CRIMINAL_INVESTIGATION: StoragePolicy(
                case_category=CaseCategory.CRIMINAL_INVESTIGATION,
                retention_policy=RetentionPolicy.LEGAL_HOLD,
                max_file_size_mb=100,
                max_files_per_case=50,
                auto_archive_days=90,
                auto_delete_days=-1,  # Never auto-delete
                requires_approval=True,
                compliance_notes="Criminal cases may require indefinite retention for legal proceedings"
            ),
            
            CaseCategory.FRAUD_CASE: StoragePolicy(
                case_category=CaseCategory.FRAUD_CASE,
                retention_policy=RetentionPolicy.LONG_TERM,
                max_file_size_mb=75,
                max_files_per_case=30,
                auto_archive_days=120,
                auto_delete_days=730,  # 2 years
                requires_approval=True,
                compliance_notes="Fraud cases require extended retention for financial investigations"
            ),
            
            CaseCategory.SURVEILLANCE_REQUEST: StoragePolicy(
                case_category=CaseCategory.SURVEILLANCE_REQUEST,
                retention_policy=RetentionPolicy.MEDIUM_TERM,
                max_file_size_mb=200,  # Larger for video files
                max_files_per_case=15,
                auto_archive_days=60,
                auto_delete_days=90,
                requires_approval=False,
                compliance_notes="Surveillance requests have shorter retention due to privacy concerns"
            ),
            
            CaseCategory.BACKGROUND_CHECK: StoragePolicy(
                case_category=CaseCategory.BACKGROUND_CHECK,
                retention_policy=RetentionPolicy.SHORT_TERM,
                max_file_size_mb=25,
                max_files_per_case=10,
                auto_archive_days=30,
                auto_delete_days=90,
                requires_approval=False,
                compliance_notes="Background checks have limited retention for privacy protection"
            )
        }
    
    def get_policy(self, case_category: str) -> StoragePolicy:
        """Get storage policy for a case category"""
        try:
            category_enum = CaseCategory(case_category)
            return self.policies.get(category_enum, self.policies[CaseCategory.MISSING_PERSON])
        except ValueError:
            # Default to missing person policy for unknown categories
            return self.policies[CaseCategory.MISSING_PERSON]
    
    def get_retention_days(self, case_category: str, case_status: str) -> int:
        """Get retention days based on case category and status"""
        policy = self.get_policy(case_category)
        
        # Adjust retention based on case status
        status_multipliers = {
            'Pending Approval': 0.5,    # Shorter retention for pending
            'Rejected': 0.3,            # Even shorter for rejected
            'Approved': 1.0,            # Full retention
            'Under Investigation': 1.5,  # Extended retention
            'Completed': 1.0,           # Full retention
            'Withdrawn': 0.1            # Minimal retention
        }
        
        base_days = policy.auto_delete_days
        if base_days == -1:  # Never delete
            return -1
        
        multiplier = status_multipliers.get(case_status, 1.0)
        return int(base_days * multiplier)
    
    def validate_file_upload(self, case_category: str, file_size_mb: float, current_file_count: int) -> tuple[bool, str]:
        """Validate if file upload is allowed based on policy"""
        policy = self.get_policy(case_category)
        
        # Check file size
        if file_size_mb > policy.max_file_size_mb:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({policy.max_file_size_mb}MB)"
        
        # Check file count
        if current_file_count >= policy.max_files_per_case:
            return False, f"Maximum files per case ({policy.max_files_per_case}) exceeded"
        
        return True, "Upload allowed"
    
    def get_cleanup_schedule(self) -> Dict[str, List[Dict]]:
        """Get automated cleanup schedule for all policies"""
        schedule = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }
        
        for category, policy in self.policies.items():
            if policy.retention_policy == RetentionPolicy.IMMEDIATE:
                schedule['daily'].append({
                    'category': category.value,
                    'action': 'delete_rejected',
                    'days': 1
                })
            elif policy.retention_policy == RetentionPolicy.SHORT_TERM:
                schedule['weekly'].append({
                    'category': category.value,
                    'action': 'archive_old',
                    'days': policy.auto_archive_days
                })
            elif policy.retention_policy in [RetentionPolicy.MEDIUM_TERM, RetentionPolicy.LONG_TERM]:
                schedule['monthly'].append({
                    'category': category.value,
                    'action': 'archive_old',
                    'days': policy.auto_archive_days
                })
        
        return schedule
    
    def export_policies(self) -> str:
        """Export policies as JSON for documentation"""
        export_data = {}
        for category, policy in self.policies.items():
            export_data[category.value] = {
                'retention_policy': policy.retention_policy.value,
                'max_file_size_mb': policy.max_file_size_mb,
                'max_files_per_case': policy.max_files_per_case,
                'auto_archive_days': policy.auto_archive_days,
                'auto_delete_days': policy.auto_delete_days,
                'requires_approval': policy.requires_approval,
                'compliance_notes': policy.compliance_notes
            }
        
        return json.dumps(export_data, indent=2)

# Global instance
storage_policy_manager = StoragePolicyManager()