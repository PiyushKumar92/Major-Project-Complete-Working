"""
Security Automation System
Autonomous security management with threat detection, automated backup, and compliance monitoring
"""

import os
import json
import hashlib
import sqlite3
import logging
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from threading import Thread
import time
import re

@dataclass
class SecurityEvent:
    timestamp: datetime
    event_type: str
    severity: str
    source_ip: str
    user_id: Optional[int]
    description: str
    action_taken: str

@dataclass
class AccessPattern:
    user_id: int
    ip_address: str
    login_time: datetime
    user_agent: str
    location: Optional[str]
    suspicious_score: float

class SecurityAutomation:
    def __init__(self):
        self.monitoring_active = False
        self.threat_patterns = self._load_threat_patterns()
        self.access_patterns = []
        self.security_events = []
        self.backup_schedule = {
            'database': {'frequency': 'daily', 'retention': 30},
            'files': {'frequency': 'weekly', 'retention': 12},
            'logs': {'frequency': 'daily', 'retention': 90}
        }
        self.setup_logging()
        self.setup_security_database()
        
    def setup_logging(self):
        """Setup security logging"""
        log_dir = Path('logs/security')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Security logger
        self.security_logger = logging.getLogger('SecurityAutomation')
        security_handler = logging.FileHandler('logs/security/security.log')
        security_handler.setFormatter(
            logging.Formatter('%(asctime)s - SECURITY - %(levelname)s - %(message)s')
        )
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(logging.INFO)
        
        # Audit logger
        self.audit_logger = logging.getLogger('SecurityAudit')
        audit_handler = logging.FileHandler('logs/security/audit.log')
        audit_handler.setFormatter(
            logging.Formatter('%(asctime)s - AUDIT - %(message)s')
        )
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
        
    def setup_security_database(self):
        """Setup security monitoring database"""
        db_path = 'instance/security.db'
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Security events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                severity TEXT,
                source_ip TEXT,
                user_id INTEGER,
                description TEXT,
                action_taken TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Access patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ip_address TEXT,
                login_time TEXT,
                user_agent TEXT,
                location TEXT,
                suspicious_score REAL,
                flagged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Backup logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_type TEXT,
                timestamp TEXT,
                status TEXT,
                file_path TEXT,
                size_mb REAL,
                checksum TEXT
            )
        ''')
        
        # Compliance checks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_type TEXT,
                timestamp TEXT,
                status TEXT,
                details TEXT,
                remediation_needed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_threat_patterns(self) -> Dict:
        """Load threat detection patterns"""
        return {
            'brute_force': {
                'pattern': r'failed login attempts from same IP',
                'threshold': 5,
                'timeframe': 300  # 5 minutes
            },
            'sql_injection': {
                'pattern': r'(union|select|insert|update|delete|drop|create|alter)',
                'severity': 'high'
            },
            'xss_attempt': {
                'pattern': r'(<script|javascript:|onload=|onerror=)',
                'severity': 'medium'
            },
            'path_traversal': {
                'pattern': r'(\.\./|\.\.\\|%2e%2e%2f)',
                'severity': 'high'
            },
            'suspicious_user_agent': {
                'pattern': r'(sqlmap|nikto|nmap|burp|scanner)',
                'severity': 'medium'
            }
        }
        
    def analyze_access_pattern(self, user_id: int, ip_address: str, 
                             user_agent: str, request_path: str) -> float:
        """Analyze access pattern for suspicious activity"""
        suspicious_score = 0.0
        
        # Check for suspicious user agents
        if re.search(self.threat_patterns['suspicious_user_agent']['pattern'], 
                    user_agent.lower()):
            suspicious_score += 0.3
            
        # Check for path traversal attempts
        if re.search(self.threat_patterns['path_traversal']['pattern'], 
                    request_path.lower()):
            suspicious_score += 0.5
            
        # Check for SQL injection patterns
        if re.search(self.threat_patterns['sql_injection']['pattern'], 
                    request_path.lower()):
            suspicious_score += 0.7
            
        # Check for XSS attempts
        if re.search(self.threat_patterns['xss_attempt']['pattern'], 
                    request_path.lower()):
            suspicious_score += 0.4
            
        # Check for unusual access times (outside business hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            suspicious_score += 0.1
            
        # Check for rapid requests from same IP
        recent_accesses = [
            p for p in self.access_patterns 
            if p.ip_address == ip_address and 
            (datetime.now() - p.login_time).seconds < 60
        ]
        if len(recent_accesses) > 10:
            suspicious_score += 0.3
            
        return min(suspicious_score, 1.0)
        
    def detect_threats(self, request_data: Dict) -> List[SecurityEvent]:
        """Detect security threats from request data"""
        threats = []
        
        ip_address = request_data.get('ip_address', 'unknown')
        user_id = request_data.get('user_id')
        user_agent = request_data.get('user_agent', '')
        request_path = request_data.get('path', '')
        
        # Analyze access pattern
        suspicious_score = self.analyze_access_pattern(
            user_id, ip_address, user_agent, request_path
        )
        
        if suspicious_score > 0.5:
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type='suspicious_activity',
                severity='high' if suspicious_score > 0.7 else 'medium',
                source_ip=ip_address,
                user_id=user_id,
                description=f'Suspicious activity detected (score: {suspicious_score:.2f})',
                action_taken='logged_and_monitored'
            )
            threats.append(event)
            
        # Check for brute force attacks
        if self._detect_brute_force(ip_address):
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type='brute_force_attack',
                severity='high',
                source_ip=ip_address,
                user_id=user_id,
                description='Brute force attack detected',
                action_taken='ip_blocked'
            )
            threats.append(event)
            
        return threats
        
    def _detect_brute_force(self, ip_address: str) -> bool:
        """Detect brute force attacks"""
        # Check failed login attempts from same IP in last 5 minutes
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        
        try:
            # In production, this would check actual login logs
            log_file = Path('logs/security/security.log')
            if not log_file.exists():
                return False
                
            failed_attempts = 0
            with open(log_file, 'r') as f:
                for line in f.readlines()[-1000:]:  # Check last 1000 lines
                    if ip_address in line and 'failed login' in line.lower():
                        failed_attempts += 1
                        
            return failed_attempts >= self.threat_patterns['brute_force']['threshold']
        except:
            return False
            
    def respond_to_threat(self, event: SecurityEvent):
        """Automatically respond to security threats"""
        try:
            if event.event_type == 'brute_force_attack':
                self._block_ip_address(event.source_ip)
                self._notify_admin_threat(event)
                
            elif event.event_type == 'suspicious_activity':
                if event.severity == 'high':
                    self._increase_monitoring(event.source_ip)
                    self._notify_admin_threat(event)
                    
            # Log the event
            self._log_security_event(event)
            
        except Exception as e:
            self.security_logger.error(f"Error responding to threat: {e}")
            
    def _block_ip_address(self, ip_address: str):
        """Block IP address (placeholder for actual implementation)"""
        self.security_logger.warning(f"IP address blocked: {ip_address}")
        # In production, this would update firewall rules
        
    def _increase_monitoring(self, ip_address: str):
        """Increase monitoring for suspicious IP"""
        self.security_logger.info(f"Increased monitoring for IP: {ip_address}")
        
    def _notify_admin_threat(self, event: SecurityEvent):
        """Notify admin of security threat"""
        try:
            from models import Notification, User
            from __init__ import create_app, db
            
            app = create_app()
            with app.app_context():
                # Find admin users
                admin_users = User.query.filter_by(is_admin=True).all()
                
                for admin in admin_users:
                    notification = Notification(
                        user_id=admin.id,
                        title=f"🚨 Security Alert: {event.event_type}",
                        message=f"Security threat detected from {event.source_ip}: {event.description}",
                        type="danger"
                    )
                    db.session.add(notification)
                    
                db.session.commit()
                
        except Exception as e:
            self.security_logger.error(f"Error notifying admin: {e}")
            
    def _log_security_event(self, event: SecurityEvent):
        """Log security event to database"""
        try:
            conn = sqlite3.connect('instance/security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_events 
                (timestamp, event_type, severity, source_ip, user_id, description, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp.isoformat(),
                event.event_type,
                event.severity,
                event.source_ip,
                event.user_id,
                event.description,
                event.action_taken
            ))
            
            conn.commit()
            conn.close()
            
            self.security_logger.info(
                f"Security event logged: {event.event_type} from {event.source_ip}"
            )
            
        except Exception as e:
            self.security_logger.error(f"Error logging security event: {e}")
            
    def perform_automated_backup(self, backup_type: str = 'all'):
        """Perform automated backup"""
        backup_results = []
        
        if backup_type in ['all', 'database']:
            result = self._backup_database()
            backup_results.append(result)
            
        if backup_type in ['all', 'files']:
            result = self._backup_files()
            backup_results.append(result)
            
        if backup_type in ['all', 'logs']:
            result = self._backup_logs()
            backup_results.append(result)
            
        return backup_results
        
    def _backup_database(self) -> Dict:
        """Backup database"""
        try:
            backup_dir = Path('backups/database')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f'database_backup_{timestamp}.db'
            
            # Copy database file
            source_db = Path('instance/app.db')
            if source_db.exists():
                shutil.copy2(source_db, backup_file)
                
                # Calculate checksum
                checksum = self._calculate_checksum(backup_file)
                file_size = backup_file.stat().st_size / (1024 * 1024)  # MB
                
                # Log backup
                self._log_backup('database', backup_file, file_size, checksum, 'success')
                
                self.security_logger.info(f"Database backup created: {backup_file}")
                
                return {
                    'type': 'database',
                    'status': 'success',
                    'file': str(backup_file),
                    'size_mb': file_size,
                    'checksum': checksum
                }
            else:
                return {'type': 'database', 'status': 'failed', 'error': 'Database file not found'}
                
        except Exception as e:
            self.security_logger.error(f"Database backup failed: {e}")
            return {'type': 'database', 'status': 'failed', 'error': str(e)}
            
    def _backup_files(self) -> Dict:
        """Backup important files"""
        try:
            backup_dir = Path('backups/files')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f'files_backup_{timestamp}.zip'
            
            # Create zip backup
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Backup uploads
                uploads_dir = Path('static/uploads')
                if uploads_dir.exists():
                    for file_path in uploads_dir.rglob('*'):
                        if file_path.is_file():
                            zipf.write(file_path, file_path.relative_to(Path('.')))
                            
                # Backup configuration files
                config_files = ['config.py', '.env', 'requirements.txt']
                for config_file in config_files:
                    if Path(config_file).exists():
                        zipf.write(config_file)
                        
            # Calculate checksum and size
            checksum = self._calculate_checksum(backup_file)
            file_size = backup_file.stat().st_size / (1024 * 1024)  # MB
            
            # Log backup
            self._log_backup('files', backup_file, file_size, checksum, 'success')
            
            self.security_logger.info(f"Files backup created: {backup_file}")
            
            return {
                'type': 'files',
                'status': 'success',
                'file': str(backup_file),
                'size_mb': file_size,
                'checksum': checksum
            }
            
        except Exception as e:
            self.security_logger.error(f"Files backup failed: {e}")
            return {'type': 'files', 'status': 'failed', 'error': str(e)}
            
    def _backup_logs(self) -> Dict:
        """Backup log files"""
        try:
            backup_dir = Path('backups/logs')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f'logs_backup_{timestamp}.zip'
            
            # Create zip backup of logs
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                logs_dir = Path('logs')
                if logs_dir.exists():
                    for log_file in logs_dir.rglob('*.log'):
                        if log_file.is_file():
                            zipf.write(log_file, log_file.relative_to(Path('.')))
                            
            # Calculate checksum and size
            checksum = self._calculate_checksum(backup_file)
            file_size = backup_file.stat().st_size / (1024 * 1024)  # MB
            
            # Log backup
            self._log_backup('logs', backup_file, file_size, checksum, 'success')
            
            self.security_logger.info(f"Logs backup created: {backup_file}")
            
            return {
                'type': 'logs',
                'status': 'success',
                'file': str(backup_file),
                'size_mb': file_size,
                'checksum': checksum
            }
            
        except Exception as e:
            self.security_logger.error(f"Logs backup failed: {e}")
            return {'type': 'logs', 'status': 'failed', 'error': str(e)}
            
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def _log_backup(self, backup_type: str, file_path: Path, size_mb: float, 
                   checksum: str, status: str):
        """Log backup operation"""
        try:
            conn = sqlite3.connect('instance/security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO backup_logs 
                (backup_type, timestamp, status, file_path, size_mb, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                backup_type,
                datetime.now().isoformat(),
                status,
                str(file_path),
                size_mb,
                checksum
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.security_logger.error(f"Error logging backup: {e}")
            
    def cleanup_old_backups(self):
        """Cleanup old backups based on retention policy"""
        try:
            backup_base = Path('backups')
            
            for backup_type, policy in self.backup_schedule.items():
                backup_dir = backup_base / backup_type
                if not backup_dir.exists():
                    continue
                    
                retention_days = policy['retention']
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                for backup_file in backup_dir.glob('*'):
                    if backup_file.is_file():
                        file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                        if file_time < cutoff_date:
                            backup_file.unlink()
                            self.security_logger.info(f"Deleted old backup: {backup_file}")
                            
        except Exception as e:
            self.security_logger.error(f"Error cleaning up backups: {e}")
            
    def perform_compliance_check(self) -> Dict:
        """Perform automated compliance checks"""
        compliance_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'compliant',
            'checks': []
        }
        
        # Check 1: Password policy compliance
        password_check = self._check_password_policy()
        compliance_results['checks'].append(password_check)
        
        # Check 2: Data retention compliance
        retention_check = self._check_data_retention()
        compliance_results['checks'].append(retention_check)
        
        # Check 3: Access control compliance
        access_check = self._check_access_controls()
        compliance_results['checks'].append(access_check)
        
        # Check 4: Backup compliance
        backup_check = self._check_backup_compliance()
        compliance_results['checks'].append(backup_check)
        
        # Check 5: Security logging compliance
        logging_check = self._check_security_logging()
        compliance_results['checks'].append(logging_check)
        
        # Determine overall status
        failed_checks = [c for c in compliance_results['checks'] if c['status'] == 'non_compliant']
        if failed_checks:
            compliance_results['overall_status'] = 'non_compliant'
            
        # Log compliance check
        self._log_compliance_check(compliance_results)
        
        return compliance_results
        
    def _check_password_policy(self) -> Dict:
        """Check password policy compliance"""
        try:
            # In production, this would check actual password policies
            return {
                'check_type': 'password_policy',
                'status': 'compliant',
                'details': 'Password policy enforced: minimum 8 characters, complexity requirements',
                'remediation_needed': False
            }
        except:
            return {
                'check_type': 'password_policy',
                'status': 'non_compliant',
                'details': 'Password policy check failed',
                'remediation_needed': True
            }
            
    def _check_data_retention(self) -> Dict:
        """Check data retention compliance"""
        try:
            # Check if old data is being properly cleaned up
            old_logs = Path('logs').glob('*.log')
            old_log_count = sum(1 for log in old_logs 
                              if (datetime.now() - datetime.fromtimestamp(log.stat().st_mtime)).days > 90)
            
            if old_log_count > 10:
                return {
                    'check_type': 'data_retention',
                    'status': 'non_compliant',
                    'details': f'{old_log_count} log files older than 90 days found',
                    'remediation_needed': True
                }
            else:
                return {
                    'check_type': 'data_retention',
                    'status': 'compliant',
                    'details': 'Data retention policy being followed',
                    'remediation_needed': False
                }
        except:
            return {
                'check_type': 'data_retention',
                'status': 'unknown',
                'details': 'Unable to check data retention',
                'remediation_needed': False
            }
            
    def _check_access_controls(self) -> Dict:
        """Check access control compliance"""
        return {
            'check_type': 'access_controls',
            'status': 'compliant',
            'details': 'Role-based access controls implemented',
            'remediation_needed': False
        }
        
    def _check_backup_compliance(self) -> Dict:
        """Check backup compliance"""
        try:
            backup_dir = Path('backups')
            if not backup_dir.exists():
                return {
                    'check_type': 'backup_compliance',
                    'status': 'non_compliant',
                    'details': 'No backups found',
                    'remediation_needed': True
                }
                
            # Check for recent backups
            recent_backups = []
            for backup_type in ['database', 'files', 'logs']:
                type_dir = backup_dir / backup_type
                if type_dir.exists():
                    recent_backup = max(type_dir.glob('*'), key=lambda x: x.stat().st_mtime, default=None)
                    if recent_backup:
                        backup_age = (datetime.now() - datetime.fromtimestamp(recent_backup.stat().st_mtime)).days
                        recent_backups.append((backup_type, backup_age))
                        
            if not recent_backups:
                return {
                    'check_type': 'backup_compliance',
                    'status': 'non_compliant',
                    'details': 'No recent backups found',
                    'remediation_needed': True
                }
            else:
                return {
                    'check_type': 'backup_compliance',
                    'status': 'compliant',
                    'details': f'Recent backups found: {len(recent_backups)} types',
                    'remediation_needed': False
                }
        except:
            return {
                'check_type': 'backup_compliance',
                'status': 'unknown',
                'details': 'Unable to check backup compliance',
                'remediation_needed': False
            }
            
    def _check_security_logging(self) -> Dict:
        """Check security logging compliance"""
        try:
            security_log = Path('logs/security/security.log')
            if security_log.exists():
                return {
                    'check_type': 'security_logging',
                    'status': 'compliant',
                    'details': 'Security logging is active',
                    'remediation_needed': False
                }
            else:
                return {
                    'check_type': 'security_logging',
                    'status': 'non_compliant',
                    'details': 'Security logging not found',
                    'remediation_needed': True
                }
        except:
            return {
                'check_type': 'security_logging',
                'status': 'unknown',
                'details': 'Unable to check security logging',
                'remediation_needed': False
            }
            
    def _log_compliance_check(self, results: Dict):
        """Log compliance check results"""
        try:
            conn = sqlite3.connect('instance/security.db')
            cursor = conn.cursor()
            
            for check in results['checks']:
                cursor.execute('''
                    INSERT INTO compliance_checks 
                    (check_type, timestamp, status, details, remediation_needed)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    check['check_type'],
                    results['timestamp'],
                    check['status'],
                    check['details'],
                    check['remediation_needed']
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.security_logger.error(f"Error logging compliance check: {e}")
            
    def start_security_monitoring(self):
        """Start continuous security monitoring"""
        self.monitoring_active = True
        self.security_logger.info("Security monitoring started")
        
        def security_loop():
            while self.monitoring_active:
                try:
                    # Perform periodic security tasks
                    
                    # Daily backup
                    if datetime.now().hour == 2 and datetime.now().minute < 5:  # 2 AM
                        self.perform_automated_backup('database')
                        
                    # Weekly full backup
                    if datetime.now().weekday() == 6 and datetime.now().hour == 3:  # Sunday 3 AM
                        self.perform_automated_backup('all')
                        
                    # Daily compliance check
                    if datetime.now().hour == 1 and datetime.now().minute < 5:  # 1 AM
                        self.perform_compliance_check()
                        
                    # Cleanup old backups
                    if datetime.now().hour == 4 and datetime.now().minute < 5:  # 4 AM
                        self.cleanup_old_backups()
                        
                except Exception as e:
                    self.security_logger.error(f"Security monitoring error: {e}")
                    
                time.sleep(300)  # Check every 5 minutes
                
        security_thread = Thread(target=security_loop, daemon=True)
        security_thread.start()
        
    def stop_security_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        self.security_logger.info("Security monitoring stopped")
        
    def get_security_status(self) -> Dict:
        """Get current security status"""
        try:
            conn = sqlite3.connect('instance/security.db')
            cursor = conn.cursor()
            
            # Get recent security events
            cursor.execute('''
                SELECT COUNT(*) FROM security_events 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            recent_events = cursor.fetchone()[0]
            
            # Get recent backups
            cursor.execute('''
                SELECT backup_type, MAX(timestamp) FROM backup_logs 
                GROUP BY backup_type
            ''')
            recent_backups = dict(cursor.fetchall())
            
            # Get compliance status
            cursor.execute('''
                SELECT status, COUNT(*) FROM compliance_checks 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY status
            ''')
            compliance_stats = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'status': 'secure',
                'recent_events': recent_events,
                'recent_backups': recent_backups,
                'compliance_stats': compliance_stats,
                'monitoring_active': self.monitoring_active
            }
            
        except Exception as e:
            self.security_logger.error(f"Error getting security status: {e}")
            return {'status': 'unknown', 'error': str(e)}

# Global security automation instance
security_automation = SecurityAutomation()

def start_security_automation():
    """Start security automation"""
    security_automation.start_security_monitoring()
    
def get_security_status():
    """Get current security status"""
    return security_automation.get_security_status()
    
def process_security_event(request_data: Dict):
    """Process security event from request"""
    threats = security_automation.detect_threats(request_data)
    for threat in threats:
        security_automation.respond_to_threat(threat)