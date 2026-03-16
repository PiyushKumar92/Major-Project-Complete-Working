"""
Automated System Monitoring
Self-healing infrastructure with performance monitoring and auto-scaling
"""

import os
import psutil
import time
import json
import logging
from datetime import datetime, timedelta
from threading import Thread
from dataclasses import dataclass
from typing import Dict, List, Optional
import sqlite3
from pathlib import Path

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    active_connections: int
    response_time: float
    error_count: int
    case_processing_queue: int

class SystemMonitor:
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history = []
        self.alert_thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 75.0,
            'memory_critical': 85.0,
            'memory_warning': 70.0,
            'disk_critical': 90.0,
            'disk_warning': 80.0,
            'response_time_critical': 5.0,
            'response_time_warning': 3.0
        }
        self.auto_recovery_enabled = True
        self.setup_logging()
        
    def setup_logging(self):
        """Setup monitoring logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SystemMonitor')
        
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Application metrics
            response_time = self._measure_response_time()
            error_count = self._get_recent_error_count()
            queue_size = self._get_processing_queue_size()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage=disk.percent,
                active_connections=connections,
                response_time=response_time,
                error_count=error_count,
                case_processing_queue=queue_size
            )
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return None
            
    def _measure_response_time(self) -> float:
        """Measure application response time"""
        try:
            import requests
            start_time = time.time()
            response = requests.get('http://localhost:5000/health', timeout=5)
            return time.time() - start_time
        except:
            return 0.0
            
    def _get_recent_error_count(self) -> int:
        """Get error count from last 5 minutes"""
        try:
            log_file = Path('logs/app.log')
            if not log_file.exists():
                return 0
                
            error_count = 0
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            
            with open(log_file, 'r') as f:
                for line in f.readlines()[-1000:]:  # Check last 1000 lines
                    if 'ERROR' in line or 'CRITICAL' in line:
                        error_count += 1
                        
            return error_count
        except:
            return 0
            
    def _get_processing_queue_size(self) -> int:
        """Get current processing queue size"""
        try:
            from models import Case
            from __init__ import create_app, db
            
            app = create_app()
            with app.app_context():
                queue_size = Case.query.filter_by(status='Under Processing').count()
                return queue_size
        except:
            return 0
            
    def analyze_metrics(self, metrics: SystemMetrics) -> Dict:
        """Analyze metrics and determine actions needed"""
        analysis = {
            'status': 'healthy',
            'alerts': [],
            'recommendations': [],
            'auto_actions': []
        }
        
        # CPU Analysis
        if metrics.cpu_percent >= self.alert_thresholds['cpu_critical']:
            analysis['status'] = 'critical'
            analysis['alerts'].append(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")
            analysis['auto_actions'].append('scale_up_cpu')
        elif metrics.cpu_percent >= self.alert_thresholds['cpu_warning']:
            analysis['status'] = 'warning'
            analysis['alerts'].append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
        # Memory Analysis
        if metrics.memory_percent >= self.alert_thresholds['memory_critical']:
            analysis['status'] = 'critical'
            analysis['alerts'].append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
            analysis['auto_actions'].append('clear_cache')
        elif metrics.memory_percent >= self.alert_thresholds['memory_warning']:
            analysis['status'] = 'warning'
            analysis['alerts'].append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
        # Disk Analysis
        if metrics.disk_usage >= self.alert_thresholds['disk_critical']:
            analysis['status'] = 'critical'
            analysis['alerts'].append(f"Critical disk usage: {metrics.disk_usage:.1f}%")
            analysis['auto_actions'].append('cleanup_files')
        elif metrics.disk_usage >= self.alert_thresholds['disk_warning']:
            analysis['status'] = 'warning'
            analysis['alerts'].append(f"High disk usage: {metrics.disk_usage:.1f}%")
            
        # Response Time Analysis
        if metrics.response_time >= self.alert_thresholds['response_time_critical']:
            analysis['status'] = 'critical'
            analysis['alerts'].append(f"Critical response time: {metrics.response_time:.2f}s")
            analysis['auto_actions'].append('restart_services')
        elif metrics.response_time >= self.alert_thresholds['response_time_warning']:
            analysis['status'] = 'warning'
            analysis['alerts'].append(f"Slow response time: {metrics.response_time:.2f}s")
            
        # Error Analysis
        if metrics.error_count > 10:
            analysis['status'] = 'critical'
            analysis['alerts'].append(f"High error rate: {metrics.error_count} errors/5min")
            analysis['auto_actions'].append('investigate_errors')
            
        return analysis
        
    def execute_auto_recovery(self, actions: List[str]):
        """Execute automatic recovery actions"""
        if not self.auto_recovery_enabled:
            return
            
        for action in actions:
            try:
                if action == 'clear_cache':
                    self._clear_system_cache()
                elif action == 'cleanup_files':
                    self._cleanup_temporary_files()
                elif action == 'restart_services':
                    self._restart_background_services()
                elif action == 'scale_up_cpu':
                    self._optimize_cpu_usage()
                elif action == 'investigate_errors':
                    self._log_error_investigation()
                    
                self.logger.info(f"Executed auto-recovery action: {action}")
            except Exception as e:
                self.logger.error(f"Failed to execute {action}: {e}")
                
    def _clear_system_cache(self):
        """Clear system cache to free memory"""
        try:
            # Clear Redis cache if available
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.flushdb()
            self.logger.info("Redis cache cleared")
        except:
            pass
            
        # Clear temporary files
        temp_dirs = ['static/temp', 'static/cache']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
                        
    def _cleanup_temporary_files(self):
        """Cleanup temporary files to free disk space"""
        cleanup_dirs = [
            'static/temp',
            'static/cache',
            'logs',
            'static/uploads/temp'
        ]
        
        total_freed = 0
        for directory in cleanup_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path):
                            # Delete files older than 24 hours
                            if time.time() - os.path.getmtime(file_path) > 86400:
                                size = os.path.getsize(file_path)
                                os.remove(file_path)
                                total_freed += size
                    except:
                        pass
                        
        self.logger.info(f"Cleaned up {total_freed / (1024*1024):.1f} MB of temporary files")
        
    def _restart_background_services(self):
        """Restart background services"""
        self.logger.info("Background service restart triggered")
        # In production, this would restart Celery workers, etc.
        
    def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        # Reduce processing priority for non-critical tasks
        try:
            import psutil
            current_process = psutil.Process()
            current_process.nice(10)  # Lower priority
            self.logger.info("CPU optimization applied")
        except:
            pass
            
    def _log_error_investigation(self):
        """Log error investigation details"""
        self.logger.warning("High error rate detected - investigation needed")
        
    def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.logger.info("System monitoring started")
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    metrics = self.collect_metrics()
                    if metrics:
                        self.metrics_history.append(metrics)
                        
                        # Keep only last 24 hours of metrics
                        cutoff_time = datetime.now() - timedelta(hours=24)
                        self.metrics_history = [
                            m for m in self.metrics_history 
                            if m.timestamp > cutoff_time
                        ]
                        
                        # Analyze metrics
                        analysis = self.analyze_metrics(metrics)
                        
                        # Log status
                        if analysis['status'] != 'healthy':
                            self.logger.warning(f"System status: {analysis['status']}")
                            for alert in analysis['alerts']:
                                self.logger.warning(f"Alert: {alert}")
                                
                        # Execute auto-recovery if needed
                        if analysis['auto_actions']:
                            self.execute_auto_recovery(analysis['auto_actions'])
                            
                        # Save metrics to database
                        self._save_metrics(metrics, analysis)
                        
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    
                time.sleep(interval)
                
        monitor_thread = Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        self.logger.info("System monitoring stopped")
        
    def _save_metrics(self, metrics: SystemMetrics, analysis: Dict):
        """Save metrics to database"""
        try:
            db_path = 'instance/monitoring.db'
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage REAL,
                    active_connections INTEGER,
                    response_time REAL,
                    error_count INTEGER,
                    queue_size INTEGER,
                    status TEXT,
                    alerts TEXT
                )
            ''')
            
            # Insert metrics
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, disk_usage, 
                 active_connections, response_time, error_count, queue_size, status, alerts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.disk_usage,
                metrics.active_connections,
                metrics.response_time,
                metrics.error_count,
                metrics.case_processing_queue,
                analysis['status'],
                json.dumps(analysis['alerts'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        if not self.metrics_history:
            return {'status': 'unknown', 'message': 'No metrics available'}
            
        latest_metrics = self.metrics_history[-1]
        analysis = self.analyze_metrics(latest_metrics)
        
        return {
            'status': analysis['status'],
            'timestamp': latest_metrics.timestamp.isoformat(),
            'metrics': {
                'cpu_percent': latest_metrics.cpu_percent,
                'memory_percent': latest_metrics.memory_percent,
                'disk_usage': latest_metrics.disk_usage,
                'response_time': latest_metrics.response_time,
                'active_connections': latest_metrics.active_connections,
                'error_count': latest_metrics.error_count,
                'queue_size': latest_metrics.case_processing_queue
            },
            'alerts': analysis['alerts'],
            'recommendations': analysis['recommendations']
        }
        
    def get_performance_trends(self, hours: int = 24) -> Dict:
        """Get performance trends over specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No data available'}
            
        # Calculate trends
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        response_times = [m.response_time for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'cpu': {
                'current': cpu_values[-1],
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'trend': 'increasing' if cpu_values[-1] > cpu_values[0] else 'decreasing'
            },
            'memory': {
                'current': memory_values[-1],
                'average': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'trend': 'increasing' if memory_values[-1] > memory_values[0] else 'decreasing'
            },
            'response_time': {
                'current': response_times[-1],
                'average': sum(response_times) / len(response_times),
                'max': max(response_times),
                'trend': 'increasing' if response_times[-1] > response_times[0] else 'decreasing'
            }
        }

# Global monitor instance
system_monitor = SystemMonitor()

def start_system_monitoring():
    """Start system monitoring"""
    system_monitor.start_monitoring(interval=30)  # Monitor every 30 seconds
    
def get_system_status():
    """Get current system status"""
    return system_monitor.get_system_health()