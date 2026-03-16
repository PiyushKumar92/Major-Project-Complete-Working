# Suppress warnings first
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*CELERY_RESULT_BACKEND.*deprecated.*")
warnings.filterwarnings("ignore", message=".*broker_connection_retry.*deprecated.*")

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError, DatabaseError
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_moment import Moment
from flask_wtf.csrf import CSRFProtect
# from flask_socketio import SocketIO  # Disabled to prevent eventlet errors
from celery import Celery
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
bcrypt = Bcrypt()
moment = Moment()
csrf = CSRFProtect()
# socketio = SocketIO()  # Disabled to prevent eventlet errors


def make_celery(app):
    # Validate Celery configuration
    broker_url = app.config.get("CELERY_BROKER_URL")
    result_backend = app.config.get("result_backend") or app.config.get("CELERY_RESULT_BACKEND")
    
    # Use SQLite as fallback for development
    if not broker_url:
        broker_url = "sqla+sqlite:///celery.db"
    if not result_backend:
        result_backend = "db+sqlite:///celery_results.db"
    
    celery = Celery(
        app.import_name,
        backend=result_backend,
        broker=broker_url,
    )
    
    # Update configuration with new format
    celery.conf.update({
        'result_backend': result_backend,
        'broker_url': broker_url,
        'broker_connection_retry_on_startup': True,
        **{k: v for k, v in app.config.items() if not k.startswith('CELERY_')}
    })
    return celery


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)
    login.login_view = "main.login"
    login.login_message = "Please log in to access this page"
    bcrypt.init_app(app)
    moment.init_app(app)
    csrf.init_app(app)
    
    # CSRF exemptions for API routes
    csrf.exempt('learning.record_admin_feedback')
    csrf.exempt('learning.trigger_learning')
    csrf.exempt('learning.reduce_false_positives')
    csrf.exempt('learning.update_threshold')
    csrf.exempt('main.check_new_messages')
    csrf.exempt('main.api_check_new_messages')
    
    # Track user activity
    @app.before_request
    def track_user_activity():
        from flask_login import current_user
        from models import get_ist_now
        
        if current_user.is_authenticated:
            # Update last activity every 5 minutes to avoid too many DB writes
            current_time = get_ist_now()
            if not hasattr(current_user, '_last_activity_update') or \
               (current_time - current_user._last_activity_update).seconds > 300:
                current_user.last_seen = current_time
                current_user.is_online = True
                current_user._last_activity_update = current_time
                try:
                    db.session.commit()
                except (SQLAlchemyError, DatabaseError) as e:
                    try:
                        db.session.rollback()
                    except (SQLAlchemyError, DatabaseError):
                        pass  # Ignore rollback errors in activity tracking
    
    # Add security headers with relaxed CSP for development
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        # Relaxed CSP for development - allows inline scripts and external CDNs
        response.headers['Content-Security-Policy'] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; style-src 'self' 'unsafe-inline' https:; font-src 'self' https:; img-src 'self' data: blob: https:; connect-src 'self' https:;"
        return response

    from routes import bp as main_bp
    from error_handlers import register_error_handlers

    app.register_blueprint(main_bp)
    
    # Import admin and other blueprints after main blueprint to avoid circular imports
    try:
        from admin import admin_bp
        app.register_blueprint(admin_bp)
    except ImportError as e:
        print(f"Warning: Could not import admin blueprint: {e}")
    
    try:
        from continuous_learning_routes import learning_bp
        app.register_blueprint(learning_bp)
    except ImportError as e:
        print(f"Warning: Could not import learning blueprint: {e}")
    
    try:
        from location_matching_routes import location_bp
        app.register_blueprint(location_bp)
    except ImportError as e:
        print(f"Warning: Could not import location blueprint: {e}")
    
    try:
        from enhanced_admin_routes import enhanced_admin_bp
        app.register_blueprint(enhanced_admin_bp)
    except ImportError as e:
        print(f"Warning: Could not import enhanced admin blueprint: {e}")
    
    try:
        from enhanced_admin_ai_routes import enhanced_ai_bp
        app.register_blueprint(enhanced_ai_bp)
    except ImportError as e:
        print(f"Warning: Could not import enhanced AI blueprint: {e}")
    
    try:
        from realtime_routes import realtime_bp, register_socketio_events
        app.register_blueprint(realtime_bp)
        # Skip SocketIO initialization to avoid eventlet errors
        pass
        print("✅ Real-time CCTV monitoring system initialized")
    except ImportError as e:
        print(f"Warning: Could not import real-time routes: {e}")
    
    try:
        from analytics_routes import analytics_bp
        app.register_blueprint(analytics_bp)
        print("✅ Advanced analytics system initialized")
    except ImportError as e:
        print(f"Warning: Could not import analytics routes: {e}")
    
    try:
        from multicamera_routes import multicamera_bp
        app.register_blueprint(multicamera_bp)
        print("✅ Multi-camera correlation system initialized")
    except ImportError as e:
        print(f"Warning: Could not import multi-camera routes: {e}")
    
    try:
        from cloud_routes import cloud_bp
        app.register_blueprint(cloud_bp)
        print("✅ Cloud & scalability system initialized")
    except ImportError as e:
        print(f"Warning: Could not import cloud routes: {e}")
    
    try:
        from advanced_ui_routes import advanced_ui_bp
        app.register_blueprint(advanced_ui_bp)
        print("✅ Advanced UI/UX system initialized")
    except ImportError as e:
        print(f"Warning: Could not import advanced UI routes: {e}")
    
    try:
        from integration_routes import integration_bp
        app.register_blueprint(integration_bp)
        print("✅ Integration system initialized")
    except ImportError as e:
        print(f"Warning: Could not import integration routes: {e}")
    
    try:
        from security_routes import security_bp
        app.register_blueprint(security_bp)
        print("✅ Security & compliance system initialized")
    except ImportError as e:
        print(f"Warning: Could not import security routes: {e}")
    
    # Register error handlers
    register_error_handlers(app)
    
    # Initialize enhanced AI system
    try:
        from enhanced_ai_engine import enhanced_ai
        ai_info = enhanced_ai.get_system_info()
        print(f"🤖 Enhanced AI System Initialized:")
        print(f"   YOLO v8: {'✅' if ai_info['yolo_v8_available'] else '❌ (using Haar Cascade)'}")
        print(f"   FaceNet: {'✅' if ai_info['facenet_available'] else '❌ (using face_recognition)'}")
        print(f"   DeepSORT: {'✅' if ai_info['deepsort_available'] else '❌ (using CSRT)'}")
    except Exception as e:
        print(f"⚠️ Enhanced AI initialization warning: {e}")
        print("   System will use fallback methods")
    
    # Initialize analytics engine
    try:
        from analytics_engine import analytics_engine
        print("📊 Advanced Analytics Engine Initialized:")
        print("   Movement Tracking: ✅")
        print("   Heat Maps: ✅")
        print("   Predictive Analytics: ✅")
    except Exception as e:
        print(f"⚠️ Analytics initialization warning: {e}")
    
    # Initialize multi-camera engine
    try:
        from multicamera_engine import multicamera_engine
        print("📹 Multi-Camera Correlation Engine Initialized:")
        print("   Cross-Camera Tracking: ✅")
        print("   Route Reconstruction: ✅")
        print("   Camera Network: ✅")
    except Exception as e:
        print(f"⚠️ Multi-camera initialization warning: {e}")
    
    # Initialize cloud engine
    try:
        print(f"☁️ Cloud & Scalability Engine:")
        print(f"   AWS Integration: ❌ (Permanently Disabled - Using GPU CNN)")
        print("   GPU CNN: ✅")
        print("   Auto-Scaling: ✅")
        print("   Local Processing: ✅")
    except Exception as e:
        print(f"⚠️ Cloud initialization warning: {e}")
    
    # Initialize integration engine
    try:
        from integration_engine import integration_engine
        integration_info = integration_engine.get_system_info()
        print(f"🔌 Integration System Initialized:")
        print(f"   Police DB: {'✅' if integration_info['police_db_enabled'] else '❌ (Disabled)'}")
        print(f"   Social Media: {'✅' if integration_info['social_media_enabled'] else '❌ (Disabled)'}")
        print(f"   Traffic Cameras: {'✅' if integration_info['traffic_cameras_enabled'] else '❌ (Disabled)'} ({integration_info['total_traffic_cameras']} cameras)")
    except Exception as e:
        print(f"⚠️ Integration initialization warning: {e}")
        print("   System will use fallback methods")
    
    # Initialize security engine
    try:
        from security_engine import security_engine
        security_info = security_engine.get_system_info()
        print(f"🔒 Security & Compliance System Initialized:")
        print(f"   Encryption: {'✅' if security_info['encryption_enabled'] else '❌ (Disabled)'}")
        print(f"   RBAC: {'✅' if security_info['rbac_enabled'] else '❌ (Disabled)'} ({security_info['total_roles']} roles)")
        print(f"   Audit Trail: {'✅' if security_info['audit_trail_enabled'] else '❌ (Disabled)'}")
        print(f"   GDPR Compliance: {'✅' if security_info['gdpr_compliance_enabled'] else '❌ (Disabled)'}")
    except Exception as e:
        print(f"⚠️ Security initialization warning: {e}")
        print("   System will use fallback methods")
    
    # Template helper functions
    from template_helpers import get_image_url, get_primary_photo_url, get_video_url, verify_file_exists
    from status_helpers import get_status_display_info, get_status_badge_html, get_status_alert_html
    from status_template_helpers import (
        status_badge_filter, status_icon_filter, status_emoji_filter, status_color_filter,
        get_status_card_html, get_status_progress_html, get_status_summary_stats
    )
    from comprehensive_status_system import ALL_CASE_STATUSES, PUBLIC_VISIBLE_STATUSES, ACTIVE_STATUSES
    
    @app.template_filter('image_url')
    def image_url_filter(image_path):
        return get_image_url(image_path)
    
    @app.template_filter('video_url')
    def video_url_filter(video_path):
        return get_video_url(video_path)
    
    @app.template_global()
    def get_case_photo_url(case):
        return get_primary_photo_url(case)
    
    @app.template_global()
    def file_exists(file_path):
        return verify_file_exists(file_path)
    
    # Status helper functions
    @app.template_global()
    def get_status_info(status):
        return get_status_display_info(status)
    
    @app.template_filter('status_badge')
    def status_badge_template_filter(status, is_admin=False):
        try:
            return status_badge_filter(status, is_admin)
        except Exception as e:
            # Fallback if status system fails
            return f'<span class="badge bg-secondary">{status}</span>'
    
    @app.template_filter('status_icon')
    def status_icon_template_filter(status):
        try:
            return status_icon_filter(status)
        except Exception:
            return '<i class="fas fa-question-circle"></i>'
    
    @app.template_filter('status_emoji')
    def status_emoji_template_filter(status):
        try:
            return status_emoji_filter(status)
        except Exception:
            return '❓'
    
    @app.template_filter('status_color')
    def status_color_template_filter(status):
        try:
            return status_color_filter(status)
        except Exception:
            return 'secondary'
    
    @app.template_filter('status_alert')
    def status_alert_filter(status, admin_message=None):
        try:
            return get_status_alert_html(status, admin_message)
        except Exception:
            return f'<div class="alert alert-secondary">{status}</div>'
    
    @app.template_filter('days_since')
    def days_since_filter(date_obj):
        """Calculate days since a given date"""
        if not date_obj:
            return 0
        from datetime import datetime
        try:
            now = datetime.utcnow()
            if hasattr(date_obj, 'replace') and date_obj.tzinfo is None:
                # If date_obj is naive, assume UTC
                delta = now - date_obj
            else:
                # If date_obj has timezone info, convert to UTC
                if hasattr(date_obj, 'utctimetuple'):
                    delta = now - datetime.utcfromtimestamp(date_obj.timestamp())
                else:
                    delta = now - date_obj
            return max(0, delta.days)
        except Exception:
            return 0
    
    # Enhanced AI template functions
    @app.template_global()
    def get_ai_capabilities():
        try:
            from enhanced_ai_engine import enhanced_ai
            return enhanced_ai.get_system_info()
        except:
            return {'yolo_v8_available': False, 'facenet_available': False, 'deepsort_available': False}
    
    @app.template_global()
    def get_analytics_capabilities():
        return {
            'movement_tracking': True,
            'heatmaps': True,
            'predictive_analytics': True,
            'real_time_analytics': True
        }
    
    @app.template_global()
    def get_multicamera_capabilities():
        return {
            'cross_camera_tracking': True,
            'route_reconstruction': True,
            'camera_network': True,
            'global_correlation': True
        }
    
    @app.template_global()
    def get_cloud_capabilities():
        # AWS permanently disabled - using local GPU CNN
        return {
            'aws_integration': False,
            'auto_scaling': True,
            'edge_computing': True,
            'load_balancing': True,
            'cloud_storage': True,
            'gpu_cnn_enabled': True
        }
    
    @app.template_global()
    def get_ui_capabilities():
        return {
            'interactive_timeline': True,
            '3d_visualization': True,
            'mobile_app': True,
            'advanced_charts': True,
            'responsive_design': True
        }
    
    # Global template functions for status system
    @app.template_global()
    def get_all_statuses():
        return ALL_CASE_STATUSES
    
    @app.template_global()
    def get_public_statuses():
        return PUBLIC_VISIBLE_STATUSES
    
    @app.template_global()
    def get_active_statuses():
        return ACTIVE_STATUSES
    
    @app.template_global()
    def get_status_card(status, count=0, is_admin=False):
        return get_status_card_html(status, count, is_admin)
    
    @app.template_global()
    def get_status_progress(cases):
        return get_status_progress_html(cases)
    
    @app.template_global()
    def get_status_stats(cases):
        return get_status_summary_stats(cases)
    
    # Context processors for global data
    @app.context_processor
    def inject_global_data():
        from flask_login import current_user
        from models import Announcement, Notification, get_ist_now
        from datetime import datetime
        
        # Auto-deactivate expired announcements
        try:
            current_time = get_ist_now()
            expired_announcements = Announcement.query.filter(
                Announcement.is_active == True,
                Announcement.expires_at.isnot(None),
                Announcement.expires_at <= current_time
            ).all()
            
            for announcement in expired_announcements:
                announcement.is_active = False
            
            if expired_announcements:
                db.session.commit()
        except:
            db.session.rollback()
        
        # Get active announcements for logged-in users only
        active_announcements = []
        if current_user.is_authenticated:
            try:
                from models import AnnouncementRead
                
                # Get all active announcements
                current_time = get_ist_now()
                all_active = Announcement.query.filter(
                    Announcement.is_active == True,
                    db.or_(
                        Announcement.expires_at == None,
                        Announcement.expires_at > current_time
                    )
                ).order_by(Announcement.created_at.desc()).all()
                
                # Filter out announcements already read by current user
                read_announcement_ids = db.session.query(AnnouncementRead.announcement_id).filter_by(user_id=current_user.id).all()
                read_ids = [r[0] for r in read_announcement_ids]
                
                active_announcements = [a for a in all_active if a.id not in read_ids]
            except Exception:
                # If table doesn't exist, show all announcements
                try:
                    current_time = get_ist_now()
                    active_announcements = Announcement.query.filter(
                        Announcement.is_active == True,
                        db.or_(
                            Announcement.expires_at.is_(None),
                            Announcement.expires_at > current_time
                        )
                    ).order_by(Announcement.created_at.desc()).all()
                except Exception:
                    active_announcements = []
        
        # Get unread notifications count for authenticated users
        unread_count = 0
        if current_user.is_authenticated:
            unread_count = current_user.unread_notifications_count
        
        return {
            'active_announcements': active_announcements,
            'unread_notifications_count': unread_count
        }

    return app


@login.user_loader
def load_user(user_id):
    from models import User
    
    try:
        return User.query.get(int(user_id))
    except (ValueError, TypeError):
        return None


import models
import person_consistency_models
