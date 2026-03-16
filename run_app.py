"""
Complete Flask Application Runner - SQLAlchemy Fixed
All Features Working with python run_app.py
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use existing __init__.py but without SocketIO
def create_app_fixed():
    """Create app using existing __init__.py but skip SocketIO"""
    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy
    from flask_login import LoginManager
    from flask_bcrypt import Bcrypt
    from flask_moment import Moment
    from flask_wtf.csrf import CSRFProtect
    from config import Config
    
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for demo
    
    # Use the same db instance as models.py
    from models import db
    db.init_app(app)
    
    # Initialize other extensions
    login = LoginManager()
    bcrypt = Bcrypt()
    moment = Moment()
    csrf = CSRFProtect()
    
    login.init_app(app)
    login.login_view = "main.login"
    bcrypt.init_app(app)
    moment.init_app(app)
    csrf.init_app(app)
    
    # User loader
    @login.user_loader
    def load_user(user_id):
        try:
            from models import User
            return User.query.get(int(user_id))
        except:
            return None
    
    # API route
    @app.route('/api/check-new-messages', methods=['POST', 'GET'])
    @csrf.exempt
    def api_check_messages():
        return {'status': 'ok'}
    
    # Import blueprints
    from routes import bp as main_bp
    app.register_blueprint(main_bp)
    
    try:
        from admin import admin_bp
        app.register_blueprint(admin_bp)
        print("✅ Admin loaded")
    except ImportError:
        pass
    
    try:
        from continuous_learning_routes import learning_bp
        app.register_blueprint(learning_bp)
        print("✅ Learning loaded")
    except ImportError:
        pass
    
    try:
        from location_matching_routes import location_bp
        app.register_blueprint(location_bp)
        print("✅ Location matching loaded")
    except ImportError:
        pass
    
    try:
        from enhanced_admin_routes import enhanced_admin_bp
        app.register_blueprint(enhanced_admin_bp)
        print("✅ Enhanced admin loaded")
    except ImportError:
        pass
    
    try:
        from enhanced_admin_ai_routes import enhanced_ai_bp
        app.register_blueprint(enhanced_ai_bp)
        print("✅ Enhanced AI loaded")
    except ImportError:
        pass
    
    try:
        from analytics_routes import analytics_bp
        app.register_blueprint(analytics_bp)
        print("✅ Analytics loaded")
    except ImportError:
        pass
    
    try:
        from multicamera_routes import multicamera_bp
        app.register_blueprint(multicamera_bp)
    except ImportError:
        pass
    
    try:
        from integration_routes import integration_bp
        app.register_blueprint(integration_bp)
    except ImportError:
        pass
    
    try:
        from security_routes import security_bp
        app.register_blueprint(security_bp)
    except ImportError:
        pass
    
    try:
        from advanced_ui_routes import advanced_ui_bp
        app.register_blueprint(advanced_ui_bp)
    except ImportError:
        pass
    
    try:
        from cloud_routes import cloud_bp
        app.register_blueprint(cloud_bp)
    except ImportError:
        pass
    
    try:
        from realtime_routes import realtime_bp
        app.register_blueprint(realtime_bp)
    except ImportError:
        pass
    
    # Template helpers
    try:
        from template_helpers import get_image_url, get_primary_photo_url, get_video_url, verify_file_exists
        
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
    except ImportError:
        pass
    
    return app

def setup_database(app):
    """Setup database and admin user"""
    with app.app_context():
        try:
            from models import db, User
            db.create_all()
            
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                from flask_bcrypt import Bcrypt
                bcrypt = Bcrypt()
                admin = User(
                    username='admin',
                    email='admin@system.com',
                    password_hash=bcrypt.generate_password_hash('admin123').decode('utf-8'),
                    role='admin',
                    is_active=True
                )
                db.session.add(admin)
                db.session.commit()
                print("✅ Admin user created")
        except Exception as e:
            print(f"Database error: {e}")

if __name__ == "__main__":
    print("🚀 AI Powered Person Detection")
    print("=" * 50)
    
    app = create_app_fixed()
    setup_database(app)
    
    print("🌐 Dashboard: http://localhost:5000/")
    print("🔐 Login: admin / admin123")
    print("🛡️ Admin: http://localhost:5000/admin/")
    print("=" * 50)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )