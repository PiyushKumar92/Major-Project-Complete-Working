"""
Flask application runner without Redis/Celery for development
Use this if Redis setup is problematic
"""
from app import create_app
import os

def main():
    # Create Flask app
    app = create_app()
    
    # Set development configuration
    app.config['DEBUG'] = True
    app.config['TESTING'] = False
    
    # Disable Celery for development
    app.config['CELERY_BROKER_URL'] = None
    app.config['CELERY_RESULT_BACKEND'] = None
    
    print("🚀 Starting Flask app without Redis/Celery...")
    print("📍 Auto location matching will be disabled")
    print("🌐 App will run on http://127.0.0.1:5000")
    
    # Run the app
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        use_reloader=True
    )

if __name__ == '__main__':
    main()