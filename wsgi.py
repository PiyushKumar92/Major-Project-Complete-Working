import os
import sys

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from __init__ import create_app, db

app = create_app()

# Start automatic location matching service
try:
    from auto_location_service import start_auto_service
    start_auto_service()
    print("✅ Advanced Location Matching Service started")
except ImportError:
    print("⚠️ Auto service not available")
except Exception as e:
    print(f"⚠️ Auto service startup failed: {str(e)}")
    import traceback
    traceback.print_exc()

# Start continuous learning system
try:
    from continuous_learning_system import continuous_learning_system
    continuous_learning_system._init_learning_db()
    print("✅ Continuous Learning System initialized")
except ImportError:
    print("⚠️ Learning system not available")
except Exception as e:
    print(f"⚠️ Learning system startup failed: {str(e)}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)