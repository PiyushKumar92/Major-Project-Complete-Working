"""
Auto-verify existing detections based on confidence score
"""
from models import PersonDetection
from __init__ import db, create_app

def auto_verify_detections():
    """Auto-verify detections with confidence >= 60%"""
    app = create_app()
    
    with app.app_context():
        # Get all unverified detections
        detections = PersonDetection.query.filter_by(verified=False).all()
        print(f"Found {len(detections)} unverified detections")
        
        verified_count = 0
        for detection in detections:
            # Auto-verify if confidence >= 60%
            if detection.confidence_score >= 0.60:
                detection.verified = True
                detection.notes = 'Auto-verified by AI based on confidence score'
                verified_count += 1
                print(f"[OK] Detection {detection.id}: {detection.confidence_score*100:.1f}% - Auto-verified")
            else:
                print(f"[SKIP] Detection {detection.id}: {detection.confidence_score*100:.1f}% - Needs manual review")
        
        db.session.commit()
        print(f"\n[SUCCESS] Auto-verified {verified_count} out of {len(detections)} detections")

if __name__ == '__main__':
    auto_verify_detections()
