#!/usr/bin/env python3
"""
Automatic CCTV Processing - Integrates with case registration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from __init__ import create_app, db
from models import Case, TargetImage, SearchVideo, Sighting
from cctv_search_engine import search_person_in_cctv_footage
import json

def process_approved_case(case_id):
    """
    Automatically process approved case for CCTV search
    """
    app = create_app()
    with app.app_context():
        try:
            case = Case.query.get(case_id)
            if not case or case.status != 'Approved':
                return {"error": "Case not found or not approved"}
            
            print(f"🔍 Starting CCTV processing for Case {case_id}: {case.person_name}")
            
            # Get reference photos
            reference_photos = []
            for img in case.target_images:
                photo_path = os.path.join('static', img.image_path.replace('/', os.sep))
                if os.path.exists(photo_path):
                    reference_photos.append(photo_path)
            
            if not reference_photos:
                return {"error": "No reference photos found"}
            
            # Get search videos (CCTV footage)
            cctv_videos = []
            for video in case.search_videos:
                video_path = os.path.join('static', video.video_path.replace('/', os.sep))
                if os.path.exists(video_path):
                    cctv_videos.append(video_path)
            
            if not cctv_videos:
                return {"error": "No CCTV footage provided"}
            
            print(f"📸 Reference photos: {len(reference_photos)}")
            print(f"📹 CCTV videos: {len(cctv_videos)}")
            
            # Start CCTV search
            search_results = search_person_in_cctv_footage(case_id, reference_photos, cctv_videos)
            
            # Save results to database
            total_sightings = 0
            for video_result in search_results.get('search_results', []):
                for match in video_result.get('matches_found', []):
                    # Create sighting record
                    sighting = Sighting(
                        case_id=case_id,
                        video_name=os.path.basename(video_result.get('cctv_video', '')),
                        timestamp=match['timestamp_formatted'],
                        confidence_score=match['confidence'],
                        frame_number=match['frame_number'],
                        detection_details=json.dumps(match)
                    )
                    db.session.add(sighting)
                    total_sightings += 1
            
            # Update case status
            if total_sightings > 0:
                case.status = 'Under Processing'
                case.admin_message = f"🎯 CCTV Analysis Complete!\n\n✅ Found {total_sightings} potential sightings\n📹 Processed {len(cctv_videos)} CCTV videos\n📸 Used {len(reference_photos)} reference photos\n\n🔍 Results are being reviewed. You will be notified of any confirmed matches."
            else:
                case.admin_message = f"🔍 CCTV Analysis Complete\n\n📹 Processed {len(cctv_videos)} CCTV videos\n📸 Used {len(reference_photos)} reference photos\n\n❌ No matches found in provided footage\n💡 Consider providing additional CCTV footage or higher quality reference photos"
            
            db.session.commit()
            
            print(f"✅ CCTV processing completed. Found {total_sightings} sightings.")
            
            return {
                "status": "success",
                "case_id": case_id,
                "sightings_found": total_sightings,
                "videos_processed": len(cctv_videos),
                "photos_used": len(reference_photos)
            }
            
        except Exception as e:
            print(f"❌ Error processing case {case_id}: {str(e)}")
            return {"error": str(e)}

def batch_process_all_approved_cases():
    """
    Process all approved cases that haven't been processed yet
    """
    app = create_app()
    with app.app_context():
        approved_cases = Case.query.filter_by(status='Approved').all()
        
        results = []
        for case in approved_cases:
            print(f"\n🔄 Processing Case {case.id}...")
            result = process_approved_case(case.id)
            results.append(result)
        
        return results

if __name__ == "__main__":
    # Test with specific case
    if len(sys.argv) > 1:
        case_id = int(sys.argv[1])
        result = process_approved_case(case_id)
        print(f"Result: {result}")
    else:
        # Process all approved cases
        results = batch_process_all_approved_cases()
        print(f"Processed {len(results)} cases")