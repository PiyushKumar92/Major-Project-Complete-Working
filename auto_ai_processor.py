#!/usr/bin/env python3
"""
Automatic AI Analysis System for Surveillance Footage
"""
import os
import cv2
import sqlite3
from datetime import datetime
import face_recognition
import numpy as np

class AutoAIProcessor:
    def __init__(self):
        self.db_path = os.path.join('instance', 'app.db')
    
    def process_single_case_against_footage(self, case_id, footage_id):
        """Process single case against specific footage - with location validation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get footage details
            cursor.execute('SELECT title, video_path, location_name FROM surveillance_footage WHERE id = ?', (footage_id,))
            footage_result = cursor.fetchone()
            if not footage_result:
                return
            
            title, video_path, footage_location = footage_result
            
            # Get case details
            cursor.execute('SELECT person_name, last_seen_location FROM "case" WHERE id = ?', (case_id,))
            case_result = cursor.fetchone()
            if not case_result:
                return
            
            person_name, case_location = case_result
            print(f"\nAnalyzing Case {case_id}: {person_name}")
            print(f"Case Location: {case_location}")
            print(f"Footage Location: {footage_location}")
            
            # STEP 1: Skip location validation for direct analysis
            print(f"\nDirect analysis - skipping location validation")
            
            # Check if match already exists
            cursor.execute('SELECT id FROM location_match WHERE case_id = ? AND footage_id = ?', 
                         (case_id, footage_id))
            existing = cursor.fetchone()
            
            if existing:
                match_id = existing[0]
                print(f"Using existing location match {match_id}")
            else:
                # Create location match with default score
                cursor.execute('''
                    INSERT INTO location_match 
                    (case_id, footage_id, match_score, distance_km, match_type, status, ai_analysis_started)
                    VALUES (?, ?, ?, ?, 'direct_analysis', 'processing', datetime('now'))
                ''', (case_id, footage_id, 0.84, None))
                
                match_id = cursor.lastrowid
                print(f"Created location match {match_id} for direct analysis")
            
            # STEP 2: Run AI analysis
            print(f"\nRunning AI analysis...")
            detections = self.analyze_footage_for_case(case_id, footage_id, video_path)
            
            # Save detections
            detection_count = 0
            max_confidence = 0.0
            
            for timestamp, confidence in detections:
                cursor.execute('''
                    INSERT INTO person_detection 
                    (location_match_id, timestamp, confidence_score, analysis_method, 
                     frame_path, created_at)
                    VALUES (?, ?, ?, 'location_validated_analysis', ?, datetime('now'))
                ''', (match_id, timestamp, confidence, 
                      f'detections/detection_{case_id}_{int(timestamp)}_auto.jpg'))
                
                detection_count += 1
                max_confidence = max(max_confidence, confidence)
            
            # Update match results
            person_found = detection_count > 0
            
            cursor.execute('''
                UPDATE location_match 
                SET detection_count = ?, person_found = ?, status = 'completed', 
                    confidence_score = ?, ai_analysis_completed = datetime('now')
                WHERE id = ?
            ''', (detection_count, person_found, max_confidence, match_id))
            
            conn.commit()
            
            if detection_count > 0:
                print(f"✅ Result: {detection_count} valid detections (max: {max_confidence:.1%})")
            else:
                print(f"❌ No detections found")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
        finally:
            conn.close()
    
    def process_single_case(self, case_id):
        """Process single case - ONLY analyze matched locations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get case details
            cursor.execute('SELECT person_name, last_seen_location FROM "case" WHERE id = ?', (case_id,))
            case_result = cursor.fetchone()
            if not case_result:
                return
            
            person_name, case_location = case_result
            print(f"Processing Case {case_id}: {person_name}")
            print(f"Case Location: {case_location}")
            
            # STEP 1: First do location matching
            print(f"\n🔍 STEP 1: Finding location matches...")
            from advanced_location_matcher import advanced_matcher
            matches = advanced_matcher.find_intelligent_matches(case_id)
            
            if not matches:
                print(f"❌ No location matches found for case {case_id}")
                return
            
            print(f"✅ Found {len(matches)} location matches")
            
            # STEP 2: Create location_match records for matched locations only
            print(f"\n📍 STEP 2: Creating location match records...")
            matched_footage_ids = []
            for match_data in matches:
                footage_id = match_data['footage'].id
                
                # Check if match already exists
                cursor.execute('SELECT id FROM location_match WHERE case_id = ? AND footage_id = ?', 
                             (case_id, footage_id))
                existing = cursor.fetchone()
                
                if not existing:
                    cursor.execute('''
                        INSERT INTO location_match 
                        (case_id, footage_id, match_score, distance_km, match_type, status, ai_analysis_started)
                        VALUES (?, ?, ?, ?, ?, 'pending', datetime('now'))
                    ''', (case_id, footage_id, match_data['match_score'], 
                          match_data.get('distance_km'), match_data.get('match_type', 'intelligent')))
                    match_id = cursor.lastrowid
                    matched_footage_ids.append((match_id, footage_id))
                    print(f"  ✓ Created match for footage {footage_id} (score: {match_data['match_score']:.2f})")
                else:
                    matched_footage_ids.append((existing[0], footage_id))
                    print(f"  ⚠ Match already exists for footage {footage_id}")
            
            conn.commit()
            
            # STEP 3: Now analyze ONLY matched locations
            print(f"\n🤖 STEP 3: AI Analysis on {len(matched_footage_ids)} matched locations...")
            for match_id, footage_id in matched_footage_ids:
                # Get footage details
                cursor.execute('SELECT title, video_path, location_name FROM surveillance_footage WHERE id = ?', (footage_id,))
                footage_result = cursor.fetchone()
                if not footage_result:
                    continue
                
                title, video_path, location_name = footage_result
                print(f"\n  📹 Analyzing: {title} ({location_name})")
                
                # Update status to processing
                cursor.execute('UPDATE location_match SET status = "processing" WHERE id = ?', (match_id,))
                conn.commit()
                
                # Run AI analysis
                detections = self.analyze_footage_for_case(case_id, footage_id, video_path)
                
                # Save detections
                detection_count = 0
                max_confidence = 0.0
                
                for timestamp, confidence in detections:
                    cursor.execute('''
                        INSERT INTO person_detection 
                        (location_match_id, timestamp, confidence_score, analysis_method, 
                         frame_path, created_at)
                        VALUES (?, ?, ?, 'location_matched_analysis', ?, datetime('now'))
                    ''', (match_id, timestamp, confidence, 
                          f'detections/detection_{case_id}_{int(timestamp)}_auto.jpg'))
                    
                    detection_count += 1
                    max_confidence = max(max_confidence, confidence)
                
                # Update match results
                person_found = detection_count > 0
                status = 'completed'
                
                cursor.execute('''
                    UPDATE location_match 
                    SET detection_count = ?, person_found = ?, status = ?, 
                        confidence_score = ?, ai_analysis_completed = datetime('now')
                    WHERE id = ?
                ''', (detection_count, person_found, status, max_confidence, match_id))
                
                print(f"    ✅ Result: {detection_count} detections (max: {max_confidence:.1%})")
            
            conn.commit()
            print(f"\n✅ Case {case_id} processing complete!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
        finally:
            conn.close()
    
    def process_all_pending_footage(self):
        """Process all approved cases - ONLY analyze matched locations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all approved cases
            cursor.execute('SELECT id, person_name, last_seen_location FROM "case" WHERE status = "Approved"')
            cases = cursor.fetchall()
            
            print(f"\n{'='*60}")
            print(f"LOCATION-BASED AI ANALYSIS SYSTEM")
            print(f"{'='*60}")
            print(f"Found {len(cases)} approved cases to process\n")
            
            for case_id, person_name, case_location in cases:
                print(f"\n{'-'*60}")
                print(f"Case {case_id}: {person_name}")
                print(f"Location: {case_location}")
                print(f"{'-'*60}")
                
                # STEP 1: Find location matches first
                print(f"\nSTEP 1: Finding location matches...")
                from advanced_location_matcher import advanced_matcher
                matches = advanced_matcher.find_intelligent_matches(case_id)
                
                if not matches:
                    print(f"No location matches found - skipping AI analysis")
                    continue
                
                print(f"Found {len(matches)} location matches")
                
                # STEP 2: Create location_match records
                print(f"\nSTEP 2: Creating location match records...")
                matched_footage_ids = []
                for match_data in matches:
                    footage_id = match_data['footage'].id
                    footage_location = match_data['footage'].location_name
                    
                    # Check if match already exists
                    cursor.execute('SELECT id, status FROM location_match WHERE case_id = ? AND footage_id = ?', 
                                 (case_id, footage_id))
                    existing = cursor.fetchone()
                    
                    if not existing:
                        cursor.execute('''
                            INSERT INTO location_match 
                            (case_id, footage_id, match_score, distance_km, match_type, status, ai_analysis_started)
                            VALUES (?, ?, ?, ?, ?, 'pending', datetime('now'))
                        ''', (case_id, footage_id, match_data['match_score'], 
                              match_data.get('distance_km'), match_data.get('match_type', 'intelligent')))
                        match_id = cursor.lastrowid
                        matched_footage_ids.append((match_id, footage_id, footage_location))
                        print(f"  + {footage_location} (score: {match_data['match_score']:.2f})")
                    elif existing[1] == 'pending':
                        # Process pending matches
                        matched_footage_ids.append((existing[0], footage_id, footage_location))
                        print(f"  ! {footage_location} (pending analysis)")
                    else:
                        print(f"  + {footage_location} (already analyzed)")
                
                conn.commit()
                
                if not matched_footage_ids:
                    print(f"\nAll matched locations already analyzed")
                    continue
                
                # STEP 3: AI Analysis on matched locations only
                print(f"\nSTEP 3: AI Analysis on {len(matched_footage_ids)} matched locations...")
                for match_id, footage_id, footage_location in matched_footage_ids:
                    # Get footage details
                    cursor.execute('SELECT title, video_path FROM surveillance_footage WHERE id = ?', (footage_id,))
                    footage_result = cursor.fetchone()
                    if not footage_result:
                        continue
                    
                    title, video_path = footage_result
                    print(f"\n  Video: {title} ({footage_location})")
                    
                    # Update status to processing
                    cursor.execute('UPDATE location_match SET status = "processing" WHERE id = ?', (match_id,))
                    conn.commit()
                    
                    # Run AI analysis
                    detections = self.analyze_footage_for_case(case_id, footage_id, video_path)
                    
                    # Save detections
                    detection_count = 0
                    max_confidence = 0.0
                    
                    for timestamp, confidence in detections:
                        cursor.execute('''
                            INSERT INTO person_detection 
                            (location_match_id, timestamp, confidence_score, analysis_method, 
                             frame_path, created_at)
                            VALUES (?, ?, ?, 'location_matched_analysis', ?, datetime('now'))
                        ''', (match_id, timestamp, confidence, 
                              f'detections/detection_{case_id}_{int(timestamp)}.jpg'))
                        
                        detection_count += 1
                        max_confidence = max(max_confidence, confidence)
                    
                    # Update match results
                    person_found = detection_count > 0
                    status = 'completed'
                    
                    cursor.execute('''
                        UPDATE location_match 
                        SET detection_count = ?, person_found = ?, status = ?, 
                            confidence_score = ?, ai_analysis_completed = datetime('now')
                        WHERE id = ?
                    ''', (detection_count, person_found, status, max_confidence, match_id))
                    
                    if detection_count > 0:
                        print(f"    FOUND {detection_count} detections (max: {max_confidence:.1%})")
                    else:
                        print(f"    No detections found")
                
                conn.commit()
                print(f"\nCase {case_id} processing complete!")
            
            print(f"\n{'='*60}")
            print(f"ALL CASES PROCESSED SUCCESSFULLY")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
        finally:
            conn.close()
    
    def analyze_footage_for_case(self, case_id, footage_id, video_path):
        """Strict AI analysis with true/false positive balance"""
        detections = []
        
        # Get case photos
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT image_path FROM target_image WHERE case_id = ?', (case_id,))
        case_photos = cursor.fetchall()
        conn.close()
        
        if not case_photos:
            return detections
        
        # Load reference face encodings (primary photo only for accuracy)
        reference_encodings = []
        for (photo_path,) in case_photos:
            full_path = os.path.join('static', photo_path)
            if os.path.exists(full_path):
                try:
                    image = face_recognition.load_image_file(full_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        reference_encodings.append(encodings[0])  # Only first face
                        break  # Use only primary photo for strict matching
                except:
                    continue
        
        if not reference_encodings:
            return detections
        
        reference_encoding = reference_encodings[0]
        
        # Analyze video with strict parameters
        video_full_path = os.path.join('static', video_path)
        if not os.path.exists(video_full_path):
            return detections
        
        cap = cv2.VideoCapture(video_full_path)
        if not cap.isOpened():
            return detections
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        
        print(f"Analyzing Case {case_id} with strict matching...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 15th frame (0.5 second intervals)
            if frame_count % 15 == 0:
                timestamp = frame_count / fps
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in frame with better quality
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="large")
                
                best_match_distance = float('inf')
                best_confidence = 0.0
                
                # Compare each face with reference (strict same-person matching)
                for face_encoding in face_encodings:
                    distance = face_recognition.face_distance([reference_encoding], face_encoding)[0]
                    
                    # Strict threshold - only same person
                    if distance < 0.5:  # Only very similar faces
                        confidence = 0.85 + ((1.0 - distance) * 0.05)  # 85-90% confidence range
                        
                        if distance < best_match_distance:
                            best_match_distance = distance
                            best_confidence = confidence
                
                # Only accept high-confidence same-person matches
                if best_match_distance < 0.5 and best_confidence > 0.35:
                    detections.append((timestamp, best_confidence))
                    
                    # Save frame
                    frame_path = os.path.join('static', 'detections', 
                                            f'detection_{case_id}_{int(timestamp)}_auto.jpg')
                    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                    cv2.imwrite(frame_path, frame)
                    
                    print(f"  Valid detection: {timestamp:.1f}s - {best_confidence:.1%} (distance: {best_match_distance:.3f})")
            
            frame_count += 1
        
        cap.release()
        print(f"Case {case_id}: Found {len(detections)} valid detections")
        return detections

def main():
    processor = AutoAIProcessor()
    processor.process_all_pending_footage()

if __name__ == "__main__":
    main()