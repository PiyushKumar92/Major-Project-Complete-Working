"""
Advanced AI Location Matcher - Fully Automated Professional System
Handles all case types with intelligent matching and zero human intervention
"""
import os
import json
import logging
import requests
from datetime import datetime, timedelta
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import threading
import time
from __init__ import db
from models import Case, SurveillanceFootage, LocationMatch, PersonDetection
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedLocationMatcher:
    def __init__(self):
        self.geocoder = Nominatim(user_agent="investigation_platform")
        self.processing_active = False
        
        # Case-specific matching criteria
        self.CASE_CRITERIA = {
            'missing_person': {
                'radius_km': 25,
                'time_window_hours': 168,  # 7 days
                'priority_multiplier': 1.5,
                'confidence_threshold': 0.4
            },
            'criminal_investigation': {
                'radius_km': 10,
                'time_window_hours': 72,   # 3 days
                'priority_multiplier': 2.0,
                'confidence_threshold': 0.6
            },
            'surveillance_request': {
                'radius_km': 5,
                'time_window_hours': 24,   # 1 day
                'priority_multiplier': 1.8,
                'confidence_threshold': 0.7
            },
            'person_tracking': {
                'radius_km': 50,
                'time_window_hours': 336,  # 14 days
                'priority_multiplier': 1.3,
                'confidence_threshold': 0.3
            },
            'evidence_analysis': {
                'radius_km': 15,
                'time_window_hours': 120,  # 5 days
                'priority_multiplier': 1.6,
                'confidence_threshold': 0.5
            }
        }
        
        # Priority-based radius adjustment
        self.PRIORITY_RADIUS = {
            'Critical': 2.0,  # Double radius
            'High': 1.5,      # 1.5x radius
            'Medium': 1.0,    # Normal radius
            'Low': 0.7        # Reduced radius
        }
        
        # Requester type priority boost
        self.REQUESTER_BOOST = {
            'police': 2.0,
            'government': 1.8,
            'private_investigator': 1.5,
            'organization': 1.3,
            'family': 1.0
        }

    def geocode_location(self, location_string):
        """Convert address to GPS coordinates with fallback methods"""
        try:
            if not location_string:
                return None, None
            
            # Clean location string
            clean_location = self._clean_location_string(location_string)
            
            # Try primary geocoding
            location = self.geocoder.geocode(clean_location, timeout=10)
            if location:
                return location.latitude, location.longitude
            
            # Fallback: Try with just city/area
            location_parts = clean_location.split(',')
            for i in range(len(location_parts)):
                partial_location = ','.join(location_parts[i:]).strip()
                if partial_location:
                    location = self.geocoder.geocode(partial_location, timeout=10)
                    if location:
                        return location.latitude, location.longitude
            
            return None, None
            
        except Exception as e:
            logger.error(f"Geocoding error for '{location_string}': {str(e)}")
            return None, None

    def _clean_location_string(self, location):
        """Clean and standardize location string"""
        if not location:
            return ""
        
        # Remove extra spaces and normalize
        clean = ' '.join(location.split())
        
        # Remove common prefixes/suffixes
        prefixes = ['near', 'close to', 'around', 'at']
        for prefix in prefixes:
            if clean.lower().startswith(prefix + ' '):
                clean = clean[len(prefix):].strip()
        
        return clean

    def calculate_smart_radius(self, case):
        """Calculate intelligent radius based on case parameters"""
        try:
            # Base radius from case type
            case_type = case.case_type or 'missing_person'
            base_radius = self.CASE_CRITERIA.get(case_type, {}).get('radius_km', 15)
            
            # Priority adjustment
            priority_multiplier = self.PRIORITY_RADIUS.get(case.priority, 1.0)
            
            # Requester type boost
            requester_boost = self.REQUESTER_BOOST.get(case.requester_type, 1.0)
            
            # Urgency adjustment
            urgency_multiplier = {
                'critical': 2.5,
                'high': 1.8,
                'medium': 1.0,
                'low': 0.6
            }.get(case.urgency_level, 1.0)
            
            # Calculate final radius
            final_radius = base_radius * priority_multiplier * requester_boost * urgency_multiplier
            
            # Ensure reasonable bounds
            return max(2, min(final_radius, 100))  # 2km to 100km
            
        except Exception as e:
            logger.error(f"Error calculating radius for case {case.id}: {str(e)}")
            return 15  # Default radius

    def find_intelligent_matches(self, case_id):
        """Advanced intelligent location matching"""
        try:
            case = Case.query.get(case_id)
            if not case:
                return []
            
            # Geocode case location
            case_lat, case_lon = self.geocode_location(case.last_seen_location)
            if not case_lat or not case_lon:
                # Fallback to string matching
                return self._fallback_string_matching(case)
            
            # Calculate smart radius
            search_radius = self.calculate_smart_radius(case)
            
            # Get time window
            case_type = case.case_type or 'missing_person'
            time_window = self.CASE_CRITERIA.get(case_type, {}).get('time_window_hours', 168)
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window)
            
            # Find footage within radius and time window
            all_footage = SurveillanceFootage.query.filter(
                SurveillanceFootage.is_active == True,
                SurveillanceFootage.date_recorded >= cutoff_time
            ).all()
            
            matches = []
            for footage in all_footage:
                match_data = self._calculate_match_score(case, footage, case_lat, case_lon, search_radius)
                if match_data and match_data['match_score'] > 0.2:  # Lowered from 0.3 to 0.2
                    matches.append(match_data)
            
            # Sort by match score
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            return matches[:20]  # Top 20 matches
            
        except Exception as e:
            logger.error(f"Error finding matches for case {case_id}: {str(e)}")
            return []

    def _calculate_match_score(self, case, footage, case_lat, case_lon, search_radius):
        """Calculate comprehensive match score"""
        try:
            match_score = 0.0
            factors = {}
            
            # Geographic proximity (40% weight)
            if footage.latitude and footage.longitude:
                distance = geodesic((case_lat, case_lon), (footage.latitude, footage.longitude)).kilometers
                if distance <= search_radius:
                    geo_score = max(0, 1 - (distance / search_radius))
                    match_score += geo_score * 0.4
                    factors['geographic'] = geo_score
                else:
                    return None  # Outside radius
            
            # Location name similarity (25% weight)
            name_score = self._calculate_name_similarity(case.last_seen_location, footage.location_name)
            match_score += name_score * 0.25
            factors['name_similarity'] = name_score
            
            # If name similarity is very high but no GPS, still consider it
            if not (footage.latitude and footage.longitude) and name_score >= 0.7:
                # No GPS but good name match - use name score as primary
                match_score = name_score * 0.8  # 80% weight to name when no GPS
                factors['match_type'] = 'name_based'
            
            # Time relevance (20% weight)
            time_score = self._calculate_time_relevance(case, footage)
            match_score += time_score * 0.20
            factors['time_relevance'] = time_score
            
            # Camera quality and type (10% weight)
            quality_score = self._calculate_quality_score(footage)
            match_score += quality_score * 0.10
            factors['quality'] = quality_score
            
            # Case priority boost (5% weight)
            priority_score = self._calculate_priority_boost(case)
            match_score += priority_score * 0.05
            factors['priority'] = priority_score
            
            return {
                'footage': footage,
                'match_score': min(match_score, 1.0),
                'distance_km': distance if footage.latitude and footage.longitude else None,
                'factors': factors,
                'match_type': 'intelligent'
            }
            
        except Exception as e:
            logger.error(f"Error calculating match score: {str(e)}")
            return None

    def _calculate_name_similarity(self, case_location, footage_location):
        """Advanced location name similarity with better matching"""
        if not case_location or not footage_location:
            return 0.0
        
        case_clean = case_location.lower().strip()
        footage_clean = footage_location.lower().strip()
        
        # Exact match
        if case_clean == footage_clean:
            return 1.0
        
        # Remove spaces for better matching (pantnagar = pant nagar)
        case_no_space = case_clean.replace(' ', '')
        footage_no_space = footage_clean.replace(' ', '')
        
        if case_no_space == footage_no_space:
            return 0.95
        
        # Substring match (without spaces)
        if case_no_space in footage_no_space or footage_no_space in case_no_space:
            return 0.90
        
        # Substring match (with spaces)
        if case_clean in footage_clean or footage_clean in case_clean:
            return 0.85
        
        # Word-based similarity (improved)
        case_words = set(case_clean.split())
        footage_words = set(footage_clean.split())
        
        if case_words and footage_words:
            intersection = case_words.intersection(footage_words)
            union = case_words.union(footage_words)
            
            if intersection:
                # Jaccard similarity
                jaccard = len(intersection) / len(union)
                
                # Bonus for matching important words
                important_match = any(word in intersection for word in case_words if len(word) > 3)
                bonus = 0.1 if important_match else 0.0
                
                return min(jaccard * 0.8 + bonus, 0.85)
        
        return 0.0

    def _calculate_time_relevance(self, case, footage):
        """Calculate time-based relevance score"""
        try:
            if not case.date_missing or not footage.date_recorded:
                return 0.5  # Default score
            
            # Time difference in hours
            time_diff = abs((case.date_missing - footage.date_recorded).total_seconds() / 3600)
            
            # Get case-specific time window
            case_type = case.case_type or 'missing_person'
            max_hours = self.CASE_CRITERIA.get(case_type, {}).get('time_window_hours', 168)
            
            if time_diff <= max_hours:
                # Closer in time = higher score
                return max(0, 1 - (time_diff / max_hours))
            
            return 0.0
            
        except Exception:
            return 0.5

    def _calculate_quality_score(self, footage):
        """Calculate footage quality score"""
        score = 0.5  # Base score
        
        # Resolution bonus
        if footage.resolution:
            if 'FHD' in footage.resolution or '1080' in footage.resolution:
                score += 0.3
            elif 'HD' in footage.resolution or '720' in footage.resolution:
                score += 0.2
        
        # Camera type bonus
        if footage.camera_type:
            if 'CCTV' in footage.camera_type.upper():
                score += 0.1
            elif 'Security' in footage.camera_type:
                score += 0.1
        
        # Quality rating bonus
        if footage.quality:
            quality_bonus = {
                '4K': 0.2,
                'FHD': 0.15,
                'HD': 0.1,
                'SD': 0.0
            }.get(footage.quality, 0.0)
            score += quality_bonus
        
        return min(score, 1.0)

    def _calculate_priority_boost(self, case):
        """Calculate priority-based boost"""
        priority_scores = {
            'Critical': 1.0,
            'High': 0.8,
            'Medium': 0.5,
            'Low': 0.2
        }
        
        requester_scores = {
            'police': 1.0,
            'government': 0.9,
            'private_investigator': 0.7,
            'organization': 0.5,
            'family': 0.3
        }
        
        priority_score = priority_scores.get(case.priority, 0.5)
        requester_score = requester_scores.get(case.requester_type, 0.3)
        
        return (priority_score + requester_score) / 2

    def _fallback_string_matching(self, case):
        """Fallback string-based matching when geocoding fails"""
        try:
            if not case.last_seen_location:
                return []
            
            footage_list = SurveillanceFootage.query.filter_by(is_active=True).all()
            matches = []
            
            for footage in footage_list:
                name_score = self._calculate_name_similarity(case.last_seen_location, footage.location_name)
                if name_score > 0.3:
                    matches.append({
                        'footage': footage,
                        'match_score': name_score * 0.7,  # Reduced confidence for string-only
                        'distance_km': None,
                        'match_type': 'string_fallback'
                    })
            
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            return matches[:10]
            
        except Exception as e:
            logger.error(f"Error in fallback matching: {str(e)}")
            return []

    def auto_process_approved_case(self, case_id):
        """Automatically process approved case"""
        try:
            logger.info(f"Auto-processing case {case_id}")
            
            # Find intelligent matches
            matches = self.find_intelligent_matches(case_id)
            
            created_matches = 0
            for match_data in matches:
                # Check if match already exists
                existing = LocationMatch.query.filter_by(
                    case_id=case_id,
                    footage_id=match_data['footage'].id
                ).first()
                
                if not existing:
                    location_match = LocationMatch(
                        case_id=case_id,
                        footage_id=match_data['footage'].id,
                        match_score=match_data['match_score'],
                        distance_km=match_data['distance_km'],
                        match_type=match_data['match_type'],
                        status='pending'
                    )
                    db.session.add(location_match)
                    created_matches += 1
            
            if created_matches > 0:
                db.session.commit()
                logger.info(f"Created {created_matches} location matches for case {case_id}")
                
                # Auto-start AI analysis for high-confidence matches
                self._auto_start_ai_analysis(case_id)
            
            return created_matches
            
        except Exception as e:
            logger.error(f"Error auto-processing case {case_id}: {str(e)}")
            db.session.rollback()
            return 0

    def _auto_start_ai_analysis(self, case_id):
        """Automatically start AI analysis for high-confidence matches"""
        try:
            # Get high-confidence matches
            high_confidence_matches = LocationMatch.query.filter(
                LocationMatch.case_id == case_id,
                LocationMatch.match_score >= 0.7,
                LocationMatch.status == 'pending'
            ).limit(5).all()
            
            for match in high_confidence_matches:
                # Start AI analysis in background
                threading.Thread(
                    target=self._analyze_match_async,
                    args=(match.id,),
                    daemon=True
                ).start()
                
        except Exception as e:
            logger.error(f"Error starting auto AI analysis: {str(e)}")

    def _analyze_match_async(self, match_id):
        """Asynchronous AI analysis"""
        try:
            from ai_location_matcher import ai_matcher
            ai_matcher.analyze_footage_for_person(match_id)
        except Exception as e:
            logger.error(f"Error in async AI analysis for match {match_id}: {str(e)}")

    def start_continuous_monitoring(self):
        """Start continuous monitoring for new cases and footage"""
        if self.processing_active:
            return
        
        self.processing_active = True
        threading.Thread(target=self._continuous_monitor, daemon=True).start()
        logger.info("Started continuous location matching monitoring")

    def _continuous_monitor(self):
        """Continuous monitoring loop"""
        from __init__ import create_app
        app = create_app()
        
        while self.processing_active:
            try:
                with app.app_context():
                    # Process pending approved cases
                    pending_cases = Case.query.filter_by(status='Approved').all()
                    for case in pending_cases:
                        # Check if already processed
                        existing_matches = LocationMatch.query.filter_by(case_id=case.id).count()
                        if existing_matches == 0:
                            self.auto_process_approved_case(case.id)
                    
                    # Process new surveillance footage
                    self._process_new_footage()
                
                # Sleep before next cycle
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {str(e)}")
                time.sleep(600)  # 10 minutes on error

    def _process_new_footage(self):
        """Process newly added surveillance footage"""
        try:
            # Get footage added in last hour
            recent_time = datetime.utcnow() - timedelta(hours=1)
            new_footage = SurveillanceFootage.query.filter(
                SurveillanceFootage.created_at >= recent_time,
                SurveillanceFootage.is_active == True
            ).all()
            
            if not new_footage:
                return
            
            # Get active cases
            active_cases = Case.query.filter(
                Case.status.in_(['Approved', 'Under Investigation', 'Processing'])
            ).all()
            
            for footage in new_footage:
                for case in active_cases:
                    # Check if match already exists
                    existing = LocationMatch.query.filter_by(
                        case_id=case.id,
                        footage_id=footage.id
                    ).first()
                    
                    if not existing:
                        # Calculate match
                        case_lat, case_lon = self.geocode_location(case.last_seen_location)
                        if case_lat and case_lon:
                            search_radius = self.calculate_smart_radius(case)
                            match_data = self._calculate_match_score(case, footage, case_lat, case_lon, search_radius)
                            
                            if match_data and match_data['match_score'] > 0.4:
                                location_match = LocationMatch(
                                    case_id=case.id,
                                    footage_id=footage.id,
                                    match_score=match_data['match_score'],
                                    distance_km=match_data['distance_km'],
                                    match_type='auto_new_footage',
                                    status='pending'
                                )
                                db.session.add(location_match)
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error processing new footage: {str(e)}")
            db.session.rollback()

    def get_matching_statistics(self, case_id):
        """Get detailed matching statistics for a case"""
        try:
            matches = LocationMatch.query.filter_by(case_id=case_id).all()
            
            stats = {
                'total_matches': len(matches),
                'high_confidence': len([m for m in matches if m.match_score >= 0.7]),
                'medium_confidence': len([m for m in matches if 0.4 <= m.match_score < 0.7]),
                'low_confidence': len([m for m in matches if m.match_score < 0.4]),
                'avg_distance': np.mean([m.distance_km for m in matches if m.distance_km]) if matches else 0,
                'processing_status': {
                    'pending': len([m for m in matches if m.status == 'pending']),
                    'processing': len([m for m in matches if m.status == 'processing']),
                    'completed': len([m for m in matches if m.status == 'completed']),
                    'failed': len([m for m in matches if m.status == 'failed'])
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics for case {case_id}: {str(e)}")
            return {}

# Global advanced matcher instance
advanced_matcher = AdvancedLocationMatcher()