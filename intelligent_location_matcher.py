"""
Intelligent Location Matching System for CCTV Footage and Missing Person Cases
"""

import re
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from __init__ import db
from models import Case, SurveillanceFootage

class IntelligentLocationMatcher:
    def __init__(self):
        self.location_weights = {
            'street_address': 0.35,
            'area_locality': 0.30,
            'city': 0.20,
            'pincode': 0.10,
            'state': 0.05
        }
        
    def find_matching_footage_for_case(self, case_id):
        """
        Find all CCTV footage that matches a case's last seen location
        """
        case = Case.query.get(case_id)
        if not case:
            return {"error": "Case not found"}
        
        # Parse case location
        case_location = self._parse_location(case.last_seen_location)
        
        # Get all available footage
        all_footage = SurveillanceFootage.query.all()
        
        # Match and score each footage
        matches = []
        for footage in all_footage:
            footage_location = self._parse_footage_location(footage)
            match_score = self._calculate_location_match_score(case_location, footage_location)
            
            if match_score > 0.3:  # 30% minimum match threshold
                matches.append({
                    'footage_id': footage.id,
                    'footage_title': footage.title,
                    'location_name': footage.location_name,
                    'match_score': match_score,
                    'match_details': self._get_match_details(case_location, footage_location),
                    'footage_data': {
                        'date_recorded': footage.date_recorded,
                        'duration': footage.duration,
                        'quality': footage.quality,
                        'file_size_mb': footage.file_size_mb,
                        'camera_type': footage.camera_type
                    }
                })
        
        # Sort by match score (highest first)
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return {
            'case_id': case_id,
            'case_location': case.last_seen_location,
            'parsed_location': case_location,
            'total_footage_available': len(all_footage),
            'matching_footage_count': len(matches),
            'matches': matches,
            'match_criteria': self._get_match_criteria_explanation()
        }
    
    def find_matching_cases_for_footage(self, footage_id):
        """
        Find all cases that match a CCTV footage location
        """
        footage = SurveillanceFootage.query.get(footage_id)
        if not footage:
            return {"error": "Footage not found"}
        
        footage_location = self._parse_footage_location(footage)
        
        # Get all active cases
        active_cases = Case.query.filter(
            Case.status.in_(['Approved', 'Under Processing', 'Pending Approval'])
        ).all()
        
        matches = []
        for case in active_cases:
            case_location = self._parse_location(case.last_seen_location)
            match_score = self._calculate_location_match_score(case_location, footage_location)
            
            if match_score > 0.3:
                matches.append({
                    'case_id': case.id,
                    'person_name': case.person_name,
                    'last_seen_location': case.last_seen_location,
                    'match_score': match_score,
                    'match_details': self._get_match_details(case_location, footage_location),
                    'case_data': {
                        'age': case.age,
                        'date_missing': case.date_missing,
                        'status': case.status,
                        'priority': case.priority
                    }
                })
        
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return {
            'footage_id': footage_id,
            'footage_location': f"{footage.location_name} - {footage.street_address}, {footage.area_locality}",
            'parsed_location': footage_location,
            'total_cases_checked': len(active_cases),
            'matching_cases_count': len(matches),
            'matches': matches
        }
    
    def _parse_location(self, location_string):
        """
        Parse case location string into components
        """
        if not location_string:
            return {}
        
        # Clean and normalize
        location = location_string.strip().lower()
        
        # Extract components using patterns
        components = {
            'street_address': '',
            'area_locality': '',
            'city': '',
            'state': '',
            'pincode': '',
            'landmarks': ''
        }
        
        # Extract PIN code
        pincode_match = re.search(r'\b(\d{6})\b', location)
        if pincode_match:
            components['pincode'] = pincode_match.group(1)
            location = location.replace(pincode_match.group(0), '').strip()
        
        # Extract landmarks (text in parentheses)
        landmarks_match = re.search(r'\(([^)]+)\)', location)
        if landmarks_match:
            components['landmarks'] = landmarks_match.group(1).strip()
            location = location.replace(landmarks_match.group(0), '').strip()
        
        # Split by commas and clean
        parts = [part.strip() for part in location.split(',') if part.strip()]
        
        # Assign parts based on common patterns
        if len(parts) >= 1:
            components['street_address'] = parts[0]
        if len(parts) >= 2:
            components['area_locality'] = parts[1]
        if len(parts) >= 3:
            components['city'] = parts[2]
        if len(parts) >= 4:
            components['state'] = parts[3]
        
        return components
    
    def _parse_footage_location(self, footage):
        """
        Parse footage location from database fields
        """
        return {
            'street_address': (footage.street_address or '').lower().strip(),
            'area_locality': (footage.area_locality or '').lower().strip(),
            'city': (footage.city or '').lower().strip(),
            'state': (footage.state or '').lower().strip(),
            'pincode': (footage.pincode or '').strip(),
            'landmarks': (footage.landmarks or '').lower().strip()
        }
    
    def _calculate_location_match_score(self, location1, location2):
        """
        Calculate match score between two locations
        """
        total_score = 0.0
        
        for component, weight in self.location_weights.items():
            val1 = location1.get(component, '').strip()
            val2 = location2.get(component, '').strip()
            
            if not val1 or not val2:
                continue
            
            # Calculate similarity
            if component == 'pincode':
                # Exact match for PIN code
                similarity = 1.0 if val1 == val2 else 0.0
            else:
                # Fuzzy match for text fields
                similarity = self._calculate_text_similarity(val1, val2)
            
            total_score += similarity * weight
        
        # Bonus for landmarks match
        landmarks1 = location1.get('landmarks', '').strip()
        landmarks2 = location2.get('landmarks', '').strip()
        if landmarks1 and landmarks2:
            landmarks_similarity = self._calculate_text_similarity(landmarks1, landmarks2)
            total_score += landmarks_similarity * 0.1  # 10% bonus
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings
        """
        if not text1 or not text2:
            return 0.0
        
        # Direct match
        if text1 == text2:
            return 1.0
        
        # Sequence matcher
        seq_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_overlap = 0.0
        
        # Substring match
        substring_match = 0.0
        if text1 in text2 or text2 in text1:
            substring_match = 0.5
        
        # Combined score
        return max(seq_similarity, word_overlap, substring_match)
    
    def _get_match_details(self, location1, location2):
        """
        Get detailed breakdown of location matches
        """
        details = {}
        
        for component in self.location_weights.keys():
            val1 = location1.get(component, '').strip()
            val2 = location2.get(component, '').strip()
            
            if val1 and val2:
                if component == 'pincode':
                    match = val1 == val2
                    similarity = 1.0 if match else 0.0
                else:
                    similarity = self._calculate_text_similarity(val1, val2)
                    match = similarity > 0.7
                
                details[component] = {
                    'case_value': val1,
                    'footage_value': val2,
                    'similarity': similarity,
                    'match': match
                }
        
        return details
    
    def _get_match_criteria_explanation(self):
        """
        Get explanation of matching criteria
        """
        return {
            'criteria': [
                {
                    'component': 'Street Address',
                    'weight': '35%',
                    'description': 'House/building number and street name'
                },
                {
                    'component': 'Area/Locality',
                    'weight': '30%',
                    'description': 'Sector, colony, or neighborhood'
                },
                {
                    'component': 'City',
                    'weight': '20%',
                    'description': 'City name'
                },
                {
                    'component': 'PIN Code',
                    'weight': '10%',
                    'description': 'Exact 6-digit postal code match'
                },
                {
                    'component': 'State',
                    'weight': '5%',
                    'description': 'State name (optional for admin)'
                }
            ],
            'minimum_threshold': '30%',
            'bonus_factors': [
                'Landmarks similarity (+10%)',
                'Exact PIN code match (high priority)',
                'Multiple component matches (cumulative)'
            ]
        }
    
    def get_location_based_suggestions(self, case_id, admin_approval_required=True):
        """
        Get AI suggestions for CCTV footage analysis based on location matching
        """
        matches = self.find_matching_footage_for_case(case_id)
        
        if 'error' in matches:
            return matches
        
        suggestions = {
            'case_id': case_id,
            'total_suggestions': len(matches['matches']),
            'analysis_ready': [],
            'requires_approval': [],
            'low_confidence': []
        }
        
        for match in matches['matches']:
            if match['match_score'] >= 0.8:
                suggestions['analysis_ready'].append({
                    **match,
                    'recommendation': 'High confidence match - Ready for AI analysis',
                    'auto_process': not admin_approval_required
                })
            elif match['match_score'] >= 0.6:
                suggestions['requires_approval'].append({
                    **match,
                    'recommendation': 'Good match - Admin approval recommended',
                    'auto_process': False
                })
            else:
                suggestions['low_confidence'].append({
                    **match,
                    'recommendation': 'Low confidence - Manual review required',
                    'auto_process': False
                })
        
        return suggestions

# Global instance
location_matcher = IntelligentLocationMatcher()