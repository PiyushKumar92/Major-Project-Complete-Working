import json
import math
from datetime import datetime

class LocationAnalyzer:
    def __init__(self):
        self.earth_radius_km = 6371
        
    def parse_location_details(self, location_string, details_string=None):
        """Parse comprehensive location from case data"""
        location_data = {
            'street_address': '',
            'area': '',
            'city': '',
            'state': '',
            'pincode': '',
            'landmarks': '',
            'full_address': location_string or ''
        }
        
        if details_string:
            # Extract location details from case details
            lines = details_string.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'Location Area':
                        location_data['area'] = value
                    elif key == 'Location City':
                        location_data['city'] = value
                    elif key == 'Location State':
                        location_data['state'] = value
                    elif key == 'Location PIN':
                        location_data['pincode'] = value
                    elif key == 'Landmarks':
                        location_data['landmarks'] = value
        
        return location_data
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using Haversine formula"""
        try:
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return self.earth_radius_km * c
        except:
            return float('inf')
    
    def match_locations(self, case_location, footage_location, max_distance_km=5):
        """Match case location with footage location"""
        match_score = 0
        match_details = {
            'exact_match': False,
            'city_match': False,
            'state_match': False,
            'pincode_match': False,
            'area_match': False,
            'distance_km': None,
            'match_score': 0
        }
        
        # Parse both locations
        case_data = self.parse_location_details(
            case_location.get('full_address', ''),
            case_location.get('details', '')
        )
        
        footage_data = footage_location
        
        # Check exact matches
        if case_data['pincode'] and footage_data.get('pincode'):
            if case_data['pincode'] == footage_data['pincode']:
                match_details['pincode_match'] = True
                match_score += 40  # High weight for PIN code match
        
        if case_data['city'] and footage_data.get('city'):
            if case_data['city'].lower() == footage_data['city'].lower():
                match_details['city_match'] = True
                match_score += 25
        
        if case_data['state'] and footage_data.get('state'):
            if case_data['state'].lower() == footage_data['state'].lower():
                match_details['state_match'] = True
                match_score += 15
        
        if case_data['area'] and footage_data.get('area_locality'):
            if case_data['area'].lower() in footage_data['area_locality'].lower() or \
               footage_data['area_locality'].lower() in case_data['area'].lower():
                match_details['area_match'] = True
                match_score += 20
        
        # Calculate coordinate distance if available
        if (case_location.get('latitude') and case_location.get('longitude') and
            footage_data.get('latitude') and footage_data.get('longitude')):
            
            distance = self.calculate_distance(
                case_location['latitude'], case_location['longitude'],
                footage_data['latitude'], footage_data['longitude']
            )
            
            match_details['distance_km'] = distance
            
            if distance <= max_distance_km:
                # Closer distance = higher score
                distance_score = max(0, (max_distance_km - distance) / max_distance_km * 20)
                match_score += distance_score
        
        # Landmark matching
        if case_data['landmarks'] and footage_data.get('landmarks'):
            case_landmarks = case_data['landmarks'].lower().split(',')
            footage_landmarks = footage_data['landmarks'].lower().split(',')
            
            landmark_matches = 0
            for case_landmark in case_landmarks:
                case_landmark = case_landmark.strip()
                for footage_landmark in footage_landmarks:
                    footage_landmark = footage_landmark.strip()
                    if case_landmark in footage_landmark or footage_landmark in case_landmark:
                        landmark_matches += 1
                        break
            
            if landmark_matches > 0:
                match_score += min(landmark_matches * 10, 30)  # Max 30 points for landmarks
        
        # Normalize score to 0-100
        match_details['match_score'] = min(match_score, 100) / 100
        
        # Determine if it's a good match
        if match_details['match_score'] >= 0.7:
            match_details['exact_match'] = True
        
        return match_details
    
    def find_matching_footage(self, case_location_data, footage_list):
        """Find all footage that matches case location"""
        matches = []
        
        for footage in footage_list:
            footage_location = {
                'street_address': footage.get('street_address', ''),
                'area_locality': footage.get('area_locality', ''),
                'city': footage.get('city', ''),
                'state': footage.get('state', ''),
                'pincode': footage.get('pincode', ''),
                'landmarks': footage.get('landmarks', ''),
                'latitude': footage.get('latitude'),
                'longitude': footage.get('longitude')
            }
            
            match_result = self.match_locations(case_location_data, footage_location)
            
            if match_result['match_score'] >= 0.3:  # Minimum threshold
                matches.append({
                    'footage_id': footage.get('id'),
                    'footage_title': footage.get('title', ''),
                    'location_name': footage.get('location_name', ''),
                    'match_details': match_result,
                    'priority': 'high' if match_result['match_score'] >= 0.7 else 
                               'medium' if match_result['match_score'] >= 0.5 else 'low'
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_details']['match_score'], reverse=True)
        
        return matches
    
    def generate_location_report(self, case_id, matches):
        """Generate detailed location analysis report"""
        report = {
            'case_id': case_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_footage_analyzed': len(matches),
            'high_priority_matches': len([m for m in matches if m['priority'] == 'high']),
            'medium_priority_matches': len([m for m in matches if m['priority'] == 'medium']),
            'low_priority_matches': len([m for m in matches if m['priority'] == 'low']),
            'matches': matches,
            'recommendations': []
        }
        
        # Add recommendations
        if report['high_priority_matches'] > 0:
            report['recommendations'].append("High priority matches found - immediate AI analysis recommended")
        
        if report['medium_priority_matches'] > 0:
            report['recommendations'].append("Medium priority matches available for secondary analysis")
        
        if report['total_footage_analyzed'] == 0:
            report['recommendations'].append("No matching footage found - consider expanding search radius")
        
        return report

def analyze_case_location(case_id, case_location_data, available_footage):
    """Main function to analyze case location against available footage"""
    analyzer = LocationAnalyzer()
    
    # Find matching footage
    matches = analyzer.find_matching_footage(case_location_data, available_footage)
    
    # Generate report
    report = analyzer.generate_location_report(case_id, matches)
    
    return report