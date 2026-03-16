"""
Advanced Location Matching System
Handles complex location matching scenarios with high accuracy
"""

import re
from geopy.distance import geodesic
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

class AdvancedLocationMatcher:
    """Advanced location matching with fuzzy logic and geographic intelligence"""
    
    def __init__(self):
        # Location synonyms and variations
        self.location_synonyms = {
            'cp': ['connaught place', 'central park', 'cp metro'],
            'delhi': ['new delhi', 'ncr', 'national capital region'],
            'mumbai': ['bombay', 'mumbai city', 'greater mumbai'],
            'bangalore': ['bengaluru', 'silicon city', 'garden city'],
            'metro': ['metro station', 'subway', 'underground'],
            'market': ['bazaar', 'shopping area', 'commercial area'],
            'hospital': ['medical center', 'healthcare', 'clinic'],
            'school': ['college', 'university', 'educational institute'],
            'mall': ['shopping mall', 'shopping center', 'plaza']
        }
        
        # Common location keywords
        self.location_keywords = [
            'near', 'opposite', 'behind', 'front', 'beside', 'adjacent',
            'sector', 'block', 'phase', 'area', 'colony', 'nagar',
            'road', 'street', 'lane', 'marg', 'path', 'avenue'
        ]
        
        # Distance thresholds for matching
        self.distance_thresholds = {
            'exact': 0.1,      # Within 100m
            'very_close': 0.5,  # Within 500m
            'close': 1.0,       # Within 1km
            'nearby': 5.0,      # Within 5km
            'same_area': 10.0   # Within 10km
        }
    
    def find_location_matches(self, case_location, footage_locations):
        """Find all matching footage locations for a case location"""
        
        if not case_location:
            return []
        
        matches = []
        
        for footage in footage_locations:
            match_result = self.calculate_location_match(case_location, footage)
            if match_result['score'] > 0.3:  # Minimum threshold
                matches.append({
                    'footage': footage,
                    'match_score': match_result['score'],
                    'match_type': match_result['type'],
                    'match_reasons': match_result['reasons'],
                    'distance_km': match_result.get('distance_km'),
                    'confidence': match_result['confidence']
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matches
    
    def calculate_location_match(self, case_location, footage):
        """Calculate detailed location match score"""
        
        footage_location = footage.location_name
        footage_address = footage.location_address or ""
        
        if not footage_location:
            return {'score': 0.0, 'type': 'no_location', 'reasons': [], 'confidence': 0.0}
        
        # Normalize locations
        case_norm = self._normalize_location(case_location)
        footage_norm = self._normalize_location(footage_location)
        address_norm = self._normalize_location(footage_address)
        
        match_results = []
        
        # 1. Exact string matching
        exact_score = self._exact_match_score(case_norm, footage_norm, address_norm)
        if exact_score > 0:
            match_results.append({
                'score': exact_score,
                'type': 'exact_match',
                'reason': 'Exact location name match'
            })
        
        # 2. Fuzzy string matching
        fuzzy_score = self._fuzzy_match_score(case_norm, footage_norm, address_norm)
        if fuzzy_score > 0:
            match_results.append({
                'score': fuzzy_score,
                'type': 'fuzzy_match',
                'reason': 'Similar location names'
            })
        
        # 3. Keyword matching
        keyword_score = self._keyword_match_score(case_norm, footage_norm, address_norm)
        if keyword_score > 0:
            match_results.append({
                'score': keyword_score,
                'type': 'keyword_match',
                'reason': 'Common location keywords'
            })
        
        # 4. Synonym matching
        synonym_score = self._synonym_match_score(case_norm, footage_norm, address_norm)
        if synonym_score > 0:
            match_results.append({
                'score': synonym_score,
                'type': 'synonym_match',
                'reason': 'Location synonyms match'
            })
        
        # 5. Geographic distance matching (if coordinates available)
        distance_score = 0.0
        distance_km = None
        
        if hasattr(footage, 'latitude') and hasattr(footage, 'longitude'):
            if footage.latitude and footage.longitude:
                # For now, we don't have case coordinates, but this is where we'd calculate
                # distance_score, distance_km = self._geographic_match_score(case_coords, footage_coords)
                pass
        
        # Calculate overall match
        if not match_results:
            return {'score': 0.0, 'type': 'no_match', 'reasons': [], 'confidence': 0.0}
        
        # Get best match
        best_match = max(match_results, key=lambda x: x['score'])
        
        # Calculate confidence based on match type and score
        confidence = self._calculate_confidence(best_match, match_results)
        
        return {
            'score': best_match['score'],
            'type': best_match['type'],
            'reasons': [r['reason'] for r in match_results],
            'confidence': confidence,
            'distance_km': distance_km
        }
    
    def _normalize_location(self, location):
        """Normalize location string for better matching"""
        if not location:
            return ""
        
        # Convert to lowercase
        normalized = location.lower().strip()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common stop words
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = normalized.split()
        words = [w for w in words if w not in stop_words]
        
        return ' '.join(words)
    
    def _exact_match_score(self, case_norm, footage_norm, address_norm):
        """Calculate exact match score"""
        
        # Direct exact match
        if case_norm == footage_norm:
            return 1.0
        
        # Case location in footage address
        if case_norm in address_norm:
            return 0.9
        
        # Footage location in case location
        if footage_norm in case_norm:
            return 0.8
        
        # Check individual words
        case_words = set(case_norm.split())
        footage_words = set(footage_norm.split())
        address_words = set(address_norm.split())
        
        # All case words in footage location
        if case_words.issubset(footage_words):
            return 0.7
        
        # All case words in address
        if case_words.issubset(address_words):
            return 0.6
        
        return 0.0
    
    def _fuzzy_match_score(self, case_norm, footage_norm, address_norm):
        """Calculate fuzzy string similarity score"""
        
        # Direct similarity
        similarity = SequenceMatcher(None, case_norm, footage_norm).ratio()
        
        # Address similarity
        address_similarity = SequenceMatcher(None, case_norm, address_norm).ratio()
        
        # Take the best similarity
        best_similarity = max(similarity, address_similarity)
        
        # Only consider if similarity is reasonable
        if best_similarity >= 0.6:
            return best_similarity * 0.8  # Scale down fuzzy matches
        
        return 0.0
    
    def _keyword_match_score(self, case_norm, footage_norm, address_norm):
        """Calculate score based on common keywords"""
        
        case_words = set(case_norm.split())
        footage_words = set(footage_norm.split())
        address_words = set(address_norm.split())
        
        # Find common words
        common_with_footage = case_words.intersection(footage_words)
        common_with_address = case_words.intersection(address_words)
        
        all_common = common_with_footage.union(common_with_address)
        
        if not all_common:
            return 0.0
        
        # Calculate score based on number and importance of common words
        score = 0.0
        
        for word in all_common:
            if len(word) >= 4:  # Longer words are more significant
                if word in self.location_keywords:
                    score += 0.1  # Location keywords
                else:
                    score += 0.2  # Regular words
        
        # Normalize by total words
        total_words = len(case_words)
        if total_words > 0:
            score = min(score / total_words * 2, 0.8)  # Cap at 0.8
        
        return score
    
    def _synonym_match_score(self, case_norm, footage_norm, address_norm):
        """Calculate score based on location synonyms"""
        
        case_words = set(case_norm.split())
        footage_words = set(footage_norm.split())
        address_words = set(address_norm.split())
        
        score = 0.0
        
        for case_word in case_words:
            # Check if case word has synonyms
            if case_word in self.location_synonyms:
                synonyms = self.location_synonyms[case_word]
                
                # Check if any synonym appears in footage location or address
                for synonym in synonyms:
                    if synonym in footage_norm or synonym in address_norm:
                        score += 0.3
                        break
            
            # Check reverse - if footage word has synonyms matching case
            for footage_word in footage_words.union(address_words):
                if footage_word in self.location_synonyms:
                    synonyms = self.location_synonyms[footage_word]
                    if case_word in synonyms:
                        score += 0.3
                        break
        
        return min(score, 0.8)  # Cap at 0.8
    
    def _geographic_match_score(self, case_coords, footage_coords):
        """Calculate score based on geographic distance"""
        
        try:
            distance_km = geodesic(case_coords, footage_coords).kilometers
            
            # Calculate score based on distance
            if distance_km <= self.distance_thresholds['exact']:
                return 1.0, distance_km
            elif distance_km <= self.distance_thresholds['very_close']:
                return 0.9, distance_km
            elif distance_km <= self.distance_thresholds['close']:
                return 0.8, distance_km
            elif distance_km <= self.distance_thresholds['nearby']:
                return 0.6, distance_km
            elif distance_km <= self.distance_thresholds['same_area']:
                return 0.4, distance_km
            else:
                return 0.0, distance_km
                
        except Exception as e:
            logger.error(f"Error calculating geographic distance: {e}")
            return 0.0, None
    
    def _calculate_confidence(self, best_match, all_matches):
        """Calculate overall confidence in the match"""
        
        base_confidence = best_match['score']
        
        # Boost confidence if multiple match types agree
        if len(all_matches) > 1:
            avg_score = sum(m['score'] for m in all_matches) / len(all_matches)
            if avg_score > 0.5:
                base_confidence = min(base_confidence + 0.1, 1.0)
        
        # Adjust based on match type
        match_type_multipliers = {
            'exact_match': 1.0,
            'fuzzy_match': 0.9,
            'keyword_match': 0.8,
            'synonym_match': 0.7,
            'geographic_match': 0.95
        }
        
        multiplier = match_type_multipliers.get(best_match['type'], 0.8)
        final_confidence = base_confidence * multiplier
        
        return min(final_confidence, 1.0)


# Global location matcher instance
advanced_location_matcher = AdvancedLocationMatcher()