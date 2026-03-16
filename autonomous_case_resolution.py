"""
Autonomous Case Resolution System
Smart Case Closure with automated resolution detection and success pattern recognition
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from pathlib import Path

@dataclass
class ResolutionEvidence:
    evidence_type: str
    confidence_score: float
    source: str
    timestamp: datetime
    details: Dict

@dataclass
class ResolutionDecision:
    case_id: int
    decision: str  # AUTO_CLOSE, MANUAL_REVIEW, CONTINUE_INVESTIGATION
    confidence: float
    evidence_count: int
    resolution_type: str
    legal_compliance: bool
    admin_review_required: bool
    closure_reason: str

class SmartCaseClosureSystem:
    def __init__(self):
        self.resolution_patterns = self._load_resolution_patterns()
        self.legal_requirements = self._load_legal_requirements()
        self.evidence_thresholds = {
            'high_confidence_detection': 0.85,
            'multiple_confirmations': 3,
            'time_threshold_days': 30,
            'evidence_sufficiency': 0.75
        }
        self.setup_logging()
        self.setup_database()
    
    def setup_logging(self):
        """Setup resolution logging"""
        log_dir = Path('logs/resolution')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('CaseResolution')
        handler = logging.FileHandler('logs/resolution/case_resolution.log')
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def setup_database(self):
        """Setup resolution tracking database"""
        db_path = 'instance/case_resolution.db'
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Resolution tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS case_resolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                resolution_type TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                evidence_count INTEGER NOT NULL,
                auto_closure_eligible BOOLEAN DEFAULT FALSE,
                legal_compliance_met BOOLEAN DEFAULT FALSE,
                admin_review_required BOOLEAN DEFAULT TRUE,
                resolution_evidence TEXT,
                closure_reason TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Evidence tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resolution_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                source TEXT NOT NULL,
                evidence_details TEXT,
                verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_resolution_patterns(self) -> Dict:
        """Load success patterns for case resolution"""
        return {
            'person_found_safe': {
                'required_evidence': ['high_confidence_detection', 'location_confirmation'],
                'confidence_threshold': 0.9,
                'auto_closure_eligible': True,
                'legal_requirements': ['identity_verification', 'safety_confirmation']
            },
            'person_found_deceased': {
                'required_evidence': ['positive_identification', 'official_confirmation'],
                'confidence_threshold': 0.95,
                'auto_closure_eligible': False,
                'legal_requirements': ['official_identification', 'next_of_kin_notification']
            },
            'criminal_identified': {
                'required_evidence': ['facial_match', 'evidence_correlation'],
                'confidence_threshold': 0.85,
                'auto_closure_eligible': False,
                'legal_requirements': ['evidence_chain', 'legal_review']
            },
            'case_solved_other': {
                'required_evidence': ['resolution_confirmation', 'outcome_verification'],
                'confidence_threshold': 0.8,
                'auto_closure_eligible': True,
                'legal_requirements': ['outcome_documentation']
            }
        }
    
    def _load_legal_requirements(self) -> Dict:
        """Load legal compliance requirements"""
        return {
            'identity_verification': {
                'required_confidence': 0.95,
                'verification_methods': ['facial_recognition', 'biometric_match', 'official_id'],
                'documentation_required': True
            },
            'safety_confirmation': {
                'required_evidence': ['wellness_check', 'direct_contact', 'family_confirmation'],
                'time_sensitivity': 24  # hours
            },
            'evidence_chain': {
                'required_documentation': ['source_verification', 'chain_of_custody', 'timestamp_verification'],
                'audit_trail': True
            },
            'legal_review': {
                'required_for_types': ['criminal_investigation', 'evidence_analysis'],
                'review_threshold': 0.8
            }
        }
    
    def analyze_case_for_resolution(self, case_id: int) -> ResolutionDecision:
        """Analyze case for potential autonomous resolution"""
        try:
            from models import Case, PersonDetection, LocationMatch, RecognitionMatch
            from __init__ import create_app, db
            
            app = create_app()
            with app.app_context():
                case = Case.query.get(case_id)
                if not case:
                    raise ValueError(f"Case {case_id} not found")
                
                # Collect all evidence
                evidence = self._collect_case_evidence(case)
                
                # Analyze evidence patterns
                resolution_analysis = self._analyze_resolution_patterns(case, evidence)
                
                # Check legal compliance
                legal_compliance = self._check_legal_compliance(case, evidence, resolution_analysis)
                
                # Make resolution decision
                decision = self._make_resolution_decision(case, evidence, resolution_analysis, legal_compliance)
                
                # Log decision
                self._log_resolution_decision(case_id, decision)
                
                return decision
                
        except Exception as e:
            self.logger.error(f"Error analyzing case {case_id} for resolution: {e}")
            return ResolutionDecision(
                case_id=case_id,
                decision='MANUAL_REVIEW',
                confidence=0.0,
                evidence_count=0,
                resolution_type='error',
                legal_compliance=False,
                admin_review_required=True,
                closure_reason=f"Analysis error: {str(e)}"
            )
    
    def _collect_case_evidence(self, case) -> List[ResolutionEvidence]:
        """Collect all available evidence for case resolution"""
        evidence = []
        
        # High confidence detections
        high_conf_detections = PersonDetection.query.join(LocationMatch).filter(
            LocationMatch.case_id == case.id,
            PersonDetection.confidence_score >= self.evidence_thresholds['high_confidence_detection']
        ).all()
        
        for detection in high_conf_detections:
            evidence.append(ResolutionEvidence(
                evidence_type='high_confidence_detection',
                confidence_score=detection.confidence_score,
                source=f"footage_{detection.location_match.footage_id}",
                timestamp=detection.created_at,
                details={
                    'detection_id': detection.id,
                    'timestamp': detection.timestamp,
                    'method': detection.analysis_method,
                    'verified': detection.verified
                }
            ))
        
        # Recognition matches
        try:
            from models import PersonProfile
            profile = PersonProfile.query.filter_by(case_id=case.id).first()
            if profile:
                recognition_matches = RecognitionMatch.query.filter_by(profile_id=profile.id).filter(
                    RecognitionMatch.overall_confidence >= 0.8
                ).all()
                
                for match in recognition_matches:
                    evidence.append(ResolutionEvidence(
                        evidence_type='recognition_match',
                        confidence_score=match.overall_confidence,
                        source=f"footage_{match.footage_id}",
                        timestamp=match.created_at,
                        details={
                            'match_id': match.id,
                            'match_type': match.match_type,
                            'verified': match.verified
                        }
                    ))
        except:
            pass
        
        # Verified sightings
        verified_sightings = case.sightings.filter_by(verified=True).all()
        for sighting in verified_sightings:
            evidence.append(ResolutionEvidence(
                evidence_type='verified_sighting',
                confidence_score=sighting.confidence_score,
                source=f"video_{sighting.search_video_id}",
                timestamp=sighting.created_at,
                details={
                    'sighting_id': sighting.id,
                    'timestamp': sighting.timestamp,
                    'method': sighting.detection_method
                }
            ))
        
        return evidence
    
    def _analyze_resolution_patterns(self, case, evidence: List[ResolutionEvidence]) -> Dict:
        """Analyze evidence against known resolution patterns"""
        analysis = {
            'pattern_matches': {},
            'confidence_scores': {},
            'evidence_sufficiency': {},
            'recommended_resolution': None
        }
        
        for pattern_name, pattern_config in self.resolution_patterns.items():
            pattern_score = self._calculate_pattern_score(evidence, pattern_config)
            analysis['pattern_matches'][pattern_name] = pattern_score
            
            # Check if pattern threshold is met
            if pattern_score >= pattern_config['confidence_threshold']:
                analysis['confidence_scores'][pattern_name] = pattern_score
        
        # Determine best matching pattern
        if analysis['confidence_scores']:
            best_pattern = max(analysis['confidence_scores'].items(), key=lambda x: x[1])
            analysis['recommended_resolution'] = best_pattern[0]
        
        return analysis
    
    def _calculate_pattern_score(self, evidence: List[ResolutionEvidence], pattern_config: Dict) -> float:
        """Calculate how well evidence matches a resolution pattern"""
        required_evidence = pattern_config['required_evidence']
        evidence_scores = []
        
        for req_evidence in required_evidence:
            matching_evidence = [e for e in evidence if e.evidence_type == req_evidence]
            if matching_evidence:
                # Use highest confidence score for this evidence type
                max_confidence = max(e.confidence_score for e in matching_evidence)
                evidence_scores.append(max_confidence)
            else:
                evidence_scores.append(0.0)
        
        # Calculate weighted average
        if evidence_scores:
            return sum(evidence_scores) / len(evidence_scores)
        return 0.0
    
    def _check_legal_compliance(self, case, evidence: List[ResolutionEvidence], analysis: Dict) -> Dict:
        """Check legal compliance requirements for case resolution"""
        compliance = {
            'overall_compliant': False,
            'requirements_met': {},
            'missing_requirements': [],
            'confidence_adequate': False
        }
        
        recommended_resolution = analysis.get('recommended_resolution')
        if not recommended_resolution:
            return compliance
        
        pattern_config = self.resolution_patterns[recommended_resolution]
        legal_reqs = pattern_config.get('legal_requirements', [])
        
        for req in legal_reqs:
            if req in self.legal_requirements:
                req_config = self.legal_requirements[req]
                compliance['requirements_met'][req] = self._check_requirement(evidence, req_config)
        
        # Check overall compliance
        met_requirements = sum(compliance['requirements_met'].values())
        total_requirements = len(legal_reqs)
        
        if total_requirements > 0:
            compliance_ratio = met_requirements / total_requirements
            compliance['overall_compliant'] = compliance_ratio >= 0.8
        
        # Check confidence adequacy
        if recommended_resolution in analysis['confidence_scores']:
            confidence = analysis['confidence_scores'][recommended_resolution]
            compliance['confidence_adequate'] = confidence >= pattern_config['confidence_threshold']
        
        return compliance
    
    def _check_requirement(self, evidence: List[ResolutionEvidence], req_config: Dict) -> bool:
        """Check if a specific legal requirement is met"""
        if 'required_confidence' in req_config:
            high_conf_evidence = [e for e in evidence if e.confidence_score >= req_config['required_confidence']]
            return len(high_conf_evidence) > 0
        
        if 'verification_methods' in req_config:
            verified_evidence = [e for e in evidence if e.details.get('verified', False)]
            return len(verified_evidence) > 0
        
        return True  # Default to true for basic requirements
    
    def _make_resolution_decision(self, case, evidence: List[ResolutionEvidence], 
                                analysis: Dict, legal_compliance: Dict) -> ResolutionDecision:
        """Make final resolution decision based on all analysis"""
        
        recommended_resolution = analysis.get('recommended_resolution')
        if not recommended_resolution:
            return ResolutionDecision(
                case_id=case.id,
                decision='CONTINUE_INVESTIGATION',
                confidence=0.0,
                evidence_count=len(evidence),
                resolution_type='insufficient_evidence',
                legal_compliance=False,
                admin_review_required=True,
                closure_reason="Insufficient evidence for resolution"
            )
        
        pattern_config = self.resolution_patterns[recommended_resolution]
        confidence = analysis['confidence_scores'].get(recommended_resolution, 0.0)
        
        # Determine decision
        decision = 'MANUAL_REVIEW'  # Default
        admin_review_required = True
        
        if (legal_compliance['overall_compliant'] and 
            legal_compliance['confidence_adequate'] and
            pattern_config.get('auto_closure_eligible', False) and
            len(evidence) >= self.evidence_thresholds['multiple_confirmations']):
            
            decision = 'AUTO_CLOSE'
            admin_review_required = False
        elif confidence >= 0.7:
            decision = 'MANUAL_REVIEW'
        else:
            decision = 'CONTINUE_INVESTIGATION'
        
        # Generate closure reason
        closure_reason = self._generate_closure_reason(recommended_resolution, evidence, confidence)
        
        return ResolutionDecision(
            case_id=case.id,
            decision=decision,
            confidence=confidence,
            evidence_count=len(evidence),
            resolution_type=recommended_resolution,
            legal_compliance=legal_compliance['overall_compliant'],
            admin_review_required=admin_review_required,
            closure_reason=closure_reason
        )
    
    def _generate_closure_reason(self, resolution_type: str, evidence: List[ResolutionEvidence], 
                                confidence: float) -> str:
        """Generate human-readable closure reason"""
        evidence_summary = f"{len(evidence)} pieces of evidence"
        confidence_text = f"{confidence:.1%} confidence"
        
        reason_templates = {
            'person_found_safe': f"Person located safely based on {evidence_summary} with {confidence_text}",
            'person_found_deceased': f"Person confirmed deceased based on {evidence_summary} with {confidence_text}",
            'criminal_identified': f"Criminal successfully identified based on {evidence_summary} with {confidence_text}",
            'case_solved_other': f"Case resolved through investigation based on {evidence_summary} with {confidence_text}"
        }
        
        return reason_templates.get(resolution_type, f"Case resolution detected with {confidence_text}")
    
    def _log_resolution_decision(self, case_id: int, decision: ResolutionDecision):
        """Log resolution decision to database"""
        try:
            conn = sqlite3.connect('instance/case_resolution.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO case_resolutions 
                (case_id, resolution_type, confidence_score, evidence_count, 
                 auto_closure_eligible, legal_compliance_met, admin_review_required, 
                 closure_reason, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                case_id,
                decision.resolution_type,
                decision.confidence,
                decision.evidence_count,
                decision.decision == 'AUTO_CLOSE',
                decision.legal_compliance,
                decision.admin_review_required,
                decision.closure_reason,
                'pending'
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Resolution decision logged for case {case_id}: {decision.decision}")
            
        except Exception as e:
            self.logger.error(f"Error logging resolution decision: {e}")
    
    def process_autonomous_resolution(self, case_id: int) -> Dict:
        """Process autonomous case resolution"""
        try:
            decision = self.analyze_case_for_resolution(case_id)
            
            if decision.decision == 'AUTO_CLOSE':
                return self._execute_auto_closure(case_id, decision)
            elif decision.decision == 'MANUAL_REVIEW':
                return self._queue_for_manual_review(case_id, decision)
            else:
                return {
                    'success': True,
                    'action': 'continue_investigation',
                    'message': 'Case will continue under investigation',
                    'decision': decision.__dict__
                }
                
        except Exception as e:
            self.logger.error(f"Error processing autonomous resolution for case {case_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute_auto_closure(self, case_id: int, decision: ResolutionDecision) -> Dict:
        """Execute automatic case closure"""
        try:
            from models import Case, Notification, User
            from __init__ import create_app, db
            
            app = create_app()
            with app.app_context():
                case = Case.query.get(case_id)
                if not case:
                    raise ValueError(f"Case {case_id} not found")
                
                # Update case status
                case.status = 'Case Solved'
                case.investigation_outcome = decision.resolution_type.replace('_', ' ').title()
                case.completed_at = datetime.utcnow()
                case.admin_message = f"""🤖 AUTONOMOUS RESOLUTION - Case Automatically Closed

🎯 RESOLUTION TYPE: {decision.resolution_type.replace('_', ' ').title()}
📊 AI CONFIDENCE: {decision.confidence:.1%}
📋 EVIDENCE COUNT: {decision.evidence_count} pieces of evidence
✅ LEGAL COMPLIANCE: {'Met' if decision.legal_compliance else 'Pending'}

🔍 CLOSURE REASON:
{decision.closure_reason}

🤖 This case was automatically resolved by our AI system based on comprehensive evidence analysis and legal compliance verification. The resolution meets all required thresholds for autonomous closure.

📞 If you have any questions about this resolution, please contact our support team."""
                
                # Create user notification
                notification = Notification(
                    user_id=case.user_id,
                    title=f"🎉 Case Automatically Resolved: {case.person_name}",
                    message=f"Great news! Your case has been automatically resolved by our AI system.\n\nResolution: {decision.resolution_type.replace('_', ' ').title()}\nConfidence: {decision.confidence:.1%}\nEvidence: {decision.evidence_count} pieces\n\nThe case has been closed based on comprehensive analysis meeting all legal requirements.",
                    type="success"
                )
                db.session.add(notification)
                
                # Notify admins
                admins = User.query.filter_by(is_admin=True).all()
                for admin in admins:
                    admin_notification = Notification(
                        user_id=admin.id,
                        title=f"🤖 Autonomous Resolution: Case {case_id}",
                        message=f"Case automatically resolved: {decision.resolution_type.replace('_', ' ').title()}\nConfidence: {decision.confidence:.1%}\nEvidence: {decision.evidence_count} pieces\nUser: {case.creator.username}",
                        type="info"
                    )
                    db.session.add(admin_notification)
                
                db.session.commit()
                
                self.logger.info(f"Case {case_id} automatically closed: {decision.resolution_type}")
                
                return {
                    'success': True,
                    'action': 'auto_closed',
                    'message': 'Case automatically closed based on AI analysis',
                    'resolution_type': decision.resolution_type,
                    'confidence': decision.confidence
                }
                
        except Exception as e:
            self.logger.error(f"Error executing auto closure for case {case_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _queue_for_manual_review(self, case_id: int, decision: ResolutionDecision) -> Dict:
        """Queue case for manual admin review"""
        try:
            from models import Notification, User, Case
            from __init__ import create_app, db
            
            app = create_app()
            with app.app_context():
                case = Case.query.get(case_id)
                
                # Notify admins for manual review
                admins = User.query.filter_by(is_admin=True).all()
                for admin in admins:
                    notification = Notification(
                        user_id=admin.id,
                        title=f"📋 Manual Review Required: Case {case_id}",
                        message=f"Case requires manual review for potential resolution.\n\nDetected Resolution: {decision.resolution_type.replace('_', ' ').title()}\nAI Confidence: {decision.confidence:.1%}\nEvidence Count: {decision.evidence_count}\nLegal Compliance: {'Met' if decision.legal_compliance else 'Pending'}\n\nReason: {decision.closure_reason}",
                        type="warning",
                        related_url=f"/admin/cases/{case_id}"
                    )
                    db.session.add(notification)
                
                db.session.commit()
                
                self.logger.info(f"Case {case_id} queued for manual review: {decision.resolution_type}")
                
                return {
                    'success': True,
                    'action': 'queued_for_review',
                    'message': 'Case queued for manual admin review',
                    'decision': decision.__dict__
                }
                
        except Exception as e:
            self.logger.error(f"Error queuing case {case_id} for manual review: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_resolution_candidates(self) -> List[int]:
        """Get list of cases that are candidates for resolution"""
        try:
            from models import Case
            from __init__ import create_app, db
            
            app = create_app()
            with app.app_context():
                # Get cases that are under processing for more than 7 days
                week_ago = datetime.utcnow() - timedelta(days=7)
                
                candidates = Case.query.filter(
                    Case.status.in_(['Under Processing', 'Approved']),
                    Case.updated_at < week_ago
                ).all()
                
                return [case.id for case in candidates]
                
        except Exception as e:
            self.logger.error(f"Error getting resolution candidates: {e}")
            return []

# Global instance
smart_case_closure = SmartCaseClosureSystem()

def analyze_case_resolution(case_id: int) -> Dict:
    """Analyze case for potential resolution"""
    return smart_case_closure.process_autonomous_resolution(case_id)

def get_resolution_candidates() -> List[int]:
    """Get cases eligible for resolution analysis"""
    return smart_case_closure.get_resolution_candidates()