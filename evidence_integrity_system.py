"""
Evidence Integrity System
Provides cryptographic evidence integrity and legal-grade audit trails
"""

import hashlib
import json
import os
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EvidenceFrame:
    """Cryptographically secured evidence frame"""
    frame_id: str
    detection_id: str
    case_id: int
    footage_id: int
    timestamp: float
    
    # Frame data
    frame_path: str
    frame_hash: str
    frame_size: Tuple[int, int]
    frame_format: str
    
    # Detection data
    bounding_box: Tuple[int, int, int, int]
    confidence_score: float
    detection_method: str
    
    # Integrity data
    created_at: str
    verified_by: Optional[int] = None
    verification_timestamp: Optional[str] = None
    chain_hash: str = ""
    
    # Legal metadata
    evidence_number: str = ""
    legal_status: str = "pending"  # pending, verified, court_ready
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class EvidenceChain:
    """Complete evidence chain for a case"""
    chain_id: str
    case_id: int
    created_at: str
    
    # Chain integrity
    chain_hash: str
    previous_chain_hash: str = ""
    
    # Evidence frames
    frames: List[EvidenceFrame] = None
    
    # Verification data
    verified_by: Optional[int] = None
    verification_timestamp: Optional[str] = None
    legal_officer: Optional[str] = None
    
    def __post_init__(self):
        if self.frames is None:
            self.frames = []
    
    def add_frame(self, frame: EvidenceFrame):
        """Add frame to evidence chain"""
        self.frames.append(frame)
        self._update_chain_hash()
    
    def _update_chain_hash(self):
        """Update chain hash with all frames"""
        chain_data = {
            'chain_id': self.chain_id,
            'case_id': self.case_id,
            'frames': [frame.frame_hash for frame in self.frames],
            'created_at': self.created_at
        }
        
        chain_string = json.dumps(chain_data, sort_keys=True)
        self.chain_hash = hashlib.sha256(chain_string.encode()).hexdigest()
    
    def verify_integrity(self) -> Dict[str, bool]:
        """Verify complete chain integrity"""
        results = {
            'chain_integrity': True,
            'frame_integrity': True,
            'hash_verification': True,
            'file_verification': True
        }
        
        # Verify chain hash
        original_hash = self.chain_hash
        self._update_chain_hash()
        if original_hash != self.chain_hash:
            results['chain_integrity'] = False
            results['hash_verification'] = False
        
        # Verify each frame
        for frame in self.frames:
            if not self._verify_frame_integrity(frame):
                results['frame_integrity'] = False
                results['file_verification'] = False
        
        return results
    
    def _verify_frame_integrity(self, frame: EvidenceFrame) -> bool:
        """Verify individual frame integrity"""
        try:
            if not os.path.exists(frame.frame_path):
                return False
            
            # Recalculate frame hash
            current_hash = self._calculate_frame_hash(frame.frame_path)
            return current_hash == frame.frame_hash
            
        except Exception as e:
            logger.error(f"Error verifying frame integrity: {e}")
            return False
    
    def _calculate_frame_hash(self, frame_path: str) -> str:
        """Calculate SHA-256 hash of frame file"""
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                return ""
            
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            return hashlib.sha256(frame_bytes).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating frame hash: {e}")
            return ""

class EvidenceIntegritySystem:
    """Cryptographic evidence integrity and legal audit system"""
    
    def __init__(self):
        self.evidence_dir = Path("static/evidence")
        self.evidence_dir.mkdir(exist_ok=True)
        
        # Legal evidence numbering
        self.evidence_counter = self._load_evidence_counter()
        
        # Chain storage
        self.chains: Dict[str, EvidenceChain] = {}
        
    def _load_evidence_counter(self) -> int:
        """Load evidence counter from file"""
        counter_file = self.evidence_dir / "evidence_counter.txt"
        try:
            if counter_file.exists():
                return int(counter_file.read_text().strip())
            return 1
        except:
            return 1
    
    def _save_evidence_counter(self):
        """Save evidence counter to file"""
        counter_file = self.evidence_dir / "evidence_counter.txt"
        counter_file.write_text(str(self.evidence_counter))
    
    def create_evidence_frame(self, detection_data: Dict, frame_image: np.ndarray) -> EvidenceFrame:
        """Create cryptographically secured evidence frame"""
        
        # Generate unique frame ID
        frame_id = self._generate_frame_id(detection_data)
        
        # Generate evidence number
        evidence_number = f"EVD-{self.evidence_counter:06d}"
        self.evidence_counter += 1
        self._save_evidence_counter()
        
        # Save frame securely
        frame_path = self._save_evidence_frame(frame_image, frame_id, evidence_number)
        
        # Calculate frame hash
        frame_hash = self._calculate_secure_frame_hash(frame_path)
        
        # Create evidence frame
        evidence_frame = EvidenceFrame(
            frame_id=frame_id,
            detection_id=detection_data.get('detection_id', ''),
            case_id=detection_data.get('case_id', 0),
            footage_id=detection_data.get('footage_id', 0),
            timestamp=detection_data.get('timestamp', 0.0),
            frame_path=frame_path,
            frame_hash=frame_hash,
            frame_size=(frame_image.shape[1], frame_image.shape[0]),
            frame_format="jpg",
            bounding_box=detection_data.get('bbox', (0, 0, 0, 0)),
            confidence_score=detection_data.get('confidence', 0.0),
            detection_method=detection_data.get('method', 'unknown'),
            created_at=datetime.now(timezone.utc).isoformat(),
            evidence_number=evidence_number,
            legal_status="pending"
        )
        
        logger.info(f"Created evidence frame {evidence_number} with hash {frame_hash[:16]}...")
        return evidence_frame
    
    def _generate_frame_id(self, detection_data: Dict) -> str:
        """Generate unique frame ID"""
        timestamp = detection_data.get('timestamp', 0.0)
        case_id = detection_data.get('case_id', 0)
        footage_id = detection_data.get('footage_id', 0)
        
        id_string = f"frame_{case_id}_{footage_id}_{timestamp}_{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def _save_evidence_frame(self, frame_image: np.ndarray, frame_id: str, evidence_number: str) -> str:
        """Save evidence frame with secure naming"""
        
        # Create case-specific directory
        case_dir = self.evidence_dir / f"case_{frame_id[:8]}"
        case_dir.mkdir(exist_ok=True)
        
        # Save with evidence number
        filename = f"{evidence_number}_{frame_id}.jpg"
        frame_path = case_dir / filename
        
        # Save with high quality for legal purposes
        cv2.imwrite(str(frame_path), frame_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return str(frame_path)
    
    def _calculate_secure_frame_hash(self, frame_path: str) -> str:
        """Calculate cryptographically secure frame hash"""
        try:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError("Cannot read frame")
            
            # Convert to bytes with consistent encoding
            frame_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            
            # Calculate SHA-256 hash
            hash_obj = hashlib.sha256()
            hash_obj.update(frame_bytes)
            
            # Add metadata for additional security
            metadata = {
                'file_path': frame_path,
                'file_size': len(frame_bytes),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            hash_obj.update(json.dumps(metadata, sort_keys=True).encode())
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating secure frame hash: {e}")
            raise
    
    def create_evidence_chain(self, case_id: int) -> EvidenceChain:
        """Create new evidence chain for case"""
        
        chain_id = f"chain_{case_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        evidence_chain = EvidenceChain(
            chain_id=chain_id,
            case_id=case_id,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.chains[chain_id] = evidence_chain
        
        logger.info(f"Created evidence chain {chain_id} for case {case_id}")
        return evidence_chain
    
    def add_frame_to_chain(self, chain_id: str, evidence_frame: EvidenceFrame):
        """Add evidence frame to chain"""
        
        if chain_id not in self.chains:
            raise ValueError(f"Evidence chain {chain_id} not found")
        
        chain = self.chains[chain_id]
        chain.add_frame(evidence_frame)
        
        # Save chain to disk
        self._save_evidence_chain(chain)
        
        logger.info(f"Added frame {evidence_frame.evidence_number} to chain {chain_id}")
    
    def _save_evidence_chain(self, chain: EvidenceChain):
        """Save evidence chain to disk"""
        
        chain_file = self.evidence_dir / f"{chain.chain_id}.json"
        
        chain_data = {
            'chain_id': chain.chain_id,
            'case_id': chain.case_id,
            'created_at': chain.created_at,
            'chain_hash': chain.chain_hash,
            'previous_chain_hash': chain.previous_chain_hash,
            'frames': [frame.to_dict() for frame in chain.frames],
            'verified_by': chain.verified_by,
            'verification_timestamp': chain.verification_timestamp,
            'legal_officer': chain.legal_officer
        }\n        \n        with open(chain_file, 'w') as f:\n            json.dump(chain_data, f, indent=2)\n    \n    def load_evidence_chain(self, chain_id: str) -> Optional[EvidenceChain]:\n        \"\"\"Load evidence chain from disk\"\"\"\n        \n        chain_file = self.evidence_dir / f\"{chain_id}.json\"\n        \n        if not chain_file.exists():\n            return None\n        \n        try:\n            with open(chain_file, 'r') as f:\n                chain_data = json.load(f)\n            \n            # Reconstruct evidence chain\n            chain = EvidenceChain(\n                chain_id=chain_data['chain_id'],\n                case_id=chain_data['case_id'],\n                created_at=chain_data['created_at'],\n                chain_hash=chain_data['chain_hash'],\n                previous_chain_hash=chain_data.get('previous_chain_hash', ''),\n                verified_by=chain_data.get('verified_by'),\n                verification_timestamp=chain_data.get('verification_timestamp'),\n                legal_officer=chain_data.get('legal_officer')\n            )\n            \n            # Reconstruct frames\n            for frame_data in chain_data['frames']:\n                frame = EvidenceFrame(**frame_data)\n                chain.frames.append(frame)\n            \n            self.chains[chain_id] = chain\n            return chain\n            \n        except Exception as e:\n            logger.error(f\"Error loading evidence chain {chain_id}: {e}\")\n            return None\n    \n    def verify_evidence_integrity(self, chain_id: str) -> Dict:\n        \"\"\"Verify complete evidence integrity\"\"\"\n        \n        chain = self.chains.get(chain_id) or self.load_evidence_chain(chain_id)\n        \n        if not chain:\n            return {\"error\": \"Evidence chain not found\"}\n        \n        # Verify chain integrity\n        integrity_results = chain.verify_integrity()\n        \n        # Additional verification\n        verification_report = {\n            \"chain_id\": chain_id,\n            \"case_id\": chain.case_id,\n            \"verification_timestamp\": datetime.now(timezone.utc).isoformat(),\n            \"total_frames\": len(chain.frames),\n            \"integrity_results\": integrity_results,\n            \"frame_details\": [],\n            \"overall_status\": \"verified\" if all(integrity_results.values()) else \"compromised\"\n        }\n        \n        # Verify each frame individually\n        for frame in chain.frames:\n            frame_verification = {\n                \"evidence_number\": frame.evidence_number,\n                \"frame_id\": frame.frame_id,\n                \"hash_verified\": chain._verify_frame_integrity(frame),\n                \"file_exists\": os.path.exists(frame.frame_path),\n                \"legal_status\": frame.legal_status\n            }\n            verification_report[\"frame_details\"].append(frame_verification)\n        \n        return verification_report\n    \n    def generate_legal_evidence_report(self, case_id: int) -> Dict:\n        \"\"\"Generate comprehensive legal evidence report\"\"\"\n        \n        # Find all chains for case\n        case_chains = [chain for chain in self.chains.values() if chain.case_id == case_id]\n        \n        # Load chains from disk if not in memory\n        for chain_file in self.evidence_dir.glob(f\"chain_{case_id}_*.json\"):\n            chain_id = chain_file.stem\n            if chain_id not in self.chains:\n                loaded_chain = self.load_evidence_chain(chain_id)\n                if loaded_chain:\n                    case_chains.append(loaded_chain)\n        \n        if not case_chains:\n            return {\"error\": \"No evidence chains found for case\"}\n        \n        # Generate comprehensive report\n        report = {\n            \"report_id\": f\"legal_report_{case_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}\",\n            \"case_id\": case_id,\n            \"generated_at\": datetime.now(timezone.utc).isoformat(),\n            \"total_evidence_chains\": len(case_chains),\n            \"total_evidence_frames\": sum(len(chain.frames) for chain in case_chains),\n            \"evidence_summary\": [],\n            \"integrity_verification\": {},\n            \"legal_compliance\": {\n                \"cryptographic_integrity\": True,\n                \"chain_of_custody\": True,\n                \"evidence_numbering\": True,\n                \"timestamp_verification\": True\n            }\n        }\n        \n        # Process each chain\n        for chain in case_chains:\n            # Verify integrity\n            integrity_results = self.verify_evidence_integrity(chain.chain_id)\n            report[\"integrity_verification\"][chain.chain_id] = integrity_results\n            \n            # Evidence summary\n            chain_summary = {\n                \"chain_id\": chain.chain_id,\n                \"created_at\": chain.created_at,\n                \"total_frames\": len(chain.frames),\n                \"evidence_numbers\": [frame.evidence_number for frame in chain.frames],\n                \"confidence_range\": {\n                    \"min\": min([frame.confidence_score for frame in chain.frames], default=0.0),\n                    \"max\": max([frame.confidence_score for frame in chain.frames], default=0.0),\n                    \"avg\": sum([frame.confidence_score for frame in chain.frames]) / len(chain.frames) if chain.frames else 0.0\n                },\n                \"detection_methods\": list(set([frame.detection_method for frame in chain.frames])),\n                \"legal_status\": chain.legal_officer is not None\n            }\n            report[\"evidence_summary\"].append(chain_summary)\n            \n            # Check legal compliance\n            if not integrity_results.get(\"overall_status\") == \"verified\":\n                report[\"legal_compliance\"][\"cryptographic_integrity\"] = False\n        \n        return report\n    \n    def mark_evidence_court_ready(self, chain_id: str, legal_officer: str, verification_notes: str = \"\") -> bool:\n        \"\"\"Mark evidence chain as court-ready\"\"\"\n        \n        chain = self.chains.get(chain_id) or self.load_evidence_chain(chain_id)\n        \n        if not chain:\n            return False\n        \n        # Verify integrity before marking court-ready\n        integrity_results = chain.verify_integrity()\n        \n        if not all(integrity_results.values()):\n            logger.error(f\"Cannot mark chain {chain_id} as court-ready: integrity verification failed\")\n            return False\n        \n        # Mark as court-ready\n        chain.legal_officer = legal_officer\n        chain.verification_timestamp = datetime.now(timezone.utc).isoformat()\n        \n        # Mark all frames as court-ready\n        for frame in chain.frames:\n            frame.legal_status = \"court_ready\"\n            frame.verification_timestamp = chain.verification_timestamp\n        \n        # Save updated chain\n        self._save_evidence_chain(chain)\n        \n        logger.info(f\"Evidence chain {chain_id} marked as court-ready by {legal_officer}\")\n        return True\n\n# Global evidence integrity system\nevidence_system = EvidenceIntegritySystem()\n\ndef create_evidence_frame(detection_data: Dict, frame_image: np.ndarray) -> EvidenceFrame:\n    \"\"\"Global function to create evidence frame\"\"\"\n    return evidence_system.create_evidence_frame(detection_data, frame_image)\n\ndef create_evidence_chain(case_id: int) -> EvidenceChain:\n    \"\"\"Global function to create evidence chain\"\"\"\n    return evidence_system.create_evidence_chain(case_id)\n\ndef verify_evidence_integrity(chain_id: str) -> Dict:\n    \"\"\"Global function to verify evidence integrity\"\"\"\n    return evidence_system.verify_evidence_integrity(chain_id)\n\ndef generate_legal_evidence_report(case_id: int) -> Dict:\n    \"\"\"Global function to generate legal evidence report\"\"\"\n    return evidence_system.generate_legal_evidence_report(case_id)