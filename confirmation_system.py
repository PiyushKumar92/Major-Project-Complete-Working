"""
Manual Confirmation System for Uncertain Detections
Handles cases where AI needs human verification
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from models import PersonDetection, LocationMatch, Case, User
from __init__ import db
import os
import json

confirmation_bp = Blueprint('confirmation', __name__)

@confirmation_bp.route('/confirmation-queue')
@login_required
def confirmation_queue():
    """Show pending confirmations to admin"""
    if not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('main.index'))
    
    # Get all detections needing confirmation
    pending_confirmations = PersonDetection.query.filter(
        PersonDetection.confidence_score.between(0.65, 0.84),
        PersonDetection.verified == False
    ).join(LocationMatch).join(Case).all()
    
    return render_template('admin/confirmation_queue.html', 
                         confirmations=pending_confirmations)

@confirmation_bp.route('/confirm-detection/<int:detection_id>', methods=['POST'])
@login_required
def confirm_detection(detection_id):
    """Confirm or reject a detection"""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    detection = PersonDetection.query.get_or_404(detection_id)
    action = request.json.get('action')  # 'confirm' or 'reject'
    notes = request.json.get('notes', '')
    
    if action == 'confirm':
        detection.verified = True
        detection.confidence_score = min(detection.confidence_score + 0.1, 1.0)
        detection.notes = f"Manually confirmed by admin. {notes}"
        
        # Update case status
        location_match = detection.location_match
        location_match.person_found = True
        
        # Notify case owner
        from models import Notification
        notification = Notification(
            user_id=location_match.case.user_id,
            sender_id=current_user.id,
            title=f"Person Confirmed in CCTV: {location_match.case.person_name}",
            message=f"Admin has confirmed person detection at {detection.formatted_timestamp}",
            type="success"
        )
        db.session.add(notification)
        
    elif action == 'reject':
        detection.verified = False
        detection.confidence_score = max(detection.confidence_score - 0.2, 0.0)
        detection.notes = f"Manually rejected by admin. {notes}"
    
    db.session.commit()
    
    return jsonify({'success': True, 'action': action})

@confirmation_bp.route('/batch-confirm', methods=['POST'])
@login_required
def batch_confirm():
    """Batch confirm/reject multiple detections"""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    detection_ids = request.json.get('detection_ids', [])
    action = request.json.get('action')
    
    for detection_id in detection_ids:
        detection = PersonDetection.query.get(detection_id)
        if detection:
            if action == 'confirm':
                detection.verified = True
                detection.confidence_score = min(detection.confidence_score + 0.1, 1.0)
            elif action == 'reject':
                detection.verified = False
                detection.confidence_score = max(detection.confidence_score - 0.2, 0.0)
    
    db.session.commit()
    
    return jsonify({'success': True, 'processed': len(detection_ids)})


class SmartConfirmationSystem:
    """Smart system to minimize false positives and confirmations needed"""
    
    def __init__(self):
        self.confidence_thresholds = {
            'auto_approve': 0.85,
            'needs_confirmation': 0.65,
            'auto_reject': 0.45
        }
    
    def analyze_detection_quality(self, detection_data):
        """Analyze detection quality to determine if confirmation is needed"""
        
        quality_factors = {
            'face_size': self._assess_face_size(detection_data.get('face_size')),
            'face_angle': self._assess_face_angle(detection_data.get('bbox')),
            'lighting': self._assess_lighting(detection_data.get('face_region')),
            'occlusion': self._assess_occlusion(detection_data.get('face_region')),
            'blur': self._assess_blur(detection_data.get('face_region'))
        }
        
        # Calculate overall quality score
        quality_score = sum(quality_factors.values()) / len(quality_factors)
        
        # Adjust confidence based on quality
        adjusted_confidence = detection_data['confidence'] * quality_score
        
        return {
            'adjusted_confidence': adjusted_confidence,
            'quality_score': quality_score,
            'quality_factors': quality_factors,
            'needs_confirmation': self._needs_confirmation(adjusted_confidence, quality_factors)
        }
    
    def _assess_face_size(self, face_size):
        """Assess if face size is optimal for recognition"""
        if not face_size:
            return 0.5
        
        width, height = face_size
        area = width * height
        
        # Optimal face size range
        if 50 <= width <= 200 and 50 <= height <= 200:
            return 1.0
        elif 30 <= width <= 300 and 30 <= height <= 300:
            return 0.8
        else:
            return 0.4
    
    def _assess_face_angle(self, bbox):
        """Assess face angle (frontal vs profile)"""
        if not bbox:
            return 0.5
        
        # Simple heuristic based on face width/height ratio
        x, y, w, h = bbox
        ratio = w / h if h > 0 else 1
        
        # Frontal faces have ratio closer to 1
        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif 0.6 <= ratio <= 1.4:
            return 0.7
        else:
            return 0.4
    
    def _assess_lighting(self, face_region):
        """Assess lighting conditions"""
        if face_region is None:
            return 0.5
        
        try:
            import cv2
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray.mean()
            
            # Optimal brightness range
            if 80 <= mean_brightness <= 180:
                return 1.0
            elif 50 <= mean_brightness <= 220:
                return 0.7
            else:
                return 0.3
        except:
            return 0.5
    
    def _assess_occlusion(self, face_region):
        """Assess if face is occluded"""
        # Simplified occlusion detection
        # In real implementation, use more sophisticated methods
        return 0.8  # Assume minimal occlusion for now
    
    def _assess_blur(self, face_region):
        """Assess image blur"""
        if face_region is None:
            return 0.5
        
        try:
            import cv2
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Higher variance means less blur
            if laplacian_var > 500:
                return 1.0
            elif laplacian_var > 100:
                return 0.7
            else:
                return 0.3
        except:
            return 0.5
    
    def _needs_confirmation(self, adjusted_confidence, quality_factors):
        """Determine if manual confirmation is needed"""
        
        # Auto approve high confidence with good quality
        if (adjusted_confidence >= self.confidence_thresholds['auto_approve'] and 
            quality_factors['face_size'] >= 0.8 and 
            quality_factors['lighting'] >= 0.7):
            return False
        
        # Auto reject very low confidence
        if adjusted_confidence < self.confidence_thresholds['auto_reject']:
            return False
        
        # Need confirmation for medium confidence
        if (self.confidence_thresholds['auto_reject'] <= adjusted_confidence < 
            self.confidence_thresholds['auto_approve']):
            return True
        
        # Need confirmation for high confidence but poor quality
        if (adjusted_confidence >= self.confidence_thresholds['auto_approve'] and
            (quality_factors['face_size'] < 0.6 or quality_factors['lighting'] < 0.5)):
            return True
        
        return False


# Global confirmation system instance
smart_confirmation = SmartConfirmationSystem()