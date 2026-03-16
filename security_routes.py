"""
Security & Compliance Routes - Flask API endpoints for security features
"""

from flask import Blueprint, request, jsonify, render_template, session
from flask_login import login_required, current_user
from datetime import datetime, timedelta
import json
from security_engine import security_engine, AuditAction, UserRole
from models import db

security_bp = Blueprint('security', __name__, url_prefix='/security')

@security_bp.route('/dashboard')
@login_required
def security_dashboard():
    """Security dashboard"""
    system_info = security_engine.get_system_info()
    security_status = security_engine.get_security_status()
    
    return render_template('security_dashboard.html',
                         system_info=system_info,
                         security_status=security_status)

@security_bp.route('/api/authenticate', methods=['POST'])
def authenticate():
    """User authentication with audit logging"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        result = security_engine.authenticate_user(username, password, ip_address, user_agent)
        
        if result['success']:
            session['security_token'] = result['session_token']
            session['user_id'] = result['user_id']
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/validate-session', methods=['POST'])
def validate_session():
    """Validate session token"""
    try:
        data = request.get_json()
        session_token = data.get('session_token') or session.get('security_token')
        
        if not session_token:
            return jsonify({'valid': False, 'error': 'No session token'})
        
        result = security_engine.validate_session(session_token)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/permissions/<user_id>')
@login_required
def get_user_permissions(user_id):
    """Get user permissions"""
    try:
        permissions = security_engine.rbac.get_user_permissions(user_id)
        return jsonify({
            'success': True,
            'user_id': user_id,
            'permissions': permissions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/assign-role', methods=['POST'])
@login_required
def assign_role():
    """Assign role to user (Admin only)"""
    try:
        # Check admin permission
        user_id = session.get('user_id', 'anonymous')
        if not security_engine.rbac.check_permission(user_id, 'manage_users'):
            return jsonify({
                'success': False,
                'error': 'Insufficient permissions'
            }), 403
        
        data = request.get_json()
        target_user_id = data.get('user_id')
        role_name = data.get('role')
        
        try:
            role = UserRole(role_name)
            security_engine.rbac.assign_role(target_user_id, role)
            
            # Log the action
            security_engine.log_data_access(
                user_id=user_id,
                action=AuditAction.UPDATE_CASE,  # Using closest available action
                resource_id=target_user_id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', ''),
                details={'action': 'role_assignment', 'new_role': role_name}
            )
            
            return jsonify({
                'success': True,
                'message': f'Role {role_name} assigned to {target_user_id}'
            })
            
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid role: {role_name}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/audit-logs')
@login_required
def get_audit_logs():
    """Get audit logs (Admin only)"""
    try:
        user_id = session.get('user_id', 'anonymous')
        if not security_engine.rbac.check_permission(user_id, 'view_audit'):
            return jsonify({
                'success': False,
                'error': 'Insufficient permissions'
            }), 403
        
        # Get query parameters
        target_user_id = request.args.get('user_id')
        action = request.args.get('action')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', 100))
        
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        action_enum = AuditAction(action) if action else None
        
        logs = security_engine.audit.get_audit_logs(
            user_id=target_user_id,
            action=action_enum,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'logs': logs,
            'total': len(logs)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/encrypt-data', methods=['POST'])
@login_required
def encrypt_data():
    """Encrypt sensitive data"""
    try:
        data = request.get_json()
        sensitive_data = data.get('data', {})
        
        encrypted_data = security_engine.encrypt_sensitive_fields(sensitive_data)
        
        return jsonify({
            'success': True,
            'encrypted_data': encrypted_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/decrypt-data', methods=['POST'])
@login_required
def decrypt_data():
    """Decrypt sensitive data"""
    try:
        user_id = session.get('user_id', 'anonymous')
        if not security_engine.rbac.check_permission(user_id, 'access_pii'):
            return jsonify({
                'success': False,
                'error': 'Insufficient permissions to access PII'
            }), 403
        
        data = request.get_json()
        encrypted_data = data.get('data', {})
        
        decrypted_data = security_engine.decrypt_sensitive_fields(encrypted_data)
        
        # Log PII access
        security_engine.log_data_access(
            user_id=user_id,
            action=AuditAction.ACCESS_PII,
            resource_id=data.get('resource_id'),
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', ''),
            details={'fields_accessed': list(encrypted_data.keys())}
        )
        
        return jsonify({
            'success': True,
            'decrypted_data': decrypted_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/gdpr/consent', methods=['POST'])
@login_required
def record_consent():
    """Record GDPR consent"""
    try:
        data = request.get_json()
        subject_id = data.get('subject_id')
        consent_type = data.get('consent_type')
        consent_given = data.get('consent_given', True)
        purpose = data.get('purpose')
        legal_basis = data.get('legal_basis', 'consent')
        
        security_engine.gdpr.record_consent(
            subject_id=subject_id,
            consent_type=consent_type,
            consent_given=consent_given,
            purpose=purpose,
            legal_basis=legal_basis
        )
        
        return jsonify({
            'success': True,
            'message': 'Consent recorded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/gdpr/check-consent/<subject_id>/<consent_type>')
@login_required
def check_consent(subject_id, consent_type):
    """Check GDPR consent status"""
    try:
        has_consent = security_engine.gdpr.check_consent(subject_id, consent_type)
        
        return jsonify({
            'success': True,
            'subject_id': subject_id,
            'consent_type': consent_type,
            'has_consent': has_consent
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/gdpr/export-data/<subject_id>')
@login_required
def export_user_data(subject_id):
    """Export user data (GDPR Article 20)"""
    try:
        user_id = session.get('user_id', 'anonymous')
        if not security_engine.rbac.check_permission(user_id, 'export_data'):
            return jsonify({
                'success': False,
                'error': 'Insufficient permissions'
            }), 403
        
        exported_data = security_engine.gdpr.export_user_data(subject_id)
        
        # Log data export
        security_engine.log_data_access(
            user_id=user_id,
            action=AuditAction.EXPORT_DATA,
            resource_id=subject_id,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', ''),
            details={'export_type': 'gdpr_article_20'}
        )
        
        return jsonify({
            'success': True,
            'exported_data': exported_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/gdpr/delete-data/<subject_id>', methods=['DELETE'])
@login_required
def delete_user_data(subject_id):
    """Delete user data (Right to be forgotten)"""
    try:
        user_id = session.get('user_id', 'anonymous')
        if not security_engine.rbac.check_permission(user_id, 'delete_case'):
            return jsonify({
                'success': False,
                'error': 'Insufficient permissions'
            }), 403
        
        success = security_engine.gdpr.delete_user_data(subject_id)
        
        # Log data deletion
        security_engine.log_data_access(
            user_id=user_id,
            action=AuditAction.DELETE_CASE,
            resource_id=subject_id,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', ''),
            details={'deletion_type': 'gdpr_right_to_be_forgotten'}
        )
        
        return jsonify({
            'success': success,
            'message': 'Data deletion completed' if success else 'Data deletion failed'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/anonymize-data', methods=['POST'])
@login_required
def anonymize_data():
    """Anonymize personal data"""
    try:
        data = request.get_json()
        personal_data = data.get('data', {})
        
        anonymized_data = security_engine.gdpr.anonymize_data(personal_data)
        
        return jsonify({
            'success': True,
            'anonymized_data': anonymized_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/status')
@login_required
def security_status():
    """Get security system status"""
    try:
        status = security_engine.get_security_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@security_bp.route('/api/system-info')
@login_required
def system_info():
    """Get security system information"""
    try:
        info = security_engine.get_system_info()
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500