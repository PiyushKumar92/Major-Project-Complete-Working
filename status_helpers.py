"""
Status Display Helpers
Centralized status management and display functions
"""

def get_status_display_info(status):
    """Get comprehensive status display information - uses comprehensive_status_system"""
    from comprehensive_status_system import get_comprehensive_status_info
    return get_comprehensive_status_info(status)

def get_legacy_status_display_info(status):
    """Get comprehensive status display information"""
    status_info = {
        'Pending Approval': {
            'emoji': '[PENDING]',
            'color': 'warning',
            'user_friendly': 'Pending Approval',
            'admin_friendly': 'Pending Approval',
            'description': 'Awaiting admin review and approval',
            'user_message': 'Your case is currently under review by our admin team.',
            'admin_message': 'This case requires admin approval before processing can begin.',
            'can_edit': True,
            'is_final': False,
            'next_actions': ['Approve', 'Reject']
        },
        'Approved': {
            'emoji': '[APPROVED]',
            'color': 'info',
            'user_friendly': 'Approved',
            'admin_friendly': 'Approved',
            'description': 'Case approved, ready for AI processing',
            'user_message': 'Your case has been approved and AI analysis will begin shortly.',
            'admin_message': 'Case approved and ready for AI processing.',
            'can_edit': True,
            'is_final': False,
            'next_actions': ['Start Processing', 'Reject']
        },
        'Rejected': {
            'emoji': '[REJECTED]',
            'color': 'danger',
            'user_friendly': 'Rejected',
            'admin_friendly': 'Rejected',
            'description': 'Case rejected, needs revision',
            'user_message': 'Your case needs revision before it can be processed.',
            'admin_message': 'Case rejected and requires user revision.',
            'can_edit': True,
            'is_final': False,
            'next_actions': ['Approve after revision']
        },
        'Under Processing': {
            'emoji': '[PROCESSING]',
            'color': 'primary',
            'user_friendly': 'Under Investigation',
            'admin_friendly': 'Under Processing',
            'description': 'AI analysis and investigation in progress',
            'user_message': 'Your case is under active investigation using AI analysis.',
            'admin_message': 'AI processing and investigation is currently running.',
            'can_edit': False,
            'is_final': False,
            'next_actions': ['Mark as Solved', 'Mark as Over']
        },
        'Case Solved': {
            'emoji': '[SOLVED]',
            'color': 'success',
            'user_friendly': 'Investigation Complete',
            'admin_friendly': 'Case Solved',
            'description': 'Investigation completed successfully',
            'user_message': 'Great news! Your case has been successfully resolved.',
            'admin_message': 'Case has been successfully solved and completed.',
            'can_edit': False,
            'is_final': True,
            'next_actions': []
        },
        'Case Over': {
            'emoji': '[CLOSED]',
            'color': 'secondary',
            'user_friendly': 'Case Closed',
            'admin_friendly': 'Case Over',
            'description': 'Investigation closed',
            'user_message': 'Your case investigation has been completed and closed.',
            'admin_message': 'Case has been closed and investigation is complete.',
            'can_edit': False,
            'is_final': True,
            'next_actions': []
        },
        'Withdrawn': {
            'emoji': '[WITHDRAWN]',
            'color': 'dark',
            'user_friendly': 'Withdrawn',
            'admin_friendly': 'Withdrawn',
            'description': 'Case withdrawn by user',
            'user_message': 'This case has been withdrawn.',
            'admin_message': 'Case has been withdrawn by the user.',
            'can_edit': False,
            'is_final': True,
            'next_actions': []
        }
    }
    
    return status_info.get(status, {
        'emoji': '[UNKNOWN]',
        'color': 'secondary',
        'user_friendly': status,
        'admin_friendly': status,
        'description': 'Unknown status',
        'user_message': f'Case status: {status}',
        'admin_message': f'Case status: {status}',
        'can_edit': False,
        'is_final': False,
        'next_actions': []
    })

def get_status_badge_html(status, is_admin=False):
    """Get HTML for status badge"""
    info = get_status_display_info(status)
    display_text = info['admin_friendly'] if is_admin else info['user_friendly']
    
    return f'<span class="badge bg-{info["color"]}" title="{info["description"]}">{info["emoji"]} {display_text}</span>'

def get_status_alert_html(status, admin_message=None):
    """Get HTML for status alert/notification with comprehensive information"""
    info = get_status_display_info(status)
    
    # Status-specific icons and styling
    status_icons = {
        'Pending Approval': 'fas fa-clock',
        'Approved': 'fas fa-check-circle', 
        'Rejected': 'fas fa-times-circle',
        'Under Processing': 'fas fa-cog fa-spin',
        'Case Solved': 'fas fa-trophy',
        'Case Over': 'fas fa-lock',
        'Withdrawn': 'fas fa-archive'
    }
    
    icon = status_icons.get(status, 'fas fa-info-circle')
    
    alert_html = f'''
    <div class="alert alert-{info["color"]} py-3 px-4 border-0 shadow-sm" style="border-radius: 10px;">
        <div class="d-flex align-items-start">
            <div class="me-3 mt-1">
                <i class="{icon} fa-lg"></i>
            </div>
            <div class="flex-grow-1">
                <h6 class="alert-heading mb-2">{info["emoji"]} {info["user_friendly"]}</h6>
                <p class="mb-2">{info["user_message"]}</p>
    '''
    
    # Add status-specific action buttons
    if status == 'Rejected' and admin_message:
        alert_html += f'''
                <div class="mt-3 p-3 bg-light rounded">
                    <small class="fw-bold text-dark"><i class="fas fa-comment-dots me-1"></i>Admin Feedback:</small>
                    <div class="mt-2 text-dark">{admin_message}</div>
                </div>
                <div class="mt-3">
                    <a href="#" onclick="window.location.href=window.location.pathname.replace('/case_details/', '/edit_case/')" class="btn btn-warning btn-sm">
                        <i class="fas fa-edit me-1"></i>Edit & Resubmit Case
                    </a>
                </div>
        '''
    elif status == 'Withdrawn':
        alert_html += f'''
                {f'<div class="mt-3 p-3 bg-light rounded"><small class="fw-bold text-dark"><i class="fas fa-info-circle me-1"></i>Withdrawal Info:</small><div class="mt-2 text-dark">{admin_message}</div></div>' if admin_message else ''}
                <div class="mt-3">
                    <a href="#" onclick="window.location.href=window.location.pathname.replace('/case_details/', '/edit_case/')" class="btn btn-primary btn-sm">
                        <i class="fas fa-redo me-1"></i>Resubmit Case
                    </a>
                </div>
        '''
    elif admin_message and status not in ['Pending Approval']:
        alert_html += f'''
                <div class="mt-3 p-3 bg-light rounded">
                    <small class="fw-bold text-dark"><i class="fas fa-comment-dots me-1"></i>Admin Update:</small>
                    <div class="mt-2 text-dark">{admin_message}</div>
                </div>
        '''
    
    # Add estimated time information
    if 'estimated_time' in info and info['estimated_time'] != 'Unknown':
        alert_html += f'''
                <div class="mt-2">
                    <small class="text-muted"><i class="fas fa-clock me-1"></i>Estimated Time: {info["estimated_time"]}</small>
                </div>
        '''
    
    alert_html += '''
            </div>
        </div>
    </div>
    '''
    
    return alert_html

def get_all_status_choices(admin_only=False):
    """Get all available status choices for forms"""
    if admin_only:
        return [
            ('Pending Approval', '⏳ Pending Approval'),
            ('Approved', '✅ Approved'),
            ('Rejected', '❌ Rejected'),
            ('Under Processing', '🔄 Under Processing'),
            ('Case Solved', '🎉 Case Solved'),
            ('Case Over', '🔒 Case Over'),
            ('Withdrawn', '🚫 Withdrawn')
        ]
    else:
        # User can only see non-admin statuses
        return [
            ('Pending Approval', '⏳ Pending Approval'),
            ('Approved', '✅ Approved'),
            ('Rejected', '❌ Rejected'),
            ('Under Processing', '🔄 Under Investigation')
        ]

def get_admin_only_statuses():
    """Get list of admin-only statuses"""
    return ['Case Solved', 'Case Over']

def get_final_statuses():
    """Get list of final statuses that cannot be edited"""
    return ['Case Solved', 'Case Over', 'Withdrawn']

def can_user_edit_case(status):
    """Check if user can edit case based on status"""
    return status not in get_final_statuses()

def get_status_workflow():
    """Get the complete status workflow"""
    return {
        'user_flow': [
            'Pending Approval',
            'Approved', 
            'Under Processing',
            'Case Solved/Case Over'  # Admin decides
        ],
        'admin_transitions': {
            'Pending Approval': ['Approved', 'Rejected'],
            'Approved': ['Under Processing', 'Rejected'],
            'Rejected': ['Approved'],  # After user revision
            'Under Processing': ['Case Solved', 'Case Over'],
            'Case Solved': [],  # Final
            'Case Over': [],   # Final
            'Withdrawn': []    # Final
        }
    }