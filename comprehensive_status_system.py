"""
Comprehensive Status System for Missing Person Investigation
Defines all case statuses and provides helper functions
"""

# All possible case statuses
ALL_CASE_STATUSES = [
    'Pending Approval',
    'Approved', 
    'Under Processing',
    'Case Solved',
    'Case Over',
    'Rejected',
    'Withdrawn'
]

# Statuses visible to public
PUBLIC_VISIBLE_STATUSES = [
    'Approved',
    'Under Processing', 
    'Case Solved'
]

# Active statuses (cases being worked on)
ACTIVE_STATUSES = [
    'Approved',
    'Under Processing'
]

# Final statuses (cannot be changed)
FINAL_STATUSES = [
    'Case Solved',
    'Case Over',
    'Withdrawn'
]

def get_dashboard_status_counts(cases, user_type='user'):
    """Get status counts for dashboard display"""
    status_counts = {
        'total': len(cases),
        'active': 0,
        'pending_approval': 0,
        'approved': 0,
        'rejected': 0,
        'under_processing': 0,
        'completed': 0,
        'withdrawn': 0
    }
    
    for case in cases:
        status = case.status
        if status == 'Pending Approval':
            status_counts['pending_approval'] += 1
        elif status == 'Approved':
            status_counts['approved'] += 1
            status_counts['active'] += 1
        elif status == 'Under Processing':
            status_counts['under_processing'] += 1
            status_counts['active'] += 1
        elif status == 'Rejected':
            status_counts['rejected'] += 1
        elif status in ['Case Solved', 'Case Over']:
            status_counts['completed'] += 1
        elif status == 'Withdrawn':
            status_counts['withdrawn'] += 1
    
    return status_counts

def get_comprehensive_status_info(status):
    """Get comprehensive status information"""
    status_info = {
        'Pending Approval': {
            'user_friendly': 'Pending Approval',
            'admin_friendly': 'Pending Approval',
            'short_name': 'Pending',
            'description': 'Case is waiting for admin approval',
            'color': 'warning',
            'bg_color': 'bg-warning',
            'text_color': 'text-warning',
            'icon': 'fas fa-clock',
            'emoji': '⏳',
            'estimated_time': '24-48 hours'
        },
        'Approved': {
            'user_friendly': 'Approved',
            'admin_friendly': 'Approved',
            'short_name': 'Approved',
            'description': 'Case has been approved and is ready for processing',
            'color': 'info',
            'bg_color': 'bg-info',
            'text_color': 'text-info',
            'icon': 'fas fa-check-circle',
            'emoji': '✅',
            'estimated_time': 'Ready for processing'
        },
        'Rejected': {
            'user_friendly': 'Rejected',
            'admin_friendly': 'Rejected',
            'short_name': 'Rejected',
            'description': 'Case has been rejected and needs improvements',
            'color': 'danger',
            'bg_color': 'bg-danger',
            'text_color': 'text-danger',
            'icon': 'fas fa-times-circle',
            'emoji': '❌',
            'estimated_time': 'Needs revision'
        },
        'Under Processing': {
            'user_friendly': 'Under Investigation',
            'admin_friendly': 'Under Processing',
            'short_name': 'Processing',
            'description': 'Case is being actively investigated',
            'color': 'primary',
            'bg_color': 'bg-primary',
            'text_color': 'text-primary',
            'icon': 'fas fa-spinner fa-spin',
            'emoji': '🔄',
            'estimated_time': 'In progress'
        },
        'Case Solved': {
            'user_friendly': 'Investigation Complete',
            'admin_friendly': 'Case Solved',
            'short_name': 'Solved',
            'description': 'Case has been successfully resolved',
            'color': 'success',
            'bg_color': 'bg-success',
            'text_color': 'text-success',
            'icon': 'fas fa-trophy',
            'emoji': '🎉',
            'estimated_time': 'Completed'
        },
        'Case Over': {
            'user_friendly': 'Case Closed',
            'admin_friendly': 'Case Over',
            'short_name': 'Closed',
            'description': 'Case has been closed',
            'color': 'secondary',
            'bg_color': 'bg-secondary',
            'text_color': 'text-secondary',
            'icon': 'fas fa-lock',
            'emoji': '🔒',
            'estimated_time': 'Final'
        },
        'Withdrawn': {
            'user_friendly': 'Withdrawn',
            'admin_friendly': 'Withdrawn',
            'short_name': 'Withdrawn',
            'description': 'Case has been withdrawn by user',
            'color': 'dark',
            'bg_color': 'bg-dark',
            'text_color': 'text-dark',
            'icon': 'fas fa-ban',
            'emoji': '🚫',
            'estimated_time': 'User action'
        }
    }
    
    return status_info.get(status, {
        'user_friendly': status,
        'admin_friendly': status,
        'short_name': status,
        'description': 'Unknown status',
        'color': 'secondary',
        'bg_color': 'bg-secondary',
        'text_color': 'text-secondary',
        'icon': 'fas fa-question-circle',
        'emoji': '❓',
        'estimated_time': 'Unknown'
    })

def get_status_statistics(cases):
    """Get detailed status statistics"""
    stats = {}
    total_cases = len(cases)
    
    # Initialize stats for all statuses
    for status in ALL_CASE_STATUSES:
        stats[status] = {
            'count': 0,
            'percentage': 0.0
        }
    
    # Count cases by status
    for case in cases:
        if case.status in stats:
            stats[case.status]['count'] += 1
    
    # Calculate percentages
    if total_cases > 0:
        for status in stats:
            stats[status]['percentage'] = round((stats[status]['count'] / total_cases) * 100, 1)
    
    # Summary statistics
    active_count = sum(stats[status]['count'] for status in ACTIVE_STATUSES)
    final_count = sum(stats[status]['count'] for status in FINAL_STATUSES)
    public_count = sum(stats[status]['count'] for status in PUBLIC_VISIBLE_STATUSES)
    pending_count = stats.get('Pending Approval', {}).get('count', 0)
    
    stats['_summary'] = {
        'total_cases': total_cases,
        'active_cases': active_count,
        'final_cases': final_count,
        'public_cases': public_count,
        'pending_cases': pending_count
    }
    
    return stats