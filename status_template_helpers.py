"""
Status Template Helpers
Template functions for displaying status information consistently across all templates
"""

from comprehensive_status_system import (
    get_comprehensive_status_info, 
    get_status_statistics,
    ALL_CASE_STATUSES,
    PUBLIC_VISIBLE_STATUSES,
    ACTIVE_STATUSES,
    FINAL_STATUSES
)

def get_status_badge_html(status, is_admin=False, size='normal', show_icon=True):
    """Generate HTML badge for status display"""
    info = get_comprehensive_status_info(status)
    display_text = info['admin_friendly'] if is_admin else info['user_friendly']
    
    size_class = {
        'small': 'badge-sm',
        'normal': '',
        'large': 'badge-lg'
    }.get(size, '')
    
    icon_html = f'<i class="{info["icon"]} me-1"></i>' if show_icon and 'icon' in info else info['emoji'] + ' '
    
    return f'<span class="badge {info["bg_color"]} {size_class}" title="{info["description"]}" data-bs-toggle="tooltip">{icon_html}{display_text}</span>'

def get_status_card_html(status, case_count=0, is_admin=False):
    """Generate status card for dashboard display"""
    info = get_comprehensive_status_info(status)
    display_text = info['admin_friendly'] if is_admin else info['user_friendly']
    
    card_html = f'''
    <div class="col-md-6 col-lg-4 mb-3">
        <div class="card border-{info["color"]} h-100">
            <div class="card-body text-center">
                <div class="mb-3">
                    <i class="{info["icon"]} fa-3x {info["text_color"]}"></i>
                </div>
                <h5 class="card-title">{display_text}</h5>
                <h2 class="display-4 {info["text_color"]}">{case_count}</h2>
                <p class="card-text text-muted">{info["description"]}</p>
                <small class="text-muted">
                    <i class="fas fa-clock me-1"></i>
                    {info["estimated_time"]}
                </small>
            </div>
        </div>
    </div>
    '''
    return card_html

def get_status_progress_html(cases, show_percentages=True):
    """Generate progress bar showing status distribution"""
    stats = get_status_statistics(cases)
    total = stats['_summary']['total_cases']
    
    if total == 0:
        return '<div class="alert alert-info">No cases found</div>'
    
    progress_html = '<div class="progress mb-3" style="height: 25px;">'
    
    for status in ALL_CASE_STATUSES:
        if stats[status]['count'] > 0:
            info = get_comprehensive_status_info(status)
            percentage = stats[status]['percentage']
            
            progress_html += f'''
            <div class="progress-bar {info["bg_color"]}" 
                 role="progressbar" 
                 style="width: {percentage}%"
                 title="{info["user_friendly"]}: {stats[status]["count"]} cases ({percentage}%)"
                 data-bs-toggle="tooltip">
                {f"{percentage}%" if show_percentages and percentage > 10 else ""}
            </div>
            '''
    
    progress_html += '</div>'
    
    # Add legend
    progress_html += '<div class="row g-2 mb-3">'
    for status in ALL_CASE_STATUSES:
        if stats[status]['count'] > 0:
            info = get_comprehensive_status_info(status)
            progress_html += f'''
            <div class="col-auto">
                <small class="d-flex align-items-center">
                    <span class="badge {info["bg_color"]} me-1">{info["emoji"]}</span>
                    {info["short_name"]}: {stats[status]["count"]}
                </small>
            </div>
            '''
    progress_html += '</div>'
    
    return progress_html

def get_status_filter_buttons(current_filter='all', base_url='/cases'):
    """Generate filter buttons for status filtering"""
    filters = [
        ('all', 'All Cases', 'fas fa-list', 'secondary'),
        ('active', 'Active', 'fas fa-play-circle', 'primary'),
        ('pending', 'Pending', 'fas fa-clock', 'warning'),
        ('final', 'Completed', 'fas fa-check-circle', 'success'),
        ('public', 'Public', 'fas fa-eye', 'info')
    ]
    
    buttons_html = '<div class="btn-group mb-3" role="group" aria-label="Status filters">'
    
    for filter_key, label, icon, color in filters:
        active_class = 'active' if current_filter == filter_key else ''
        buttons_html += f'''
        <a href="{base_url}?filter={filter_key}" 
           class="btn btn-outline-{color} {active_class}">
            <i class="{icon} me-1"></i>
            {label}
        </a>
        '''
    
    buttons_html += '</div>'
    return buttons_html

def get_case_status_timeline(case):
    """Generate timeline showing case status progression"""
    # This would need case history tracking - placeholder for now
    info = get_comprehensive_status_info(case.status)
    
    timeline_html = f'''
    <div class="timeline-item">
        <div class="timeline-marker {info["bg_color"]}">
            <i class="{info["icon"]}"></i>
        </div>
        <div class="timeline-content">
            <h6>{info["user_friendly"]}</h6>
            <p class="text-muted mb-1">{info["description"]}</p>
            <small class="text-muted">
                Updated: {case.updated_at.strftime('%d %b %Y at %I:%M %p') if case.updated_at else 'Unknown'}
            </small>
        </div>
    </div>
    '''
    
    return timeline_html

def get_status_summary_stats(cases):
    """Get summary statistics for status display"""
    stats = get_status_statistics(cases)
    
    return {
        'total_cases': stats['_summary']['total_cases'],
        'active_cases': stats['_summary']['active_cases'],
        'final_cases': stats['_summary']['final_cases'],
        'public_cases': stats['_summary']['public_cases'],
        'pending_cases': stats['_summary']['pending_cases'],
        'completion_rate': round((stats['_summary']['final_cases'] / stats['_summary']['total_cases'] * 100), 1) if stats['_summary']['total_cases'] > 0 else 0,
        'active_rate': round((stats['_summary']['active_cases'] / stats['_summary']['total_cases'] * 100), 1) if stats['_summary']['total_cases'] > 0 else 0
    }

# Template filter functions
def status_badge_filter(status, is_admin=False):
    """Template filter for status badges"""
    return get_status_badge_html(status, is_admin)

def status_icon_filter(status):
    """Template filter for status icons"""
    info = get_comprehensive_status_info(status)
    return f'<i class="{info["icon"]} {info["text_color"]}"></i>'

def status_emoji_filter(status):
    """Template filter for status emojis"""
    info = get_comprehensive_status_info(status)
    return info['emoji']

def status_color_filter(status):
    """Template filter for status colors"""
    info = get_comprehensive_status_info(status)
    return info['color']