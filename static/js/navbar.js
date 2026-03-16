// Navbar functionality
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileToggle = document.getElementById('mobileMenuToggle');
    const navbarMenu = document.getElementById('navbarMenu');
    
    if (mobileToggle && navbarMenu) {
        mobileToggle.addEventListener('click', function() {
            navbarMenu.classList.toggle('active');
            this.classList.toggle('active');
        });
    }
    
    // User menu toggle
    window.toggleUserMenu = function() {
        const dropdown = document.getElementById('userMenuDropdown');
        if (dropdown) {
            dropdown.classList.toggle('active');
        }
    }
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', function(e) {
        const userMenu = document.querySelector('.user-menu');
        const dropdown = document.getElementById('userMenuDropdown');
        
        if (userMenu && dropdown && !userMenu.contains(e.target)) {
            dropdown.classList.remove('active');
        }
        
        if (navbarMenu && mobileToggle && 
            !navbarMenu.contains(e.target) && 
            !mobileToggle.contains(e.target)) {
            navbarMenu.classList.remove('active');
            mobileToggle.classList.remove('active');
        }
    });
});