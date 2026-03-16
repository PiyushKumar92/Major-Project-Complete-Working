// Enhanced Landing Page Animations
document.addEventListener('DOMContentLoaded', function() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    // Observe all elements that should animate on scroll
    document.querySelectorAll('.scroll-fade, .feature-card, .testimonial-card').forEach(el => {
        observer.observe(el);
    });

    // Animate search items in hero visual
    function animateSearchItems() {
        const searchItems = document.querySelectorAll('.search-item');
        let currentIndex = 0;

        setInterval(() => {
            // Remove active class from all items
            searchItems.forEach(item => item.classList.remove('active'));
            
            // Add active class to current item
            if (searchItems[currentIndex]) {
                searchItems[currentIndex].classList.add('active');
            }
            
            // Move to next item
            currentIndex = (currentIndex + 1) % searchItems.length;
        }, 3000);
    }

    // Start search animation
    animateSearchItems();

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading animation to buttons
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            // Don't prevent default for actual navigation
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });

    // Parallax effect for hero background
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const heroPattern = document.querySelector('.hero-pattern');
        if (heroPattern) {
            heroPattern.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    });

    // Counter animation for metrics
    function animateCounters() {
        const counters = document.querySelectorAll('.metric-number, .stat-number, .mini-number');
        
        counters.forEach(counter => {
            const target = counter.textContent;
            if (!isNaN(target) && target !== '') {
                const increment = target / 100;
                let current = 0;
                
                const timer = setInterval(() => {
                    current += increment;
                    counter.textContent = Math.floor(current);
                    
                    if (current >= target) {
                        counter.textContent = target;
                        clearInterval(timer);
                    }
                }, 20);
            }
        });
    }

    // Trigger counter animation when metrics come into view
    const metricsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounters();
                metricsObserver.unobserve(entry.target);
            }
        });
    });

    const metricsSection = document.querySelector('.impact-metrics');
    if (metricsSection) {
        metricsObserver.observe(metricsSection);
    }
});