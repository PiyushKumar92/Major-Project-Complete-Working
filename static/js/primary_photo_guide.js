/**
 * Primary Photo Selection Guide
 * Helps users understand the importance of selecting the best primary photo
 */

document.addEventListener('DOMContentLoaded', function() {
    // Add primary photo selection guide
    addPrimaryPhotoGuide();
    
    // Monitor photo uploads and provide guidance
    monitorPhotoUploads();
});

function addPrimaryPhotoGuide() {
    const photoSection = document.querySelector('#photos-section') || document.querySelector('.photo-upload-section');
    if (!photoSection) return;
    
    const guideHTML = `
        <div class="alert alert-info primary-photo-guide mb-3">
            <h6><i class="fas fa-star text-warning"></i> Primary Photo Selection Guide</h6>
            <div class="row">
                <div class="col-md-6">
                    <h6 class="text-success">✅ Best Primary Photo:</h6>
                    <ul class="small mb-0">
                        <li>Clear, front-facing photo</li>
                        <li>Good lighting (natural light preferred)</li>
                        <li>High resolution & sharp focus</li>
                        <li>Minimal background distractions</li>
                        <li>Recent photo (within 6 months)</li>
                        <li>Person looking directly at camera</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6 class="text-danger">❌ Avoid as Primary:</h6>
                    <ul class="small mb-0">
                        <li>Blurry or low-quality images</li>
                        <li>Side profile or back view</li>
                        <li>Group photos with multiple people</li>
                        <li>Sunglasses or face coverings</li>
                        <li>Very old photos (different appearance)</li>
                        <li>Poor lighting or shadows</li>
                    </ul>
                </div>
            </div>
            <div class="mt-2">
                <small class="text-muted">
                    <i class="fas fa-info-circle"></i> 
                    <strong>Why Primary Photo Matters:</strong> 
                    AI uses your primary photo as the main reference for face recognition in CCTV footage. 
                    A high-quality primary photo significantly improves detection accuracy.
                </small>
            </div>
        </div>
    `;
    
    photoSection.insertAdjacentHTML('afterbegin', guideHTML);
}

function monitorPhotoUploads() {
    const photoInput = document.querySelector('input[type="file"][accept*="image"]');
    if (!photoInput) return;
    
    photoInput.addEventListener('change', function(e) {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            showPhotoQualityTips(files);
        }
    });
}

function showPhotoQualityTips(files) {
    // Show tips for selecting primary photo when multiple photos are uploaded
    if (files.length > 1) {
        setTimeout(() => {
            const tipHTML = `
                <div class="alert alert-warning photo-selection-tip">
                    <h6><i class="fas fa-lightbulb"></i> Photo Selection Tip</h6>
                    <p class="mb-2">You've uploaded ${files.length} photos. Please select the <strong>best quality photo</strong> as your primary photo:</p>
                    <div class="row">
                        <div class="col-md-8">
                            <ol class="small mb-0">
                                <li>Look for the <strong>clearest face</strong> with good lighting</li>
                                <li>Choose a <strong>recent photo</strong> that looks like the person now</li>
                                <li>Prefer <strong>front-facing</strong> over side angles</li>
                                <li>Avoid photos with <strong>sunglasses or masks</strong></li>
                            </ol>
                        </div>
                        <div class="col-md-4">
                            <div class="text-center">
                                <i class="fas fa-camera fa-3x text-primary mb-2"></i>
                                <div class="small text-muted">Primary photo = Better AI accuracy</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            const photoSection = document.querySelector('.primary-photo-guide');
            if (photoSection) {
                photoSection.insertAdjacentHTML('afterend', tipHTML);
                
                // Auto-hide tip after 10 seconds
                setTimeout(() => {
                    const tip = document.querySelector('.photo-selection-tip');
                    if (tip) {
                        tip.style.transition = 'opacity 0.5s';
                        tip.style.opacity = '0';
                        setTimeout(() => tip.remove(), 500);
                    }
                }, 10000);
            }
        }, 1000);
    }
}

// Add primary photo indicator to photo previews
function addPrimaryPhotoIndicators() {
    const photoContainers = document.querySelectorAll('.photo-preview-container');
    
    photoContainers.forEach((container, index) => {
        const isPrimary = container.querySelector('input[name="primary_photo_index"]')?.checked;
        
        if (isPrimary || index === 0) {
            const primaryBadge = document.createElement('div');
            primaryBadge.className = 'primary-photo-badge';
            primaryBadge.innerHTML = '<i class="fas fa-star"></i> Primary';
            container.appendChild(primaryBadge);
        }
    });
}

// CSS for primary photo indicators
const primaryPhotoCSS = `
    .primary-photo-badge {
        position: absolute;
        top: 5px;
        left: 5px;
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: #333;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        z-index: 10;
    }
    
    .primary-photo-guide {
        border-left: 4px solid #17a2b8;
    }
    
    .photo-selection-tip {
        border-left: 4px solid #ffc107;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
`;

// Inject CSS
const style = document.createElement('style');
style.textContent = primaryPhotoCSS;
document.head.appendChild(style);