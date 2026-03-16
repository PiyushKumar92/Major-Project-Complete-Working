/**
 * Enhanced Detection Viewer with Bounding Box Interactions
 * Provides interactive features for AI detection results
 */

class DetectionViewer {
    constructor() {
        this.currentDetection = null;
        this.boundingBoxes = new Map();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDetectionData();
        this.initializeTooltips();
    }

    setupEventListeners() {
        // Detection frame hover events
        document.addEventListener('mouseover', (e) => {
            if (e.target.closest('.detection-frame-container')) {
                this.handleDetectionHover(e.target.closest('.detection-frame-container'));
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.closest('.detection-frame-container')) {
                this.handleDetectionLeave(e.target.closest('.detection-frame-container'));
            }
        });

        // Detection click events
        document.addEventListener('click', (e) => {
            if (e.target.closest('.detection-frame')) {
                this.handleDetectionClick(e.target.closest('.detection-frame'));
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardNavigation(e);
        });
    }

    handleDetectionHover(container) {
        const boundingBox = container.querySelector('.bounding-box-overlay');
        const confidenceLabel = container.querySelector('.confidence-label');
        
        if (boundingBox) {
            // Enhanced hover effect
            boundingBox.style.borderWidth = '3px';
            boundingBox.style.background = 'rgba(0, 123, 255, 0.2)';
            boundingBox.style.animation = 'none';
        }

        if (confidenceLabel) {
            confidenceLabel.style.transform = 'scale(1.1)';
        }

        // Show tooltip with detection details
        this.showDetectionTooltip(container);
    }

    handleDetectionLeave(container) {
        const boundingBox = container.querySelector('.bounding-box-overlay');
        const confidenceLabel = container.querySelector('.confidence-label');
        
        if (boundingBox) {
            // Reset to original state
            boundingBox.style.borderWidth = '2px';
            boundingBox.style.animation = 'pulse-border 2s infinite';
        }

        if (confidenceLabel) {
            confidenceLabel.style.transform = 'scale(1)';
        }

        // Hide tooltip
        this.hideDetectionTooltip();
    }

    handleDetectionClick(frame) {
        const container = frame.closest('.detection-frame-container');
        const detectionId = container.dataset.detectionId;
        
        if (detectionId) {
            this.showDetectionModal(detectionId);
        }
    }

    showDetectionTooltip(container) {
        // Remove existing tooltip
        this.hideDetectionTooltip();

        const confidence = container.dataset.confidence || '0';
        const detectionId = container.dataset.detectionId || 'Unknown';
        const timestamp = container.dataset.timestamp || 'Unknown';

        const tooltip = document.createElement('div');
        tooltip.className = 'detection-tooltip';
        tooltip.innerHTML = `
            <strong>Detection #${detectionId}</strong><br>
            Confidence: ${(parseFloat(confidence) * 100).toFixed(1)}%<br>
            Time: ${timestamp}<br>
            <small>Click to view details</small>
        `;

        // Position tooltip
        const rect = container.getBoundingClientRect();
        tooltip.style.left = (rect.left + rect.width + 10) + 'px';
        tooltip.style.top = rect.top + 'px';

        document.body.appendChild(tooltip);
        
        // Fade in
        setTimeout(() => {
            tooltip.style.opacity = '1';
        }, 10);
    }

    hideDetectionTooltip() {
        const existingTooltip = document.querySelector('.detection-tooltip');
        if (existingTooltip) {
            existingTooltip.remove();
        }
    }

    showDetectionModal(detectionId) {
        // Create enhanced modal with bounding box
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'enhancedDetectionModal';
        modal.innerHTML = `
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-search-plus me-2"></i>
                            Enhanced Detection View #${detectionId}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="enhancedDetectionContent">
                        <div class="text-center py-4">
                            <div class="detection-loading"></div>
                            <p class="mt-2">Loading detection details...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();

        // Load detection details
        this.loadDetectionDetails(detectionId);

        // Clean up modal on hide
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });
    }

    loadDetectionDetails(detectionId) {
        fetch(`/admin/detection/${detectionId}/details`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.renderEnhancedDetectionView(data.detection);
                } else {
                    this.showDetectionError('Failed to load detection details');
                }
            })
            .catch(error => {
                console.error('Error loading detection:', error);
                this.showDetectionError('Error loading detection details');
            });
    }

    renderEnhancedDetectionView(detection) {
        const content = document.getElementById('enhancedDetectionContent');
        if (!content) return;

        const confidenceColor = this.getConfidenceColor(detection.confidence_score);
        
        content.innerHTML = `
            <div class="row">
                <div class="col-lg-8">
                    <div class="enhanced-detection-frame" style="position: relative; text-align: center;">
                        <img src="/static/${detection.frame_path}" 
                             class="img-fluid rounded shadow" 
                             style="max-height: 500px;">
                        
                        <!-- Enhanced bounding box -->
                        <div class="enhanced-bounding-box" 
                             style="position: absolute; 
                                    border: 4px solid ${confidenceColor}; 
                                    background: rgba(${this.getConfidenceRGB(detection.confidence_score)}, 0.15);
                                    top: 8%; left: 12%; 
                                    width: 76%; height: 70%;
                                    pointer-events: none;
                                    border-radius: 6px;
                                    animation: pulse-border 2s infinite;">
                            
                            <!-- Confidence label -->
                            <div style="position: absolute; 
                                       top: -35px; left: 0;
                                       background: ${confidenceColor}; 
                                       color: white; 
                                       padding: 8px 16px; 
                                       font-size: 16px; 
                                       border-radius: 6px;
                                       font-weight: bold;
                                       box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                                Detection #${detection.id} - ${(detection.confidence_score * 100).toFixed(1)}% Confidence
                            </div>
                            
                            <!-- Analysis method badge -->
                            <div style="position: absolute; 
                                       bottom: -35px; right: 0;
                                       background: rgba(0, 123, 255, 0.9); 
                                       color: white; 
                                       padding: 6px 12px; 
                                       font-size: 12px; 
                                       border-radius: 4px;
                                       font-weight: bold;">
                                ${detection.analysis_method || 'AI Analysis'}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-4">
                    <div class="detection-details">
                        <h6 class="text-primary mb-3">
                            <i class="fas fa-info-circle me-2"></i>Detection Information
                        </h6>
                        
                        <div class="detail-card mb-3">
                            <div class="card">
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-6">
                                            <div class="text-center">
                                                <div class="h4 text-${this.getConfidenceBootstrapColor(detection.confidence_score)}">
                                                    ${(detection.confidence_score * 100).toFixed(1)}%
                                                </div>
                                                <small class="text-muted">Overall Confidence</small>
                                            </div>
                                        </div>
                                        <div class="col-6">
                                            <div class="text-center">
                                                <div class="h4 text-info">
                                                    ${detection.formatted_timestamp}
                                                </div>
                                                <small class="text-muted">Timestamp</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="detail-list">
                            <div class="detail-item mb-2">
                                <strong>Detection ID:</strong> #${detection.id}
                            </div>
                            <div class="detail-item mb-2">
                                <strong>Video Time:</strong> ${detection.timestamp.toFixed(2)}s
                            </div>
                            ${detection.face_match_score ? `
                            <div class="detail-item mb-2">
                                <strong>Face Match:</strong> 
                                <span class="badge bg-info">${(detection.face_match_score * 100).toFixed(1)}%</span>
                            </div>
                            ` : ''}
                            ${detection.clothing_match_score ? `
                            <div class="detail-item mb-2">
                                <strong>Clothing Match:</strong> 
                                <span class="badge bg-secondary">${(detection.clothing_match_score * 100).toFixed(1)}%</span>
                            </div>
                            ` : ''}
                            <div class="detail-item mb-2">
                                <strong>Analysis Method:</strong> 
                                <span class="badge bg-primary">${detection.analysis_method || 'Unknown'}</span>
                            </div>
                            <div class="detail-item mb-2">
                                <strong>Verification:</strong> 
                                ${this.getVerificationBadge(detection.verified)}
                            </div>
                            ${detection.notes ? `
                            <div class="detail-item mb-2">
                                <strong>Notes:</strong><br>
                                <small class="text-muted">${detection.notes}</small>
                            </div>
                            ` : ''}
                        </div>
                        
                        <div class="action-buttons mt-4">
                            <button class="btn btn-success btn-sm me-2" onclick="verifyDetection(${detection.id}, true)">
                                <i class="fas fa-check me-1"></i>Verify
                            </button>
                            <button class="btn btn-danger btn-sm me-2" onclick="verifyDetection(${detection.id}, false)">
                                <i class="fas fa-times me-1"></i>Reject
                            </button>
                            <button class="btn btn-info btn-sm" onclick="downloadDetection(${detection.id})">
                                <i class="fas fa-download me-1"></i>Download
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    showDetectionError(message) {
        const content = document.getElementById('enhancedDetectionContent');
        if (content) {
            content.innerHTML = `
                <div class="alert alert-danger text-center">
                    <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                    <h5>Error</h5>
                    <p>${message}</p>
                </div>
            `;
        }
    }

    getConfidenceColor(confidence) {
        if (confidence > 0.8) return '#28a745'; // Green
        if (confidence > 0.6) return '#ffc107'; // Orange
        return '#dc3545'; // Red
    }

    getConfidenceRGB(confidence) {
        if (confidence > 0.8) return '40, 167, 69'; // Green RGB
        if (confidence > 0.6) return '255, 193, 7'; // Orange RGB
        return '220, 53, 69'; // Red RGB
    }

    getConfidenceBootstrapColor(confidence) {
        if (confidence > 0.8) return 'success';
        if (confidence > 0.6) return 'warning';
        return 'danger';
    }

    getVerificationBadge(verified) {
        if (verified === true) {
            return '<span class="badge bg-success"><i class="fas fa-check me-1"></i>Verified</span>';
        } else if (verified === false) {
            return '<span class="badge bg-danger"><i class="fas fa-times me-1"></i>Rejected</span>';
        } else {
            return '<span class="badge bg-warning"><i class="fas fa-clock me-1"></i>Pending</span>';
        }
    }

    handleKeyboardNavigation(e) {
        // ESC to close modals/tooltips
        if (e.key === 'Escape') {
            this.hideDetectionTooltip();
        }
    }

    loadDetectionData() {
        // Load detection coordinates if available
        document.querySelectorAll('.detection-frame-container').forEach(container => {
            const detectionId = container.dataset.detectionId;
            if (detectionId) {
                this.loadBoundingBoxCoordinates(detectionId, container);
            }
        });
    }

    loadBoundingBoxCoordinates(detectionId, container) {
        // If detection has stored coordinates, use them
        // Otherwise use default positioning
        const boundingBox = container.querySelector('.bounding-box-overlay');
        if (boundingBox && container.dataset.coordinates) {
            try {
                const coords = JSON.parse(container.dataset.coordinates);
                boundingBox.style.left = coords.x + '%';
                boundingBox.style.top = coords.y + '%';
                boundingBox.style.width = coords.width + '%';
                boundingBox.style.height = coords.height + '%';
            } catch (e) {
                console.log('Invalid coordinates for detection', detectionId);
            }
        }
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }
}

// Global functions for detection actions
function verifyDetection(detectionId, isValid) {
    const action = isValid ? 'verify' : 'reject';
    if (confirm(`${isValid ? 'Verify' : 'Reject'} this detection?`)) {
        fetch(`/admin/detection/${detectionId}/${action}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || ''
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Close modal and refresh page
                const modal = bootstrap.Modal.getInstance(document.getElementById('enhancedDetectionModal'));
                if (modal) modal.hide();
                location.reload();
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing request');
        });
    }
}

function downloadDetection(detectionId) {
    // Create download link for detection frame
    const detection = document.querySelector(`[data-detection-id="${detectionId}"]`);
    if (detection) {
        const img = detection.querySelector('img');
        if (img) {
            const link = document.createElement('a');
            link.href = img.src;
            link.download = `detection_${detectionId}.jpg`;
            link.click();
        }
    }
}

// Initialize detection viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.detectionViewer = new DetectionViewer();
});