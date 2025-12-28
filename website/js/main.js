/**
 * Main JavaScript for NMN Website
 * Initializes all visualizations and handles interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize KaTeX
    initMathRendering();

    // Initialize visualizations with delay for DOM readiness
    setTimeout(() => {
        initHeatmapViz();
        initGradientViz();
        initXORDemo();
        initDecisionBoundaryViz();
        initLossLandscapeViz();
        initTopologicalDistortionViz();
    }, 100);

    // Initialize UI interactions
    initNavigation();
    initCodeTabs();
    initScrollAnimations();
    initParticles();
});

/**
 * Fullscreen Visualization System
 */
let currentFullscreenViz = null;
let fullscreenRenderInterval = null;

function openFullscreen(canvasId) {
    const overlay = document.getElementById('fullscreenOverlay');
    const fsCanvas = document.getElementById('fullscreen-canvas');
    const label = document.getElementById('fullscreen-label');
    const sourceCanvas = document.getElementById(canvasId);

    if (!overlay || !fsCanvas || !sourceCanvas) return;

    // Determine visualization type and label
    const isHeatmap = canvasId.startsWith('heatmap-');
    const isGradient = canvasId.startsWith('gradient-');
    const metric = canvasId.replace('heatmap-', '').replace('gradient-', '');

    const metricLabels = {
        'dot': 'Dot Product',
        'euclidean': 'Euclidean Distance²',
        'yat': 'ⵟ-Product',
        'cosine': 'Cosine Similarity'
    };

    const vizType = isHeatmap ? 'Similarity Heatmap' : 'Gradient Field';
    label.textContent = `${metricLabels[metric] || metric} — ${vizType}`;

    currentFullscreenViz = { canvasId, metric, isHeatmap, isGradient };

    // Setup fullscreen canvas
    const fsCtx = fsCanvas.getContext('2d');
    const range = { min: -8, max: 8 };

    // Add drag events to fullscreen canvas
    setupFullscreenDrag(fsCanvas, range);

    // Show overlay
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Initial render
    renderFullscreen();

    // Setup continuous render loop
    fullscreenRenderInterval = setInterval(renderFullscreen, 100);
}

function renderFullscreen() {
    if (!currentFullscreenViz) return;

    const fsCanvas = document.getElementById('fullscreen-canvas');
    const fsCtx = fsCanvas.getContext('2d');

    if (currentFullscreenViz.isHeatmap && typeof heatmapViz !== 'undefined' && heatmapViz) {
        heatmapViz.renderHeatmap(fsCtx, fsCanvas, currentFullscreenViz.metric);
    } else if (currentFullscreenViz.isGradient && typeof gradientViz !== 'undefined' && gradientViz) {
        gradientViz.renderGradientField(fsCtx, fsCanvas, currentFullscreenViz.metric);
    }
}

function setupFullscreenDrag(canvas, range) {
    let isDragging = false;

    const getMousePos = (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    };

    const pixelToCoord = (px, py) => {
        const r = range.max - range.min;
        return [
            range.min + (px / canvas.width) * r,
            range.max - (py / canvas.height) * r
        ];
    };

    const coordToPixel = (x, y) => {
        const r = range.max - range.min;
        return {
            x: ((x - range.min) / r) * canvas.width,
            y: ((range.max - y) / r) * canvas.height
        };
    };

    canvas.onmousedown = (e) => {
        if (!heatmapViz) return;
        const pos = getMousePos(e);
        const anchorPx = coordToPixel(heatmapViz.anchor[0], heatmapViz.anchor[1]);
        const dist = Math.sqrt(Math.pow(pos.x - anchorPx.x, 2) + Math.pow(pos.y - anchorPx.y, 2));
        if (dist < 30) isDragging = true;
    };

    canvas.onmousemove = (e) => {
        if (isDragging && heatmapViz) {
            const pos = getMousePos(e);
            const coord = pixelToCoord(pos.x, pos.y);
            heatmapViz.anchor = coord;
            heatmapViz.updateAnchorDisplay();
            heatmapViz.render();
            renderFullscreen();
        }
    };

    canvas.onmouseup = () => { isDragging = false; };
    canvas.onmouseleave = () => { };
}

function closeFullscreen() {
    const overlay = document.getElementById('fullscreenOverlay');
    if (overlay) overlay.classList.remove('active');
    document.body.style.overflow = '';
    currentFullscreenViz = null;

    if (fullscreenRenderInterval) {
        clearInterval(fullscreenRenderInterval);
        fullscreenRenderInterval = null;
    }
}

// Escape key to close fullscreen
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && currentFullscreenViz) {
        closeFullscreen();
    }
});

// Make functions globally available
window.openFullscreen = openFullscreen;
window.closeFullscreen = closeFullscreen;

/**
 * Initialize KaTeX math rendering
 */
function initMathRendering() {
    if (typeof renderMathInElement === 'function') {
        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\[', right: '\\]', display: true },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false,
            macros: {
                "\\ⵟ": "\\text{ⵟ}"
            }
        });
    }
}

/**
 * Navigation handling
 */
function initNavigation() {
    const nav = document.querySelector('.main-nav');
    let lastScroll = 0;

    // Hide/show nav on scroll
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 100) {
            nav.style.background = 'rgba(10, 10, 15, 0.95)';
        } else {
            nav.style.background = 'rgba(10, 10, 15, 0.85)';
        }

        lastScroll = currentScroll;
    });

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-links a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            const target = document.querySelector(href);
            if (target) {
                // Make target section and its children visible
                target.classList.add('animate-in');
                const parentSection = target.closest('.section') || target;
                parentSection.classList.add('animate-in');

                // Animate all blog elements within the section
                parentSection.querySelectorAll('.theorem-post, .theory-nav, .theory-summary').forEach(el => {
                    el.classList.add('animate-in');
                });

                const offset = 80; // Nav height
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });

                // Update URL hash
                history.pushState(null, null, href);
            }
        });
    });
}

/**
 * Code tabs functionality
 */
function initCodeTabs() {
    const tabs = document.querySelectorAll('.code-tab');
    const panels = document.querySelectorAll('.code-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;

            // Update tabs
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update panels
            panels.forEach(p => {
                p.classList.remove('active');
                if (p.id === `panel-${targetId}`) {
                    p.classList.add('active');
                }
            });
        });
    });
}

/**
 * Scroll-triggered animations
 */
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.05,  // Lower threshold for better detection
        rootMargin: '50px 0px -20px 0px'  // More generous margins
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');

                // Trigger visualization updates when visible
                if (entry.target.classList.contains('viz-block')) {
                    // Re-render visualizations that might have been hidden
                    if (typeof heatmapViz !== 'undefined' && heatmapViz) {
                        heatmapViz.render();
                    }
                    if (typeof gradientViz !== 'undefined' && gradientViz) {
                        gradientViz.render();
                    }
                }
            }
        });
    }, observerOptions);

    // Observe sections, viz blocks, and blog elements
    const animatedElements = document.querySelectorAll(
        '.section, .viz-block, .arch-card, .property-card, .problem-card, ' +
        '.theorem-post, .theory-nav, .theory-summary'
    );

    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Handle hash navigation - make target section visible immediately
    function handleHashNavigation() {
        const hash = window.location.hash;
        if (hash) {
            const target = document.querySelector(hash);
            if (target) {
                // Make the target and its parent section visible
                target.classList.add('animate-in');
                const parentSection = target.closest('.section');
                if (parentSection) {
                    parentSection.classList.add('animate-in');
                    // Also animate all children in the section
                    parentSection.querySelectorAll('.theorem-post, .theory-nav, .theory-summary').forEach(el => {
                        el.classList.add('animate-in');
                    });
                }
            }
        }
    }

    // Run on load and hash change
    handleHashNavigation();
    window.addEventListener('hashchange', handleHashNavigation);
}

/**
 * Add animate-in class styles
 */
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);

/**
 * Hero section particles
 */
function initParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let particles = [];
    let animationId;

    function resize() {
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    }

    function createParticles() {
        particles = [];
        const numParticles = Math.floor((canvas.width * canvas.height) / 15000);

        for (let i = 0; i < numParticles; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                radius: Math.random() * 1.5 + 0.5,
                alpha: Math.random() * 0.5 + 0.1
            });
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Update and draw particles
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;

            // Wrap around
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 212, 255, ${p.alpha})`;
            ctx.fill();
        });

        // Draw connections
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.05)';
        ctx.lineWidth = 0.5;

        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
        }

        animationId = requestAnimationFrame(animate);
    }

    resize();
    createParticles();
    animate();

    window.addEventListener('resize', () => {
        resize();
        createParticles();
    });
}

/**
 * Copy citation to clipboard
 */
function copyBibtex() {
    const bibtex = `@article{bouhsine2025nomoredelulu,
  author = {Taha Bouhsine},
  title = {No More DeLuLu: A Kernel-Based Activation-Free Neural Networks},
  year = {2025},
  url = {https://github.com/mlnomadpy/nmn}
}`;

    navigator.clipboard.writeText(bibtex).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg> Copied!';
        btn.style.color = '#10b981';

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.style.color = '';
        }, 2000);
    });
}

// Make copyBibtex globally available
window.copyBibtex = copyBibtex;

/**
 * Handle window resize for visualizations
 */
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        // Re-render visualizations on resize
        if (typeof heatmapViz !== 'undefined' && heatmapViz) {
            heatmapViz.render();
        }
        if (typeof gradientViz !== 'undefined' && gradientViz) {
            gradientViz.render();
        }
        if (typeof xorDemo !== 'undefined' && xorDemo) {
            xorDemo.render();
        }
        if (typeof decisionBoundaryViz !== 'undefined' && decisionBoundaryViz) {
            decisionBoundaryViz.render();
        }
    }, 250);
});

/**
 * Debug mode - expose visualizations globally
 */
if (window.location.hash === '#debug') {
    window.NMN = {
        heatmapViz: () => heatmapViz,
        gradientViz: () => gradientViz,
        xorDemo: () => xorDemo,
        decisionBoundaryViz: () => decisionBoundaryViz,
        lossLandscapeViz: () => lossLandscapeViz,
        MathUtils
    };
    console.log('NMN Debug mode enabled. Access visualizations via window.NMN');
}

/**
 * Blog Modal System
 */
let currentModal = null;

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    const overlay = document.getElementById('modalOverlay');

    if (modal && overlay) {
        // Close any existing modal
        closeModal();

        // Open new modal
        overlay.classList.add('active');
        modal.classList.add('active');
        currentModal = modal;

        // Prevent body scroll
        document.body.style.overflow = 'hidden';

        // Re-render math in modal
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(modal, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false }
                ],
                throwOnError: false
            });
        }

        // Scroll modal body to top
        const modalBody = modal.querySelector('.blog-modal-body');
        if (modalBody) {
            modalBody.scrollTop = 0;
        }
    }
}

function closeModal() {
    const overlay = document.getElementById('modalOverlay');
    const modals = document.querySelectorAll('.blog-modal');

    overlay.classList.remove('active');
    modals.forEach(modal => modal.classList.remove('active'));
    currentModal = null;

    // Restore body scroll
    document.body.style.overflow = '';
}

function navigateModal(modalId) {
    openModal(modalId);
}

// Initialize modal event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Blog card clicks
    const blogCards = document.querySelectorAll('.blog-card');
    blogCards.forEach(card => {
        card.addEventListener('click', () => {
            const modalId = card.getAttribute('data-modal');
            if (modalId) {
                openModal(modalId);
            }
        });
    });

    // Overlay click to close
    const overlay = document.getElementById('modalOverlay');
    if (overlay) {
        overlay.addEventListener('click', closeModal);
    }

    // Escape key to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && currentModal) {
            closeModal();
        }
    });

    // Prevent modal body clicks from closing
    const modals = document.querySelectorAll('.blog-modal');
    modals.forEach(modal => {
        modal.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    });
});

