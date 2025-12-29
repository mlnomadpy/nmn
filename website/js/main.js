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
 * Navigation handling with UX enhancements
 */
function initNavigation() {
    const nav = document.querySelector('.main-nav');
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
    let lastScroll = 0;

    // Create scroll progress bar
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    document.body.prepend(progressBar);

    // Create back to top button
    const backToTop = document.createElement('button');
    backToTop.className = 'back-to-top';
    backToTop.innerHTML = '↑';
    backToTop.title = 'Back to top';
    backToTop.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
    document.body.appendChild(backToTop);

    // Create mobile hamburger
    const hamburger = document.createElement('div');
    hamburger.className = 'nav-hamburger';
    hamburger.innerHTML = '<span></span><span></span><span></span>';
    document.querySelector('.nav-container').appendChild(hamburger);

    // Create mobile drawer
    const drawer = document.createElement('div');
    drawer.className = 'nav-drawer';
    drawer.innerHTML = `
        <a href="#introduction">Introduction</a>
        <a href="#yat-product">ⵟ-Product</a>
        <a href="#visualizations">Visualizations</a>
        <a href="#results">Results</a>
        <a href="#theory">Blog</a>
        <a href="#code">Code</a>
        <a href="https://github.com/mlnomadpy/nmn" target="_blank">GitHub →</a>
    `;
    nav.after(drawer);

    // Mobile menu toggle
    hamburger.onclick = () => {
        hamburger.classList.toggle('open');
        drawer.classList.toggle('open');
    };

    // Close drawer on link click
    drawer.querySelectorAll('a').forEach(link => {
        link.onclick = () => {
            hamburger.classList.remove('open');
            drawer.classList.remove('open');
        };
    });

    // Scroll handler for progress, nav bg, back to top, and active section
    const sections = document.querySelectorAll('section[id]');

    function updateOnScroll() {
        const currentScroll = window.pageYOffset;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;

        // Update progress bar
        const scrollPercent = (currentScroll / docHeight) * 100;
        progressBar.style.width = `${scrollPercent}%`;

        // Update nav background
        nav.style.background = currentScroll > 100 ? 'rgba(10, 10, 15, 0.95)' : 'rgba(10, 10, 15, 0.85)';

        // Show/hide back to top button
        backToTop.classList.toggle('visible', currentScroll > 500);

        // Update active section in nav
        let activeSection = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionBottom = sectionTop + section.offsetHeight;
            if (currentScroll >= sectionTop && currentScroll < sectionBottom) {
                activeSection = section.id;
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${activeSection}`) {
                link.classList.add('active');
            }
        });

        lastScroll = currentScroll;
    }

    window.addEventListener('scroll', updateOnScroll, { passive: true });
    updateOnScroll(); // Initial call

    // Smooth scroll for nav links
    navLinks.forEach(anchor => {
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
 * Hero section particles - Terminal Matrix Rain Effect
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
    let matrixDrops = [];
    let animationId;
    let time = 0;

    // Terminal theme colors
    const colors = {
        primary: 'rgba(79, 249, 117, ',      // Terminal green
        secondary: 'rgba(77, 238, 234, ',    // Cyan
        accent: 'rgba(249, 215, 28, ',       // Yellow
    };

    function resize() {
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
        initMatrixDrops();
    }

    // Matrix-style falling characters
    function initMatrixDrops() {
        matrixDrops = [];
        const columns = Math.floor(canvas.width / 20);
        for (let i = 0; i < columns; i++) {
            matrixDrops.push({
                x: i * 20,
                y: Math.random() * canvas.height,
                speed: Math.random() * 2 + 1,
                chars: [],
                length: Math.floor(Math.random() * 15) + 5
            });
        }
    }

    function createParticles() {
        particles = [];
        const numParticles = Math.floor((canvas.width * canvas.height) / 20000);

        for (let i = 0; i < numParticles; i++) {
            const colorChoice = Math.random();
            let color;
            if (colorChoice < 0.7) color = colors.primary;
            else if (colorChoice < 0.9) color = colors.secondary;
            else color = colors.accent;

            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                radius: Math.random() * 2 + 1,
                alpha: Math.random() * 0.6 + 0.2,
                color: color,
                pulse: Math.random() * Math.PI * 2,
                pulseSpeed: Math.random() * 0.02 + 0.01
            });
        }
    }

    function animate() {
        time += 0.01;
        ctx.fillStyle = 'rgba(5, 5, 5, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw matrix rain (subtle)
        ctx.font = '12px "Share Tech Mono", monospace';
        matrixDrops.forEach(drop => {
            const char = String.fromCharCode(0x30A0 + Math.random() * 96);
            const alpha = 0.08 + Math.random() * 0.05;
            ctx.fillStyle = colors.primary + alpha + ')';
            ctx.fillText(char, drop.x, drop.y);

            drop.y += drop.speed;
            if (drop.y > canvas.height) {
                drop.y = 0;
                drop.speed = Math.random() * 2 + 1;
            }
        });

        // Update and draw particles with glow
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.pulse += p.pulseSpeed;

            // Wrap around
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            // Pulsing alpha
            const pulseAlpha = p.alpha * (0.7 + Math.sin(p.pulse) * 0.3);

            // Outer glow
            const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 3);
            gradient.addColorStop(0, p.color + pulseAlpha + ')');
            gradient.addColorStop(0.5, p.color + (pulseAlpha * 0.3) + ')');
            gradient.addColorStop(1, p.color + '0)');

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius * 3, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Core
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color + pulseAlpha + ')';
            ctx.fill();
        });

        // Draw neural connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 120) {
                    const alpha = (1 - dist / 120) * 0.15;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = colors.primary + alpha + ')';
                    ctx.lineWidth = 0.5;
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

