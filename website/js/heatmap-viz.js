/**
 * Heatmap Visualization Module
 * Interactive 2D heatmaps comparing different similarity measures
 */

class HeatmapVisualization {
    constructor() {
        this.canvases = {
            dot: document.getElementById('heatmap-dot'),
            euclidean: document.getElementById('heatmap-euclidean'),
            yat: document.getElementById('heatmap-yat'),
            cosine: document.getElementById('heatmap-cosine')
        };

        this.contexts = {};
        this.anchor = [3, 4];  // Default anchor position (w vector)
        this.epsilon = 1.0;
        this.range = { min: -8, max: 8 };
        this.isDragging = false;
        this.activeCanvas = null;

        this.init();
    }

    init() {
        // Initialize contexts
        for (const [name, canvas] of Object.entries(this.canvases)) {
            if (canvas) {
                this.contexts[name] = canvas.getContext('2d');
                this.setupCanvasEvents(canvas);
            }
        }

        // Setup controls
        this.setupControls();

        // Initial render
        this.render();
    }

    setupControls() {
        const epsilonSlider = document.getElementById('epsilon-slider');
        const epsilonDisplay = document.getElementById('epsilon-display');

        if (epsilonSlider) {
            epsilonSlider.addEventListener('input', (e) => {
                this.epsilon = Math.pow(10, parseFloat(e.target.value));
                if (epsilonDisplay) {
                    epsilonDisplay.textContent = this.epsilon.toFixed(4);
                }
                this.render();
            });
        }
    }

    setupCanvasEvents(canvas) {
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
            const range = this.range.max - this.range.min;
            return [
                this.range.min + (px / canvas.width) * range,
                this.range.max - (py / canvas.height) * range  // Y is inverted
            ];
        };

        canvas.addEventListener('mousedown', (e) => {
            const pos = getMousePos(e);
            const coord = pixelToCoord(pos.x, pos.y);

            // Check if click is near anchor
            const anchorPx = this.coordToPixel(this.anchor[0], this.anchor[1], canvas);
            const dist = Math.sqrt(Math.pow(pos.x - anchorPx.x, 2) + Math.pow(pos.y - anchorPx.y, 2));

            if (dist < 20) {
                this.isDragging = true;
                this.activeCanvas = canvas;
            }
        });

        canvas.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const pos = getMousePos(e);
                const coord = pixelToCoord(pos.x, pos.y);
                this.anchor = coord;
                this.updateAnchorDisplay();
                this.render();
            }
        });

        canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.activeCanvas = null;
        });

        canvas.addEventListener('mouseleave', () => {
            if (this.isDragging && this.activeCanvas === canvas) {
                // Continue tracking outside canvas
            }
        });

        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const pos = {
                x: (touch.clientX - rect.left) * (canvas.width / rect.width),
                y: (touch.clientY - rect.top) * (canvas.height / rect.height)
            };

            const anchorPx = this.coordToPixel(this.anchor[0], this.anchor[1], canvas);
            const dist = Math.sqrt(Math.pow(pos.x - anchorPx.x, 2) + Math.pow(pos.y - anchorPx.y, 2));

            if (dist < 30) {
                this.isDragging = true;
            }
        });

        canvas.addEventListener('touchmove', (e) => {
            if (this.isDragging) {
                e.preventDefault();
                const touch = e.touches[0];
                const rect = canvas.getBoundingClientRect();
                const px = (touch.clientX - rect.left) * (canvas.width / rect.width);
                const py = (touch.clientY - rect.top) * (canvas.height / rect.height);
                this.anchor = pixelToCoord(px, py);
                this.updateAnchorDisplay();
                this.render();
            }
        });

        canvas.addEventListener('touchend', () => {
            this.isDragging = false;
        });
    }

    coordToPixel(x, y, canvas) {
        const range = this.range.max - this.range.min;
        return {
            x: ((x - this.range.min) / range) * canvas.width,
            y: ((this.range.max - y) / range) * canvas.height  // Y inverted
        };
    }

    updateAnchorDisplay() {
        const display = document.getElementById('anchor-display');
        if (display) {
            display.textContent = `(${this.anchor[0].toFixed(1)}, ${this.anchor[1].toFixed(1)})`;
        }
    }

    /**
     * Compute value at a point for given metric
     */
    computeValue(metric, x, y) {
        const w = this.anchor;
        const point = [x, y];

        switch (metric) {
            case 'dot':
                return MathUtils.dotProduct(w, point);
            case 'euclidean':
                return MathUtils.squaredDistance(w, point);
            case 'yat':
                return Math.log1p(MathUtils.yatProduct(w, point, this.epsilon));
            case 'cosine':
                return MathUtils.cosineSimilarity(w, point);
            default:
                return 0;
        }
    }

    /**
     * Render a single heatmap
     */
    renderHeatmap(ctx, canvas, metric) {
        const width = canvas.width;
        const height = canvas.height;
        const imageData = ctx.createImageData(width, height);

        // Compute all values first to find min/max
        const values = [];
        const resolution = 1; // Full resolution for quality

        for (let py = 0; py < height; py += resolution) {
            for (let px = 0; px < width; px += resolution) {
                const x = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y = this.range.max - (py / height) * (this.range.max - this.range.min);
                values.push({
                    px, py,
                    value: this.computeValue(metric, x, y)
                });
            }
        }

        // Find min/max for normalization
        let minVal = Infinity, maxVal = -Infinity;
        for (const v of values) {
            if (isFinite(v.value)) {
                minVal = Math.min(minVal, v.value);
                maxVal = Math.max(maxVal, v.value);
            }
        }

        // Handle edge cases
        if (!isFinite(minVal)) minVal = 0;
        if (!isFinite(maxVal)) maxVal = 1;
        if (maxVal === minVal) maxVal = minVal + 1;

        // Choose colormap based on metric - using terminal theme
        let colormap;
        let normalizeForDiverging = false;

        switch (metric) {
            case 'dot':
                colormap = MathUtils.colorMaps.terminalDiverging;
                normalizeForDiverging = true;
                break;
            case 'cosine':
                colormap = MathUtils.colorMaps.terminalDiverging;
                normalizeForDiverging = true;
                break;
            case 'yat':
                colormap = MathUtils.colorMaps.terminal;
                break;
            case 'euclidean':
                colormap = MathUtils.colorMaps.terminal;
                break;
            default:
                colormap = MathUtils.colorMaps.terminal;
        }

        // Render pixels
        for (const v of values) {
            let normalized;
            if (normalizeForDiverging) {
                // For diverging colormaps, center at 0
                const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal));
                normalized = (v.value + absMax) / (2 * absMax);
            } else {
                normalized = (v.value - minVal) / (maxVal - minVal);
            }

            const color = colormap(normalized);
            const rgb = color.match(/\d+/g).map(Number);

            // Fill resolution x resolution block
            for (let dy = 0; dy < resolution && v.py + dy < height; dy++) {
                for (let dx = 0; dx < resolution && v.px + dx < width; dx++) {
                    const idx = ((v.py + dy) * width + (v.px + dx)) * 4;
                    imageData.data[idx] = rgb[0];
                    imageData.data[idx + 1] = rgb[1];
                    imageData.data[idx + 2] = rgb[2];
                    imageData.data[idx + 3] = 255;
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);

        // Draw contour lines
        this.drawContours(ctx, canvas, metric, minVal, maxVal);

        // Draw anchor point
        this.drawAnchor(ctx, canvas);

        // Draw axes
        this.drawAxes(ctx, canvas);
    }

    /**
     * Draw contour lines
     */
    drawContours(ctx, canvas, metric, minVal, maxVal) {
        const numContours = 8;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 0.5;

        for (let i = 1; i < numContours; i++) {
            const level = minVal + (i / numContours) * (maxVal - minVal);
            this.drawContourLine(ctx, canvas, metric, level);
        }
    }

    /**
     * Draw a single contour line using marching squares (simplified)
     */
    drawContourLine(ctx, canvas, metric, level) {
        const step = 10;
        const width = canvas.width;
        const height = canvas.height;

        ctx.beginPath();

        for (let py = 0; py < height - step; py += step) {
            for (let px = 0; px < width - step; px += step) {
                const corners = [];
                for (let dy = 0; dy <= step; dy += step) {
                    for (let dx = 0; dx <= step; dx += step) {
                        const x = this.range.min + ((px + dx) / width) * (this.range.max - this.range.min);
                        const y = this.range.max - ((py + dy) / height) * (this.range.max - this.range.min);
                        corners.push({
                            px: px + dx,
                            py: py + dy,
                            value: this.computeValue(metric, x, y)
                        });
                    }
                }

                // Simple contour: draw line segments where level crosses
                const above = corners.map(c => c.value > level);

                // Check for crossing on edges
                if (above[0] !== above[1]) {
                    const t = (level - corners[0].value) / (corners[1].value - corners[0].value);
                    ctx.moveTo(corners[0].px + t * step, corners[0].py);
                    ctx.lineTo(corners[0].px + t * step, corners[0].py + 2);
                }
            }
        }

        ctx.stroke();
    }

    /**
     * Draw the anchor point (w vector) - Terminal Style
     */
    drawAnchor(ctx, canvas) {
        const pos = this.coordToPixel(this.anchor[0], this.anchor[1], canvas);

        // Outer glow effect - terminal green
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 25);
        gradient.addColorStop(0, 'rgba(79, 249, 117, 0.6)');
        gradient.addColorStop(0.5, 'rgba(79, 249, 117, 0.2)');
        gradient.addColorStop(1, 'rgba(79, 249, 117, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 25, 0, Math.PI * 2);
        ctx.fill();

        // Pulsing ring
        const pulsePhase = (Date.now() % 2000) / 2000;
        const pulseRadius = 12 + Math.sin(pulsePhase * Math.PI * 2) * 3;
        ctx.strokeStyle = `rgba(77, 238, 234, ${0.3 + Math.sin(pulsePhase * Math.PI * 2) * 0.2})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, pulseRadius, 0, Math.PI * 2);
        ctx.stroke();

        // Inner star - terminal colors
        ctx.fillStyle = '#4ff975';
        ctx.strokeStyle = '#4deeea';
        ctx.lineWidth = 2;

        this.drawStar(ctx, pos.x, pos.y, 5, 10, 5);

        // Coordinate label with terminal font
        ctx.font = '11px "Share Tech Mono", monospace';
        ctx.fillStyle = '#4ff975';
        ctx.strokeStyle = '#050505';
        ctx.lineWidth = 3;
        const label = `w(${this.anchor[0].toFixed(1)}, ${this.anchor[1].toFixed(1)})`;
        ctx.strokeText(label, pos.x + 15, pos.y - 8);
        ctx.fillText(label, pos.x + 15, pos.y - 8);
    }

    /**
     * Draw a star shape
     */
    drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius) {
        let rot = Math.PI / 2 * 3;
        const step = Math.PI / spikes;

        ctx.beginPath();
        ctx.moveTo(cx, cy - outerRadius);

        for (let i = 0; i < spikes; i++) {
            ctx.lineTo(
                cx + Math.cos(rot) * outerRadius,
                cy + Math.sin(rot) * outerRadius
            );
            rot += step;

            ctx.lineTo(
                cx + Math.cos(rot) * innerRadius,
                cy + Math.sin(rot) * innerRadius
            );
            rot += step;
        }

        ctx.lineTo(cx, cy - outerRadius);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }

    /**
     * Draw coordinate axes - Terminal Style with grid
     */
    drawAxes(ctx, canvas) {
        // Faint grid lines
        ctx.setLineDash([2, 4]);
        ctx.strokeStyle = 'rgba(79, 249, 117, 0.08)';
        ctx.lineWidth = 1;

        for (let x = this.range.min; x <= this.range.max; x += 2) {
            const px = this.coordToPixel(x, 0, canvas).x;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, canvas.height);
            ctx.stroke();
        }

        for (let y = this.range.min; y <= this.range.max; y += 2) {
            const py = this.coordToPixel(0, y, canvas).y;
            ctx.beginPath();
            ctx.moveTo(0, py);
            ctx.lineTo(canvas.width, py);
            ctx.stroke();
        }

        ctx.setLineDash([]);

        // Main axes - terminal green
        ctx.strokeStyle = 'rgba(79, 249, 117, 0.4)';
        ctx.lineWidth = 1;

        // X axis (y = 0)
        const y0 = this.coordToPixel(0, 0, canvas).y;
        ctx.beginPath();
        ctx.moveTo(0, y0);
        ctx.lineTo(canvas.width, y0);
        ctx.stroke();

        // Y axis (x = 0)
        const x0 = this.coordToPixel(0, 0, canvas).x;
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x0, canvas.height);
        ctx.stroke();

        // Axis labels - terminal font
        ctx.font = '10px "Share Tech Mono", monospace';
        ctx.fillStyle = 'rgba(77, 238, 234, 0.7)';
        ctx.textAlign = 'center';

        // X axis labels
        for (let x = this.range.min; x <= this.range.max; x += 4) {
            if (x === 0) continue;
            const px = this.coordToPixel(x, 0, canvas);
            ctx.fillText(x.toString(), px.x, canvas.height - 5);
        }

        // Y axis labels
        ctx.textAlign = 'right';
        for (let y = this.range.min; y <= this.range.max; y += 4) {
            if (y === 0) continue;
            const px = this.coordToPixel(0, y, canvas);
            ctx.fillText(y.toString(), 22, px.y + 4);
        }

        // Axis titles with glow
        ctx.fillStyle = '#4ff975';
        ctx.font = '11px "Share Tech Mono", monospace';
        ctx.textAlign = 'center';
        ctx.shadowColor = '#4ff975';
        ctx.shadowBlur = 5;
        ctx.fillText('x₁', canvas.width - 15, y0 - 8);
        ctx.fillText('x₂', x0 + 15, 15);
        ctx.shadowBlur = 0;
    }

    /**
     * Render all heatmaps
     */
    render() {
        for (const [metric, canvas] of Object.entries(this.canvases)) {
            if (canvas && this.contexts[metric]) {
                this.renderHeatmap(this.contexts[metric], canvas, metric);
            }
        }

        // Also update gradient visualization (shares anchor/epsilon)
        if (typeof gradientViz !== 'undefined' && gradientViz) {
            gradientViz.render();
        }
    }
}

// Initialize when DOM is ready
let heatmapViz = null;

function initHeatmapViz() {
    if (document.getElementById('heatmap-dot')) {
        heatmapViz = new HeatmapVisualization();
    }
}

// Export for external access
if (typeof window !== 'undefined') {
    window.HeatmapVisualization = HeatmapVisualization;
    window.initHeatmapViz = initHeatmapViz;
}

