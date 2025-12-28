/**
 * Level Set Curves Visualization Module
 * Interactive contour/level set visualizations for similarity measures
 */

class LevelSetVisualization {
    constructor() {
        this.canvases = {
            dot: document.getElementById('levelset-dot'),
            euclidean: document.getElementById('levelset-euclidean'),
            yat: document.getElementById('levelset-yat'),
            cosine: document.getElementById('levelset-cosine')
        };

        this.contexts = {};
        this.range = { min: -10, max: 10 };
        this.numContours = 12;

        // Colors for contour levels (gradient from cool to warm)
        this.contourColors = [
            '#4c1d95', '#5b21b6', '#7c3aed', '#8b5cf6',
            '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe',
            '#fecaca', '#fca5a5', '#f87171', '#ef4444'
        ];

        this.init();
    }

    init() {
        for (const [metric, canvas] of Object.entries(this.canvases)) {
            if (canvas) {
                this.contexts[metric] = canvas.getContext('2d');
            }
        }
    }

    // Get anchor and epsilon from heatmap viz (shared state)
    getSharedState() {
        if (typeof heatmapViz !== 'undefined' && heatmapViz) {
            return {
                anchor: heatmapViz.anchor,
                epsilon: heatmapViz.epsilon
            };
        }
        return {
            anchor: [3, 4],
            epsilon: 1.0
        };
    }

    coordToPixel(x, y, canvas) {
        const range = this.range.max - this.range.min;
        return {
            x: ((x - this.range.min) / range) * canvas.width,
            y: ((this.range.max - y) / range) * canvas.height
        };
    }

    /**
     * Compute value at a point for given metric
     */
    computeValue(metric, x, y) {
        const { anchor, epsilon } = this.getSharedState();
        const w = anchor;
        const xVec = [x, y];

        switch (metric) {
            case 'dot':
                return MathUtils.dotProduct(w, xVec);
            case 'euclidean':
                return MathUtils.squaredDistance(w, xVec);
            case 'yat':
                return MathUtils.yatProduct(w, xVec, epsilon);
            case 'cosine':
                return MathUtils.cosineSimilarity(w, xVec);
            default:
                return 0;
        }
    }

    /**
     * Render level set curves for a metric
     */
    renderLevelSet(ctx, canvas, metric) {
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        this.drawGrid(ctx, canvas);

        // Sample values to determine range
        const resolution = 50;
        let minVal = Infinity;
        let maxVal = -Infinity;
        const values = [];

        for (let py = 0; py <= height; py += height / resolution) {
            const row = [];
            for (let px = 0; px <= width; px += width / resolution) {
                const x = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y = this.range.max - (py / height) * (this.range.max - this.range.min);
                const val = this.computeValue(metric, x, y);
                row.push(val);
                if (isFinite(val)) {
                    minVal = Math.min(minVal, val);
                    maxVal = Math.max(maxVal, val);
                }
            }
            values.push(row);
        }

        // Normalize range
        if (!isFinite(minVal)) minVal = 0;
        if (!isFinite(maxVal)) maxVal = 1;
        if (minVal === maxVal) maxVal = minVal + 1;

        // Draw contour lines using marching squares
        for (let i = 1; i <= this.numContours; i++) {
            const level = minVal + (i / (this.numContours + 1)) * (maxVal - minVal);
            const colorIndex = Math.floor((i - 1) / this.numContours * this.contourColors.length);
            const color = this.contourColors[Math.min(colorIndex, this.contourColors.length - 1)];

            this.drawContourLine(ctx, canvas, metric, level, color);
        }

        // Draw anchor point
        this.drawAnchor(ctx, canvas);

        // Draw axes
        this.drawAxes(ctx, canvas);
    }

    /**
     * Draw contour line using marching squares algorithm
     */
    drawContourLine(ctx, canvas, metric, level, color) {
        const step = 6;
        const width = canvas.width;
        const height = canvas.height;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Marching squares with linear interpolation
        for (let py = 0; py < height - step; py += step) {
            for (let px = 0; px < width - step; px += step) {
                // Get corner values
                const x0 = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y0 = this.range.max - (py / height) * (this.range.max - this.range.min);
                const x1 = this.range.min + ((px + step) / width) * (this.range.max - this.range.min);
                const y1 = this.range.max - ((py + step) / height) * (this.range.max - this.range.min);

                const v00 = this.computeValue(metric, x0, y0);
                const v10 = this.computeValue(metric, x1, y0);
                const v01 = this.computeValue(metric, x0, y1);
                const v11 = this.computeValue(metric, x1, y1);

                // Marching squares case
                let code = 0;
                if (v00 >= level) code |= 1;
                if (v10 >= level) code |= 2;
                if (v01 >= level) code |= 4;
                if (v11 >= level) code |= 8;

                // Skip if all same
                if (code === 0 || code === 15) continue;

                // Interpolation helper
                const lerp = (a, b, va, vb) => {
                    if (Math.abs(vb - va) < 1e-10) return 0.5;
                    return (level - va) / (vb - va);
                };

                // Edge midpoints
                const top = px + step * lerp(x0, x1, v00, v10);
                const bottom = px + step * lerp(x0, x1, v01, v11);
                const left = py + step * lerp(y0, y1, v00, v01);
                const right = py + step * lerp(y0, y1, v10, v11);

                ctx.beginPath();

                // Draw line segments based on marching squares case
                switch (code) {
                    case 1: case 14:
                        ctx.moveTo(px, left);
                        ctx.lineTo(top, py);
                        break;
                    case 2: case 13:
                        ctx.moveTo(top, py);
                        ctx.lineTo(px + step, right);
                        break;
                    case 3: case 12:
                        ctx.moveTo(px, left);
                        ctx.lineTo(px + step, right);
                        break;
                    case 4: case 11:
                        ctx.moveTo(px, left);
                        ctx.lineTo(bottom, py + step);
                        break;
                    case 5: case 10:
                        ctx.moveTo(px, left);
                        ctx.lineTo(top, py);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(bottom, py + step);
                        ctx.lineTo(px + step, right);
                        break;
                    case 6: case 9:
                        ctx.moveTo(top, py);
                        ctx.lineTo(bottom, py + step);
                        break;
                    case 7: case 8:
                        ctx.moveTo(bottom, py + step);
                        ctx.lineTo(px + step, right);
                        break;
                }

                ctx.stroke();
            }
        }
    }

    /**
     * Draw background grid
     */
    drawGrid(ctx, canvas) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;

        const gridStep = canvas.width / 10;

        for (let i = 0; i <= 10; i++) {
            const pos = i * gridStep;
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, canvas.height);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(canvas.width, pos);
            ctx.stroke();
        }
    }

    /**
     * Draw coordinate axes
     */
    drawAxes(ctx, canvas) {
        const origin = this.coordToPixel(0, 0, canvas);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;

        // X axis
        ctx.beginPath();
        ctx.moveTo(0, origin.y);
        ctx.lineTo(canvas.width, origin.y);
        ctx.stroke();

        // Y axis
        ctx.beginPath();
        ctx.moveTo(origin.x, 0);
        ctx.lineTo(origin.x, canvas.height);
        ctx.stroke();
    }

    /**
     * Draw the anchor point
     */
    drawAnchor(ctx, canvas) {
        const { anchor } = this.getSharedState();
        const pos = this.coordToPixel(anchor[0], anchor[1], canvas);

        // Glow
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 20);
        gradient.addColorStop(0, 'rgba(255, 215, 0, 0.5)');
        gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
        ctx.fill();

        // Star
        ctx.fillStyle = '#ffd700';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        this.drawStar(ctx, pos.x, pos.y, 5, 8, 4);
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
     * Render all level set visualizations
     */
    render() {
        for (const [metric, canvas] of Object.entries(this.canvases)) {
            if (canvas && this.contexts[metric]) {
                this.renderLevelSet(this.contexts[metric], canvas, metric);
            }
        }
    }
}

// Initialize when DOM is ready
let levelSetViz = null;

function initLevelSetViz() {
    if (document.getElementById('levelset-dot')) {
        levelSetViz = new LevelSetVisualization();
        levelSetViz.render();
    }
}

// Export for external access
if (typeof window !== 'undefined') {
    window.LevelSetVisualization = LevelSetVisualization;
    window.initLevelSetViz = initLevelSetViz;
}
