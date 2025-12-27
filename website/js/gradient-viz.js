/**
 * Gradient Vector Field Visualization Module
 * Shows gradient directions for different similarity measures
 */

class GradientVisualization {
    constructor() {
        this.canvases = {
            dot: document.getElementById('gradient-dot'),
            euclidean: document.getElementById('gradient-euclidean'),
            yat: document.getElementById('gradient-yat'),
            cosine: document.getElementById('gradient-cosine')
        };
        
        this.contexts = {};
        this.anchor = [3, 4];  // Same as heatmap
        this.epsilon = 1.0;
        this.range = { min: -8, max: 8 };
        
        this.init();
    }

    init() {
        for (const [name, canvas] of Object.entries(this.canvases)) {
            if (canvas) {
                this.contexts[name] = canvas.getContext('2d');
            }
        }
        
        // Listen for anchor updates from heatmap
        if (typeof heatmapViz !== 'undefined' && heatmapViz) {
            // Sync with heatmap visualization
            const originalRender = heatmapViz.render.bind(heatmapViz);
            heatmapViz.render = () => {
                originalRender();
                this.anchor = heatmapViz.anchor;
                this.epsilon = heatmapViz.epsilon;
                this.render();
            };
        }
        
        this.render();
    }

    /**
     * Compute gradient at a point for given metric
     */
    computeGradient(metric, x, y) {
        const w = this.anchor;
        const point = [x, y];
        
        switch (metric) {
            case 'dot':
                return MathUtils.dotGradient(w);
            case 'euclidean':
                return MathUtils.euclideanGradient(w, point);
            case 'yat':
                return MathUtils.yatGradient(w, point, this.epsilon);
            case 'cosine':
                return MathUtils.cosineGradient(w, point);
            default:
                return [0, 0];
        }
    }

    /**
     * Compute scalar value for background coloring
     */
    computeValue(metric, x, y) {
        const w = this.anchor;
        const point = [x, y];
        
        switch (metric) {
            case 'dot':
                return MathUtils.dotProduct(w, point);
            case 'euclidean':
                return -MathUtils.squaredDistance(w, point); // Negative for descent
            case 'yat':
                return MathUtils.yatProduct(w, point, this.epsilon);
            case 'cosine':
                return MathUtils.cosineSimilarity(w, point);
            default:
                return 0;
        }
    }

    coordToPixel(x, y, canvas) {
        const range = this.range.max - this.range.min;
        return {
            x: ((x - this.range.min) / range) * canvas.width,
            y: ((this.range.max - y) / range) * canvas.height
        };
    }

    /**
     * Render a single gradient field
     */
    renderGradientField(ctx, canvas, metric) {
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, width, height);
        
        // Draw background gradient based on scalar value
        this.drawBackground(ctx, canvas, metric);
        
        // Draw vector field
        const gridStep = 25;
        const arrowScale = 15;
        
        // Collect all gradients for normalization
        const gradients = [];
        for (let py = gridStep / 2; py < height; py += gridStep) {
            for (let px = gridStep / 2; px < width; px += gridStep) {
                const x = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y = this.range.max - (py / height) * (this.range.max - this.range.min);
                const grad = this.computeGradient(metric, x, y);
                const magnitude = MathUtils.norm(grad);
                gradients.push({ px, py, x, y, grad, magnitude });
            }
        }
        
        // Find max magnitude for normalization
        const maxMag = Math.max(...gradients.map(g => g.magnitude), 0.001);
        
        // Draw arrows
        for (const g of gradients) {
            if (g.magnitude < 0.001) continue;
            
            // Normalize gradient for display
            const normalizedMag = g.magnitude / maxMag;
            const dx = (g.grad[0] / g.magnitude) * arrowScale * Math.sqrt(normalizedMag);
            const dy = -(g.grad[1] / g.magnitude) * arrowScale * Math.sqrt(normalizedMag);  // Y inverted
            
            // Color based on magnitude
            const color = this.getArrowColor(normalizedMag);
            
            this.drawArrow(ctx, g.px, g.py, g.px + dx, g.py + dy, color, normalizedMag);
        }
        
        // Draw anchor point
        this.drawAnchor(ctx, canvas);
    }

    /**
     * Draw faint background showing scalar field
     */
    drawBackground(ctx, canvas, metric) {
        const width = canvas.width;
        const height = canvas.height;
        const imageData = ctx.createImageData(width, height);
        const resolution = 4;
        
        // Compute values for normalization
        const values = [];
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
        
        let minVal = Infinity, maxVal = -Infinity;
        for (const v of values) {
            if (isFinite(v.value)) {
                minVal = Math.min(minVal, v.value);
                maxVal = Math.max(maxVal, v.value);
            }
        }
        
        if (!isFinite(minVal)) minVal = 0;
        if (!isFinite(maxVal)) maxVal = 1;
        if (maxVal === minVal) maxVal = minVal + 1;
        
        // Render as faint background
        for (const v of values) {
            const normalized = (v.value - minVal) / (maxVal - minVal);
            
            // Dark background with subtle gradient
            const intensity = 20 + normalized * 40;
            
            for (let dy = 0; dy < resolution && v.py + dy < height; dy++) {
                for (let dx = 0; dx < resolution && v.px + dx < width; dx++) {
                    const idx = ((v.py + dy) * width + (v.px + dx)) * 4;
                    imageData.data[idx] = intensity * 0.8;
                    imageData.data[idx + 1] = intensity * 0.6;
                    imageData.data[idx + 2] = intensity;
                    imageData.data[idx + 3] = 255;
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    /**
     * Get arrow color based on magnitude
     */
    getArrowColor(normalizedMag) {
        // Orange to yellow gradient
        const r = 255;
        const g = Math.round(140 + normalizedMag * 100);
        const b = Math.round(50 - normalizedMag * 50);
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Draw an arrow
     */
    drawArrow(ctx, fromX, fromY, toX, toY, color, magnitude) {
        const headLength = 6 + magnitude * 4;
        const angle = Math.atan2(toY - fromY, toX - fromX);
        
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1 + magnitude * 1.5;
        ctx.lineCap = 'round';
        
        // Draw line
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        
        // Draw arrowhead
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(
            toX - headLength * Math.cos(angle - Math.PI / 6),
            toY - headLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            toX - headLength * Math.cos(angle + Math.PI / 6),
            toY - headLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
    }

    /**
     * Draw the anchor point (w vector)
     */
    drawAnchor(ctx, canvas) {
        const pos = this.coordToPixel(this.anchor[0], this.anchor[1], canvas);
        
        // Glow
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 15);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.6)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 15, 0, Math.PI * 2);
        ctx.fill();
        
        // Point
        ctx.fillStyle = '#3b82f6';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Label
        ctx.font = 'bold 12px "Space Grotesk", sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'left';
        ctx.fillText('w', pos.x + 12, pos.y - 8);
    }

    /**
     * Sync anchor with heatmap
     */
    syncWithHeatmap() {
        if (typeof heatmapViz !== 'undefined' && heatmapViz) {
            this.anchor = heatmapViz.anchor;
            this.epsilon = heatmapViz.epsilon;
        }
    }

    /**
     * Render all gradient fields
     */
    render() {
        this.syncWithHeatmap();
        
        for (const [metric, canvas] of Object.entries(this.canvases)) {
            if (canvas && this.contexts[metric]) {
                this.renderGradientField(this.contexts[metric], canvas, metric);
            }
        }
    }
}

// Initialize when DOM is ready
let gradientViz = null;

function initGradientViz() {
    if (document.getElementById('gradient-dot')) {
        gradientViz = new GradientVisualization();
    }
}

// Export
if (typeof window !== 'undefined') {
    window.GradientVisualization = GradientVisualization;
    window.initGradientViz = initGradientViz;
}

