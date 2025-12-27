/**
 * Decision Boundary Comparison
 * Shows linear vs ⵟ-product decision boundaries with multiple prototypes
 */

class DecisionBoundaryViz {
    constructor() {
        this.linearCanvas = document.getElementById('boundary-linear');
        this.yatCanvas = document.getElementById('boundary-yat');
        
        // Multiple prototypes (class centers)
        this.prototypes = [
            { w: [-0.6, 0.7], color: '#ef4444' },   // Red
            { w: [0.5, 0.8], color: '#fbbf24' },    // Yellow
            { w: [-0.4, -0.5], color: '#22c55e' },  // Green
            { w: [0.7, -0.3], color: '#3b82f6' },   // Blue
            { w: [0.1, 0.2], color: '#a855f7' },    // Purple
            { w: [-0.7, -0.1], color: '#06b6d4' },  // Cyan
            { w: [0.3, -0.7], color: '#f97316' },   // Orange
            { w: [-0.2, 0.3], color: '#ec4899' },   // Pink
        ];
        
        this.epsilon = 0.1;
        this.range = { min: -1, max: 1 };
        
        if (this.linearCanvas && this.yatCanvas) {
            this.linearCtx = this.linearCanvas.getContext('2d');
            this.yatCtx = this.yatCanvas.getContext('2d');
            this.render();
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
     * Find winning prototype using linear dot product
     */
    linearArgmax(x, y) {
        let maxVal = -Infinity;
        let maxIdx = 0;
        
        for (let i = 0; i < this.prototypes.length; i++) {
            const dot = MathUtils.dotProduct(this.prototypes[i].w, [x, y]);
            if (dot > maxVal) {
                maxVal = dot;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Find winning prototype using ⵟ-product
     */
    yatArgmax(x, y) {
        let maxVal = -Infinity;
        let maxIdx = 0;
        
        for (let i = 0; i < this.prototypes.length; i++) {
            const yat = MathUtils.yatProduct(this.prototypes[i].w, [x, y], this.epsilon);
            if (yat > maxVal) {
                maxVal = yat;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Parse color to RGB
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }

    /**
     * Render linear decision boundaries
     */
    renderLinear() {
        const ctx = this.linearCtx;
        const canvas = this.linearCanvas;
        const width = canvas.width;
        const height = canvas.height;
        const resolution = 2;
        
        // Clear
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, width, height);
        
        // Draw regions
        const imageData = ctx.createImageData(width, height);
        
        for (let py = 0; py < height; py += resolution) {
            for (let px = 0; px < width; px += resolution) {
                const x = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y = this.range.max - (py / height) * (this.range.max - this.range.min);
                
                const winnerIdx = this.linearArgmax(x, y);
                const rgb = this.hexToRgb(this.prototypes[winnerIdx].color);
                
                for (let dy = 0; dy < resolution && py + dy < height; dy++) {
                    for (let dx = 0; dx < resolution && px + dx < width; dx++) {
                        const idx = ((py + dy) * width + (px + dx)) * 4;
                        imageData.data[idx] = rgb.r;
                        imageData.data[idx + 1] = rgb.g;
                        imageData.data[idx + 2] = rgb.b;
                        imageData.data[idx + 3] = 120;  // Semi-transparent
                    }
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        // Draw grid
        this.drawGrid(ctx, canvas);
        
        // Draw decision boundaries (where argmax changes)
        this.drawBoundaries(ctx, canvas, 'linear');
        
        // Draw prototypes as stars
        this.drawPrototypes(ctx, canvas);
    }

    /**
     * Render YAT decision boundaries
     */
    renderYAT() {
        const ctx = this.yatCtx;
        const canvas = this.yatCanvas;
        const width = canvas.width;
        const height = canvas.height;
        const resolution = 2;
        
        // Clear
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, width, height);
        
        // Draw regions
        const imageData = ctx.createImageData(width, height);
        
        for (let py = 0; py < height; py += resolution) {
            for (let px = 0; px < width; px += resolution) {
                const x = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y = this.range.max - (py / height) * (this.range.max - this.range.min);
                
                const winnerIdx = this.yatArgmax(x, y);
                const rgb = this.hexToRgb(this.prototypes[winnerIdx].color);
                
                // Also get the winning value for intensity modulation
                const yatVal = MathUtils.yatProduct(this.prototypes[winnerIdx].w, [x, y], this.epsilon);
                const intensity = Math.min(100 + yatVal * 20, 180);
                
                for (let dy = 0; dy < resolution && py + dy < height; dy++) {
                    for (let dx = 0; dx < resolution && px + dx < width; dx++) {
                        const idx = ((py + dy) * width + (px + dx)) * 4;
                        imageData.data[idx] = rgb.r;
                        imageData.data[idx + 1] = rgb.g;
                        imageData.data[idx + 2] = rgb.b;
                        imageData.data[idx + 3] = intensity;
                    }
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        // Draw grid
        this.drawGrid(ctx, canvas);
        
        // Draw decision boundaries
        this.drawBoundaries(ctx, canvas, 'yat');
        
        // Draw prototypes as stars
        this.drawPrototypes(ctx, canvas);
    }

    /**
     * Draw coordinate grid
     */
    drawGrid(ctx, canvas) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1;
        
        // Grid at 0.5 intervals
        for (let v = -1; v <= 1; v += 0.5) {
            const pos = this.coordToPixel(v, 0, canvas);
            
            ctx.beginPath();
            ctx.moveTo(pos.x, 0);
            ctx.lineTo(pos.x, canvas.height);
            ctx.stroke();
            
            const pos2 = this.coordToPixel(0, v, canvas);
            ctx.beginPath();
            ctx.moveTo(0, pos2.y);
            ctx.lineTo(canvas.width, pos2.y);
            ctx.stroke();
        }
    }

    /**
     * Draw decision boundaries (edges between regions)
     */
    drawBoundaries(ctx, canvas, method) {
        const step = 3;
        
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        
        for (let py = 0; py < canvas.height - step; py += step) {
            for (let px = 0; px < canvas.width - step; px += step) {
                const x1 = this.range.min + (px / canvas.width) * (this.range.max - this.range.min);
                const y1 = this.range.max - (py / canvas.height) * (this.range.max - this.range.min);
                const x2 = this.range.min + ((px + step) / canvas.width) * (this.range.max - this.range.min);
                const y2 = this.range.max - ((py + step) / canvas.height) * (this.range.max - this.range.min);
                
                let w1, w2, w3, w4;
                if (method === 'linear') {
                    w1 = this.linearArgmax(x1, y1);
                    w2 = this.linearArgmax(x2, y1);
                    w3 = this.linearArgmax(x1, y2);
                    w4 = this.linearArgmax(x2, y2);
                } else {
                    w1 = this.yatArgmax(x1, y1);
                    w2 = this.yatArgmax(x2, y1);
                    w3 = this.yatArgmax(x1, y2);
                    w4 = this.yatArgmax(x2, y2);
                }
                
                // If not all corners have the same winner, there's a boundary
                if (w1 !== w2 || w1 !== w3 || w1 !== w4) {
                    ctx.fillRect(px, py, step, step);
                }
            }
        }
    }

    /**
     * Draw prototype points as stars
     */
    drawPrototypes(ctx, canvas) {
        for (let i = 0; i < this.prototypes.length; i++) {
            const proto = this.prototypes[i];
            const pos = this.coordToPixel(proto.w[0], proto.w[1], canvas);
            
            // Glow
            const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 20);
            gradient.addColorStop(0, proto.color + '80');
            gradient.addColorStop(1, proto.color + '00');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
            ctx.fill();
            
            // Star
            ctx.fillStyle = proto.color;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            this.drawStar(ctx, pos.x, pos.y, 5, 12, 6);
        }
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

    render() {
        if (this.linearCtx) this.renderLinear();
        if (this.yatCtx) this.renderYAT();
    }
}

// Initialize
let decisionBoundaryViz = null;

function initDecisionBoundaryViz() {
    if (document.getElementById('boundary-linear') && document.getElementById('boundary-yat')) {
        decisionBoundaryViz = new DecisionBoundaryViz();
    }
}

// Export
if (typeof window !== 'undefined') {
    window.DecisionBoundaryViz = DecisionBoundaryViz;
    window.initDecisionBoundaryViz = initDecisionBoundaryViz;
}

