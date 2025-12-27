/**
 * XOR Problem Demonstration
 * Shows how ⵟ-product solves XOR with a single neuron
 */

class XORDemo {
    constructor() {
        this.linearCanvas = document.getElementById('xor-linear');
        this.yatCanvas = document.getElementById('xor-yat');
        
        this.xorPoints = [
            { x: [0, 0], y: 0, label: '0' },
            { x: [0, 1], y: 1, label: '1' },
            { x: [1, 0], y: 1, label: '1' },
            { x: [1, 1], y: 0, label: '0' }
        ];
        
        // Optimal weight for YAT (w = [1, -1] separates XOR)
        this.yatWeight = [1, -1];
        // Linear weight (any weight fails)
        this.linearWeight = [1, 1];
        
        this.epsilon = 0.1;
        this.range = { min: -0.5, max: 1.5 };
        
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
     * Render the linear neuron visualization
     */
    renderLinear() {
        const ctx = this.linearCtx;
        const canvas = this.linearCanvas;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, width, height);
        
        // Draw decision boundary (hyperplane w·x = 0)
        // For w = [1, 1], the boundary is x + y = 0, or y = -x
        this.drawLinearBoundary(ctx, canvas);
        
        // Draw region coloring (showing failure)
        this.drawLinearRegions(ctx, canvas);
        
        // Draw grid
        this.drawGrid(ctx, canvas);
        
        // Draw XOR points
        this.drawXORPoints(ctx, canvas);
        
        // Draw weight vector
        this.drawWeightVector(ctx, canvas, this.linearWeight, '#ef4444');
    }

    /**
     * Draw linear decision boundary
     */
    drawLinearBoundary(ctx, canvas) {
        const w = this.linearWeight;
        
        // w·x = 0 means w[0]*x + w[1]*y = 0
        // y = -w[0]*x / w[1]
        
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 4]);
        
        ctx.beginPath();
        
        // Find intersections with canvas boundaries
        const points = [];
        for (const xVal of [this.range.min, this.range.max]) {
            const yVal = -w[0] * xVal / (w[1] || 0.001);
            if (yVal >= this.range.min && yVal <= this.range.max) {
                points.push({ x: xVal, y: yVal });
            }
        }
        for (const yVal of [this.range.min, this.range.max]) {
            const xVal = -w[1] * yVal / (w[0] || 0.001);
            if (xVal >= this.range.min && xVal <= this.range.max) {
                points.push({ x: xVal, y: yVal });
            }
        }
        
        if (points.length >= 2) {
            const p1 = this.coordToPixel(points[0].x, points[0].y, canvas);
            const p2 = this.coordToPixel(points[1].x, points[1].y, canvas);
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
        
        ctx.setLineDash([]);
    }

    /**
     * Draw linear regions (showing classification)
     */
    drawLinearRegions(ctx, canvas) {
        const width = canvas.width;
        const height = canvas.height;
        const resolution = 4;
        
        for (let py = 0; py < height; py += resolution) {
            for (let px = 0; px < width; px += resolution) {
                const x = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y = this.range.max - (py / height) * (this.range.max - this.range.min);
                
                const dot = MathUtils.dotProduct(this.linearWeight, [x, y]);
                const pred = MathUtils.sigmoid(dot);
                
                // Blue for < 0.5, Red for >= 0.5
                if (pred >= 0.5) {
                    ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';
                } else {
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
                }
                ctx.fillRect(px, py, resolution, resolution);
            }
        }
    }

    /**
     * Render the YAT neuron visualization
     */
    renderYAT() {
        const ctx = this.yatCtx;
        const canvas = this.yatCanvas;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, width, height);
        
        // Draw YAT decision regions
        this.drawYATRegions(ctx, canvas);
        
        // Draw grid
        this.drawGrid(ctx, canvas);
        
        // Draw decision boundary contour
        this.drawYATBoundary(ctx, canvas);
        
        // Draw XOR points
        this.drawXORPoints(ctx, canvas);
        
        // Draw weight vector
        this.drawWeightVector(ctx, canvas, this.yatWeight, '#10b981');
    }

    /**
     * Draw YAT decision regions
     */
    drawYATRegions(ctx, canvas) {
        const width = canvas.width;
        const height = canvas.height;
        const resolution = 3;
        
        // Find threshold that separates XOR
        // ⵟ([0,0]) = 0, ⵟ([0,1]) > 0, ⵟ([1,0]) > 0, ⵟ([1,1]) = 0
        const threshold = 0.1; // Small positive threshold
        
        for (let py = 0; py < height; py += resolution) {
            for (let px = 0; px < width; px += resolution) {
                const x = this.range.min + (px / width) * (this.range.max - this.range.min);
                const y = this.range.max - (py / height) * (this.range.max - this.range.min);
                
                const yatVal = MathUtils.yatProduct(this.yatWeight, [x, y], this.epsilon);
                
                // Color based on YAT value
                if (yatVal > threshold) {
                    // Predicted 1 (green region)
                    const intensity = Math.min(yatVal * 50, 255);
                    ctx.fillStyle = `rgba(16, 185, 129, ${0.1 + intensity / 1000})`;
                } else {
                    // Predicted 0 (darker region)
                    ctx.fillStyle = 'rgba(139, 92, 246, 0.08)';
                }
                ctx.fillRect(px, py, resolution, resolution);
            }
        }
    }

    /**
     * Draw YAT decision boundary (contour where ⵟ = threshold)
     */
    drawYATBoundary(ctx, canvas) {
        const threshold = 0.1;
        
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
        ctx.lineWidth = 2;
        
        // Draw contour using simple marching
        const step = 5;
        
        for (let py = 0; py < canvas.height - step; py += step) {
            for (let px = 0; px < canvas.width - step; px += step) {
                const x1 = this.range.min + (px / canvas.width) * (this.range.max - this.range.min);
                const y1 = this.range.max - (py / canvas.height) * (this.range.max - this.range.min);
                const x2 = this.range.min + ((px + step) / canvas.width) * (this.range.max - this.range.min);
                const y2 = this.range.max - ((py + step) / canvas.height) * (this.range.max - this.range.min);
                
                const v1 = MathUtils.yatProduct(this.yatWeight, [x1, y1], this.epsilon);
                const v2 = MathUtils.yatProduct(this.yatWeight, [x2, y1], this.epsilon);
                const v3 = MathUtils.yatProduct(this.yatWeight, [x1, y2], this.epsilon);
                const v4 = MathUtils.yatProduct(this.yatWeight, [x2, y2], this.epsilon);
                
                // Check for crossing
                const above1 = v1 > threshold;
                const above2 = v2 > threshold;
                const above3 = v3 > threshold;
                const above4 = v4 > threshold;
                
                if (above1 !== above2 || above1 !== above3 || above1 !== above4) {
                    // There's a boundary crossing in this cell
                    ctx.beginPath();
                    ctx.arc(px + step/2, py + step/2, 1, 0, Math.PI * 2);
                    ctx.stroke();
                }
            }
        }
    }

    /**
     * Draw coordinate grid
     */
    drawGrid(ctx, canvas) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        
        // Grid lines at 0.5 intervals
        for (let v = 0; v <= 1; v += 0.5) {
            const pos = this.coordToPixel(v, v, canvas);
            
            // Vertical line
            ctx.beginPath();
            ctx.moveTo(pos.x, 0);
            ctx.lineTo(pos.x, canvas.height);
            ctx.stroke();
            
            // Horizontal line
            const pos2 = this.coordToPixel(0, v, canvas);
            ctx.beginPath();
            ctx.moveTo(0, pos2.y);
            ctx.lineTo(canvas.width, pos2.y);
            ctx.stroke();
        }
        
        // Axis labels
        ctx.font = '11px "Space Grotesk", sans-serif';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.textAlign = 'center';
        
        for (let v = 0; v <= 1; v += 0.5) {
            const pos = this.coordToPixel(v, 0, canvas);
            ctx.fillText(v.toString(), pos.x, canvas.height - 8);
        }
        
        ctx.textAlign = 'right';
        for (let v = 0; v <= 1; v += 0.5) {
            const pos = this.coordToPixel(0, v, canvas);
            ctx.fillText(v.toString(), 20, pos.y + 4);
        }
    }

    /**
     * Draw XOR data points
     */
    drawXORPoints(ctx, canvas) {
        for (const point of this.xorPoints) {
            const pos = this.coordToPixel(point.x[0], point.x[1], canvas);
            
            // Outer glow
            const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 25);
            if (point.y === 1) {
                gradient.addColorStop(0, 'rgba(251, 191, 36, 0.4)');
                gradient.addColorStop(1, 'rgba(251, 191, 36, 0)');
            } else {
                gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
                gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
            }
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 25, 0, Math.PI * 2);
            ctx.fill();
            
            // Point
            ctx.fillStyle = point.y === 1 ? '#fbbf24' : '#6366f1';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            
            // Label
            ctx.font = 'bold 14px "Space Grotesk", sans-serif';
            ctx.fillStyle = point.y === 1 ? '#000' : '#fff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(point.label, pos.x, pos.y);
        }
    }

    /**
     * Draw weight vector
     */
    drawWeightVector(ctx, canvas, w, color) {
        const origin = this.coordToPixel(0.5, 0.5, canvas);
        const scale = 80;
        const endX = origin.x + w[0] * scale;
        const endY = origin.y - w[1] * scale;  // Y inverted
        
        // Arrow
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 3;
        
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Arrowhead
        const angle = Math.atan2(endY - origin.y, endX - origin.x);
        const headLength = 12;
        
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
            endX - headLength * Math.cos(angle - Math.PI / 6),
            endY - headLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            endX - headLength * Math.cos(angle + Math.PI / 6),
            endY - headLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
        
        // Label
        ctx.font = 'bold 13px "Space Grotesk", sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'left';
        ctx.fillText(`w=[${w[0]}, ${w[1]}]`, endX + 10, endY - 5);
    }

    render() {
        if (this.linearCtx) this.renderLinear();
        if (this.yatCtx) this.renderYAT();
    }
}

// Initialize
let xorDemo = null;

function initXORDemo() {
    if (document.getElementById('xor-linear') && document.getElementById('xor-yat')) {
        xorDemo = new XORDemo();
    }
}

// Export
if (typeof window !== 'undefined') {
    window.XORDemo = XORDemo;
    window.initXORDemo = initXORDemo;
}

