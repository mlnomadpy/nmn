/**
 * Math Utilities for NMN Website
 * Core mathematical functions for the ⵟ-product and similarity measures
 */

const MathUtils = {
    // Small constant for numerical stability
    EPSILON: 1.0,

    /**
     * Dot product of two vectors
     * @param {number[]} w - Weight vector
     * @param {number[]} x - Input vector
     * @returns {number}
     */
    dotProduct(w, x) {
        let sum = 0;
        for (let i = 0; i < w.length; i++) {
            sum += w[i] * x[i];
        }
        return sum;
    },

    /**
     * Euclidean norm (magnitude) of a vector
     * @param {number[]} v - Vector
     * @returns {number}
     */
    norm(v) {
        let sum = 0;
        for (let i = 0; i < v.length; i++) {
            sum += v[i] * v[i];
        }
        return Math.sqrt(sum);
    },

    /**
     * Squared Euclidean distance between two vectors
     * @param {number[]} w - First vector
     * @param {number[]} x - Second vector
     * @returns {number}
     */
    squaredDistance(w, x) {
        let sum = 0;
        for (let i = 0; i < w.length; i++) {
            const diff = w[i] - x[i];
            sum += diff * diff;
        }
        return sum;
    },

    /**
     * ⵟ-Product (Yat Product) - The core operation
     * ⵟ(w, x) = (w·x)² / (||w - x||² + ε)
     * @param {number[]} w - Weight vector
     * @param {number[]} x - Input vector
     * @param {number} epsilon - Stability constant (default: 1.0)
     * @returns {number}
     */
    yatProduct(w, x, epsilon = 1.0) {
        const dot = this.dotProduct(w, x);
        const sqDist = this.squaredDistance(w, x);
        return (dot * dot) / (sqDist + epsilon);
    },

    /**
     * Cosine similarity between two vectors
     * cos(θ) = (w·x) / (||w|| ||x||)
     * @param {number[]} w - First vector
     * @param {number[]} x - Second vector
     * @returns {number}
     */
    cosineSimilarity(w, x) {
        const dot = this.dotProduct(w, x);
        const normW = this.norm(w);
        const normX = this.norm(x);
        if (normW < 1e-10 || normX < 1e-10) return 0;
        return dot / (normW * normX);
    },

    /**
     * Gradient of ⵟ-product with respect to x
     * ∇_x ⵟ(w, x) for visualization
     * @param {number[]} w - Weight vector
     * @param {number[]} x - Input vector
     * @param {number} epsilon - Stability constant
     * @returns {number[]} Gradient vector
     */
    yatGradient(w, x, epsilon = 1.0) {
        const dot = this.dotProduct(w, x);
        const sqDist = this.squaredDistance(w, x);
        const denom = sqDist + epsilon;
        
        // ∂(dot²)/∂x_i = 2 * dot * w_i
        // ∂(sqDist)/∂x_i = 2 * (x_i - w_i)
        // Using quotient rule: d/dx [f/g] = (f'g - fg') / g²
        
        const grad = [];
        for (let i = 0; i < w.length; i++) {
            const fPrime = 2 * dot * w[i];  // derivative of numerator
            const gPrime = 2 * (x[i] - w[i]);  // derivative of denominator
            grad[i] = (fPrime * denom - (dot * dot) * gPrime) / (denom * denom);
        }
        return grad;
    },

    /**
     * Gradient of dot product with respect to x
     * @param {number[]} w - Weight vector
     * @returns {number[]} Gradient vector (just w)
     */
    dotGradient(w) {
        return [...w];
    },

    /**
     * Gradient of squared Euclidean distance with respect to x
     * @param {number[]} w - Weight vector
     * @param {number[]} x - Input vector
     * @returns {number[]} Gradient vector
     */
    euclideanGradient(w, x) {
        return x.map((xi, i) => 2 * (xi - w[i]));
    },

    /**
     * Gradient of cosine similarity with respect to x
     * @param {number[]} w - Weight vector
     * @param {number[]} x - Input vector
     * @returns {number[]} Gradient vector
     */
    cosineGradient(w, x) {
        const dot = this.dotProduct(w, x);
        const normW = this.norm(w);
        const normX = this.norm(x);
        
        if (normW < 1e-10 || normX < 1e-10) {
            return w.map(() => 0);
        }
        
        const grad = [];
        for (let i = 0; i < w.length; i++) {
            // ∂/∂x_i [w·x / (||w|| ||x||)]
            // = w_i/(||w|| ||x||) - (w·x) x_i / (||w|| ||x||³)
            grad[i] = w[i] / (normW * normX) - (dot * x[i]) / (normW * normX * normX * normX);
        }
        return grad;
    },

    /**
     * Normalize a gradient vector for visualization
     * @param {number[]} grad - Gradient vector
     * @param {number} maxLength - Maximum length
     * @returns {number[]} Normalized gradient
     */
    normalizeGradient(grad, maxLength = 1) {
        const magnitude = this.norm(grad);
        if (magnitude < 1e-10) return grad.map(() => 0);
        return grad.map(g => (g / magnitude) * maxLength);
    },

    /**
     * Color mapping utilities
     */
    colorMaps: {
        /**
         * Blue-White-Red diverging colormap
         * @param {number} value - Value between 0 and 1
         * @returns {string} RGB color string
         */
        diverging(value) {
            // Clamp value
            value = Math.max(0, Math.min(1, value));
            
            let r, g, b;
            if (value < 0.5) {
                // Blue to white
                const t = value * 2;
                r = Math.round(59 + t * (255 - 59));
                g = Math.round(76 + t * (255 - 76));
                b = Math.round(192 + t * (255 - 192));
            } else {
                // White to red
                const t = (value - 0.5) * 2;
                r = 255;
                g = Math.round(255 - t * (255 - 59));
                b = Math.round(255 - t * (255 - 48));
            }
            return `rgb(${r}, ${g}, ${b})`;
        },

        /**
         * Viridis-like colormap (purple to yellow)
         * @param {number} value - Value between 0 and 1
         * @returns {string} RGB color string
         */
        viridis(value) {
            value = Math.max(0, Math.min(1, value));
            
            // Simplified viridis
            const colors = [
                [68, 1, 84],      // 0.0 - dark purple
                [72, 40, 120],    // 0.2
                [62, 74, 137],    // 0.4
                [49, 104, 142],   // 0.5
                [38, 130, 142],   // 0.6
                [53, 183, 121],   // 0.8
                [253, 231, 37]    // 1.0 - yellow
            ];
            
            const idx = value * (colors.length - 1);
            const low = Math.floor(idx);
            const high = Math.min(low + 1, colors.length - 1);
            const t = idx - low;
            
            const r = Math.round(colors[low][0] + t * (colors[high][0] - colors[low][0]));
            const g = Math.round(colors[low][1] + t * (colors[high][1] - colors[low][1]));
            const b = Math.round(colors[low][2] + t * (colors[high][2] - colors[low][2]));
            
            return `rgb(${r}, ${g}, ${b})`;
        },

        /**
         * Inferno-like colormap (black to yellow via red)
         * @param {number} value - Value between 0 and 1
         * @returns {string} RGB color string
         */
        inferno(value) {
            value = Math.max(0, Math.min(1, value));
            
            const colors = [
                [0, 0, 4],        // 0.0
                [40, 11, 84],     // 0.2
                [101, 21, 110],   // 0.4
                [159, 42, 99],    // 0.5
                [212, 72, 66],    // 0.6
                [245, 125, 21],   // 0.8
                [252, 255, 164]   // 1.0
            ];
            
            const idx = value * (colors.length - 1);
            const low = Math.floor(idx);
            const high = Math.min(low + 1, colors.length - 1);
            const t = idx - low;
            
            const r = Math.round(colors[low][0] + t * (colors[high][0] - colors[low][0]));
            const g = Math.round(colors[low][1] + t * (colors[high][1] - colors[low][1]));
            const b = Math.round(colors[low][2] + t * (colors[high][2] - colors[low][2]));
            
            return `rgb(${r}, ${g}, ${b})`;
        },

        /**
         * Plasma-like colormap
         * @param {number} value - Value between 0 and 1
         * @returns {string} RGB color string
         */
        plasma(value) {
            value = Math.max(0, Math.min(1, value));
            
            const colors = [
                [13, 8, 135],     // 0.0 - deep blue
                [84, 2, 163],     // 0.2
                [139, 10, 165],   // 0.4
                [185, 50, 137],   // 0.5
                [219, 92, 104],   // 0.6
                [244, 136, 73],   // 0.8
                [240, 249, 33]    // 1.0 - yellow
            ];
            
            const idx = value * (colors.length - 1);
            const low = Math.floor(idx);
            const high = Math.min(low + 1, colors.length - 1);
            const t = idx - low;
            
            const r = Math.round(colors[low][0] + t * (colors[high][0] - colors[low][0]));
            const g = Math.round(colors[low][1] + t * (colors[high][1] - colors[low][1]));
            const b = Math.round(colors[low][2] + t * (colors[high][2] - colors[low][2]));
            
            return `rgb(${r}, ${g}, ${b})`;
        }
    },

    /**
     * Sigmoid activation function
     * @param {number} x - Input value
     * @returns {number}
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    },

    /**
     * Cross-entropy loss for binary classification
     * @param {number} pred - Predicted probability
     * @param {number} target - Target (0 or 1)
     * @returns {number}
     */
    binaryCrossEntropy(pred, target) {
        pred = Math.max(1e-7, Math.min(1 - 1e-7, pred));
        return -(target * Math.log(pred) + (1 - target) * Math.log(1 - pred));
    },

    /**
     * XOR targets
     */
    xorTargets: [
        { x: [0, 0], y: 0 },
        { x: [0, 1], y: 1 },
        { x: [1, 0], y: 1 },
        { x: [1, 1], y: 0 }
    ],

    /**
     * Calculate total XOR loss for given weights using dot product + sigmoid
     * @param {number[]} w - Weight vector [w1, w2]
     * @returns {number} Total loss
     */
    xorLossDot(w) {
        let totalLoss = 0;
        for (const sample of this.xorTargets) {
            const dot = this.dotProduct(w, sample.x);
            const pred = this.sigmoid(dot);
            totalLoss += this.binaryCrossEntropy(pred, sample.y);
        }
        return totalLoss / this.xorTargets.length;
    },

    /**
     * Calculate total XOR loss for given weights using ⵟ-product
     * @param {number[]} w - Weight vector [w1, w2]
     * @param {number} epsilon - Stability constant
     * @returns {number} Total loss
     */
    xorLossYat(w, epsilon = 0.1) {
        let totalLoss = 0;
        for (const sample of this.xorTargets) {
            const yat = this.yatProduct(w, sample.x, epsilon);
            // Normalize yat output for sigmoid
            const pred = this.sigmoid(yat - 0.5);
            totalLoss += this.binaryCrossEntropy(pred, sample.y);
        }
        return totalLoss / this.xorTargets.length;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MathUtils;
}

