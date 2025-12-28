/**
 * 3D Loss Landscape Visualization using Three.js
 * Compares loss surfaces for dot product vs ⵟ-product on XOR problem
 */

class LossLandscapeViz {
    constructor() {
        this.dotContainer = document.getElementById('loss-3d-dot');
        this.yatContainer = document.getElementById('loss-3d-yat');

        this.range = { min: -8, max: 8 };
        this.resolution = 60;

        if (this.dotContainer && this.yatContainer && typeof THREE !== 'undefined') {
            this.initDot();
            this.initYat();
            this.animate();
        }
    }

    /**
     * Compute XOR loss for dot product + sigmoid
     */
    computeDotLoss(w1, w2) {
        const w = [w1, w2];
        let totalLoss = 0;

        for (const sample of MathUtils.xorTargets) {
            const dot = MathUtils.dotProduct(w, sample.x);
            const pred = MathUtils.sigmoid(dot);
            totalLoss += MathUtils.binaryCrossEntropy(pred, sample.y);
        }
        return totalLoss / 4;
    }

    /**
     * Compute XOR loss for ⵟ-product
     */
    computeYatLoss(w1, w2) {
        const w = [w1, w2];
        const epsilon = 0.1;
        let totalLoss = 0;

        for (const sample of MathUtils.xorTargets) {
            const yat = MathUtils.yatProduct(w, sample.x, epsilon);
            // Use sigmoid with proper scaling for YAT output
            const pred = MathUtils.sigmoid((yat - 0.5) * 2);
            totalLoss += MathUtils.binaryCrossEntropy(pred, sample.y);
        }
        return totalLoss / 4;
    }

    /**
     * Initialize dot product loss surface
     */
    initDot() {
        const width = this.dotContainer.clientWidth;
        const height = this.dotContainer.clientHeight;

        // Scene
        this.dotScene = new THREE.Scene();
        this.dotScene.background = new THREE.Color(0x0d0d12);

        // Camera
        this.dotCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.dotCamera.position.set(15, 12, 15);
        this.dotCamera.lookAt(0, 0, 0);

        // Renderer
        this.dotRenderer = new THREE.WebGLRenderer({ antialias: true });
        this.dotRenderer.setSize(width, height);
        this.dotRenderer.setPixelRatio(window.devicePixelRatio);
        this.dotContainer.appendChild(this.dotRenderer.domElement);

        // Create surface
        this.createSurface(this.dotScene, 'dot');

        // Add axes
        this.addAxes(this.dotScene);

        // Add lighting
        this.addLighting(this.dotScene);

        // Mouse interaction
        this.setupMouseControls(this.dotRenderer.domElement, this.dotCamera, 'dot');
    }

    /**
     * Initialize YAT loss surface
     */
    initYat() {
        const width = this.yatContainer.clientWidth;
        const height = this.yatContainer.clientHeight;

        // Scene
        this.yatScene = new THREE.Scene();
        this.yatScene.background = new THREE.Color(0x0d0d12);

        // Camera
        this.yatCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.yatCamera.position.set(15, 12, 15);
        this.yatCamera.lookAt(0, 0, 0);

        // Renderer
        this.yatRenderer = new THREE.WebGLRenderer({ antialias: true });
        this.yatRenderer.setSize(width, height);
        this.yatRenderer.setPixelRatio(window.devicePixelRatio);
        this.yatContainer.appendChild(this.yatRenderer.domElement);

        // Create surface
        this.createSurface(this.yatScene, 'yat');

        // Add axes
        this.addAxes(this.yatScene);

        // Add lighting
        this.addLighting(this.yatScene);

        // Mouse interaction
        this.setupMouseControls(this.yatRenderer.domElement, this.yatCamera, 'yat');
    }

    /**
     * Create loss surface geometry
     */
    createSurface(scene, type) {
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        const colors = [];
        const indices = [];

        const size = this.resolution;
        const scale = 8 / size;

        // Generate vertices
        let maxLoss = 0;
        let minLoss = Infinity;
        const lossValues = [];

        for (let i = 0; i <= size; i++) {
            for (let j = 0; j <= size; j++) {
                const w1 = -8 + i * 16 / size;
                const w2 = -8 + j * 16 / size;

                let loss;
                if (type === 'dot') {
                    loss = this.computeDotLoss(w1, w2);
                } else {
                    loss = this.computeYatLoss(w1, w2);
                }

                // Clamp extreme values
                loss = Math.min(loss, 5);

                lossValues.push({ w1, w2, loss });
                maxLoss = Math.max(maxLoss, loss);
                minLoss = Math.min(minLoss, loss);
            }
        }

        // Normalize and create vertices
        const lossRange = maxLoss - minLoss || 1;

        for (const v of lossValues) {
            const x = v.w1;
            const z = v.w2;
            const y = ((v.loss - minLoss) / lossRange) * 6;  // Scale height

            vertices.push(x, y, z);

            // Color based on height (viridis-like)
            const t = (v.loss - minLoss) / lossRange;
            const color = this.getViridisColor(t);
            colors.push(color.r, color.g, color.b);
        }

        // Generate indices for triangles
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const a = i * (size + 1) + j;
                const b = a + 1;
                const c = a + (size + 1);
                const d = c + 1;

                indices.push(a, b, c);
                indices.push(b, d, c);
            }
        }

        geometry.setIndex(indices);
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeVertexNormals();

        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            shininess: 30,
            flatShading: false
        });

        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        // Add wireframe
        const wireGeometry = new THREE.WireframeGeometry(geometry);
        const wireMaterial = new THREE.LineBasicMaterial({
            color: 0xffffff,
            opacity: 0.1,
            transparent: true
        });
        const wireframe = new THREE.LineSegments(wireGeometry, wireMaterial);
        scene.add(wireframe);
    }

    /**
     * Get terminal-themed color (dark to green to cyan to yellow)
     */
    getViridisColor(t) {
        t = Math.max(0, Math.min(1, t));

        // Terminal green theme
        const colors = [
            { r: 0.02, g: 0.06, b: 0.04 },   // 0.0 - almost black green
            { r: 0.08, g: 0.20, b: 0.15 },   // 0.25
            { r: 0.20, g: 0.50, b: 0.35 },   // 0.5
            { r: 0.31, g: 0.98, b: 0.46 },   // 0.75 - terminal green
            { r: 0.98, g: 0.84, b: 0.11 }    // 1.0 - terminal yellow
        ];

        const idx = t * (colors.length - 1);
        const low = Math.floor(idx);
        const high = Math.min(low + 1, colors.length - 1);
        const frac = idx - low;

        return {
            r: colors[low].r + frac * (colors[high].r - colors[low].r),
            g: colors[low].g + frac * (colors[high].g - colors[low].g),
            b: colors[low].b + frac * (colors[high].b - colors[low].b)
        };
    }

    /**
     * Add axes to scene - Terminal Style
     */
    addAxes(scene) {
        // X axis (w1) - terminal magenta
        const xGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-9, 0, 0),
            new THREE.Vector3(9, 0, 0)
        ]);
        const xMat = new THREE.LineBasicMaterial({ color: 0xff4f9a, opacity: 0.7, transparent: true });
        scene.add(new THREE.Line(xGeom, xMat));

        // Z axis (w2) - terminal cyan
        const zGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, -9),
            new THREE.Vector3(0, 0, 9)
        ]);
        const zMat = new THREE.LineBasicMaterial({ color: 0x4deeea, opacity: 0.7, transparent: true });
        scene.add(new THREE.Line(zGeom, zMat));

        // Y axis (loss) - terminal yellow
        const yGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 7, 0)
        ]);
        const yMat = new THREE.LineBasicMaterial({ color: 0xf9d71c, opacity: 0.7, transparent: true });
        scene.add(new THREE.Line(yGeom, yMat));

        // Grid on XZ plane - terminal green
        const gridHelper = new THREE.GridHelper(16, 16, 0x2a3a30, 0x1a2a20);
        gridHelper.position.y = 0;
        scene.add(gridHelper);
    }

    /**
     * Add lighting
     */
    addLighting(scene) {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        scene.add(directionalLight);

        const directionalLight2 = new THREE.DirectionalLight(0x8888ff, 0.3);
        directionalLight2.position.set(-10, 10, -10);
        scene.add(directionalLight2);
    }

    /**
     * Setup mouse controls for rotation
     */
    setupMouseControls(domElement, camera, type) {
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };

        const rotation = { x: 0, y: 0 };
        const distance = 22;

        domElement.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        domElement.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            rotation.y += deltaX * 0.01;
            rotation.x += deltaY * 0.01;
            rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotation.x));

            camera.position.x = distance * Math.sin(rotation.y) * Math.cos(rotation.x);
            camera.position.y = distance * Math.sin(rotation.x) + 5;
            camera.position.z = distance * Math.cos(rotation.y) * Math.cos(rotation.x);
            camera.lookAt(0, 2, 0);

            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        domElement.addEventListener('mouseup', () => {
            isDragging = false;
        });

        domElement.addEventListener('mouseleave', () => {
            isDragging = false;
        });

        // Touch events
        domElement.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                isDragging = true;
                previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
            }
        });

        domElement.addEventListener('touchmove', (e) => {
            if (!isDragging || e.touches.length !== 1) return;
            e.preventDefault();

            const touch = e.touches[0];
            const deltaX = touch.clientX - previousMousePosition.x;
            const deltaY = touch.clientY - previousMousePosition.y;

            rotation.y += deltaX * 0.01;
            rotation.x += deltaY * 0.01;
            rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotation.x));

            camera.position.x = distance * Math.sin(rotation.y) * Math.cos(rotation.x);
            camera.position.y = distance * Math.sin(rotation.x) + 5;
            camera.position.z = distance * Math.cos(rotation.y) * Math.cos(rotation.x);
            camera.lookAt(0, 2, 0);

            previousMousePosition = { x: touch.clientX, y: touch.clientY };
        });

        domElement.addEventListener('touchend', () => {
            isDragging = false;
        });
    }

    /**
     * Animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.dotRenderer && this.dotScene && this.dotCamera) {
            this.dotRenderer.render(this.dotScene, this.dotCamera);
        }

        if (this.yatRenderer && this.yatScene && this.yatCamera) {
            this.yatRenderer.render(this.yatScene, this.yatCamera);
        }
    }
}

// Initialize
let lossLandscapeViz = null;

function initLossLandscapeViz() {
    if (document.getElementById('loss-3d-dot') && document.getElementById('loss-3d-yat')) {
        lossLandscapeViz = new LossLandscapeViz();
    }
}

// Export
if (typeof window !== 'undefined') {
    window.LossLandscapeViz = LossLandscapeViz;
    window.initLossLandscapeViz = initLossLandscapeViz;
}

