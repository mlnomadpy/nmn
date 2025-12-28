/**
 * Topological Distortion by Activations Visualization
 * Shows how manifolds are distorted by different activation functions
 * Supports multiple manifold types and activation functions
 */

class TopologicalDistortionViz {
    constructor() {
        this.containers = {
            original: document.getElementById('manifold-original'),
            activated: document.getElementById('manifold-activated')
        };

        this.scenes = {};
        this.cameras = {};
        this.renderers = {};
        this.meshes = {};

        this.range = { min: -2, max: 2 };
        this.resolution = 40;
        this.animationId = null;
        this.time = 0;

        // Current settings
        this.manifoldType = 'swiss-roll';
        this.activationType = 'relu';

        // Terminal theme colors
        this.colors = {
            primary: 0x4ff975,
            secondary: 0x4deeea,
            accent: 0xf9d71c,
            surface: 0x2a3a30,
            wireframe: 0x4ff975,
            background: 0x050a08
        };

        // Activation function descriptions - note which preserve topology
        this.activationInfo = {
            'relu': { title: 'After ReLU', desc: '⚠️ Creates sharp folds - discontinuous gradient at zero', preserves: false },
            'sigmoid': { title: 'After Sigmoid', desc: '✓ Homeomorphic - stretches but preserves topology', preserves: true },
            'tanh': { title: 'After Tanh', desc: '✓ Homeomorphic - stretches but preserves topology', preserves: true },
            'leaky-relu': { title: 'After Leaky ReLU', desc: '⚠️ Kink at zero - continuous but not smooth', preserves: false },
            'elu': { title: 'After ELU', desc: '✓ Smooth everywhere - nearly preserves topology', preserves: true },
            'softplus': { title: 'After Softplus', desc: '✓ Smooth approximation of ReLU - preserves topology', preserves: true }
        };

        this.init();
    }

    init() {
        if (typeof THREE === 'undefined') {
            console.error('TopologicalDistortionViz: THREE.js not loaded');
            return;
        }

        let hasValidContainer = false;
        for (const [name, container] of Object.entries(this.containers)) {
            if (container) {
                hasValidContainer = true;
                this.initScene(name, container);
            }
        }

        if (!hasValidContainer) {
            console.error('TopologicalDistortionViz: No valid containers found');
            return;
        }

        this.createManifolds();
        this.animate();
        this.setupControls();

        window.addEventListener('resize', () => this.onResize());
    }

    setupControls() {
        const manifoldSelect = document.getElementById('manifold-type-select');
        const activationSelect = document.getElementById('activation-select');

        if (manifoldSelect) {
            manifoldSelect.addEventListener('change', (e) => {
                this.manifoldType = e.target.value;
                this.recreateManifolds();
            });
        }

        if (activationSelect) {
            activationSelect.addEventListener('change', (e) => {
                this.activationType = e.target.value;
                this.updateActivationInfo();
                this.recreateActivatedManifold();
            });
        }
    }

    updateActivationInfo() {
        const titleEl = document.getElementById('activation-title');
        const descEl = document.getElementById('activation-desc');
        const info = this.activationInfo[this.activationType];

        if (titleEl && info) titleEl.textContent = info.title;
        if (descEl && info) descEl.textContent = info.desc;
    }

    initScene(name, container) {
        let width = container.clientWidth;
        let height = container.clientHeight;

        if (width === 0) width = 400;
        if (height === 0) height = 320;

        try {
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(this.colors.background);
            this.scenes[name] = scene;

            const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(4, 3, 4);
            camera.lookAt(0, 0, 0);
            this.cameras[name] = camera;

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

            renderer.domElement.style.width = '100%';
            renderer.domElement.style.height = '100%';
            renderer.domElement.style.display = 'block';

            container.appendChild(renderer.domElement);
            this.renderers[name] = renderer;

            this.addLighting(scene);
            this.addGrid(scene);
            this.addAxes(scene);
            this.setupMouseControls(container, camera, name);
        } catch (e) {
            console.error('TopologicalDistortionViz: Error initializing scene', name, e);
        }
    }

    addLighting(scene) {
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambient);

        const directional = new THREE.DirectionalLight(0x4ff975, 0.8);
        directional.position.set(5, 10, 7);
        scene.add(directional);

        const secondary = new THREE.DirectionalLight(0x4deeea, 0.4);
        secondary.position.set(-5, 5, -5);
        scene.add(secondary);
    }

    addGrid(scene) {
        const gridHelper = new THREE.GridHelper(6, 12, 0x1a2a20, 0x0f1a15);
        gridHelper.position.y = -1.5;
        scene.add(gridHelper);
    }

    addAxes(scene) {
        const axesLength = 2.5;

        const xGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-axesLength, 0, 0),
            new THREE.Vector3(axesLength, 0, 0)
        ]);
        const xMat = new THREE.LineBasicMaterial({ color: 0xff4f9a, opacity: 0.7, transparent: true });
        scene.add(new THREE.Line(xGeom, xMat));

        const yGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, -1.5, 0),
            new THREE.Vector3(0, 2, 0)
        ]);
        const yMat = new THREE.LineBasicMaterial({ color: 0xf9d71c, opacity: 0.7, transparent: true });
        scene.add(new THREE.Line(yGeom, yMat));

        const zGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, -axesLength),
            new THREE.Vector3(0, 0, axesLength)
        ]);
        const zMat = new THREE.LineBasicMaterial({ color: 0x4deeea, opacity: 0.7, transparent: true });
        scene.add(new THREE.Line(zGeom, zMat));
    }

    // Activation functions - mathematically correct implementations
    applyActivation(value) {
        switch (this.activationType) {
            case 'relu':
                // Non-homeomorphic: creates fold at 0
                return Math.max(0, value);
            case 'sigmoid':
                // Homeomorphic: continuous bijection (0,1), preserves topology
                return 2 * (1 / (1 + Math.exp(-value * 2))) - 1; // Centered sigmoid
            case 'tanh':
                // Homeomorphic: continuous bijection (-1,1), preserves topology
                return Math.tanh(value);
            case 'leaky-relu':
                // Almost homeomorphic but has a kink (discontinuous derivative)
                return value > 0 ? value : 0.2 * value;
            case 'elu':
                // Smooth everywhere - nearly homeomorphic
                const alpha = 1.0;
                return value > 0 ? value : alpha * (Math.exp(value) - 1);
            case 'softplus':
                // Smooth approximation of ReLU - homeomorphic
                return Math.log(1 + Math.exp(value));
            default:
                return value;
        }
    }

    // Generate manifold point based on type
    getManifoldPoint(u, v) {
        let x, y, z;

        switch (this.manifoldType) {
            case 'swiss-roll':
                const t = u * Math.PI * 1.5;
                x = u * 2;
                z = v * 2;
                y = Math.sin(t) * 0.5 + Math.cos(v * Math.PI) * 0.3;
                break;

            case 'saddle':
                x = u * 2;
                z = v * 2;
                y = (u * u - v * v) * 0.5;
                break;

            case 'sphere':
                const theta = (u + 1) * Math.PI / 2;
                const phi = v * Math.PI;
                const r = 1.5;
                x = r * Math.sin(theta) * Math.cos(phi);
                z = r * Math.sin(theta) * Math.sin(phi);
                y = r * Math.cos(theta) - 0.5;
                break;

            case 'torus':
                const R = 1.2;
                const sr = 0.5;
                const angle1 = u * Math.PI;
                const angle2 = v * Math.PI * 2;
                x = (R + sr * Math.cos(angle2)) * Math.cos(angle1);
                z = (R + sr * Math.cos(angle2)) * Math.sin(angle1);
                y = sr * Math.sin(angle2);
                break;

            case 'wave':
                x = u * 2;
                z = v * 2;
                y = Math.sin(u * Math.PI * 2) * Math.cos(v * Math.PI * 2) * 0.5;
                break;

            default:
                x = u * 2;
                z = v * 2;
                y = 0;
        }

        return { x, y, z };
    }

    createManifolds() {
        if (this.scenes.original) {
            this.meshes.original = this.createOriginalManifold();
            this.scenes.original.add(this.meshes.original.surface);
            this.scenes.original.add(this.meshes.original.wireframe);
        }

        if (this.scenes.activated) {
            this.meshes.activated = this.createActivatedManifold();
            this.scenes.activated.add(this.meshes.activated.surface);
            this.scenes.activated.add(this.meshes.activated.wireframe);
        }
    }

    recreateManifolds() {
        // Clear existing meshes
        for (const [name, mesh] of Object.entries(this.meshes)) {
            if (mesh && this.scenes[name]) {
                this.scenes[name].remove(mesh.surface);
                this.scenes[name].remove(mesh.wireframe);
                mesh.surface.geometry.dispose();
                mesh.wireframe.geometry.dispose();
            }
        }
        this.meshes = {};
        this.createManifolds();
    }

    recreateActivatedManifold() {
        if (this.meshes.activated && this.scenes.activated) {
            this.scenes.activated.remove(this.meshes.activated.surface);
            this.scenes.activated.remove(this.meshes.activated.wireframe);
            this.meshes.activated.surface.geometry.dispose();
            this.meshes.activated.wireframe.geometry.dispose();
        }

        this.meshes.activated = this.createActivatedManifold();
        this.scenes.activated.add(this.meshes.activated.surface);
        this.scenes.activated.add(this.meshes.activated.wireframe);
    }

    createOriginalManifold() {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const indices = [];
        const res = this.resolution;

        for (let i = 0; i <= res; i++) {
            for (let j = 0; j <= res; j++) {
                const u = (i / res) * 2 - 1;
                const v = (j / res) * 2 - 1;

                const pt = this.getManifoldPoint(u, v);
                positions.push(pt.x, pt.y, pt.z);

                const heightNorm = (pt.y + 1) / 2;
                const color = this.getTerminalColor(heightNorm);
                colors.push(color.r, color.g, color.b);
            }
        }

        for (let i = 0; i < res; i++) {
            for (let j = 0; j < res; j++) {
                const a = i * (res + 1) + j;
                const b = a + 1;
                const c = a + (res + 1);
                const d = c + 1;
                indices.push(a, c, b, b, c, d);
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();

        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            shininess: 50,
            transparent: true,
            opacity: 0.85
        });

        const surface = new THREE.Mesh(geometry, material);

        const wireframeMat = new THREE.LineBasicMaterial({
            color: this.colors.wireframe,
            opacity: 0.3,
            transparent: true
        });
        const wireframe = new THREE.LineSegments(
            new THREE.WireframeGeometry(geometry),
            wireframeMat
        );

        return { surface, wireframe };
    }

    createActivatedManifold() {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const indices = [];
        const res = this.resolution;
        const info = this.activationInfo[this.activationType];
        const preservesTopology = info ? info.preserves : false;

        for (let i = 0; i <= res; i++) {
            for (let j = 0; j <= res; j++) {
                const u = (i / res) * 2 - 1;
                const v = (j / res) * 2 - 1;

                const pt = this.getManifoldPoint(u, v);

                // Apply activation element-wise to simulate neural network layer
                // The key insight: ReLU folds the manifold, sigmoid/tanh stretch it
                const preActivation = pt.y;
                const activatedY = this.applyActivation(preActivation);

                positions.push(pt.x, activatedY, pt.z);

                // Coloring based on whether topology is preserved
                if (preservesTopology) {
                    // Smooth gradient for homeomorphic functions - green tones
                    const heightNorm = (activatedY + 1) / 2;
                    const color = this.getTerminalColor(heightNorm, 0.1);
                    colors.push(color.r, color.g, color.b);
                } else {
                    // Show dead regions for ReLU-like - darker for negative pre-activation
                    const isActive = preActivation > 0;
                    if (isActive) {
                        const heightNorm = Math.min(1, 0.5 + activatedY * 0.3);
                        const color = this.getTerminalColor(heightNorm, 0);
                        colors.push(color.r, color.g, color.b);
                    } else {
                        // Dead/folded region - dark magenta
                        colors.push(0.2, 0.08, 0.15);
                    }
                }
            }
        }

        for (let i = 0; i < res; i++) {
            for (let j = 0; j < res; j++) {
                const a = i * (res + 1) + j;
                const b = a + 1;
                const c = a + (res + 1);
                const d = c + 1;
                indices.push(a, c, b, b, c, d);
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();

        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            shininess: 50,
            transparent: true,
            opacity: 0.85
        });

        const surface = new THREE.Mesh(geometry, material);

        const wireframeMat = new THREE.LineBasicMaterial({
            color: this.colors.accent,
            opacity: 0.3,
            transparent: true
        });
        const wireframe = new THREE.LineSegments(
            new THREE.WireframeGeometry(geometry),
            wireframeMat
        );

        return { surface, wireframe };
    }

    getTerminalColor(t, cyanMix = 0) {
        t = Math.max(0, Math.min(1, t));
        cyanMix = Math.max(0, Math.min(1, cyanMix));

        const green = {
            r: 0.31 + t * 0.67,
            g: 0.98,
            b: 0.46 - t * 0.35
        };

        const cyan = { r: 0.30, g: 0.93, b: 0.92 };

        return {
            r: green.r * (1 - cyanMix) + cyan.r * cyanMix,
            g: green.g * (1 - cyanMix) + cyan.g * cyanMix,
            b: green.b * (1 - cyanMix) + cyan.b * cyanMix
        };
    }

    setupMouseControls(container, camera, type) {
        let isDragging = false;
        let previousMouse = { x: 0, y: 0 };
        let rotation = { x: 0.3, y: 0.5 };
        const radius = 6;

        const updateCamera = () => {
            camera.position.x = radius * Math.sin(rotation.y) * Math.cos(rotation.x);
            camera.position.y = radius * Math.sin(rotation.x) + 1;
            camera.position.z = radius * Math.cos(rotation.y) * Math.cos(rotation.x);
            camera.lookAt(0, 0, 0);
        };

        updateCamera();

        container.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousMouse = { x: e.clientX, y: e.clientY };
        });

        container.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaX = e.clientX - previousMouse.x;
            const deltaY = e.clientY - previousMouse.y;
            rotation.y += deltaX * 0.01;
            rotation.x = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, rotation.x + deltaY * 0.01));
            updateCamera();
            previousMouse = { x: e.clientX, y: e.clientY };
        });

        container.addEventListener('mouseup', () => isDragging = false);
        container.addEventListener('mouseleave', () => isDragging = false);

        container.addEventListener('touchstart', (e) => {
            isDragging = true;
            previousMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        });

        container.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            e.preventDefault();
            const deltaX = e.touches[0].clientX - previousMouse.x;
            const deltaY = e.touches[0].clientY - previousMouse.y;
            rotation.y += deltaX * 0.01;
            rotation.x = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, rotation.x + deltaY * 0.01));
            updateCamera();
            previousMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        });

        container.addEventListener('touchend', () => isDragging = false);
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        this.time += 0.01;

        for (const [name, mesh] of Object.entries(this.meshes)) {
            if (mesh && mesh.surface) {
                mesh.surface.rotation.y += 0.002;
                mesh.wireframe.rotation.y += 0.002;
            }
        }

        for (const [name, renderer] of Object.entries(this.renderers)) {
            if (renderer && this.scenes[name] && this.cameras[name]) {
                renderer.render(this.scenes[name], this.cameras[name]);
            }
        }
    }

    onResize() {
        for (const [name, container] of Object.entries(this.containers)) {
            if (container && this.cameras[name] && this.renderers[name]) {
                const width = container.clientWidth;
                const height = container.clientHeight;
                this.cameras[name].aspect = width / height;
                this.cameras[name].updateProjectionMatrix();
                this.renderers[name].setSize(width, height);
            }
        }
    }

    dispose() {
        if (this.animationId) cancelAnimationFrame(this.animationId);
        for (const renderer of Object.values(this.renderers)) {
            if (renderer) renderer.dispose();
        }
    }
}

// Initialize
let topologicalDistortionViz = null;

function initTopologicalDistortionViz() {
    if (document.getElementById('manifold-original')) {
        topologicalDistortionViz = new TopologicalDistortionViz();
    }
}

// Export
if (typeof window !== 'undefined') {
    window.TopologicalDistortionViz = TopologicalDistortionViz;
    window.initTopologicalDistortionViz = initTopologicalDistortionViz;
}
