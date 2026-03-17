# NMN Website Ecosystem

This directory contains the source code for the NMN project's web presence, which consists of three integrated components:

1.  **Interactive Paper/Blog**: A visually rich, custom-built static site explaining the theoretical foundations (the "Visual Paper").
2.  **Documentation**: A Docusaurus-based site for API references, tutorials, and guides.
3.  **Deployment**: A workflow that merges these two components into a single cohesive website.

---

## 1. 🎨 Interactive Paper (Visual Blog)

The "Visual Paper" is the landing page and core educational content, featuring interactive 3D visualizations and mathematical explanations. It creates the `index.html` in this directory.

### Structure
```
website/
├── templates/
│   └── index.html          # Main HTML skeleton
├── blog/                   # Content for the "blog" sections of the paper
│   ├── 01-mercer-kernel.html
│   └── ...
├── visualizations/         # Interactive JS/Canvas visualizations
│   └── visualizations.html
├── build.py                # Script to assemble the visual paper
└── index.html              # GENERATED OUTPUT (Do not edit directly)
```

### Development
To update the visual paper:
1.  Edit content in `blog/`, `visualizations/`, or `templates/`.
2.  Run the build script:
    ```bash
    python build.py
    ```
3.  Open `index.html` locally to verify changes.

---

## 2. 📚 Documentation (Docusaurus)

The technical documentation, API reference, and user guides are built using [Docusaurus](https://docusaurus.io/).

### Structure
```
website/docusaurus/
├── docs/                   # Markdown files for documentation pages
├── src/                    # React components and pages
├── docusaurus.config.js    # Docusaurus configuration
└── static/                 # Static assets (images, files) for docs
```

### Development
To run the documentation site locally:
```bash
cd docusaurus
npm install
npm start
```
This will start a local server at `http://localhost:3000`.

---

## 3. 🚀 Deployment & Integration

The final deployed website combines both components. The **Interactive Paper** is embedded into the Docusaurus site.

### Deployment Workflow (`deploy.yml`)
1.  **Builds the Visual Paper**: Uses the existing `website/index.html` and assets.
2.  **Builds Docusaurus**: Compiles the documentation site.
3.  **Merges**:
    *   The Visual Paper (`index.html`, `css/`, `js/`, `assets/`) is copied into the Docusaurus `static/paper/` directory.
    *   This makes the visual paper accessible at `https://azettaai.github.io/nmn/paper/`.
4.  **Deploys**: The combined artifact is pushed to GitHub Pages.

### Key Links
*   **Documentation Home**: [https://azettaai.github.io/nmn/](https://azettaai.github.io/nmn/)
*   **Visual Paper**: [https://azettaai.github.io/nmn/paper/](https://azettaai.github.io/nmn/paper/)
