# Website Build System

This website uses a modular build system where blog posts and visualizations are extracted into separate files and combined during build.

## Structure

```
website/
├── templates/
│   └── index.html          # Template with placeholders
├── blog/
│   ├── _overlay.html       # Modal overlay
│   ├── 01-mercer-kernel.html
│   ├── 02-universal-approximation.html
│   ├── 03-self-regulation.html
│   ├── 04-stable-gradients.html
│   ├── 05-information-theory.html
│   └── 06-topology.html
├── visualizations/
│   └── visualizations.html # All visualization sections
├── build.py                # Build script
├── extract_content.py     # Extraction script (run once)
└── index.html             # Generated file (do not edit directly)
```

## Building

To build the website:

```bash
python build.py
```

This will:
1. Read the template from `templates/index.html`
2. Combine all blog posts from `blog/` directory
3. Include visualizations from `visualizations/visualizations.html`
4. Generate `index.html` with all content injected

## Editing Content

### Blog Posts

Edit individual blog post files in `blog/` directory. Files are numbered to maintain order:
- `01-mercer-kernel.html`
- `02-universal-approximation.html`
- etc.

### Visualizations

Edit `visualizations/visualizations.html` to modify visualization sections.

### Template

Edit `templates/index.html` to modify the main structure, navigation, hero section, etc.

**Important:** After editing blog posts or visualizations, run `python build.py` to regenerate `index.html`.

## Initial Setup

If you need to re-extract content from a full `index.html`:

```bash
python extract_content.py
```

This extracts:
- Visualizations section → `visualizations/visualizations.html`
- Blog modals → `blog/*.html` files

Then create the template:

```bash
python create_template.py
```

## Development Workflow

1. Edit content in `blog/` or `visualizations/`
2. Run `python build.py`
3. Test the generated `index.html`
4. Commit both the source files and the generated `index.html`

