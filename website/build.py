#!/usr/bin/env python3
"""
Build script to combine template with extracted blog posts and visualizations
"""
import os
import glob

# Read template
with open('templates/index.html', 'r', encoding='utf-8') as f:
    template = f.read()

# Read and combine blog modals
blog_dir = 'blog'
blog_files = sorted(glob.glob(os.path.join(blog_dir, '*.html')))
blog_content = '\n'.join([open(f, 'r', encoding='utf-8').read() for f in blog_files])

# Read visualizations
viz_file = 'visualizations/visualizations.html'
viz_content = open(viz_file, 'r', encoding='utf-8').read() if os.path.exists(viz_file) else ''

# Replace placeholders
output = template.replace('<!-- VISUALIZATIONS_PLACEHOLDER -->', viz_content)
output = output.replace('<!-- BLOG_MODALS_PLACEHOLDER -->', blog_content)

# Write final index.html
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(output)

print(f'[OK] Build complete!')
print(f'   - {len(blog_files)} blog posts included')
print(f'   - Visualizations: {"included" if viz_content else "not found"}')
print(f'   - Output: index.html')

