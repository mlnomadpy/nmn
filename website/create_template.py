#!/usr/bin/env python3
"""Create template from current index.html"""
import os

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find visualization section
viz_start = content.find('<section id="visualizations"')
viz_end = content.find('</section>', viz_start) + len('</section>')

# Find blog modals section
blog_start = content.find('<!-- Blog Modals -->')
scripts_start = content.find('<!-- Scripts -->')

# Create template with placeholders
template = (
    content[:viz_start] + 
    '    <!-- VISUALIZATIONS_PLACEHOLDER -->\n' + 
    content[viz_end:blog_start] + 
    '    <!-- BLOG_MODALS_PLACEHOLDER -->\n' + 
    content[scripts_start:]
)

# Write template
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(template)

print('[OK] Template created at templates/index.html')

