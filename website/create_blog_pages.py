#!/usr/bin/env python3
"""
Convert blog modal content to standalone HTML pages
"""
import os
import re
from pathlib import Path

# Read the main index.html to get the head section
with open('index.html', 'r', encoding='utf-8') as f:
    index_content = f.read()

# Extract head section (everything from <!DOCTYPE to </head>)
head_match = re.search(r'(<!DOCTYPE html>.*?</head>)', index_content, re.DOTALL)
if head_match:
    head_section = head_match.group(1)
else:
    # Fallback: extract from <head> tag
    head_match = re.search(r'(<head[^>]*>.*?</head>)', index_content, re.DOTALL)
    head_section = '<!DOCTYPE html>\n<html lang="en">\n' + (head_match.group(1) if head_match else '<head></head>')

# Extract navigation and footer from index
nav_match = re.search(r'(<nav[^>]*>.*?</nav>)', index_content, re.DOTALL)
nav_section = nav_match.group(1) if nav_match else ''

footer_match = re.search(r'(<footer[^>]*>.*?</footer>)', index_content, re.DOTALL)
footer_section = footer_match.group(1) if footer_match else ''

# Extract scripts section
scripts_match = re.search(r'(<!-- Scripts -->.*?</body>)', index_content, re.DOTALL)
scripts_section = scripts_match.group(1) if scripts_match else ''

# Blog post metadata
blog_metadata = {
    '01-mercer-kernel': {
        'title': 'The ⵟ-Product is a Mercer Kernel',
        'number': 1,
        'prev': None,
        'next': '02-universal-approximation'
    },
    '02-universal-approximation': {
        'title': 'Universal Approximation Theorem',
        'number': 2,
        'prev': '01-mercer-kernel',
        'next': '03-self-regulation'
    },
    '03-self-regulation': {
        'title': 'Self-Regulation & Bounded Outputs',
        'number': 3,
        'prev': '02-universal-approximation',
        'next': '04-stable-gradients'
    },
    '04-stable-gradients': {
        'title': 'Stable Learning & Gradient Localization',
        'number': 4,
        'prev': '03-self-regulation',
        'next': '05-information-theory'
    },
    '05-information-theory': {
        'title': 'Information-Geometric Foundations',
        'number': 5,
        'prev': '04-stable-gradients',
        'next': '06-topology'
    },
    '06-topology': {
        'title': 'Topological Organization: Neural Fiber Bundles',
        'number': 6,
        'prev': '05-information-theory',
        'next': None
    }
}

# Create blog pages directory
os.makedirs('blog-pages', exist_ok=True)

# Process each blog file
blog_dir = Path('blog')
for blog_file in sorted(blog_dir.glob('*.html')):
    if blog_file.name.startswith('_'):
        continue
    
    # Extract blog ID (e.g., '01-mercer-kernel' from '01-mercer-kernel.html')
    blog_id = blog_file.stem
    if blog_id not in blog_metadata:
        continue
    
    meta = blog_metadata[blog_id]
    
    # Read blog content
    with open(blog_file, 'r', encoding='utf-8') as f:
        blog_content = f.read()
    
    # Extract the modal body content (remove modal wrapper)
    # Remove the outer div and modal header/close button
    body_match = re.search(r'<div class="blog-modal-body">(.*?)</div>\s*<div class="blog-modal-nav">', blog_content, re.DOTALL)
    if body_match:
        body_content = body_match.group(1)
    else:
        # Fallback: try to extract everything between modal-body and modal-nav
        body_match = re.search(r'blog-modal-body">(.*?)blog-modal-nav', blog_content, re.DOTALL)
        body_content = body_match.group(1) if body_match else blog_content
    
    # Create navigation links
    nav_links = '<div class="blog-page-nav">'
    if meta['prev']:
        nav_links += f'<a href="{meta["prev"]}.html" class="blog-nav-btn">← Previous: {blog_metadata[meta["prev"]]["title"]}</a>'
    else:
        nav_links += '<a href="index.html#theory" class="blog-nav-btn">← Back to Theory</a>'
    
    nav_links += '<a href="index.html#theory" class="blog-nav-btn">All Theorems</a>'
    
    if meta['next']:
        nav_links += f'<a href="{meta["next"]}.html" class="blog-nav-btn">Next: {blog_metadata[meta["next"]]["title"]} →</a>'
    else:
        nav_links += '<a href="index.html#theory" class="blog-nav-btn">Back to Theory →</a>'
    nav_links += '</div>'
    
    # Update title in head section and fix CSS paths
    head_with_title = re.sub(r'<title>.*?</title>', f'<title>{meta["title"]} — Neural Matter Networks</title>', head_section)
    # Fix CSS paths (blog-pages are in a subdirectory)
    head_with_title = head_with_title.replace('href="css/', 'href="../css/')
    # Add blog-pages CSS if not already present
    if 'blog-pages.css' not in head_with_title:
        head_with_title = head_with_title.replace('</head>', '    <link rel="stylesheet" href="../css/blog-pages.css">\n</head>')
    
    # Create full HTML page
    page_html = f'''{head_with_title}
<body>
    {nav_section}
    
    <!-- Blog Post Page -->
    <main class="blog-page">
        <div class="container">
            <article class="blog-post">
                <header class="blog-post-header">
                    <a href="index.html#theory" class="blog-back-link">← Back to Theory</a>
                    <span class="blog-post-badge">Theorem {meta['number']}</span>
                    <h1 class="blog-post-title">{meta['title']}</h1>
                </header>
                
                <div class="blog-post-content">
{body_content}
                </div>
                
                {nav_links}
            </article>
        </div>
    </main>
    
    {footer_section}
    {scripts_section}
</body>
</html>'''
    
    # Write the page
    output_path = Path('blog-pages') / blog_file.name
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(page_html)
    
    print(f'[OK] Created: blog-pages/{blog_file.name}')

print(f'\n[OK] Created {len(blog_metadata)} blog pages')

