#!/usr/bin/env python3
import re
import os

# Read index.html
with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract visualizations section
viz_start = content.find('<section id="visualizations"')
viz_end = content.find('</section>', viz_start) + len('</section>')
viz_section = content[viz_start:viz_end]

with open('visualizations/visualizations.html', 'w', encoding='utf-8') as f:
    f.write(viz_section)

print('[OK] Extracted visualizations section')

# Extract blog modals
blog_start = content.find('<!-- Blog Modals -->')
scripts_start = content.find('<!-- Scripts -->')
blog_section = content[blog_start:scripts_start]

# Extract overlay
overlay_match = re.search(r'<div class="blog-modal-overlay"[^>]*>.*?</div>', blog_section, re.DOTALL)
if overlay_match:
    with open('blog/_overlay.html', 'w', encoding='utf-8') as f:
        f.write(overlay_match.group(0) + '\n')
    print('[OK] Extracted overlay')

# Extract each modal - need to match the full modal div by counting opening/closing tags
def extract_modal_div(text, start_pos):
    """Extract a complete div tag with all nested content"""
    if text[start_pos:start_pos+4] != '<div':
        return None, start_pos
    
    # Find the opening <div class="blog-modal"
    div_start = text.find('<div class="blog-modal"', start_pos)
    if div_start == -1:
        return None, start_pos
    
    # Count div tags to find the matching closing tag
    pos = div_start + 4  # After '<div'
    depth = 1
    while pos < len(text) and depth > 0:
        # Look for <div or </div
        next_div = text.find('<div', pos)
        next_close = text.find('</div>', pos)
        
        if next_close == -1:
            break
        
        if next_div != -1 and next_div < next_close:
            # Found opening div before closing
            depth += 1
            pos = next_div + 4
        else:
            # Found closing div
            depth -= 1
            if depth == 0:
                # Found matching closing tag
                return text[div_start:next_close + 6], next_close + 6
            pos = next_close + 6
    
    return None, start_pos

# Find all modal comments and extract the modals
modal_pattern = r'<!-- Modal (\d+):\s*([^(]+?)\s*\([^)]*\) -->'
modal_comments = list(re.finditer(modal_pattern, blog_section))

modal_files = []
for i, match in enumerate(modal_comments):
    modal_num = match.group(1)
    modal_name = match.group(2).strip().lower().replace(' ', '-')
    
    # Find the modal div after this comment
    start_pos = match.end()
    modal_content, end_pos = extract_modal_div(blog_section, start_pos)
    
    if modal_content:
        filename = f'{modal_num.zfill(2)}-{modal_name}.html'
        filepath = os.path.join('blog', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modal_content + '\n')
        
        modal_files.append(filename)
        print(f'[OK] Extracted: {filename} ({len(modal_content)} chars)')
    else:
        print(f'[ERROR] Failed to extract modal {modal_num}')

print(f'\n[OK] Extracted {len(modal_files)} modals')
