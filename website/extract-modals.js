const fs = require('fs');
const path = require('path');

// Read the full index.html
const indexPath = path.join(__dirname, 'index.html');
const content = fs.readFileSync(indexPath, 'utf8');

// Extract blog modals section (from <!-- Blog Modals --> to <!-- Scripts -->)
const blogModalsStart = content.indexOf('<!-- Blog Modals -->');
const scriptsStart = content.indexOf('<!-- Scripts -->');
const blogModalsSection = content.substring(blogModalsStart, scriptsStart);

// Extract overlay
const overlayMatch = blogModalsSection.match(/<div class="blog-modal-overlay"[^>]*>[\s\S]*?<\/div>/);
if (overlayMatch) {
    fs.writeFileSync(
        path.join(__dirname, 'blog', '_overlay.html'),
        overlayMatch[0] + '\n',
        'utf8'
    );
}

// Extract each modal (they start with <!-- Modal N: and end with </div> before next <!-- Modal)
const modalRegex = /<!-- Modal \d+:[\s\S]*?<\/div>\s*(?=<!-- Modal \d+:|$)/g;
const modals = blogModalsSection.match(modalRegex);

if (modals) {
    modals.forEach((modal, index) => {
        // Extract modal number and name from comment
        const commentMatch = modal.match(/<!-- Modal (\d+):\s*(.+?)\s*\(/);
        if (commentMatch) {
            const modalNum = commentMatch[1];
            const modalName = commentMatch[2].toLowerCase().replace(/\s+/g, '-');
            const filename = `${modalNum.padStart(2, '0')}-${modalName}.html`;
            
            // Remove the comment line
            const cleanModal = modal.replace(/<!-- Modal \d+:.*?-->\s*/, '');
            
            fs.writeFileSync(
                path.join(__dirname, 'blog', filename),
                cleanModal + '\n',
                'utf8'
            );
            console.log(`Extracted: ${filename}`);
        }
    });
}

console.log(`\n✅ Extracted ${modals ? modals.length : 0} modals`);


