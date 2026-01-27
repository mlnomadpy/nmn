const fs = require('fs');
const path = require('path');

// Read the template
const templatePath = path.join(__dirname, 'templates', 'index.html');
const template = fs.readFileSync(templatePath, 'utf8');

// Read all blog posts
const blogDir = path.join(__dirname, 'blog');
const blogFiles = fs.readdirSync(blogDir)
    .filter(file => file.endsWith('.html'))
    .sort(); // Sort to maintain order

let blogContent = '<div class="blog-modal-overlay" id="modalOverlay"></div>\n';
for (const file of blogFiles) {
    const content = fs.readFileSync(path.join(blogDir, file), 'utf8');
    blogContent += content + '\n';
}

// Read all visualization sections
const vizDir = path.join(__dirname, 'visualizations');
const vizFiles = fs.readdirSync(vizDir)
    .filter(file => file.endsWith('.html'))
    .sort(); // Sort to maintain order

let vizContent = '';
for (const file of vizFiles) {
    const content = fs.readFileSync(path.join(vizDir, file), 'utf8');
    vizContent += content + '\n';
}

// Replace placeholders in template
let output = template
    .replace('<!-- BLOG_MODALS_PLACEHOLDER -->', blogContent)
    .replace('<!-- VISUALIZATIONS_PLACEHOLDER -->', vizContent);

// Write the final index.html
const outputPath = path.join(__dirname, 'index.html');
fs.writeFileSync(outputPath, output, 'utf8');

console.log('✅ Build complete!');
console.log(`   - ${blogFiles.length} blog posts included`);
console.log(`   - ${vizFiles.length} visualization sections included`);
console.log(`   - Output: ${outputPath}`);


