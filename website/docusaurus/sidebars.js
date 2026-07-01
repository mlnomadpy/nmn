// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
export default {
    tutorialSidebar: [
        'intro',
        {
            type: 'category',
            label: 'Getting Started',
            items: ['installation', 'quick-start', 'migration-guide'],
        },
        {
            type: 'category',
            label: 'Layers',
            items: [
                'layers/yat-nmn',
                'layers/yat-conv',
                'layers/yat-conv-transpose',
            ],
        },
        {
            type: 'category',
            label: 'Attention',
            items: [
                'attention/yat-attention',
                'attention/multi-head',
                'attention/rotary',
                'attention/linear-attention',
            ],
        },
        {
            type: 'category',
            label: 'Guides',
            items: [
                'guides/lazy-training',
            ],
        },
        {
            type: 'category',
            label: 'CLI',
            items: [
                'cli',
            ],
        },
        {
            type: 'category',
            label: 'Examples',
            items: [
                'examples/mnist',
                'examples/cifar10',
                'examples/transformer',
            ],
        },
    ],
};
