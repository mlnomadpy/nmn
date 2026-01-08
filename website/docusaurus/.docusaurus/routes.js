import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/nmn/__docusaurus/debug',
    component: ComponentCreator('/nmn/__docusaurus/debug', 'dde'),
    exact: true
  },
  {
    path: '/nmn/__docusaurus/debug/config',
    component: ComponentCreator('/nmn/__docusaurus/debug/config', '190'),
    exact: true
  },
  {
    path: '/nmn/__docusaurus/debug/content',
    component: ComponentCreator('/nmn/__docusaurus/debug/content', '5fa'),
    exact: true
  },
  {
    path: '/nmn/__docusaurus/debug/globalData',
    component: ComponentCreator('/nmn/__docusaurus/debug/globalData', 'a50'),
    exact: true
  },
  {
    path: '/nmn/__docusaurus/debug/metadata',
    component: ComponentCreator('/nmn/__docusaurus/debug/metadata', '98a'),
    exact: true
  },
  {
    path: '/nmn/__docusaurus/debug/registry',
    component: ComponentCreator('/nmn/__docusaurus/debug/registry', '7ab'),
    exact: true
  },
  {
    path: '/nmn/__docusaurus/debug/routes',
    component: ComponentCreator('/nmn/__docusaurus/debug/routes', '6dc'),
    exact: true
  },
  {
    path: '/nmn/blog',
    component: ComponentCreator('/nmn/blog', '067'),
    exact: true
  },
  {
    path: '/nmn/blog/archive',
    component: ComponentCreator('/nmn/blog/archive', 'a2b'),
    exact: true
  },
  {
    path: '/nmn/blog/authors',
    component: ComponentCreator('/nmn/blog/authors', '39e'),
    exact: true
  },
  {
    path: '/nmn/blog/tags',
    component: ComponentCreator('/nmn/blog/tags', 'eea'),
    exact: true
  },
  {
    path: '/nmn/blog/tags/announcement',
    component: ComponentCreator('/nmn/blog/tags/announcement', '2cc'),
    exact: true
  },
  {
    path: '/nmn/blog/welcome',
    component: ComponentCreator('/nmn/blog/welcome', '052'),
    exact: true
  },
  {
    path: '/nmn/docs',
    component: ComponentCreator('/nmn/docs', '39b'),
    routes: [
      {
        path: '/nmn/docs',
        component: ComponentCreator('/nmn/docs', '8ab'),
        routes: [
          {
            path: '/nmn/docs',
            component: ComponentCreator('/nmn/docs', 'bb1'),
            routes: [
              {
                path: '/nmn/docs/attention/multi-head',
                component: ComponentCreator('/nmn/docs/attention/multi-head', 'c85'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/attention/rotary',
                component: ComponentCreator('/nmn/docs/attention/rotary', '92e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/attention/yat-attention',
                component: ComponentCreator('/nmn/docs/attention/yat-attention', '750'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/examples/cifar10',
                component: ComponentCreator('/nmn/docs/examples/cifar10', '066'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/examples/mnist',
                component: ComponentCreator('/nmn/docs/examples/mnist', '925'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/examples/transformer',
                component: ComponentCreator('/nmn/docs/examples/transformer', 'd78'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/installation',
                component: ComponentCreator('/nmn/docs/installation', '45b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/intro',
                component: ComponentCreator('/nmn/docs/intro', 'a8e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/layers/yat-conv',
                component: ComponentCreator('/nmn/docs/layers/yat-conv', '9c0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/layers/yat-conv-transpose',
                component: ComponentCreator('/nmn/docs/layers/yat-conv-transpose', '701'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/layers/yat-nmn',
                component: ComponentCreator('/nmn/docs/layers/yat-nmn', 'd3e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/quick-start',
                component: ComponentCreator('/nmn/docs/quick-start', '766'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/rnn/gru',
                component: ComponentCreator('/nmn/docs/rnn/gru', '01e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/rnn/lstm',
                component: ComponentCreator('/nmn/docs/rnn/lstm', '783'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/nmn/docs/rnn/simple',
                component: ComponentCreator('/nmn/docs/rnn/simple', '5b9'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/nmn/',
    component: ComponentCreator('/nmn/', '24f'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
