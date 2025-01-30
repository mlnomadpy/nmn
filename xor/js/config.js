// Plot dimensions and margins
export const width = 350;
export const height = 350;
export const margin = { top: 20, right: 20, bottom: 30, left: 30 };
export const innerWidth = width - margin.left - margin.right;
export const innerHeight = height - margin.top - margin.bottom;

// XOR dataset
export const data = [
  { x: 0, y: 0, label: 0 },
  { x: 0, y: 1, label: 1 },
  { x: 1, y: 0, label: 1 },
  { x: 1, y: 1, label: 0 },
];

// Scales
export const xScale = d3.scaleLinear()
  .domain([-0.2, 1.2])
  .range([0, innerWidth]);

export const yScale = d3.scaleLinear()
  .domain([-0.2, 1.2])
  .range([innerHeight, 0]);

export const weightScale = d3.scaleLinear()
  .domain([-3, 3])
  .range([0, innerWidth]);
