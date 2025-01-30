import {
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  xScale,
  yScale,
  weightScale,
  data,
} from "./config.js";

// Add these utility functions at the top
const throttle = (func, limit) => {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

function createPlot(containerId) {
  const svg = d3
    .select(containerId)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("class", "plot-svg")  // Add this class for theme handling
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Add background rect that responds to theme
  svg.append("rect")
    .attr("class", "plot-background")
    .attr("width", innerWidth + margin.left + margin.right)
    .attr("height", innerHeight + margin.top + margin.bottom)
    .attr("x", -margin.left)
    .attr("y", -margin.top);

  svg
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(xScale));

  svg.append("g").attr("class", "y-axis").call(d3.axisLeft(yScale));

  return svg;
}

function createWeightPlot(containerId) {
  const svg = d3
    .select(containerId)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("class", "plot-svg")
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  svg.append("rect")
    .attr("class", "plot-background")
    .attr("width", innerWidth + margin.left + margin.right)
    .attr("height", innerHeight + margin.top + margin.bottom)
    .attr("x", -margin.left)
    .attr("y", -margin.top);

  svg
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(weightScale));

  svg
    .append("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(weightScale.range([innerHeight, 0])));

  return svg;
}

function createHistoryPlot(containerId) {
  const plotHeight = 150; // Increased height
  const plotWidth =
    document.querySelector(containerId).clientWidth -
    margin.left -
    margin.right;
  const graphHeight = plotHeight - margin.top - margin.bottom;

  const svg = d3
    .select(containerId)
    .append("svg")
    .attr("width", "100%")
    .attr("height", plotHeight)
    .style("display", "block")
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Add background
  svg
    .append("rect")
    .attr("width", plotWidth)
    .attr("height", graphHeight)
    .attr("fill", "#f8f8f8")
    .attr("rx", 4);

  // Add grid lines
  svg
    .append("g")
    .attr("class", "grid")
    .style("stroke", "#ddd")
    .style("stroke-opacity", 0.5);

  // Add axes
  svg
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${graphHeight})`)
    .style("font-size", "10px");

  svg.append("g").attr("class", "y-axis").style("font-size", "10px");

  // Add the line path
  svg
    .append("path")
    .attr("class", "line")
    .attr("fill", "none")
    .attr("stroke-width", 2)
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round");

  return svg;
}

function updatePlot(g, neuronFn, params) {
  // Create grid data
  const gridSize = 30; // Reduced from 40 for better performance
  const points = [];

  // Use faster array initialization
  const rows = Array.from({ length: gridSize + 1 });
  const cols = Array.from({ length: gridSize + 1 });

  rows.forEach((_, x) => {
    cols.forEach((_, y) => {
      const xVal = -0.2 + (1.4 * x) / gridSize;
      const yVal = -0.2 + (1.4 * y) / gridSize;
      const output = neuronFn(xVal, yVal, params.w1, params.w2, params.b);
      points.push({
        x: xVal,
        y: yVal,
        value: output,
      });
    });
  });

  // Batch DOM updates
  requestAnimationFrame(() => {
    const pointElements = g.selectAll(".grid-point")
      .data(points);

    pointElements.enter()
      .append("circle")
      .merge(pointElements)
      .attr("class", "grid-point")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 4)
      .attr("fill", d => {
        if (d.value > 0.5) {
          const intensity = 0.3 + 0.7 * (d.value - 0.5) * 2;
          return `rgba(255, 50, 50, ${intensity})`;
        }
        const intensity = 0.3 + 0.7 * (0.5 - d.value) * 2;
        return `rgba(50, 50, 255, ${intensity})`;
      });

    pointElements.exit().remove();

    // Update XOR points with enhanced visibility - place this after grid points
    g.selectAll(".xor-point")
      .data(data)
      .join("circle")
      .attr("class", "xor-point")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 8)  // Larger points
      .attr("fill", d => d.label === 1 ? "#000" : "#fff")
      .attr("stroke", "#000")
      .attr("stroke-width", 2);
  });
}

// Throttle the heavy update functions
const throttledUpdatePlot = throttle(updatePlot, 32); // ~30fps
const throttledUpdateCombinedPlot = throttle(updateCombinedPlot, 32);

function updateWeightPlot(g, params, weightPath) {
  // Draw weight path
  const line = d3
    .line()
    .x((d) => weightScale(d.w1))
    .y((d) => weightScale(d.w2));

  g.selectAll(".weight-path")
    .data([weightPath])
    .join("path")
    .attr("class", "weight-path")
    .attr("d", line)
    .attr("stroke", "#666")
    .attr("stroke-width", 1)
    .attr("fill", "none");

  // Update current weight point
  g.selectAll(".weight-point")
    .data([params])
    .join("circle")
    .attr("class", "weight-point")
    .attr("cx", (d) => weightScale(d.w1))
    .attr("cy", (d) => weightScale(d.w2))
    .attr("r", 4)
    .attr("fill", "#ff0000");
}

function updateHistoryPlots(plots, history) {
  const updateHistoryPlot = (
    plot,
    data,
    yDomain = [0, 1],
    color = "steelblue"
  ) => {
    const plotHeight = 80;
    const plotWidth =
      plot.node().parentNode.getBoundingClientRect().width -
      margin.left -
      margin.right;

    const xScale = d3
      .scaleLinear()
      .domain([0, Math.max(10, data.length - 1)])
      .range([0, plotWidth]);

    const yScale = d3.scaleLinear().domain(yDomain).range([plotHeight, 0]);

    // Update grid with more visible lines
    const gridLines = plot.select(".grid");
    gridLines.selectAll("*").remove(); // Clear existing grid lines

    // Add horizontal grid lines
    gridLines
      .selectAll(".horizontal")
      .data(yScale.ticks(5))
      .join("line")
      .attr("class", "horizontal")
      .attr("x1", 0)
      .attr("x2", plotWidth)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .style("stroke-dasharray", "3,3");

    // Add vertical grid lines
    gridLines
      .selectAll(".vertical")
      .data(xScale.ticks(5))
      .join("line")
      .attr("class", "vertical")
      .attr("x1", (d) => xScale(d))
      .attr("x2", (d) => xScale(d))
      .attr("y1", 0)
      .attr("y2", plotHeight)
      .style("stroke-dasharray", "3,3");

    // Update axes
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(5)
      .tickSize(5)
      .tickFormat(d3.format("d"));

    const yAxis = d3
      .axisLeft(yScale)
      .ticks(5)
      .tickSize(5)
      .tickFormat((d) => d.toFixed(2));

    plot.select(".x-axis").call(xAxis);
    plot.select(".y-axis").call(yAxis);

    // Add smoother curve
    const line = d3
      .line()
      .x((d, i) => xScale(i))
      .y((d) => yScale(d))
      .curve(d3.curveBasis);

    // Add area under the curve
    const area = d3
      .area()
      .x((d, i) => xScale(i))
      .y0(plotHeight)
      .y1((d) => yScale(d))
      .curve(d3.curveBasis);

    // Add area with transparency
    plot
      .select(".line")
      .datum(data)
      .attr("stroke", color)
      .attr("d", line)
      .style("fill", `${color}20`); // Add transparent fill
  };

  if (history.loss.length > 0) {
    updateHistoryPlot(
      plots.loss,
      history.loss,
      [0, Math.max(...history.loss) * 1.1],
      "#ff4444"
    );
    updateHistoryPlot(plots.accuracy, history.accuracy, [0, 1], "#44ff44");
    updateHistoryPlot(
      plots.magnitude,
      history.magnitude,
      [0, Math.max(...history.magnitude) * 1.1],
      "#4444ff"
    );
  }
}

function createCombinedPlot(containerId) {
  const combinedWidth = 800;
  const combinedHeight = 600;

  const svg = d3
    .select(containerId)
    .append("svg")
    .attr("width", combinedWidth)
    .attr("height", combinedHeight)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Add axes
  svg
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${combinedHeight - margin.bottom})`)
    .call(d3.axisBottom(xScale));

  svg.append("g").attr("class", "y-axis").call(d3.axisLeft(yScale));

  return svg;
}

function updateCombinedPlot(svg, neurons) {
  const colors = {
    dot: "rgba(255, 68, 68, 0.5)",    // Brighter red with increased opacity
    yat: "rgba(68, 255, 68, 0.5)",    // Brighter green with increased opacity
    posi: "rgba(68, 68, 255, 0.5)"    // Brighter blue with increased opacity
  };

  // Clear previous points
  svg.selectAll(".neuron-points").remove();

  // Draw decision boundaries for each neuron
  Object.entries(neurons).forEach(([type, neuron]) => {
    const gridSize = 50;
    const points = [];

    for (let x = 0; x <= gridSize; x++) {
      for (let y = 0; y <= gridSize; y++) {
        const xVal = -0.2 + (1.4 * x) / gridSize;
        const yVal = -0.2 + (1.4 * y) / gridSize;
        const output = neuron.fn(
          xVal,
          yVal,
          neuron.params.w1,
          neuron.params.w2,
          neuron.params.b
        );
        points.push({ x: xVal, y: yVal, value: output });
      }
    }

    svg
      .selectAll(`.${type}-points`)
      .data(points)
      .join("circle")
      .attr("class", `neuron-points ${type}-points`)
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 3)  // Slightly larger points
      .attr("fill", (d) => d.value > 0.5 ? colors[type] : "transparent")
      .attr("stroke", (d) => d.value > 0.5 ? colors[type] : "transparent")
      .attr("stroke-width", 1);
  });

  // Draw XOR points on top with enhanced visibility
  svg
    .selectAll(".xor-point")
    .data(data)
    .join("circle")
    .attr("class", "xor-point")
    .attr("cx", (d) => xScale(d.x))
    .attr("cy", (d) => yScale(d.y))
    .attr("r", 8)  // Larger points
    .attr("fill", (d) => d.label === 1 ? "#000" : "#fff")
    .attr("stroke", "#000")
    .attr("stroke-width", 2);
}

function updateCombinedWeightPlot(svg, neurons) {
  const colors = {
    dot: "#ff4444",
    yat: "#44ff44",
    posi: "#4444ff"
  };

  // Draw weight paths for each neuron
  Object.entries(neurons).forEach(([type, neuron]) => {
    const line = d3.line()
      .x(d => weightScale(d.w1))
      .y(d => weightScale(d.w2));

    // Draw path
    svg.selectAll(`.weight-path-${type}`)
      .data([neuron.weightPath])
      .join("path")
      .attr("class", `weight-path weight-path-${type}`)
      .attr("d", line)
      .attr("stroke", colors[type])
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .attr("opacity", 0.6);

    // Draw current point
    if (neuron.weightPath.length > 0) {
      const currentPoint = neuron.weightPath[neuron.weightPath.length - 1];
      svg.selectAll(`.weight-point-${type}`)
        .data([currentPoint])
        .join("circle")
        .attr("class", `weight-point weight-point-${type}`)
        .attr("cx", d => weightScale(d.w1))
        .attr("cy", d => weightScale(d.w2))
        .attr("r", 4)
        .attr("fill", colors[type])
        .attr("stroke", "white")
        .attr("stroke-width", 1);
    }
  });
}

// Single export statement at the end of the file
export {
  createPlot,
  createWeightPlot,
  createHistoryPlot,
  createCombinedPlot,
  updatePlot,
  updateWeightPlot,
  updateHistoryPlots,
  updateCombinedPlot,
  updateCombinedWeightPlot,
};
