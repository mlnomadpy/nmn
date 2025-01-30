// Set up constants and data
const width = 350;
const height = 350;
const margin = { top: 20, right: 20, bottom: 30, left: 30 };
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

// XOR dataset
const data = [
  { x: 0, y: 0, label: 0 },
  { x: 0, y: 1, label: 1 },
  { x: 1, y: 0, label: 1 },
  { x: 1, y: 1, label: 0 },
];

// Neuron functions
function dotNeuron(x, y, w1, w2, b) {
  return w1 * x + w2 * y + b;
}

function yatNeuron(x, y, w1, w2, b) {
  const dot = w1 * x + w2 * y;
  const dist = Math.pow(w1 - x, 2) + Math.pow(w2 - y, 2);
  const epsilon = 1e-6;
  return (dot * dot) / (dist + epsilon) + b;
}

function posiYatNeuron(x, y, w1, w2, b) {
  const dot = w1 * x + w2 * y;
  const dist = Math.pow(w1 - x, 2) + Math.pow(w2 - y, 2);
  const epsilon = 1e-6;
  return Math.sqrt(dist) / (dot * dot + epsilon) + b;
}

// Initialize weights and biases
let dotParams = {
  w1: Math.random() - 0.5,
  w2: Math.random() - 0.5,
  b: Math.random() - 0.5,
};
let yatParams = {
  w1: Math.random() - 0.5,
  w2: Math.random() - 0.5,
  b: Math.random() - 0.5,
};
let posiParams = {
  w1: Math.random() - 0.5,
  w2: Math.random() - 0.5,
  b: Math.random() - 0.5,
};

// Weight paths
let dotWeightPath = [];
let yatWeightPath = [];
let posiWeightPath = [];

// Create scales
const xScale = d3
  .scaleLinear()
  .domain([-0.2, 1.2]) // Adjusted domain
  .range([0, innerWidth]);

const yScale = d3
  .scaleLinear()
  .domain([-0.2, 1.2]) // Adjusted domain
  .range([innerHeight, 0]);

// Create weight plot scales
const weightScale = d3
  .scaleLinear()
  .domain([-3, 3]) // Adjusted domain for weights
  .range([0, innerWidth]);

// Create SVG containers
function createPlot(containerId) {
  const svg = d3
    .select(`#${containerId}`)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Add axes
  g.append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(xScale));

  g.append("g").call(d3.axisLeft(yScale));

  // Add grid
  const gridSize = 30;
  const gridData = [];
  for (let x = 0; x <= gridSize; x++) {
    for (let y = 0; y <= gridSize; y++) {
      gridData.push({
        x: -0.5 + (2 * x) / gridSize,
        y: -0.5 + (2 * y) / gridSize,
      });
    }
  }

  // Add grid points
  g.selectAll(".grid-point")
    .data(gridData)
    .enter()
    .append("circle")
    .attr("class", "grid-point")
    .attr("cx", (d) => xScale(d.x))
    .attr("cy", (d) => yScale(d.y))
    .attr("r", 2)
    .attr("fill", "#ddd");

  // Add XOR points
  g.selectAll(".point")
    .data(data)
    .enter()
    .append("circle")
    .attr("class", "point")
    .attr("cx", (d) => xScale(d.x))
    .attr("cy", (d) => yScale(d.y))
    .attr("r", 6)
    .attr("fill", (d) => (d.label === 1 ? "#ff4444" : "#4444ff"));

  return g;
}

// Create weight plot
function createWeightPlot(containerId) {
  const svg = d3
    .select(`#${containerId}`)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Add axes
  g.append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(weightScale));

  g.append("g").call(d3.axisLeft(weightScale));

  // Add grid lines
  const gridlines = g.append("g").attr("class", "grid-lines");

  // Vertical grid lines
  gridlines
    .selectAll(".vertical")
    .data(weightScale.ticks(10))
    .enter()
    .append("line")
    .attr("class", "vertical")
    .attr("x1", (d) => weightScale(d))
    .attr("x2", (d) => weightScale(d))
    .attr("y1", 0)
    .attr("y2", innerHeight)
    .attr("stroke", "#ddd")
    .attr("stroke-width", 1);

  // Horizontal grid lines
  gridlines
    .selectAll(".horizontal")
    .data(weightScale.ticks(10))
    .enter()
    .append("line")
    .attr("class", "horizontal")
    .attr("x1", 0)
    .attr("x2", innerWidth)
    .attr("y1", (d) => weightScale(d))
    .attr("y2", (d) => weightScale(d))
    .attr("stroke", "#ddd")
    .attr("stroke-width", 1);

  // Add weight path
  g.append("path").attr("class", "weight-path");

  // Add current weight point
  g.append("circle").attr("class", "weight-point").attr("r", 4);

  return g;
}

// Create history plot
function createHistoryPlot(containerId, height = 100) {
  const width = (innerWidth - 20) / 3;

  const svg = d3
    .select(`#${containerId}`)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g").attr("transform", "translate(5,5)");

  g.append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${height - 20})`);

  g.append("g").attr("class", "y-axis").attr("transform", "translate(25,0)");

  g.append("path").attr("class", "line").attr("stroke", "#007bff");

  return g;
}

// Create all plots
const dotPlot = createPlot("dot-plot");
const yatPlot = createPlot("yat-plot");
const posiPlot = createPlot("posi-plot");

// Create weight plots
const dotWeightPlot = createWeightPlot("dot-weight-plot");
const yatWeightPlot = createWeightPlot("yat-weight-plot");
const posiWeightPlot = createWeightPlot("posi-weight-plot");

// Create history plots
const historyPlots = {
  dot: {
    loss: createHistoryPlot("dot-loss-history"),
    accuracy: createHistoryPlot("dot-accuracy-history"),
    magnitude: createHistoryPlot("dot-magnitude-history"),
  },
  yat: {
    loss: createHistoryPlot("yat-loss-history"),
    accuracy: createHistoryPlot("yat-accuracy-history"),
    magnitude: createHistoryPlot("yat-magnitude-history"),
  },
  posi: {
    loss: createHistoryPlot("posi-loss-history"),
    accuracy: createHistoryPlot("posi-accuracy-history"),
    magnitude: createHistoryPlot("posi-magnitude-history"),
  },
};

// Update functions
function updatePlot(g, neuronFn, params) {
  const gridSize = 30;
  const points = [];

  for (let x = 0; x <= gridSize; x++) {
    for (let y = 0; y <= gridSize; y++) {
      const xVal = -0.5 + (2 * x) / gridSize;
      const yVal = -0.5 + (2 * y) / gridSize;
      const output = neuronFn(xVal, yVal, params.w1, params.w2, params.b);
      points.push({
        x: xVal,
        y: yVal,
        value: output,
      });
    }
  }

  g.selectAll(".grid-point")
    .data(points)
    .attr("fill", (d) =>
      d.value > 0.5 ? "rgba(255,100,100,0.2)" : "rgba(100,100,255,0.2)"
    );
}

// Update weight plot
function updateWeightPlot(g, params, weightPath) {
  const line = d3
    .line()
    .x((d) => weightScale(d.w1))
    .y((d) => weightScale(d.w2));

  g.select(".weight-path").datum(weightPath).attr("d", line);

  g.select(".weight-point")
    .attr("cx", weightScale(params.w1))
    .attr("cy", weightScale(params.w2));
}

// Loss calculation
function calculateLoss(neuronFn, params) {
  return (
    data.reduce((sum, point) => {
      const output = neuronFn(point.x, point.y, params.w1, params.w2, params.b);
      return sum + Math.pow(output - point.label, 2);
    }, 0) / data.length
  );
}

// Gradient descent
function gradientDescent(neuronFn, params, learningRate = 0.01) {
  const epsilon = 1e-7;
  const gradients = { w1: 0, w2: 0, b: 0 };

  // Calculate gradients numerically
  for (let param in params) {
    const originalValue = params[param];
    params[param] = originalValue + epsilon;
    const lossPlus = calculateLoss(neuronFn, params);
    params[param] = originalValue - epsilon;
    const lossMinus = calculateLoss(neuronFn, params);
    gradients[param] = (lossPlus - lossMinus) / (2 * epsilon);
    params[param] = originalValue;
  }

  // Update parameters
  for (let param in params) {
    params[param] -= learningRate * gradients[param];
  }

  return calculateLoss(neuronFn, params);
}

// Calculate accuracy
function calculateAccuracy(neuronFn, params) {
  return (
    data.reduce((correct, point) => {
      const output = neuronFn(point.x, point.y, params.w1, params.w2, params.b);
      return correct + ((output > 0.5 ? 1 : 0) === point.label ? 1 : 0);
    }, 0) / data.length
  );
}

// Calculate neuron magnitude
function calculateMagnitude(params) {
  return Math.sqrt(params.w1 * params.w1 + params.w2 * params.w2);
}

// Update history plots
function updateHistoryPlots(plots, history) {
  const updatePlot = (plot, data, color) => {
    const width = (innerWidth - 20) / 3;
    const height = 100;

    const xScale = d3
      .scaleLinear()
      .domain([0, data.length - 1])
      .range([30, width - 5]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(data) * 1.1])
      .range([height - 20, 5]);

    const line = d3
      .line()
      .x((d, i) => xScale(i))
      .y((d) => yScale(d));

    plot.select(".line").datum(data).attr("d", line).attr("stroke", color);

    // Update axes
    const xAxis = d3.axisBottom(xScale).ticks(5).tickSize(5);

    const yAxis = d3.axisLeft(yScale).ticks(5).tickSize(5);

    plot.select(".x-axis").call(xAxis);
    plot.select(".y-axis").call(yAxis);
  };

  updatePlot(plots.loss, history.loss, "#ff4444");
  updatePlot(plots.accuracy, history.accuracy, "#44ff44");
  updatePlot(plots.magnitude, history.magnitude, "#4444ff");
}

// Optimization functions
function optimize(
  neuronFn,
  params,
  plot,
  weightPlot,
  weightPath,
  lossDisplay,
  historyPlots,
  iterations = 100
) {
  weightPath.length = 0; // Clear previous path
  const history = {
    loss: [],
    accuracy: [],
    magnitude: [],
  };

  let i = 0;
  const interval = setInterval(() => {
    const loss = gradientDescent(neuronFn, params, 0.1);
    const accuracy = calculateAccuracy(neuronFn, params);
    const magnitude = calculateMagnitude(params);

    weightPath.push({ w1: params.w1, w2: params.w2 });

    history.loss.push(loss);
    history.accuracy.push(accuracy);
    history.magnitude.push(magnitude);

    updatePlot(plot, neuronFn, params);
    updateWeightPlot(weightPlot, params, weightPath);
    updateHistoryPlots(historyPlots, history);
    lossDisplay.textContent = `Loss: ${loss.toFixed(4)} | Accuracy: ${(
      accuracy * 100
    ).toFixed(1)}% | w1: ${params.w1.toFixed(3)} | w2: ${params.w2.toFixed(
      3
    )} | b: ${params.b.toFixed(3)}`;

    i++;
    if (i >= iterations) {
      clearInterval(interval);
    }
  }, 50);
}

// Reset functions
function resetParams(params) {
  params.w1 = Math.random() - 0.5;
  params.w2 = Math.random() - 0.5;
  params.b = Math.random() - 0.5;
}

// Event handlers
function optimizeDot() {
  const iterations = parseInt(document.getElementById("dot-iterations").value);
  optimize(
    dotNeuron,
    dotParams,
    dotPlot,
    dotWeightPlot,
    dotWeightPath,
    document.getElementById("dot-loss"),
    historyPlots.dot,
    iterations
  );
}

function optimizeYat() {
  const iterations = parseInt(document.getElementById("yat-iterations").value);
  optimize(
    yatNeuron,
    yatParams,
    yatPlot,
    yatWeightPlot,
    yatWeightPath,
    document.getElementById("yat-loss"),
    historyPlots.yat,
    iterations
  );
}

function optimizePosi() {
  const iterations = parseInt(document.getElementById("posi-iterations").value);
  optimize(
    posiYatNeuron,
    posiParams,
    posiPlot,
    posiWeightPlot,
    posiWeightPath,
    document.getElementById("posi-loss"),
    historyPlots.posi,
    iterations
  );
}

function resetDot() {
  resetParams(dotParams);
  dotWeightPath.length = 0;
  updatePlot(dotPlot, dotNeuron, dotParams);
  updateWeightPlot(dotWeightPlot, dotParams, dotWeightPath);
  document.getElementById("dot-loss").textContent = `Loss: ${calculateLoss(
    dotNeuron,
    dotParams
  ).toFixed(4)} | w1: ${dotParams.w1.toFixed(3)} | w2: ${dotParams.w2.toFixed(
    3
  )} | b: ${dotParams.b.toFixed(3)}`;
}

function resetYat() {
  resetParams(yatParams);
  yatWeightPath.length = 0;
  updatePlot(yatPlot, yatNeuron, yatParams);
  updateWeightPlot(yatWeightPlot, yatParams, yatWeightPath);
  document.getElementById("yat-loss").textContent = `Loss: ${calculateLoss(
    yatNeuron,
    yatParams
  ).toFixed(4)} | w1: ${yatParams.w1.toFixed(3)} | w2: ${yatParams.w2.toFixed(
    3
  )} | b: ${yatParams.b.toFixed(3)}`;
}

function resetPosi() {
  resetParams(posiParams);
  posiWeightPath.length = 0;
  updatePlot(posiPlot, posiYatNeuron, posiParams);
  updateWeightPlot(posiWeightPlot, posiParams, posiWeightPath);
  document.getElementById("posi-loss").textContent = `Loss: ${calculateLoss(
    posiYatNeuron,
    posiParams
  ).toFixed(4)} | w1: ${posiParams.w1.toFixed(3)} | w2: ${posiParams.w2.toFixed(
    3
  )} | b: ${posiParams.b.toFixed(3)}`;
}

// Initial updates
updatePlot(dotPlot, dotNeuron, dotParams);
updatePlot(yatPlot, yatNeuron, yatParams);
updatePlot(posiPlot, posiYatNeuron, posiParams);

updateWeightPlot(dotWeightPlot, dotParams, dotWeightPath);
updateWeightPlot(yatWeightPlot, yatParams, yatWeightPath);
updateWeightPlot(posiWeightPlot, posiParams, posiWeightPath);
