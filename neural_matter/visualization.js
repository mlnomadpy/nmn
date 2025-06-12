import {
  network,
  config,
  sigmoid,
  dotProduct,
  recalculateNetwork,
} from "./network.js";

// Define colors as a constant at the top level
const colors = {
  input: "#10B981",
  hidden: "#F59E0B",
  concat: "#8B5CF6",
  output: "#3B82F6",
  softmax: "#EC4899", // Add pink color for softmax
};

// D3 visualization setup
const networkSvg = d3
  .select("#network")
  .append("svg")
  .attr("width", "100%")
  .attr("height", "100%");

const networkZoom = d3
  .zoom()
  .scaleExtent([0.1, 4])
  .on("zoom", (event) => {
    networkContainer.attr("transform", event.transform);
  });

networkSvg.call(networkZoom);

const networkContainer = networkSvg.append("g");

const vectorSpaceSvg = d3
  .select("#vectorSpace")
  .append("svg")
  .attr("width", "100%")
  .attr("height", "100%");

// Force simulation setup
const simulation = d3
  .forceSimulation()
  .force(
    "link",
    d3.forceLink().id((d) => d.id)
  )
  .force("charge", d3.forceManyBody().strength(-1000)) // Increased repulsion
  .force("collision", d3.forceCollide().radius(50)) // Add collision detection
  .force(
    "x",
    d3
      .forceX()
      .x((d) => {
        const width = document.getElementById("network").clientWidth;
        const totalLayers = config.hiddenLayers.length + 2;
        // Spread layers horizontally with more space
        return width * ((d.layer + 0.5) / totalLayers);
      })
      .strength(0.5) // Adjust strength to allow some flexibility
  )
  .force(
    "y",
    d3
      .forceY()
      .y((d) => {
        const height = document.getElementById("network").clientHeight;
        const layerSize =
          d.type === "input"
            ? 1 // Single input vector
            : d.type === "output"
            ? config.outputSize
            : config.hiddenLayers[d.layer - 1];
        // More vertical spread
        return height * (0.2 + (d.index / Math.max(1, layerSize - 1)) * 0.6);
      })
      .strength(0.3) // Adjust strength to allow some flexibility
  );

function updateForceLayout() {
  const totalLayers = config.layers.length;
  
  simulation
    .alpha(1)
    .force("charge", d3.forceManyBody().strength(-1000))
    .force("collision", d3.forceCollide().radius(50))
    .force(
      "x",
      d3
        .forceX()
        .x((d) => {
          const width = document.getElementById("network").clientWidth;
          // Use layer index for x-position
          return width * ((d.layer + 0.5) / totalLayers);
        })
        .strength(0.5)
    )
    .force(
      "y",
      d3
        .forceY()
        .y((d) => {
          const height = document.getElementById("network").clientHeight;
          // Calculate nodes in same layer for vertical distribution
          const nodesInLayer = network.nodes.filter(n => n.layer === d.layer).length;
          const indexInLayer = network.nodes.filter(n => n.layer === d.layer)
            .findIndex(n => n.id === d.id);
          
          // Distribute nodes vertically within their layer
          return height * (0.2 + (indexInLayer / Math.max(1, nodesInLayer - 1)) * 0.6);
        })
        .strength(0.3)
    );
}

export function reinitializeSimulation() {
  // Initialize positions with better distribution
  network.nodes.forEach((node) => {
    const width = document.getElementById("network").clientWidth;
    const height = document.getElementById("network").clientHeight;
    const totalLayers = config.layers.length;
    
    // Set initial positions based on layer
    node.x = width * ((node.layer + 0.5) / totalLayers);
    node.y = height / 2 + (Math.random() - 0.5) * height * 0.4;
    
    // Clear any fixed positions
    node.fx = null;
    node.fy = null;
  });

  simulation.nodes(network.nodes).force("link").links(network.links);
  updateForceLayout();
  simulation.alpha(1).restart();
}

export function updateVisualization() {
  networkContainer.selectAll("*").remove();
  vectorSpaceSvg.selectAll("*").remove();

  // Reset zoom
  networkSvg.call(networkZoom.transform, d3.zoomIdentity);

  // Create color gradients
  const defs = networkContainer.append("defs");

  Object.entries(colors).forEach(([type, color]) => {
    const gradient = defs
      .append("radialGradient")
      .attr("id", `${type}-gradient`);

    gradient
      .append("stop")
      .attr("offset", "0%")
      .attr("stop-color", color)
      .attr("stop-opacity", 0.8);

    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", color)
      .attr("stop-opacity", 0);
  });

  // Update force simulation
  reinitializeSimulation(); // Reinitialize the simulation

  // Draw links with dot product values
  const links = networkContainer
    .append("g")
    .selectAll("line")
    .data(network.links)
    .enter()
    .append("line")
    .attr("stroke", (d) => {
      const value = d.source === "input-vector" ? d.weight : Math.abs(d.weight);
      return d3.interpolateRdYlBu(1 - (value + 1) / 2);
    })
    .attr("stroke-width", (d) => Math.abs(d.weight) * 2 + 1);

  // Add dot product labels
  networkContainer
    .append("g")
    .selectAll("text")
    .data(network.links.filter((l) => l.source === "input-vector"))
    .enter()
    .append("text")
    .attr("class", "link-label")
    .attr("fill", "white")
    .attr("font-size", "10px")
    .text((d) => d.weight.toFixed(2));

  // Draw nodes
  const nodes = networkContainer
    .append("g")
    .selectAll("g")
    .data(network.nodes)
    .enter()
    .append("g")
    .on("click", (event, d) => showNeuronDetails(event, d))
    .call(
      d3
        .drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
        })
    );

  // Node glow
  nodes
    .append("circle")
    .attr("r", 30)
    .attr("fill", (d) => `url(#${d.type}-gradient)`);

  // Node circle
  nodes
    .append("circle")
    .attr("r", (d) => (d.type === "concat" ? 20 : 15)) // Make concat nodes bigger
    .attr("fill", (d) => colors[d.type])
    .attr("stroke", "white")
    .attr("stroke-width", (d) => (d.type === "concat" ? 3 : 2))
    .attr("stroke-dasharray", (d) => (d.type === "concat" ? "4,2" : "none"));

  // Vector labels
  if (document.getElementById("showVectors").checked) {
    nodes
      .append("text")
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .attr("font-family", "monospace")
      .attr("font-size", "12px")
      .text((d) => `[${d.vector.map((v) => v.toFixed(1)).join(",")}]`);

    nodes
      .filter((d) => d.type === "hidden")
      .append("text")
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .attr("font-size", "12px")
      .text((d) => `out: ${d.output?.toFixed(2)}`);
  }

  // Update positions
  simulation.on("tick", () => {
    // Ensure all coordinates are valid numbers
    links
      .attr("x1", (d) => (isFinite(d.source.x) ? d.source.x : 0))
      .attr("y1", (d) => (isFinite(d.source.y) ? d.source.y : 0))
      .attr("x2", (d) => (isFinite(d.target.x) ? d.target.x : 0))
      .attr("y2", (d) => (isFinite(d.target.y) ? d.target.y : 0));

    nodes.attr("transform", (d) => {
      const x = isFinite(d.x) ? d.x : 0;
      const y = isFinite(d.y) ? d.y : 0;
      return `translate(${x},${y})`;
    });
  });

  updateVectorSpaceVisualization();
}

function updateVectorSpaceVisualization() {
  const margin = 40;
  const width = document.getElementById("vectorSpace").clientWidth;
  const height = document.getElementById("vectorSpace").clientHeight;

  // Clear previous visualization
  vectorSpaceSvg.selectAll("*").remove();

  // Draw background and grid
  vectorSpaceSvg
    .append("rect")
    .attr("width", width)
    .attr("height", height)
    .attr("fill", "rgba(30, 30, 30, 0.8)");

  // Draw grid
  for (let i = -2; i <= 2; i++) {
    vectorSpaceSvg
      .append("line")
      .attr("x1", margin + ((width - 2 * margin) * (i + 2)) / 4)
      .attr("y1", margin)
      .attr("x2", margin + ((width - 2 * margin) * (i + 2)) / 4)
      .attr("y2", height - margin)
      .attr("stroke", "#ffffff20")
      .attr("stroke-dasharray", "2,2");

    vectorSpaceSvg
      .append("line")
      .attr("x1", margin)
      .attr("y1", margin + ((height - 2 * margin) * (i + 2)) / 4)
      .attr("x2", width - margin)
      .attr("y2", margin + ((height - 2 * margin) * (i + 2)) / 4)
      .attr("stroke", "#ffffff20")
      .attr("stroke-dasharray", "2,2");
  }

  // Draw axes
  vectorSpaceSvg
    .append("line")
    .attr("x1", margin)
    .attr("y1", height / 2)
    .attr("x2", width - margin)
    .attr("y2", height / 2)
    .attr("stroke", "#ffffff40")
    .attr("stroke-width", 1);

  vectorSpaceSvg
    .append("line")
    .attr("x1", width / 2)
    .attr("y1", margin)
    .attr("x2", width / 2)
    .attr("y2", height - margin)
    .attr("stroke", "#ffffff40")
    .attr("stroke-width", 1);

  // Plot vectors - now with safety checks
  network.nodes.forEach((node) => {
    if (!node.vector || node.vector.length < 2) return;

    // Ensure we have valid numbers for visualization
    const x = node.vector[0];
    const y = node.vector[1];
    
    if (isNaN(x) || isNaN(y)) return;

    const scaledX = margin + ((width - 2 * margin) * (x + 2)) / 4;
    const scaledY = margin + ((height - 2 * margin) * (2 - y)) / 4;

    if (isNaN(scaledX) || isNaN(scaledY)) return;

    // Draw vector point
    vectorSpaceSvg
      .append("circle")
      .attr("cx", scaledX)
      .attr("cy", scaledY)
      .attr("r", node.type === "input" ? 6 : 4)
      .attr("fill", colors[node.type])
      .attr("stroke", "white")
      .attr("stroke-width", 1);

    // Draw vector line from origin with safety checks
    const originX = width / 2;
    const originY = height / 2;
    
    vectorSpaceSvg
      .append("line")
      .attr("x1", originX)
      .attr("y1", originY)
      .attr("x2", scaledX)
      .attr("y2", scaledY)
      .attr("stroke", `${colors[node.type]}40`)
      .attr("stroke-width", node.type === "input" ? 2 : 1);
  });
}

function showNeuronDetails(event, neuron) {
  hideNeuronDetails();

  const detailsPanel = document.createElement("div");
  detailsPanel.className = "neuron-details";
  detailsPanel.id = "neuron-details";

  detailsPanel.style.left = `${event.pageX + 10}px`;
  detailsPanel.style.top = `${event.pageY + 10}px`;

  let content = `<h4>${
    neuron.type.charAt(0).toUpperCase() + neuron.type.slice(1)
  } ${neuron.type !== "concat" ? `Neuron ${neuron.index + 1}` : "Layer"}</h4>`;

  // Helper function to safely format numbers
  const safeFormat = (value, decimals = 2) => {
    if (value === undefined || value === null || isNaN(value)) return "0.00";
    return Number(value).toFixed(decimals);
  };

  // Helper function to safely get vector values
  const safeVector = (vector) => {
    if (!Array.isArray(vector)) return [];
    return vector.map(v => isNaN(v) ? 0 : v);
  };

  if (neuron.type === "output") {
    content += `
      <div class="vector-editor">
        <label>Weight Vector (from concatenated inputs):</label>
        ${safeVector(neuron.vector)
          .map((val, idx) => `
          <div class="weight-input">
            <label>Weight ${idx + 1}:</label>
            <input type="number" step="0.1" value="${safeFormat(val)}" 
                   onchange="window.updateNeuronVector('${neuron.id}', ${idx}, this.value)" />
          </div>
        `).join("")}
      </div>
      <div class="weight-input" style="margin-top: 10px">
        <label>Output:</label>
        <input type="number" value="${safeFormat(neuron.output, 4)}" disabled />
      </div>
    `;
  } else if (neuron.type === "softmax") {
    const outputs = Array.isArray(neuron.output) ? neuron.output : [neuron.output];
    content += `
      <div class="vector-editor">
        <label>Softmax Probabilities:</label>
        ${outputs.map((val, idx) => `
          <div class="weight-input">
            <label>Class ${idx + 1}:</label>
            <input type="number" value="${safeFormat(val, 4)}" disabled />
          </div>
        `).join("")}
      </div>
    `;
  } else {
    // Add vector editor for all neuron types except concat
    if (neuron.type !== "concat") {
      content += `
        <div class="vector-editor">
          <label>Vector Values:</label>
          ${safeVector(neuron.vector)
            .map((val, idx) => `
            <div class="weight-input">
              <label>Dimension ${idx + 1}:</label>
              <input type="number" step="0.1" value="${safeFormat(val)}" 
                     onchange="window.updateNeuronVector('${neuron.id}', ${idx}, this.value)" />
            </div>
          `).join("")}
        </div>
      `;
    }

    // Show weights for all nodes except input
    if (neuron.type !== "input" && neuron.type !== "concat") {
      const incomingLinks = network.links.filter(l => l.target === neuron.id);
      if (incomingLinks.length > 0) {
        content += '<div style="margin: 10px 0"><strong>Input Weights:</strong></div>';
        incomingLinks.forEach(link => {
          const sourceNode = network.nodes.find(n => n.id === link.source);
          content += `
            <div class="weight-input">
              <label>From ${sourceNode.type} ${sourceNode.type !== "concat" ? sourceNode.index + 1 : ""}</label>
              <input type="number" step="0.1" value="${safeFormat(link.weight)}" 
                     onchange="window.updateWeight('${link.source}', '${link.target}', this.value)" />
            </div>
          `;
        });
      }
    }

    // Show output value for hidden and output nodes
    if (neuron.type === "hidden" || neuron.type === "output") {
      content += `
        <div class="weight-input" style="margin-top: 10px">
          <label>Output:</label>
          <input type="number" value="${safeFormat(neuron.output)}" disabled />
        </div>
      `;
    }
  }

  detailsPanel.innerHTML = content;
  document.body.appendChild(detailsPanel);

  setTimeout(() => {
    document.addEventListener("click", hideNeuronDetailsOnClickOutside);
  }, 0);
}

function hideNeuronDetails() {
  const existingPanel = document.getElementById("neuron-details");
  if (existingPanel) {
    existingPanel.remove();
    document.removeEventListener("click", hideNeuronDetailsOnClickOutside);
  }
}

function hideNeuronDetailsOnClickOutside(event) {
  const panel = document.getElementById("neuron-details");
  if (panel && !panel.contains(event.target)) {
    hideNeuronDetails();
  }
}

function updateNeuronValue(neuronId, value) {
  const neuron = network.nodes.find((n) => n.id === neuronId);
  neuron.value = parseFloat(value);
  forwardPass();
  updateVisualization();
}

function updateWeight(sourceId, targetId, value) {
  const link = network.links.find(
    (l) => l.source === sourceId && l.target === targetId
  );
  link.weight = parseFloat(value);
  forwardPass();
  updateVisualization();
}

window.updateNeuronVector = function (neuronId, dimension, value) {
  const neuron = network.nodes.find((n) => n.id === neuronId);
  if (neuron && neuron.vector) {
    neuron.vector[dimension] = parseFloat(value);
    recalculateNetwork();
    updateVisualization();
  }
};

window.updateNeuronValue = function (neuronId, value) {
  const neuron = network.nodes.find((n) => n.id === neuronId);
  if (neuron) {
    neuron.value = parseFloat(value);
    recalculateNetwork();
    forwardPass();
    updateVisualization();
  }
};

window.updateWeight = function (sourceId, targetId, value) {
  const link = network.links.find(
    (l) => l.source === sourceId && l.target === targetId
  );
  if (link) {
    link.weight = parseFloat(value);
    recalculateNetwork();
    forwardPass();
    updateVisualization();
  }
};

window.updateInputAndPropagate = function(neuronId, dimension, value) {
  const neuron = network.nodes.find(n => n.id === neuronId);
  if (neuron && neuron.vector) {
    // Update the input value
    neuron.vector[dimension] = parseFloat(value);
    
    // Set initial output for input neuron
    neuron.output = neuron.vector[0];
    
    // Recalculate all weights based on new input
    network.links.forEach(link => {
      if (link.source === neuronId) {
        const targetNode = network.nodes.find(n => n.id === link.target);
        if (targetNode) {
          link.weight = dotProduct(neuron.vector, targetNode.vector);
        }
      }
    });

    // Recalculate the entire network
    recalculateNetwork();
    
    // Update visualization
    updateVisualization();
  }
};

export { networkContainer, showNeuronDetails, networkSvg, networkZoom };
