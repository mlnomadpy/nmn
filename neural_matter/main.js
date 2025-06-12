import { initializeNetwork, config } from "./network.js";
import {
  updateVisualization,
  networkContainer,
  showNeuronDetails,
  reinitializeSimulation,
  networkSvg,
  networkZoom,
} from "./visualization.js";
import { trainStep, forwardPass } from "./training.js";

let isTraining = false;
let epoch = 0;
let error = 0;

document.addEventListener("DOMContentLoaded", function () {
  // Initialize default layers configuration if not exists
  if (!config.layers) {
    config.layers = [
      { type: "input", size: config.inputSize },
      ...config.hiddenLayers.map((size) => ({ type: "hidden", size })),
      { type: "output", size: config.outputSize },
    ];
  }

  // Initialize the network
  initializeNetwork();
  // Initialize layer configurations UI
  initializeLayerConfigs();
  updateVisualization();

  // Event listeners
  const inputDimension = document.getElementById("inputDimension");
  if (inputDimension) {
    inputDimension.addEventListener("change", updateInputDimension);
  }

  // Add output size change listener
  const outputSize = document.getElementById("outputSize");
  if (outputSize) {
    outputSize.addEventListener("change", updateOutputSize);
  }

  document.getElementById("addLayerBtn").addEventListener("click", addLayer);
  document
    .getElementById("trainButton")
    .addEventListener("click", toggleTraining);
  document
    .getElementById("resetButton")
    .addEventListener("click", resetNetwork);
  document
    .getElementById("centerViewButton")
    .addEventListener("click", centerView);
  document
    .getElementById("showVectors")
    .addEventListener("change", updateVisualization);
  document
    .getElementById("showForces")
    .addEventListener("change", updateVisualization);
  document
    .getElementById("learningRate")
    .addEventListener("change", updateConfig);
  document.getElementById("batchSize").addEventListener("change", updateConfig);
  document
    .getElementById("trainingSpeed")
    .addEventListener("change", updateConfig);

  // Center the view after a short delay
  setTimeout(centerView, 100);
});

function initializeLayerConfigs() {
  const layerConfigs = document.getElementById("layerConfigs");
  layerConfigs.innerHTML = ""; // Clear existing configs

  // Create sortable container
  const sortableContainer = document.createElement("div");
  sortableContainer.id = "sortableLayers";
  layerConfigs.appendChild(sortableContainer);

  // Make layers sortable
  new Sortable(sortableContainer, {
    animation: 150,
    handle: ".layer-config", // Drag by the layer div
    onEnd: function (evt) {
      const newIndex = evt.newIndex;
      const oldIndex = evt.oldIndex;

      // Don't allow moving input or output layers
      if (
        config.layers[oldIndex].type === "input" ||
        config.layers[oldIndex].type === "output" ||
        config.layers[newIndex].type === "input" ||
        config.layers[newIndex].type === "output"
      ) {
        // Revert the move
        sortableContainer.insertBefore(
          evt.item,
          sortableContainer.children[oldIndex]
        );
        return;
      }

      // Update config.layers array
      const layer = config.layers.splice(oldIndex, 1)[0];
      config.layers.splice(newIndex, 0, layer);

      // Update network
      initializeNetwork();
      updateVisualization();
    },
  });

  // Show all layers in the UI
  if (config.layers) {
    config.layers.forEach((layer, index) => {
      const layerDiv = document.createElement("div");
      layerDiv.className = "layer-config";

      // Add drag handle
      layerDiv.style.cursor = "move";

      // Create layer configuration UI based on type
      switch (layer.type) {
        case "input":
          layerDiv.innerHTML = `
            <label>Input Layer</label>
            <span class="layer-info">Dimension: ${config.inputSize}</span>
          `;
          break;

        case "hidden":
          layerDiv.innerHTML = `
            <label>Hidden Layer ${index}</label>
            <input type="number" class="layer-size" value="${layer.size}" min="1" max="10"/>
            <button class="remove-layer" onclick="removeLayer(${index})">×</button>
          `;
          break;

        case "activation":
          layerDiv.innerHTML = `
            <label>Activation Layer</label>
            <select class="activation-type" data-index="${index}">
              <option value="relu" ${
                layer.activation === "relu" ? "selected" : ""
              }>ReLU</option>
              <option value="gelu" ${
                layer.activation === "gelu" ? "selected" : ""
              }>GELU</option>
              <option value="sigmoid" ${
                layer.activation === "sigmoid" ? "selected" : ""
              }>Sigmoid</option>
              <option value="softmax" ${
                layer.activation === "softmax" ? "selected" : ""
              }>Softmax</option>
            </select>
            <button class="remove-layer" onclick="removeLayer(${index})">×</button>
          `;
          break;

        case "concat":
          layerDiv.innerHTML = `
            <label>Concatenation Layer</label>
            <span class="layer-info">From previous layer</span>
            <button class="remove-layer" onclick="removeLayer(${index})">×</button>
          `;
          break;

        case "fusion":
          layerDiv.innerHTML = `
            <label>Fusion Layer</label>
            <span class="layer-info">From: ${
              layer.sourceLayer1?.split("-")[0] || "unknown"
            }, 
                                         ${
                                           layer.sourceLayer2?.split("-")[0] ||
                                           "unknown"
                                         }</span>
            <button class="remove-layer" data-index="${index}">×</button>
          `;
          break;

        case "output":
          layerDiv.innerHTML = `
            <label>Output Layer</label>
            <span class="layer-info">Size: ${config.outputSize}</span>
          `;
          break;
      }

      // Add event listeners for inputs
      const sizeInput = layerDiv.querySelector(".layer-size");
      if (sizeInput) {
        sizeInput.addEventListener("change", (e) => {
          config.layers[index].size = parseInt(e.target.value);
          initializeNetwork();
          updateVisualization();
        });
      }

      const activationSelect = layerDiv.querySelector(".activation-type");
      if (activationSelect) {
        activationSelect.addEventListener("change", (e) => {
          config.layers[index].activation = e.target.value;
          initializeNetwork();
          updateVisualization();
        });
      }

      sortableContainer.appendChild(layerDiv);
    });
  }
}

function updateInputDimension(event) {
  const newDimension = parseInt(event.target.value);
  if (newDimension >= 2 && newDimension <= 10) {
    config.inputSize = newDimension;
    resetNetwork();
  }
}

function updateOutputSize(event) {
  const newSize = parseInt(event.target.value);
  if (newSize >= 1 && newSize <= 10) {
    config.outputSize = newSize;
    resetNetwork();
  }
}

function addLayer() {
  const dialog = document.createElement("div");
  dialog.className = "layer-dialog";
  dialog.innerHTML = `
    <h3>Add New Layer</h3>
    <button class="layer-type-btn" data-type="input">Input Layer</button>
    <button class="layer-type-btn" data-type="hidden">Hidden Layer</button>
    <button class="layer-type-btn" data-type="concat">Concatenation Layer</button>
    <button class="layer-type-btn" data-type="activation">Activation Layer</button>
    <button class="layer-type-btn" data-type="fusion">Fusion Layer</button>
    <button class="layer-type-btn" data-type="output">Output Layer</button>
  `;

  document.body.appendChild(dialog);

  dialog.querySelectorAll(".layer-type-btn").forEach((btn) => {
    btn.addEventListener("click", () =>
      showLayerOptions(btn.dataset.type, dialog)
    );
  });
}

function showLayerOptions(type, dialog) {
  const options = document.createElement("div");
  options.className = "layer-dialog-options";

  switch (type) {
    case "activation":
      options.innerHTML = `
        <select id="activation-type">
          <option value="relu">ReLU</option>
          <option value="gelu">GELU</option>
          <option value="softmax">Softmax</option>
          <option value="sigmoid">Sigmoid</option>
        </select>
      `;
      break;

    case "concat":
    case "fusion":
      // Use config.layers instead of network.nodes
      const layers = config.layers
        .filter(
          (layer, idx) =>
            layer.type !== "concat" &&
            layer.type !== "fusion" &&
            idx < config.layers.length - 1
        ) // Exclude output layer
        .map(
          (layer, idx) =>
            `<option value="layer-${idx}">${layer.type} ${idx}</option>`
        );

      options.innerHTML = `
        <select id="source-layer-1">
          <option value="">Select first layer</option>
          ${layers.join("")}
        </select>
        ${
          type === "fusion"
            ? `
          <select id="source-layer-2">
            <option value="">Select second layer</option>
            ${layers.join("")}
          </select>
        `
            : ""
        }
      `;
      break;

    case "hidden":
      options.innerHTML = `
        <div class="form-group">
          <label>Layer Size</label>
          <input type="number" id="layer-size" value="4" min="1" max="10">
        </div>
      `;
      break;
  }

  options.innerHTML += `
    <div class="dialog-buttons">
      <button onclick="this.closest('.layer-dialog').remove()">Cancel</button>
      <button onclick="createNewLayer('${type}')">Add Layer</button>
    </div>
  `;

  // Remove any existing options
  dialog.querySelector(".layer-dialog-options")?.remove();
  dialog.appendChild(options);
}

window.createNewLayer = function (type) {
  const layerConfig = {
    type,
    size:
      type === "hidden"
        ? parseInt(document.getElementById("layer-size")?.value || 4)
        : 1,
    activation: document.getElementById("activation-type")?.value,
    sourceLayer1: document.getElementById("source-layer-1")?.value,
    sourceLayer2: document.getElementById("source-layer-2")?.value,
  };

  // Initialize layers array if it doesn't exist
  if (!config.layers) {
    config.layers = [
      { type: "input", size: config.inputSize },
      ...config.hiddenLayers.map((size) => ({ type: "hidden", size })),
      { type: "output", size: config.outputSize },
    ];
  }

  // Insert new layer before output layer
  const outputLayerIndex = config.layers.findIndex((l) => l.type === "output");
  if (outputLayerIndex !== -1) {
    config.layers.splice(outputLayerIndex, 0, layerConfig);
  } else {
    config.layers.push(layerConfig);
  }

  // Update network
  initializeNetwork();
  // Update UI
  initializeLayerConfigs();
  // Update visualization with new layout
  updateVisualization();

  // Close dialog
  document.querySelector(".layer-dialog").remove();
};

function toggleTraining() {
  isTraining = !isTraining;
  this.textContent = isTraining ? "Stop Training" : "Start Training";
  this.style.background = isTraining ? "#dc2626" : "#2196f3";

  if (isTraining) {
    updateConfig();
    function train() {
      if (isTraining) {
        trainStep();
        setTimeout(train, 1000 / config.trainingSpeed);
      }
    }
    train();
  }
}

function resetNetwork() {
  epoch = 0;
  error = 0;
  document.getElementById("epoch").textContent = "0";
  document.getElementById("error").textContent = "0.0000";
  initializeNetwork();
  updateVisualization();
  reinitializeSimulation(); // Reinitialize the simulation
}

function centerView() {
  const bounds = networkContainer.node().getBBox();
  const fullWidth = document.getElementById("network").clientWidth;
  const fullHeight = document.getElementById("network").clientHeight;

  const scale = Math.min(
    (0.8 * fullWidth) / bounds.width,
    (0.8 * fullHeight) / bounds.height
  );

  const transform = d3.zoomIdentity
    .translate(
      (fullWidth - bounds.width * scale) / 2 - bounds.x * scale,
      (fullHeight - bounds.height * scale) / 2 - bounds.y * scale
    )
    .scale(scale);

  networkSvg.transition().duration(750).call(networkZoom.transform, transform);
}

function updateConfig() {
  config.learningRate = parseFloat(
    document.getElementById("learningRate").value
  );
  config.batchSize = parseInt(document.getElementById("batchSize").value);
  config.trainingSpeed = parseInt(
    document.getElementById("trainingSpeed").value
  );
}

function updateNetwork() {
  // Get all layer sizes from inputs
  const layerSizes = Array.from(document.querySelectorAll(".layer-size")).map(
    (input) => parseInt(input.value) || 4 // Default to 4 if invalid
  );

  // Only update if we have valid layer sizes
  if (layerSizes.length > 0) {
    config.hiddenLayers = layerSizes;
    initializeNetwork();
    initializeLayerConfigs(); // Refresh the UI to match the network
    updateVisualization();
    reinitializeSimulation();
  }
}

function removeLayer(button) {
  const layerDiv = button.parentElement;
  layerDiv.remove();
  // Update layer numbers
  const layers = document.querySelectorAll(".layer-config");
  layers.forEach((layer, index) => {
    layer.querySelector("label").textContent = `Layer ${index + 1}`;
  });
  updateNetwork();
  reinitializeSimulation(); // Reinitialize the simulation
}

// Make removeLayer globally available
window.removeLayer = function (index) {
  if (config.layers) {
    // Don't remove input or output layers
    if (
      config.layers[index].type !== "input" &&
      config.layers[index].type !== "output"
    ) {
      config.layers.splice(index, 1);
      initializeNetwork();
      initializeLayerConfigs();
      updateVisualization();
      reinitializeSimulation();
    }
  }
};
