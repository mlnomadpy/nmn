// Network configuration
export const config = {
  inputSize: 3,
  hiddenLayers: [4],
  outputSize: 2,
  learningRate: 0.1,
  batchSize: 1,
  trainingSpeed: 50,
  layers: [
    { type: "input", size: 3 },
    { type: "hidden", size: 4 },
    { type: "output", size: 2 },
  ],
};

// Network state
export let network = {
  nodes: [],
  links: [],
  weights: [],
};

// Helper functions
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
export const dotProduct = (vec1, vec2) =>
  vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);

// Add softmax helper function at the top with other helpers
export const softmax = (vector) => {
  const expValues = vector.map((v) => Math.exp(v));
  const sumExp = expValues.reduce((sum, v) => sum + v, 0);
  return expValues.map((v) => v / sumExp);
};

// Update activation functions
export const activations = {
  relu: (x) => Math.max(0, x),
  gelu: (x) =>
    x *
    0.5 *
    (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  softmax: (input) => {
    // Ensure input is an array
    const arr = Array.isArray(input) ? input : [input];
    // Apply exp to all values and normalize
    const expValues = arr.map((x) => Math.exp(x));
    const sumExp = expValues.reduce((sum, val) => sum + val, 0);
    return expValues.map((x) => x / sumExp);
  },
};

// Add this new function
export function recalculateNetwork() {
  // Process each layer in order, ensuring proper forward propagation
  config.layers.forEach((layer, layerIndex) => {
    const layerNodes = network.nodes.filter(n => n.layer === layerIndex);
    
    layerNodes.forEach(node => {
      switch (node.type) {
        case 'input':
          // Input nodes keep their vector values and set output to first element
          node.output = node.vector[0];
          break;

        case 'hidden':
          // Get incoming connections
          const incomingLinks = network.links.filter(l => l.target === node.id);
          const inputs = incomingLinks.map(link => {
            const sourceNode = network.nodes.find(n => n.id === link.source);
            if (sourceNode.type === 'concat') {
              // For concat nodes, use their vector values
              return sourceNode.vector.map(v => v * link.weight);
            } else if (sourceNode.type === 'input') {
              return sourceNode.vector.map(v => v * link.weight);
            }
            return (sourceNode.output || 0) * link.weight;
          }).flat();
          
          // Sum all inputs and apply activation
          const sum = inputs.reduce((acc, val) => acc + val, 0);
          node.output = sigmoid(sum);
          break;

        case 'output':
          // Get incoming connections
          const outputLinks = network.links.filter(l => l.target === node.id);
          const prevLayer = network.nodes.filter(n => n.layer === layerIndex - 1);
          
          // Calculate weighted sum of inputs
          const weightedSum = outputLinks.reduce((sum, link) => {
            const sourceNode = network.nodes.find(n => n.id === link.source);
            return sum + (sourceNode.output || 0) * link.weight;
          }, 0);
          
          node.output = sigmoid(weightedSum);
          break;

        case 'activation':
          const sourceNode = network.nodes.find(n => n.layer === layerIndex - 1);
          if (sourceNode) {
            if (node.activationType === 'softmax') {
              // Get all outputs from previous layer for softmax
              const prevOutputs = prevLayer.map(n => n.output);
              node.output = activations.softmax(prevOutputs);
            } else {
              // Apply activation function to single input
              node.output = activations[node.activationType](sourceNode.output);
            }
          }
          break;

        case 'concat':
          // Get all nodes from previous layer
          const prevNodes = network.nodes.filter(n => n.layer === layerIndex - 1);
          
          // Collect all values to concatenate
          const valuesToConcat = prevNodes.map(prevNode => {
            // If the previous node has a vector, take all values
            if (Array.isArray(prevNode.vector)) {
              return prevNode.vector;
            }
            // If it has an output, take that
            if (typeof prevNode.output !== 'undefined') {
              return [prevNode.output];
            }
            // Fallback to 0
            return [0];
          }).flat();

          // Store concatenated values in both vector and output
          node.vector = valuesToConcat;
          // Set output as first value for compatibility
          node.output = valuesToConcat[0] || 0;
          break;
      }
    });
  });
}

// Initialize the network
export function initializeNetwork() {
  const nodes = [];
  const links = [];
  let currentLayer = 0;

  if (config.layers && config.layers.length > 0) {
    // Process layers in order
    config.layers.forEach((layerConfig, idx) => {
      switch (layerConfig.type) {
        case "input":
          nodes.push({
            id: "input-vector",
            layer: currentLayer,
            type: "input",
            vector: Array(config.inputSize)
              .fill(0)
              .map(() => Math.random() * 2 - 1),
          });
          break;

        case "activation":
          nodes.push({
            id: `activation-${currentLayer}`,
            layer: currentLayer,
            type: "activation",
            activationType: layerConfig.activation,
            vector: [],
            output: [],
          });

          // Connect to previous layer
          if (nodes.length > 1) {
            links.push({
              source: nodes[nodes.length - 2].id,
              target: `activation-${currentLayer}`,
              weight: 1,
            });
          }
          break;

        case "hidden":
          for (let i = 0; i < layerConfig.size; i++) {
            nodes.push({
              id: `hidden-${currentLayer}-${i}`,
              layer: currentLayer,
              index: i,
              type: "hidden",
              vector: Array(config.inputSize)
                .fill(0)
                .map(() => Math.random() * 2 - 1),
              output: 0,
            });

            // Connect to previous layer
            if (nodes.length > 1) {
              const prevLayer = nodes.filter(
                (n) => n.layer === currentLayer - 1
              );
              prevLayer.forEach((prevNode) => {
                links.push({
                  source: prevNode.id,
                  target: `hidden-${currentLayer}-${i}`,
                  weight: Math.random() * 2 - 1,
                });
              });
            }
          }
          break;

        case "concat":
          nodes.push({
            id: `concat-${currentLayer}`,
            layer: currentLayer,
            type: "concat",
            vector: [],
          });

          // Connect to source layer
          if (layerConfig.sourceLayer1) {
            const sourceNodes = nodes.filter(
              (n) => n.type !== "concat" && n.layer === currentLayer - 1
            );
            sourceNodes.forEach((sourceNode) => {
              links.push({
                source: sourceNode.id,
                target: `concat-${currentLayer}`,
                weight: 1,
              });
            });
          }
          break;

        case "fusion":
          nodes.push({
            id: `fusion-${currentLayer}`,
            layer: currentLayer,
            type: "fusion",
            vector: [],
          });

          // Connect both source layers
          if (layerConfig.sourceLayer1 && layerConfig.sourceLayer2) {
            [layerConfig.sourceLayer1, layerConfig.sourceLayer2].forEach(
              (sourceId) => {
                links.push({
                  source: sourceId,
                  target: `fusion-${currentLayer}`,
                  weight: 1,
                });
              }
            );
          }
          break;

        case "output":
          // Create multiple output nodes based on outputSize
          for (let i = 0; i < config.outputSize; i++) {
            const prevNode = nodes.find((n) => n.layer === currentLayer - 1);
            const vectorSize = prevNode?.vector?.length || config.inputSize;

            nodes.push({
              id: `output-${i}`,
              layer: currentLayer,
              type: "output",
              index: i,
              vector: Array(vectorSize)
                .fill(0)
                .map(() => Math.random() * 2 - 1),
              output: 0,
            });

            // Connect to previous layer
            if (prevNode) {
              links.push({
                source: prevNode.id,
                target: `output-${i}`,
                weight: Math.random() * 2 - 1,
              });
              // Calculate initial output
              const outputNode = nodes[nodes.length - 1];
              outputNode.output = sigmoid(
                dotProduct(prevNode.vector, outputNode.vector)
              );
            }
          }
          break;
      }
      currentLayer++;
    });
  } else {
    // Use old initialization code
    // Get initial dimensions for positioning
    const width = document.getElementById("network")?.clientWidth || 800;
    const height = document.getElementById("network")?.clientHeight || 600;

    // Initialize input node with position
    nodes.push({
      id: "input-vector",
      layer: 0,
      type: "input",
      vector: Array(config.inputSize)
        .fill(0)
        .map(() => Math.random() * 2 - 1),
      x: width * 0.2, // Set initial position
      y: height * 0.5,
    });

    // Hidden layers with concatenation layers between them
    let currentLayer = 1;
    config.hiddenLayers.forEach((layerSize, layerIndex) => {
      // Add hidden layer
      for (let i = 0; i < layerSize; i++) {
        nodes.push({
          id: `hidden-${currentLayer}-${i}`,
          layer: currentLayer,
          index: i,
          type: "hidden",
          vector: Array(config.inputSize)
            .fill(0)
            .map(() => Math.random() * 2 - 1),
          output: Math.random(),
          x: width * ((currentLayer + 1) / (config.hiddenLayers.length + 2)),
          y: height * 0.5,
        });

        // Connect to previous layer or input
        const prevNodeId =
          layerIndex === 0 ? "input-vector" : `concat-${currentLayer - 1}`;
        links.push({
          source: prevNodeId,
          target: `hidden-${currentLayer}-${i}`,
          weight: dotProduct(nodes[0].vector, nodes[nodes.length - 1].vector),
        });
      }

      // Add concatenation layer after hidden layer
      nodes.push({
        id: `concat-${currentLayer}`,
        layer: currentLayer + 0.5,
        type: "concat",
        vector: [], // Will be populated during forward pass
      });

      // Connect all hidden layer nodes to concatenation node
      for (let i = 0; i < layerSize; i++) {
        links.push({
          source: `hidden-${currentLayer}-${i}`,
          target: `concat-${currentLayer}`,
          weight: 1, // Fixed weight for concatenation
        });
      }

      currentLayer += 1;
    });

    // Output layer
    for (let i = 0; i < config.outputSize; i++) {
      // Calculate input size for output layer based on last concatenation layer
      const lastConcatSize =
        config.hiddenLayers[config.hiddenLayers.length - 1];

      nodes.push({
        id: `output-${i}`,
        layer: currentLayer,
        index: i,
        type: "output",
        vector: Array(lastConcatSize) // Vector size matches previous layer size
          .fill(0)
          .map(() => Math.random() * 2 - 1),
        output: 0,
      });

      // Connect to last concatenation layer
      links.push({
        source: `concat-${currentLayer - 1}`,
        target: `output-${i}`,
        weight: Math.random() * 2 - 1,
      });
    }

    // Add final concatenation layer after output
    nodes.push({
      id: `concat-final`,
      layer: currentLayer + 0.5,
      type: "concat",
      vector: [], // Will be populated during forward pass
    });

    // Connect all output nodes to final concatenation
    for (let i = 0; i < config.outputSize; i++) {
      links.push({
        source: `output-${i}`,
        target: "concat-final",
        weight: 1, // Fixed weight for concatenation
      });
    }

    // Add softmax layer after final concatenation
    nodes.push({
      id: "softmax",
      layer: currentLayer + 1,
      type: "softmax",
      vector: [], // Will be populated with softmax probabilities
      output: [], // Will store softmax outputs
    });

    // Connect final concatenation to softmax
    links.push({
      source: "concat-final",
      target: "softmax",
      weight: 1,
    });
  }

  network = { nodes, links };
  // Calculate initial outputs
  recalculateNetwork();
}
