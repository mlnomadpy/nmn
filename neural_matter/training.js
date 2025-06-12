import { network, config, sigmoid, dotProduct, recalculateNetwork } from "./network.js";

let trainingData = [];

export function trainStep() {
  if (!trainingData.length) return;

  const sample = trainingData[Math.floor(Math.random() * trainingData.length)];
  const { hiddenOutputs, outputs } = forwardPass(sample.input);

  // Calculate output layer error
  const outputErrors = outputs.map(
    (output, i) => (sample.output[i] - output) * output * (1 - output)
  );

  let totalError = outputErrors.reduce((sum, err) => sum + Math.abs(err), 0);

  // Update weights
  network.links.forEach((link) => {
    if (link.target.startsWith("output")) {
      const outputIndex = parseInt(link.target.split("-")[1]);
      const hiddenIndex = parseInt(link.source.split("-")[1]);
      const lastHiddenOutput = hiddenOutputs[hiddenOutputs.length - 1];
      link.weight +=
        config.learningRate *
        outputErrors[outputIndex] *
        lastHiddenOutput[hiddenIndex];
    }
  });

  // Update display
  document.getElementById("epoch").textContent = ++epoch;
  document.getElementById("error").textContent = totalError.toFixed(4);

  updateVisualization();
}

export function forwardPass(inputValues) {
  // Set input values
  const inputNode = network.nodes.find(n => n.type === 'input');
  if (inputNode) {
    inputNode.vector = inputValues;
    inputNode.output = inputValues[0];
  }

  // Process each layer in order
  config.layers.forEach((layer, idx) => {
    const layerNodes = network.nodes.filter(n => n.layer === idx);
    
    layerNodes.forEach(node => {
      switch (node.type) {
        case 'hidden':
          const inputs = network.links
            .filter(l => l.target === node.id)
            .map(l => {
              const sourceNode = network.nodes.find(n => n.id === l.source);
              return sourceNode.output || sourceNode.vector[0] || 0;
            });
          const weights = network.links
            .filter(l => l.target === node.id)
            .map(l => l.weight);
          node.output = sigmoid(dotProduct(inputs, weights));
          break;

        case 'output':
          const prevNode = network.nodes.find(n => n.layer === idx - 1);
          if (prevNode) {
            const input = prevNode.vector || [prevNode.output || 0];
            node.output = sigmoid(dotProduct(input, node.vector));
          }
          break;

        // Handle other layer types...
      }
    });
  });

  // Return final outputs
  const outputs = network.nodes
    .filter(n => n.type === 'output')
    .map(n => n.output);

  return {
    outputs,
    hiddenOutputs: network.nodes
      .filter(n => n.type === 'hidden')
      .map(n => n.output)
  };
}
