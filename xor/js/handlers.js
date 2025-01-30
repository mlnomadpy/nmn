import { optimize, optimizeAll } from './optimization.js';  // Add optimizeAll to imports
import { dotNeuron, yatNeuron, posiYatNeuron } from './neurons.js';
import { 
  dotParams, 
  yatParams, 
  posiParams, 
  dotWeightPath, 
  yatWeightPath, 
  posiWeightPath,
  dotPlot,
  yatPlot,
  posiPlot,
  dotWeightPlot,
  yatWeightPlot,
  posiWeightPlot,
  historyPlots,
  combinedPlot  // Add this
} from './main.js';
import { updatePlot, updateWeightPlot, updateCombinedPlot } from './plots.js';

function optimizeDot() {
  const iterations = parseInt(document.getElementById("dot-iterations").value);
  const learningRate = parseFloat(document.getElementById("dot-learning-rate").value);
  const method = document.getElementById("dot-optimization-method").value;
  optimize(
    dotNeuron,
    dotParams,
    dotPlot,
    dotWeightPlot,
    dotWeightPath,
    document.getElementById("dot-loss"),
    historyPlots.dot,
    iterations,
    learningRate,
    method
  );
}

function optimizeYat() {
  const iterations = parseInt(document.getElementById("yat-iterations").value);
  const learningRate = parseFloat(document.getElementById("yat-learning-rate").value);
  const method = document.getElementById("yat-optimization-method").value;
  optimize(
    yatNeuron,
    yatParams,
    yatPlot,
    yatWeightPlot,
    yatWeightPath,
    document.getElementById("yat-loss"),
    historyPlots.yat,
    iterations,
    learningRate,
    method
  );
}

function optimizePosi() {
  const iterations = parseInt(document.getElementById("posi-iterations").value);
  const learningRate = parseFloat(document.getElementById("posi-learning-rate").value);
  const method = document.getElementById("posi-optimization-method").value;
  optimize(
    posiYatNeuron,
    posiParams,
    posiPlot,
    posiWeightPlot,
    posiWeightPath,
    document.getElementById("posi-loss"),
    historyPlots.posi,
    iterations,
    learningRate,
    method
  );
}

function resetDot() {
  resetParams(dotParams);
  dotWeightPath.length = 0;
  updatePlot(dotPlot, dotNeuron, dotParams);
  updateWeightPlot(dotWeightPlot, dotParams, dotWeightPath);
}

function resetYat() {
  resetParams(yatParams);
  yatWeightPath.length = 0;
  updatePlot(yatPlot, yatNeuron, yatParams);
  updateWeightPlot(yatWeightPlot, yatParams, yatWeightPath);
}

function resetPosi() {
  resetParams(posiParams);
  posiWeightPath.length = 0;
  updatePlot(posiPlot, posiYatNeuron, posiParams);
  updateWeightPlot(posiWeightPlot, posiParams, posiWeightPath);
}

function resetParams(params) {
  params.w1 = Math.random() - 0.5;
  params.w2 = Math.random() - 0.5;
  params.b = Math.random() - 0.5;
}

function optimizeAllNeurons() {
  const learningRate = parseFloat(document.getElementById("combined-learning-rate").value);
  const method = document.getElementById("combined-optimization-method").value;
  // Initialize combined weight paths
  const combinedWeightPaths = {
    dot: [],
    yat: [],
    posi: []
  };

  const neurons = {
    dot: { fn: dotNeuron, params: dotParams, weightPath: combinedWeightPaths.dot },
    yat: { fn: yatNeuron, params: yatParams, weightPath: combinedWeightPaths.yat },
    posi: { fn: posiYatNeuron, params: posiParams, weightPath: combinedWeightPaths.posi }
  };

  const lossDisplays = {
    dot: document.getElementById('dot-combined-loss'),
    yat: document.getElementById('yat-combined-loss'),
    posi: document.getElementById('posi-combined-loss')
  };

  const combinedWeightPlot = createWeightPlot("#combined-weight-plot");
  
  optimizeAll(neurons, combinedPlot, combinedWeightPlot, lossDisplays, learningRate, method);
}

function resetAllNeurons() {
  resetDot();
  resetYat();
  resetPosi();
  
  // Reset combined plot
  const neurons = {
    dot: { fn: dotNeuron, params: dotParams },
    yat: { fn: yatNeuron, params: yatParams },
    posi: { fn: posiYatNeuron, params: posiParams }
  };
  updateCombinedPlot(combinedPlot, neurons);
}

function runComparison() {
  const neuron1 = document.getElementById('neuron1-select').value;
  const neuron2 = document.getElementById('neuron2-select').value;
  
  const comparisonView = document.querySelector('.comparison-view');
  if (!comparisonView) return;
  
  comparisonView.innerHTML = '';
  
  const neuron1Card = document.getElementById(`${neuron1}-card`);
  const neuron2Card = document.getElementById(`${neuron2}-card`);
  
  if (!neuron1Card || !neuron2Card) {
    console.error('Could not find neuron cards');
    return;
  }
  
  const neuron1Clone = neuron1Card.cloneNode(true);
  const neuron2Clone = neuron2Card.cloneNode(true);
  
  // Update IDs to avoid duplicates
  neuron1Clone.id = `${neuron1}-comparison`;
  neuron2Clone.id = `${neuron2}-comparison`;
  
  comparisonView.appendChild(neuron1Clone);
  comparisonView.appendChild(neuron2Clone);
}

// Remove individual exports from function declarations and use a single export statement
export {
  optimizeDot,
  optimizeYat,
  optimizePosi,
  resetDot,
  resetYat,
  resetPosi,
  optimizeAllNeurons,
  resetAllNeurons,
  runComparison
};

// Make functions available on window object
Object.assign(window, {
  optimizeDot,
  optimizeYat,
  optimizePosi,
  resetDot,
  resetYat,
  resetPosi,
  optimizeAllNeurons,
  resetAllNeurons,
  runComparison
});