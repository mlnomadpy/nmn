import { data } from './config.js';
import { 
  updatePlot, 
  updateWeightPlot, 
  updateHistoryPlots,
  updateCombinedPlot,  // Add this import
  updateCombinedWeightPlot // Add this import
} from './plots.js';

function calculateLoss(neuronFn, params) {
  return data.reduce((sum, point) => {
    const output = neuronFn(point.x, point.y, params.w1, params.w2, params.b);
    return sum + Math.pow(output - point.label, 2);
  }, 0) / data.length;
}

function gradientDescent(neuronFn, params, learningRate = 0.01) {
  const epsilon = 1e-7;
  const gradients = { w1: 0, w2: 0, b: 0 };

  for (let param in params) {
    const originalValue = params[param];
    params[param] = originalValue + epsilon;
    const lossPlus = calculateLoss(neuronFn, params);
    params[param] = originalValue - epsilon;
    const lossMinus = calculateLoss(neuronFn, params);
    gradients[param] = (lossPlus - lossMinus) / (2 * epsilon);
    params[param] = originalValue;
  }

  for (let param in params) {
    params[param] -= learningRate * gradients[param];
  }

  return calculateLoss(neuronFn, params);
}

function calculateHessian(neuronFn, params) {
  const epsilon = 1e-4;
  const hessian = {
    w1w1: 0, w1w2: 0, w1b: 0,
    w2w1: 0, w2w2: 0, w2b: 0,
    bw1: 0, bw2: 0, bb: 0
  };
  
  // Calculate second derivatives
  for (let param1 of ['w1', 'w2', 'b']) {
    for (let param2 of ['w1', 'w2', 'b']) {
      const orig1 = params[param1];
      const orig2 = params[param2];
      
      // f(x+h, y+h)
      params[param1] = orig1 + epsilon;
      params[param2] = orig2 + epsilon;
      const fpp = calculateLoss(neuronFn, params);
      
      // f(x+h, y-h)
      params[param2] = orig2 - epsilon;
      const fpm = calculateLoss(neuronFn, params);
      
      // f(x-h, y+h)
      params[param1] = orig1 - epsilon;
      params[param2] = orig2 + epsilon;
      const fmp = calculateLoss(neuronFn, params);
      
      // f(x-h, y-h)
      params[param2] = orig2 - epsilon;
      const fmm = calculateLoss(neuronFn, params);
      
      const secondDerivative = (fpp - fpm - fmp + fmm) / (4 * epsilon * epsilon);
      hessian[param1 + param2] = secondDerivative;
      
      // Reset parameters
      params[param1] = orig1;
      params[param2] = orig2;
    }
  }
  
  return hessian;
}

function newtonStep(neuronFn, params, learningRate = 1.0) {
  const gradients = { w1: 0, w2: 0, b: 0 };
  const epsilon = 1e-7;

  // Calculate gradients
  for (let param in params) {
    const originalValue = params[param];
    params[param] = originalValue + epsilon;
    const lossPlus = calculateLoss(neuronFn, params);
    params[param] = originalValue - epsilon;
    const lossMinus = calculateLoss(neuronFn, params);
    gradients[param] = (lossPlus - lossMinus) / (2 * epsilon);
    params[param] = originalValue;
  }

  // Calculate Hessian
  const hessian = calculateHessian(neuronFn, params);
  
  // Compute inverse Hessian times gradient
  const det = hessian.w1w1 * (hessian.w2w2 * hessian.bb - hessian.w2b * hessian.bw2) -
              hessian.w1w2 * (hessian.w2w1 * hessian.bb - hessian.w2b * hessian.bw1) +
              hessian.w1b * (hessian.w2w1 * hessian.bw2 - hessian.w2w2 * hessian.bw1);
  
  // Add regularization to prevent singular matrix
  const lambda = 1e-6;
  for (let i in hessian) {
    if (i[0] === i[1]) hessian[i] += lambda;
  }
  
  // Update parameters using Newton's method
  const step = {
    w1: -(gradients.w1 / hessian.w1w1),
    w2: -(gradients.w2 / hessian.w2w2),
    b: -(gradients.b / hessian.bb)
  };

  for (let param in params) {
    params[param] += learningRate * step[param];
  }

  return calculateLoss(neuronFn, params);
}

function calculateAccuracy(neuronFn, params) {
  return data.reduce((correct, point) => {
    const output = neuronFn(point.x, point.y, params.w1, params.w2, params.b);
    return correct + ((output > 0.5 ? 1 : 0) === point.label ? 1 : 0);
  }, 0) / data.length;
}

function calculateMagnitude(params) {
  return Math.sqrt(params.w1 * params.w1 + params.w2 * params.w2);
}

function momentumStep(neuronFn, params, state, learningRate = 0.01, momentum = 0.9) {
  if (!state.velocity) {
    state.velocity = { w1: 0, w2: 0, b: 0 };
  }

  const gradients = calculateGradients(neuronFn, params);
  
  for (let param in params) {
    state.velocity[param] = momentum * state.velocity[param] - learningRate * gradients[param];
    params[param] += state.velocity[param];
  }

  return calculateLoss(neuronFn, params);
}

function adamStep(neuronFn, params, state, learningRate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
  if (!state.step) {
    state.step = 0;
    state.m = { w1: 0, w2: 0, b: 0 };
    state.v = { w1: 0, w2: 0, b: 0 };
  }

  state.step += 1;
  const gradients = calculateGradients(neuronFn, params);

  for (let param in params) {
    // Update biased first moment estimate
    state.m[param] = beta1 * state.m[param] + (1 - beta1) * gradients[param];
    // Update biased second raw moment estimate
    state.v[param] = beta2 * state.v[param] + (1 - beta2) * gradients[param] * gradients[param];
    
    // Compute bias-corrected moment estimates
    const mHat = state.m[param] / (1 - Math.pow(beta1, state.step));
    const vHat = state.v[param] / (1 - Math.pow(beta2, state.step));
    
    params[param] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
  }

  return calculateLoss(neuronFn, params);
}

function rmspropStep(neuronFn, params, state, learningRate = 0.01, decay = 0.9, epsilon = 1e-8) {
  if (!state.cache) {
    state.cache = { w1: 0, w2: 0, b: 0 };
  }

  const gradients = calculateGradients(neuronFn, params);

  for (let param in params) {
    state.cache[param] = decay * state.cache[param] + (1 - decay) * gradients[param] * gradients[param];
    params[param] -= learningRate * gradients[param] / (Math.sqrt(state.cache[param]) + epsilon);
  }

  return calculateLoss(neuronFn, params);
}

function nagStep(neuronFn, params, state, learningRate = 0.01, momentum = 0.9) {
  if (!state.velocity) {
    state.velocity = { w1: 0, w2: 0, b: 0 };
  }

  const lookAheadParams = { ...params };
  for (let param in params) {
    lookAheadParams[param] += momentum * state.velocity[param];
  }

  const gradients = calculateGradients(neuronFn, lookAheadParams);

  for (let param in params) {
    state.velocity[param] = momentum * state.velocity[param] - learningRate * gradients[param];
    params[param] += state.velocity[param];
  }

  return calculateLoss(neuronFn, params);
}

function optimize(neuronFn, params, plot, weightPlot, weightPath, lossDisplay, historyPlots, iterations = 200, learningRate = 0.1, method = 'gradient') {
  weightPath.length = 0;
  const history = { loss: [], accuracy: [], magnitude: [] };
  const state = {}; // State for optimizers that need it

  let i = 0;
  const interval = setInterval(() => {
    let loss;
    switch (method) {
      case 'newton':
        loss = newtonStep(neuronFn, params, learningRate);
        break;
      case 'momentum':
        loss = momentumStep(neuronFn, params, state, learningRate);
        break;
      case 'adam':
        loss = adamStep(neuronFn, params, state, learningRate);
        break;
      case 'rmsprop':
        loss = rmspropStep(neuronFn, params, state, learningRate);
        break;
      case 'nag':
        loss = nagStep(neuronFn, params, state, learningRate);
        break;
      default: // gradient descent
        loss = gradientDescent(neuronFn, params, learningRate);
    }
    const accuracy = calculateAccuracy(neuronFn, params);
    const magnitude = calculateMagnitude(params);

    weightPath.push({ w1: params.w1, w2: params.w2 });
    history.loss.push(loss);
    history.accuracy.push(accuracy);
    history.magnitude.push(magnitude);

    updatePlot(plot, neuronFn, params);
    updateWeightPlot(weightPlot, params, weightPath);
    updateHistoryPlots(historyPlots, history);

    lossDisplay.textContent = `Loss: ${loss.toFixed(4)} | Accuracy: ${(accuracy * 100).toFixed(1)}% | w1: ${params.w1.toFixed(3)} | w2: ${params.w2.toFixed(3)} | b: ${params.b.toFixed(3)}`;

    i++;
    if (i >= iterations) clearInterval(interval);
  }, 50);
}

function optimizeAll(neurons, combinedPlot, combinedWeightPlot, lossDisplays, learningRate = 0.1, method = 'gradient') {
  const iterations = 100;
  let i = 0;
  
  // Clear weight paths
  Object.values(neurons).forEach(neuron => {
    neuron.weightPath = [];
  });

  const interval = setInterval(() => {
    // Optimize each neuron
    Object.entries(neurons).forEach(([type, neuron]) => {
      const loss = method === 'gradient'
        ? gradientDescent(neuron.fn, neuron.params, learningRate)
        : newtonStep(neuron.fn, neuron.params, learningRate);
      const accuracy = calculateAccuracy(neuron.fn, neuron.params);
      
      // Track weight path
      neuron.weightPath.push({
        w1: neuron.params.w1,
        w2: neuron.params.w2
      });

      // Update displays
      const lossDisplay = lossDisplays[type];
      const weightDisplay = document.getElementById(`${type}-combined-weights`);
      if (lossDisplay) {
        lossDisplay.textContent = `Loss: ${loss.toFixed(4)} | Acc: ${(accuracy * 100).toFixed(1)}%`;
      }
      if (weightDisplay) {
        weightDisplay.textContent = `w1: ${neuron.params.w1.toFixed(3)} | w2: ${neuron.params.w2.toFixed(3)} | b: ${neuron.params.b.toFixed(3)}`;
      }
    });

    // Update plots
    updateCombinedPlot(combinedPlot, neurons);
    updateCombinedWeightPlot(combinedWeightPlot, neurons);

    i++;
    if (i >= iterations) clearInterval(interval);
  }, 50);
}

// Helper function to calculate gradients
function calculateGradients(neuronFn, params) {
  const epsilon = 1e-7;
  const gradients = { w1: 0, w2: 0, b: 0 };

  for (let param in params) {
    const originalValue = params[param];
    params[param] = originalValue + epsilon;
    const lossPlus = calculateLoss(neuronFn, params);
    params[param] = originalValue - epsilon;
    const lossMinus = calculateLoss(neuronFn, params);
    gradients[param] = (lossPlus - lossMinus) / (2 * epsilon);
    params[param] = originalValue;
  }

  return gradients;
}

// Make sure optimizeAll is exported
export { 
  calculateLoss, 
  gradientDescent, 
  calculateAccuracy, 
  calculateMagnitude, 
  optimize,
  optimizeAll 
};