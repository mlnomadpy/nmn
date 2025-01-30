import { dotNeuron, yatNeuron, posiYatNeuron, createParams } from './neurons.js';
import { 
  createPlot, 
  createWeightPlot, 
  createHistoryPlot, 
  createCombinedPlot,  // Add this
  updatePlot, 
  updateWeightPlot,
  updateCombinedPlot   // Add this
} from './plots.js';
import * as handlers from './handlers.js';
import { ThemeController } from './theme.js';

// Initialize theme controller
const themeController = new ThemeController();
window.themeController = themeController;

// Initialize parameters
export const dotParams = createParams();
export const yatParams = createParams();
export const posiParams = createParams();

// Initialize weight paths
export const dotWeightPath = [];
export const yatWeightPath = [];
export const posiWeightPath = [];

// Create plots
export const dotPlot = createPlot("#dot-plot");
export const yatPlot = createPlot("#yat-plot");
export const posiPlot = createPlot("#posi-plot");

// Create weight plots
export const dotWeightPlot = createWeightPlot("#dot-weight-plot");
export const yatWeightPlot = createWeightPlot("#yat-weight-plot");
export const posiWeightPlot = createWeightPlot("#posi-weight-plot");
// Create combined plot
export const combinedPlot = createCombinedPlot("#combined-plot");

// Create and export history plots
export const historyPlots = {
  dot: {
    loss: createHistoryPlot("#dot-loss-history"),
    accuracy: createHistoryPlot("#dot-accuracy-history"),
    magnitude: createHistoryPlot("#dot-magnitude-history"),
  },
  yat: {
    loss: createHistoryPlot("#yat-loss-history"),
    accuracy: createHistoryPlot("#yat-accuracy-history"),
    magnitude: createHistoryPlot("#yat-magnitude-history"),
  },
  posi: {
    loss: createHistoryPlot("#posi-loss-history"),
    accuracy: createHistoryPlot("#posi-accuracy-history"),
    magnitude: createHistoryPlot("#posi-magnitude-history"),
  }
};

// Initial updates
updatePlot(dotPlot, dotNeuron, dotParams);
updatePlot(yatPlot, yatNeuron, yatParams);
updatePlot(posiPlot, posiYatNeuron, posiParams);

updateWeightPlot(dotWeightPlot, dotParams, dotWeightPath);
updateWeightPlot(yatWeightPlot, yatParams, yatWeightPath);
updateWeightPlot(posiWeightPlot, posiParams, posiWeightPath);

// Initial combined plot update
updateCombinedPlot(combinedPlot, {
  dot: { fn: dotNeuron, params: dotParams },
  yat: { fn: yatNeuron, params: yatParams },
  posi: { fn: posiYatNeuron, params: posiParams }
});

// Expose handlers to window for HTML event handlers
Object.assign(window, {
  optimizeDot: handlers.optimizeDot,
  optimizeYat: handlers.optimizeYat,
  optimizePosi: handlers.optimizePosi,
  resetDot: handlers.resetDot,
  resetYat: handlers.resetYat,
  resetPosi: handlers.resetPosi,
  optimizeAllNeurons: handlers.optimizeAllNeurons,
  resetAllNeurons: handlers.resetAllNeurons,
  runComparison: handlers.runComparison
});
