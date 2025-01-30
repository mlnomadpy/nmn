export function dotNeuron(x, y, w1, w2, b) {
  return w1 * x + w2 * y + b;
}

export function yatNeuron(x, y, w1, w2, b) {
  const dot = w1 * x + w2 * y;
  const dist = Math.pow(w1 - x, 2) + Math.pow(w2 - y, 2);
  const epsilon = 1e-4;
  return (dot * dot) / (dist + epsilon) + b;
}

export function posiYatNeuron(x, y, w1, w2, b) {
  const dot = w1 * x + w2 * y;
  const dist = Math.pow(w1 - x, 2) + Math.pow(w2 - y, 2);
  const epsilon = 1e-4;
  return Math.sqrt(dist) / (dot * dot + epsilon) + b;
}

// Parameter initialization
export function createParams() {
  return {
    w1: Math.random() - 0.5,
    w2: Math.random() - 0.5,
    b: Math.random() - 0.5,
  };
}
