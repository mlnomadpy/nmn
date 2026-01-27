---
sidebar_position: 3
---

# SimpleRNN

Basic recurrent neural network with ⵟ-product.

## Import

```python
from nmn.nnx.rnn import SimpleYatRNN
```

## Usage

```python
from nmn.nnx.rnn import SimpleYatRNN

rnn = SimpleYatRNN(
    input_size=128,
    hidden_size=256,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 100, 128))
outputs, h_n = rnn(x)
```

## See Also

- [YatLSTM](/docs/rnn/lstm) - LSTM (more expressive)
- [YatGRU](/docs/rnn/gru) - GRU variant
