---
sidebar_position: 2
---

# YatGRU

Gated Recurrent Unit with ⵟ-product gates.

## Import

```python
from nmn.nnx.rnn import YatGRU
```

## Usage

```python
from nmn.nnx.rnn import YatGRU

gru = YatGRU(
    input_size=128,
    hidden_size=256,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 100, 128))
outputs, h_n = gru(x)
```

## See Also

- [YatLSTM](/docs/rnn/lstm) - LSTM variant
