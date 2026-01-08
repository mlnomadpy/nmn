---
sidebar_position: 1
---

# YatLSTM

Long Short-Term Memory (LSTM) with ⵟ-product gates.

## Import

```python
from nmn.nnx.rnn import YatLSTM
```

## Constructor

```python
YatLSTM(
    input_size: int,
    hidden_size: int,
    *,
    use_bias: bool = True,
    constant_alpha: Optional[Union[bool, float]] = True,
    epsilon: float = 1e-5,
    rngs: nnx.Rngs,
)
```

## Usage

```python
from nmn.nnx.rnn import YatLSTM

lstm = YatLSTM(
    input_size=128,
    hidden_size=256,
    rngs=nnx.Rngs(0)
)

# Process sequence
x = jnp.ones((16, 100, 128))  # (batch, seq_len, features)
outputs, (h_n, c_n) = lstm(x)
```

## With Initial State

```python
h_0 = jnp.zeros((16, 256))
c_0 = jnp.zeros((16, 256))

outputs, (h_n, c_n) = lstm(x, initial_state=(h_0, c_0))
```

## See Also

- [YatGRU](/docs/rnn/gru) - GRU variant
- [SimpleRNN](/docs/rnn/simple) - Basic RNN
