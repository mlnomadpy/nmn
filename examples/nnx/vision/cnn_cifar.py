import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# Import libraries for comparison functions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from flax import nnx
from flax.nnx.module import Module
from flax.nnx import rnglib
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Initializer,
  PrecisionLike,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
  PromoteDtypeFn,
)

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import opt_einsum

from flax.core.frozen_dict import FrozenDict
from flax import nnx
from flax.nnx import rnglib, variablelib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
  PromoteDtypeFn,
  EinsumT,
)


import tensorflow_datasets as tfds
import tensorflow as tf

from flax import nnx  # The Flax NNX API.
from functools import partial
import optax

Array = jax.Array

# Default initializers
default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()

# Helper functions
def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """ "Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, tp.Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    ' int or pair of ints.'
  )

def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

class YatConv(Module):
  """Yat Convolution Module wrapping ``lax.conv_general_dilated``.

  Example usage::

    >>> from flax.nnx import conv # Assuming this file is flax/nnx/conv.py
    >>> import jax, jax.numpy as jnp
    >>> from flax.nnx import rnglib, state

    >>> rngs = rnglib.Rngs(0)
    >>> x = jnp.ones((1, 8, 3))

    >>> # valid padding
    >>> layer = conv.Conv(in_features=3, out_features=4, kernel_size=(3,),
    ...                  padding='VALID', rngs=rngs)
    >>> s = state(layer)
    >>> print(s['kernel'].value.shape)
    (3, 3, 4)
    >>> print(s['bias'].value.shape)
    (4,)
    >>> out = layer(x)
    >>> print(out.shape)
    (1, 6, 4)

  Args:
    in_features: int or tuple with number of input features.
    out_features: int or tuple with number of output features.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer, which will be interpreted
      as a tuple of the single integer. For all other cases, it must be a
      sequence of integers.
    strides: an integer or a sequence of ``n`` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), the string `'REFLECT'`
      (reflection across the padding boundary), or a sequence of ``n``
      ``(low, high)`` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides. ``'CAUSAL'`` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of ``inputs``
      (default: 1). Convolution with input dilation ``d`` is equivalent to
      transposed convolution with stride ``d``.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
      and a ``dtype`` keyword argument, and return a tuple of arrays with the
      promoted dtype.
    epsilon: A small float added to the denominator to prevent division by zero.
    rngs: rng key.
  """

  __data__ = ('kernel', 'bias', 'mask', 'alpha')

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: int | tp.Sequence[int],
    strides: tp.Union[None, int, tp.Sequence[int]] = 1,
    *,
    padding: PaddingLike = 'SAME',
    input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    mask: tp.Optional[Array] = None,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    epsilon: float = 1/137,
    use_alpha: bool = True,
    alpha_init: Initializer = default_alpha_init,

    rngs: rnglib.Rngs,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)
    else:
      kernel_size = tuple(kernel_size)

    self.kernel_shape = kernel_size + (
      in_features // feature_group_count,
      out_features,
    )
    kernel_key = rngs.params()
    self.kernel = nnx.Param(kernel_init(kernel_key, self.kernel_shape, param_dtype))

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_shape = (out_features,)
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
    else:
      self.bias = None


    self.alpha: nnx.Param[jax.Array] | None
    if use_alpha:
      alpha_key = rngs.params()
      self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
    else:
      self.alpha = None


    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.feature_group_count = feature_group_count
    self.use_bias = use_bias
    self.mask = mask
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.conv_general_dilated = conv_general_dilated
    self.promote_dtype = promote_dtype
    self.epsilon = epsilon
    self.use_alpha = use_alpha
    self.alpha_init = alpha_init

  def __call__(self, inputs: Array) -> Array:
    assert isinstance(self.kernel_size, tuple)

    def maybe_broadcast(
      x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
    ) -> tuple[int, ...]:
      if x is None:
        x = 1
      if isinstance(x, int):
        return (x,) * len(self.kernel_size)
      return tuple(x)

    num_batch_dimensions = inputs.ndim - (len(self.kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
        num_batch_dimensions:
      ]
      inputs_flat = jnp.reshape(inputs, flat_input_shape)
    else:
      inputs_flat = inputs
      input_batch_shape = ()

    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(self.kernel_size))
    if padding_lax in ('CIRCULAR', 'REFLECT'):
      assert isinstance(padding_lax, str)
      kernel_size_dilated = [
        (k - 1) * d + 1 for k, d in zip(self.kernel_size, kernel_dilation)
      ]
      zero_pad: tp.List[tuple[int, int]] = [(0, 0)]
      pads = (
        zero_pad
        + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
        + [(0, 0)]
      )
      padding_mode = {'CIRCULAR': 'wrap', 'REFLECT': 'reflect'}[padding_lax]
      inputs_flat = jnp.pad(inputs_flat, pads, mode=padding_mode)
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(self.kernel_size) != 1:
        raise ValueError(
          'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (self.kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs_flat = jnp.pad(inputs_flat, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs_flat.shape)
    assert self.in_features % self.feature_group_count == 0

    kernel_val = self.kernel.value

    current_mask = self.mask
    if current_mask is not None:
      if current_mask.shape != self.kernel_shape:
        raise ValueError(
          'Mask needs to have the same shape as weights. '
          f'Shapes are: {current_mask.shape}, {self.kernel_shape}'
        )
      kernel_val *= current_mask

    bias_val = self.bias.value if self.bias is not None else None

    inputs_promoted, kernel_promoted, bias_promoted = self.promote_dtype(
      (inputs_flat, kernel_val, bias_val), dtype=self.dtype
    )
    inputs_flat = inputs_promoted
    kernel_val = kernel_promoted
    bias_val = bias_promoted

    dot_prod_map = self.conv_general_dilated(
      inputs_flat,
      kernel_val,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    inputs_flat_squared = inputs_flat**2
    kernel_in_channels_for_sum_sq = self.kernel_shape[-2]
    kernel_for_patch_sq_sum_shape = self.kernel_size + (kernel_in_channels_for_sum_sq, 1)
    kernel_for_patch_sq_sum = jnp.ones(kernel_for_patch_sq_sum_shape, dtype=kernel_val.dtype)

    patch_sq_sum_map_raw = self.conv_general_dilated(
      inputs_flat_squared,
      kernel_for_patch_sq_sum,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    if self.feature_group_count > 1:
      num_out_channels_per_group = self.out_features // self.feature_group_count
      if num_out_channels_per_group == 0 :
          raise ValueError(
              "out_features must be a multiple of feature_group_count and greater or equal."
          )
      patch_sq_sum_map = jnp.repeat(patch_sq_sum_map_raw, num_out_channels_per_group, axis=-1)
    else:
      patch_sq_sum_map = patch_sq_sum_map_raw

    reduce_axes_for_kernel_sq = tuple(range(kernel_val.ndim - 1))
    kernel_sq_sum_per_filter = jnp.sum(kernel_val**2, axis=reduce_axes_for_kernel_sq)

    distance_sq_map = patch_sq_sum_map + kernel_sq_sum_per_filter - 2 * dot_prod_map
    y = dot_prod_map**2 / (distance_sq_map + self.epsilon)

    if self.use_bias and bias_val is not None:
      bias_reshape_dims = (1,) * (y.ndim - 1) + (-1,)
      y += jnp.reshape(bias_val, bias_reshape_dims)

    if self.use_alpha and self.alpha is not None:
      alpha_val = self.alpha.value
      # Ensure alpha_val is promoted to the same dtype as y if needed, though usually it's float32.
      # This might require using self.promote_dtype or ensuring consistent dtypes.
      # For simplicity, assuming alpha_val.dtype is compatible or jax handles promotion.
      scale = (jnp.sqrt(jnp.array(self.out_features, dtype=y.dtype)) /
               jnp.log(1 + jnp.array(self.out_features, dtype=y.dtype))) ** alpha_val
      y = y * scale

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y

Array = jax.Array
Axis = int
Size = int


default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()

class YatNMN(Module):
  """A linear transformation applied over the last dimension of the input.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> layer = nnx.Linear(in_features=3, out_features=4, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': VariableState(
        type=Param,
        value=(4,)
      ),
      'kernel': VariableState(
        type=Param,
        value=(3, 4)
      )
    })

  Args:
    in_features: the number of input features.
    out_features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    dot_general: dot product function.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
      and a ``dtype`` keyword argument, and return a tuple of arrays with the
      promoted dtype.
    rngs: rng key.
  """

  __data__ = ('kernel', 'bias')

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    use_alpha: bool = True,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    alpha_init: Initializer = default_alpha_init,
    dot_general: DotGeneralT = lax.dot_general,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    rngs: rnglib.Rngs,
    epsilon: float = 1/137,
  ):

    kernel_key = rngs.params()
    self.kernel = nnx.Param(
      kernel_init(kernel_key, (in_features, out_features), param_dtype)
    )
    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
    else:
      self.bias = None

    self.alpha: nnx.Param[jax.Array] | None
    if use_alpha:
      alpha_key = rngs.params()
      self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
    else:
      self.alpha = None

    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias
    self.use_alpha = use_alpha
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.dot_general = dot_general
    self.promote_dtype = promote_dtype
    self.epsilon = epsilon

  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.kernel.value
    bias = self.bias.value if self.bias is not None else None
    alpha = self.alpha.value if self.alpha is not None else None

    inputs, kernel, bias, alpha = self.promote_dtype(
      (inputs, kernel, bias, alpha), dtype=self.dtype
    )
    y = self.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )

    assert self.use_bias == (bias is not None)
    assert self.use_alpha == (alpha is not None)

    inputs_squared_sum = jnp.sum(inputs**2, axis=-1, keepdims=True)
    kernel_squared_sum = jnp.sum(kernel**2, axis=0, keepdims=True)  # Change axis to 0 and keepdims to True
    distances = inputs_squared_sum + kernel_squared_sum - 2 * y

    # # Element-wise operation
    y = y ** 2 /  (distances + self.epsilon)

    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    if alpha is not None:
      scale = (jnp.sqrt(self.out_features) / jnp.log(1 + self.out_features)) ** alpha
      y = y * scale


    return y

def loss_fn(model, batch):
  logits = model(batch['image'], training=True)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

tf.random.set_seed(0)

# ===== DATASET CONFIGURATIONS =====
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 5, 'eval_every': 200, 'batch_size': 128
    },
    'cifar100': {
        'num_classes': 100, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 5, 'eval_every': 200, 'batch_size': 128
    },
    'stl10': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 5, 'eval_every': 200, 'batch_size': 128
    },
    'eurosat/rgb': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
        'image_key': 'image', 'label_key': 'label', # EuroSAT label key is 'label' in TFDS
        'num_epochs': 5, 'eval_every': 100, 'batch_size': 128
    },
    'eurosat/all': {
        'num_classes': 10, 'input_channels': 13,
        'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 5, 'eval_every': 100, 'batch_size': 16 # Smaller batch for more channels
    },
    # Example for a dataset that might need specific image resizing if models were not robust
    # 'some_other_dataset': {
    #     'num_classes': X, 'input_channels': Y, 
    #     'train_split': 'train', 'test_split': 'validation',
    #     'image_key': 'image_data', 'label_key': 'class_id',
    #     'target_image_size': [H, W] # Optional: for explicit resizing
    # },
}

# Original global dataset setup (will be superseded by _train_model_loop for actual training runs)
# These might still be used by some top-level calls if not careful, or for initial exploration.
_DEFAULT_DATASET_FOR_GLOBALS = 'cifar10'

# Get default training parameters from the default dataset's config or set fallbacks
_default_config_for_globals = DATASET_CONFIGS.get(_DEFAULT_DATASET_FOR_GLOBALS, {})
_global_num_epochs = _default_config_for_globals.get('num_epochs', 10) # Default to 10 epochs
_global_eval_every = _default_config_for_globals.get('eval_every', 200)
_global_batch_size = _default_config_for_globals.get('batch_size', 64)


_global_ds_builder = tfds.builder(_DEFAULT_DATASET_FOR_GLOBALS)
_global_ds_info = _global_ds_builder.info

train_ds_global_tf: tf.data.Dataset = tfds.load(_DEFAULT_DATASET_FOR_GLOBALS, split='train')
test_ds_global_tf: tf.data.Dataset = tfds.load(_DEFAULT_DATASET_FOR_GLOBALS, split='test')

def _global_preprocess(sample):
    return {
        'image': tf.cast(sample[DATASET_CONFIGS[_DEFAULT_DATASET_FOR_GLOBALS]['image_key']], tf.float32) / 255,
        'label': sample[DATASET_CONFIGS[_DEFAULT_DATASET_FOR_GLOBALS]['label_key']],
    }

train_ds_global_tf = train_ds_global_tf.map(_global_preprocess)
test_ds_global_tf = test_ds_global_tf.map(_global_preprocess)

# Original global TF dataset iterators (used for some analysis functions if they don't reload)
# It's better if analysis functions requiring data get it passed or reload it with correct dataset_name
# Removing .take() from global train_ds to align with epoch-based approach; consumers must manage iterations.
train_ds = train_ds_global_tf.repeat().shuffle(1024).batch(_global_batch_size, drop_remainder=True).prefetch(1)
test_ds = test_ds_global_tf.batch(_global_batch_size, drop_remainder=True).prefetch(1)

# ===== MODEL COMPARISON FUNCTIONS =====

def compare_training_curves(yat_history, linear_history):
    """
    Compare training curves between YAT and Linear models.
    Plots side-by-side comparison of loss and accuracy over training steps.
    """
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison: YAT vs Linear Models', fontsize=16, fontweight='bold')
    
    steps = range(len(yat_history['train_loss']))
    
    # Training Loss
    ax1.plot(steps, yat_history['train_loss'], 'b-', label='YAT Model', linewidth=2)
    ax1.plot(steps, linear_history['train_loss'], 'r--', label='Linear Model', linewidth=2)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Evaluation Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test Loss
    ax2.plot(steps, yat_history['test_loss'], 'b-', label='YAT Model', linewidth=2)
    ax2.plot(steps, linear_history['test_loss'], 'r--', label='Linear Model', linewidth=2)
    ax2.set_title('Test Loss', fontweight='bold')
    ax2.set_xlabel('Evaluation Steps')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax3.plot(steps, yat_history['train_accuracy'], 'b-', label='YAT Model', linewidth=2)
    ax3.plot(steps, linear_history['train_accuracy'], 'r--', label='Linear Model', linewidth=2)
    ax3.set_title('Training Accuracy', fontweight='bold')
    ax3.set_xlabel('Evaluation Steps')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Test Accuracy
    ax4.plot(steps, yat_history['test_accuracy'], 'b-', label='YAT Model', linewidth=2)
    ax4.plot(steps, linear_history['test_accuracy'], 'r--', label='Linear Model', linewidth=2)
    ax4.set_title('Test Accuracy', fontweight='bold')
    ax4.set_xlabel('Evaluation Steps')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📈 Training curves comparison plotted successfully!")

def print_final_metrics_comparison(yat_history, linear_history):
    """
    Print a detailed comparison table of final metrics.
    """
    print("\n📊 FINAL METRICS COMPARISON")
    print("=" * 60)
    
    yat_final = {
        'train_loss': yat_history['train_loss'][-1],
        'train_accuracy': yat_history['train_accuracy'][-1],
        'test_loss': yat_history['test_loss'][-1],
        'test_accuracy': yat_history['test_accuracy'][-1]
    }
    
    linear_final = {
        'train_loss': linear_history['train_loss'][-1],
        'train_accuracy': linear_history['train_accuracy'][-1],
        'test_loss': linear_history['test_loss'][-1],
        'test_accuracy': linear_history['test_accuracy'][-1]
    }
    
    print(f"{'Metric':<20} {'YAT Model':<15} {'Linear Model':<15} {'Difference':<15}")
    print("-" * 65)
    
    for metric in ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']:
        yat_val = yat_final[metric]
        linear_val = linear_final[metric]
        diff = yat_val - linear_val
        diff_str = f"{diff:+.4f}"
        if 'accuracy' in metric:
            if diff > 0:
                diff_str += " (YAT better)"
            elif diff < 0:
                diff_str += " (Linear better)"
        else:  # loss
            if diff < 0:
                diff_str += " (YAT better)"
            elif diff > 0:
                diff_str += " (Linear better)"
        
        print(f"{metric:<20} {yat_val:<15.4f} {linear_val:<15.4f} {diff_str:<15}")
    
    # Summary
    print("\n🏆 SUMMARY:")
    if yat_final['test_accuracy'] > linear_final['test_accuracy']:
        winner = "YAT Model"
        margin = yat_final['test_accuracy'] - linear_final['test_accuracy']
    else:
        winner = "Linear Model"
        margin = linear_final['test_accuracy'] - yat_final['test_accuracy']
    
    print(f"   Better Test Accuracy: {winner} (by {margin:.4f})")
    print(f"   YAT Test Accuracy: {yat_final['test_accuracy']:.4f}")
    print(f"   Linear Test Accuracy: {linear_final['test_accuracy']:.4f}")

def analyze_convergence(yat_history, linear_history):
    """
    Analyze convergence speed and stability of both models.
    """
    print("\n🔍 CONVERGENCE ANALYSIS")
    print("=" * 50)
    
    def calculate_convergence_metrics(history):
        test_acc = history['test_accuracy']
        train_acc = history['train_accuracy']
        test_loss = history['test_loss']
        
        # Find step where model reaches 50% of final accuracy
        final_acc = test_acc[-1]
        target_acc = 0.5 * final_acc
        convergence_step = 0
        for i, acc in enumerate(test_acc):
            if acc >= target_acc:
                convergence_step = i
                break
        
        # Calculate stability (variance in last 25% of training)
        last_quarter = len(test_acc) // 4
        stability = np.std(test_acc[-last_quarter:])
        
        # Calculate final overfitting (train_acc - test_acc)
        overfitting = train_acc[-1] - test_acc[-1]
        
        return {
            'convergence_step': convergence_step,
            'stability': stability,
            'overfitting': overfitting,
            'final_loss': test_loss[-1]
        }
    
    yat_conv = calculate_convergence_metrics(yat_history)
    linear_conv = calculate_convergence_metrics(linear_history)
    
    print(f"{'Metric':<25} {'YAT Model':<15} {'Linear Model':<15}")
    print("-" * 55)
    print(f"{'Convergence Speed':<25} {yat_conv['convergence_step']:<15} {linear_conv['convergence_step']:<15}")
    print(f"{'Stability (std)':<25} {yat_conv['stability']:<15.4f} {linear_conv['stability']:<15.4f}")
    print(f"{'Overfitting Gap':<25} {yat_conv['overfitting']:<15.4f} {linear_conv['overfitting']:<15.4f}")
    print(f"{'Final Test Loss':<25} {yat_conv['final_loss']:<15.4f} {linear_conv['final_loss']:<15.4f}")
    
    # Analysis
    print("\n📋 ANALYSIS:")
    if yat_conv['convergence_step'] < linear_conv['convergence_step']:
        print(f"   🚀 YAT model converges faster (step {yat_conv['convergence_step']} vs {linear_conv['convergence_step']})")
    else:
        print(f"   🚀 Linear model converges faster (step {linear_conv['convergence_step']} vs {yat_conv['convergence_step']})")
    
    if yat_conv['stability'] < linear_conv['stability']:
        print(f"   📈 YAT model is more stable (std: {yat_conv['stability']:.4f} vs {linear_conv['stability']:.4f})")
    else:
        print(f"   📈 Linear model is more stable (std: {linear_conv['stability']:.4f} vs {yat_conv['stability']:.4f})")
    
    if abs(yat_conv['overfitting']) < abs(linear_conv['overfitting']):
        print(f"   🎯 YAT model has less overfitting (gap: {yat_conv['overfitting']:.4f} vs {linear_conv['overfitting']:.4f})")
    else:
        print(f"   🎯 Linear model has less overfitting (gap: {linear_conv['overfitting']:.4f} vs {yat_conv['overfitting']:.4f})")

def detailed_test_evaluation(yat_model, linear_model, test_ds_iter, class_names: list[str]):
    """
    Perform detailed evaluation on test set including per-class accuracy and model agreement.
    test_ds_iter: An iterable TFDS dataset (already batched and preprocessed).
    class_names: List of class names for the current dataset.
    """
    print("Running detailed test evaluation...")
    
    # CIFAR-10 class names # This will be replaced by the passed class_names
    # cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(class_names)
    
    yat_predictions = []
    linear_predictions = []
    true_labels = []
    
    # Collect predictions from both models
    for batch in test_ds_iter.as_numpy_iterator():
        batch_images, batch_labels = batch['image'], batch['label']
        
        # YAT model predictions
        yat_logits = yat_model(batch_images, training=False)
        yat_preds = jnp.argmax(yat_logits, axis=1)
        
        # Linear model predictions
        linear_logits = linear_model(batch_images, training=False)
        linear_preds = jnp.argmax(linear_logits, axis=1)
        
        yat_predictions.extend(yat_preds.tolist())
        linear_predictions.extend(linear_preds.tolist())
        true_labels.extend(batch_labels.tolist())
    
    yat_predictions = np.array(yat_predictions)
    linear_predictions = np.array(linear_predictions)
    true_labels = np.array(true_labels)
    
    # Calculate per-class accuracies
    print("\n🎯 PER-CLASS ACCURACY COMPARISON")
    print("=" * 70)
    print(f"{'Class':<12} {'YAT Acc':<10} {'Linear Acc':<12} {'Difference':<15} {'Sample Count':<12}")
    print("-" * 70)
    
    for class_idx in range(num_classes): # Use num_classes from passed class_names
        class_mask = true_labels == class_idx
        class_samples = np.sum(class_mask)
        
        if class_samples > 0:
            yat_class_acc = np.mean(yat_predictions[class_mask] == true_labels[class_mask])
            linear_class_acc = np.mean(linear_predictions[class_mask] == true_labels[class_mask])
            diff = yat_class_acc - linear_class_acc
            diff_str = f"{diff:+.4f}"
            
            print(f"{class_names[class_idx]:<12} {yat_class_acc:<10.4f} {linear_class_acc:<12.4f} {diff_str:<15} {class_samples:<12}")
        elif num_classes <= 20: # Only print for manageable number of classes if no samples
             print(f"{class_names[class_idx]:<12} {'N/A':<10} {'N/A':<12} {'N/A':<15} {class_samples:<12}")

    # Model agreement analysis
    agreement = np.mean(yat_predictions == linear_predictions)
    both_correct = np.mean((yat_predictions == true_labels) & (linear_predictions == true_labels))
    yat_correct_linear_wrong = np.mean((yat_predictions == true_labels) & (linear_predictions != true_labels))
    linear_correct_yat_wrong = np.mean((linear_predictions == true_labels) & (yat_predictions != true_labels))
    both_wrong = np.mean((yat_predictions != true_labels) & (linear_predictions != true_labels))
    
    print(f"\n🤝 MODEL AGREEMENT ANALYSIS")
    print("=" * 40)
    print(f"Overall Agreement: {agreement:.4f}")
    print(f"Both Correct: {both_correct:.4f}")
    print(f"YAT Correct, Linear Wrong: {yat_correct_linear_wrong:.4f}")
    print(f"Linear Correct, YAT Wrong: {linear_correct_yat_wrong:.4f}")
    print(f"Both Wrong: {both_wrong:.4f}")
    
    return {
        'yat_predictions': yat_predictions,
        'linear_predictions': linear_predictions,
        'true_labels': true_labels,
        'class_names': class_names,
        'agreement': agreement,
        'both_correct': both_correct
    }

def plot_confusion_matrices(predictions_data):
    """
    Plot confusion matrices for both models side by side.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    yat_preds = predictions_data['yat_predictions']
    linear_preds = predictions_data['linear_predictions']
    true_labels = predictions_data['true_labels']
    class_names = predictions_data['class_names']
    
    # Calculate confusion matrices
    yat_cm = confusion_matrix(true_labels, yat_preds)
    linear_cm = confusion_matrix(true_labels, linear_preds)
    
    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # YAT model confusion matrix
    sns.heatmap(yat_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('YAT Model - Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Linear model confusion matrix
    sns.heatmap(linear_cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Linear Model - Confusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Confusion matrices plotted successfully!")

def generate_summary_report(yat_history, linear_history, predictions_data):
    """
    Generate a comprehensive summary report of the comparison.
    """
    print("\n" + "="*80)
    print("                    COMPREHENSIVE SUMMARY REPORT")
    print("="*80)
    
    # Final metrics
    yat_final_acc = yat_history['test_accuracy'][-1]
    linear_final_acc = linear_history['test_accuracy'][-1]
    
    print(f"\n🏆 OVERALL WINNER:")
    if yat_final_acc > linear_final_acc:
        winner = "YAT Model"
        margin = yat_final_acc - linear_final_acc
        print(f"   🥇 {winner} wins by {margin:.4f} accuracy points!")
    elif linear_final_acc > yat_final_acc:
        winner = "Linear Model"
        margin = linear_final_acc - yat_final_acc
        print(f"   🥇 {winner} wins by {margin:.4f} accuracy points!")
    else:
        print(f"   🤝 It's a tie! Both models achieved {yat_final_acc:.4f} accuracy")
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   YAT Model Test Accuracy: {yat_final_acc:.4f}")
    print(f"   Linear Model Test Accuracy: {linear_final_acc:.4f}")
    print(f"   Model Agreement: {predictions_data['agreement']:.4f}")
    print(f"   Both Models Correct: {predictions_data['both_correct']:.4f}")
    
    # Best and worst performing classes
    class_names = predictions_data['class_names']
    true_labels = predictions_data['true_labels']
    yat_preds = predictions_data['yat_predictions']
    linear_preds = predictions_data['linear_predictions']
    
    yat_class_accs = []
    linear_class_accs = []
    
    for class_idx in range(len(class_names)):
        class_mask = true_labels == class_idx
        if np.sum(class_mask) > 0:
            yat_acc = np.mean(yat_preds[class_mask] == true_labels[class_mask])
            linear_acc = np.mean(linear_preds[class_mask] == true_labels[class_mask])
            yat_class_accs.append((class_names[class_idx], yat_acc))
            linear_class_accs.append((class_names[class_idx], linear_acc))
    
    # Sort by accuracy
    yat_class_accs.sort(key=lambda x: x[1], reverse=True)
    linear_class_accs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 BEST PERFORMING CLASSES (Top 3 if available):")
    for i in range(min(3, len(yat_class_accs))):
        print(f"   YAT Model: {yat_class_accs[i][0]} ({yat_class_accs[i][1]:.4f})")
    for i in range(min(3, len(linear_class_accs))):
        print(f"   Linear Model: {linear_class_accs[i][0]} ({linear_class_accs[i][1]:.4f})")
    
    print(f"\n🎯 WORST PERFORMING CLASSES (Bottom 3 if available):")
    for i in range(min(3, len(yat_class_accs))):
        print(f"   YAT Model: {yat_class_accs[-(i+1)][0]} ({yat_class_accs[-(i+1)][1]:.4f})")
    for i in range(min(3, len(linear_class_accs))):
        print(f"   Linear Model: {linear_class_accs[-(i+1)][0]} ({linear_class_accs[-(i+1)][1]:.4f})")
        
    print(f"\n📊 TRAINING CHARACTERISTICS:")
    if yat_history['train_accuracy'] and linear_history['train_accuracy'] and \
       yat_history['test_accuracy'] and linear_history['test_accuracy']:
        print(f"   YAT Final Train Accuracy: {yat_history['train_accuracy'][-1]:.4f}")
        print(f"   Linear Final Train Accuracy: {linear_history['train_accuracy'][-1]:.4f}")
        print(f"   YAT Overfitting Gap: {yat_history['train_accuracy'][-1] - yat_history['test_accuracy'][-1]:.4f}")
        print(f"   Linear Overfitting Gap: {linear_history['train_accuracy'][-1] - linear_history['test_accuracy'][-1]:.4f}")
    else:
        print("   Training/Test accuracy history missing for full overfitting gap analysis.")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if yat_final_acc > linear_final_acc:
        print(f"   ✅ YAT model architecture shows superior performance")
        print(f"   ✅ Consider using YAT layers for similar classification tasks")
    else:
        print(f"   ✅ Linear model architecture is sufficient for this task")
        print(f"   ✅ Standard convolution layers perform well on CIFAR-10")
    
    if predictions_data['agreement'] > 0.8:
        print(f"   🤝 High model agreement suggests stable learning")
    else:
        print(f"   🔍 Low model agreement suggests different learning patterns")
    
    print("="*80)

# ===== COMPLETE IMPLEMENTATION EXAMPLE =====

# Moved YatCNN class definition to module level
class YatCNN(nnx.Module):
    """YAT CNN model with custom layers."""

    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.conv1 = YatConv(input_channels, 32, kernel_size=(5, 5), rngs=rngs)
        self.conv2 = YatConv(32, 64, kernel_size=(5, 5), rngs=rngs)
        self.conv3 = YatConv(64, 128, kernel_size=(5, 5), rngs=rngs)
        self.conv4 = YatConv(128, 128, kernel_size=(5, 5), rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.3, rngs=rngs)

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.non_linear2 = YatNMN(128, num_classes, use_bias=False, use_alpha=False, rngs=rngs)

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        activations = {}
        x = self.conv1(x)
        activations['conv1'] = x
        if return_activations_for_layer == 'conv1': return x
        x = self.dropout1(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv2(x)
        activations['conv2'] = x
        if return_activations_for_layer == 'conv2': return x
        x = self.dropout2(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv3(x)
        activations['conv3'] = x
        if return_activations_for_layer == 'conv3': return x
        x = self.dropout3(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv4(x)
        activations['conv4'] = x
        if return_activations_for_layer == 'conv4': return x
        x = self.dropout4(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = jnp.mean(x, axis=(1, 2))
        activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': return x
        
        x = self.non_linear2(x)
        activations['final_layer'] = x
        if return_activations_for_layer == 'final_layer': return x
        
        if return_activations_for_layer is not None and return_activations_for_layer not in activations:
            print(f"Warning: Layer '{return_activations_for_layer}' not found in YatCNN. Available: {list(activations.keys())}")
            # Fallback to returning final output if requested layer is not found after checking all
        return x

# Moved LinearCNN class definition to module level
class LinearCNN(nnx.Module):
    """Standard CNN model with linear layers."""

    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(input_channels, 32, kernel_size=(5, 5), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(5, 5), rngs=rngs)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(5, 5), rngs=rngs)
        self.conv4 = nnx.Conv(128, 128, kernel_size=(5, 5), rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.3, rngs=rngs) # Note: different dropout rate
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.3, rngs=rngs)

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear2 = nnx.Linear(128, num_classes, rngs=rngs, use_bias=False)

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        activations = {}
        x = self.conv1(x)
        activations['conv1_raw'] = x # Raw output before ReLU
        if return_activations_for_layer == 'conv1_raw': return x
        x = nnx.relu(x)
        activations['conv1'] = x # Output after ReLU
        if return_activations_for_layer == 'conv1': return x
        x = self.dropout1(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv2(x)
        activations['conv2_raw'] = x
        if return_activations_for_layer == 'conv2_raw': return x
        x = nnx.relu(x)
        activations['conv2'] = x
        if return_activations_for_layer == 'conv2': return x
        x = self.dropout2(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv3(x)
        activations['conv3_raw'] = x
        if return_activations_for_layer == 'conv3_raw': return x
        x = nnx.relu(x)
        activations['conv3'] = x
        if return_activations_for_layer == 'conv3': return x
        x = self.dropout3(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv4(x)
        activations['conv4_raw'] = x
        if return_activations_for_layer == 'conv4_raw': return x
        x = nnx.relu(x)
        activations['conv4'] = x
        if return_activations_for_layer == 'conv4': return x
        x = self.dropout4(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = jnp.mean(x, axis=(1, 2))
        activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': return x
        
        x = self.linear2(x)
        activations['final_layer'] = x
        if return_activations_for_layer == 'final_layer': return x

        if return_activations_for_layer is not None and return_activations_for_layer not in activations:
            print(f"Warning: Layer '{return_activations_for_layer}' not found in LinearCNN. Available: {list(activations.keys())}")
        return x

# New helper function for the training loop
def _train_model_loop(
    model_class: tp.Type[nnx.Module],
    model_name: str,
    dataset_name: str, # New argument
    rng_seed: int,
    learning_rate: float,
    momentum: float,
    optimizer_constructor: tp.Callable,
):
    """Helper function to train a model and return it with its metrics history."""
    print(f"Initializing {model_name} model for dataset {dataset_name}...")

    config = DATASET_CONFIGS.get(dataset_name)
    ds_builder = tfds.builder(dataset_name)
    ds_info_for_model = ds_builder.info

    if not config:
        try:
            num_classes = ds_info_for_model.features['label'].num_classes
            image_shape = ds_info_for_model.features['image'].shape
            input_channels = image_shape[-1] if len(image_shape) >= 3 else 1
            train_split_name = 'train'
            test_split_name = 'test'
            image_key = 'image'
            label_key = 'label'
            # Fallback training parameters if not in config
            current_num_epochs = _global_num_epochs # Use global default epochs
            current_eval_every = _global_eval_every
            current_batch_size = _global_batch_size
            print(f"Warning: Dataset '{dataset_name}' not in pre-defined configs. Inferred: num_classes={num_classes}, input_channels={input_channels}. Using global defaults for training params.")
        except Exception as e:
            raise ValueError(f"Dataset '{dataset_name}' not in configs and could not infer info: {e}")
    else:
        num_classes = config['num_classes']
        input_channels = config['input_channels']
        train_split_name = config['train_split']
        test_split_name = config['test_split']
        image_key = config['image_key']
        label_key = config['label_key']
        current_num_epochs = config['num_epochs']
        current_eval_every = config['eval_every']
        current_batch_size = config['batch_size']

    model = model_class(num_classes=num_classes, input_channels=input_channels, rngs=nnx.Rngs(rng_seed))
    optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate, momentum))
    metrics_computer = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    def preprocess_data_fn(sample):
        image = tf.cast(sample[image_key], tf.float32) / 255.0
        return {'image': image, 'label': sample[label_key]}

    loaded_train_ds = tfds.load(dataset_name, split=train_split_name, as_supervised=False, shuffle_files=True)
    loaded_test_ds = tfds.load(dataset_name, split=test_split_name, as_supervised=False)

    dataset_size = loaded_train_ds.cardinality().numpy()
    if dataset_size == tf.data.UNKNOWN_CARDINALITY or dataset_size == tf.data.INFINITE_CARDINALITY:
        raise ValueError(
            f"Cannot determine dataset size for '{dataset_name}' split '{train_split_name}' for epoch-based training. "
            f"Please ensure the dataset split has a known finite cardinality or revert to step-based training with .take()."
        )
    steps_per_epoch = dataset_size // current_batch_size
    total_expected_steps = current_num_epochs * steps_per_epoch

    print(f"Training {model_name} on {dataset_name} for {current_num_epochs} epochs ({steps_per_epoch} steps/epoch, total {total_expected_steps} steps). Evaluating every {current_eval_every} steps.")

    # Test dataset iterator (created once)
    dataset_test_iter = loaded_test_ds.map(preprocess_data_fn, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(current_batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    metrics_history = {
        'train_loss': [], 'train_accuracy': [],
        'test_loss': [], 'test_accuracy': [],
    }

    global_step_counter = 0
    for epoch in range(current_num_epochs):
        print(f"  Epoch {epoch + 1}/{current_num_epochs}")
        # Create a new iterator for each epoch to ensure data is reshuffled if shuffle_files=True in tfds.load or .shuffle() is used effectively
        epoch_train_ds = loaded_train_ds.shuffle(buffer_size=1024) \
            .map(preprocess_data_fn, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(current_batch_size, drop_remainder=True) \
            .prefetch(tf.data.AUTOTUNE)

        for batch_in_epoch, batch_data in enumerate(epoch_train_ds.as_numpy_iterator()):
            train_step(model, optimizer, metrics_computer, batch_data)

            # Evaluation logic based on global step
            if global_step_counter > 0 and \
               (global_step_counter % current_eval_every == 0 or global_step_counter == total_expected_steps - 1) and \
               not (epoch == current_num_epochs - 1 and batch_in_epoch == steps_per_epoch -1 ): # Avoid double eval on last step if it aligns

                computed_train_metrics = metrics_computer.compute()
                for metric_name_key, value in computed_train_metrics.items():
                    metrics_history[f'train_{metric_name_key}'].append(value)
                metrics_computer.reset()

                for test_batch in dataset_test_iter.as_numpy_iterator():
                    eval_step(model, metrics_computer, test_batch)
                computed_test_metrics = metrics_computer.compute()
                for metric_name_key, value in computed_test_metrics.items():
                    metrics_history[f'test_{metric_name_key}'].append(value)
                metrics_computer.reset()
                print(f"    Step {global_step_counter}: {model_name} Train Acc = {metrics_history['train_accuracy'][-1]:.4f}, Test Acc = {metrics_history['test_accuracy'][-1]:.4f}")
            
            global_step_counter += 1
            if global_step_counter >= total_expected_steps:
                break # Exit if total_expected_steps reached (e.g. if steps_per_epoch was rounded)
        
        if global_step_counter >= total_expected_steps:
            break # Exit epoch loop as well

    # Final evaluation at the end of all epochs if not captured by the step-based eval above
    print(f"  Performing final evaluation for {model_name} after {current_num_epochs} epochs...")
    # Ensure train metrics for the last part of training are captured
    computed_train_metrics = metrics_computer.compute() # This captures metrics since last reset
    if computed_train_metrics and computed_train_metrics.get('loss') is not None: # Check if there are new metrics
        for metric_name_key, value in computed_train_metrics.items():
            metrics_history[f'train_{metric_name_key}'].append(value)
    metrics_computer.reset() # Reset for final test eval

    for test_batch in dataset_test_iter.as_numpy_iterator():
        eval_step(model, metrics_computer, test_batch)
    computed_test_metrics = metrics_computer.compute()
    for metric_name_key, value in computed_test_metrics.items():
        metrics_history[f'test_{metric_name_key}'].append(value)
    metrics_computer.reset()

    print(f"✅ {model_name} Model Training Complete on {dataset_name} after {current_num_epochs} epochs ({global_step_counter} steps)!")
    if metrics_history['test_accuracy']:
        print(f"   Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")
    else:
        print(f"   No test accuracy recorded for {model_name}.")
    
    return model, metrics_history


# ===== NEW ADVANCED ANALYSIS FUNCTIONS =====

def visualize_kernels(yat_model, linear_model, layer_name='conv1', num_kernels_to_show=16):
    """
    Visualize the kernels of the first convolutional layer for both models.
    """
    print(f"\n🎨 VISUALIZING KERNELS FROM LAYER: {layer_name}")
    print("=" * 50)

    def get_kernels(model, layer_name_str):
        try:
            layer = getattr(model, layer_name_str)
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                kernels = layer.kernel.value
                return kernels
            else:
                print(f"Kernel not found or is None in layer {layer_name_str} of {model.__class__.__name__}")
                return None
        except AttributeError:
            print(f"Layer {layer_name_str} not found in {model.__class__.__name__}")
            return None

    yat_kernels = get_kernels(yat_model, layer_name)
    linear_kernels = get_kernels(linear_model, layer_name)

    if yat_kernels is None and linear_kernels is None:
        print("Could not retrieve kernels for either model.")
        return

    def plot_kernel_grid(kernels, model_name_str, num_kernels):
        if kernels is None:
            print(f"No kernels to plot for {model_name_str}")
            return

        kh, kw, in_c, out_c = kernels.shape
        num_kernels = min(num_kernels, out_c)
        cols = int(np.ceil(np.sqrt(num_kernels)))
        rows = int(np.ceil(num_kernels / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(f'{model_name_str} - Layer {layer_name} Kernels (First {num_kernels} of {out_c})', fontsize=16)
        
        for i in range(num_kernels):
            ax = axes.flat[i] if num_kernels > 1 else axes
            if i < out_c:
                kernel_slice = kernels[:, :, 0, i] 
                kernel_slice = (kernel_slice - np.min(kernel_slice)) / (np.max(kernel_slice) - np.min(kernel_slice) + 1e-5)
                ax.imshow(kernel_slice, cmap='viridis')
                ax.set_title(f'Kernel {i+1}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        for i in range(num_kernels, len(axes.flat)):
             axes.flat[i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    if yat_kernels is not None:
        plot_kernel_grid(np.array(yat_kernels), "YAT Model", num_kernels_to_show)
    if linear_kernels is not None:
        plot_kernel_grid(np.array(linear_kernels), "Linear Model", num_kernels_to_show)
        
    print("🖼️ Kernel visualization complete (if kernels were found and plotted).")


def get_activation_maps(model, layer_name, input_sample, training=False):
    """
    Extracts activation maps from a specified layer of the model
    by calling the model with the 'return_activations_for_layer' argument.
    """
    try:
        # Call the model, requesting activations for the specified layer
        activations = model(input_sample, training=training, return_activations_for_layer=layer_name)
        
        # The model's __call__ method now handles printing a warning if the layer is not found
        # and will return the final output in that case. Consumers of this function
        # should be aware of this behavior if the layer_name is mistyped.
        return activations
        
    except Exception as e:
        print(f"Error getting activations for {layer_name} in {model.__class__.__name__}: {e}")
        return None


def activation_map_visualization(yat_model, linear_model, test_ds_iter, layer_name='conv1', num_maps_to_show=16):
    """
    Visualize activation maps from a specified layer for a sample input.
    test_ds_iter: An iterable TFDS dataset (already batched and preprocessed).
    """
    print(f"\n🗺️ VISUALIZING ACTIVATION MAPS FROM LAYER: {layer_name}")
    print("=" * 50)

    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except tf.errors.OutOfRangeError:
        print("ERROR: Test dataset iterator for activation maps is exhausted. Consider re-creating it or passing a fresh one.")
        # Fallback: Try to use the global test_ds if available, but warn this might be for the wrong dataset
        try:
            print(f"Warning: Falling back to global test_ds for activation maps. This might use data from '{_DEFAULT_DATASET_FOR_GLOBALS}'.")
            sample_batch = next(test_ds.as_numpy_iterator()) # Global test_ds
        except Exception as e_global:
            print(f"Error: Could not get sample batch for activation maps: {e_global}")
            return

    sample_image = sample_batch['image'][0:1]  # Take the first image, keep batch dim

    yat_activations = get_activation_maps(yat_model, layer_name, sample_image)
    linear_activations = get_activation_maps(linear_model, layer_name, sample_image)

    if yat_activations is None and linear_activations is None:
        print("Could not retrieve activation maps for either model.")
        return

    def plot_activation_grid(activations, model_name_str, num_maps):
        if activations is None:
            print(f"No activation maps to plot for {model_name_str}")
            return
        
        activations_np = np.array(activations)
        if activations_np.ndim == 4:
            activations_np = activations_np[0]
        else:
            print(f"Unexpected activation shape for {model_name_str}: {activations_np.shape}")
            return

        num_channels = activations_np.shape[-1]
        num_maps = min(num_maps, num_channels)
        cols = int(np.ceil(np.sqrt(num_maps)))
        rows = int(np.ceil(num_maps / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(f'{model_name_str} - Layer {layer_name} Activation Maps (First {num_maps})', fontsize=16)
        
        for i in range(num_maps):
            ax = axes.flat[i] if num_maps > 1 else axes
            if i < num_channels:
                ax.imshow(activations_np[:, :, i], cmap='viridis')
                ax.set_title(f'Map {i+1}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        for i in range(num_maps, len(axes.flat)):
             axes.flat[i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    if yat_activations is not None:
        plot_activation_grid(yat_activations, "YAT Model", num_maps_to_show)
    if linear_activations is not None:
        plot_activation_grid(linear_activations, "Linear Model", num_maps_to_show)

    print("🗺️ Activation map visualization complete (if maps were found and plotted).")


def saliency_map_analysis(yat_model, linear_model, test_ds_iter, class_names: list[str]):
    """
    Generate and visualize saliency maps for both models.
    test_ds_iter: An iterable TFDS dataset (already batched and preprocessed).
    class_names: List of class names for the current dataset.
    """
    print(f"\n🔥 SALIENCY MAP ANALYSIS for {len(class_names)} classes")
    print("=" * 50)

    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except tf.errors.OutOfRangeError:
        print("ERROR: Test dataset iterator for saliency maps is exhausted. Consider re-creating it or passing a fresh one.")
        # Fallback: Try to use the global test_ds if available, but warn this might be for the wrong dataset
        try:
            print(f"Warning: Falling back to global test_ds for saliency maps. This might use data from '{_DEFAULT_DATASET_FOR_GLOBALS}'.")
            sample_batch = next(test_ds.as_numpy_iterator()) # Global test_ds
        except Exception as e_global:
            print(f"Error: Could not get sample batch for saliency maps: {e_global}")
            return
            
    sample_image = sample_batch['image'][0:1]  # Take the first image, keep batch dim
    sample_label = int(sample_batch['label'][0]) # Ensure sample_label is a Python int

    # cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    true_class_name = class_names[sample_label]

    @partial(jax.jit, static_argnums=(0, 2)) # Modified: Added 2 to static_argnums
    def get_saliency_map(model, image_input, class_index=None):
        def model_output_for_grad(img):
            logits = model(img, training=False)
            if class_index is not None:
                # Ensure class_index is valid for the current model's output logits
                num_model_classes = logits.shape[-1]
                if class_index >= num_model_classes:
                    print(f"Warning: class_index {class_index} is out of bounds for model with {num_model_classes} classes. Using 0 instead.")
                    safe_class_index = 0
                else:
                    safe_class_index = class_index
                return logits[0, safe_class_index]
            else:
                return jnp.max(logits[0]) # Logit for the predicted class

        grads = jax.grad(model_output_for_grad)(image_input)
        saliency = jnp.max(jnp.abs(grads[0]), axis=-1)
        return saliency

    yat_logits_sample = yat_model(sample_image, training=False)
    yat_predicted_class_idx = int(jnp.argmax(yat_logits_sample, axis=1)[0]) # Ensure is Python int
    yat_predicted_class_name = class_names[yat_predicted_class_idx]

    linear_logits_sample = linear_model(sample_image, training=False)
    linear_predicted_class_idx = int(jnp.argmax(linear_logits_sample, axis=1)[0]) # Ensure is Python int
    linear_predicted_class_name = class_names[linear_predicted_class_idx]

    print(f"Sample image true class: {true_class_name} (Index: {sample_label})")
    print(f"YAT predicted class: {yat_predicted_class_name} (Index: {yat_predicted_class_idx})")
    print(f"Linear predicted class: {linear_predicted_class_name} (Index: {linear_predicted_class_idx})")

    yat_saliency_true_class = get_saliency_map(yat_model, sample_image, class_index=sample_label)
    linear_saliency_true_class = get_saliency_map(linear_model, sample_image, class_index=sample_label)
    
    yat_saliency_pred_class = get_saliency_map(yat_model, sample_image, class_index=yat_predicted_class_idx)
    linear_saliency_pred_class = get_saliency_map(linear_model, sample_image, class_index=linear_predicted_class_idx)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Saliency Map Comparison (True Class: {true_class_name})", fontsize=16)

    img_display = sample_image[0]
    img_display = (img_display - np.min(img_display)) / (np.max(img_display) - np.min(img_display) + 1e-5)

    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title(f"YAT Input (True: {true_class_name})")
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(np.array(yat_saliency_true_class), cmap='hot')
    axes[0, 1].set_title(f"YAT Saliency (for True: {true_class_name})")
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0,1])

    im2 = axes[0, 2].imshow(np.array(yat_saliency_pred_class), cmap='hot')
    axes[0, 2].set_title(f"YAT Saliency (for Pred: {yat_predicted_class_name})")
    axes[0, 2].axis('off')
    fig.colorbar(im2, ax=axes[0,2])

    axes[1, 0].imshow(img_display)
    axes[1, 0].set_title(f"Linear Input (True: {true_class_name})")
    axes[1, 0].axis('off')
    
    im3 = axes[1, 1].imshow(np.array(linear_saliency_true_class), cmap='hot')
    axes[1, 1].set_title(f"Linear Saliency (for True: {true_class_name})")
    axes[1, 1].axis('off')
    fig.colorbar(im3, ax=axes[1,1])

    im4 = axes[1, 2].imshow(np.array(linear_saliency_pred_class), cmap='hot')
    axes[1, 2].set_title(f"Linear Saliency (for Pred: {linear_predicted_class_name})")
    axes[1, 2].axis('off')
    fig.colorbar(im4, ax=axes[1,2])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    print("🔥 Saliency map analysis and visualization complete.")




def run_complete_comparison(dataset_name: str = 'cifar10'): # Added dataset_name argument
    """
    Complete implementation that runs both models, saves metrics, and performs comparison.
    Run this function to get a full comparison between YAT and Linear models on the specified dataset.
    """
    
    print("\n" + "="*80)
    print(f"                    RUNNING COMPLETE MODEL COMPARISON FOR: {dataset_name.upper()}")
    print("="*80)
    
    # Common training parameters (could be moved to DATASET_CONFIGS if they vary a lot)
    learning_rate = 0.003
    momentum = 0.9
    # current_train_steps, current_eval_every, current_batch_size are now fetched/calculated inside _train_model_loop

    # Fetch dataset info for analysis functions that need it (e.g. class names)
    dataset_config = DATASET_CONFIGS.get(dataset_name, {})
    if not dataset_config:
        print(f"Warning: Dataset '{dataset_name}' not in DATASET_CONFIGS. Some features might use defaults or fail.")
        # Attempt to get class names using default label key if config is missing
        ds_builder_comp_fallback = tfds.builder(dataset_name)
        ds_info_comp_fallback = ds_builder_comp_fallback.info
        try:
            class_names_comp = ds_info_comp_fallback.features['label'].names
        except (KeyError, AttributeError):
            print(f"Could not infer class names for {dataset_name}, using a placeholder list.")
            # Fallback if even 'label' key doesn't work or has no names (e.g. regression task)
            try: # Try to get num_classes and create generic names
                num_classes_fallback = ds_info_comp_fallback.features['label'].num_classes
                class_names_comp = [f"Class {i}" for i in range(num_classes_fallback)]
            except: # Absolute fallback
                class_names_comp = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"] # Default to 10 generic classes
    else:
        ds_builder_comp = tfds.builder(dataset_name)
        ds_info_comp = ds_builder_comp.info
        class_names_comp = ds_info_comp.features[dataset_config.get('label_key', 'label')].names
    
    # Get batch size for the current dataset to correctly batch the evaluation dataset
    # If dataset_name is not in DATASET_CONFIGS, use a global default or a fallback.
    current_batch_size_for_eval = dataset_config.get('batch_size', _global_batch_size)


    # Step 1: Train YAT Model
    print(f"\n🚀 STEP 1: Training YAT Model on {dataset_name}...")
    print("-" * 50)
    
    yat_model, yat_metrics_history = _train_model_loop(
        model_class=YatCNN,
        model_name="YAT",
        dataset_name=dataset_name, # Pass dataset_name
        rng_seed=0,
        learning_rate=learning_rate,
        momentum=momentum,
        optimizer_constructor=optax.adamw
    )

    # Step 2: Train Linear Model
    print(f"\n🚀 STEP 2: Training Linear Model on {dataset_name}...")
    print("-" * 50)
    
    linear_model, linear_metrics_history = _train_model_loop(
        model_class=LinearCNN,
        model_name="Linear",
        dataset_name=dataset_name, # Pass dataset_name
        rng_seed=0, 
        learning_rate=learning_rate,
        momentum=momentum,
        optimizer_constructor=optax.adamw
    )

    # Step 3: Run All Comparisons
    print(f"\n📊 STEP 3: Running Complete Comparison Analysis for {dataset_name}...")
    print("-" * 50)
    
    # 3.1 Compare training curves
    print("\n📈 Comparing training curves...")
    compare_training_curves(yat_metrics_history, linear_metrics_history)
    
    # 3.2 Print final metrics comparison
    print_final_metrics_comparison(yat_metrics_history, linear_metrics_history)
    
    # 3.3 Analyze convergence
    analyze_convergence(yat_metrics_history, linear_metrics_history)
    
    # 3.4 Detailed test evaluation
    # Need to reload the test dataset here as the one from _train_model_loop is consumed / specific to its scope
    # Or pass the models and a fresh test_ds_iterable to detailed_test_evaluation
    print("\n🎯 Running detailed test evaluation...")
    # Prepare test_ds specifically for detailed_test_evaluation and other analysis functions
    eval_config = DATASET_CONFIGS.get(dataset_name, {})
    eval_image_key = eval_config.get('image_key', 'image')
    eval_label_key = eval_config.get('label_key', 'label')
    eval_test_split = eval_config.get('test_split', 'test')

    def eval_preprocess_fn(sample):
        return {
            'image': tf.cast(sample[eval_image_key], tf.float32) / 255.0,
            'label': sample[eval_label_key]
        }
    current_test_ds_for_eval = tfds.load(dataset_name, split=eval_test_split, as_supervised=False) \
        .map(eval_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(current_batch_size_for_eval, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    predictions_data = detailed_test_evaluation(yat_model, linear_model, current_test_ds_for_eval, class_names=class_names_comp)
    
    # 3.5 Plot confusion matrices
    print("\n📊 Plotting confusion matrices...")
    plot_confusion_matrices(predictions_data) # predictions_data now contains class_names
    
    # 3.6 Generate comprehensive summary report
    generate_summary_report(yat_metrics_history, linear_metrics_history, predictions_data)
    
    # Step 4: Advanced Analysis (New)
    print("\n🔬 STEP 4: Running Advanced Analysis...")
    print("-" * 50)

    # 4.1 Visualize Kernels (e.g., from 'conv1')
    visualize_kernels(yat_model, linear_model, layer_name='conv1', num_kernels_to_show=16)

    # 4.2 Visualize Activation Maps (e.g., from 'conv1' for a sample from test_ds)
    # Use current_test_ds_for_eval or reload a small part of it
    activation_map_visualization(yat_model, linear_model, current_test_ds_for_eval, layer_name='conv1', num_maps_to_show=16)
    
    # 4.3 Saliency Map Analysis
    saliency_map_analysis(yat_model, linear_model, current_test_ds_for_eval, class_names=class_names_comp)
    
    print("\n" + "="*80)
    print(f"                    COMPARISON ANALYSIS FOR {dataset_name.upper()} COMPLETE! ✅")
    print("="*80)
    
    return {
        'yat_model': yat_model,
        'linear_model': linear_model,
        'yat_metrics_history': yat_metrics_history,
        'linear_metrics_history': linear_metrics_history,
        'predictions_data': predictions_data
    }

# ===== QUICK START FUNCTIONS =====

def quick_comparison_demo():
    """
    Quick demo that shows how to use the comparison functions with dummy data.
    Use this to test the comparison functions before running full training.
    """
    print("\n🎬 RUNNING QUICK COMPARISON DEMO...")
    print("-" * 50)
    
    # Create dummy metrics history for demonstration
    import random
    random.seed(42)
    
    steps = 30
    yat_dummy = {
        'train_loss': [1.5 - 0.04*i + random.random()*0.1 for i in range(steps)],
        'train_accuracy': [0.2 + 0.025*i + random.random()*0.05 for i in range(steps)],
        'test_loss': [1.6 - 0.035*i + random.random()*0.15 for i in range(steps)],
        'test_accuracy': [0.15 + 0.022*i + random.random()*0.08 for i in range(steps)]
    }
    
    linear_dummy = {
        'train_loss': [1.6 - 0.045*i + random.random()*0.1 for i in range(steps)],
        'train_accuracy': [0.18 + 0.024*i + random.random()*0.05 for i in range(steps)],
        'test_loss': [1.7 - 0.04*i + random.random()*0.15 for i in range(steps)],
        'test_accuracy': [0.12 + 0.023*i + random.random()*0.08 for i in range(steps)]
    }
    
    print("📈 Comparing dummy training curves...")
    compare_training_curves(yat_dummy, linear_dummy)
    
    print_final_metrics_comparison(yat_dummy, linear_dummy)
    analyze_convergence(yat_dummy, linear_dummy)
    
    print("✅ Demo complete! Now you can run the full comparison with real models.")

def save_metrics_example():
    """
    Shows how to properly save metrics history during training.
    """
    print("\n💾 HOW TO SAVE METRICS DURING TRAINING:")
    print("-" * 50)
    print("""
# After training your YAT model:
yat_metrics_history = metrics_history.copy()

# After training your Linear model:  
linear_metrics_history = metrics_history.copy()

# Or save to files:
import pickle
with open('yat_metrics.pkl', 'wb') as f:
    pickle.dump(yat_metrics_history, f)
    
with open('linear_metrics.pkl', 'wb') as f:
    pickle.dump(linear_metrics_history, f)

# Load later:
with open('yat_metrics.pkl', 'rb') as f:
    yat_metrics_history = pickle.load(f)
    
with open('linear_metrics.pkl', 'rb') as f:
    linear_metrics_history = pickle.load(f)
""")

# Print final instructions
print("\n" + "="*80)
print("="*80)
print("\n🚀 TO RUN THE COMPLETE COMPARISON (e.g., for CIFAR-10):")
print("   results = run_complete_comparison(dataset_name='cifar10')")
print("\n   Other examples:")
print("   results_cifar100 = run_complete_comparison(dataset_name='cifar100')")
print("   results_stl10 = run_complete_comparison(dataset_name='stl10')")
print("   results_eurosat_rgb = run_complete_comparison(dataset_name='eurosat/rgb')")
# print("   results_eurosat_all = run_complete_comparison(dataset_name='eurosat/all')") # Might be slow due to 13 channels

print("\n🎬 TO RUN A QUICK DEMO (uses dummy data, not specific dataset):")
print("   quick_comparison_demo()")
print("\n💾 TO SEE HOW TO SAVE METRICS:")
print("   save_metrics_example()")
print("\n📖 The comparison functions are ready to use:")
print("   - compare_training_curves(yat_history, linear_history)")
print("   - print_final_metrics_comparison(yat_history, linear_history)")
print("   - analyze_convergence(yat_history, linear_history)")
print("   - detailed_test_evaluation(yat_model, linear_model, test_ds)")
print("   - plot_confusion_matrices(predictions_data)")
print("   - generate_summary_report(yat_history, linear_history, predictions_data)")
print("\n   ✨ NEW ADVANCED ANALYSIS:")
print("   - visualize_kernels(yat_model, linear_model, layer_name='conv1', num_kernels_to_show=16)")
print("   - activation_map_visualization(yat_model, linear_model, test_ds_iter, layer_name='conv1', num_maps_to_show=16)")
print("   - saliency_map_analysis(yat_model, linear_model, test_ds_iter, class_names=class_names_comp)")
print("="*80)


results_stl10 = run_complete_comparison(dataset_name='stl10')