# Install required packages:
!pip install -Uq tiktoken grain matplotlib datasets
# pip install tensorflow-cpu wandb
# pip install --upgrade jax jaxlib flax grain
# pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

import jax
jax.devices()

# Download TinyStories dataset (can be replaced with HuggingFace datasets)
# wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true -O TinyStories-train.txt

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding # For data and model parallelism (explained in more detail later)
from jax.experimental import mesh_utils

import flax.nnx as nnx
import optax

from dataclasses import dataclass
import grain.python as pygrain
import pandas as pd
import tiktoken
import time
from typing import Optional, Dict, List, Union
import warnings

# Hugging Face datasets integration
try:
    from datasets import load_dataset, DatasetDict
    import datasets
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    warnings.warn("Hugging Face datasets not available. Install with: pip install datasets")

# Create a `Mesh` object representing TPU device arrangement.
mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))

tokenizer = tiktoken.get_encoding("gpt2")

# Dataset Configuration
@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing"""
    name: str = "roneneldan/TinyStories"  # Default dataset
    subset: Optional[str] = None  # For datasets with multiple subsets
    split: str = "train"
    text_column: str = "text"
    streaming: bool = False  # Use streaming for large datasets
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    
    # Text preprocessing options
    separator: str = "<|endoftext|>"  # Token to separate documents
    min_length: int = 10  # Minimum text length to include
    max_length: Optional[int] = None  # Maximum text length before truncation
    
    # File-based dataset options (for local files)
    file_path: Optional[str] = None
    file_type: str = "txt"  # txt, json, csv, parquet

# Predefined dataset configurations
DATASET_CONFIGS = {
    "tinystories": DatasetConfig(
        name="roneneldan/TinyStories", 
        text_column="text",
        separator="<|endoftext|>"
    ),
    "wikitext": DatasetConfig(
        name="Salesforce/wikitext", 
        subset="wikitext-2-raw-v1",
        text_column="text",
        separator="\n\n"
    ),
    "openwebtext": DatasetConfig(
        name="Skylion007/openwebtext", 
        text_column="text",
        streaming=True,  # Large dataset, use streaming
        separator="<|endoftext|>"
    ),
    "bookscorpus": DatasetConfig(
        name="bookcorpus/bookcorpus", 
        text_column="text",
        trust_remote_code=True,
        separator="<|endoftext|>"
    ),
    "c4": DatasetConfig(
        name="allenai/c4", 
        subset="en",
        text_column="text",
        streaming=True,
        separator="<|endoftext|>"
    ),
    "tiny_shakespeare": DatasetConfig(
        name="tiny_shakespeare", 
        text_column="text",
        separator="\n\n"
    ),
    "gutenberg": DatasetConfig(
        name="sedthh/gutenberg_english", 
        text_column="text",
        separator="<|endoftext|>"
    ),
    "pile": DatasetConfig(
        name="EleutherAI/pile", 
        text_column="text",
        streaming=True,
        separator="<|endoftext|>"
    ),
    "common_crawl": DatasetConfig(
        name="oscar", 
        subset="unshuffled_deduplicated_en",
        text_column="text",
        streaming=True,
        separator="<|endoftext|>"
    ),
    "local_file": DatasetConfig(
        name="local",
        file_path="TinyStories-train.txt",
        file_type="txt",
        separator="<|endoftext|>"
    ),
    "fineweb": DatasetConfig(
        name="VisionTheta/fineweb-100B",
        file_type="txt",
        streaming=True,
        separator="<|endoftext|>"
    ),
    
}

def load_huggingface_dataset(config: DatasetConfig) -> List[str]:
    """Load dataset from Hugging Face Hub"""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("Hugging Face datasets not available. Install with: pip install datasets")
    
    print(f"Loading dataset: {config.name}")
    if config.subset:
        print(f"  Subset: {config.subset}")
    
    try:
        # Load dataset
        load_kwargs = {
            "path": config.name,
            "split": config.split,
            "streaming": config.streaming,
            "trust_remote_code": config.trust_remote_code
        }
        
        if config.subset:
            load_kwargs["name"] = config.subset
        if config.cache_dir:
            load_kwargs["cache_dir"] = config.cache_dir
            
        dataset = load_dataset(**load_kwargs)
        
        print(f"Dataset loaded successfully. Processing text...")
        
        # Extract text
        texts = []
        count = 0
        max_samples = 50000 if config.streaming else None  # Limit for streaming datasets
        
        for item in dataset:
            if max_samples and count >= max_samples:
                break
                
            # Extract text from item
            if config.text_column in item:
                text = item[config.text_column]
            elif isinstance(item, str):
                text = item
            else:
                print(f"Warning: Text column '{config.text_column}' not found in item: {item.keys()}")
                continue
            
            # Filter by length
            if len(text) < config.min_length:
                continue
            if config.max_length and len(text) > config.max_length:
                text = text[:config.max_length]
            
            texts.append(text)
            count += 1
            
            if count % 10000 == 0:
                print(f"  Processed {count} samples...")
        
        print(f"Dataset processing complete. Total samples: {len(texts)}")
        return texts
        
    except Exception as e:
        print(f"Error loading dataset {config.name}: {e}")
        print("Falling back to local file if available...")
        return []

def load_local_file(config: DatasetConfig) -> List[str]:
    """Load dataset from local file"""
    if not config.file_path:
        raise ValueError("file_path must be specified for local datasets")
    
    print(f"Loading local file: {config.file_path}")
    
    try:
        if config.file_type == "txt":
            with open(config.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split text by separator
            if config.separator in text:
                texts = text.split(config.separator)
                texts = [t.strip() + config.separator for t in texts if t.strip()]
            else:
                # Split by paragraphs if no separator found
                texts = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        elif config.file_type == "json":
            import json
            with open(config.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                texts = [item[config.text_column] if isinstance(item, dict) else str(item) for item in data]
            else:
                texts = [data[config.text_column]] if isinstance(data, dict) else [str(data)]
        
        elif config.file_type == "csv":
            import csv
            texts = []
            with open(config.file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if config.text_column in row:
                        texts.append(row[config.text_column])
        
        else:
            raise ValueError(f"Unsupported file type: {config.file_type}")
        
        # Filter by length
        filtered_texts = []
        for text in texts:
            if len(text) >= config.min_length:
                if config.max_length and len(text) > config.max_length:
                    text = text[:config.max_length]
                filtered_texts.append(text)
        
        print(f"Local file loaded successfully. Total samples: {len(filtered_texts)}")
        return filtered_texts
        
    except Exception as e:
        print(f"Error loading local file: {e}")
        return []

def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about available datasets"""
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        return {
            "name": config.name,
            "subset": config.subset,
            "text_column": config.text_column,
            "separator": config.separator,
            "streaming": config.streaming,
            "description": f"Predefined configuration for {dataset_name}"
        }
    else:
        return {"error": f"Dataset '{dataset_name}' not found in predefined configurations"}

def list_available_datasets() -> List[str]:
    """List all available dataset configurations"""
    return list(DATASET_CONFIGS.keys())

"""Neural-Matter Network Definition"""

# Commented out IPython magic to ensure Python compatibility.
# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention core modules for Flax."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional
import typing as tp

import jax
import jax.numpy as jnp
from jax import lax, random

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import initializers
from flax.nnx.nn.dtypes import promote_dtype
from flax.nnx.nn.linear import (
  LinearGeneral,
  default_kernel_init,
)
from flax.nnx.nn.normalization import LayerNorm
from flax.typing import (
  Dtype,
  Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)



def yat_attention_weights(
  query: Array,
  key: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
  epsilon: float = 1e-5,
):
  """Computes attention weights using YatNMN distance-based calculation."""
  query, key = promote_dtype((query, key), dtype=dtype)
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # YatNMN-style attention calculation using the cleaner approach
  # query shape: [..., q_length, num_heads, head_dim]
  # key shape: [..., kv_length, num_heads, head_dim]

  # Calculate dot product attention scores
  attn = jnp.einsum('...qhd,...khd->...hqk', query, key, precision=precision)
  squared_dot_product = jnp.square(attn)

  # Calculate norms
  q_norm = jnp.sum(jnp.square(query), axis=-1, keepdims=True)  # [..., q_length, num_heads, 1]
  k_norm = jnp.sum(jnp.square(key), axis=-1, keepdims=True)    # [..., kv_length, num_heads, 1]
  qk_norm_sum = q_norm + k_norm  # Broadcasting: [..., q_length, num_heads, 1] + [..., kv_length, num_heads, 1]

  # Transpose to match attention dimensions [..., num_heads, q_length, kv_length]
  # The transpose converts [..., q_length, num_heads, kv_length] -> [..., num_heads, q_length, kv_length]
  batch_dims = len(qk_norm_sum.shape) - 3
  transpose_axes = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
  qk_norm_sum_transposed = qk_norm_sum.transpose(transpose_axes)

  # Calculate squared distances: ||q||² + ||k||² - 2*(q·k)²
  squared_dist = qk_norm_sum_transposed - 2.0 * squared_dot_product

  # YatNMN attention scores: (q·k)² / (squared_distance + ε)
  attn_weights = squared_dot_product / (squared_dist + epsilon)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if module:
    module.sow(nnx.Intermediate, 'attention_weights', attn_weights)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights


def yat_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
  epsilon: float = 1e-5,
):
  """Computes attention using YatNMN distance-based calculation."""
  query, key, value = promote_dtype((query, key, value), dtype=dtype)
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights using YatNMN
  attn_weights = yat_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    module,
    epsilon,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
    '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )

Array = jax.Array

# Add YatNMN class implementation
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()

class YatNMN(Module):
  """A linear transformation with custom distance-based computation."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    use_alpha: bool = True,
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    alpha_init: Initializer = default_alpha_init,
    dot_general: DotGeneralT = lax.dot_general,
    rngs: rnglib.Rngs,
    epsilon: float = 1e-5,
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
    self.epsilon = epsilon

  def __call__(self, inputs: Array) -> Array:
    """Applies YatNMN transformation to inputs."""
    kernel = self.kernel.value
    bias = self.bias.value if self.bias is not None else None
    alpha = self.alpha.value if self.alpha is not None else None

    y = self.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )

    inputs_squared_sum = jnp.sum(inputs**2, axis=-1, keepdims=True)
    kernel_squared_sum = jnp.sum(kernel**2, axis=0, keepdims=True)
    distances = inputs_squared_sum + kernel_squared_sum - 2 * y

    # Element-wise operation
    y = y ** 2 / (distances + self.epsilon)

    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    if alpha is not None:
      scale = (jnp.sqrt(self.out_features) / jnp.log(1 + self.out_features)) ** alpha
      y = y * scale

    return y


def dot_product_attention_weights(
  query: Array,
  key: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      ``nnx.Intermediate`` collection. If ``module`` is None, the attention
      weights will not be sowed.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  query, key = promote_dtype((query, key), dtype=dtype)  # type: ignore[bad-unpacking]
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
    '...qhd,...khd->...hqk', query, key, precision=precision
  )

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if module:
    module.sow(nnx.Intermediate, 'attention_weights', attn_weights)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights


def dot_product_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  .. note::
    ``query``, ``key``, ``value`` needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of ``[batch..., q_length,
      num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    value: values to be used in attention with shape of ``[batch..., kv_length,
      num_heads, v_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      ``nnx.Intermediate`` collection. If ``module`` is None, the attention
      weights will not be sowed.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  query, key, value = promote_dtype((query, key, value), dtype=dtype)  # type: ignore[bad-unpacking]
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    module,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
    '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )


class MultiHeadAttention(Module):
  """Multi-head attention.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax

    >>> layer = nn.MultiHeadAttention(num_heads=8, qkv_features=16)
    >>> key1, key2, key3, key4, key5, key6 = jax.random.split(jax.random.key(0), 6)
    >>> shape = (4, 3, 2, 5)
    >>> q, k, v = jax.random.uniform(key1, shape), jax.random.uniform(key2, shape), jax.random.uniform(key3, shape)
    >>> variables = layer.init(jax.random.key(0), q)

    >>> # different inputs for inputs_q, inputs_k and inputs_v
    >>> out = layer.apply(variables, q, k, v)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=k, inputs_v=k)
    >>> out = layer.apply(variables, q, k)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=q) and layer.apply(variables, inputs_q=q, inputs_k=q, inputs_v=q)
    >>> out = layer.apply(variables, q)

    >>> attention_kwargs = dict(
    ...     num_heads=8,
    ...     qkv_features=16,
    ...     kernel_init=nn.initializers.ones,
    ...     bias_init=nn.initializers.zeros,
    ...     dropout_rate=0.5,
    ...     deterministic=False,
    ...     )
    >>> class Module(nn.Module):
    ...   attention_kwargs: dict
    ...
    ...   @nn.compact
    ...   def __call__(self, x, dropout_rng=None):
    ...     out1 = nn.MultiHeadAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     out2 = nn.MultiHeadAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     return out1, out2
    >>> module = Module(attention_kwargs)
    >>> variables = module.init({'params': key1, 'dropout': key2}, q)

    >>> # out1 and out2 are different.
    >>> out1, out2 = module.apply(variables, q, rngs={'dropout': key3})
    >>> # out3 and out4 are different.
    >>> # out1 and out3 are different. out2 and out4 are different.
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key4})
    >>> # out1 and out2 are the same.
    >>> out1, out2 = module.apply(variables, q, dropout_rng=key5)
    >>> # out1 and out2 are the same as out3 and out4.
    >>> # providing a `dropout_rng` arg will take precedence over the `rngs` arg in `.apply`
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key6}, dropout_rng=key5)

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: infer from inputs and params)
    param_dtype: the dtype passed to parameter initializers (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly using
      dropout, whereas if true, the attention weights are deterministic.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    out_kernel_init: optional initializer for the kernel of the output Dense layer,
      if None, the kernel_init is used.
    bias_init: initializer for the bias of the Dense layers.
    out_bias_init: optional initializer for the bias of the output Dense layer,
      if None, the bias_init is used.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    decode: whether to prepare and use an autoregressive cache.
    normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
  """

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    qkv_features: int | None = None,
    out_features: int | None = None,
    *,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.0,
    deterministic: bool | None = None,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    out_kernel_init: Initializer | None = None,
    bias_init: Initializer = initializers.zeros_init(),
    out_bias_init: Initializer | None = None,
    use_bias: bool = True,
    attention_fn: Callable[..., Array] = yat_attention,
    decode: bool | None = None,
    normalize_qk: bool = False,
    # Deprecated, will be removed.
    qkv_dot_general: DotGeneralT | None = None,
    out_dot_general: DotGeneralT | None = None,
    qkv_dot_general_cls: Any = None,
    out_dot_general_cls: Any = None,
    rngs: rnglib.Rngs,
    epsilon: float = 1e-5,
  ):
    self.num_heads = num_heads
    self.in_features = in_features
    self.qkv_features = (
      qkv_features if qkv_features is not None else in_features
    )
    self.out_features = (
      out_features if out_features is not None else in_features
    )
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.broadcast_dropout = broadcast_dropout
    self.dropout_rate = dropout_rate
    self.deterministic = deterministic
    self.precision = precision
    self.kernel_init = kernel_init
    self.out_kernel_init = out_kernel_init
    self.bias_init = bias_init
    self.out_bias_init = out_bias_init
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.decode = decode
    self.normalize_qk = normalize_qk
    self.qkv_dot_general = qkv_dot_general
    self.out_dot_general = out_dot_general
    self.qkv_dot_general_cls = qkv_dot_general_cls
    self.out_dot_general_cls = out_dot_general_cls
    self.epsilon = epsilon

    if self.qkv_features % self.num_heads != 0:
      raise ValueError(
        f'Memory dimension ({self.qkv_features}) must be divisible by '
        f"'num_heads' heads ({self.num_heads})."
      )

    self.head_dim = self.qkv_features // self.num_heads

    # Replace LinearGeneral with YatNMN for query, key, value projections
    yat_linear = functools.partial(
      YatNMN,
      in_features=self.in_features,
      out_features=self.qkv_features,  # Output total features, will reshape later
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      epsilon=self.epsilon,
    )

    # project inputs_q to multi-headed q/k/v
    # dimensions will be reshaped to [batch..., length, n_heads, n_features_per_head]
    self.query = yat_linear(rngs=rngs)
    self.key = yat_linear(rngs=rngs)
    self.value = yat_linear(rngs=rngs)

    self.query_ln: LayerNorm | None
    self.key_ln: LayerNorm | None
    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      self.query_ln = LayerNorm(
        self.head_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
      )
      self.key_ln = LayerNorm(
        self.head_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
      )
    else:
      self.query_ln = None
      self.key_ln = None

    # Remove the output layer - no more self.out
    self.rngs = rngs if dropout_rate > 0.0 else None

    self.cached_key: nnx.Cache[Array] | None = None
    self.cached_value: nnx.Cache[Array] | None = None
    self.cache_index: nnx.Cache[Array] | None = None

  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Array | None = None,
    inputs_v: Array | None = None,
    *,
    mask: Array | None = None,
    deterministic: bool | None = None,
    rngs: rnglib.Rngs | None = None,
    sow_weights: bool = False,
    decode: bool | None = None,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
        inputs_v will copy the value of inputs_k.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      rngs: container for random number generators to generate the dropout
        mask when `deterministic` is False. The `rngs` container should have a
        `dropout` key.
      sow_weights: if ``True``, the attention weights are sowed into the
        'intermediates' collection.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if rngs is None:
      rngs = self.rngs

    if inputs_k is None:
      if inputs_v is not None:
        raise ValueError(
          '`inputs_k` cannot be None if `inputs_v` is not None. '
          'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
          'value to `inputs_k` and leave `inputs_v` as None.'
        )
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k

    if inputs_q.shape[-1] != self.in_features:
      raise ValueError(
        f'Incompatible input dimension, got {inputs_q.shape[-1]} '
        f'but module expects {self.in_features}.'
      )

    # Apply YatNMN transformations and reshape to multi-head format
    query = self.query(inputs_q)
    key = self.key(inputs_k)
    value = self.value(inputs_v)

    # Reshape from [batch..., length, qkv_features] to [batch..., length, num_heads, head_dim]
    query = query.reshape(query.shape[:-1] + (self.num_heads, self.head_dim))
    key = key.reshape(key.shape[:-1] + (self.num_heads, self.head_dim))
    value = value.reshape(value.shape[:-1] + (self.num_heads, self.head_dim))

    if self.normalize_qk:
      assert self.query_ln is not None and self.key_ln is not None
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = self.query_ln(query)
      key = self.key_ln(key)

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    decode = first_from(
      decode,
      self.decode,
      error_msg="""No `decode` argument was provided to MultiHeadAttention
        as either a __call__ argument, class attribute, or nnx.flag.""",
    )

    if decode:
      if (
        self.cached_key is None
        or self.cached_value is None
        or self.cache_index is None
      ):
        raise ValueError(
          'Autoregressive cache not initialized, call ``init_cache`` first.'
        )
      (
        *batch_dims,
        max_length,
        num_heads,
        depth_per_head,
      ) = self.cached_key.value.shape
      # shape check of cached keys against query input
      expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
      if expected_shape != query.shape:
        raise ValueError(
          'Autoregressive cache shape error, '
          'expected query shape %s instead got %s.'
#           % (expected_shape, query.shape)
        )
      # update key, value caches with our new 1d spatial slices
      cur_index = self.cache_index.value
      zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
      indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
      key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
      value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
      self.cached_key.value = key
      self.cached_value.value = value
      self.cache_index.value += 1
      # causal mask for cached decoder self-attention:
      # our single query position should only attend to those key
      # positions that have already been generated and cached,
      # not the remaining zero elements.
      mask = combine_masks(
        mask,
        jnp.broadcast_to(
          jnp.arange(max_length) <= cur_index,
          tuple(batch_dims) + (1, 1, max_length),
        ),
      )

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      deterministic = first_from(
        deterministic,
        self.deterministic,
        error_msg="""No `deterministic` argument was provided to MultiHeadAttention
          as either a __call__ argument, class attribute, or nnx.flag.""",
      )
      if not deterministic:
        if rngs is None:
          raise ValueError(
            "'rngs' must be provided if 'dropout_rng' is not given."
          )
        dropout_rng = rngs.dropout()
      else:
        dropout_rng = None
    else:
      deterministic = True
      dropout_rng = None

    # apply attention with epsilon parameter for YatNMN
    x = self.attention_fn(
      query,
      key,
      value,
      mask=mask,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      deterministic=deterministic,
      dtype=self.dtype,
      precision=self.precision,
      module=self if sow_weights else None,
      epsilon=self.epsilon,  # Pass epsilon to yat_attention
    )
    # Reshape attention output back to original embedding dimension
    # from [batch..., length, num_heads, head_dim] to [batch..., length, qkv_features]
    x = x.reshape(x.shape[:-2] + (self.qkv_features,))
    return x

  def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
    """Initializes cache for fast autoregressive decoding. When
    ``decode=True``, this method must be called first before performing
    forward inference.

    Example usage::

      >>> from flax import nnx
      >>> import jax.numpy as jnp
      ...
      >>> rngs = nnx.Rngs(42)
      ...
      >>> x = jnp.ones((1, 3))
      >>> model_nnx = nnx.MultiHeadAttention(
      ...   num_heads=2,
      ...   in_features=3,
      ...   qkv_features=6,
      ...   out_features=6,
      ...   decode=True,
      ...   rngs=rngs,
      ... )
      ...
      >>> # out_nnx = model_nnx(x)  <-- throws an error because cache isn't initialized
      ...
      >>> model_nnx.init_cache(x.shape)
      >>> out_nnx = model_nnx(x)
    """
    cache_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
    self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype))
    self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype))
    self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))


# mask-making utility functions


def make_attention_mask(
  query_input: Array,
  key_input: Array,
  pairwise_fn: Callable[..., Any] = jnp.multiply,
  extra_batch_dims: int = 0,
  dtype: Dtype = jnp.float32,
):
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
  attention weights will be `[batch..., heads, len_q, len_kv]` and this
  function will produce `[batch..., 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  mask = pairwise_fn(
    jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2)
  )
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(
  x: Array, extra_batch_dims: int = 0, dtype: Dtype = jnp.float32
) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
  will be `[batch..., heads, len, len]` and this function will produce a
  causal mask of shape `[batch..., 1, len, len]`.

  Args:
    x: input array of shape `[batch..., len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
    idxs,
    idxs,
    jnp.greater_equal,
    extra_batch_dims=extra_batch_dims,
    dtype=dtype,
  )


def combine_masks(
  *masks: Optional[Array], dtype: Dtype = jnp.float32
) -> Array | None:
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: dtype for the returned mask.

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks_list = [m for m in masks if m is not None]
  if not masks_list:
    return None
  assert all(
    map(lambda x: x.ndim == masks_list[0].ndim, masks_list)
  ), f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks_list))}'
  mask, *other_masks = masks_list
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)



# Define a triangular mask for causal attention with `jax.numpy.tril` and `jax.numpy.ones`.
def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):
    """ A single Transformer block.

    Each Transformer block processes input sequences via self-attention and feed-forward networks.

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1):
        # Multi-Head Attention (MHA) with `flax.nnx.MultiHeadAttention`.
        # Specifies tensor sharding (depending on the mesh configuration)
        # where we shard the weights across devices for parallel computation.
        self.mha = MultiHeadAttention(num_heads=num_heads,
                                          in_features=embed_dim,
                                          kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                          bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                          rngs=rngs)
        # The first dropout with `flax.nnx.Dropout`.
        self.dropout1 = nnx.Dropout(rate=rate)
        # The first linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.nonlinear1 = YatNMN(in_features=embed_dim,
                                  out_features=embed_dim,
                                  kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                  bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                  alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                  rngs=rngs)
        # The second dropout with `flax.nnx.Dropout`.
        self.dropout2 = nnx.Dropout(rate=rate)


    # Apply the Transformer block to the input sequence.
    def __call__(self, inputs, training: bool = False):
        input_shape = inputs.shape
        _, seq_len, _ = input_shape

        # Instantiate the causal attention mask.
        mask = causal_attention_mask(seq_len)

        # Apply Multi-Head Attention with the causal attention mask.
        attention_output = self.mha(
            inputs_q=inputs,
            mask=mask,
            decode=False
        )
        # Apply the first dropout.
        attention_output = self.dropout1(attention_output, deterministic=not training)
        # Apply the first layer normalization.
        out1 = inputs + attention_output

        # The feed-forward network.
        # Apply the first linear transformation.
        ffn_output = self.nonlinear1(out1)
        # Apply the second dropout.
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        # Apply the second layer normalization and return the output of the Transformer block.
        return out1 + ffn_output

class TokenAndPositionEmbedding(nnx.Module):
    """ Combines token embeddings (words in an input sentence) with
    positional embeddings (the position of each word in a sentence).

    Args:
        maxlen (int): Matimum sequence length.
        vocal_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        # Initialize token embeddings (using `flax.nnx.Embed`).
        # Each unique word has an embedding vector.
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        # Initialize positional embeddings (using `flax.nnx.Embed`).
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    # Takes a token sequence (integers) and returns the combined token and positional embeddings.
    def __call__(self, x):
        # Generate a sequence of positions for the input tokens.
        positions = jnp.arange(0, x.shape[1])[None, :]
        # Look up the positional embeddings for each position in the input sequence.
        position_embedding = self.pos_emb(positions)
        # Look up the token embeddings for each token in the input sequence.
        token_embedding = self.token_emb(x)
        # Combine token and positional embeddings.
        return token_embedding + position_embedding

class MiniGPT(nnx.Module):
    """ A miniGPT transformer model, inherits from `flax.nnx.Module`.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        feed_forward_dim (int): Dimensionality of the feed-forward network.
        num_transformer_blocks (int): Number of transformer blocks. Each block contains attention and feed-forward networks.
        rngs (nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    # Initialize miniGPT model components.
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        # Initiliaze the `TokenAndPositionEmbedding` that combines token and positional embeddings.
        self.embedding_layer = TokenAndPositionEmbedding(
                    maxlen, vocab_size, embed_dim, rngs=rngs
                )
        # Create a list of `TransformerBlock` instances.
        # Each block processes input sequences using attention and feed-forward networks.
        self.transformer_blocks = [TransformerBlock(
            embed_dim, num_heads, feed_forward_dim, rngs=rngs
        ) for _ in range(num_transformer_blocks)]
        # Initialize the output `flax.nnx.Linear` layer producing logits over the vocabulary for next-token prediction.
        self.output_layer = YatNMN(in_features=embed_dim,
                                       out_features=vocab_size,
                                       kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                       bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                       alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                       use_bias=False,
                                       rngs=rngs,
                                  )

    def __call__(self, inputs, training: bool = False):
        # Pass the input tokens through the `embedding_layer` to get token embeddings.
        # Apply each transformer block sequentially to the embedded input, use the `training` flag for the behavior of `flax.nnx.Dropout`.
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        # Pass the output of the transformer blocks through the output layer,
        # and obtain logits for each token in the vocabulary (for next token prediction).
        outputs = self.output_layer(x)
        return outputs

    # Text generation.
    def generate_text(self, max_tokens: int, start_tokens: [int], top_k=10):
        # Sample the next token from a probability distribution based on
        # `logits` and `tok_k` (top-k) sampling strategy.
        def sample_from(logits):
            logits, indices = jax.lax.top_k(logits, k=top_k)
            # Convert logits to probabilities (using `flax.nnx.softmax`).
            logits = nnx.softmax(logits)
            return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

        # Generate text one token at a time until the maximum token limit is reached (`maxlen`).
        def generate_step(start_tokens):
            pad_len = maxlen - len(start_tokens)
            # Index of the last token in the current sequence.
            sample_index = len(start_tokens) - 1
            # If the input is longer than `maxlen`, then truncate it.
            if pad_len < 0:
                x = jnp.array(start_tokens[:maxlen])
                sample_index = maxlen - 1
            # If the input is shorter than `maxlen`, then pad it (`pad_len`).
            elif pad_len > 0:
                x = jnp.array(start_tokens + [0] * pad_len)
            else:
                x = jnp.array(start_tokens)

            # Add a batch dimension.
            x = x[None, :]
            logits = self(x)
            next_token = sample_from(logits[0][sample_index])
            return next_token

        # Store generated tokens.
        generated = []
        # Generate tokens until the end-of-text token is encountered or the maximum token limit is reached.
        for _ in range(max_tokens):
            next_token = generate_step(start_tokens + generated)
            # Truncate whatever is after '<|endoftext|>' (stop word)
            if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
              # Stop text generation if the end-of-text token is encountered.
              break
            generated.append(int(next_token))
        # Decode the generated token IDs into text.
        return tokenizer.decode(start_tokens + generated)

# Creates the miniGPT model with 4 transformer blocks.
def create_model(rngs):
    return MiniGPT(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim, num_transformer_blocks=4, rngs=rngs)

vocab_size = tokenizer.n_vocab
num_transformer_blocks = 3
maxlen = 1024
embed_dim = 512
num_heads = 8
feed_forward_dim = 512
batch_size = 64 # You can set a bigger batch size if you use Kaggle's Cloud TPU.
num_epochs = 5

"""## Loading and preprocessing the data

Enhanced data loading with support for multiple Hugging Face datasets and local files.
"""

# Configuration for dataset selection
CURRENT_DATASET = "fineweb"  # Change this to use different datasets
# Available options: tinystories, wikitext, openwebtext, bookscorpus, c4, 
#                   tiny_shakespeare, gutenberg, pile, common_crawl, local_file

@dataclass
class EnhancedTextDataset:
    """Enhanced TextDataset with better preprocessing and flexible tokenization"""
    data: list
    maxlen: int
    tokenizer: any = tokenizer
    separator_token: str = "<|endoftext|>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        text = self.data[idx]
        # Use Tiktoken for tokenization with proper handling of special tokens
        try:
            encoding = self.tokenizer.encode(
                text, 
                allowed_special={self.separator_token}
            )[:self.maxlen]
        except Exception:
            # Fallback for texts without special tokens
            encoding = self.tokenizer.encode(text)[:self.maxlen]
        
        # Pad to maxlen
        return encoding + [0] * (self.maxlen - len(encoding))

def load_and_preprocess_data_enhanced(
    dataset_name: str = CURRENT_DATASET,
    batch_size: int = batch_size, 
    maxlen: int = maxlen,
    custom_config: Optional[DatasetConfig] = None
) -> pygrain.DataLoader:
    """
    Enhanced data loading function that supports multiple datasets
    
    Args:
        dataset_name: Name of the dataset configuration to use
        batch_size: Batch size for data loading
        maxlen: Maximum sequence length
        custom_config: Custom dataset configuration (overrides dataset_name)
    
    Returns:
        pygrain.DataLoader: Configured data loader
    """
    
    print(f"=== Dataset Loading ===")
    print(f"Requested dataset: {dataset_name}")
    
    # Use custom config if provided, otherwise get predefined config
    if custom_config:
        config = custom_config
        print("Using custom dataset configuration")
    elif dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        print(f"Using predefined configuration for {dataset_name}")
    else:
        print(f"Dataset '{dataset_name}' not found. Available datasets:")
        for name in list_available_datasets():
            info = get_dataset_info(name)
            print(f"  - {name}: {info.get('name', 'N/A')}")
        
        # Fallback to local file
        print("Falling back to local file (TinyStories)")
        config = DATASET_CONFIGS["local_file"]
    
    # Load the dataset
    texts = []
    
    if config.name == "local" or config.file_path:
        # Load from local file
        texts = load_local_file(config)
    else:
        # Try to load from Hugging Face
        if HF_DATASETS_AVAILABLE:
            texts = load_huggingface_dataset(config)
            
        # If HF loading failed or not available, try local fallback
        if not texts and dataset_name != "local_file":
            print("Attempting to load from local TinyStories file as fallback...")
            fallback_config = DATASET_CONFIGS["local_file"]
            texts = load_local_file(fallback_config)
    
    if not texts:
        raise RuntimeError(f"Failed to load any data for dataset: {dataset_name}")
    
    # Create enhanced dataset
    dataset = EnhancedTextDataset(
        data=texts, 
        maxlen=maxlen,
        separator_token=config.separator
    )
    
    print(f"Dataset created successfully:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Max sequence length: {maxlen}")
    print(f"  - Separator token: {config.separator}")
    
    # Create sampler and data loader
    sampler = pygrain.IndexSampler(
        len(dataset),
        shuffle=True,  # Enable shuffling for better training
        seed=42,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )

    dl = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
    )
    
    print(f"Data loader created with batch size: {batch_size}")
    print("=== Dataset Loading Complete ===\n")
    
    return dl

def switch_dataset(new_dataset: str) -> pygrain.DataLoader:
    """
    Utility function to quickly switch datasets
    
    Args:
        new_dataset: Name of the new dataset to load
        
    Returns:
        pygrain.DataLoader: New data loader
    """
    global CURRENT_DATASET
    CURRENT_DATASET = new_dataset
    return load_and_preprocess_data_enhanced(new_dataset, batch_size, maxlen)

def create_custom_dataset(
    name: str,
    subset: Optional[str] = None, 
    text_column: str = "text",
    separator: str = "<|endoftext|>",
    streaming: bool = False,
    min_length: int = 10
) -> pygrain.DataLoader:
    """
    Create a data loader for a custom Hugging Face dataset
    
    Args:
        name: Hugging Face dataset name (e.g., "username/dataset_name")
        subset: Dataset subset/configuration name
        text_column: Name of the text column in the dataset
        separator: Token to separate documents
        streaming: Whether to use streaming (for large datasets)
        min_length: Minimum text length to include
        
    Returns:
        pygrain.DataLoader: Configured data loader for the custom dataset
    """
    config = DatasetConfig(
        name=name,
        subset=subset,
        text_column=text_column,
        separator=separator,
        streaming=streaming,
        min_length=min_length
    )
    
    return load_and_preprocess_data_enhanced(
        dataset_name="custom",
        custom_config=config
    )

# Load the default dataset
print("Loading default dataset...")
text_dl = load_and_preprocess_data_enhanced()

# Print available datasets for reference
print(f"Available predefined datasets: {list_available_datasets()}")
print(f"Current dataset: {CURRENT_DATASET}")

"""## Defining the loss function and training step function"""

# Defines the loss function using `optax.softmax_cross_entropy_with_integer_labels`.
def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
    return loss, logits

# Define the training step with the `flax.nnx.jit` transformation decorator.
@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, lables=batch[1])
    optimizer.update(grads)

model = create_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))
metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)
rng = jax.random.PRNGKey(0)

start_prompt = "Once upon a time, "
start_tokens = tokenizer.encode(start_prompt)[:maxlen]
generated_text = model.generate_text(
    maxlen, start_tokens
)
print(f"Initial generated text:\n{generated_text}\n")


metrics_history = {
  'train_loss': [],
}

prep_target_batch = jax.vmap(lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0]))))

step = 0
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in text_dl:
        if len(batch) % len(jax.devices()) != 0:
          continue  # skip the remaining elements
        input_batch = jnp.array(jnp.array(batch).T)
        target_batch = prep_target_batch(input_batch)
        train_step(model, optimizer, metrics, jax.device_put((input_batch, target_batch), NamedSharding(mesh, P('batch', None))))

        if (step + 1) % 200 == 0:
          for metric, value in metrics.compute().items():
              metrics_history[f'train_{metric}'].append(value)
          metrics.reset()

          elapsed_time = time.time() - start_time
          print(f"Step {step + 1}, Loss: {metrics_history['train_loss'][-1]}, Elapsed Time: {elapsed_time:.2f} seconds")
          start_time = time.time()

          generated_text = model.generate_text(
              maxlen, start_tokens
          )
          print(f"Generated text:\n{generated_text}\n")
        step += 1

# Final text generation
generated_text = model.generate_text(
    maxlen, start_tokens
)
print(f"Final generated text:\n{generated_text}")

"""Visualize the training loss."""

import matplotlib.pyplot as plt
plt.plot(metrics_history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
