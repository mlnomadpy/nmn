"""GPT-2 Training Script with Spherical YAT Performer Attention and YatNMN MLP.

This script implements a GPT-2 style autoregressive language model using:
- Spherical YAT Performer attention for O(n) complexity
- YatNMN (spherical E-product MLP) for feed-forward layers
- RMSNorm for normalization
- Pre-norm architecture
- Learned positional embeddings

The model uses the YAT (Yet Another Transformer) attention formula:
    ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

With Multi-Scale FAVOR+ features for linear-time attention approximation.
"""

import os
import time
from dataclasses import dataclass
from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as orbax
import wandb
from datasets import load_dataset
from tokenizers import Tokenizer

from nmn.nnx.nmn import YatNMN
from nmn.nnx.attention.spherical_yat_performer import (
    create_yat_tp_projection,
    yat_tp_attention,
)




# --- Model Configuration ---
@dataclass
class GPT2Config:
    """GPT-2 model configuration."""
    vocab_size: int = 50257  # GPT-2 BPE vocab size
    maxlen: int = 1024       # Maximum sequence length
    vocab_embed_dim: int = 128 # Factorized embedding dimension (ALBERT-style)
    embed_dim: int = 768     # Hidden dimension (d_model)
    num_heads: int = 12      # Number of attention heads
    num_layers: int = 12     # Number of transformer blocks
    ff_dim: int = 768       # Feed-forward intermediate dimension (4 * embed_dim)
    dropout_rate: float = 0.1
    
    # YAT Performer settings
    # YAT Performer settings
    num_features_per_scale: int = 8    # Features per scale (Tiny: 8*2=16 exp features)
    num_scales: int = 2                # Number of scales
    poly_sketch_dim: int = 8           # Dimension for Polynomial TensorSketch (Total feats: 8*16=128)
    
    # Training settings
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100_000
    warmup_steps: int = 2000
    eval_interval: int = 500
    checkpoint_interval: int = 5000


# --- Model Components ---

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization (LLaMA-style)."""
    
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs = None):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim))

    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + self.eps) * self.weight


class YatPerformerAttention(nnx.Module):
    """Multi-head YAT Attention with Spherical FAVOR+ Performer approximation.
    
    Uses the Multi-Scale FAVOR+ approximation for the YAT kernel:
        K(x,y) = (x·y)² / (C - 2x·y)
    
    This provides O(n) linear complexity instead of O(n²) quadratic.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        num_features_per_scale: int = 64,
        num_scales: int = 8,
        poly_sketch_dim: int = 64,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        kernel_init=None,
        causal: bool = True,
        epsilon: float = 1e-5,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features_per_scale = num_features_per_scale
        self.num_scales = num_scales
        self.poly_sketch_dim = poly_sketch_dim
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.epsilon = epsilon
        self.rngs = rngs
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        if kernel_init is None:
            kernel_init = nnx.initializers.xavier_uniform()
        
        # Q, K, V projections
        self.q_proj = nnx.Linear(
            embed_dim, embed_dim, use_bias=use_bias, kernel_init=kernel_init, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            embed_dim, embed_dim, use_bias=use_bias, kernel_init=kernel_init, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            embed_dim, embed_dim, use_bias=use_bias, kernel_init=kernel_init, rngs=rngs
        )
        
        # Output projection
        self.o_proj = nnx.Linear(
            embed_dim, embed_dim, use_bias=use_bias, kernel_init=kernel_init, rngs=rngs
        )
        
        # Create YAT Performer projection parameters
        _performer_params = create_yat_tp_projection(
            rngs.params(),
            self.head_dim,
            num_prf_features=num_features_per_scale,
            num_quad_nodes=num_scales,
            epsilon=epsilon,
            poly_sketch_dim=poly_sketch_dim,
        )
        # Store as nnx.Cache for proper state handling (don't store the dict)
        self.projections = nnx.Cache(_performer_params['projections'])
        self.scales = nnx.Cache(_performer_params['scales'])
        self.sketch_h = nnx.Cache(_performer_params['sketch_params']['h'])
        self.sketch_s = nnx.Cache(_performer_params['sketch_params']['s'])
        
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        """Apply YAT Performer attention.
        
        Args:
            x: Input of shape [batch, seq_len, embed_dim]
            deterministic: If True, no dropout
            
        Returns:
            Output of shape [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Build performer params dict
        params = {
            'projections': jax.device_put(self.projections.value),
            'scales': jax.device_put(self.scales.value),
            'head_dim': self.head_dim,
            'num_prf_features': self.num_features_per_scale,
            'num_scales': self.num_scales,
            'sketch_params': {'h': self.sketch_h.value, 's': self.sketch_s.value},
            'poly_sketch_dim': self.poly_sketch_dim,
        }
        
        # Apply YAT Performer attention
        output = yat_tp_attention(
            q, k, v,
            params,
            causal=self.causal,
            epsilon=self.epsilon,
        )
        
        # Reshape back: [batch, seq, num_heads, head_dim] -> [batch, seq, embed_dim]
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection with dropout
        output = self.o_proj(output)
        output = self.dropout(output, deterministic=deterministic)
        
        return output


class GPT2Block(nnx.Module):
    """GPT-2 Transformer Block with YAT Performer Attention and YatNMN MLP.
    
    Architecture (Pre-Norm):
        x = x + Attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.1,
        num_features_per_scale: int = 64,
        num_scales: int = 8,
        poly_sketch_dim: int = 64,
        causal: bool = True,
    ):
        kernel_init = nnx.initializers.xavier_uniform()

        self.rngs = rngs
        
        # YAT Performer Attention (Spherical Multi-Scale FAVOR+)
        self.attn = YatPerformerAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rngs=rngs,
            num_features_per_scale=num_features_per_scale,
            num_scales=num_scales,
            poly_sketch_dim=poly_sketch_dim,
            dropout_rate=dropout_rate,
            use_bias=False,
            kernel_init=kernel_init,
            causal=causal,
        )
        
        # YatNMN MLP (spherical E-product based)
        # GPT-2 style: embed_dim -> 4*embed_dim -> embed_dim
        self.ffn = nnx.Sequential(
            YatNMN(embed_dim, ff_dim, kernel_init=kernel_init, rngs=rngs, use_bias=False),
            nnx.Linear(ff_dim, embed_dim, kernel_init=kernel_init, rngs=rngs, use_bias=False),
        )
        self.ffn_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    @nnx.remat(static_argnums=2)
    def __call__(self, x, training: bool = False):
        # Pre-Norm Attention Block
        h = x
        h = self.attn(h, deterministic=not training)
        x = x + h
        
        # Pre-Norm MLP Block
        h = x
        h = self.ffn(h)
        h = self.ffn_dropout(h, deterministic=not training)
        x = x + h
        
        return x


class GPT2(nnx.Module):
    """GPT-2 Language Model with Spherical YAT Performer Attention.
    
    Features:
    - Token embeddings + learned positional embeddings
    - ALBERT-style weight sharing: single transformer block used across all layers
    - YAT Performer attention and YatNMN MLP
    - Weight-tied output head
    
    ALBERT-style sharing reduces parameters dramatically by reusing the same
    transformer block weights across all layers (from ~124M to ~12M params).
    """
    
    def __init__(self, config: GPT2Config, *, rngs: nnx.Rngs):
        self.config = config
        
        # Token embeddings
        # Token embeddings (Factorized: vocab_size -> vocab_embed_dim)
        self.token_emb = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.vocab_embed_dim,
            rngs=rngs,
        )
        
        # Learned positional embeddings (maxlen -> vocab_embed_dim)
        self.pos_emb = nnx.Embed(
            num_embeddings=config.maxlen,
            features=config.vocab_embed_dim,
            rngs=rngs,
        )
        
        # Factorized embedding projection: vocab_embed_dim -> embed_dim
        # ALBERT projects E -> H before the transformer layers
        self.input_proj = nnx.Linear(
            config.vocab_embed_dim,
            config.embed_dim,
            rngs=rngs,
        )
        
        # Output projection: embed_dim -> vocab_embed_dim 
        # Project back to embedding size before reusing embedding matrix
        self.output_proj = nnx.Linear(
            config.embed_dim,
            config.vocab_embed_dim,
            rngs=rngs,
        )
        
        self.embed_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        
        # ALBERT-style: Single shared transformer block (called num_layers times)
        # This reduces parameters by ~10x compared to separate blocks
        self.shared_block = GPT2Block(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            rngs=rngs,
            dropout_rate=config.dropout_rate,
            num_features_per_scale=config.num_features_per_scale,
            num_scales=config.num_scales,
            causal=True,
        )
        
        # Store head_dim for reference
        self.head_dim = config.embed_dim // config.num_heads

    def __call__(self, input_ids, training: bool = False):
        """Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs of shape [batch, seq_len]
            training: Whether to apply dropout
            
        Returns:
            Logits of shape [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeddings = self.token_emb(input_ids)
        
        # Positional embeddings
        positions = jnp.arange(seq_len)
        pos_embeddings = self.pos_emb(positions)
        
        # Combine embeddings
        x = token_embeddings + pos_embeddings
        x = self.embed_dropout(x, deterministic=not training)
        
        # Project to hidden size (E -> H)
        x = self.input_proj(x)
        
        # Apply shared transformer block num_layers times (ALBERT-style)
        for _ in range(self.config.num_layers):
            x = self.shared_block(x, training)
        
        # Project back to embedding size (H -> E)
        x = self.output_proj(x)
        
        # Weight-tied output projection: reuse token embedding weights
        embedding_weights = self.token_emb.embedding.value
        logits = x @ embedding_weights.T
        
        return logits


def create_model(rngs: nnx.Rngs, config: GPT2Config) -> GPT2:
    """Create a GPT-2 model with the given configuration."""
    return GPT2(config, rngs=rngs)


# --- Data Preprocessing ---

def create_data_iterator(
    dataset,
    tokenizer: Tokenizer,
    maxlen: int,
    batch_size: int,
):
    """Create an iterator that yields batches of tokenized text for language modeling."""
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        pad_token_id = 0
    
    def tokenize_batch(examples):
        input_ids = []
        labels = []
        
        for text in examples["text"]:
            # Tokenize and truncate/pad to maxlen + 1 for shifted labels
            encoded = tokenizer.encode(text)
            tokens = encoded.ids[: maxlen + 1]
            
            # Pad if needed
            if len(tokens) < maxlen + 1:
                tokens = tokens + [pad_token_id] * (maxlen + 1 - len(tokens))
            
            # Input is tokens[:-1], labels is tokens[1:]
            input_ids.append(tokens[:-1])
            labels.append(tokens[1:])
        
        return {"input_ids": input_ids, "labels": labels}
    
    # Process dataset
    columns_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "labels"]]
    processed = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        remove_columns=columns_to_remove,
    )
    processed = processed.shuffle(buffer_size=10_000, seed=42)
    
    return iter(processed.iter(batch_size=batch_size, drop_last_batch=True))


# --- Training Functions ---

def loss_fn(model: GPT2, batch, training: bool):
    """Compute cross-entropy loss for language modeling."""
    logits = model(batch["input_ids"], training=training)
    labels = batch["labels"]
    
    # Flatten for loss computation
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    
    # Cross-entropy loss (ignore padding tokens with label -100)
    loss_per_token = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_flat, labels=labels_flat
    )
    
    # Mask out padding
    mask = labels_flat != -100
    num_valid = jnp.sum(mask)
    loss = jnp.where(num_valid > 0, jnp.sum(loss_per_token * mask) / num_valid, 0.0)
    
    return loss, logits


@nnx.jit
def train_step(model: GPT2, optimizer: nnx.Optimizer, batch):
    """Single training step."""
    grad_fn = nnx.value_and_grad(lambda m, b: loss_fn(m, b, training=True), has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    optimizer.update(model, grads)  # Flax 0.11.0+ requires (model, grads)
    return loss


@nnx.jit
def eval_step(model: GPT2, batch):
    """Single evaluation step."""
    loss, _ = loss_fn(model, batch, training=False)
    return loss


# --- Learning Rate Schedule ---

def create_lr_schedule(config: GPT2Config):
    """Create learning rate schedule with warmup and cosine decay."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.max_steps - config.warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )
    return schedule_fn


# --- Main Training Loop ---

def main():
    """Main training function."""
    # Configuration
    config = GPT2Config(
        num_layers=12,
        maxlen=1024,
        embed_dim=768,
        num_heads=12,
        ff_dim=3072,
        batch_size=16,
        learning_rate=3e-4,
        max_steps=100_000,
        warmup_steps=2000,
        eval_interval=500,
        checkpoint_interval=5000,
        num_features_per_scale=64,
        num_scales=8,
    )
    
    checkpoint_dir = os.path.abspath("./gpt2_yat_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="gpt2-yat-performer",
        config=config.__dict__,
        name="gpt2-yat-performer-training",
    )
    
    # Initialize RNGs
    rngs = nnx.Rngs(0)
    
    # Load tokenizer (GPT-2 style)
    print("\n=== Loading Tokenizer ===")
    tokenizer = Tokenizer.from_pretrained("gpt2")
    tokenizer.enable_truncation(max_length=config.maxlen)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab Size: {config.vocab_size}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_model(rngs, config)
    
    # Count parameters
    _, param_state, _ = nnx.split(model, nnx.Param, ...)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(param_state))
    print(f"Model Parameters: {num_params:,}")
    
    # Create optimizer with weight decay and learning rate schedule
    lr_schedule = create_lr_schedule(config)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay),
        ),
        wrt=nnx.Param,  # Specify which parameters to optimize (required in Flax 0.11.0+)
    )
    
    # Load dataset (streaming for large datasets)
    print("\n=== Loading Dataset ===")
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    
    # Small validation set
    val_size = 1000
    val_dataset = dataset.take(val_size)
    train_dataset = dataset.skip(val_size)
    
    train_iterator = create_data_iterator(train_dataset, tokenizer, config.maxlen, config.batch_size)
    val_iterator = create_data_iterator(val_dataset, tokenizer, config.maxlen, config.batch_size)
    
    # Training loop
    print("\n=== Starting Training ===")
    start_time = time.time()
    
    for step in range(config.max_steps):
        # Get batch
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = create_data_iterator(train_dataset, tokenizer, config.maxlen, config.batch_size)
            batch = next(train_iterator)
        
        # Put batch on device
        batch = {k: jax.device_put(jnp.array(v)) for k, v in batch.items()}
        
        # Training step
        loss = train_step(model, optimizer, batch)
        
        # Logging
        wandb.log({"train/loss": loss.item(), "train/step": step}, step=step)
        
        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            # Compute validation loss
            try:
                val_batch = next(val_iterator)
            except StopIteration:
                val_iterator = create_data_iterator(val_dataset, tokenizer, config.maxlen, config.batch_size)
                val_batch = next(val_iterator)
            
            val_batch = {k: jax.device_put(jnp.array(v)) for k, v in val_batch.items()}
            val_loss = eval_step(model, val_batch)
            
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * config.batch_size * config.maxlen / elapsed
            
            print(
                f"Step {step + 1}/{config.max_steps} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Tokens/sec: {tokens_per_sec:.0f}"
            )
            wandb.log({
                "val/loss": val_loss.item(),
                "train/tokens_per_sec": tokens_per_sec,
            }, step=step)
        
        # Checkpointing
        if (step + 1) % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step + 1}")
            checkpointer = orbax.PyTreeCheckpointer()
            _, param_state, _ = nnx.split(model, nnx.Param, ...)
            checkpointer.save(ckpt_path, item=param_state)
            checkpointer.close()
            print(f"Checkpoint saved to {ckpt_path}")
    
    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, "final")
    checkpointer = orbax.PyTreeCheckpointer()
    _, param_state, _ = nnx.split(model, nnx.Param, ...)
    checkpointer.save(final_path, item=param_state)
    checkpointer.close()
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
