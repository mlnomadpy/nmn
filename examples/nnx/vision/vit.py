import os
import sys
import argparse
import warnings
from typing import Any, List, Optional, Tuple, Callable
from functools import partial

# JAX / Flax / Optax / Sharding
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx
import optax

# Checkpointing & Logging
import orbax.checkpoint as ocp
import wandb

# Data Loading
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from torchvision import transforms
from PIL import Image, PngImagePlugin
from tqdm import tqdm
import numpy as np

# Fix PIL PNG decompression limit
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------------------------------------------------
# 1. Neural Network Blocks (ViT - NNX)
# -----------------------------------------------------------------------------

class PatchEmbedding(nnx.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, rngs: nnx.Rngs):
        self.proj = nnx.Conv(in_channels, embed_dim, kernel_size=(patch_size, patch_size),
                             strides=(patch_size, patch_size), padding='VALID', rngs=rngs)
    
    def __call__(self, x):
        x = self.proj(x)
        # B, H, W, C -> B, (H*W), C
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        return x


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization (modern alternative to LayerNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6, rngs: nnx.Rngs = None):
        self.eps = eps
        self.scale = nnx.Param(jnp.ones(dim))
    
    def __call__(self, x):
        # RMSNorm: x * scale / sqrt(mean(x^2) + eps)
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale[...]


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding (RoPE) for Vision Transformers.
    
    Implements 2D RoPE by treating patch positions as (row, col) pairs
    and applying separate rotations for each spatial dimension.
    """
    def __init__(self, dim: int, max_seq_len: int = 256, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies for RoPE
        # Split dim in half for 2D (rows and cols)
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (jnp.arange(0, half_dim, 2).astype(jnp.float32) / half_dim))
        self.inv_freq = inv_freq
    
    def _get_rotary_emb(self, seq_len: int, grid_size: int):
        """Generate rotary embeddings for 2D grid positions."""
        # Create 2D position grid
        positions = jnp.arange(seq_len)
        rows = positions // grid_size
        cols = positions % grid_size
        
        # Compute sinusoidal embeddings for rows and cols
        # Each gets half the dimensions, then concatenated to form full dim
        half_dim = self.dim // 2
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, half_dim, 1).astype(jnp.float32) / half_dim))
        
        # Outer product: (seq_len,) x (half_dim,) -> (seq_len, half_dim)
        row_freqs = jnp.outer(rows, inv_freq)
        col_freqs = jnp.outer(cols, inv_freq)
        
        # Combine into full dimension embeddings
        # sin/cos for rows in first half, sin/cos for cols in second half
        sin_emb = jnp.concatenate([jnp.sin(row_freqs), jnp.sin(col_freqs)], axis=-1)
        cos_emb = jnp.concatenate([jnp.cos(row_freqs), jnp.cos(col_freqs)], axis=-1)
        
        return sin_emb, cos_emb
    
    def __call__(self, x, grid_size: int):
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (B, seq_len, num_heads, head_dim) or (B, num_heads, seq_len, head_dim)
            grid_size: Size of the spatial grid (e.g., 14 for 224/16 patches)
        
        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[1] if x.ndim == 4 else x.shape[2]
        sin_emb, cos_emb = self._get_rotary_emb(seq_len, grid_size)
        
        # Reshape for broadcasting: (1, seq_len, 1, head_dim)
        sin_emb = sin_emb[None, :, None, :]
        cos_emb = cos_emb[None, :, None, :]
        
        return x * cos_emb + self._rotate_half(x) * sin_emb
    
    def _rotate_half(self, x):
        """Rotate half the dimensions for RoPE."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)


class MultiHeadAttentionWithRoPE(nnx.Module):
    """Multi-Head Attention with Rotary Position Embeddings."""
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.0, 
                 grid_size: int = 14, rngs: nnx.Rngs = None):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        
        # Q, K, V projections
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=grid_size * grid_size + 1)
        
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    
    def __call__(self, x, deterministic: bool = True):
        B, N, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape: (B, N, num_heads, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K (skip CLS token at position 0)
        # For CLS token, we don't apply RoPE
        q_cls, q_patches = q[:, :1, :, :], q[:, 1:, :, :]
        k_cls, k_patches = k[:, :1, :, :], k[:, 1:, :, :]
        
        q_patches = self.rope(q_patches, self.grid_size)
        k_patches = self.rope(k_patches, self.grid_size)
        
        q = jnp.concatenate([q_cls, q_patches], axis=1)
        k = jnp.concatenate([k_cls, k_patches], axis=1)
        
        # Transpose for attention: (B, num_heads, N, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = nnx.softmax(attn, axis=-1)
        attn = self.dropout(attn, deterministic=deterministic)
        
        # Output
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

class MLP(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout_rate: float, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, in_features, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    
    def __call__(self, x, training: bool = True):
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.dropout(x, deterministic=not training)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=not training)
        return x

class Block(nnx.Module):
    """Transformer block with RMSNorm and RoPE attention."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout_rate: float = 0.0, attention_dropout: float = 0.0,
                 grid_size: int = 14, rngs: nnx.Rngs = None):
        self.norm1 = RMSNorm(dim, rngs=rngs)
        self.attn = MultiHeadAttentionWithRoPE(
            dim=dim,
            num_heads=num_heads,
            dropout_rate=attention_dropout,
            grid_size=grid_size,
            rngs=rngs
        )
        self.norm2 = RMSNorm(dim, rngs=rngs)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                       dropout_rate=dropout_rate, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, training: bool = True):
        # Attention Block (Pre-Norm)
        residual = x
        x = self.norm1(x)
        x = self.attn(x, deterministic=not training)
        x = self.dropout(x, deterministic=not training)
        x = residual + x
        
        # MLP Block (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x, training=training)
        x = residual + x
        return x



class VisionTransformer(nnx.Module):
    """Vision Transformer with RoPE and RMSNorm (modern BERT-style)."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 num_classes: int = 1000, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, dropout_rate: float = 0.1,
                 dtype=jnp.float32, rngs: nnx.Rngs = None):
        
        self.dtype = dtype
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(in_chans, embed_dim, patch_size, rngs=rngs)
        
        # 2. Class Token (no positional embedding - RoPE handles positions)
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, embed_dim)) * 0.02)
        
        self.pos_drop = nnx.Dropout(dropout_rate, rngs=rngs)

        # 3. Transformer Encoder Blocks with RoPE
        layers = []
        for _ in range(depth):
            layers.append(Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                grid_size=self.grid_size,
                rngs=rngs
            ))
        self.blocks = nnx.List(layers)

        # 4. Final RMSNorm
        self.norm = RMSNorm(embed_dim, rngs=rngs)

        # 5. Classification Head
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs)

    def __call__(self, x, training: bool = True, return_features: bool = False):
        x = x.astype(self.dtype)
        
        # Patch Embed
        x = self.patch_embed(x)  # (B, N, D)
        b = x.shape[0]

        # Add CLS token
        cls_token_arr = self.cls_token[...]
        cls_token = jnp.broadcast_to(cls_token_arr, (b, 1, cls_token_arr.shape[-1]))
        x = jnp.concatenate((cls_token, x), axis=1)  # (B, N+1, D)
        
        # No positional embedding added - RoPE handles position in attention
        x = self.pos_drop(x, deterministic=not training)

        # Apply Blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final Norm
        x = self.norm(x)

        # Extract CLS token output
        x = x[:, 0]

        if return_features:
            return x

        x = self.head(x)
        return x
        
# -----------------------------------------------------------------------------
# 2. Loss Functions
# -----------------------------------------------------------------------------

def custom_softmax_loss(logits, targets, power=1.0):
    y_pred = nnx.softmax(logits, axis=1)
    y_true = jax.nn.one_hot(targets, num_classes=logits.shape[1])
    loss_per_sample = jnp.sum(y_true + y_pred - 2 * y_true * y_pred, axis=1)
    if power != 1.0:
        loss_per_sample = jnp.power(loss_per_sample, power)
    return jnp.mean(loss_per_sample)



# -----------------------------------------------------------------------------
# 3. Data Loading
# -----------------------------------------------------------------------------

class ImageNetStreamDataset(IterableDataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset('mlnomad/imagenet-1k-224', split=split, streaming=True)
        self.transform = transform

    def __iter__(self):
        for sample in self.dataset:
            try:
                image = sample['image']
                label = sample['label']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                yield image, label
            except Exception:
                continue

def get_transforms(is_train=True):
    # ViT typically uses Resize 256 -> Crop 224
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_numpy_batch(batch):
    images, labels = batch
    # PyTorch NCHW -> JAX NHWC
    images_np = images.numpy().transpose(0, 2, 3, 1)
    labels_np = labels.numpy()
    return images_np, labels_np

# -----------------------------------------------------------------------------
# 4. Training Step (NNX + Mesh)
# -----------------------------------------------------------------------------

@nnx.jit(static_argnames=('loss_type', 'loss_power'))
def train_step(model, optimizer, batch_images, batch_labels, loss_type, loss_power):
    
    def loss_fn(model):
        outputs = model(batch_images, training=True)
        
        if loss_type == 'custom':
            loss = custom_softmax_loss(outputs, batch_labels, power=loss_power)
            acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
            return loss, acc
        else:  # CrossEntropy (default)
            loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
            loss = loss.astype(jnp.float32)
            acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
            return loss, acc

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(model)
    
    optimizer.update(model, grads)
    
    return loss, acc

@nnx.jit(static_argnames=('loss_type', 'loss_power'))
def val_step(model, batch_images, batch_labels, loss_type, loss_power):
    outputs = model(batch_images, training=False)
    
    if loss_type == 'custom':
        loss = custom_softmax_loss(outputs, batch_labels, power=loss_power)
    else:  # CrossEntropy (default)
        loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
        
    acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
    return loss, acc

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Model Args
    parser.add_argument('--model', type=str, default='vit_small', choices=['vit_tiny', 'vit_small', 'vit_base'])
    parser.add_argument('--batch-size', type=int, default=512, help="Global batch size across all devices")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001, help="Lower LR recommended for ViT (e.g. 1e-3 or 3e-4)")
    parser.add_argument('--loss-type', type=str, default='crossentropy', choices=['custom', 'crossentropy'])
    parser.add_argument('--mixed-precision', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Checkpointing & Logging Args
    parser.add_argument('--save-dir', type=str, default='./checkpoints_flax_vit')
    parser.add_argument('--checkpoint-keep', type=int, default=3, help='Number of checkpoints to keep')
    parser.add_argument('--wandb-project', type=str, default="imagenet-flax", help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default="irf-sic", help='WandB entity/username')
    parser.add_argument('--wandb-name', type=str, default=None, help='WandB run name')
    
    parser.add_argument('-f', '--file', type=str, default='', help='Jupyter kernel file (ignored)')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    # 1. Initialize WandB
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )
        print(f"WandB initialized: {args.wandb_project}")

    # 2. Setup Orbax
    ckpt_dir = os.path.abspath(args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    options = ocp.CheckpointManagerOptions(max_to_keep=args.checkpoint_keep, create=True)
    mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options)
    print(f"Orbax Checkpoint Manager initialized at {ckpt_dir}")

    # -------------------------------------------------------------------------
    # MESH SETUP
    # -------------------------------------------------------------------------
    devices = jax.devices()
    num_devices = len(devices)
    print(f"JAX Devices: {num_devices} ({devices[0].platform})")

    # Define Mesh: Pure Data Parallelism
    mesh = Mesh(np.array(devices), ('data',))
    data_sharding = NamedSharding(mesh, P('data', None, None, None)) 
    label_sharding = NamedSharding(mesh, P('data'))
    replicated_sharding = NamedSharding(mesh, P())

    # -------------------------------------------------------------------------
    # MODEL INIT
    # -------------------------------------------------------------------------
    rngs = nnx.Rngs(0)
    
    # ViT Configurations
    vit_configs = {
        'vit_tiny':  {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'vit_small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'vit_base':  {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    }
    config = vit_configs[args.model]
    dtype = jnp.bfloat16 if args.mixed_precision else jnp.float32

    with mesh:
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            dtype=dtype,
            rngs=rngs
        )
                
        IMAGENET_SIZE = 1_281_167
        steps_per_epoch = IMAGENET_SIZE // args.batch_size
        total_steps = steps_per_epoch * args.epochs
        
        # Warmup + Cosine Decay is critical for ViT stability
        warmup_steps = int(steps_per_epoch * 1) 
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.lr,
            warmup_steps=warmup_steps, 
            decay_steps=total_steps,
        )        
        # AdamW is standard for ViT (Weight decay 0.05-0.1 usually)
        optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=schedule, weight_decay=0.05), wrt=nnx.Param)

        # Replicate state across mesh
        state = nnx.state((model, optimizer))
        
        def shard_leaf(leaf):
            return jax.device_put(leaf, replicated_sharding)
        sharded_state = jax.tree_util.tree_map(shard_leaf, state)
        
        nnx.update((model, optimizer), sharded_state)

    print(f"Model {args.model} initialized and replicated.")

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    train_dataset = ImageNetStreamDataset(split='train', transform=get_transforms(True))
    val_dataset = ImageNetStreamDataset(split='validation', transform=get_transforms(False))
    
    if args.batch_size % num_devices != 0:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by device count {num_devices}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_metrics = [] # Acc or SimO2 aux
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            global_step += 1
            imgs_np, lbls_np = get_numpy_batch(batch)
            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)
            
            loss, acc = train_step(model, optimizer, imgs_sharded, lbls_sharded, 
                             args.loss_type, 1.0)
            
            loss_val = float(loss)
            acc_val = float(acc)
            epoch_losses.append(loss_val)
            epoch_metrics.append(acc_val)
            
            if args.wandb_project:
                wandb.log({
                    'train/iter_loss': loss_val,
                    'train/iter_acc': acc_val,
                    'trainer/global_step': global_step,
                    'epoch': epoch
                })

            pbar.set_postfix({'acc': float(np.mean(epoch_metrics[-10:]))})

        avg_train_loss = np.mean(epoch_losses)
        avg_train_metric = np.mean(epoch_metrics)

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        model.eval()
        total_acc = []
        total_loss = []
        for batch in tqdm(val_loader, desc='Val'):
            imgs_np, lbls_np = get_numpy_batch(batch)
            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)
            
            loss, acc = val_step(model, imgs_sharded, lbls_sharded, args.loss_type, 1.0)
            total_acc.append(acc)
            total_loss.append(loss)
        
        val_acc = np.mean(total_acc) * 100
        val_loss = np.mean(total_loss)
        print(f"Epoch {epoch} Val Acc: {val_acc:.2f}%")
        
        # ---------------------------------------------------------------------
        # LOGGING (WandB)
        # ---------------------------------------------------------------------
        if args.wandb_project:
            wandb.log({
                'epoch': epoch,
                'train/loss': avg_train_loss,
                'train/acc': avg_train_metric,
                'val/loss': val_loss,
                'val/acc': val_acc
            })

        # ---------------------------------------------------------------------
        # CHECKPOINTING (Orbax)
        # ---------------------------------------------------------------------
        should_save = False
        if val_acc > best_acc:
            best_acc = val_acc
            should_save = True
            print(f"New best model: {best_acc:.2f}%")

        if should_save:
            raw_state = nnx.state((model, optimizer))
            save_args = ocp.args.StandardSave(raw_state)
            mngr.save(step=epoch, args=save_args)
            mngr.wait_until_finished()
            print(f"Checkpoint saved for epoch {epoch}")

    if args.wandb_project:
        wandb.finish()

if __name__ == '__main__':
    main()