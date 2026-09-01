import argparse
import os
import sys
import warnings
from typing import List, Optional

# JAX / Flax / Optax / Sharding
import jax
import jax.numpy as jnp
import optax

# Checkpointing & Logging
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Data Loading
try:
    import grain.python as grain

    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False
    from PIL import PngImagePlugin
    from torch.utils.data import DataLoader, Dataset, IterableDataset
    from torchvision import transforms

    PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from nmn.nnx.layers import YatNMN
from nmn.nnx.layers.conv import YatConv

# Training Monitor
try:
    from tpu_monitor import TrainingMonitor

    MONITOR_AVAILABLE = True
except ImportError:
    try:
        import importlib.util
        import pathlib

        _mon_path = str(pathlib.Path(__file__).parent / "tpu_monitor.py")
        _spec = importlib.util.spec_from_file_location("tpu_monitor", _mon_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Could not load training monitor from {_mon_path}")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        TrainingMonitor = _mod.TrainingMonitor
        MONITOR_AVAILABLE = True
    except Exception:
        MONITOR_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


def _is_notebook_frontend() -> bool:
    """Return True when running inside an interactive notebook frontend."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return bool(ip.__class__.__name__ == "ZMQInteractiveShell")
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 1. Neural Network Blocks (NNX)
# -----------------------------------------------------------------------------


class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nnx.Module] = None,
        dtype=jnp.float32,
        tie_yat_kernels: bool = False,
        yat_kernel_bank_size: Optional[int] = None,
        yat_kernel_bank_id: str = "resnet-yat",
        rngs: nnx.Rngs = None,
    ):
        self.conv1 = YatConv(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            constant_alpha=True,
            tie_kernel_bank=tie_yat_kernels,
            kernel_bank_size=yat_kernel_bank_size,
            kernel_bank_id=yat_kernel_bank_id,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        return out


class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nnx.Module] = None,
        dtype=jnp.float32,
        tie_yat_kernels: bool = False,
        yat_kernel_bank_size: Optional[int] = None,
        yat_kernel_bank_id: str = "resnet-yat",
        rngs: nnx.Rngs = None,
    ):
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv2 = YatConv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            constant_alpha=True,
            tie_kernel_bank=tie_yat_kernels,
            kernel_bank_size=yat_kernel_bank_size,
            kernel_bank_id=yat_kernel_bank_id,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            out_channels,
            out_channels * self.expansion,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        return out


class DownsampleBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, stride, dtype, rngs):
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            strides=stride,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x, training=True):
        x = self.conv(x)
        return x


class ResNet(nnx.Module):
    def __init__(
        self,
        block_cls,
        layers: List[int],
        num_classes: int = 1000,
        dtype=jnp.float32,
        tie_yat_kernels: bool = False,
        yat_kernel_bank_size: Optional[int] = None,
        yat_kernel_bank_id: str = "resnet-yat",
        use_yat_head: bool = True,
        tie_yat_nmn_kernels: bool = False,
        yat_nmn_kernel_bank_size: Optional[int] = None,
        yat_nmn_kernel_bank_id: str = "resnet-yat-head",
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = 64
        self.dtype = dtype
        self.tie_yat_kernels = tie_yat_kernels
        self.yat_kernel_bank_size = yat_kernel_bank_size
        self.yat_kernel_bank_id = yat_kernel_bank_id
        self.use_yat_head = use_yat_head
        self.tie_yat_nmn_kernels = tie_yat_nmn_kernels
        self.yat_nmn_kernel_bank_size = yat_nmn_kernel_bank_size
        self.yat_nmn_kernel_bank_id = yat_nmn_kernel_bank_id

        self.conv1 = nnx.Conv(
            3,
            64,
            kernel_size=(7, 7),
            strides=2,
            padding=3,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        # ResNet Layers
        self.layer1 = self._make_layer(block_cls, 64, layers[0], dtype=dtype, rngs=rngs)
        self.layer2 = self._make_layer(
            block_cls, 128, layers[1], stride=2, dtype=dtype, rngs=rngs
        )
        self.layer3 = self._make_layer(
            block_cls, 256, layers[2], stride=2, dtype=dtype, rngs=rngs
        )
        self.layer4 = self._make_layer(
            block_cls, 512, layers[3], stride=2, dtype=dtype, rngs=rngs
        )

        self.feature_dim = 512 * block_cls.expansion
        if self.use_yat_head:
            self.head = YatNMN(
                self.feature_dim,
                num_classes,
                use_bias=False,
                constant_alpha=True,
                tie_kernel_bank=self.tie_yat_nmn_kernels,
                kernel_bank_size=self.yat_nmn_kernel_bank_size,
                kernel_bank_id=self.yat_nmn_kernel_bank_id,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.head = nnx.Linear(
                self.feature_dim, num_classes, use_bias=False, dtype=dtype, rngs=rngs
            )

    def _make_layer(
        self, block_cls, out_channels, blocks, stride=1, dtype=jnp.float32, rngs=None
    ):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block_cls.expansion:
            downsample = DownsampleBlock(
                self.in_channels,
                out_channels * block_cls.expansion,
                stride,
                dtype,
                rngs,
            )

        layers = []
        layers.append(
            block_cls(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                dtype=dtype,
                tie_yat_kernels=self.tie_yat_kernels,
                yat_kernel_bank_size=self.yat_kernel_bank_size,
                yat_kernel_bank_id=self.yat_kernel_bank_id,
                rngs=rngs,
            )
        )
        self.in_channels = out_channels * block_cls.expansion
        for _ in range(1, blocks):
            layers.append(
                block_cls(
                    self.in_channels,
                    out_channels,
                    dtype=dtype,
                    tie_yat_kernels=self.tie_yat_kernels,
                    yat_kernel_bank_size=self.yat_kernel_bank_size,
                    yat_kernel_bank_id=self.yat_kernel_bank_id,
                    rngs=rngs,
                )
            )

        return nnx.List(layers)

    def __call__(self, x, training: bool = True, return_features: bool = False):
        x = x.astype(self.dtype)
        x = self.conv1(x)
        x = nnx.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, training=training)

        x = jnp.mean(x, axis=(1, 2))

        if return_features:
            return x

        x = self.head(x)
        return x


# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------

if GRAIN_AVAILABLE:
    # Grain-based data pipeline (optimized for TPU)
    class HFDataSource(grain.RandomAccessDataSource):
        """Grain data source wrapping HuggingFace dataset."""

        def __init__(self, split="train", cache_dir=None):
            self.dataset = load_dataset(
                "mlnomad/imagenet-1k-224",
                split=split,
                streaming=False,
                cache_dir=cache_dir,
            )

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            image = sample["image"]
            label = sample["label"]
            # Convert PIL to numpy (HWC format)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_np = np.array(image, dtype=np.uint8)
            return {"image": image_np, "label": label}

    class PreprocessOp(grain.MapTransform):
        """JAX-native preprocessing operations."""

        def __init__(self, is_train=True, image_size=224):
            self.is_train = is_train
            self.image_size = image_size
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def map(self, element):
            image = element["image"]
            label = element["label"]

            # Random crop/resize
            if self.is_train:
                # Simple random crop for now (could add more augmentations)
                h, w = image.shape[:2]
                if h > self.image_size or w > self.image_size:
                    # Center crop as fallback
                    top = (h - self.image_size) // 2
                    left = (w - self.image_size) // 2
                    image = image[
                        top : top + self.image_size, left : left + self.image_size
                    ]
            else:
                # Center crop
                h, w = image.shape[:2]
                top = (h - self.image_size) // 2
                left = (w - self.image_size) // 2
                image = image[
                    top : top + self.image_size, left : left + self.image_size
                ]

            # Normalize
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std

            return {"image": image, "label": label}

else:
    # Fallback to PyTorch DataLoader
    class ImageNetStreamDataset(IterableDataset):
        def __init__(self, split="train", transform=None):
            self.dataset = load_dataset(
                "mlnomad/imagenet-1k-224", split=split, streaming=True
            )
            self.transform = transform

        def __iter__(self):
            for sample in self.dataset:
                try:
                    image = sample["image"]
                    label = sample["label"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    yield image, label
                except Exception:
                    continue

    class ImageNetMapDataset(Dataset):
        def __init__(
            self, split="train", transform=None, cache_dir=None, keep_in_memory=False
        ):
            self.dataset = load_dataset(
                "mlnomad/imagenet-1k-224",
                split=split,
                streaming=False,
                cache_dir=cache_dir,
                keep_in_memory=keep_in_memory,
            )
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            try:
                sample = self.dataset[idx]
                image = sample["image"]
                label = sample["label"]
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                raise e

    def get_transforms(is_train=True):
        if is_train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def get_numpy_batch(batch):
        images, labels = batch
        # PyTorch NCHW -> JAX NHWC
        images_np = images.numpy().transpose(0, 2, 3, 1)
        labels_np = labels.numpy()
        return images_np, labels_np


def create_grain_dataloader(
    split, batch_size, is_train=True, cache_dir=None, num_epochs=None, num_devices=1
):
    """Create Grain DataLoader with proper prefetching."""
    if not GRAIN_AVAILABLE:
        raise RuntimeError(
            "FATAL: Grain is required but not available. Install with: pip install grain"
        )

    try:
        # Create data source
        data_source = HFDataSource(split=split, cache_dir=cache_dir)
        total_samples = len(data_source)

        # Create sampler - CRITICAL: num_epochs=1 means one pass per dataloader creation
        if is_train:
            sampler = grain.IndexSampler(
                total_samples,
                shuffle=True,
                seed=42 + (hash(split) % 1000 if split != "train" else 0),
                num_epochs=1,  # One pass per dataloader instance
                shard_options=grain.ShardByJaxProcess(drop_remainder=True),
            )
        else:
            sampler = grain.IndexSampler(
                total_samples,
                shuffle=False,
                num_epochs=1,  # One validation pass
                shard_options=grain.ShardByJaxProcess(drop_remainder=True),
            )

        # Create transformations (including batching)
        transformations = [
            PreprocessOp(is_train=is_train),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ]

        # Create DataLoader
        loader = grain.DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=transformations,
            worker_count=0,  # Use 0 for TPU to avoid multiprocessing overhead
            worker_buffer_size=2,
            read_options=grain.ReadOptions(
                num_threads=1,
                prefetch_buffer_size=2,
            ),
        )

        # Debug info
        # `batch_size` is per JAX process for Grain. On single-host TPU this equals
        # global batch size, so expected batches should be ~ total_samples // batch_size.
        num_jax_processes = jax.process_count()
        samples_per_process = total_samples // num_jax_processes
        expected_batches = samples_per_process // batch_size
        print(
            f"[DEBUG] Created Grain loader for {split}: {total_samples:,} samples, "
            f"batch_size={batch_size}, jax_processes={num_jax_processes}, "
            f"expected_batches≈{expected_batches}"
        )

        return loader
    except Exception as e:
        print(f"\n" + "!" * 80)
        print(f"FATAL: Failed to create Grain dataloader for split={split}")
        print(f"Error: {e}")
        print(f"!" * 80)
        raise


# -----------------------------------------------------------------------------
# 3. Training Step (NNX + Mesh)
# -----------------------------------------------------------------------------


@nnx.jit
def train_step(model, optimizer, batch_images, batch_labels):

    def loss_fn(model):
        outputs = model(batch_images, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            outputs, batch_labels
        ).mean()
        acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
        return loss, acc

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model)

    optimizer.update(model, grads)

    return loss, aux


@nnx.jit
def val_step(model, batch_images, batch_labels):
    outputs = model(batch_images, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
    acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
    return loss, acc


# -----------------------------------------------------------------------------
# 4. Model Analysis Helper
# -----------------------------------------------------------------------------


def analyze_model(model):
    """Analyze and log model statistics: total parameters, neuron types, and counts."""
    total_params = 0
    linear_neurons = 0
    yat_neurons = 0
    linear_params = 0
    yat_params = 0

    def traverse_module(module, prefix=""):
        nonlocal total_params, linear_neurons, yat_neurons, linear_params, yat_params

        # Check if this is a Linear layer
        if isinstance(module, nnx.Linear):
            out_features = module.out_features
            in_features = module.in_features
            params = in_features * out_features
            if module.bias is not None and hasattr(module, "bias"):
                params += out_features
            linear_neurons += out_features
            linear_params += params
            total_params += params

        # Check if this is a YatNMN layer
        elif isinstance(module, YatNMN):
            out_features = module.out_features
            in_features = module.in_features
            # YatNMN typically has similar parameter structure to Linear
            params = in_features * out_features
            if hasattr(module, "bias") and module.bias is not None:
                params += out_features
            yat_neurons += out_features
            yat_params += params
            total_params += params

        # Check if this is a Conv layer
        elif isinstance(module, (nnx.Conv, YatConv)):
            # Count conv parameters
            if hasattr(module, "kernel"):
                try:
                    kernel_shape = module.kernel.shape
                    kernel_params = int(jnp.prod(jnp.array(kernel_shape)))
                except (AttributeError, TypeError):
                    kernel_params = 0
            else:
                kernel_params = 0

            bias_params = 0
            if hasattr(module, "bias") and module.bias is not None:
                bias_params = (
                    module.out_channels if hasattr(module, "out_channels") else 0
                )

            conv_params = kernel_params + bias_params
            total_params += conv_params

            if isinstance(module, YatConv):
                yat_params += conv_params
            else:
                linear_params += conv_params

        # Recursively traverse child modules
        if hasattr(module, "__dict__"):
            for name, child in module.__dict__.items():
                if isinstance(child, nnx.Module):
                    traverse_module(child, prefix + f"{name}.")
                elif isinstance(child, nnx.List):
                    for i, item in enumerate(child):
                        if isinstance(item, nnx.Module):
                            traverse_module(item, prefix + f"{name}[{i}].")

    traverse_module(model)

    # Log statistics
    print("\n" + "=" * 80)
    print("MODEL STATISTICS")
    print("=" * 80)
    print(f"Total Parameters: {total_params:,}")
    print(f"\nNeuron Types:")
    print(f"  Linear Neurons: {linear_neurons:,}")
    print(f"  YAT Neurons: {yat_neurons:,}")
    print(f"  Total Neurons: {linear_neurons + yat_neurons:,}")
    print(f"\nParameter Distribution:")
    print(f"  Linear Parameters: {linear_params:,}")
    print(f"  YAT Parameters: {yat_params:,}")
    print(f"  Total (check): {linear_params + yat_params:,}")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# 5. Model Initialization Helper
# -----------------------------------------------------------------------------


@nnx.jit(
    static_argnames=(
        "block_cls",
        "layers",
        "num_classes",
        "dtype",
        "lr",
        "epochs",
        "tie_yat_kernels",
        "yat_kernel_bank_size",
        "yat_kernel_bank_id",
        "use_yat_head",
        "tie_yat_nmn_kernels",
        "yat_nmn_kernel_bank_size",
        "yat_nmn_kernel_bank_id",
    )
)
def create_model_and_optimizer(
    block_cls,
    layers,
    num_classes,
    dtype,
    lr,
    epochs,
    tie_yat_kernels=False,
    yat_kernel_bank_size=None,
    yat_kernel_bank_id="resnet-yat",
    use_yat_head=True,
    tie_yat_nmn_kernels=False,
    yat_nmn_kernel_bank_size=None,
    yat_nmn_kernel_bank_id="resnet-yat-head",
):
    rngs = nnx.Rngs(0)
    model = ResNet(
        block_cls=block_cls,
        layers=layers,
        num_classes=num_classes,
        dtype=dtype,
        tie_yat_kernels=tie_yat_kernels,
        yat_kernel_bank_size=yat_kernel_bank_size,
        yat_kernel_bank_id=yat_kernel_bank_id,
        use_yat_head=use_yat_head,
        tie_yat_nmn_kernels=tie_yat_nmn_kernels,
        yat_nmn_kernel_bank_size=yat_nmn_kernel_bank_size,
        yat_nmn_kernel_bank_id=yat_nmn_kernel_bank_id,
        rngs=rngs,
    )

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs * 1000)
    optimizer = nnx.Optimizer(
        model, optax.adamw(schedule, weight_decay=0.01), wrt=nnx.Param
    )
    return model, optimizer


# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    # Model Args
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Global batch size across all devices",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Directory to cache the dataset"
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Cache the dataset in RAM (keep_in_memory=True)",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming mode (default: False)"
    )
    parser.add_argument(
        "--tie-yat-kernels",
        action="store_true",
        help="Tie YatConv kernels via shared banks and slice first k filters",
    )
    parser.add_argument(
        "--yat-kernel-bank-size",
        type=int,
        default=None,
        help="Shared bank size for tied YatConv kernels (must be >= out_channels)",
    )
    parser.add_argument(
        "--yat-kernel-bank-id",
        type=str,
        default="resnet-yat",
        help="Namespace/id for shared YatConv kernel banks",
    )
    parser.add_argument(
        "--use-yat-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use YatNMN classifier head instead of nnx.Linear",
    )
    parser.add_argument(
        "--tie-yat-nmn-kernels",
        action="store_true",
        help="Tie YatNMN kernels via shared banks and slice first k neurons",
    )
    parser.add_argument(
        "--yat-nmn-kernel-bank-size",
        type=int,
        default=None,
        help="Shared bank size for tied YatNMN kernels (must be >= out_features)",
    )
    parser.add_argument(
        "--yat-nmn-kernel-bank-id",
        type=str,
        default="resnet-yat-head",
        help="Namespace/id for shared YatNMN kernel banks",
    )

    # Checkpointing & Logging Args
    parser.add_argument("--save-dir", type=str, default="./checkpoints_flax")
    parser.add_argument(
        "--checkpoint-keep", type=int, default=3, help="Number of checkpoints to keep"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="imagenet-flax", help="WandB project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default="irf-sic", help="WandB entity/username"
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    parser.add_argument(
        "--wandb-log-freq", type=int, default=50, help="Log to WandB every N iterations"
    )

    # Monitor options
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=True,
        help="Enable Jupyter training monitor",
    )
    parser.add_argument(
        "--no-monitor",
        dest="monitor",
        action="store_false",
        help="Disable Jupyter training monitor",
    )
    parser.add_argument(
        "--monitor-freq", type=int, default=10, help="Update monitor every N iterations"
    )

    parser.add_argument(
        "-f", "--file", type=str, default="", help="Jupyter kernel file (ignored)"
    )

    # Use parse_known_args to avoid crashing on Jupyter/IPython arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    # 1. Initialize WandB
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )
        print(f"WandB initialized: {args.wandb_project}")

    # 2. Setup Orbax
    ckpt_dir = os.path.abspath(args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Use StandardCheckpointer for PyTrees
    options = ocp.CheckpointManagerOptions(
        max_to_keep=args.checkpoint_keep, create=True
    )
    mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options)
    print(f"Orbax Checkpoint Manager initialized at {ckpt_dir}")

    # -------------------------------------------------------------------------
    # MESH SETUP
    # -------------------------------------------------------------------------
    # Simulate devices if running on CPU-only local machine for testing
    if jax.default_backend() == "cpu" and len(jax.devices()) == 1:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        if not jax._src.xla_bridge.backends_are_initialized():
            jax.config.update("jax_num_cpu_devices", 8)

    devices = jax.devices()
    num_devices = len(devices)
    print(f"JAX Devices: {num_devices} ({devices[0].platform})")

    mesh_shape = (num_devices, 1)

    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ("data", "model"))
    print(f"Mesh Shape: {mesh_shape}, Axis Names: ('data', 'model')")

    # Logical Axis Rules
    # Map 'batch' -> 'data', others (embed, hidden, etc.) to 'model' if desired
    # For ResNet, we primarily care about batch sharding.
    logical_axis_rules = [
        ("batch", "data"),
        ("embed", "model"),
        ("features", "model"),
        ("channels", "model"),
    ]

    # Enable Eager Sharding
    nnx.use_eager_sharding(True)

    # -------------------------------------------------------------------------
    # MODEL INIT
    # -------------------------------------------------------------------------
    block_map = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
        "resnet101": (Bottleneck, [3, 4, 23, 3]),
    }
    block_cls, layer_counts = block_map[args.model]
    layers = tuple(layer_counts)  # Make hashable for JIT
    dtype = jnp.bfloat16 if args.mixed_precision else jnp.float32

    # Initialize Model under Mesh & Rules Context
    with mesh, nnx.logical_axis_rules(logical_axis_rules):
        model, optimizer = create_model_and_optimizer(
            block_cls,
            layers,
            1000,
            dtype,
            args.lr,
            args.epochs,
            args.tie_yat_kernels,
            args.yat_kernel_bank_size,
            args.yat_kernel_bank_id,
            args.use_yat_head,
            args.tie_yat_nmn_kernels,
            args.yat_nmn_kernel_bank_size,
            args.yat_nmn_kernel_bank_id,
        )

    state = nnx.state((model, optimizer))
    # jax.debug.visualize_array_sharding(state['model']['conv1']['kernel'].value)

    print(f"Model {args.model} initialized and sharded.")

    # Analyze and log model statistics
    analyze_model(model)

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    # Divide batch size by number of devices since Grain's ShardByJaxProcess
    # will give each process a full batch. Sharding within a device mesh
    # happens at the jax.device_put level.

    per_device_batch = args.batch_size // num_devices
    num_jax_processes = jax.process_count()
    print(
        f"Per-device batch size: {per_device_batch} (global: {args.batch_size} / devices: {num_devices})"
    )
    print(f"JAX processes: {num_jax_processes}")

    # On single-host TPU (1 JAX process, N devices): ShardByJaxProcess does NOT shard.
    # We must use global batch size for Grain so we get the right iteration count.
    # jax.device_put with NamedSharding handles splitting across devices.
    # On multi-host TPU pods: ShardByJaxProcess shards across hosts, so use per-host batch.
    grain_batch_size = args.batch_size // num_jax_processes
    print(
        f"Grain batch size: {grain_batch_size} (global: {args.batch_size} / jax_processes: {num_jax_processes})"
    )

    if GRAIN_AVAILABLE:
        print("Using Grain DataLoader (optimized for TPU)")
    else:
        print("\n" + "!" * 80)
        print("WARNING: Grain not available!")
        print("Grain is REQUIRED for TPU training. Install with: pip install grain")
        print("Falling back to PyTorch DataLoader (significantly slower on TPU)")
        print("!" * 80 + "\n")

        # Setup PyTorch DataLoader fallback with in-memory option
        if args.streaming:
            print("Using Streaming Dataset")
            train_dataset = ImageNetStreamDataset(
                split="train", transform=get_transforms(True)
            )
            val_dataset = ImageNetStreamDataset(
                split="validation", transform=get_transforms(False)
            )
        else:
            print(f"Using Map-Style Dataset")
            if args.in_memory:
                print(f"  → Caching dataset in memory (keep_in_memory=True)")
            if args.cache_dir:
                print(f"  → Cache directory: {args.cache_dir}")
            train_dataset = ImageNetMapDataset(
                split="train",
                transform=get_transforms(True),
                cache_dir=args.cache_dir,
                keep_in_memory=args.in_memory,
            )
            val_dataset = ImageNetMapDataset(
                split="validation",
                transform=get_transforms(False),
                cache_dir=args.cache_dir,
                keep_in_memory=args.in_memory,
            )

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    best_acc = 0.0
    global_step = 0

    # Create sharding objects once (outside loop for efficiency)
    data_sharding = NamedSharding(mesh, P("data", None, None, None))
    label_sharding = NamedSharding(mesh, P("data"))

    # Initialize Training Monitor
    monitor = None
    if MONITOR_AVAILABLE and args.monitor:
        monitor_update_freq = getattr(args, "monitor_freq", 10)
        monitor = TrainingMonitor(
            num_devices=num_devices,
            use_widgets=True,
            update_freq=monitor_update_freq,
        )
        if _is_notebook_frontend():
            monitor.display()
        else:
            print("Monitor active in text mode (widget UI available in notebooks).")
        monitor.log(
            f"Monitor initialized: {num_devices} devices, update every {monitor_update_freq} steps"
        )
        monitor.log(
            f"Model: {args.model} | Batch: {args.batch_size} | Epochs: {args.epochs}"
        )
        monitor.log_memory()  # Initial memory snapshot

    # Estimate training time
    # Count total iterations
    train_samples = 1281167  # ImageNet train set size
    val_samples = 50000  # ImageNet val set size
    iterations_per_epoch = train_samples // args.batch_size
    val_iterations_per_epoch = val_samples // args.batch_size
    total_iterations = iterations_per_epoch * args.epochs

    print(f"\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Total training samples: {train_samples:,}")
    print(f"Batch size (global): {args.batch_size}")
    print(f"Iterations per epoch: ~{iterations_per_epoch:,}")
    print(f"Validation iterations per epoch: ~{val_iterations_per_epoch:,}")
    print(f"Total iterations: ~{total_iterations:,}")
    print(f"Total epochs: {args.epochs}")
    print("=" * 80)
    print("Time estimates will be displayed after first few iterations...\n")

    # Helper to process batches
    def process_batch(batch):
        if GRAIN_AVAILABLE:
            # Grain Batch operation already stacks into dicts with arrays
            imgs_np = batch["image"]
            lbls_np = batch["label"]
        else:
            # PyTorch DataLoader
            imgs_np, lbls_np = get_numpy_batch(batch)
        return imgs_np, lbls_np

    import time

    epoch_times = []
    iter_times = []

    print(f"\n" + "=" * 80)
    print("STARTING TRAINING LOOP")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs} - Creating fresh dataloaders...")
        print(f"{'='*80}")

        # Recreate dataloaders for each epoch
        if GRAIN_AVAILABLE:
            try:
                train_loader = create_grain_dataloader(
                    split="train",
                    batch_size=grain_batch_size,
                    is_train=True,
                    cache_dir=args.cache_dir,
                    num_epochs=1,
                    num_devices=num_devices,
                )
                val_loader = create_grain_dataloader(
                    split="validation",
                    batch_size=grain_batch_size,
                    is_train=False,
                    cache_dir=args.cache_dir,
                    num_epochs=1,
                    num_devices=num_devices,
                )
                print(f"✓ Grain dataloaders created successfully for epoch {epoch}")
            except Exception as e:
                print(f"\n" + "!" * 80)
                print(f"FATAL ERROR creating Grain dataloaders for epoch {epoch}:")
                print(f"{e}")
                print(f"!" * 80)
                raise
        else:
            # PyTorch DataLoader fallback
            try:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size // num_devices,
                    num_workers=0,
                    drop_last=True,
                    shuffle=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size // num_devices,
                    num_workers=0,
                    drop_last=True,
                    shuffle=False,
                )
                print(f"✓ PyTorch dataloaders created for epoch {epoch}")
                if args.in_memory:
                    print(f"  (using in-memory dataset)")
            except Exception as e:
                print(f"\n" + "!" * 80)
                print(f"FATAL ERROR creating PyTorch dataloaders for epoch {epoch}:")
                print(f"{e}")
                print(f"!" * 80)
                raise

        epoch_start_time = time.time()
        model.train()
        epoch_losses = []
        epoch_metrics = []  # Accuracy

        print(f"Starting training for epoch {epoch}...")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=iterations_per_epoch)
        for batch_idx, batch in enumerate(pbar):
            iter_start_time = time.time()
            global_step += 1
            imgs_np, lbls_np = process_batch(batch)

            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)

            loss, aux = train_step(model, optimizer, imgs_sharded, lbls_sharded)

            loss_val = float(loss)
            acc_val = float(aux)
            epoch_losses.append(loss_val)
            epoch_metrics.append(acc_val)

            iter_time = time.time() - iter_start_time
            iter_times.append(iter_time)

            # Update training monitor
            if monitor is not None:
                monitor.log_iteration(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=loss_val,
                    acc=acc_val,
                    iter_time=iter_time,
                    batch_size=args.batch_size,
                )

            # Iteration Logging - log every N iterations instead of every iteration
            if args.wandb_project and (batch_idx + 1) % args.wandb_log_freq == 0:
                wandb.log(
                    {
                        "train/iter_loss": loss_val,
                        "train/iter_acc": acc_val,
                        "trainer/global_step": global_step,
                        "epoch": epoch,
                    }
                )

            # Update progress bar with time estimate
            if len(iter_times) >= 5:
                avg_iter_time = np.mean(iter_times[-5:])
                remaining_iters = max(0, iterations_per_epoch - batch_idx - 1)
                remaining_epochs = args.epochs - epoch

                epoch_time_estimate = avg_iter_time * (
                    remaining_iters + val_iterations_per_epoch
                )
                total_time_estimate = epoch_time_estimate + (
                    remaining_epochs
                    * avg_iter_time
                    * (iterations_per_epoch + val_iterations_per_epoch)
                )

                pbar.set_postfix(
                    {
                        "metric": float(np.mean(epoch_metrics[-10:])),
                        "iter_time": f"{avg_iter_time:.2f}s",
                        "epoch_eta": f"{epoch_time_estimate/60:.1f}m",
                    }
                )

            # Safety: break if we've done enough iterations (in case dataloader is cycling)
            if batch_idx + 1 >= iterations_per_epoch * 2:
                print(
                    f"\n⚠️  WARNING: Epoch {epoch} has {batch_idx + 1} iterations (expected ~{iterations_per_epoch})"
                )
                print("Breaking to prevent epoch from continuing indefinitely")
                break

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Log actual iterations completed
        actual_train_iters = min(batch_idx + 1, iterations_per_epoch)
        print(
            f"✓ Training complete: {actual_train_iters} iterations in {epoch_time/60:.1f}m"
        )

        # Calculate epoch averages
        avg_train_loss = np.mean(epoch_losses)
        avg_train_metric = np.mean(epoch_metrics)

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        val_start_time = time.time()
        val_acc = 0.0
        val_loss = 0.0

        total_acc = []
        total_loss = []
        val_batch_idx = 0
        for val_batch_idx, batch in enumerate(
            tqdm(val_loader, desc="Val", total=val_iterations_per_epoch)
        ):
            imgs_np, lbls_np = process_batch(batch)

            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)

            loss, acc = val_step(model, imgs_sharded, lbls_sharded)
            total_acc.append(float(acc))
            total_loss.append(float(loss))

            # Safety: break if validation has too many batches
            if val_batch_idx + 1 >= val_iterations_per_epoch * 2:
                print(
                    f"WARNING: Validation has {val_batch_idx + 1} batches (expected ~{val_iterations_per_epoch})"
                )
                print("Breaking to prevent validation from continuing indefinitely")
                break

        val_time = time.time() - val_start_time
        actual_val_iters = val_batch_idx + 1
        print(
            f"[Epoch {epoch}] Completed {actual_val_iters} validation iterations in {val_time/60:.1f}m"
        )

        val_acc = float(np.mean(total_acc) * 100)
        val_loss = float(np.mean(total_loss))

        # Update monitor with epoch summary
        if monitor is not None:
            monitor.log_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                train_acc=avg_train_metric,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_time=time.time() - epoch_start_time,
            )
            monitor.log_memory()  # Memory snapshot after validation

        # Calculate time estimates
        avg_iter_time = (
            np.mean(iter_times[-100:])
            if len(iter_times) >= 100
            else np.mean(iter_times)
        )
        remaining_epochs = args.epochs - epoch
        total_time_per_epoch = avg_iter_time * (
            iterations_per_epoch + val_iterations_per_epoch
        )
        total_remaining_time = total_time_per_epoch * remaining_epochs

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Acc: {avg_train_metric:.4f} | Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")
        print(
            f"  Epoch Time: {epoch_time/60:.1f}m (Train: {(epoch_time-val_time)/60:.1f}m + Val: {val_time/60:.1f}m)"
        )
        print(f"  Avg Iter Time: {avg_iter_time:.2f}s")
        if remaining_epochs > 0:
            print(
                f"  Estimated Time Remaining: {total_remaining_time/3600:.1f}h ({remaining_epochs} epochs)"
            )
        print(f"  Total Time So Far: {sum(epoch_times)/3600:.1f}h")

        # ---------------------------------------------------------------------
        # LOGGING (WandB)
        # ---------------------------------------------------------------------
        if args.wandb_project:
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "train/metric": avg_train_metric,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "timing/epoch_time_min": epoch_time / 60,
                "timing/avg_iter_time_s": avg_iter_time,
            }
            wandb.log(log_dict)

        # ---------------------------------------------------------------------
        # CHECKPOINTING (Orbax)
        # ---------------------------------------------------------------------
        should_save = False
        if val_acc > best_acc:
            best_acc = val_acc
            should_save = True
            print(f"New best model: {best_acc:.2f}%")

        if should_save:
            # Get current state (parameters + optimizer stats)
            raw_state = nnx.state((model, optimizer))

            # Save using Orbax
            save_args = ocp.args.StandardSave(raw_state)
            mngr.save(step=epoch, args=save_args)
            mngr.wait_until_finished()  # Ensure save completes before proceeding
            print(f"Checkpoint saved for epoch {epoch}")

    # Print final summary
    total_training_time = sum(epoch_times)
    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(
        f"Total Training Time: {total_training_time/3600:.1f}h ({int(total_training_time/60)}m)"
    )
    print(f"Average Time per Epoch: {total_training_time/len(epoch_times)/60:.1f}m")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("=" * 80 + "\n")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
