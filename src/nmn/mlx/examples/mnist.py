"""MLX MNIST training with YatNMN.

Run:
    python -m nmn.mlx.examples.mnist
or:
    python src/nmn/mlx/examples/mnist.py

Trains a 2-layer YatNMN MLP for 3 epochs on Apple Silicon (Metal GPU by
default). Uses ``torchvision`` to fetch MNIST so no tensorflow-datasets
dependency is required.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from torchvision import datasets, transforms

from nmn.mlx import YatNMN


class YatMLP(nn.Module):
    def __init__(self, in_features: int = 28 * 28, hidden1: int = 256,
                 hidden2: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = YatNMN(features=hidden1)
        self.fc2 = YatNMN(features=hidden2)
        self.out = nn.Linear(hidden2, num_classes)
        # Build the YatNMN parameters eagerly so optimizer / value_and_grad
        # see the full parameter tree from step 0.
        dummy = mx.zeros((1, in_features))
        _ = self.fc1(dummy)
        _ = self.fc2(self.fc1(dummy))

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


def loss_fn(model: YatMLP, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def eval_batches(model: YatMLP, images: np.ndarray, labels: np.ndarray,
                 batch_size: int = 512) -> tuple[float, float]:
    n = images.shape[0]
    total_loss = 0.0
    correct = 0
    for start in range(0, n, batch_size):
        x = mx.array(images[start:start + batch_size])
        y = mx.array(labels[start:start + batch_size])
        logits = model(x)
        total_loss += float(mx.sum(nn.losses.cross_entropy(logits, y)))
        correct += int(mx.sum(mx.argmax(logits, axis=1) == y))
    return total_loss / n, correct / n


def iter_batches(images: np.ndarray, labels: np.ndarray, batch_size: int,
                 shuffle: bool, rng: np.random.Generator):
    n = images.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n - batch_size + 1, batch_size):
        b = idx[start:start + batch_size]
        yield mx.array(images[b]), mx.array(labels[b])


def load_mnist(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    x_train = (train.data.numpy().astype(np.float32) / 255.0)[..., None]
    y_train = train.targets.numpy().astype(np.int32)
    x_test = (test.data.numpy().astype(np.float32) / 255.0)[..., None]
    y_test = test.targets.numpy().astype(np.int32)
    return x_train, y_train, x_test, y_test


def _count_params(params) -> int:
    total = 0
    for v in mx.tree_flatten(params)[0] if hasattr(mx, "tree_flatten") else []:
        total += v.size
    if total == 0:
        # Fallback: walk the dict manually.
        def walk(p):
            nonlocal total
            if isinstance(p, mx.array):
                total += p.size
            elif isinstance(p, dict):
                for v in p.values():
                    walk(v)
            elif isinstance(p, (list, tuple)):
                for v in p:
                    walk(v)
        walk(params)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="YatNMN MNIST training (MLX)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=".data")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="MLX device (default: gpu / Metal)")
    parser.add_argument("--report", type=str, default=None,
                        help="optional path to write JSON results")
    args = parser.parse_args()

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    print(f"mlx device: {mx.default_device()}")

    x_train, y_train, x_test, y_test = load_mnist(args.data_root)
    rng = np.random.default_rng(args.seed)

    model = YatMLP()
    optimizer = optim.AdamW(learning_rate=args.lr)

    n_params = _count_params(model.parameters())
    print(f"params={n_params:,}")

    grad_fn = nn.value_and_grad(model, loss_fn)

    history: list[dict] = []
    wall0 = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        running = 0.0
        steps = 0
        for x, y in iter_batches(x_train, y_train, args.batch_size, True, rng):
            loss, grads = grad_fn(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            running += float(loss)
            steps += 1
        train_loss = running / steps
        epoch_s = time.perf_counter() - t0

        test_loss, test_acc = eval_batches(model, x_test, y_test)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"time={epoch_s:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "test_acc": test_acc, "epoch_s": epoch_s,
        })
    total_s = time.perf_counter() - wall0

    result = {
        "framework": "mlx",
        "device": str(mx.default_device()),
        "params": int(n_params),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "total_s": total_s,
        "final_test_acc": history[-1]["test_acc"],
        "final_test_loss": history[-1]["test_loss"],
        "history": history,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2))
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
