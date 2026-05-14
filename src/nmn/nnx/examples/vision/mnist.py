"""Flax NNX MNIST training with YatNMN.

Run:
    python -m nmn.nnx.examples.vision.mnist
or:
    python src/nmn/nnx/examples/vision/mnist.py

Trains a 2-layer YatNMN MLP for 3 epochs. Uses torchvision to fetch MNIST so
no tensorflow-datasets dependency is required.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from torchvision import datasets, transforms

from nmn.nnx import YatNMN


class YatMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10):
        self.fc1 = YatNMN(in_features=28 * 28, out_features=hidden1, rngs=rngs)
        self.fc2 = YatNMN(in_features=hidden1, out_features=hidden2, rngs=rngs)
        self.out = nnx.Linear(hidden2, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


def loss_fn(model: YatMLP, x: jax.Array, y: jax.Array) -> jax.Array:
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@nnx.jit
def train_step(model: YatMLP, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: YatMLP, x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).sum()
    correct = (jnp.argmax(logits, axis=1) == y).sum()
    return loss, correct


def iter_batches(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    n = images.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n - batch_size + 1, batch_size):
        b = idx[start:start + batch_size]
        yield jnp.asarray(images[b]), jnp.asarray(labels[b])


def load_mnist(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    x_train = (train.data.numpy().astype(np.float32) / 255.0)[..., None]
    y_train = train.targets.numpy().astype(np.int32)
    x_test = (test.data.numpy().astype(np.float32) / 255.0)[..., None]
    y_test = test.targets.numpy().astype(np.int32)
    return x_train, y_train, x_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="YatNMN MNIST training (Flax NNX)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=".data")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    print(f"jax devices: {jax.devices()}")

    x_train, y_train, x_test, y_test = load_mnist(args.data_root)
    rng = np.random.default_rng(args.seed)

    model = YatMLP(rngs=nnx.Rngs(args.seed))
    optimizer = nnx.Optimizer(model, optax.adamw(args.lr), wrt=nnx.Param)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"params={n_params:,}")

    history: list[dict] = []
    wall0 = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        running = 0.0
        steps = 0
        for x, y in iter_batches(x_train, y_train, args.batch_size, True, rng):
            loss = train_step(model, optimizer, x, y)
            running += float(loss)
            steps += 1
        train_loss = running / steps
        epoch_s = time.perf_counter() - t0

        total_loss = 0.0
        total_correct = 0
        n_eval = 0
        for x, y in iter_batches(x_test, y_test, 512, False, rng):
            loss_b, correct_b = eval_step(model, x, y)
            total_loss += float(loss_b)
            total_correct += int(correct_b)
            n_eval += int(y.shape[0])
        test_loss = total_loss / n_eval
        test_acc = total_correct / n_eval
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
        "framework": "flax-nnx",
        "device": str(jax.devices()[0]),
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
