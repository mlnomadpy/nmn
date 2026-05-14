"""TensorFlow MNIST training with YatNMN (native tf.Module + GradientTape).

Run:
    python -m nmn.tf.examples.mnist
or:
    python src/nmn/tf/examples/mnist.py

Requires:
    pip install "nmn[tf]" tensorflow tensorflow-datasets
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from nmn.tf import YatNMN


class YatMLP(tf.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10):
        self.fc1 = YatNMN(features=hidden1)
        self.fc2 = YatNMN(features=hidden2)
        self.out = tf.keras.layers.Dense(num_classes)

    def __call__(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="YatNMN MNIST training (TensorFlow)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    print(f"tf devices: {tf.config.list_physical_devices()}")

    def prep(x, y):
        return tf.cast(x, tf.float32) / 255.0, y

    train_ds = (tfds.load("mnist", split="train", as_supervised=True)
                .map(prep).shuffle(10_000).batch(args.batch_size))
    test_ds = (tfds.load("mnist", split="test", as_supervised=True)
               .map(prep).batch(512))

    model = YatMLP()
    # Build variables with a dummy call so optimizer can see them.
    _ = model(tf.zeros((1, 28, 28, 1)))
    opt = tf.keras.optimizers.AdamW(args.lr)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_step(x, y):
        logits = model(x)
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)
        )
        correct = tf.reduce_sum(
            tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32)
                    == tf.cast(y, tf.int32), tf.int32)
        )
        return loss, correct

    n_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"params={n_params:,}")

    history: list[dict] = []
    wall0 = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        running = 0.0
        steps = 0
        for x, y in train_ds:
            loss = train_step(x, y)
            running += float(loss)
            steps += 1
        train_loss = running / max(steps, 1)
        epoch_s = time.perf_counter() - t0

        total_loss = 0.0
        total_correct = 0
        n_eval = 0
        for x, y in test_ds:
            loss_b, correct_b = eval_step(x, y)
            total_loss += float(loss_b)
            total_correct += int(correct_b)
            n_eval += int(tf.shape(y)[0])
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
        "framework": "tensorflow",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "params": n_params,
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
