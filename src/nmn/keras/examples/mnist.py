"""Keras 3 MNIST training with YatNMN (backend-agnostic).

Run:
    python -m nmn.keras.examples.mnist
or:
    python src/nmn/keras/examples/mnist.py

Set KERAS_BACKEND={jax,tensorflow,torch} before importing keras to choose backend.

Requires:
    pip install "nmn[keras]" keras tensorflow      # or jax / torch
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import keras
import numpy as np

from nmn.keras import YatNMN


def build_model(hidden1: int = 256, hidden2: int = 128, num_classes: int = 10) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input((28, 28)),
        keras.layers.Flatten(),
        YatNMN(features=hidden1),
        YatNMN(features=hidden2),
        keras.layers.Dense(num_classes),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="YatNMN MNIST training (Keras 3)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    print(f"keras backend: {keras.backend.backend()}")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    n_params = model.count_params()
    print(f"params={n_params:,}")

    wall0 = time.perf_counter()
    hist = model.fit(
        x_train, y_train,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(x_test, y_test), verbose=2,
    )
    total_s = time.perf_counter() - wall0

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    history = [
        {"epoch": i,
         "train_loss": float(hist.history["loss"][i]),
         "test_loss": float(hist.history["val_loss"][i]),
         "test_acc": float(hist.history["val_accuracy"][i])}
        for i in range(args.epochs)
    ]
    result = {
        "framework": "keras",
        "backend": keras.backend.backend(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "params": int(n_params),
        "total_s": total_s,
        "final_test_acc": float(test_acc),
        "final_test_loss": float(test_loss),
        "history": history,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2))
        print(f"report -> {args.report}")
    print(f"final: test_acc={test_acc:.4f} test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
