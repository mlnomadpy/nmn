"""PyTorch MNIST training with YatNMN.

Run:
    python -m nmn.torch.examples.vision.mnist
or:
    python src/nmn/torch/examples/vision/mnist.py

Trains a 2-layer YatNMN MLP for 3 epochs and prints train loss + test accuracy.
On CPU (Apple Silicon, batch size 128) ~30-60s per epoch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nmn.torch import YatNMN


class YatMLP(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = YatNMN(in_features=28 * 28, out_features=hidden1)
        self.fc2 = YatNMN(in_features=hidden1, out_features=hidden2)
        self.out = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser(description="YatNMN MNIST training (PyTorch)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=".data")
    parser.add_argument("--report", type=str, default=None,
                        help="optional path to write JSON results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(args.data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.data_root, train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model = YatMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,}")

    history: list[dict] = []
    wall0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        steps = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        train_loss = running / steps
        epoch_s = time.perf_counter() - t0
        test_loss, test_acc = evaluate(model, test_dl, device)
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
        "framework": "pytorch",
        "device": device,
        "params": n_params,
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
