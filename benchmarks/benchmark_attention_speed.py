"""Explicit JAX attention benchmark with reproducibility metadata.

Run from the repository root with::

    pytest benchmarks/benchmark_attention_speed.py -m benchmark -v -s

This is intentionally outside the default ``tests/`` collection root. It is a
performance report, not a correctness or wall-clock regression gate.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Callable

import jax
import pytest
from flax import nnx

from benchmarks._timing import TimingStats, measure, validate_results
from nmn.nnx.layers.attention import (
    RotaryYatAttention,
    create_yat_projection,
    dot_product_attention,
    yat_attention,
    yat_attention_normalized,
    yat_performer_attention,
)

pytestmark = pytest.mark.benchmark


def _integer_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


SEQUENCE_LENGTHS = tuple(
    int(value)
    for value in os.environ.get(
        "NMN_BENCHMARK_SEQUENCE_LENGTHS", "64,128,256,512,1024"
    ).split(",")
)
if not SEQUENCE_LENGTHS or any(length < 1 for length in SEQUENCE_LENGTHS):
    raise ValueError("NMN_BENCHMARK_SEQUENCE_LENGTHS must contain positive integers")
BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64
EMBED_DIM = NUM_HEADS * HEAD_DIM
NUM_FEATURES = 256
WARMUP_ITERATIONS = _integer_environment("NMN_BENCHMARK_WARMUP", 3)
SAMPLE_ITERATIONS = _integer_environment("NMN_BENCHMARK_ITERATIONS", 10)

IMPLEMENTATIONS = (
    "dot_product",
    "yat_standard",
    "yat_normalized",
    "yat_performer",
    "yat_performer_normalized",
    "rotary_yat",
    "rotary_yat_performer",
    "rotary_yat_performer_normalized",
)


def _synchronize(result) -> None:
    jax.block_until_ready(result)


class AttentionBenchmark:
    """Inputs and transformed callables for one sequence length."""

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        q_key, k_key, v_key, x_key, projection_key = jax.random.split(
            jax.random.key(0), 5
        )
        shape = (BATCH_SIZE, sequence_length, NUM_HEADS, HEAD_DIM)
        self.q = jax.random.normal(q_key, shape)
        self.k = jax.random.normal(k_key, shape)
        self.v = jax.random.normal(v_key, shape)
        self.x = jax.random.normal(x_key, (BATCH_SIZE, sequence_length, EMBED_DIM))
        self.projection = create_yat_projection(projection_key, NUM_FEATURES, HEAD_DIM)
        self._create_modules()

    def _create_modules(self) -> None:
        common = dict(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            max_seq_len=self.sequence_length,
        )
        self.rotary_yat = RotaryYatAttention(
            **common, use_performer=False, rngs=nnx.Rngs(1)
        )
        self.rotary_yat_performer = RotaryYatAttention(
            **common,
            use_performer=True,
            num_prf_features=NUM_FEATURES,
            performer_normalize=False,
            rngs=nnx.Rngs(2),
        )
        self.rotary_yat_performer_normalized = RotaryYatAttention(
            **common,
            use_performer=True,
            num_prf_features=NUM_FEATURES,
            performer_normalize=True,
            rngs=nnx.Rngs(3),
        )

    def cases(self) -> dict[str, tuple[Callable, tuple]]:
        return {
            "dot_product": (
                jax.jit(dot_product_attention),
                (self.q, self.k, self.v),
            ),
            "yat_standard": (jax.jit(yat_attention), (self.q, self.k, self.v)),
            "yat_normalized": (
                jax.jit(yat_attention_normalized),
                (self.q, self.k, self.v),
            ),
            "yat_performer": (
                jax.jit(
                    lambda q, k, v, projection: yat_performer_attention(
                        q, k, v, projection, normalize_inputs=False
                    )
                ),
                (self.q, self.k, self.v, self.projection),
            ),
            "yat_performer_normalized": (
                jax.jit(
                    lambda q, k, v, projection: yat_performer_attention(
                        q, k, v, projection, normalize_inputs=True
                    )
                ),
                (self.q, self.k, self.v, self.projection),
            ),
            "rotary_yat": (
                jax.jit(lambda x: self.rotary_yat(x, deterministic=True)),
                (self.x,),
            ),
            "rotary_yat_performer": (
                jax.jit(lambda x: self.rotary_yat_performer(x, deterministic=True)),
                (self.x,),
            ),
            "rotary_yat_performer_normalized": (
                jax.jit(
                    lambda x: self.rotary_yat_performer_normalized(
                        x, deterministic=True
                    )
                ),
                (self.x,),
            ),
        }


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _metadata() -> dict:
    return {
        "schema": "nmn-attention-benchmark/v1",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "versions": {
            "python": sys.version,
            "nmn": _version("nmn"),
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
            "flax": _version("flax"),
        },
        "protocol": {
            "warmup_iterations": WARMUP_ITERATIONS,
            "sample_iterations": SAMPLE_ITERATIONS,
            "lazy_compilation_excluded": True,
            "synchronization": "jax.block_until_ready",
            "sequence_lengths": list(SEQUENCE_LENGTHS),
            "batch_size": BATCH_SIZE,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "num_features": NUM_FEATURES,
        },
    }


def _write_report(results: dict[str, dict[int, TimingStats]]) -> None:
    report = _metadata()
    report["results"] = {
        name: {str(length): stats.as_dict() for length, stats in timings.items()}
        for name, timings in results.items()
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if output := os.environ.get("NMN_BENCHMARK_OUTPUT"):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")


def test_attention_benchmark() -> None:
    results: dict[str, dict[int, TimingStats]] = {name: {} for name in IMPLEMENTATIONS}
    for sequence_length in SEQUENCE_LENGTHS:
        benchmark = AttentionBenchmark(sequence_length)
        cases = benchmark.cases()
        if set(cases) != set(IMPLEMENTATIONS):
            raise AssertionError("benchmark case registry is incomplete")
        for name in IMPLEMENTATIONS:
            fn, args = cases[name]
            results[name][sequence_length] = measure(
                fn,
                *args,
                warmup=WARMUP_ITERATIONS,
                iterations=SAMPLE_ITERATIONS,
                synchronize=_synchronize,
            )

    validate_results(results, IMPLEMENTATIONS, SEQUENCE_LENGTHS)
    _write_report(results)
