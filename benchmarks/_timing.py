"""Framework-independent timing and completeness checks for benchmarks."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TimingStats:
    """Raw samples and descriptive statistics for one implementation."""

    samples_ms: tuple[float, ...]
    mean_ms: float
    std_ms: float
    median_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int,
    iterations: int,
    synchronize: Callable[[Any], None],
) -> TimingStats:
    """Measure an already-transformed callable after untimed warmup.

    The first warmup executes any lazy compilation. Exceptions deliberately
    propagate: a required implementation that cannot run is a failed benchmark,
    not a missing data point.
    """
    if warmup < 1:
        raise ValueError("warmup must be at least 1 to exclude lazy compilation")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    for _ in range(warmup):
        synchronize(fn(*args))

    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        synchronize(fn(*args))
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)

    if len(samples) != iterations or not all(
        math.isfinite(sample) and sample > 0.0 for sample in samples
    ):
        raise RuntimeError("benchmark produced missing or invalid timing samples")
    return TimingStats(
        samples_ms=tuple(samples),
        mean_ms=statistics.fmean(samples),
        std_ms=statistics.pstdev(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def validate_results(
    results: Mapping[str, Mapping[int, TimingStats]],
    required_implementations: Iterable[str],
    sequence_lengths: Sequence[int],
) -> None:
    """Fail unless every required implementation has valid samples everywhere."""
    required = set(required_implementations)
    actual = set(results)
    if actual != required:
        raise AssertionError(
            f"benchmark implementations mismatch: missing={sorted(required - actual)}, "
            f"unexpected={sorted(actual - required)}"
        )
    for name in sorted(required):
        timings = results[name]
        missing = set(sequence_lengths) - set(timings)
        if missing:
            raise AssertionError(f"{name} produced no samples for {sorted(missing)}")
        for sequence_length in sequence_lengths:
            samples = timings[sequence_length].samples_ms
            if not samples or not all(
                math.isfinite(sample) and sample > 0.0 for sample in samples
            ):
                raise AssertionError(
                    f"{name} produced invalid samples for sequence length "
                    f"{sequence_length}"
                )
