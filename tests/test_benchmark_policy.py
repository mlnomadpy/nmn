"""Regression tests for the explicit benchmark/correctness boundary."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from benchmarks._timing import TimingStats, measure, validate_results

ROOT = Path(__file__).parents[1]


class _Ready:
    pass


def test_measure_propagates_implementation_failures():
    def broken():
        raise RuntimeError("implementation failed")

    with pytest.raises(RuntimeError, match="implementation failed"):
        measure(broken, warmup=1, iterations=1, synchronize=lambda result: None)


@pytest.mark.parametrize("warmup,iterations", [(0, 1), (1, 0)])
def test_measure_rejects_protocols_that_cannot_produce_samples(warmup, iterations):
    with pytest.raises(ValueError):
        measure(
            _Ready,
            warmup=warmup,
            iterations=iterations,
            synchronize=lambda result: None,
        )


def test_result_validation_fails_for_missing_implementation_or_samples():
    stats = TimingStats((1.0,), 1.0, 0.0, 1.0, 1.0, 1.0)
    with pytest.raises(AssertionError, match="missing"):
        validate_results({"one": {64: stats}}, ("one", "two"), (64,))
    with pytest.raises(AssertionError, match="no samples"):
        validate_results({"one": {}}, ("one",), (64,))


def test_result_validation_rejects_non_finite_or_empty_samples():
    invalid = TimingStats((math.nan,), math.nan, math.nan, math.nan, math.nan, math.nan)
    with pytest.raises(AssertionError, match="invalid samples"):
        validate_results({"one": {64: invalid}}, ("one",), (64,))


def test_benchmark_is_explicit_and_default_ci_excludes_it():
    benchmark = (ROOT / "benchmarks" / "benchmark_attention_speed.py").read_text()
    correctness = (
        ROOT / "tests" / "test_nnx" / "test_attention_benchmark_correctness.py"
    ).read_text()
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    benchmark_workflow = (ROOT / ".github" / "workflows" / "benchmarks.yml").read_text()

    assert "pytestmark = pytest.mark.benchmark" in benchmark
    assert "except Exception" not in benchmark
    assert "perf_counter" not in correctness
    assert "tests/benchmarks" not in workflow
    assert "workflow_dispatch:" in benchmark_workflow
    trigger = benchmark_workflow.split("permissions:", 1)[0]
    assert "pull_request:" not in trigger and "push:" not in trigger
    assert "benchmark_attention_speed.py -m benchmark" in benchmark_workflow
    assert "NMN_BENCHMARK_OUTPUT" in benchmark_workflow
    assert "actions/upload-artifact@v7" in benchmark_workflow
    assert "continue-on-error" not in benchmark_workflow


def test_benchmark_report_records_reproducibility_metadata_and_raw_samples():
    benchmark = (ROOT / "benchmarks" / "benchmark_attention_speed.py").read_text()
    for field in (
        '"hardware"',
        '"versions"',
        '"warmup_iterations"',
        '"sample_iterations"',
        '"lazy_compilation_excluded"',
        '"synchronization"',
        "samples_ms",
    ):
        assert (
            field in benchmark
            or field in (ROOT / "benchmarks" / "_timing.py").read_text()
        )
