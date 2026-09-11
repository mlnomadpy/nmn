# Benchmarks and diagnostics

This directory contains exploratory programs that report performance,
approximation quality, or intermediate numerical values. They are intentionally
outside `tests/`: running `pytest` must execute only deterministic,
assertion-based checks.

Run a benchmark directly from the repository root, for example:

```bash
make benchmark-attention
python benchmarks/benchmark_may_ray.py
python benchmarks/benchmark_yat_performer.py
python benchmarks/diagnostics/gradient_scaling.py
```

Backend dependencies are optional. Install the corresponding project extra
before running a program, such as `pip install -e ".[nnx]"` for JAX/Flax NNX.

`benchmark_attention_speed.py` is a strict benchmark: every registered
implementation must run at every configured sequence length and emit all raw
samples. It writes a reproducible JSON report when `NMN_BENCHMARK_OUTPUT` is
set. The manually dispatched **Attention Benchmarks** GitHub Actions workflow
sets that path and retains the report as an artifact. No wall-clock threshold
is used as a correctness or pull-request gate.

For a quick pipeline smoke test, override the protocol explicitly, for example:

```bash
NMN_BENCHMARK_SEQUENCE_LENGTHS=16 \
NMN_BENCHMARK_WARMUP=1 \
NMN_BENCHMARK_ITERATIONS=2 \
make benchmark-attention
```

These resolved values are recorded in the JSON report; do not compare a smoke
run with the default benchmark protocol.
