# Benchmarks and diagnostics

This directory contains exploratory programs that report performance,
approximation quality, or intermediate numerical values. They are intentionally
outside `tests/`: running `pytest` must execute only deterministic,
assertion-based checks.

Run a benchmark directly from the repository root, for example:

```bash
python benchmarks/benchmark_may_ray.py
python benchmarks/benchmark_yat_performer.py
python benchmarks/diagnostics/gradient_scaling.py
```

Backend dependencies are optional. Install the corresponding project extra
before running a program, such as `pip install -e ".[nnx]"` for JAX/Flax NNX.
