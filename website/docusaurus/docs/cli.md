---
sidebar_position: 5
---

# CLI Reference

The `nmn` console command (also `python -m nmn`) helps you discover the API
**without importing any ML framework**. `import nmn` is import-light — a backend
is only pulled in when you import its subpackage — so the CLI is safe and fast,
which makes it ideal for checking an environment before writing code.

## Commands

| Command | What it does |
|---------|--------------|
| `nmn` / `nmn info` | Banner: version, the six frameworks, and pip extras. |
| `nmn version` | The version string only. |
| `nmn frameworks` | Table of each framework's import line + `YatNMN` signature. |
| `nmn guide <framework>` | Self-contained quickstart (import + small YatNMN MLP + attention one-liner). |
| `nmn features` | MAY/RAY performer maps and YatNMN lazy mode, with a NNX `performer_kind` snippet. |
| `nmn doctor` | Reports import OK/missing + version for each backend; never raises on a missing one. |
| `nmn examples` | Where to find runnable examples + a NNX quickstart. |

## `nmn doctor` — check the environment first

Run this before anything else to see which of the six backends are importable in
the current environment, with versions. It never raises on a missing backend:

```bash
nmn doctor
```

If a backend is missing, install it: `pip install "nmn[<framework>]"`
(`torch`, `nnx`, `linen`, `keras`, `tf`, `mlx`).

## `nmn frameworks` — the per-backend `YatNMN` signature

The `YatNMN` size argument differs per backend (the #1 gotcha). `nmn frameworks`
prints the exact import line and constructor for each:

```bash
nmn frameworks
```

## `nmn guide <framework>` — a runnable quickstart

Prints a verified, self-contained quickstart for one backend. Aliases:
`torch`/`pytorch`, `tf`/`tensorflow`, `nnx`/`flax-nnx`, `linen`/`flax-linen`,
`keras`, `mlx`.

```bash
nmn guide nnx
nmn guide pytorch
nmn guide mlx
```

## `nmn features` — MAY/RAY maps + lazy mode

Prints the MAY/RAY linear-attention performer maps and YatNMN lazy mode, with a
NNX `performer_kind` snippet:

```bash
nmn features
```

See [Linear Attention](/docs/attention/linear-attention) and
[Lazy Training](/docs/guides/lazy-training) for the details.

## Programmatic Equivalents

Two commands have Python equivalents that return values you can inspect:

```python
import nmn

nmn.help()      # same as `nmn info`
nmn.doctor()    # same as `nmn doctor`; returns {framework: version_or_None}
```

`nmn.doctor()` returns a dict such as `{"torch": "2.3.0", "nnx": None, ...}` —
handy for guarding imports in scripts.

## See Also

- [Installation](/docs/installation) - Install a backend
- [Quick Start](/docs/quick-start) - Build your first model
- [Linear Attention](/docs/attention/linear-attention) - `nmn features` in depth
- [Lazy Training](/docs/guides/lazy-training) - Freeze-kernel mode
