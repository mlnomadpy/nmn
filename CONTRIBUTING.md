# Contributing to NMN

Thanks for your interest in contributing to **Neural Matter Networks (NMN)**. This document explains how to set up your development environment, run tests, and submit a pull request.

> By contributing, you agree that your contributions will be licensed under the project's [AGPL-3.0](LICENSE) license.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Where to Start](#where-to-start)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Adding a New Layer](#adding-a-new-layer)
- [Pull Request Process](#pull-request-process)
- [Release Process (Maintainers)](#release-process-maintainers)

---

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold it. Please report unacceptable behavior to **taha@azetta.ai**.

---

## Where to Start

Good first contributions:

- 🐛 **Bug fixes** — see [open issues](https://github.com/azettaai/nmn/issues) labeled `good first issue`.
- 📚 **Documentation** — typo fixes, clarifications, new examples, new guides under [`docs/`](docs/).
- ✅ **Tests** — increasing coverage in `tests/test_*/`, adding edge cases, cross-framework consistency checks.
- ⚡ **Examples** — new application examples under `src/nmn/<framework>/examples/`.
- ✨ **New layers** — implementing parity for a layer that exists in one framework but not another (see the [Layer Support Matrix](README.md#-layer-support-matrix)).

Larger work (new layer families, new framework backends, kernel optimizations) should start with an issue or [discussion](https://github.com/azettaai/nmn/discussions) so we can align on the design.

---

## Development Setup

### 1. Fork & clone

```bash
git clone https://github.com/<your-username>/nmn.git
cd nmn
git remote add upstream https://github.com/azettaai/nmn.git
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows PowerShell
```

Python **3.10+** is required for PyTorch/Keras/TF work. Python **3.11+** is required for JAX/Flax work (`jax>=0.9.1`).

### 3. Install in editable mode

Pick the dependency set matching the framework(s) you intend to touch:

```bash
# Single framework
pip install -e ".[dev,torch]"     # PyTorch
pip install -e ".[dev,nnx]"       # Flax NNX (JAX)
pip install -e ".[dev,linen]"     # Flax Linen (JAX)
pip install -e ".[dev,keras]"     # Keras
pip install -e ".[dev,tf]"        # TensorFlow

# Everything (heavy; recommended for cross-framework consistency work)
pip install -e ".[test]"
```

### 4. Verify

```bash
python -c "import nmn; print(nmn.__version__)"
pytest tests/test_torch/ -v       # or whichever framework you installed
```

---

## Running Tests

The full suite lives under `tests/` and is organized per-framework plus `integration/`:

```text
tests/
├── test_torch/      PyTorch
├── test_keras/      Keras
├── test_tf/         TensorFlow
├── test_nnx/        Flax NNX (JAX)
├── test_linen/      Flax Linen (JAX)
├── integration/     Cross-framework numerical consistency
├── benchmarks/      Performance benchmarks
└── scripts/         One-off validation scripts
```

### Common commands

```bash
# Single framework
pytest tests/test_torch/ -v

# Cross-framework consistency (requires all frameworks installed)
pytest tests/integration/ -v

# A single file or test
pytest tests/test_torch/test_nmn.py -v
pytest tests/test_torch/test_nmn.py::TestYatNMN::test_forward -v

# With coverage
pytest tests/ --cov=nmn --cov-report=html
open htmlcov/index.html

# Skip slow tests
pytest tests/ -m "not slow"

# Only one framework's marker
pytest tests/ -m torch
```

Custom markers are declared in [`pyproject.toml`](pyproject.toml): `slow`, `integration`, `torch`, `jax`, `tf`.

### Writing tests

- Place tests in the framework directory matching the code you changed (e.g. a fix in `src/nmn/torch/` → test in `tests/test_torch/`).
- For new layers, add a **cross-framework consistency test** in `tests/integration/` that initializes the same weights in two or more frameworks and asserts max absolute error `< 1e-5` for fp32 inputs.
- Prefer small, deterministic inputs (`torch.manual_seed(0)`, `jax.random.PRNGKey(0)`) over flaky randomness.

---

## Code Style

Style is enforced by **black**, **isort**, **flake8**, and **mypy** — all configured in [`pyproject.toml`](pyproject.toml).

```bash
# Format
black src/ tests/
isort src/ tests/

# Lint
flake8 src/nmn --select=E9,F63,F7,F82       # errors only (must pass)
flake8 src/nmn --max-line-length=127        # style (informational)

# Type check (informational for now, tightening over time)
mypy src/nmn/torch --ignore-missing-imports
mypy src/nmn/nnx   --ignore-missing-imports
```

CI fails only on **flake8 errors** (`E9,F63,F7,F82`) and test failures. Black/isort/mypy are advisory but please run them locally before pushing.

### Conventions

- **Line length**: 88 (black default) is preferred; 127 is the absolute max.
- **Type hints**: required on public APIs; optional inside method bodies.
- **Docstrings**: numpy or google style, short. Document `epsilon`, `use_bias`, `alpha`, and other Yat-specific knobs explicitly.
- **No comments stating the obvious** — only explain *why* when non-obvious.
- **Naming**: keep parity across frameworks (`YatNMN`, `YatConv2D`, `MultiHeadAttention`). When a name must diverge for framework idioms (e.g. `in_features` vs `features`), document the mapping in the layer's docstring.

---

## Adding a New Layer

Layers should reach **all 5 frameworks** before being considered complete. The recommended order:

1. **Reference implementation in Flax NNX** (`src/nmn/nnx/layers/`). NNX is the cleanest place to prototype because of explicit state.
2. **PyTorch** (`src/nmn/torch/layers/` or top-level). Mirror parameter names where possible.
3. **Keras / TF** (`src/nmn/keras/`, `src/nmn/tf/`).
4. **Flax Linen** (`src/nmn/linen/`). Linen is most similar to NNX — usually a thin port.
5. **Cross-framework consistency test** in `tests/integration/`.
6. **Per-framework unit tests** in `tests/test_<framework>/`.
7. **Update**:
   - [`README.md`](README.md) Layer Support Matrix
   - [`EXAMPLES.md`](EXAMPLES.md) with a usage snippet
   - [`CHANGELOG.md`](CHANGELOG.md) under the **Unreleased** section
   - Public `__init__.py` re-exports

---

## Pull Request Process

1. **Branch** from `master`: `git checkout -b feat/short-description` (or `fix/...`, `docs/...`).
2. **Commit** in small, logical units. Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes where possible: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`.
3. **Run tests + linters** locally.
4. **Update CHANGELOG.md** under `## [Unreleased]` if the change is user-visible.
5. **Open a PR** against `master`. Fill in the PR template — describe what changed, why, and how it was tested.
6. **CI must pass** (tests + flake8 errors). Style checks are advisory but help reviewers.
7. **One reviewer approval** is required for merge. A maintainer will merge.

PRs that touch more than one framework should ideally be split, unless the change is intrinsically cross-framework (e.g. a renaming for parity).

---

## Release Process (Maintainers)

Versioning is **VCS-based** via [`hatch-vcs`](https://github.com/ofek/hatch-vcs) — the package version is derived from the latest git tag.

To cut a release:

```bash
# 1. Update CHANGELOG.md: move [Unreleased] entries under [vX.Y.Z] - YYYY-MM-DD
# 2. Commit and push
git commit -am "chore: prepare vX.Y.Z release"
git push origin master

# 3. Tag and push
git tag vX.Y.Z
git push origin vX.Y.Z
```

The [`publish.yml`](.github/workflows/publish.yml) workflow runs automatically on `v*.*.*` tags and publishes to TestPyPI then PyPI via OIDC trusted publishing (no API tokens needed).

---

## Questions?

- 💬 **General**: [GitHub Discussions](https://github.com/azettaai/nmn/discussions)
- 🐛 **Bugs**: [GitHub Issues](https://github.com/azettaai/nmn/issues)
- 📧 **Maintainer**: taha@azetta.ai

Thanks for contributing! 🧠
