# Changelog

All notable changes to NMN are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Changed
- Repaired the contributor workflow so clean dev installs can build packages,
  local pytest always resolves the checkout, Make and CI share package-wide
  type checking, and documentation/templates match the six supported backends.
- Hardened CI/CD with immutable third-party Action pins, job-scoped OIDC,
  bounded release/deploy jobs, unskippable policy checks, and scheduled runtime
  smoke tests for every declared backend minimum.
- Restored PyTorch 1.11 masked-attention compatibility and corrected the MLX
  lower bound to 0.18.1, the first published release in the 0.18 series.
- Patched the website's `serialize-javascript` and `uuid` dependency paths and
  added a fail-closed audit gate for new high/critical advisories, retaining an
  exact reviewed allowance only for two upstream `image-size` no-fix findings.
- Updated `qs` to 6.16.0 to resolve its request-buffer denial-of-service
  advisory without expanding the website audit allowlist.
- Made coverage uploads authenticated through OIDC and failure-blocking, with
  explicit project and patch coverage policies for the combined backend flags.
- Added a repository-owned coverage gate that combines JAX, PyTorch, and
  Keras/TensorFlow data, enforcing 70% project and 80% changed-line coverage
  even when an external reporting integration is unavailable.

---

## [0.3.4] — 2026-09-02

### Added
- **Complete direct-package static analysis** — MyPy 2.3.1 now checks every
  Python module under `src/nmn` through one canonical configuration. CI policy
  tests reject package exclusions, duplicate MyPy invocations, and accidental
  narrowing of the checked package surface. Imported third-party modules remain
  skipped so each NMN module owns its errors consistently across the supported
  minimum/latest dependency matrix.
- **Website pull-request validation** — Docusaurus now builds on Node.js 24 for
  website changes, using the same static-asset preparation path as deployment.
- **Minimum and latest JAX/Flax coverage** within the existing CI matrix, plus
  policy tests for dependency, documentation, optional-backend collection, and
  workflow invariants.

### Changed
- Updated every GitHub Actions checkout to `actions/checkout@v7`, artifact
  upload/download actions to their Node.js 24 releases, Docusaurus packages to
  3.10.2, and React/React DOM to 19.2.8.
- Reorganized the test tree so assertion-based regressions remain under pytest
  while exploratory diagnostics and benchmarks live outside test collection.
- Modernized Flax NNX variable updates to avoid deprecated `.value`
  assignments while preserving shared-bank identity, slices, and gradients.
- Corrected accelerator installation documentation to use the real `nmn[nnx]`
  extra with the official JAX TPU/CUDA wheels and metadata-aligned minimums.

### Fixed
- Source distributions explicitly exclude the local `.dacli` orchestration and
  evidence ledger, including run transcripts and task worktrees.
- Mirror synchronization now fails visibly when its credential is unavailable
  instead of reporting a false successful no-op.
- Pinned lint/type-check tooling and import-light CLI annotations make local and
  CI quality gates deterministic across supported backend versions.

---

## [0.3.3] — 2026-09-01

### Added
- Native Apple Silicon/Metal MLX CI, TPU-oriented Pallas validation, Windows
  process-tree cleanup regressions, safe optional-backend collection, and a
  broader cross-framework numerical-parity suite.
- Precision policy plumbing for every Pallas attention dot product and
  TPU-legal NNX Pallas BlockSpecs for forward and backward kernels.

### Changed
- Low-precision YAT score paths accumulate overflow-prone intermediates and
  operator-local cotangents safely while preserving genuine NaNs and public
  float16/bfloat16 output contracts.
- Learnable epsilon initialization uses stable inverse softplus with validated,
  dtype-aware storage across dense and convolution families.
- Fully masked attention rows, broadcast masks, normalization, output
  projection, and incremental decoding semantics are aligned across backends.

### Fixed
- Grouped and transposed convolution parity, Keras `channels_first` execution
  on TensorFlow CPU, shared-kernel-bank lifecycle behavior, and serialization.
- MLX fused epsilon gradients and transposed-convolution `SAME` behavior,
  including compiled/lazy and native Metal coverage.
- NNX attention custom-VJP gradients, cache transactionality and overflow,
  DropConnect RNG behavior, grouped patch norms, and transpose-kernel layouts.
- Release publishing now runs only for version tags and validates artifacts in
  TestPyPI before publishing to PyPI.

---

## [0.3.2] — 2026-06-09

### Added
- **`nmn` command-line interface + `python -m nmn`** — a discovery/diagnostics CLI exposed via the `nmn` console script (and `python -m nmn`). It is import-light: importing the CLI does **not** import torch/jax/tf/mlx, so it stays fast and works even when no backend is installed. Commands:
  - `nmn` / `nmn info` — banner with name, version, one-line description, the six frameworks with their pip extras (`nmn[torch]`, `nmn[nnx]`, `nmn[keras]`, `nmn[tf]`, `nmn[linen]`, `nmn[mlx]`), and a pointer to `nmn guide <framework>`.
  - `nmn version` — prints the version string only.
  - `nmn frameworks` — per-framework import line + the `YatNMN` constructor signature for that framework.
  - `nmn guide <framework>` — self-contained quickstart (import + a small `YatNMN` MLP + a one-liner attention example) for `torch`/`pytorch`, `nnx`/`flax-nnx`, `linen`/`flax-linen`, `keras`, `tf`/`tensorflow`, `mlx` (aliases accepted).
  - `nmn features` — concise usage of the MAY/RAY performer feature maps and lazy `YatNMN` mode, including a nnx `performer_kind` snippet.
  - `nmn doctor` — reports import OK/missing + version for each of the six backends, plus the running Python and `nmn` version. Never raises on a missing backend.
  - `nmn examples` — where to find runnable examples + a nnx quickstart, with a link to `EXAMPLES.md`.
- **Programmatic helpers** `nmn.help()` and `nmn.doctor()` in `src/nmn/__init__.py` (kept import-light via lazy import). `nmn.help()` prints the `nmn info` banner; `nmn.doctor()` prints the `nmn doctor` report and returns a `{framework: version_str_or_None}` dict.
- **Docs & developer-experience refresh** — updated `README.md` (CLI / discovery section, per-framework install extras), `CHANGELOG.md`, agent-oriented `llms.txt` / `AGENTS.md`, per-framework guides under `docs/guides/` (lazy-training notes + MAY/RAY/SLAY section), `EXAMPLES.md`, `docs/architecture.md`, `docs/migration.md`, plus `Makefile`, `.pre-commit-config.yaml`, and `CONTRIBUTING.md`.

### Notes
- Fully additive and backward compatible — no layer behavior changed in this release.

---

## [0.3.1] — 2026-06-08

### Added
- **Bias-aware linear-attention feature maps (MAY / RAY)** in **all six frameworks** (`nmn.torch`, `nmn.nnx`, `nmn.linen`, `nmn.keras`, `nmn.tf`, `nmn.mlx`). Linearized spherical-Yat attention approximates the kernel `κ(s) = (s + b)² / ((2 + ε) − 2s)` (with `s = q̂·k̂`) by a feature map `φ` so that `φ(q)·φ(k) ≈ κ`, giving **O(n)** attention.
  - **MAY** (Random-Maclaurin, bias-aware): `create_maclaurin_projection`, `maclaurin_features`, `maclaurin_yat_attention` (+ `maclaurin_coeffs`).
  - **RAY** (radial, sharp-`ε` route): `create_radial_projection`, `radial_features`, `radial_yat_attention`.
  - Canonical kwargs `bias` and `epsilon`. MAY/RAY are bias-aware (`b > 0`), where the `b = 0` SLAY anchor cannot follow the learned per-head bias. Features are sign-indefinite — see the module docstrings for the caveat.
  - On **Flax NNX**, the attention layer takes a selector `performer_kind="slay" | "maclaurin" | "radial"` (default `"slay"`).
- **Lazy `YatNMN` training (freeze kernel)** in **all six frameworks** — `YatNMN(..., lazy=True)` (alias `freeze_kernel=True`) freezes **only the kernel** (feature directions) while keeping `bias`, `alpha`, and `epsilon` trainable. Each backend uses its idiomatic mechanism: NNX `FrozenParam` (excluded from `nnx.state(model, nnx.Param)`), torch `requires_grad=False`, mlx `freeze`, Keras/TF `trainable=False`, linen `stop_gradient`. Default is unchanged (`lazy=False`).
- **GOAT value-only / projection-free YAT attention (MLX)** — `GoatYatAttention`, `goat_yat_attention`, `goat_yat_attention_weights` in `nmn.mlx`.

### Notes
- Backward compatible: new attention helpers and the `lazy`/`freeze_kernel` kwargs are additive; existing defaults and outputs are unchanged.

---

## [0.3.0] — 2026-05-14

### Added
- **MLX backend (Apple Silicon) — the sixth NMN framework** (`nmn.mlx`). Apple-Silicon-native YAT layers mirroring the `nmn.tf` / `nmn.keras` surface: `YatNMN` (alias `YatDense`, `features=`), `YatConv{1,2,3}D` / `YatConvTranspose{1,2,3}D`, `YatEmbed`, `MultiHeadYatAttention`, `RotaryYatAttention` (RoPE + KV cache), the Spherical-YAT-Performer (`create_yat_tp_projection` / `yat_tp_features` / `yat_tp_attention`), CIRCULAR/REFLECT/CAUSAL conv padding, a fused Metal-kernel YAT score (`fused_yat_score`, `is_gpu_available`), DropConnect + `weight_normalized` conv fast path, and the `softermax` / `softer_sigmoid` / `soft_tanh` squashers. Install with `nmn[mlx]`. MLX is wired into the cross-framework parity matrix and ships an MNIST example + framework guide.
- **`learnable_epsilon` parameter** added to **every `YatNMN` and YAT conv layer in every framework**. When `True`, epsilon becomes a trainable parameter passed through softplus to guarantee strict positivity (it can never be zero or negative). The raw parameter is initialized via inverse softplus so that `softplus(param) ≈ epsilon`.
  - **Flax NNX**: `YatNMN`, `YatConv`, `YatConvTranspose`, `Embed`, `MultiHeadAttention`, `RotaryYatAttention`. The fused kernel path in NNX `YatNMN` is automatically disabled when epsilon is learnable to allow gradient flow.
  - **PyTorch**: `YatNMN`, `YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`.
  - **Keras**: `YatNMN`, `YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`.
  - **Flax Linen**: `YatNMN`, `YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`.
  - **TensorFlow**: `YatNMN`, `YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`.

- **`constant_bias` parameter** added to **every `YatNMN` and YAT conv layer in every framework**. Pass a `float` to use a fixed (non-learnable) bias constant inside the numerator square. Useful for ablations or when bias should not be trained. Same coverage as `learnable_epsilon` above.

- **`tie_kernel_bank` (+ `kernel_bank_size`, `kernel_bank_id`) and `use_dropconnect` (+ `drop_rate`)** added to all 6 Keras conv layers (`YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`). Brings Keras conv to full feature parity with PyTorch and NNX — previously Keras conv had none of these optional knobs. Tied kernel banks use class-level registries guarded by a threading lock, auto-expand when a later layer needs more filters, and support the same multi-bank namespacing via `kernel_bank_id`.

- **`spherical` and `weight_normalized` parameters** added to Flax Linen `YatNMN` and TensorFlow `YatNMN`. These were previously only available in NNX, PyTorch, and Keras. Brings Linen and TF to feature parity for these knobs. Linen and TF kernels use a row-based layout `(features, input_dim)` so each row is one neuron — normalization runs along `axis=-1` (the input dim), making each neuron a unit vector.

- **`weight_normalized` parameter** added to all Keras conv layers (`YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`). Previously only Keras `YatNMN` and `YatEmbed` supported it. Forward conv normalizes along all axes except the last (filter axis); transpose conv normalizes along all axes except `len(kernel_size)` (the transpose-conv filter axis).

- **Cross-framework numerical parity test** at `tests/integration/test_yat_nmn_parity.py`. Runs a 24-combination matrix of `(spherical, weight_normalized, learnable_epsilon, bias_mode)` against every installed framework's `YatNMN` with identical kernel/bias/inputs and asserts all outputs match a pure-numpy reference implementation to `rtol=2e-4`. This exact test would have caught the Keras `YatNMN.spherical` axis bug, the Keras `weight_normalized` axis bug, and the Keras conv bias-after-squaring bugs fixed earlier in this release. Frameworks that aren't installed in the test environment are gracefully skipped.

### Fixed
- **Critical math bugs in Keras `YatConv2D`, `YatConv3D`, `YatConvTranspose1D`, `YatConvTranspose2D`, `YatConvTranspose3D`**:
  - **Bias was added AFTER the squaring** instead of inside the numerator — `outputs = dot ** 2 / dist; outputs = outputs + bias` instead of the correct `(dot + bias) ** 2 / dist`. This made Keras conv outputs disagree with every other framework whenever `use_bias=True`.
  - **Alpha scaling used a complex exponent formula** `(sqrt(filters) / log1p(filters)) ** alpha` instead of the simple multiplicative `output * alpha` used by every other framework and by Keras `YatNMN` and `YatConv1D`. Both bugs are now fixed.
- **Keras `YatNMN.weight_normalized` normalized the wrong axis** — it normalized rows (`axis=-1`) instead of columns (`axis=0`). Since Keras `YatNMN` lays neurons out as columns (kernel shape `(input_dim, units)`), the weight-normalization optimization that assumes `||W_col||² = 1` was silently broken whenever `input_dim ≠ units`. Now uses `axis=0`. (Same root cause as the spherical-mode bug fixed earlier this release.)
- **PyTorch `YatNMN` docstring** corrected — said "normalize each neuron (column)" but PyTorch kernels use shape `(out_features, in_features)` where rows are neurons. Now documents the row-based convention correctly.

---

## [0.2.11] — 2026-03-17

### Added
- **Migration Guide** — new Docusaurus doc page covering drop-in replacement patterns for all 5 frameworks (Flax NNX, PyTorch, Keras, TensorFlow, Flax Linen)
- **Theoretical explainer** — "Why YAT Works: Geometric Intuition" section in docs/intro, covering alignment×proximity interpretation, Mercer kernel connection, bounded response proof, and activation-free non-linearity explanation
- **`YatConvTranspose3D` for Keras** — previously missing; now parity with TF/Linen
- **`YatConvTranspose1D/2D/3D` for Flax Linen** — full transpose conv support via `lax.conv_transpose`
- **`spherical` and `weight_normalized` parameters for Keras `YatNMN`** — normalize inputs/kernel to unit sphere; per-row kernel normalization
- **Module-level docstring** for `src/nmn/tf/nmn.py`
- **Expanded `YatNMN` class docstring** in `tf/nmn.py` — documents all parameters (`use_alpha`, `constant_alpha`, `positive_init`) with usage example
- **`build`/`__call__` docstrings** for all three `YatConvTranspose*D` classes in `tf/conv.py` (were missing)

### Changed
- **Naming convention unified to `2D` (uppercase D)** across PyTorch — `YatConv1d/2d/3d` → `YatConv1D/2D/3D`, `YatConvTranspose1d/2d/3d` → `YatConvTranspose1D/2D/3D`. Lowercase aliases kept for backward compatibility in Keras/TF/Linen.
- **`print()` → `logging.getLogger(__name__)`** in all production code paths (`torch/layers/yat_conv*.py`, `torch/nmn/yat_nmn.py`, `nnx/layers/conv/yat_conv.py`)
- **Flax NNX `.value` reads migrated** — deprecated `variable.value` read syntax replaced with `variable[...]` on all known `nnx.Param` attributes (`kernel`, `bias`, `alpha`, `embedding`, `cached_key`, `cached_value`, `cache_index`, `freqs_cos`, `freqs_sin`, `perf_projections`, `perf_scales`). Write assignments left intact (still valid in Flax 0.12.5).
- **JAX/Flax version constraints updated** in `pyproject.toml` — `jax>=0.9.1`, `jaxlib>=0.9.1`, `flax>=0.12.5` (was `jax>=0.4.0`, `flax>=0.7.0`)
- **JAX/Flax pinned in CI** — `test.yml` now installs `jax==0.9.1 jaxlib==0.9.1 flax==0.12.5` (was unpinned `jax jaxlib flax`)
- **Repository URLs updated** — all `mlnomadpy/nmn` references in CI, Docusaurus config, `pyproject.toml`, `installation.md`, `authors.yml`, `index.js`, and static website files updated to `azettaai/nmn`. Historical attribution in README acknowledgments preserved.
- **Docusaurus `organizationName` and `url`** updated to `azettaai` / `https://azettaai.github.io`

### Fixed
- **`test_yat_formula_with_bias`** — test was validating the old `output + bias` formula. Updated to validate the correct `(x·W + b)² / dist` formula (bias inside the numerator square).
- **`test_yat_formula_with_alpha_scaling`** — test was checking a channel-dependent scaling formula. Updated to validate simple multiplicative alpha: `output * alpha == output_no_alpha * alpha`.
- **`test_yat_nmn_custom_dtype` / `test_yat_nmn_module_dtype`** — tests incorrectly assumed `dtype=float64` sets weight storage dtype. Fixed to use `param_dtype=float64` for weight storage; `dtype` is compute-only.
- **`test_alpha_affects_output`** — test expected `constant_alpha=1.0` to differ from `use_alpha=False`. Fixed: multiplying by 1.0 is identity, so they are equal. Test now asserts equality for α=1.0 and difference for α=2.0.
- **`TypeError: 'MultiHeadAttention' object is not subscriptable`** — overly broad regex replacement of `.value` → `[...]` was incorrectly rewriting `self.value(inputs_v)` (a sub-module method call). Fixed by scoping replacement to named Param attributes only.
- **Version mismatch** — `src/nmn/__init__.py` version `0.2.6` synced to `0.2.11` to match `pyproject.toml`.

---

## [0.2.10] — 2026-01-07

### Added
- Flax NNX support — `nmn.nnx` module with `YatNMN`, `YatConv`, `YatConvTranspose`, attention layers, RNN cells
- Docusaurus documentation site at `website/docusaurus/`
- `RotaryYatAttention` with Performer (linear attention) mode
- `YatLSTMCell`, `YatGRUCell`, `RNN`, `Bidirectional` recurrent modules (NNX)
- `YatEmbed` positional embedding layer (NNX)
- DropConnect regularization in NNX conv layers

### Changed
- Moved from `setup.py` to `pyproject.toml` / Hatch build system
- Reorganized package into `src/nmn/{nnx,linen,torch,keras,tf}` layout

---

## [0.2.0] — 2025-06-01

### Added
- Initial PyTorch implementation: `YatNMN`, `YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`
- Keras/TensorFlow implementations
- Flax Linen implementation
- Kernel bank (weight tying) feature for PyTorch conv layers
- DropConnect support for PyTorch conv layers
- `alpha` scaling parameter (learnable or constant)
- `spherical` mode (unit-sphere normalization)
- `weight_normalized` mode (per-filter kernel normalization)
