# Changelog

All notable changes to NMN are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
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
