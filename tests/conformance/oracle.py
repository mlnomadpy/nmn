"""NumPy float64 semantic oracles and canonical fixtures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenseCase:
    """Canonical backend-independent dense YAT operands."""

    inputs: np.ndarray
    kernel: np.ndarray
    bias: np.ndarray
    alpha: np.ndarray
    epsilon: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class DenseConfiguration:
    """Optional dense semantics retained from the legacy parity matrix."""

    spherical: bool = False
    weight_normalized: bool = False
    learnable_epsilon: bool = True
    bias_mode: str = "learnable"

    def __post_init__(self) -> None:
        if self.bias_mode not in {"learnable", "constant", "none"}:
            raise ValueError(f"unsupported dense bias mode: {self.bias_mode!r}")

    @property
    def use_bias(self) -> bool:
        return self.bias_mode != "none"

    @property
    def constant_bias(self) -> float | None:
        return 0.42 if self.bias_mode == "constant" else None


@dataclass(frozen=True)
class OracleResult:
    """A forward result and gradients of its cotangent projection."""

    output: np.ndarray
    gradients: dict[str, np.ndarray]


@dataclass(frozen=True)
class EmbeddingCase:
    """Canonical integer lookup operands for an embedding layer."""

    indices: np.ndarray
    embedding: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class EmbeddingAttendCase:
    """Canonical YAT embedding-attend operands with static epsilon."""

    query: np.ndarray
    embedding: np.ndarray
    alpha: np.ndarray
    epsilon: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class ConvolutionCase:
    """Canonical channels-last 1D YAT convolution operands.

    ``kernel`` is always ``[kernel_width, input_channels, output_channels]``.
    That is the native JAX layout; adapters explicitly transpose it for Torch.
    """

    inputs: np.ndarray
    kernel: np.ndarray
    bias: np.ndarray
    alpha: np.ndarray
    epsilon: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class AttentionCase:
    """Canonical rank-four YAT attention operands and boolean mask."""

    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    mask: np.ndarray
    alpha: np.ndarray
    epsilon: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class AttentionResult:
    """Canonical attention weights, readout, and scalar-loss gradients."""

    weights: np.ndarray
    output: np.ndarray
    gradients: dict[str, np.ndarray]


@dataclass(frozen=True)
class LinearAttentionCase:
    """Fixed-projection operands for MAY/RAY linear attention.

    Projections are deliberately supplied as canonical arrays rather than
    sampled through backend RNGs: JAX and NumPy sampling algorithms are not a
    cross-framework semantic contract.  They are frozen kernels, so the
    conformance VJP covers the actual differentiable inputs (q/k/v), not the
    projection bookkeeping or construction-time bias/epsilon.
    """

    kind: str
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    projection: dict[str, np.ndarray | float | int]
    causal: bool
    key_padding_mask: np.ndarray | None
    epsilon: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class LinearAttentionResult:
    """MAY/RAY feature maps, readout, and q/k/v VJPs."""

    query_features: np.ndarray
    key_features: np.ndarray
    output: np.ndarray
    gradients: dict[str, np.ndarray]


def canonical_dense_case(seed: int = 2026) -> DenseCase:
    """Return deterministic, non-degenerate float64 conformance operands."""
    rng = np.random.default_rng(seed)
    return DenseCase(
        inputs=rng.normal(size=(3, 5)),
        kernel=rng.normal(size=(5, 4)),
        bias=rng.normal(size=(4,)),
        alpha=np.asarray(0.8),
        epsilon=np.asarray(2e-3),
        cotangent=rng.normal(size=(3, 4)),
    )


def canonical_embedding_case(seed: int = 2026) -> EmbeddingCase:
    """Return a lookup case with a repeated index to exercise accumulation."""
    rng = np.random.default_rng(seed + 2)
    indices = np.asarray([[0, 3], [3, 1]], dtype=np.int32)
    embedding = rng.normal(scale=0.35, size=(5, 3))
    return EmbeddingCase(
        indices=indices,
        embedding=embedding,
        cotangent=rng.normal(scale=0.4, size=indices.shape + (embedding.shape[1],)),
    )


def canonical_embedding_attend_case(seed: int = 2026) -> EmbeddingAttendCase:
    """Return a non-degenerate YAT embedding-attend case."""
    rng = np.random.default_rng(seed + 5)
    embedding = rng.normal(scale=0.35, size=(5, 3))
    query = rng.normal(scale=0.3, size=(2, 3))
    return EmbeddingAttendCase(
        query=query,
        embedding=embedding,
        alpha=np.asarray(0.7),
        epsilon=np.asarray(2e-2),
        cotangent=rng.normal(scale=0.4, size=(2, 5)),
    )


def canonical_convolution_case(seed: int = 2026) -> ConvolutionCase:
    """Return a small VALID 1D YAT convolution case in canonical layout."""
    rng = np.random.default_rng(seed + 3)
    inputs = rng.normal(scale=0.35, size=(1, 4, 2))
    kernel = rng.normal(scale=0.3, size=(2, 2, 2))
    return ConvolutionCase(
        inputs=inputs,
        kernel=kernel,
        bias=rng.normal(scale=0.15, size=(2,)),
        alpha=np.asarray(0.7),
        epsilon=np.asarray(2e-2),
        cotangent=rng.normal(scale=0.4, size=(1, 3, 2)),
    )


def canonical_transpose_convolution_case(seed: int = 2026) -> ConvolutionCase:
    """Return a small VALID 1D transposed-YAT convolution case."""
    case = canonical_convolution_case(seed)
    # VALID stride-one transpose convolution maps length four to length five.
    return ConvolutionCase(
        inputs=case.inputs,
        kernel=case.kernel,
        bias=case.bias,
        alpha=case.alpha,
        epsilon=case.epsilon,
        cotangent=np.random.default_rng(seed + 4).normal(scale=0.4, size=(1, 5, 2)),
    )


def canonical_attention_case(seed: int = 2026) -> AttentionCase:
    """Return a deterministic masked attention case with one all-masked row."""
    rng = np.random.default_rng(seed + 1)
    return AttentionCase(
        query=rng.normal(scale=0.4, size=(1, 2, 1, 3)),
        key=rng.normal(scale=0.4, size=(1, 3, 1, 3)),
        value=rng.normal(scale=0.4, size=(1, 3, 1, 2)),
        mask=np.asarray([[[[True, False, True], [False, False, False]]]]),
        alpha=np.asarray(0.7),
        epsilon=np.asarray(2e-3),
        cotangent=rng.normal(size=(1, 2, 1, 2)),
    )


def canonical_linear_attention_case(
    kind: str, seed: int = 2026, *, key_padding: bool = False
) -> LinearAttentionCase:
    """Return deterministic MAY/RAY operands with a frozen canonical kernel."""
    if kind not in {"may", "ray"}:
        raise ValueError(f"unknown linear-attention feature map: {kind}")
    rng = np.random.default_rng(seed + (11 if kind == "may" else 12))
    query = rng.normal(scale=0.35, size=(1, 3, 1, 2))
    key = rng.normal(scale=0.35, size=(1, 3, 1, 2))
    value = rng.normal(scale=0.3, size=(1, 3, 1, 2))
    if kind == "may":
        # Canonical [features, degree slots, head dim] fixture.  The modest
        # degrees keep values and finite-difference VJPs well-conditioned.
        projection: dict[str, np.ndarray | float | int] = {
            "omegas": np.asarray(
                [
                    [[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                    [[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]],
                    [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
                    [[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
                ]
            ),
            "degrees": np.asarray([0, 1, 2, 3, 1, 2], dtype=np.int32),
            "Z": float(3.1),
            "num_features": 6,
            "nmax": 3,
            "head_dim": 2,
        }
    else:
        projection = {
            "W1": np.asarray([[0.35, -0.2, 0.45], [-0.25, 0.4, 0.15]]),
            "W2": np.asarray([[-0.3, 0.5, 0.25], [0.4, 0.1, -0.35]]),
            "omegas": np.asarray(
                [
                    [[0.25, -0.4, 0.2], [-0.15, 0.3, -0.25]],
                    [[0.35, 0.1, -0.2], [-0.2, -0.3, 0.4]],
                ]
            ),
            "phases": np.asarray([[0.2, 1.1], [2.0, 0.7]]),
            "coef": np.asarray([0.65, 0.35]),
            "bias": float(0.6),
            "sketch_m": 2,
            "num_radial": 2,
            "radial_dim": 2,
            "head_dim": 2,
        }
    return LinearAttentionCase(
        kind=kind,
        query=query,
        key=key,
        value=value,
        projection=projection,
        causal=True,
        key_padding_mask=(
            np.asarray([[[[True, False, True]]]]) if key_padding else None
        ),
        # Keep the public/readout epsilon equal to the shared normalization
        # epsilon. TF currently uses one argument for both; all other backends
        # use 1e-6 internally for normalization.
        epsilon=np.asarray(1e-6),
        cotangent=rng.normal(scale=0.35, size=(1, 3, 1, 2)),
    )


def _attention_forward(case: AttentionCase) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(case.query, dtype=np.float64)
    key = np.asarray(case.key, dtype=np.float64)
    value = np.asarray(case.value, dtype=np.float64)
    dot = np.einsum("bqhd,bkhd->bhqk", query, key)
    query_norm = np.sum(query * query, axis=-1).transpose(0, 2, 1)[..., None]
    key_norm = np.sum(key * key, axis=-1).transpose(0, 2, 1)[:, :, None, :]
    distance = np.maximum(query_norm + key_norm - 2.0 * dot, 0.0)
    scores = dot * dot / ((distance + case.epsilon) * np.sqrt(query.shape[-1]))
    scores = scores * case.alpha
    mask = np.broadcast_to(np.asarray(case.mask, dtype=bool), scores.shape)
    row_has_key = np.any(mask, axis=-1, keepdims=True)
    masked = np.where(mask, scores, -np.inf)
    masked = np.where(row_has_key, masked, 0.0)
    maximum = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.exp(masked - maximum)
    weights = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    weights = np.where(mask, weights, 0.0)
    output = np.einsum("bhqk,bkhd->bqhd", weights, value)
    return weights, output


def _central_difference(
    case: AttentionCase, field: str, cotangent: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    source = np.asarray(getattr(case, field), dtype=np.float64)
    gradient = np.empty_like(source)
    for index in np.ndindex(source.shape):
        plus = source.copy()
        minus = source.copy()
        plus[index] += step
        minus[index] -= step
        plus_case = AttentionCase(
            **{**case.__dict__, field: plus}  # type: ignore[arg-type]
        )
        minus_case = AttentionCase(
            **{**case.__dict__, field: minus}  # type: ignore[arg-type]
        )
        plus_loss = np.sum(_attention_forward(plus_case)[1] * cotangent)
        minus_loss = np.sum(_attention_forward(minus_case)[1] * cotangent)
        gradient[index] = (plus_loss - minus_loss) / (2.0 * step)
    return gradient


def yat_attention(case: AttentionCase) -> AttentionResult:
    """Evaluate masked YAT softmax attention and reference VJPs."""
    weights, output = _attention_forward(case)
    cotangent = np.asarray(case.cotangent, dtype=np.float64)
    gradients = {
        field: _central_difference(case, field, cotangent)
        for field in ("query", "key", "value", "alpha", "epsilon")
    }
    return AttentionResult(weights, output, gradients)


def _linear_features(case: LinearAttentionCase, values: np.ndarray) -> np.ndarray:
    """Evaluate the fixed MAY or RAY feature map in float64."""
    x = np.asarray(values, dtype=np.float64)
    x = x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-6)
    p = case.projection
    if case.kind == "may":
        omegas = np.asarray(p["omegas"], dtype=np.float64)
        degrees = np.asarray(p["degrees"], dtype=np.intp)
        projections = np.einsum("...d,mld->...ml", x, omegas)
        slots = np.arange(omegas.shape[1])[None, :]
        projections = np.where(slots < degrees[:, None], projections, 1.0)
        return np.sqrt(float(p["Z"]) / int(p["num_features"])) * np.prod(
            projections, axis=-1
        )

    w1 = np.asarray(p["W1"], dtype=np.float64)
    w2 = np.asarray(p["W2"], dtype=np.float64)
    omegas = np.asarray(p["omegas"], dtype=np.float64)
    phases = np.asarray(p["phases"], dtype=np.float64)
    coef = np.asarray(p["coef"], dtype=np.float64)
    bias = float(p["bias"])
    z = np.concatenate([x, np.full(x.shape[:-1] + (1,), np.sqrt(bias))], axis=-1)
    sketch_m = int(p["sketch_m"])
    radial_dim = int(p["radial_dim"])
    psi = np.einsum("...d,md->...m", z, w1) * np.einsum("...d,md->...m", z, w2)
    psi /= np.sqrt(sketch_m)
    rff = np.sqrt(2.0 / radial_dim) * np.cos(
        np.einsum("...d,jrd->...jr", z, omegas) + phases
    )
    rff *= np.sqrt(np.maximum(coef, 0.0))[..., :, None]
    radial = rff.reshape(rff.shape[:-2] + (-1,))
    return (psi[..., :, None] * radial[..., None, :]).reshape(x.shape[:-1] + (-1,))


def _linear_readout(
    query_features: np.ndarray,
    key_features: np.ndarray,
    value: np.ndarray,
    *,
    causal: bool,
    epsilon: float,
    key_padding_mask: np.ndarray | None,
) -> np.ndarray:
    """Shared, positive-denominator linear-attention readout reference."""
    keys = np.asarray(key_features, dtype=np.float64)
    if key_padding_mask is not None:
        # Canonical masks are [..., heads, 1, keys], exactly NNX's supported
        # factorable key-padding form.
        mask = np.asarray(key_padding_mask, dtype=bool)
        key_mask = np.swapaxes(np.squeeze(mask, axis=-2), -1, -2)[..., None]
        keys = np.where(key_mask, keys, 0.0)
    values = np.asarray(value, dtype=np.float64)
    if causal:
        kv = np.einsum("bkhf,bkhd->bkhfd", keys, values)
        kv = np.cumsum(kv, axis=1)
        key_sum = np.cumsum(keys, axis=1)
        numerator = np.einsum("bqhf,bqhfd->bqhd", query_features, kv)
        denominator = np.einsum("bqhf,bqhf->bqh", query_features, key_sum)
    else:
        kv = np.einsum("bkhf,bkhd->bhfd", keys, values)
        numerator = np.einsum("bqhf,bhfd->bqhd", query_features, kv)
        denominator = np.einsum("bqhf,bhf->bqh", query_features, np.sum(keys, axis=1))
    # This fixture deliberately has strictly positive denominators, making the
    # common semantics identical to NNX's sign-preserving stabilization.
    if np.any(denominator <= 0.0):
        raise AssertionError("canonical linear-attention denominator must be positive")
    return numerator / (denominator[..., None] + epsilon)


def linear_yat_attention(case: LinearAttentionCase) -> LinearAttentionResult:
    """Float64 MAY/RAY oracle for fixed projections and q/k/v gradients."""
    query_features = _linear_features(case, case.query)
    key_features = _linear_features(case, case.key)

    def forward(updated: LinearAttentionCase) -> np.ndarray:
        return _linear_readout(
            _linear_features(updated, updated.query),
            _linear_features(updated, updated.key),
            updated.value,
            causal=updated.causal,
            epsilon=float(updated.epsilon),
            key_padding_mask=updated.key_padding_mask,
        )

    output = forward(case)
    cotangent = np.asarray(case.cotangent, dtype=np.float64)
    gradients: dict[str, np.ndarray] = {}
    for field in ("query", "key", "value"):
        source = np.asarray(getattr(case, field), dtype=np.float64)
        gradient = np.empty_like(source)
        for index in np.ndindex(source.shape):
            plus, minus = source.copy(), source.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            gradients[field] = gradient
            gradient[index] = (
                np.sum(forward(replace(case, **{field: plus})) * cotangent)
                - np.sum(forward(replace(case, **{field: minus})) * cotangent)
            ) / 2e-5
    return LinearAttentionResult(query_features, key_features, output, gradients)


def yat_dense(case: DenseCase) -> OracleResult:
    """Evaluate canonical YAT dense semantics and all relevant gradients.

    The logical kernel layout is always ``[input_features, output_features]``.
    Bias is inside the squared affine term, distance is clamped at zero, and
    alpha multiplies the ratio.
    """
    x = np.asarray(case.inputs, dtype=np.float64)
    kernel = np.asarray(case.kernel, dtype=np.float64)
    bias = np.asarray(case.bias, dtype=np.float64)
    alpha = np.asarray(case.alpha, dtype=np.float64)
    epsilon = np.asarray(case.epsilon, dtype=np.float64)
    cotangent = np.asarray(case.cotangent, dtype=np.float64)

    dot = x @ kernel
    raw_distance = (
        np.sum(x * x, axis=-1, keepdims=True)
        + np.sum(kernel * kernel, axis=0, keepdims=True)
        - 2.0 * dot
    )
    distance = np.maximum(raw_distance, 0.0)
    affine = dot + bias
    denominator = distance + epsilon
    ratio = affine * affine / denominator
    output = ratio * alpha

    affine_bar = cotangent * alpha * 2.0 * affine / denominator
    distance_bar = -cotangent * alpha * affine * affine / (denominator * denominator)
    distance_bar = distance_bar * (raw_distance > 0.0)
    input_gradient = affine_bar @ kernel.T + np.sum(
        2.0 * distance_bar[..., None] * (x[:, None, :] - kernel.T[None, ...]),
        axis=1,
    )
    kernel_gradient = (
        x.T @ affine_bar
        + np.sum(
            2.0 * distance_bar[..., None] * (kernel.T[None, ...] - x[:, None, :]),
            axis=0,
        ).T
    )
    gradients = {
        "input": input_gradient,
        "kernel": kernel_gradient,
        "bias": np.sum(affine_bar, axis=0),
        "alpha": np.asarray(np.sum(cotangent * ratio)),
        "epsilon": np.asarray(
            np.sum(-cotangent * alpha * affine * affine / (denominator * denominator))
        ),
    }
    return OracleResult(output=output, gradients=gradients)


def yat_dense_configured(
    case: DenseCase, configuration: DenseConfiguration
) -> np.ndarray:
    """Evaluate the forward semantics of every supported dense configuration."""
    inputs = np.asarray(case.inputs, dtype=np.float64)
    kernel = np.asarray(case.kernel, dtype=np.float64)
    if configuration.spherical:
        inputs = inputs / (np.linalg.norm(inputs, axis=-1, keepdims=True) + 1e-8)
        kernel = kernel / (np.linalg.norm(kernel, axis=0, keepdims=True) + 1e-8)
    if configuration.weight_normalized:
        kernel = kernel / (np.linalg.norm(kernel, axis=0, keepdims=True) + 1e-8)

    dot = inputs @ kernel
    if configuration.spherical:
        distance = np.maximum(2.0 - 2.0 * dot, 0.0)
    else:
        kernel_norm = (
            np.ones((1, kernel.shape[-1]), dtype=kernel.dtype)
            if configuration.weight_normalized
            else np.sum(kernel * kernel, axis=0, keepdims=True)
        )
        distance = np.maximum(
            np.sum(inputs * inputs, axis=-1, keepdims=True) + kernel_norm - 2.0 * dot,
            0.0,
        )

    if configuration.bias_mode == "learnable":
        dot = dot + np.asarray(case.bias, dtype=np.float64)
    elif configuration.bias_mode == "constant":
        dot = dot + configuration.constant_bias
    return dot * dot / (distance + case.epsilon) * case.alpha


def yat_embedding(case: EmbeddingCase) -> OracleResult:
    """Evaluate canonical embedding lookup and its embedding VJP."""
    indices = np.asarray(case.indices, dtype=np.intp)
    embedding = np.asarray(case.embedding, dtype=np.float64)
    output = embedding[indices]
    gradient = np.zeros_like(embedding)
    np.add.at(gradient, indices, np.asarray(case.cotangent, dtype=np.float64))
    return OracleResult(output=output, gradients={"embedding": gradient})


def _embedding_attend_forward(case: EmbeddingAttendCase) -> np.ndarray:
    query = np.asarray(case.query, dtype=np.float64)
    embedding = np.asarray(case.embedding, dtype=np.float64)
    dot = query @ embedding.T
    distance = np.maximum(
        np.sum(query * query, axis=-1, keepdims=True)
        + np.sum(embedding * embedding, axis=-1)[None, :]
        - 2.0 * dot,
        0.0,
    )
    return dot * dot / (distance + case.epsilon) * case.alpha


def yat_embedding_attend(case: EmbeddingAttendCase) -> OracleResult:
    """Evaluate YAT embedding attend and query/parameter VJPs in float64."""
    output = _embedding_attend_forward(case)
    cotangent = np.asarray(case.cotangent, dtype=np.float64)

    def loss(updated: EmbeddingAttendCase) -> float:
        return float(np.sum(_embedding_attend_forward(updated) * cotangent))

    gradients: dict[str, np.ndarray] = {}
    for field, gradient_name in (
        ("query", "query"),
        ("embedding", "embedding"),
        ("alpha", "alpha"),
    ):
        source = np.asarray(getattr(case, field), dtype=np.float64)
        gradient = np.empty_like(source)
        for index in np.ndindex(source.shape):
            plus = source.copy()
            minus = source.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            gradient[index] = (
                loss(replace(case, **{field: plus}))
                - loss(replace(case, **{field: minus}))
            ) / 2e-5
        gradients[gradient_name] = gradient
    return OracleResult(output=output, gradients=gradients)


def _conv1d_forward(case: ConvolutionCase, *, transpose: bool) -> np.ndarray:
    """Reference VALID stride-one 1D convolution in channels-last layout."""
    inputs = np.asarray(case.inputs, dtype=np.float64)
    kernel = np.asarray(case.kernel, dtype=np.float64)
    bias = np.asarray(case.bias, dtype=np.float64)
    alpha = np.asarray(case.alpha, dtype=np.float64)
    epsilon = np.asarray(case.epsilon, dtype=np.float64)
    batch, input_length, input_channels = inputs.shape
    kernel_width, kernel_channels, output_channels = kernel.shape
    if input_channels != kernel_channels:
        raise ValueError("canonical convolution channel mismatch")
    output_length = (
        input_length + kernel_width - 1
        if transpose
        else input_length - kernel_width + 1
    )
    dot = np.zeros((batch, output_length, output_channels), dtype=np.float64)
    patch_norm = np.zeros_like(dot)
    if transpose:
        for position in range(input_length):
            for offset in range(kernel_width):
                # ``lax.conv_transpose`` (and Linen's wrapper) defines the
                # transposed operator relative to cross-correlation, so the
                # canonical KWIO kernel is spatially reversed on expansion.
                destination = position + kernel_width - 1 - offset
                for output_channel in range(output_channels):
                    dot[:, destination, output_channel] += np.sum(
                        inputs[:, position, :] * kernel[offset, :, output_channel],
                        axis=-1,
                    )
                patch_norm[:, destination, :] += np.sum(
                    inputs[:, position, :] ** 2, axis=-1, keepdims=True
                )
    else:
        for position in range(output_length):
            patch = inputs[:, position : position + kernel_width, :]
            dot[:, position, :] = np.einsum("bwk,wko->bo", patch, kernel)
            patch_norm[:, position, :] = np.sum(patch * patch, axis=(1, 2))[:, None]
    kernel_norm = np.sum(kernel * kernel, axis=(0, 1))[None, None, :]
    distance = np.maximum(patch_norm + kernel_norm - 2.0 * dot, 0.0)
    return (dot + bias[None, None, :]) ** 2 / (distance + epsilon) * alpha


def _convolution_result(case: ConvolutionCase, *, transpose: bool) -> OracleResult:
    output = _conv1d_forward(case, transpose=transpose)
    cotangent = np.asarray(case.cotangent, dtype=np.float64)
    if output.shape != cotangent.shape:
        raise ValueError(
            f"cotangent shape {cotangent.shape} does not match output {output.shape}"
        )

    def loss(updated: ConvolutionCase) -> float:
        return float(np.sum(_conv1d_forward(updated, transpose=transpose) * cotangent))

    gradients: dict[str, np.ndarray] = {}
    for field, gradient_name in (
        ("inputs", "input"),
        ("kernel", "kernel"),
        ("bias", "bias"),
        ("alpha", "alpha"),
        ("epsilon", "epsilon"),
    ):
        source = np.asarray(getattr(case, field), dtype=np.float64)
        gradient = np.empty_like(source)
        for index in np.ndindex(source.shape):
            plus = source.copy()
            minus = source.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            gradient[index] = (
                loss(replace(case, **{field: plus}))
                - loss(replace(case, **{field: minus}))
            ) / 2e-5
        gradients[gradient_name] = gradient
    return OracleResult(output=output, gradients=gradients)


def yat_conv1d(case: ConvolutionCase) -> OracleResult:
    """Evaluate a canonical, channels-last VALID 1D YAT convolution."""
    return _convolution_result(case, transpose=False)


def yat_conv_transpose1d(case: ConvolutionCase) -> OracleResult:
    """Evaluate a canonical, channels-last VALID 1D transposed YAT convolution."""
    return _convolution_result(case, transpose=True)


def write_dense_fixture(path: str | Path, case: DenseCase | None = None) -> None:
    """Serialize the canonical oracle fixture consumed by Apple MLX CI."""
    if case is None:
        case = canonical_dense_case()
    result = yat_dense(case)
    arrays = {
        "inputs": case.inputs,
        "kernel": case.kernel,
        "bias": case.bias,
        "alpha": case.alpha,
        "epsilon": case.epsilon,
        "cotangent": case.cotangent,
        "output": result.output,
    }
    arrays.update(
        {f"gradient_{name}": value for name, value in result.gradients.items()}
    )
    np.savez(path, **arrays)


def read_dense_fixture(path: str | Path) -> tuple[DenseCase, OracleResult]:
    """Load a canonical fixture without permitting pickled objects."""
    with np.load(path, allow_pickle=False) as fixture:
        case = DenseCase(
            inputs=fixture["inputs"],
            kernel=fixture["kernel"],
            bias=fixture["bias"],
            alpha=fixture["alpha"],
            epsilon=fixture["epsilon"],
            cotangent=fixture["cotangent"],
        )
        gradients = {
            name: fixture[f"gradient_{name}"]
            for name in ("input", "kernel", "bias", "alpha", "epsilon")
        }
        result = OracleResult(output=fixture["output"], gradients=gradients)
    return case, result
