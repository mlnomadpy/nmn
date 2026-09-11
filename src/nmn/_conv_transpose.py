"""Framework-independent transposed-convolution shape helpers."""

from __future__ import annotations

from collections.abc import Sequence


def _tuple(value: int | Sequence[int], rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (value,) * rank
    else:
        result = tuple(int(item) for item in value)
    if len(result) != rank:
        raise ValueError(f"{name} must have {rank} values, got {result}")
    return result


def canonical_transpose_config(
    kernel_size: int | Sequence[int],
    strides: int | Sequence[int],
    padding: str,
    dilation_rate: int | Sequence[int] = 1,
    output_padding: int | Sequence[int] = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str, tuple[int, ...]]:
    """Normalize and validate the canonical NMN transpose shape arguments."""
    rank = 1 if isinstance(kernel_size, int) else len(tuple(kernel_size))
    kernel = _tuple(kernel_size, rank, "kernel_size")
    stride = _tuple(strides, rank, "strides")
    dilation = _tuple(dilation_rate, rank, "dilation_rate")
    extra = _tuple(output_padding, rank, "output_padding")
    mode = padding.upper()
    if mode not in {"SAME", "VALID"}:
        raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
    for name, values in (
        ("kernel_size", kernel),
        ("strides", stride),
        ("dilation_rate", dilation),
    ):
        if any(value <= 0 for value in values):
            raise ValueError(f"{name} values must be positive, got {values}")
    if any(value < 0 for value in extra):
        raise ValueError(f"output_padding values must be nonnegative, got {extra}")
    if any(value >= step for value, step in zip(extra, stride)):
        raise ValueError(
            "output_padding values must be smaller than the corresponding "
            f"stride, got output_padding={extra}, strides={stride}"
        )
    return kernel, stride, dilation, mode, extra


def canonical_transpose_output_spatial(
    input_spatial: Sequence[int],
    kernel_size: int | Sequence[int],
    strides: int | Sequence[int],
    padding: str,
    dilation_rate: int | Sequence[int] = 1,
    output_padding: int | Sequence[int] = 0,
) -> tuple[int, ...]:
    """Return spatial sizes under the canonical NMN transpose contract."""
    kernel, stride, dilation, mode, extra = canonical_transpose_config(
        kernel_size, strides, padding, dilation_rate, output_padding
    )
    spatial = tuple(int(value) for value in input_spatial)
    if len(spatial) != len(kernel):
        raise ValueError(f"input_spatial must have {len(kernel)} values, got {spatial}")
    effective = tuple(d * (k - 1) + 1 for k, d in zip(kernel, dilation))
    if mode == "SAME":
        return tuple(
            size * step + out for size, step, out in zip(spatial, stride, extra)
        )
    return tuple(
        (size - 1) * step + width + out
        for size, step, width, out in zip(spatial, stride, effective, extra)
    )


def canonical_jax_transpose_padding(
    kernel_size: int | Sequence[int],
    strides: int | Sequence[int],
    padding: str,
    dilation_rate: int | Sequence[int] = 1,
    output_padding: int | Sequence[int] = 0,
) -> tuple[tuple[int, int], ...]:
    """Return explicit JAX padding pairs implementing the NMN contract."""
    kernel, stride, dilation, mode, extra = canonical_transpose_config(
        kernel_size, strides, padding, dilation_rate, output_padding
    )
    effective = tuple(d * (k - 1) + 1 for k, d in zip(kernel, dilation))
    pairs = []
    for width, step, out in zip(effective, stride, extra):
        if mode == "VALID":
            low = width - 1
            high = width - 1 + out
        else:
            total = width + step - 2
            low = width - 1 if step > width - 1 else (total + 1) // 2
            high = total - low + out
        pairs.append((low, high))
    return tuple(pairs)


def canonical_same_crop_or_pad(
    kernel_size: int | Sequence[int],
    strides: int | Sequence[int],
    dilation_rate: int | Sequence[int] = 1,
    output_padding: int | Sequence[int] = 0,
) -> tuple[tuple[int, int], ...]:
    """Return low/high crops (negative values are pads) from canonical VALID.

    This lets backends without explicit asymmetric transpose padding implement
    canonical ``SAME`` by evaluating ``VALID`` and then adjusting its borders.
    """
    kernel, _, dilation, _, _ = canonical_transpose_config(
        kernel_size, strides, "SAME", dilation_rate, output_padding
    )
    same_padding = canonical_jax_transpose_padding(
        kernel_size, strides, "SAME", dilation_rate, output_padding
    )
    effective = tuple(d * (k - 1) + 1 for k, d in zip(kernel, dilation))
    return tuple(
        (width - 1 - low, width - 1 - high)
        for width, (low, high) in zip(effective, same_padding)
    )
