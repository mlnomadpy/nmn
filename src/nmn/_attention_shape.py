"""Framework-neutral validation for functional attention tensor shapes."""

from __future__ import annotations

from typing import Any


def validate_attention_inputs(
    query: Any,
    key: Any,
    value: Any | None = None,
    *,
    exact_rank: int | None = None,
) -> None:
    """Validate ``[..., sequence, heads, depth]`` attention operands.

    The helper deliberately inspects only ``ndim`` and ``shape`` so all
    backends can share the same exception type, check ordering, and messages
    without importing another tensor framework.
    """
    operands: tuple[tuple[str, Any], ...] = (("query", query), ("key", key))
    if value is not None:
        operands += (("value", value),)

    ranks = {name: operand.ndim for name, operand in operands}
    if exact_rank is None:
        invalid = {name: rank for name, rank in ranks.items() if rank < 3}
        if invalid:
            details = ", ".join(f"{name}={rank}" for name, rank in ranks.items())
            raise ValueError(
                "attention inputs must have rank at least 3 "
                f"([..., sequence, heads, depth]); got {details}"
            )
    elif any(rank != exact_rank for rank in ranks.values()):
        details = ", ".join(f"{name}={rank}" for name, rank in ranks.items())
        raise ValueError(
            f"attention inputs must have rank {exact_rank} "
            f"([batch, sequence, heads, depth]); got {details}"
        )

    if len(set(ranks.values())) != 1:
        details = ", ".join(f"{name}={rank}" for name, rank in ranks.items())
        raise ValueError(f"attention inputs must have the same rank; got {details}")

    batch_shapes = {name: tuple(operand.shape[:-3]) for name, operand in operands}
    if len(set(batch_shapes.values())) != 1:
        details = ", ".join(f"{name}={shape}" for name, shape in batch_shapes.items())
        raise ValueError(
            f"attention input batch dimensions must match exactly; got {details}"
        )

    head_counts = {name: operand.shape[-2] for name, operand in operands}
    if len(set(head_counts.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in head_counts.items())
        raise ValueError(
            f"attention inputs must have the same number of heads; got {details}"
        )

    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key head depth (head_dim) must match; "
            f"got query={query.shape[-1]}, key={key.shape[-1]}"
        )

    if value is not None and key.shape[-3] != value.shape[-3]:
        raise ValueError(
            "key and value sequence lengths must match; "
            f"got key={key.shape[-3]}, value={value.shape[-3]}"
        )


__all__ = ["validate_attention_inputs"]
