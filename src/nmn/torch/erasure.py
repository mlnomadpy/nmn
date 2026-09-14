"""Fit-only covariance-direction removal in a native finite kernel bank."""

import hashlib
import math
from pathlib import Path

import torch

from .graph import YatGraph
from .interpretable import ThreeNeuronYat, YatExpansion
from .research import _json_value, collect_research_data


def erasure_study(
    model,
    dataset,
    *,
    module_name,
    labels,
    provenance,
    rtol,
    fit_split="tuning",
    evaluation_split="validation",
):
    """Learn a Euclidean projection from fit cross-covariance, then freeze it.

    This is an explicitly defined finite-bank baseline, not LEACE, an RKHS
    projection, a minimax eraser, or a guarantee against nonlinear decoding.
    Evaluation labels are used only for reporting cross-covariance residuals.
    Targets can feed the existing native preimage and receiver-read execution APIs.
    """
    if not isinstance(model, (YatGraph, ThreeNeuronYat)):
        raise TypeError("erasure requires a native graph or three-neuron model")
    if module_name not in model.state_names:
        raise ValueError("unknown native module")
    block = (
        model.blocks[module_name]
        if isinstance(model, YatGraph)
        else getattr(model, module_name)
    )
    if type(block) is not YatExpansion:
        raise ValueError("erasure requires a strict YatExpansion bank")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("label provenance is required")
    if (
        isinstance(rtol, bool)
        or not isinstance(rtol, (int, float))
        or not math.isfinite(rtol)
        or not 0 <= rtol < 1
    ):
        raise ValueError("rtol must be finite in [0,1)")
    if not fit_split or not evaluation_split or fit_split == evaluation_split:
        raise ValueError("declare distinct fit and evaluation splits")
    fit_ids = dataset.sample_ids(split=fit_split)
    eval_ids = dataset.sample_ids(split=evaluation_split)
    if (
        len(fit_ids) < 2
        or not eval_ids
        or not isinstance(labels, dict)
        or set(labels) != set(fit_ids + eval_ids)
    ):
        raise ValueError(
            "labels must cover exactly at least two fit and nonempty evaluation samples"
        )
    parameter = block.centers
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("erasure requires float32 or float64")

    def tensor(values):
        return torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device)

    y = tensor([labels[s] for s in fit_ids + eval_ids])
    if y.ndim != 2 or y.shape[1] < 1 or not bool(torch.isfinite(y).all()):
        raise ValueError(
            "labels must be finite vectors of one consistent positive width"
        )

    def collect(ids):
        snap = collect_research_data(
            model,
            tensor([dataset.sample(s).inputs for s in ids]),
            sample_ids=ids,
            derivatives=False,
        )
        points = tensor(snap["observations"]["baseline_trace"][module_name + ".input"])
        with torch.no_grad():
            features = block._features(points)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("nonfinite kernel features")
        return snap, features

    fit_snapshot, x = collect(fit_ids)
    mean = x.mean(0)
    centered = x - mean
    yf = y[: len(fit_ids)]
    covariance = centered.T @ (yf - yf.mean(0)) / len(fit_ids)
    u, singular, _ = torch.linalg.svd(covariance, full_matrices=False)
    threshold = rtol * singular.max()
    rank = int((singular > threshold).sum())
    basis = u[:, :rank]
    projection = torch.eye(x.shape[1], dtype=x.dtype, device=x.device) - basis @ basis.T
    # The projection and centering offset are frozen before evaluation execution.
    evaluation_snapshot, xe = collect(eval_ids)

    def measure(features, target_labels):
        erased = (features - mean) @ projection + mean
        centered_labels = target_labels - target_labels.mean(0)
        before = (features - features.mean(0)).T @ centered_labels / len(features)
        after = (erased - erased.mean(0)).T @ centered_labels / len(features)
        return dict(
            features=features,
            projected_features=erased,
            labels=target_labels,
            covariance_before=before,
            covariance_after=after,
            covariance_before_norm=torch.linalg.vector_norm(before).item(),
            covariance_after_norm=torch.linalg.vector_norm(after).item(),
            distortion_mse=(erased - features).square().mean().item(),
        )

    fit = measure(x, yf)
    evaluation = measure(xe, y[len(fit_ids) :])
    if not all(
        bool(torch.isfinite(v).all())
        for v in (
            projection,
            singular,
            fit["projected_features"],
            evaluation["projected_features"],
        )
    ):
        raise ValueError("nonfinite fitted projection or targets")
    return _json_value(
        dict(
            schema="nmn.erasure-study.v1",
            status="observed",
            model_snapshot=fit_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            module=module_name,
            protocol=dict(
                provenance=provenance,
                rtol=rtol,
                fit_split=fit_split,
                evaluation_split=evaluation_split,
                fit_sample_ids=fit_ids,
                evaluation_sample_ids=eval_ids,
                frozen_before_evaluation=True,
                method="Center features at fit mean; remove left singular directions of fit feature-label cross-covariance with singular value > rtol*smax; restore fit mean",
            ),
            projection=dict(
                matrix=projection,
                mean=mean,
                singular_values=singular,
                threshold=threshold.item(),
                removed_rank=rank,
                idempotence_residual=torch.linalg.vector_norm(
                    projection @ projection - projection
                ).item(),
            ),
            fit=fit,
            evaluation=evaluation,
            targets=dict(zip(eval_ids, evaluation["projected_features"])),
            cost=dict(full_forwards=2, svd_calls=1),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Euclidean geometry of unweighted finite bank evaluations, not an intrinsic RKHS norm or source-faithful LEACE implementation.",
                "Projection targets may have no realizable native preimage; measured execution is required.",
                "Small empirical cross-covariance does not imply independence, nonlinear erasure, semantic causality, or utility preservation.",
                "Evaluation labels do not fit the projection; choosing bank, labels or rtol after inspecting evaluation compromises held-out interpretation.",
                "Truncated singular directions can leave fit covariance; recorded residuals use actual floating-point arithmetic.",
            ],
        )
    )
