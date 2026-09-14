"""Finite observations of supplied affine state summaries and reduced dynamics."""

import hashlib
from pathlib import Path

import torch

from .graph import YatGraph
from .research import _json_value, collect_research_data


def reduction_study(model, dataset, *, start_layer, maps, provenance, split=None):
    """Execute a supplied encoder, decoder and one-layer summary transition.

    All maps use row-vector convention ``x @ weight + bias``. The same encoder
    and decoder apply at both boundaries. No fitting, invariant-domain claim or
    out-of-sample guarantee is performed. Raw states, summaries and residuals
    remain aligned with sample IDs in the returned portable record.
    """
    if not isinstance(model, YatGraph):
        raise ValueError("reduction studies require YatGraph")
    if type(start_layer) is not int or not 0 <= start_layer < len(model.layer_specs):
        raise ValueError("start_layer must identify a layer with a successor state")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("summary-map provenance is required")
    if not isinstance(maps, dict) or set(maps) != {"encoder", "decoder", "transition"}:
        raise ValueError("supply encoder, decoder and transition affine maps")
    ids = dataset.sample_ids(split=split)
    if not ids:
        raise ValueError("selected split has no samples")
    parameter = next(model.parameters())

    def tensor(value):
        result = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("maps and inputs must be finite")
        return result

    def affine_spec(name, rows, columns=None):
        spec = maps[name]
        if not isinstance(spec, dict) or set(spec) != {"weight", "bias"}:
            raise ValueError(f"{name} requires weight and bias")
        weight, bias = tensor(spec["weight"]), tensor(spec["bias"])
        if (
            weight.ndim != 2
            or weight.shape[0] != rows
            or weight.shape[1] < 1
            or (columns is not None and weight.shape[1] != columns)
            or bias.shape != (weight.shape[1],)
        ):
            raise ValueError(f"invalid {name} affine map dimensions")
        return weight, bias

    width = len(model.slots)
    encoder = affine_spec("encoder", width)
    rank = encoder[0].shape[1]
    decoder = affine_spec("decoder", rank, width)
    transition = affine_spec("transition", rank, rank)
    inputs = tensor([dataset.sample(sid).inputs for sid in ids])
    snapshot = collect_research_data(model, inputs, sample_ids=ids, derivatives=False)

    def apply(value, spec):
        return value @ spec[0] + spec[1]

    with torch.no_grad():
        baseline, trace = model.forward_with_trace(inputs)
        state = trace[f"state.{start_layer}"]
        next_state = trace[f"state.{start_layer + 1}"]
        summary = apply(state, encoder)
        actual_next_summary = apply(next_state, encoder)
        predicted_next_summary = apply(summary, transition)
        reconstructed_state = apply(summary, decoder)
        decoded_actual_next = apply(actual_next_summary, decoder)
        decoded_predicted_next = apply(predicted_next_summary, decoder)
        reconstructed_outputs, reconstructed_trace = model.forward_from_state(
            reconstructed_state, start_layer=start_layer
        )
        decoded_outputs, decoded_trace = model.forward_from_state(
            decoded_actual_next, start_layer=start_layer + 1
        )
        predicted_outputs, predicted_trace = model.forward_from_state(
            decoded_predicted_next, start_layer=start_layer + 1
        )
        observations = dict(
            state=state,
            next_state=next_state,
            summary=summary,
            actual_next_summary=actual_next_summary,
            predicted_next_summary=predicted_next_summary,
            summary_transition_residual=predicted_next_summary - actual_next_summary,
            reconstructed_state=reconstructed_state,
            state_reconstruction_residual=reconstructed_state - state,
            decoded_actual_next_state=decoded_actual_next,
            next_state_reconstruction_residual=decoded_actual_next - next_state,
            decoded_predicted_next_state=decoded_predicted_next,
            predicted_next_state_residual=decoded_predicted_next - next_state,
            baseline_outputs=baseline,
            reconstructed_outputs=reconstructed_outputs,
            reconstruction_output_residual=reconstructed_outputs - baseline,
            decoded_next_outputs=decoded_outputs,
            next_reconstruction_output_residual=decoded_outputs - baseline,
            predicted_outputs=predicted_outputs,
            prediction_output_residual=predicted_outputs - baseline,
            reconstructed_trace=reconstructed_trace,
            decoded_next_trace=decoded_trace,
            predicted_trace=predicted_trace,
        )
    finite = all(
        bool(value.isfinite().all())
        for value in observations.values()
        if isinstance(value, torch.Tensor)
    ) and all(
        bool(value.isfinite().all())
        for values in (trace, reconstructed_trace, decoded_trace, predicted_trace)
        for value in values.values()
        if isinstance(value, torch.Tensor)
    )
    return _json_value(
        dict(
            schema="nmn.reduction-study.v1",
            status="observed" if finite else "nonfinite-observation",
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            model_snapshot=snapshot,
            sample_ids=ids,
            maps={
                name: dict(weight=spec[0], bias=spec[1])
                for name, spec in (
                    ("encoder", encoder),
                    ("decoder", decoder),
                    ("transition", transition),
                )
            },
            protocol=dict(
                start_layer=start_layer,
                slot_order=list(model.slots),
                summary_width=rank,
                split=split,
                provenance=provenance,
                map_convention="row-vector: x @ weight + bias",
                residual_convention="reconstructed or predicted minus actual",
            ),
            observations=observations,
            cost=dict(full_forwards=2, suffix_forwards=3, samples=len(ids)),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Supplied affine maps are evaluated without fitting or validating their provenance.",
                "Finite observations do not establish closure, an invariant domain or error propagation bounds.",
                "Decoded states may be unreachable; no native edit or semantic equivalence is inferred.",
                "Only one unchanged layer transition is compared; no controlled reduced dynamics is inferred.",
            ],
        )
    )


def fit_reduction_study(
    model,
    dataset,
    *,
    start_layer,
    rank,
    ridge,
    fit_split="tuning",
    evaluation_split="validation",
):
    """Fit centered PCA and an affine ridge transition on one declared split.

    The decoder is the transpose PCA basis plus the joint boundary-state mean.
    The transition minimizes mean squared summary error plus ``ridge * ||A||²``;
    its intercept is unpenalized. All maps are frozen before evaluation inputs
    are executed. No model parameters, rank or hyperparameters are optimized.
    """
    import math

    if not isinstance(model, YatGraph):
        raise ValueError("reduction fitting requires YatGraph")
    if type(start_layer) is not int or not 0 <= start_layer < len(model.layer_specs):
        raise ValueError("start_layer must identify a layer with a successor state")
    if type(rank) is not int or not 1 <= rank <= len(model.slots):
        raise ValueError("rank must be a positive integer no larger than state width")
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("ridge must be finite and strictly positive")
    if (
        not all(isinstance(s, str) and s for s in (fit_split, evaluation_split))
        or fit_split == evaluation_split
    ):
        raise ValueError("fit and evaluation splits must be distinct named populations")
    fit_ids = dataset.sample_ids(split=fit_split)
    evaluation_ids = dataset.sample_ids(split=evaluation_split)
    if len(fit_ids) < 2 or not evaluation_ids:
        raise ValueError(
            "supply at least two fit samples and a nonempty evaluation split"
        )
    parameter = next(model.parameters())
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("summary fitting requires float32 or float64 model parameters")
    inputs = torch.tensor(
        [dataset.sample(sid).inputs for sid in fit_ids],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    with torch.no_grad():
        _, trace = model.forward_with_trace(inputs)
        before = trace[f"state.{start_layer}"]
        after = trace[f"state.{start_layer + 1}"]
        joint = torch.cat((before, after))
        if not bool(torch.isfinite(joint).all()):
            raise ValueError("fit boundary states must be finite")
        mean = joint.mean(0)
        centered = joint - mean
        _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
        threshold = (
            max(centered.shape) * torch.finfo(parameter.dtype).eps * singular_values[0]
        )
        numerical_rank = int((singular_values > threshold).sum().item())
        if rank > numerical_rank:
            raise ValueError(
                "requested rank exceeds numerical rank of fit boundary states"
            )
        basis = right[:rank].T.contiguous()
        # Canonicalize column signs for readable snapshots, not unique eigenspaces.
        pivots = basis.abs().argmax(dim=0)
        signs = basis[pivots, torch.arange(rank, device=basis.device)].sign()
        basis = basis * signs
        summary = (before - mean) @ basis
        next_summary = (after - mean) @ basis
        mean_summary, mean_next = summary.mean(0), next_summary.mean(0)
        x, y = summary - mean_summary, next_summary - mean_next
        regularizer = torch.as_tensor(
            ridge, dtype=parameter.dtype, device=parameter.device
        )
        if not bool(torch.isfinite(regularizer)) or regularizer.item() <= 0:
            raise ValueError(
                "ridge is not representable as positive finite model arithmetic"
            )
        gram = x.T @ x / len(fit_ids)
        matrix = torch.linalg.solve(
            gram
            + regularizer
            * torch.eye(rank, dtype=parameter.dtype, device=parameter.device),
            x.T @ y / len(fit_ids),
        )
        intercept = mean_next - mean_summary @ matrix
        maps = _json_value(
            {
                "encoder": {"weight": basis, "bias": -mean @ basis},
                "decoder": {"weight": basis.T, "bias": mean},
                "transition": {"weight": matrix, "bias": intercept},
            }
        )
    # Frozen JSON maps are formed before either measurement pass. In particular,
    # no evaluation state participates in PCA centering, directions or regression.
    provenance = "Fitted on split " + fit_split + "; frozen before evaluation"
    fit = reduction_study(
        model,
        dataset,
        start_layer=start_layer,
        maps=maps,
        provenance=provenance,
        split=fit_split,
    )
    evaluation = reduction_study(
        model,
        dataset,
        start_layer=start_layer,
        maps=maps,
        provenance=provenance,
        split=evaluation_split,
    )
    return _json_value(
        dict(
            schema="nmn.fitted-reduction.v1",
            status=(
                "observed"
                if all(s["status"] == "observed" for s in (fit, evaluation))
                else "nonfinite-observation"
            ),
            model_snapshot=fit["model_snapshot"],
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            maps=maps,
            protocol=dict(
                start_layer=start_layer,
                rank=rank,
                ridge=regularizer.item(),
                fit_split=fit_split,
                evaluation_split=evaluation_split,
                fit_sample_ids=fit_ids,
                evaluation_sample_ids=evaluation_ids,
                method="joint-boundary centered PCA + affine ridge transition",
                transition_objective="mean squared summary error + ridge * squared Frobenius weight norm; unpenalized intercept",
                frozen_before_evaluation=True,
            ),
            fitting=dict(
                joint_state_mean=mean,
                singular_values=singular_values,
                numerical_rank=numerical_rank,
                rank_threshold=threshold,
                fit_state_count=len(joint),
            ),
            fit=fit,
            evaluation=evaluation,
            cost=dict(
                fitting_full_forwards=1,
                measurement_full_forwards=4,
                measurement_suffix_forwards=6,
                svd_calls=1,
                linear_solves=1,
            ),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "PCA optimizes sampled state reconstruction, not semantic meaning or downstream utility.",
                "Rank and ridge are supplied; repeated evaluation-guided choices can leak evaluation information.",
                "Split/group labels are checked structurally, not proven statistically independent.",
                "Degenerate singular subspaces are not uniquely identified; no closure or uniform error bound is established.",
                "Replay the embedded fit/evaluation reduction records to check frozen-map execution; fitting itself has no replay adapter.",
            ],
        )
    )
