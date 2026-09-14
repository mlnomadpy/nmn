"""Fit-only linear classification probes of native internal trace tensors."""

import hashlib
import math
from pathlib import Path

import torch

from .research import _json_value, collect_research_data


def probe_study(
    model,
    dataset,
    *,
    feature,
    labels,
    classes,
    provenance,
    ridge,
    edits=None,
    refit_edits=False,
    fit_split="tuning",
    evaluation_split="validation",
):
    """Fit an affine ridge classifier on baseline fit features, then freeze it.

    Scores regress one-hot labels and are not probabilities. Argmax ties choose
    the first declared class. Edits are evaluated only on the held-out population
    with the unchanged probe by default. Explicit refit_edits also fits separate
    classifiers on edited fit samples before evaluation. Neither implies erasure.
    """
    if (
        not isinstance(feature, str)
        or not feature
        or not isinstance(provenance, str)
        or not provenance.strip()
    ):
        raise ValueError("feature trace key and label provenance are required")
    if (
        not isinstance(classes, (list, tuple))
        or len(classes) < 2
        or any(not isinstance(c, str) or not c for c in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError(
            "supply at least two unique nonempty class names in tie-break order"
        )
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
    fit_ids, evaluation_ids = dataset.sample_ids(split=fit_split), dataset.sample_ids(
        split=evaluation_split
    )
    if (
        len(fit_ids) < 2
        or not evaluation_ids
        or not isinstance(labels, dict)
        or set(labels) != set(fit_ids + evaluation_ids)
    ):
        raise ValueError(
            "labels must cover exactly a fit population of at least two samples and a nonempty evaluation population"
        )
    if any(
        not isinstance(value, str) or value not in classes for value in labels.values()
    ):
        raise ValueError("labels must be declared class names")
    if set(labels[sid] for sid in fit_ids) != set(classes):
        raise ValueError("every declared class must occur in the fit population")
    labels = dict(labels)
    classes = list(classes)
    edits = {} if edits is None else edits
    if any(not isinstance(name, str) or not name for name in edits):
        raise ValueError("edit names must be nonempty strings")
    parameter = next(model.parameters())
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("probe fitting requires float32 or float64")

    def tensor(values):
        return torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device)

    def snapshot(ids, controls):
        return collect_research_data(
            model,
            tensor([dataset.sample(s).inputs for s in ids]),
            sample_ids=ids,
            edits=controls,
            derivatives=False,
        )

    def features(trace, count):
        if feature not in trace:
            raise ValueError("feature trace key is absent from the model execution")
        value = tensor(trace[feature])
        if (
            value.ndim != 2
            or value.shape[0] != count
            or value.shape[1] < 1
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("probe features must be finite sample-by-feature matrices")
        return value

    if type(refit_edits) is not bool:
        raise ValueError("refit_edits must be a boolean")
    if refit_edits:
        if not edits:
            raise ValueError("refitting requires at least one named edit")
        # A scalar has identical meaning across the two different populations.
        # Per-example or per-write tensors require a separate population contract.
        for controls in edits.values():
            for control in controls.values():
                for value in (control.gate, control.replacement):
                    if value is not None and tensor(value).ndim != 0:
                        raise ValueError(
                            "refitted comparisons require population-independent scalar edit controls"
                        )
    fit_snapshot = snapshot(fit_ids, edits if refit_edits else {})
    x = features(fit_snapshot["observations"]["baseline_trace"], len(fit_ids))
    target_indices = torch.tensor(
        [classes.index(labels[s]) for s in fit_ids], device=x.device
    )
    y = torch.nn.functional.one_hot(target_indices, len(classes)).to(x.dtype)
    regularizer = tensor(ridge)
    if not bool(torch.isfinite(regularizer)) or regularizer.item() <= 0:
        raise ValueError(
            "ridge is not representable as positive finite model arithmetic"
        )

    def solve(values):
        mean_x, mean_y = values.mean(0), y.mean(0)
        centered_x, centered_y = values - mean_x, y - mean_y
        gram = centered_x.T @ centered_x / len(values)
        weight = torch.linalg.solve(
            gram
            + regularizer
            * torch.eye(values.shape[1], device=values.device, dtype=values.dtype),
            centered_x.T @ centered_y / len(values),
        )
        bias = mean_y - mean_x @ weight
        if not bool(torch.isfinite(weight).all() & torch.isfinite(bias).all()):
            raise ValueError("probe solve produced nonfinite parameters")
        return dict(weight=weight, bias=bias, feature_mean=mean_x, target_mean=mean_y)

    probe = solve(x)
    refitted_probes = {}
    if refit_edits:
        for name, row in fit_snapshot["observations"]["edits"].items():
            refitted_probes[name] = solve(features(row["trace"], len(fit_ids)))
    # All baseline and optional edited fits are frozen before evaluation forwards.
    evaluation_snapshot = snapshot(evaluation_ids, edits)

    def measure(values, ids, fitted=None):
        fitted = probe if fitted is None else fitted
        scores = values @ fitted["weight"] + fitted["bias"]
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("probe scores must be finite")
        predicted = scores.argmax(1).tolist()
        predictions = [classes[i] for i in predicted]
        correct = [p == labels[s] for p, s in zip(predictions, ids)]
        confusion = [[0 for _ in classes] for _ in classes]
        for sid, index in zip(ids, predicted):
            confusion[classes.index(labels[sid])][index] += 1
        return dict(
            features=values,
            scores=scores,
            predictions=predictions,
            labels=[labels[s] for s in ids],
            correct=correct,
            accuracy=sum(correct) / len(correct),
            confusion=confusion,
        )

    fit = measure(x, fit_ids)
    observation = evaluation_snapshot["observations"]
    baseline = measure(
        features(observation["baseline_trace"], len(evaluation_ids)), evaluation_ids
    )
    edited = {}
    for name, row in observation["edits"].items():
        result = measure(features(row["trace"], len(evaluation_ids)), evaluation_ids)
        eligible = sum(baseline["correct"])
        damaged = sum(
            a and not b for a, b in zip(baseline["correct"], result["correct"])
        )
        result.update(
            feature_delta=result["features"] - baseline["features"],
            score_delta=result["scores"] - baseline["scores"],
            baseline_correct_count=eligible,
            damaged_count=damaged,
            conditional_damage=damaged / eligible if eligible else None,
            disagreement=sum(
                a != b for a, b in zip(baseline["predictions"], result["predictions"])
            )
            / len(evaluation_ids),
        )
        if refit_edits:
            fitted = refitted_probes[name]
            result["refitted"] = dict(
                probe=fitted,
                fit=measure(
                    features(
                        fit_snapshot["observations"]["edits"][name]["trace"],
                        len(fit_ids),
                    ),
                    fit_ids,
                    fitted,
                ),
                evaluation=measure(result["features"], evaluation_ids, fitted),
            )
        edited[name] = result
    return _json_value(
        dict(
            schema="nmn.probe-study.v1",
            status="observed",
            model_snapshot=fit_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            labels=labels,
            classes=classes,
            protocol=dict(
                feature=feature,
                provenance=provenance,
                ridge=regularizer.item(),
                fit_split=fit_split,
                evaluation_split=evaluation_split,
                fit_sample_ids=fit_ids,
                evaluation_sample_ids=evaluation_ids,
                rule="argmax scores; first declared class wins ties",
                method="affine ridge regression on one-hot labels; unpenalized intercept",
                frozen_before_evaluation=True,
                **(
                    {
                        "refit_edits": True,
                        "refit_controls": "same scalar controls on both populations",
                    }
                    if refit_edits
                    else {}
                ),
            ),
            probe=probe,
            fit=fit,
            evaluation=dict(baseline=baseline, edits=edited),
            cost=dict(
                full_forwards=2 + len(edits) * (2 if refit_edits else 1),
                linear_solves=1 + (len(edits) if refit_edits else 0),
            ),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Probe accuracy measures decoding by this fitted linear classifier, not semantic causality or a stable inverse.",
                "Scores are uncalibrated ridge outputs, not class probabilities.",
                "A frozen probe can fail after a representation change while another probe recovers the labels; no erasure is certified.",
                "Declared split/group separation does not prove independence; feature and ridge selection using evaluation results can leak information.",
            ],
        )
    )
