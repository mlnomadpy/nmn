"""Split-aware low-rank measurements of finite native edit-response tables."""

import hashlib
from pathlib import Path

import torch

from .research import _json_value, collect_research_data


def response_space_study(
    model, dataset, *, edits, rank, fit_split="tuning", evaluation_split="validation"
):
    """Fit a basis over edit coordinates using fit-split responses only.

    Each row is one sample/output pair; each column is the effect of one named
    edit relative to the unchanged network. The uncentered right singular vectors
    define an edit-coordinate subspace. Evaluation measures projection error on
    fresh rows, using all measured edits; it is not an unseen-edit predictor.
    Model parameters are unchanged. SVD fitting is explicit numerical analysis.
    """
    if not edits or any(not isinstance(name, str) or not name for name in edits):
        raise ValueError("supply nonempty named edits")
    if not fit_split or not evaluation_split or fit_split == evaluation_split:
        raise ValueError("fit and evaluation splits must be distinct and nonempty")
    fit_ids = dataset.sample_ids(split=fit_split)
    evaluation_ids = dataset.sample_ids(split=evaluation_split)
    names = sorted(edits)
    if not fit_ids or not evaluation_ids:
        raise ValueError("both split populations must be nonempty")
    if type(rank) is not int or not 1 <= rank <= min(
        len(names), len(fit_ids) * len(model.output_names)
    ):
        raise ValueError("rank must fit the observed fit-matrix dimensions")
    parameter = next(model.parameters())

    def collect(ids):
        inputs = torch.tensor(
            [dataset.sample(s).inputs for s in ids],
            dtype=parameter.dtype,
            device=parameter.device,
        )
        snapshot = collect_research_data(
            model, inputs, sample_ids=ids, edits=edits, derivatives=False
        )
        tensor = torch.stack(
            [
                torch.as_tensor(
                    snapshot["observations"]["edits"][name]["delta"],
                    dtype=torch.float64,
                )
                for name in names
            ],
            dim=-1,
        )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("response-space fitting requires finite edit effects")
        return snapshot, tensor

    fit_snapshot, fit = collect(fit_ids)
    matrix = fit.reshape(-1, len(names))
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    # Fit before executing the evaluation population; never refit on its errors.
    evaluation_snapshot, evaluation = collect(evaluation_ids)

    def project(responses):
        coordinates = responses @ basis
        reconstructed = coordinates @ basis.T
        residual = responses - reconstructed
        norm = torch.linalg.vector_norm(responses)
        residual_norm = torch.linalg.vector_norm(residual)
        return dict(
            responses=responses,
            coordinates=coordinates,
            reconstructed=reconstructed,
            residual=residual,
            residual_norm=residual_norm,
            relative_residual=None if norm == 0 else residual_norm / norm,
            per_sample_residual_norm=torch.linalg.vector_norm(
                residual.reshape(len(responses), -1), dim=1
            ),
        )

    threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular_values[0]
    return _json_value(
        dict(
            schema="nmn.response-space.v1",
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            model_snapshot=fit_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            protocol=dict(
                fit_split=fit_split,
                evaluation_split=evaluation_split,
                rank=rank,
                edit_order=names,
                output_order=list(model.output_names),
                centering="none",
                row_order="sample then output",
                basis_axis="named edit effects",
            ),
            basis=basis,
            singular_values=singular_values,
            numerical_rank=int((singular_values > threshold).sum()),
            numerical_rank_threshold=threshold,
            fit=project(fit),
            evaluation=project(evaluation),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cost=dict(
                model_forward_calls=2 * (1 + len(names)),
                fit_rows=matrix.shape[0],
                evaluation_rows=len(evaluation_ids) * len(model.output_names),
                columns=len(names),
            ),
            limitations=[
                "Finite observed edit-response subspace; no whole-response RKHS norm or uniform rank bound.",
                "Evaluation reconstruction uses every measured edit response; it is not prediction from a small queried subset.",
                "SVD bases may rotate within repeated singular-value subspaces; compare projections, not unique basis semantics.",
                "Declared splits and leakage groups do not establish statistical independence.",
            ],
        )
    )
