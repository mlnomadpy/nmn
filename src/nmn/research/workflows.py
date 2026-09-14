"""Backend-free discovery of native research data workflows."""

import copy
from typing import Any

# Inputs describe research obligations, not a replacement for each command's help.
_WORKFLOWS = [
    (
        "observations",
        ["KG01", "KG03"],
        ["init", "inspect", "collect", "diagnose"],
        "Model configuration or snapshot, identified samples, optional named edits",
        "Full outputs/traces, local bank geometry, input/gate derivatives and sensor diagnostics",
        "torch; nnx fixed architecture for init/inspect/collect only",
        "Local and sampled quantities do not establish whole-response norms or stable inverse bounds",
        "native-cli.md",
    ),
    (
        "depth",
        ["KG02"],
        ["suffix", "reduce", "fit-reduction"],
        "Saved model, split-labelled samples, boundary states or supplied/fitted affine maps",
        "Executed suffixes, reconstruction, transition and downstream residuals",
        "torch; nnx fixed architecture suffix only",
        "No invariant-domain proof or general nonlinear summary closure",
        "native-reduction.md",
    ),
    (
        "probes",
        ["KG03"],
        ["probe"],
        "Trace key, class labels and provenance, fit/evaluation splits, ridge and feature-map choice",
        "Fit-only linear/quadratic probe parameters, raw scores and frozen/refitted edit evaluation",
        "torch",
        "Probe failure is not an erasure certificate",
        "native-probes.md",
    ),
    (
        "interactions",
        ["KG04"],
        ["coalitions", "path", "curvature"],
        "Model and samples, named module subsets or gate directions/endpoints, explicit budgets",
        "Finite joint effects, subset coefficients when complete, path quadrature, Hessian-vector products and residuals",
        "torch",
        "No sparse recovery or uniform finite-edit residual guarantee",
        "native-curvature.md",
    ),
    (
        "native-edits",
        ["KG05"],
        ["donor", "edges", "protect", "preimage", "apply-preimage"],
        "Model and samples, donor pairs/read-edge routes or feature targets, declared protection tasks",
        "Native intervention traces, per-example target/protection effects, realizability residuals",
        "torch",
        "No generic external-model adapter or automatic semantic reference",
        "native-preimage.md",
    ),
    (
        "selection",
        ["KG06"],
        ["plan-donors", "select", "search-gates", "response-space"],
        "Candidate family, split/group IDs, labels/objectives and explicit budgets",
        "Pair plans, candidate ledger, frozen-winner evaluation and fitted finite response spaces",
        "torch; plan-donors requires no ML backend",
        "Declared splits do not prove independence; no statistical guarantee or adaptive acquisition algorithm",
        "native-selection.md",
    ),
    (
        "semantics",
        ["KG07"],
        ["semantics", "align"],
        "Supplied counterfactual reference, fixed correspondence or candidate module pool, selection/evaluation splits",
        "Baseline/counterfactual agreement, supervised assignment scores, ties and frozen held-out outcomes",
        "torch",
        "Finite whole-module assignment is not distributed alignment, recursive scrubbing or unique identification",
        "native-alignment.md",
    ),
    (
        "training",
        ["KG05", "KG07"],
        ["train", "checkpoint", "benchmark"],
        "Initial model, training/validation labels, architecture contract, budgets/seeds and optional donor pairs",
        "Checkpoints, task/intervention losses, gradient policy, failures and matched native evaluation",
        "torch",
        "Sampling seeds share initialization; validation selects checkpoints and is not final evaluation",
        "intervention-training.md",
    ),
    (
        "bounds",
        ["KG01", "KG05"],
        ["enclose", "verify-box", "check-box"],
        "Fixed supported model, rational input domain, gates and output or difference contract",
        "Real-arithmetic enclosures, complete/pending box partitions and checked counterexamples",
        "torch; fixed Yat/IMQ/linear arithmetic only",
        "No floating-point roundoff or broader operator certification",
        "native-interval-contract.md",
    ),
    (
        "delivery",
        [],
        ["replay", "extract", "export", "verify-export", "report"],
        "Saved recognized evidence records and explicit local output paths",
        "Numerical replay differences, reusable components, exact-byte vault exports and offline dashboards",
        "replay uses the recorded torch/nnx backend; remaining commands need no ML backend",
        "Replay, file integrity and research validity are separate; optimizer replay is not supplied",
        "native-replay.md",
    ),
]


def workflow_catalog(family=None):
    """Return structured command discovery; never import or execute an ML backend."""
    if family is not None and family not in {f"KG{i:02d}" for i in range(1, 8)}:
        raise ValueError("family must be KG01 through KG07")
    fields = (
        "id",
        "families",
        "commands",
        "required_inputs",
        "data_outputs",
        "backend_scope",
        "limitations",
        "documentation",
    )
    records: list[dict[str, Any]] = [
        dict(zip(fields, copy.deepcopy(row)))
        for row in _WORKFLOWS
        if family is None or family in row[1]
    ]
    for row in records:
        row["commands"] = ["nmn research native " + c for c in row["commands"]]
        row["documentation"] = "docs/" + row["documentation"]
    return dict(
        schema="nmn.research-workflows.v1",
        workflows=records,
        filter_family=family,
        architecture_commands=[
            "nmn research architecture",
            "nmn research architecture GRAPH.json",
        ],
        assurance="Capability descriptions, not evidence of source-protocol reproduction",
        unsupported=[
            "sequence-model data adapter and frozen decoding contract",
            "distributed semantic alignment and recursive causal scrubbing",
            "statistical validation guarantees and runtime-roundoff certification",
        ],
    )
