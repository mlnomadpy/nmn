"""Contract binding, action restrictions and distinct sampled measurements."""

import copy

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import ThreeNeuronYat
from nmn.torch.replay import replay_native_record
from nmn.torch.research import collect_research_data
from nmn.torch.sampled_contract import evaluate_sampled_contract


def fixture():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    dataset = ResearchDataset(
        [ResearchSample("a", [1.0, 1.0], "evaluation", "a")],
        name="contract fixture",
        provenance="declared arithmetic",
    )
    snapshot = collect_research_data(
        model,
        torch.ones(1, 2, dtype=torch.float64),
        sample_ids=["a"],
        derivatives=False,
    )
    contract = dict(
        schema="nmn.sampled-contract.v1",
        scope="sampled",
        arithmetic="floating-point",
        model_sha256=snapshot["model_sha256"],
        dataset_sha256=dataset.sha256,
        sample_ids=["a"],
        action_domain=dict(
            kind="shared-module-writes",
            gate_bounds=[0.0, 1.0],
            allow_replacements=False,
        ),
        controls={"h": {"gate": 0.0}},
        targets=dict(
            output_names=["target"], absolute_tolerance=[0.0], expected={"a": [0.5]}
        ),
        protected=dict(output_names=["protected"], absolute_tolerance=[0.0]),
        provenance="fixture",
    )
    return model, dataset, contract


def test_sampled_compliance_protection_violation_and_bound_replay():
    model, data, contract = fixture()
    good = evaluate_sampled_contract(model, data, contract)
    assert (
        good["status"] == "observed"
        and good["assessment"] == "observed-within-tolerances"
    )
    changed = copy.deepcopy(contract)
    changed["controls"]["p"] = {"gate": 0.0}
    bad = evaluate_sampled_contract(model, data, changed)
    assert bad["violations"] == [
        dict(sample_id="a", measurement="protected", output="protected")
    ]
    assert good["contract_sha256"] != bad["contract_sha256"]
    assert replay_native_record(bad)["status"] == "matched"
    bad["contract"]["protected"]["absolute_tolerance"] = [1.0]
    with pytest.raises(ValueError, match="contract content identity"):
        replay_native_record(bad)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda c: c.update(scope="continuous"), "only sampled"),
        (lambda c: c.update(model_sha256="wrong"), "model identity"),
        (lambda c: c["controls"]["h"].update(replacement=0.0), "forbidden"),
        (lambda c: c["controls"]["h"].update(gate=2.0), "gate bounds"),
    ],
)
def test_rejects_invalid_identity_scope_and_actions(mutation, message):
    model, data, contract = fixture()
    mutation(contract)
    with pytest.raises(ValueError, match=message):
        evaluate_sampled_contract(model, data, contract)
