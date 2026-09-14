"""Bounded rational box search and search-independent partition checking."""

import copy
import hashlib
import json
from collections import deque
from pathlib import Path

from ..research.intervals import RationalInterval, rational
from .enclosure import enclose_native, enclose_native_difference

ASSURANCE = "real-valued range contract over the entire input box; exact rational enclosures; no runtime-roundoff coverage"


def _contract(snapshot, contract):
    contract = json.loads(json.dumps(contract, allow_nan=False))
    if contract.get("schema") != "nmn.interval-contract.v1":
        raise ValueError("unsupported interval contract schema")
    if not isinstance(contract.get("provenance"), str) or not contract["provenance"]:
        raise ValueError("contract provenance is required")
    if "reference_controls" in contract:
        initial = enclose_native_difference(
            snapshot,
            contract["input_box"],
            controls=contract.get("controls"),
            reference_controls=contract["reference_controls"],
        )
        contract["reference_controls"] = initial["reference_controls"]
    else:
        initial = enclose_native(
            snapshot, contract["input_box"], controls=contract.get("controls")
        )
    limits = contract["outputs"]
    if not limits or not set(limits) <= set(initial["output_bounds"]):
        raise ValueError("contract must constrain declared model outputs")
    contract["input_box"] = initial["input_box"]
    contract["outputs"] = {
        name: RationalInterval(*bounds).to_list() for name, bounds in limits.items()
    }
    contract["controls"] = initial["controls"]
    return contract, initial


def _evaluate(snapshot, box, contract):
    if "reference_controls" in contract:
        return enclose_native_difference(
            snapshot,
            box,
            controls=contract["controls"],
            reference_controls=contract["reference_controls"],
        )
    return enclose_native(snapshot, box, controls=contract["controls"])


def _covered(bounds, limits):
    return all(
        rational(limits[name][0]) <= rational(bounds[name][0])
        and rational(bounds[name][1]) <= rational(limits[name][1])
        for name in limits
    )


def _split(box, axis, midpoint):
    lower, upper = map(rational, box[axis])
    midpoint = rational(midpoint)
    if not lower < midpoint < upper:
        raise ValueError("split point must lie strictly inside its coordinate interval")
    left, right = copy.deepcopy(box), copy.deepcopy(box)
    left[axis][1], right[axis][0] = str(midpoint), str(midpoint)
    return left, right


def verify_box(snapshot, contract, *, max_boxes):
    """Bound a real-function range contract, retaining a complete box partition.

    Each processed region costs one interval enclosure; unresolved regions also
    receive an exact midpoint check. Splits bisect the widest rational coordinate
    (ties by sorted coordinate name). Budget exhaustion preserves pending regions.
    """
    if type(max_boxes) is not int or max_boxes < 1:
        raise ValueError("max_boxes must be a positive integer")
    contract, initial = _contract(snapshot, contract)
    root = contract["input_box"]
    nodes = {}
    pending = deque([("", root)])
    regions = points = 0
    witness_found = False
    while pending and regions < max_boxes:
        path, box = pending.popleft()
        bound = initial if not path else _evaluate(snapshot, box, contract)
        regions += 1
        node = {"box": box}
        nodes[path] = node
        if _covered(bound["output_bounds"], contract["outputs"]):
            node.update(kind="covered", output_bounds=bound["output_bounds"])
            continue
        midpoint = {
            name: [str((rational(pair[0]) + rational(pair[1])) / 2)] * 2
            for name, pair in box.items()
        }
        evaluated = _evaluate(snapshot, midpoint, contract)
        points += 1
        if not _covered(evaluated["output_bounds"], contract["outputs"]):
            node.update(
                kind="counterexample",
                point=midpoint,
                output_bounds=evaluated["output_bounds"],
            )
            witness_found = True
            break
        axis = min(
            box,
            key=lambda name: (-(rational(box[name][1]) - rational(box[name][0])), name),
        )
        split_value = midpoint[axis][0]
        left, right = _split(box, axis, split_value)
        node.update(kind="split", axis=axis, midpoint=split_value)
        pending.extend([(path + "0", left), (path + "1", right)])
    for path, box in pending:
        nodes[path] = dict(kind="pending", box=box)
    status = (
        "counterexample-found"
        if witness_found
        else "inconclusive" if pending else "certified-under-assumptions"
    )
    return dict(
        schema="nmn.interval-certificate.v1",
        quantity="output-difference" if "reference_controls" in contract else "output",
        status=status,
        model_snapshot=snapshot,
        contract=contract,
        nodes=nodes,
        budget=dict(
            max_boxes=max_boxes, regions_enclosed=regions, exact_points_checked=points
        ),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assurance=ASSURANCE,
        limitations=[
            "Only fixed Yat/IMQ graphs, constant controls and closed output or output-difference ranges are supported.",
            "Positive denominators make the supported rational expressions globally defined; no additional hidden-state domain constraint is asserted.",
            "Interval overestimation yields subdivision or inconclusive status, never a counterexample by itself.",
            "This record requires partition/enclosure checking; a saved status string is not trusted.",
        ],
    )


def check_box_certificate(certificate):
    """Check the saved partition and leaves without repeating the search policy.

    Reconstruct child boxes from each supplied interior split, reject gaps/extra
    nodes, recompute every claimed enclosure and evaluate each claimed witness.
    Shares rational arithmetic with the search; this is not a second arithmetic
    implementation or an external proof assistant.
    """
    if certificate.get("schema") != "nmn.interval-certificate.v1":
        raise ValueError("unsupported interval certificate schema")
    if certificate.get("assurance") != ASSURANCE:
        raise ValueError("unsupported certificate assurance claim")
    snapshot = certificate["model_snapshot"]
    contract, _ = _contract(snapshot, certificate["contract"])
    quantity = "output-difference" if "reference_controls" in contract else "output"
    if certificate.get("quantity", quantity) != quantity:
        raise ValueError("certificate quantity differs from contract")
    nodes = certificate["nodes"]
    queue = deque([("", contract["input_box"])])
    seen = set()
    counts = {"covered": 0, "pending": 0, "counterexample": 0, "split": 0}
    while queue:
        path, box = queue.popleft()
        if path not in nodes or nodes[path]["box"] != box:
            raise ValueError("partition has missing nodes or inconsistent child boxes")
        seen.add(path)
        node = nodes[path]
        kind = node["kind"]
        if kind not in counts:
            raise ValueError("unknown certificate node kind")
        counts[kind] += 1
        if kind == "split":
            if node["axis"] not in box:
                raise ValueError("unknown split coordinate")
            left, right = _split(box, node["axis"], node["midpoint"])
            queue.extend([(path + "0", left), (path + "1", right)])
        elif kind == "covered":
            actual = _evaluate(snapshot, box, contract)["output_bounds"]
            if actual != node["output_bounds"] or not _covered(
                actual, contract["outputs"]
            ):
                raise ValueError(
                    "covered leaf enclosure does not establish its contract"
                )
        elif kind == "counterexample":
            point = node["point"]
            if (
                not isinstance(point, dict)
                or set(point) != set(box)
                or any(
                    pair[0] != pair[1]
                    or not rational(box[name][0])
                    <= rational(pair[0])
                    <= rational(box[name][1])
                    for name, pair in point.items()
                )
            ):
                raise ValueError("counterexample must be a point inside its leaf")
            actual = _evaluate(snapshot, point, contract)["output_bounds"]
            if (
                any(pair[0] != pair[1] for pair in actual.values())
                or actual != node["output_bounds"]
                or _covered(actual, contract["outputs"])
            ):
                raise ValueError("claimed point does not violate the contract")
    if seen != set(nodes):
        raise ValueError("partition contains unreachable extra nodes")
    budget = certificate["budget"]
    regions = counts["covered"] + counts["split"] + counts["counterexample"]
    if (
        any(
            type(budget.get(key)) is not int
            for key in ("max_boxes", "regions_enclosed", "exact_points_checked")
        )
        or budget["max_boxes"] < 1
        or budget["regions_enclosed"] != regions
        or regions > budget["max_boxes"]
        or budget["exact_points_checked"] != counts["split"] + counts["counterexample"]
    ):
        raise ValueError("budget accounting disagrees with partition")
    outcome = (
        "counterexample-found"
        if counts["counterexample"]
        else "inconclusive" if counts["pending"] else "certified-under-assumptions"
    )
    if certificate["status"] != outcome:
        raise ValueError("saved outcome disagrees with checked leaves")
    return dict(
        schema="nmn.interval-check.v1",
        quantity=quantity,
        status="certificate-checked",
        outcome=outcome,
        counts=counts,
        model_sha256=snapshot["model_sha256"],
        certificate_sha256=hashlib.sha256(
            json.dumps(certificate, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        assurance=ASSURANCE,
        limitations=[
            "Partition checking is independent of search order, but shares the enclosure arithmetic implementation.",
            "No floating-point runtime error or statistical guarantee is certified.",
        ],
    )
