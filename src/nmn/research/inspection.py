"""Explicit, conservative dependencies for the supported fixed topology."""

from typing import Any, Dict

from .model import validate
from .reference import digest, encoded


def inspect_model(model: Dict[str, Any]) -> Dict[str, Any]:
    model = validate(model)
    edges = [
        ["u", "h"],
        ["gate", "h"],
        ["v", "p"],
        ["h", "y"],
        ["v", "y"],
        ["y", "target"],
        ["p", "protected"],
    ]
    if model["protected_leak"] != "0":
        edges.append(["y", "protected"])
    return {
        "schema": "nmn.three-neuron-inspection.v1",
        "model": model,
        "model_sha256": digest(encoded(model)),
        "encoder": "(u,v) -> (u,v,0,0,0)",
        "layers": [["h", "p"], ["y"]],
        "neurons": [
            {
                "id": name,
                "reads": reads,
                "writes": [name],
                "center": model["neurons"][name]["center"],
                "coefficient": model["neurons"][name]["coefficient"],
                "epsilon": model["epsilon"],
            }
            for name, reads in (("h", ["u"]), ("p", ["v"]), ("y", ["h", "v"]))
        ],
        "readouts": {"target": "y", "protected": f"p + ({model['protected_leak']})*y"},
        "dependency_edges": edges,
        "h_to_target_paths": [["h", "y", "target"]],
        "h_to_protected_paths": (
            [] if model["protected_leak"] == "0" else [["h", "y", "protected"]]
        ),
        "parameters": {
            "prototype_coordinates": 4,
            "coefficients": 3,
            "shared_epsilon": 1,
            "protected_readout_mix": 1,
        },
        "scope": "Declared fixed topology only; conservative paths may overstate dependence for degenerate parameters. Kernel centers are not inferred semantic labels.",
    }
