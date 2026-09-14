"""Backend-free schema and topology validation for native graph configuration."""

import copy
import hashlib
import json
import math


def architecture_schema():
    """JSON Schema covers shape; validate_architecture checks cross references."""
    names = dict(
        type="array",
        minItems=1,
        uniqueItems=True,
        items=dict(type="string", minLength=1),
    )
    module = dict(
        type="object",
        additionalProperties=False,
        required=["name", "reads", "writes"],
        properties=dict(
            name=dict(
                type="string",
                minLength=1,
                description="Python identifier; checked by native validator",
            ),
            reads=names,
            writes=names,
            num_centers=dict(type="integer", minimum=1, default=1),
            epsilon=dict(type="number", exclusiveMinimum=0, default=1.0),
            family=dict(enum=["yat", "imq", "tanh", "linear"], default="yat"),
        ),
    )
    return copy.deepcopy(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "NMN explicit-state graph configuration v1",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "class",
                "slots",
                "input_names",
                "output_names",
                "layers",
                "update",
                "distance_mode",
            ],
            "properties": {
                "class": {"const": "nmn.torch.YatGraph"},
                "slots": names,
                "input_names": names,
                "output_names": names,
                "layers": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "array", "minItems": 1, "items": module},
                },
                "update": {"const": "simultaneous-residual-add"},
                "distance_mode": {"const": "direct"},
            },
            "$comment": "Finite numbers, unique global module names, Python identifiers and slot membership require validate_architecture. Parameters, dtype and interventions are separate contracts.",
        }
    )


def validate_architecture(configuration):
    """Validate and normalize fixed graph topology without allocating a model.

    No weights, data or ML imports are needed. This describes dimensions and native
    execution rules, not trained semantics or a structural protection certificate.
    """

    def fail(path, message):
        raise ValueError(f"{path}: {message}")

    def object_fields(value, required, optional, path):
        if not isinstance(value, dict):
            fail(path, "expected an object")
        if set(required) - set(value):
            fail(path, f"missing fields {sorted(set(required)-set(value))}")
        if set(value) - set(required) - set(optional):
            fail(
                path, f"unknown fields {sorted(set(value)-set(required)-set(optional))}"
            )

    schema = architecture_schema()
    object_fields(configuration, schema["required"], [], "$")
    for key, expected in (
        ("class", "nmn.torch.YatGraph"),
        ("update", "simultaneous-residual-add"),
        ("distance_mode", "direct"),
    ):
        if configuration[key] != expected:
            fail("$." + key, f"expected {expected!r}")

    def names(value, path):
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(s, str) or not s for s in value)
        ):
            fail(path, "expected a nonempty list of nonempty strings")
        if len(set(value)) != len(value):
            fail(path, "names must be unique")
        return value

    slots = names(configuration["slots"], "$.slots")
    for key in ("input_names", "output_names"):
        value = names(configuration[key], "$." + key)
        if not set(value) <= set(slots):
            fail("$." + key, "all names must refer to declared slots")
    layers = configuration["layers"]
    if not isinstance(layers, list) or not layers:
        fail("$.layers", "expected nonempty layers")
    normalized = copy.deepcopy(configuration)
    seen = set()
    dimensions = []
    for i, layer in enumerate(layers):
        if not isinstance(layer, list) or not layer:
            fail(f"$.layers[{i}]", "expected a nonempty module list")
        for j, spec in enumerate(layer):
            path = f"$.layers[{i}][{j}]"
            object_fields(
                spec,
                ["name", "reads", "writes"],
                ["num_centers", "epsilon", "family"],
                path,
            )
            name = spec["name"]
            if not isinstance(name, str) or not name.isidentifier():
                fail(path + ".name", "expected a nonempty Python identifier")
            if name in seen:
                fail(path + ".name", "module names must be globally unique")
            seen.add(name)
            for key in ("reads", "writes"):
                value = names(spec[key], path + "." + key)
                if not set(value) <= set(slots):
                    fail(path + "." + key, "all names must refer to declared slots")
            width = spec.get("num_centers", 1)
            epsilon = spec.get("epsilon", 1.0)
            family = spec.get("family", "yat")
            if type(width) is not int or width < 1:
                fail(path + ".num_centers", "expected a positive integer")
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon <= 0
            ):
                fail(path + ".epsilon", "expected a finite positive number")
            if family not in ("yat", "imq", "tanh", "linear"):
                fail(path + ".family", "expected yat, imq, tanh or linear")
            normalized["layers"][i][j] = dict(
                spec, num_centers=width, epsilon=epsilon, family=family
            )
            dimensions.append(
                dict(
                    module=name,
                    layer=i,
                    family=family,
                    inputs=len(spec["reads"]),
                    outputs=len(spec["writes"]),
                    width=width,
                    center_or_hidden_weight_shape=[width, len(spec["reads"])],
                    coefficient_shape=[len(spec["writes"]), width],
                    hidden_bias_shape=[width] if family == "tanh" else None,
                )
            )
    return dict(
        schema="nmn.architecture-validation.v1",
        status="valid-topology",
        configuration=normalized,
        configuration_sha256=hashlib.sha256(
            json.dumps(normalized, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        dimensions=dimensions,
        execution=dict(
            encoder="copy inputs into named slots; other slots initially zero",
            layer="simultaneous reads from incoming state; sum residual writes",
            readout="select named output slots",
        ),
        limitations=[
            "Topology only: parameters, dtype/device and interventions are separate.",
            "Backend-reserved module attribute names are checked when constructing the model.",
            "No semantic correspondence, task competence or protection guarantee is established.",
        ],
    )
