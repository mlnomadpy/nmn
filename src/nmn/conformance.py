"""Import-light access to NMN's cross-framework semantic contract."""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from importlib.resources import files
from typing import Any, Mapping, cast

BACKENDS = ("torch", "nnx", "linen", "tf", "keras", "mlx")
OPERATIONS = (
    "dense",
    "embed",
    "convolution",
    "transpose_convolution",
    "attention",
    "may",
    "ray",
)
DTYPES = ("float64", "float32", "bfloat16", "float16")
EXECUTION_MODES = {"eager", "compile", "jit", "function", "compiled_call"}
SUPPORT_STATES = {"supported", "partial", "unsupported"}
CONFORMANCE_STATES = {"oracle", "fixture", "declared"}
EVIDENCE_KINDS = CONFORMANCE_STATES
KERAS_ENGINES = {"not_applicable", "tensorflow", "jax", "torch"}
CONFIG_KEYS = {
    "dense": {"case", "in_features", "out_features", "bias", "lazy"},
    "embed": {
        "case",
        "num_embeddings",
        "features",
        "lookup_shape",
        "query_length",
        "epsilon",
    },
    "convolution": {
        "case",
        "spatial_rank",
        "input_channels",
        "output_channels",
        "kernel_size",
        "bias",
        "alpha",
        "learnable_epsilon",
    },
    "transpose_convolution": {
        "case",
        "spatial_rank",
        "input_channels",
        "output_channels",
        "kernel_size",
        "groups",
        "bias",
        "alpha",
        "learnable_epsilon",
    },
    "attention": {"case", "num_heads", "head_features", "causal"},
    "may": {
        "case",
        "num_heads",
        "head_features",
        "num_features",
        "causal",
        "fixed_projection_fixture",
        "gradient_scope",
        "denominator_profile",
        "semantic_limitations",
    },
    "ray": {
        "case",
        "num_heads",
        "head_features",
        "sketch_m",
        "num_radial",
        "radial_dim",
        "causal",
        "fixed_projection_fixture",
        "gradient_scope",
        "denominator_profile",
        "semantic_limitations",
    },
}
CONFIG_CASES = {
    "dense": "dense_v1",
    "embed": "embed_lookup_and_attend_v1",
    "convolution": "convolution_1d_valid_v1",
    "transpose_convolution": "transpose_convolution_1d_valid_v1",
    "attention": "attention_v1",
    "may": "may_v1",
    "ray": "ray_v1",
}


class ContractError(ValueError):
    """Raised when a conformance manifest violates the public schema."""


def load_contract() -> dict[str, Any]:
    """Load and validate the packaged cross-framework contract.

    This function uses only the Python standard library and never imports an ML
    framework, so tooling can inspect the matrix from a base ``nmn`` install.
    """
    resource = files("nmn").joinpath("conformance_manifest.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))
    validate_contract(contract)
    return cast(dict[str, Any], contract)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be an object")
    return value


def _require_keys(
    value: Mapping[str, Any],
    keys: set[str],
    path: str,
    *,
    optional: set[str] | None = None,
) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise ContractError(f"{path} is missing required keys: {', '.join(missing)}")
    unexpected = sorted(value.keys() - keys - (optional or set()))
    if unexpected:
        raise ContractError(f"{path} has unexpected keys: {', '.join(unexpected)}")


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ContractError(f"{path} must be a nonempty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{path} must be a boolean")
    return value


def _string_list(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ContractError(f"{path} must be a list of nonempty strings")
    if nonempty and not value:
        raise ContractError(f"{path} must not be empty")
    if len(value) != len(set(value)):
        raise ContractError(f"{path} must not contain duplicates")
    return value


def _subset(values: list[str], allowed: Collection[str], path: str) -> None:
    unknown = set(values) - set(allowed)
    if unknown:
        raise ContractError(f"{path} has unsupported values: {sorted(unknown)}")


def _validate_coverage(
    value: Any, path: str, allowed: list[str]
) -> tuple[list[str], list[str]]:
    coverage = _mapping(value, path)
    _require_keys(coverage, {"supported", "tested"}, path)
    supported = _string_list(coverage["supported"], f"{path}.supported")
    tested = _string_list(coverage["tested"], f"{path}.tested")
    _subset(supported, allowed, f"{path}.supported")
    _subset(tested, supported, f"{path}.tested")
    return supported, tested


def _validate_config(
    config: Mapping[str, Any],
    *,
    path: str,
    backend_name: str,
    operation_name: str,
    operation: Mapping[str, Any],
) -> None:
    optional = {"key_padding_executed"} if backend_name == "nnx" else set()
    _require_keys(config, CONFIG_KEYS[operation_name], path, optional=optional)
    if config["case"] != CONFIG_CASES[operation_name]:
        raise ContractError(f"{path}.case must be {CONFIG_CASES[operation_name]!r}")
    if operation_name in {"convolution", "transpose_convolution"}:
        if config["spatial_rank"] != 1:
            raise ContractError(
                f"{path}.spatial_rank must be 1 for "
                f"{CONFIG_CASES[operation_name]!r}"
            )
    integer_keys = CONFIG_KEYS[operation_name] & {
        "in_features",
        "out_features",
        "num_embeddings",
        "features",
        "query_length",
        "spatial_rank",
        "input_channels",
        "output_channels",
        "kernel_size",
        "groups",
        "num_heads",
        "head_features",
        "num_features",
        "sketch_m",
        "num_radial",
        "radial_dim",
    }
    for key in integer_keys:
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ContractError(f"{path}.{key} must be a positive integer")
    for key in CONFIG_KEYS[operation_name] & {
        "bias",
        "lazy",
        "alpha",
        "learnable_epsilon",
        "causal",
    }:
        _boolean(config[key], f"{path}.{key}")
    if operation_name == "embed":
        lookup_shape = config["lookup_shape"]
        if (
            not isinstance(lookup_shape, list)
            or not lookup_shape
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in lookup_shape
            )
        ):
            raise ContractError(f"{path}.lookup_shape must contain positive integers")
        if config["epsilon"] != "static":
            raise ContractError(f"{path}.epsilon must be 'static'")
    if operation_name in {"may", "ray"}:
        if config["fixed_projection_fixture"] != "linear_attention_v1":
            raise ContractError(
                f"{path}.fixed_projection_fixture must be 'linear_attention_v1'"
            )
        if config["denominator_profile"] != "positive":
            raise ContractError(f"{path}.denominator_profile must be 'positive'")
        gradients = _string_list(
            config["gradient_scope"], f"{path}.gradient_scope", nonempty=True
        )
        if gradients != operation["gradients"]:
            raise ContractError(f"{path}.gradient_scope must match operation gradients")
        _string_list(
            config["semantic_limitations"],
            f"{path}.semantic_limitations",
            nonempty=True,
        )
    if "key_padding_executed" in config:
        _boolean(config["key_padding_executed"], f"{path}.key_padding_executed")
        if operation_name not in {"may", "ray"}:
            raise ContractError(
                f"{path}.key_padding_executed is valid only for MAY/RAY"
            )


def _validate_profile(
    profile: Any,
    *,
    path: str,
    backend_name: str,
    operation_name: str,
    backend: Mapping[str, Any],
    operation: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> None:
    profile = _mapping(profile, path)
    _require_keys(
        profile,
        {
            "dtypes",
            "modes",
            "keras_engine",
            "config",
            "layout",
            "masking",
            "serialization",
            "evidence",
        },
        path,
    )
    supported_dtypes, tested_dtypes = _validate_coverage(
        profile["dtypes"], f"{path}.dtypes", backend["dtypes"]
    )
    supported_modes, tested_modes = _validate_coverage(
        profile["modes"], f"{path}.modes", backend["execution"]
    )
    if capability["status"] in {"supported", "partial"}:
        if not supported_dtypes or not supported_modes:
            raise ContractError(
                f"{path}.{capability['status']} capability must name supported "
                "dtypes and modes"
            )
    keras_engine = _string(profile["keras_engine"], f"{path}.keras_engine")
    if keras_engine not in KERAS_ENGINES:
        raise ContractError(f"{path}.keras_engine is invalid")
    expected_engine = "tensorflow" if backend_name == "keras" else "not_applicable"
    if keras_engine != expected_engine:
        raise ContractError(f"{path}.keras_engine must be {expected_engine!r}")
    config = _mapping(profile["config"], f"{path}.config")
    if not config:
        raise ContractError(f"{path}.config must not be empty")
    _validate_config(
        config,
        path=f"{path}.config",
        backend_name=backend_name,
        operation_name=operation_name,
        operation=operation,
    )
    _string(profile["layout"], f"{path}.layout")
    masking = _string_list(profile["masking"], f"{path}.masking")
    _subset(masking, backend["masking"][operation["mask_key"]], f"{path}.masking")
    serialization = _string_list(
        profile["serialization"], f"{path}.serialization", nonempty=True
    )
    _subset(serialization, backend["serialization"], f"{path}.serialization")

    evidence = _mapping(profile["evidence"], f"{path}.evidence")
    _require_keys(evidence, {"kind", "tests", "fixture"}, f"{path}.evidence")
    evidence_kind = _string(evidence["kind"], f"{path}.evidence.kind")
    if evidence_kind not in EVIDENCE_KINDS:
        raise ContractError(f"{path}.evidence.kind is invalid")
    tests = _string_list(evidence["tests"], f"{path}.evidence.tests")
    fixture = evidence["fixture"]
    if fixture is not None:
        _string(fixture, f"{path}.evidence.fixture")
    if evidence_kind != capability["conformance"]:
        raise ContractError(f"{path}.evidence.kind must match capability conformance")
    if evidence_kind == "declared":
        if tested_dtypes or tested_modes or tests or fixture is not None:
            raise ContractError(
                f"{path}.declared evidence must not claim test coverage"
            )
        if capability["status"] in {"supported", "partial"}:
            raise ContractError(
                f"{path}.{capability['status']} capability must use oracle or "
                "fixture evidence"
            )
    else:
        if not tested_dtypes or not tested_modes or not tests:
            raise ContractError(
                f"{path}.verified evidence must name tested dtypes, modes, and tests"
            )
        if evidence_kind == "fixture" and fixture is None:
            raise ContractError(f"{path}.fixture evidence must name a fixture")
        if evidence_kind == "oracle" and fixture is not None:
            raise ContractError(f"{path}.oracle evidence must not name a fixture")
    if capability["status"] == "unsupported":
        if (
            supported_dtypes
            or supported_modes
            or tested_dtypes
            or tested_modes
            or tests
        ):
            raise ContractError(
                f"{path}.unsupported capability must not claim support or tests"
            )


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Validate a manifest without importing optional backend dependencies.

    The validator deliberately rejects malformed and internally inconsistent
    contracts.  A declared API is not evidence: only ``oracle`` or ``fixture``
    cells may list tests, tested dtypes, or tested execution modes.
    """
    contract = _mapping(contract, "contract")
    _require_keys(
        contract,
        {
            "schema_version",
            "contract_version",
            "canonical",
            "tolerances",
            "operations",
            "backends",
            "profiles",
        },
        "contract",
    )
    if contract["schema_version"] != 1:
        raise ContractError("contract.schema_version must be 1")
    _string(contract["contract_version"], "contract.contract_version")
    canonical = _mapping(contract["canonical"], "contract.canonical")
    _require_keys(
        canonical,
        {
            "array_format",
            "oracle_dtype",
            "kernel_layout",
            "fixture_format",
            "random_seed",
        },
        "contract.canonical",
    )
    for key in ("array_format", "oracle_dtype", "kernel_layout", "fixture_format"):
        _string(canonical[key], f"contract.canonical.{key}")
    if not isinstance(canonical["random_seed"], int) or isinstance(
        canonical["random_seed"], bool
    ):
        raise ContractError("contract.canonical.random_seed must be an integer")

    tolerances = _mapping(contract["tolerances"], "contract.tolerances")
    if tuple(tolerances) != DTYPES:
        raise ContractError(f"contract.tolerances must be exactly {DTYPES!r}")
    for dtype in DTYPES:
        tolerance = _mapping(tolerances[dtype], f"contract.tolerances.{dtype}")
        _require_keys(tolerance, {"rtol", "atol"}, f"contract.tolerances.{dtype}")
        for key in ("rtol", "atol"):
            value = tolerance[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ContractError(
                    f"contract.tolerances.{dtype}.{key} must be finite and nonnegative"
                )

    operations = _mapping(contract["operations"], "contract.operations")
    backends = _mapping(contract["backends"], "contract.backends")
    profiles = _mapping(contract["profiles"], "contract.profiles")
    if tuple(operations) != OPERATIONS:
        raise ContractError(f"contract.operations must be exactly {OPERATIONS!r}")
    if tuple(backends) != BACKENDS:
        raise ContractError(f"contract.backends must be exactly {BACKENDS!r}")
    if tuple(profiles) != BACKENDS:
        raise ContractError(f"contract.profiles must be exactly {BACKENDS!r}")

    for operation_name, operation_value in operations.items():
        operation = _mapping(operation_value, f"contract.operations.{operation_name}")
        operation_path = f"contract.operations.{operation_name}"
        _require_keys(
            operation,
            {"oracle", "outputs", "gradients", "masking", "layout", "mask_key"},
            operation_path,
        )
        _string(operation["oracle"], f"{operation_path}.oracle")
        _string_list(operation["outputs"], f"{operation_path}.outputs", nonempty=True)
        _string_list(operation["gradients"], f"{operation_path}.gradients")
        _string_list(operation["masking"], f"{operation_path}.masking")
        _string(operation["layout"], f"{operation_path}.layout")
        _string(operation["mask_key"], f"{operation_path}.mask_key")

    for backend_name, backend_value in backends.items():
        path = f"contract.backends.{backend_name}"
        backend = _mapping(backend_value, path)
        _require_keys(
            backend,
            {
                "display_name",
                "adapter",
                "ci_platform",
                "required_in_ci",
                "execution",
                "dtypes",
                "serialization",
                "native_layout",
                "masking",
                "operations",
            },
            path,
        )
        for key in ("display_name", "adapter", "ci_platform", "native_layout"):
            _string(backend[key], f"{path}.{key}")
        _boolean(backend["required_in_ci"], f"{path}.required_in_ci")
        execution = _string_list(
            backend["execution"], f"{path}.execution", nonempty=True
        )
        _subset(execution, EXECUTION_MODES, f"{path}.execution")
        backend_dtypes = _string_list(
            backend["dtypes"], f"{path}.dtypes", nonempty=True
        )
        _subset(backend_dtypes, DTYPES, f"{path}.dtypes")
        _string_list(backend["serialization"], f"{path}.serialization", nonempty=True)
        masking = _mapping(backend["masking"], f"{path}.masking")
        if set(masking) != {"attention", "may", "ray", "none"}:
            raise ContractError(
                f"{path}.masking must declare attention, may, ray, and none"
            )
        for mask_key, modes in masking.items():
            _string_list(modes, f"{path}.masking.{mask_key}")
        backend_operations = _mapping(backend["operations"], f"{path}.operations")
        if set(backend_operations) != set(OPERATIONS):
            raise ContractError(f"{path}.operations must declare every operation")
        backend_profiles = _mapping(
            profiles[backend_name], f"contract.profiles.{backend_name}"
        )
        if set(backend_profiles) != set(OPERATIONS):
            raise ContractError(
                f"contract.profiles.{backend_name} must declare every operation"
            )
        for operation_name in OPERATIONS:
            capability_path = f"{path}.operations.{operation_name}"
            capability = _mapping(backend_operations[operation_name], capability_path)
            required_capability_keys = {"status", "conformance", "api"}
            optional_capability_keys = {"limitations"}
            if operation_name in {"convolution", "transpose_convolution"}:
                required_capability_keys.add("declared_api")
            else:
                optional_capability_keys.add("declared_api")
            _require_keys(
                capability,
                required_capability_keys,
                capability_path,
                optional=optional_capability_keys,
            )
            status = _string(capability["status"], f"{capability_path}.status")
            conformance = _string(
                capability["conformance"], f"{capability_path}.conformance"
            )
            if status not in SUPPORT_STATES:
                raise ContractError(f"{capability_path}.status is invalid")
            if conformance not in CONFORMANCE_STATES:
                raise ContractError(f"{capability_path}.conformance is invalid")
            api_names = _string(capability["api"], f"{capability_path}.api").split("|")
            declared_api_names: list[str] = []
            if "declared_api" in capability:
                declared_api_names = _string(
                    capability["declared_api"], f"{capability_path}.declared_api"
                ).split("|")
            if "limitations" in capability:
                _string_list(
                    capability["limitations"],
                    f"{capability_path}.limitations",
                    nonempty=True,
                )
            if any(not name.startswith(f"nmn.{backend_name}.") for name in api_names):
                raise ContractError(
                    f"{capability_path}.api must contain fully qualified nmn.{backend_name} symbols"
                )
            if any(
                not name.startswith(f"nmn.{backend_name}.")
                for name in declared_api_names
            ):
                raise ContractError(
                    f"{capability_path}.declared_api must contain fully qualified "
                    f"nmn.{backend_name} symbols"
                )
            if set(api_names) & set(declared_api_names):
                raise ContractError(
                    f"{capability_path}.api and declared_api must not overlap"
                )
            operation = operations[operation_name]
            if operation["mask_key"] not in masking:
                raise ContractError(
                    f"contract.operations.{operation_name}.mask_key is unknown"
                )
            _validate_profile(
                backend_profiles[operation_name],
                path=f"contract.profiles.{backend_name}.{operation_name}",
                backend_name=backend_name,
                operation_name=operation_name,
                backend=backend,
                operation=operation,
                capability=capability,
            )


def render_support_markdown(contract: Mapping[str, Any] | None = None) -> str:
    """Render the human support table from the canonical manifest."""
    if contract is None:
        contract = load_contract()
    validate_contract(contract)
    lines = [
        "<!-- Generated by `python -m nmn.conformance`; do not edit by hand. -->",
        "# Cross-framework conformance contract",
        "",
        f"Contract version: `{contract['contract_version']}`.",
        "",
        "A cell is verified only when it names tested dtypes, execution modes, and evidence.",
        "`declared` means the public API is documented but is not yet conformance-tested.",
        "",
        "| Backend | CI | Execution | Dtypes | Serialization | Linear masks |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for backend in contract["backends"].values():
        required = "required" if backend["required_in_ci"] else "optional"
        lines.append(
            f"| {backend['display_name']} | {backend['ci_platform']} ({required}) | "
            f"{', '.join(backend['execution'])} | {', '.join(backend['dtypes'])} | "
            f"{', '.join(backend['serialization'])} | "
            f"MAY: {', '.join(backend['masking']['may'])}; RAY: {', '.join(backend['masking']['ray'])} |"
        )
    lines.extend(["", "## Operation coverage", ""])
    headings = [contract["backends"][name]["display_name"] for name in BACKENDS]
    lines.append("| Operation | " + " | ".join(headings) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in headings) + " |")
    for operation_name in OPERATIONS:
        cells = []
        for backend_name in BACKENDS:
            capability = contract["backends"][backend_name]["operations"][
                operation_name
            ]
            profile = contract["profiles"][backend_name][operation_name]
            tested = ", ".join(profile["dtypes"]["tested"]) or "not tested"
            cells.append(
                f"{capability['status']} / {capability['conformance']} ({tested})"
            )
        lines.append(f"| {operation_name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## API evidence scope",
            "",
            "Every convolution capability must name both its oracle-tested `api` "
            + "and its untested `declared_api` symbols; contract validation fails if "
            "either scope is absent.",
            "",
            "The 1D convolution symbols below are covered by the exact oracle profiles. "
            + "The 2D and 3D symbols are public APIs but remain declared until matching "
            "rank-specific profiles are added.",
            "",
        ]
    )
    for backend_name in BACKENDS:
        display_name = contract["backends"][backend_name]["display_name"]
        for operation_name in ("convolution", "transpose_convolution"):
            capability = contract["backends"][backend_name]["operations"][
                operation_name
            ]
            verified = capability["api"].replace("|", ", ")
            declared = capability["declared_api"].replace("|", ", ")
            lines.append(
                f"- {display_name} {operation_name}: oracle `{verified}`; "
                f"declared `{declared}`."
            )
    lines.extend(
        [
            "",
            "## Tolerance policy",
            "",
            "A tolerance is enforced only when at least one exact profile tests that dtype.",
        ]
    )
    lines.append("| Dtype | rtol | atol | Status |")
    lines.append("| --- | ---: | ---: | --- |")
    enforced_dtypes = {
        dtype
        for backend_profiles in contract["profiles"].values()
        for profile in backend_profiles.values()
        for dtype in profile["dtypes"]["tested"]
    }
    for dtype, tolerance in contract["tolerances"].items():
        status = "enforced" if dtype in enforced_dtypes else "declared, not tested"
        lines.append(
            f"| {dtype} | {tolerance['rtol']:.12g} | {tolerance['atol']:.12g} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print generated support documentation to stdout."""
    print(render_support_markdown(), end="")


if __name__ == "__main__":
    main()
