"""CLI for the deliberately bounded three-neuron reference experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .bundles import create_configured
from .comparison import compare
from .contract_bundles import create_contract_bundle
from .contract_bundles import export as export_bundle
from .contract_bundles import reproduce as reproduce_bundle
from .contracts import check, default_contract, load_contract
from .inspection import inspect_model
from .model import default_model, load, model_compare, model_trace, model_verify
from .reference import experiment, rational, trace, write_bundle


def main(argv: Optional[List[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "native":
        from .native_cli import main as native_main

        return native_main(arguments[1:])
    parser = argparse.ArgumentParser(prog="nmn research")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "native", help="native model data, donor studies and gate paths"
    )
    inspection = commands.add_parser(
        "inspect", help="describe the three-neuron architecture"
    )
    inspection.add_argument("--model", type=Path)
    model = commands.add_parser(
        "model", help="print the default model or validate a JSON model"
    )
    model.add_argument("path", type=Path, nargs="?")
    tracing = commands.add_parser("trace", help="exact forward and intervention trace")
    tracing.add_argument("--u", default="1")
    tracing.add_argument("--v", default="1")
    tracing.add_argument("--gate", default="1")
    tracing.add_argument(
        "--replace-h", help="replace h after gating at the layer-one cut"
    )
    comparison = commands.add_parser(
        "compare", help="compare a native edit with the unedited model"
    )
    comparison.add_argument("--u", default="1")
    comparison.add_argument("--v", default="1")
    comparison.add_argument("--gate", default="0")
    comparison.add_argument("--replace-h", help="replace h after gating")
    verify = commands.add_parser("verify", help="exhaust the fixed 25-input contract")
    verify.add_argument(
        "--leaky", action="store_true", help="check the failing p+y readout"
    )
    demo = commands.add_parser(
        "demo", help="write evidence, failure witness and report"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--model", type=Path)
    demo.add_argument("--contract", type=Path)
    demo.add_argument("--max-cases", type=int)
    reproduce = commands.add_parser("reproduce", help="check hashes and rerun a bundle")
    reproduce.add_argument("bundle", type=Path)
    export = commands.add_parser(
        "export", help="export a verified bundle into a new vault folder"
    )
    export.add_argument("bundle", type=Path)
    export.add_argument("--output", type=Path, required=True)
    for command_parser in (tracing, comparison, verify):
        command_parser.add_argument("--model", type=Path)
    contract_parser = commands.add_parser(
        "contract", help="emit or validate a finite intervention contract"
    )
    contract_parser.add_argument("path", type=Path, nargs="?")
    verify.add_argument("--contract", type=Path)
    verify.add_argument("--max-cases", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            result = load_contract(args.path) if args.path else default_contract()
        elif args.command == "model":
            result = load(args.path) if args.path else default_model()
        elif args.command == "inspect":
            result = inspect_model(load(args.model) if args.model else default_model())
        elif args.command == "trace":
            evaluator = (
                trace
                if args.model is None
                else lambda *values: model_trace(load(args.model), *values)
            )
            result = evaluator(
                rational(args.u),
                rational(args.v),
                rational(args.gate),
                None if args.replace_h is None else rational(args.replace_h),
            )
        elif args.command == "compare":
            comparator = (
                compare
                if args.model is None
                else lambda *values: model_compare(load(args.model), *values)
            )
            result = comparator(
                rational(args.u),
                rational(args.v),
                rational(args.gate),
                None if args.replace_h is None else rational(args.replace_h),
            )
        elif args.command == "verify":
            if args.contract:
                if args.leaky:
                    raise ValueError(
                        "set protected_leak in the model instead of --leaky"
                    )
                evidence = check(
                    load(args.model) if args.model else default_model(),
                    load_contract(args.contract),
                    4096 if args.max_cases is None else args.max_cases,
                )
                print(json.dumps(evidence, indent=2))
                return {
                    "certified-under-assumptions": 0,
                    "counterexample-found": 1,
                    "inconclusive": 3,
                }[evidence["status"]]
            if args.max_cases is not None:
                raise ValueError("--max-cases requires --contract")
            if args.model:
                if args.leaky:
                    raise ValueError(
                        "--leaky applies only to the fixed reference; set protected_leak in the model"
                    )
                configured = model_verify(load(args.model))
                print(json.dumps(configured, indent=2))
                return 1 if configured["protected_violations"] else 0
            data = experiment()
            result = {
                "contract": data["contract"],
                "results": data["results"],
                "witness": data["witness"],
            }
            print(json.dumps(result, indent=2))
            if args.leaky:
                return 1 if data["results"]["failure_variant_violations"] else 0
            return (
                0
                if data["results"]["protected_violations"] == 0
                and data["results"]["target_witness_pass"]
                else 1
            )
        elif args.command == "reproduce":
            result = reproduce_bundle(args.bundle)
        elif args.command == "export":
            result = export_bundle(args.bundle, args.output)
        elif args.contract:
            result = create_contract_bundle(
                load(args.model) if args.model else default_model(),
                load_contract(args.contract),
                args.output,
                4096 if args.max_cases is None else args.max_cases,
            )
        elif args.max_cases is not None:
            raise ValueError("--max-cases requires --contract")
        elif args.model:
            result = create_configured(load(args.model), args.output)
        else:
            write_bundle(args.output)
            result = {
                "status": "created",
                "bundle": str(args.output),
                "results": experiment()["results"],
            }
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
        print(f"research error: {exc}", file=sys.stderr)
        return 2
