"""CLI for the deliberately bounded three-neuron reference experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .comparison import compare
from .reference import architecture, experiment, rational, replay, trace, write_bundle


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nmn research")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="describe the fixed three-neuron architecture")
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
    reproduce = commands.add_parser("reproduce", help="check hashes and rerun a bundle")
    reproduce.add_argument("bundle", type=Path)
    export = commands.add_parser(
        "export", help="export a verified bundle into a new vault folder"
    )
    export.add_argument("bundle", type=Path)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = architecture()
        elif args.command == "trace":
            result = trace(
                rational(args.u),
                rational(args.v),
                rational(args.gate),
                None if args.replace_h is None else rational(args.replace_h),
            )
        elif args.command == "compare":
            result = compare(
                rational(args.u),
                rational(args.v),
                rational(args.gate),
                None if args.replace_h is None else rational(args.replace_h),
            )
        elif args.command == "verify":
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
            result = replay(args.bundle)
        elif args.command == "export":
            replay(args.bundle)
            write_bundle(args.output)
            result = {"status": "exported", "report": str(args.output / "Report.md")}
        else:
            write_bundle(args.output)
            result = {
                "status": "created",
                "bundle": str(args.output),
                "results": experiment()["results"],
            }
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"research error: {exc}", file=sys.stderr)
        return 2
