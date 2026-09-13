"""Lazy CLI entry point for native PyTorch research workflows."""

import argparse
import json
import sys
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nmn research native")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "init", help="write a native reference or graph model"
    )
    initialize.add_argument("--graph", type=Path, help="YatGraph configuration JSON")
    initialize.add_argument("--seed", type=int, default=0)
    initialize.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser(
        "inspect", help="show a saved native model's structure"
    )
    inspect.add_argument("--model", type=Path, required=True)
    collect = commands.add_parser(
        "collect", help="collect native observations on a dataset"
    )
    collect.add_argument("--edits", type=Path, help="named intervention mappings JSON")
    collect.add_argument("--no-derivatives", action="store_true")
    path = commands.add_parser("path", help="measure a joint gate path")
    path.add_argument(
        "--start", required=True, help="comma-separated gates in module order"
    )
    path.add_argument(
        "--end", required=True, help="comma-separated gates in module order"
    )
    path.add_argument("--steps", type=int, default=32)
    donor = commands.add_parser(
        "donor", help="run declared donor pairs and reference labels"
    )
    donor.add_argument("--pairs", type=Path, required=True)
    donor.add_argument("--protected", nargs="*", default=[])
    donor.add_argument("--match-semantics", nargs="*", default=[])
    donor.add_argument("--allow-cross-split", action="store_true")
    for command in (collect, path, donor):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    for command in (collect, path):
        command.add_argument("--split", help="restrict to one named dataset split")
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "output") and args.output.exists():
            raise ValueError("output already exists; choose a new evidence path")
        # Keep --help and the base CLI available without optional ML backends.
        import torch

        from ..torch import Intervention, ThreeNeuronYat, YatGraph
        from ..torch.paths import gate_path
        from ..torch.research import (
            _json_value,
            collect_research_data,
            model_from_snapshot,
            save_research_data,
        )
        from ..torch.studies import donor_study
        from .datasets import DonorPair, ResearchDataset

        if args.command == "init":
            torch.manual_seed(args.seed)
            model = (
                YatGraph.from_configuration(_read(args.graph), dtype=torch.float64)
                if args.graph
                else ThreeNeuronYat.reference(dtype=torch.float64)
            )
            width = len(model.input_names) if isinstance(model, YatGraph) else 2
            data = collect_research_data(
                model,
                torch.zeros(1, width, dtype=torch.float64),
                sample_ids=["initialization-placeholder"],
                derivatives=False,
            )
            result = {
                key: data[key]
                for key in (
                    "configuration",
                    "parameters",
                    "model_sha256",
                    "source_sha256",
                )
            }
            result.update(
                {
                    "schema": "nmn.native-model.v1",
                    "initialization_seed": args.seed,
                    "trained": False,
                }
            )
        else:
            model = model_from_snapshot(_read(args.model))
            if args.command == "inspect":
                print(
                    json.dumps(
                        {
                            "model_sha256": _read(args.model)["model_sha256"],
                            "module_names": model.state_names,
                            "output_names": model.output_names,
                            "parameter_count": sum(
                                p.numel() for p in model.parameters()
                            ),
                            "configuration": _read(args.model)["configuration"],
                            "dependencies": (
                                model.dependencies()
                                if isinstance(model, YatGraph)
                                else None
                            ),
                        },
                        indent=2,
                    )
                )
                return 0
            dataset = ResearchDataset.from_dict(_read(args.dataset))
            if args.command == "donor":
                pairs = [DonorPair(**pair) for pair in _read(args.pairs)]
                result = donor_study(
                    model,
                    dataset,
                    pairs,
                    protected_outputs=args.protected,
                    match_semantics=args.match_semantics,
                    allow_cross_split=args.allow_cross_split,
                )
            else:
                ids = dataset.sample_ids(split=args.split)
                if not ids:
                    raise ValueError("selected dataset split has no samples")
                inputs = torch.tensor(
                    [dataset.sample(sid).inputs for sid in ids], dtype=torch.float64
                )
                metadata = {
                    "dataset_sha256": dataset.sha256,
                    "split": args.split,
                    "dataset_provenance": dataset.to_dict()["provenance"],
                }
                if args.command == "collect":
                    controls = {} if args.edits is None else _read(args.edits)
                    edits = {
                        name: {
                            state: Intervention(**control)
                            for state, control in mapping.items()
                        }
                        for name, mapping in controls.items()
                    }
                    result = collect_research_data(
                        model,
                        inputs,
                        sample_ids=ids,
                        edits=edits,
                        metadata=metadata,
                        derivatives=not args.no_derivatives,
                    )
                    result["dataset"] = dataset.to_dict()
                else:
                    snapshot = collect_research_data(
                        model,
                        inputs,
                        sample_ids=ids,
                        metadata=metadata,
                        derivatives=False,
                    )
                    result = {
                        "schema": "nmn.gate-path-study.v1",
                        "dataset": dataset.to_dict(),
                        "model_snapshot": snapshot,
                        "path": _json_value(
                            gate_path(
                                model,
                                inputs,
                                [float(v) for v in args.start.split(",")],
                                [float(v) for v in args.end.split(",")],
                                steps=args.steps,
                            )
                        ),
                    }
        save_research_data(result, args.output)
        print(
            json.dumps(
                {
                    "status": "written",
                    "output": str(args.output),
                    "schema": result["schema"],
                }
            )
        )
        return 0
    except ImportError as exc:
        print(f"native research requires nmn[torch]: {exc}", file=sys.stderr)
        return 2
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        AttributeError,
    ) as exc:
        print(f"native research error: {exc}", file=sys.stderr)
        return 2
