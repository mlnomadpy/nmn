"""Lazy CLI entry point for native PyTorch research workflows."""

import argparse
import json
import sys
from pathlib import Path
from typing import cast


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nmn research native")
    commands = parser.add_subparsers(dest="command", required=True)
    dashboard = commands.add_parser(
        "report", help="build an offline evidence dashboard"
    )
    dashboard.add_argument("sources", nargs="+", type=Path)
    dashboard.add_argument("--output", type=Path, required=True)
    export = commands.add_parser(
        "export", help="write an Obsidian note and exact native data copy"
    )
    export.add_argument("record", type=Path)
    export.add_argument("--output", type=Path, required=True)
    integrity = commands.add_parser(
        "verify-export", help="check exported native files without replay"
    )
    integrity.add_argument("directory", type=Path)
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
    checkpoint = commands.add_parser(
        "checkpoint", help="extract one selected training checkpoint"
    )
    checkpoint.add_argument("--run", type=Path, required=True)
    checkpoint.add_argument("--seed", type=int, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    training = commands.add_parser(
        "train", help="explicitly run a bounded native training protocol"
    )
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--targets", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--contract", type=Path, required=True)
    training.add_argument("--target-provenance", required=True)
    training.add_argument(
        "--seeds", default="0", help="comma-separated minibatch sampling seeds"
    )
    training.add_argument("--pairs", type=Path)
    training.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark", help="compare saved native models under one replay contract"
    )
    benchmark.add_argument(
        "--models",
        type=Path,
        required=True,
        help="JSON map of method names to model paths",
    )
    benchmark.add_argument("--dataset", type=Path, required=True)
    benchmark.add_argument("--edits", type=Path, required=True)
    benchmark.add_argument(
        "--expected", type=Path, help="optional baseline target matrix JSON"
    )
    benchmark.add_argument("--protected", nargs="*", default=[])
    benchmark.add_argument("--split")
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--output", type=Path, required=True)
    diagnose = commands.add_parser(
        "diagnose", help="inspect a module kernel and sensor geometry"
    )
    diagnose.add_argument("--module", required=True)
    diagnose.add_argument("--noise-radius", type=float, default=0.0)
    protection = commands.add_parser(
        "protect", help="measure declared classification protection tasks"
    )
    protection.add_argument("--edits", type=Path, required=True)
    protection.add_argument("--tasks", type=Path, required=True)
    protection.add_argument("--provenance", required=True)
    protection.add_argument("--strata", nargs="*", default=[])
    donor = commands.add_parser(
        "donor", help="run declared donor pairs and reference labels"
    )
    donor.add_argument("--pairs", type=Path, required=True)
    donor.add_argument("--protected", nargs="*", default=[])
    donor.add_argument("--match-semantics", nargs="*", default=[])
    donor.add_argument("--allow-cross-split", action="store_true")
    for command in (collect, path, donor, diagnose, protection):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    for command in (collect, path, diagnose, protection):
        command.add_argument("--split", help="restrict to one named dataset split")
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "output") and args.output.exists():
            raise ValueError("output already exists; choose a new evidence path")
        if args.command == "report":
            from .dashboard import build_dashboard

            print(json.dumps(build_dashboard(args.sources, args.output)))
            return 0
        if args.command == "verify-export":
            from .native_export import verify_native_export

            print(json.dumps(verify_native_export(args.directory)))
            return 0
        if args.command == "export":
            from .native_export import export_native_record

            print(json.dumps(export_native_record(args.record, args.output)))
            return 0
        # Keep --help and the base CLI available without optional ML backends.
        import torch

        from ..torch import Intervention, ThreeNeuronYat, YatExpansion, YatGraph
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
                    "trainability",
                )
            }
            result.update(
                {
                    "schema": "nmn.native-model.v1",
                    "initialization_seed": args.seed,
                    "trained": False,
                }
            )
        elif args.command == "checkpoint":
            record = _read(args.run)
            if record.get("schema") != "nmn.native-training.v1":
                raise ValueError("expected a native training record")
            matches = [run for run in record["runs"] if run["seed"] == args.seed]
            if len(matches) != 1 or matches[0]["selected_checkpoint"] is None:
                raise ValueError("seed has no unique valid selected checkpoint")
            result = matches[0]["selected_checkpoint"]
            model_from_snapshot(result)
        elif args.command == "train":
            from ..torch.training import TrainingConfig, train_native

            result = train_native(
                _read(args.model),
                ResearchDataset.from_dict(_read(args.dataset)),
                _read(args.targets),
                config=TrainingConfig(**_read(args.config)),
                architecture_contract=_read(args.contract),
                target_provenance=args.target_provenance,
                seeds=[int(seed) for seed in args.seeds.split(",")],
                pairs=(
                    []
                    if args.pairs is None
                    else [DonorPair(**p) for p in _read(args.pairs)]
                ),
            )
        elif args.command == "benchmark":
            from ..torch.benchmark import benchmark_models

            paths = _read(args.models)
            models = {
                name: model_from_snapshot(_read(args.models.parent / path))
                for name, path in paths.items()
            }
            controls = _read(args.edits)
            edits = {
                name: {
                    state: Intervention(**control) for state, control in mapping.items()
                }
                for name, mapping in controls.items()
            }
            result = benchmark_models(
                models,
                ResearchDataset.from_dict(_read(args.dataset)),
                edits=edits,
                expected_outputs=(
                    None if args.expected is None else _read(args.expected)
                ),
                protected_outputs=args.protected,
                split=args.split,
                repeats=args.repeats,
                warmup=args.warmup,
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
            if args.command == "protect":
                from ..torch.protection import protection_study

                edits = {
                    name: {
                        state: Intervention(**control)
                        for state, control in mapping.items()
                    }
                    for name, mapping in _read(args.edits).items()
                }
                result = protection_study(
                    model,
                    dataset,
                    edits=edits,
                    tasks=_read(args.tasks),
                    provenance=args.provenance,
                    split=args.split,
                    strata=args.strata,
                )
            elif args.command == "donor":
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
                if args.command == "diagnose":
                    import hashlib

                    from ..torch import diagnostics

                    if args.module not in model.state_names:
                        raise ValueError("unknown diagnostic module")
                    block = cast(
                        YatExpansion,
                        (
                            model.blocks[args.module]
                            if isinstance(model, YatGraph)
                            else getattr(model, args.module)
                        ),
                    )
                    if not isinstance(block, YatExpansion):
                        raise ValueError(
                            "diagnose supports yat modules; collect exposes baseline feature geometry"
                        )
                    with torch.no_grad():
                        _, trace = model.forward_with_trace(inputs)
                        points = trace[f"{args.module}.input"]
                    result = {
                        "schema": "nmn.kernel-diagnostics.v1",
                        "dataset": dataset.to_dict(),
                        "sample_ids": ids,
                        "module": args.module,
                        "model_snapshot": collect_research_data(
                            model,
                            inputs,
                            sample_ids=ids,
                            metadata=metadata,
                            derivatives=False,
                        ),
                        "layer": _json_value(diagnostics.diagnose_layer(block, points)),
                        "sensors": _json_value(
                            diagnostics.sensor_diagnostics(
                                block.centers,
                                points,
                                epsilon=block.kernel.epsilon,
                                noise_radius=args.noise_radius,
                            )
                        ),
                        "source_sha256": hashlib.sha256(
                            Path(diagnostics.__file__).read_bytes()
                        ).hexdigest(),
                    }
                elif args.command == "collect":
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
        summary = {
            "status": "written",
            "output": str(args.output),
            "schema": result["schema"],
        }
        if result["schema"] == "nmn.native-training.v1":
            summary["runs"] = [
                {
                    key: run[key]
                    for key in ("seed", "status", "steps_completed", "best_step")
                }
                for run in result["runs"]
            ]
        print(json.dumps(summary))
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
