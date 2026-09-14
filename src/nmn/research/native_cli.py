"""Lazy CLI entry point for native PyTorch research workflows."""

import argparse
import json
import sys
from pathlib import Path
from typing import cast


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="nmn research native")
    commands = parser.add_subparsers(dest="command", required=True)
    dashboard = commands.add_parser(
        "report", help="build an offline evidence dashboard"
    )
    dashboard.add_argument("sources", nargs="*", type=Path)
    dashboard.add_argument(
        "--sources-file", type=Path, help="portable JSON list of local evidence paths"
    )
    dashboard.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser(
        "extract", help="list or extract reusable JSON components without a backend"
    )
    extract.add_argument("record", type=Path)
    extract.add_argument("--component", help="omit to list available components")
    extract.add_argument("--output", type=Path)
    export = commands.add_parser(
        "export", help="write an Obsidian note and exact native data copy"
    )
    export.add_argument("record", type=Path)
    export.add_argument("--output", type=Path, required=True)
    integrity = commands.add_parser(
        "verify-export", help="check exported native files without replay"
    )
    integrity.add_argument("directory", type=Path)
    box_verify = commands.add_parser(
        "verify-box", help="search a rational continuous range contract"
    )
    box_verify.add_argument("--model", type=Path, required=True)
    box_verify.add_argument("--contract", type=Path, required=True)
    box_verify.add_argument("--max-boxes", type=int, required=True)
    box_verify.add_argument("--output", type=Path, required=True)
    box_check = commands.add_parser(
        "check-box", help="check a saved rational box partition certificate"
    )
    box_check.add_argument("certificate", type=Path)
    box_check.add_argument("--output", type=Path, required=True)
    enclosure = commands.add_parser(
        "enclose",
        help="bound supported real-valued models with exact rational intervals",
    )
    enclosure.add_argument("--model", type=Path, required=True)
    enclosure.add_argument("--box", type=Path, required=True)
    enclosure.add_argument("--controls", type=Path)
    enclosure.add_argument(
        "--reference-controls",
        type=Path,
        help="bound edited-minus-reference output differences",
    )
    enclosure.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser(
        "replay", help="recompute supported native records on CPU"
    )
    replay.add_argument("record", type=Path)
    replay.add_argument("--atol", type=float, default=1e-10)
    replay.add_argument("--rtol", type=float, default=1e-8)
    replay.add_argument("--output", type=Path, required=True)
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
    response = commands.add_parser(
        "response-space", help="fit and evaluate a finite edit-response subspace"
    )
    response.add_argument("--edits", type=Path, required=True)
    response.add_argument("--rank", type=int, required=True)
    response.add_argument("--fit-split", default="tuning")
    response.add_argument("--evaluation-split", default="validation")
    selection = commands.add_parser(
        "select", help="select an edit then validate only the frozen candidate"
    )
    selection.add_argument("--candidates", type=Path, required=True)
    selection.add_argument("--targets", type=Path, required=True)
    selection.add_argument("--target-outputs", nargs="+", required=True)
    selection.add_argument("--protected-outputs", nargs="*", default=[])
    selection.add_argument("--protection-tolerance", type=float, required=True)
    selection.add_argument("--max-candidates", type=int, required=True)
    selection.add_argument("--provenance", required=True)
    selection.add_argument("--selection-split", default="tuning")
    selection.add_argument("--validation-split", default="validation")
    semantic = commands.add_parser(
        "semantics",
        help="compare supplied semantic references with native counterfactuals",
    )
    semantic.add_argument("--reference", type=Path, required=True)
    semantic.add_argument("--correspondence", type=Path, required=True)
    semantic.add_argument("--tolerance", type=float, default=1e-8)
    probe = commands.add_parser(
        "probe",
        help="fit a linear state probe then evaluate frozen decoding under edits",
    )
    probe.add_argument("--feature", required=True)
    probe.add_argument("--labels", type=Path, required=True)
    probe.add_argument("--classes", nargs="+", required=True)
    probe.add_argument("--provenance", required=True)
    probe.add_argument("--ridge", type=float, required=True)
    probe.add_argument("--edits", type=Path)
    probe.add_argument(
        "--refit-edits",
        action="store_true",
        help="also fit a separate probe on each edited fit population",
    )
    probe.add_argument("--fit-split", default="tuning")
    probe.add_argument("--evaluation-split", default="validation")
    fitted_reduction = commands.add_parser(
        "fit-reduction",
        help="fit a state summary then evaluate frozen maps on another split",
    )
    fitted_reduction.add_argument("--start-layer", type=int, required=True)
    fitted_reduction.add_argument("--rank", type=int, required=True)
    fitted_reduction.add_argument("--ridge", type=float, required=True)
    fitted_reduction.add_argument("--fit-split", default="tuning")
    fitted_reduction.add_argument("--evaluation-split", default="validation")
    edge = commands.add_parser(
        "edges", help="measure producer-specific residual read replacements"
    )
    edge.add_argument("--patches", type=Path, required=True)
    edge.add_argument("--provenance", required=True)
    reduction = commands.add_parser(
        "reduce", help="evaluate supplied state summaries and reduced dynamics"
    )
    reduction.add_argument("--maps", type=Path, required=True)
    reduction.add_argument("--start-layer", type=int, required=True)
    reduction.add_argument("--provenance", required=True)
    suffix = commands.add_parser(
        "suffix", help="measure downstream responses to supplied graph states"
    )
    suffix.add_argument("--states", type=Path, required=True)
    suffix.add_argument("--start-layer", type=int, required=True)
    suffix.add_argument("--provenance", required=True)
    coalition = commands.add_parser(
        "coalitions", help="replay a budgeted module-deletion lattice"
    )
    coalition.add_argument("--modules", nargs="+", required=True)
    coalition.add_argument("--max-evaluations", type=int, required=True)
    coalition.add_argument(
        "--background", type=Path, help="module to scalar background gate JSON"
    )
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
    donor.add_argument(
        "--read-slots",
        type=Path,
        help="optional receiving-module to read-slot lists JSON (YatGraph)",
    )
    donor.add_argument("--protected", nargs="*", default=[])
    donor.add_argument("--match-semantics", nargs="*", default=[])
    donor.add_argument("--allow-cross-split", action="store_true")
    for command in (
        collect,
        path,
        donor,
        diagnose,
        protection,
        coalition,
        suffix,
        reduction,
        edge,
        fitted_reduction,
        probe,
        semantic,
        selection,
        response,
    ):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    for command in (
        collect,
        path,
        diagnose,
        protection,
        coalition,
        suffix,
        reduction,
        edge,
    ):
        command.add_argument("--split", help="restrict to one named dataset split")
    args = parser.parse_args(argv)
    try:
        if getattr(args, "output", None) is not None and args.output.exists():
            raise ValueError("output already exists; choose a new evidence path")
        if args.command == "extract":
            from .components import extract_native_component, list_native_components

            if args.component is None:
                if args.output is not None:
                    raise ValueError("--output requires --component")
                print(json.dumps(list_native_components(args.record)))
            else:
                if args.output is None:
                    raise ValueError("--component requires a new --output directory")
                print(
                    json.dumps(
                        extract_native_component(
                            args.record, args.component, args.output
                        )
                    )
                )
            return 0
        if args.command == "report":
            from .dashboard import build_dashboard, load_dashboard_sources

            sources = list(args.sources)
            if args.sources_file is not None:
                sources.extend(load_dashboard_sources(args.sources_file))
            print(json.dumps(build_dashboard(sources, args.output)))
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

        if args.command == "verify-box":
            from ..torch.interval_contract import verify_box

            result = verify_box(
                _read(args.model), _read(args.contract), max_boxes=args.max_boxes
            )
        elif args.command == "check-box":
            from ..torch.interval_contract import check_box_certificate

            result = check_box_certificate(_read(args.certificate))
        elif args.command == "enclose":
            from ..torch.enclosure import enclose_native, enclose_native_difference

            arguments = dict(
                controls=None if args.controls is None else _read(args.controls)
            )
            if args.reference_controls is not None:
                result = enclose_native_difference(
                    _read(args.model),
                    _read(args.box),
                    reference_controls=_read(args.reference_controls),
                    **arguments,
                )
            else:
                result = enclose_native(_read(args.model), _read(args.box), **arguments)
        elif args.command == "replay":
            from ..torch.replay import replay_native_record

            result = replay_native_record(
                _read(args.record), atol=args.atol, rtol=args.rtol
            )
        elif args.command == "init":
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
            if args.command == "response-space":
                from ..torch.response_space import response_space_study

                edits = {
                    name: {
                        module: Intervention(**control)
                        for module, control in mapping.items()
                    }
                    for name, mapping in _read(args.edits).items()
                }
                result = response_space_study(
                    model,
                    dataset,
                    edits=edits,
                    rank=args.rank,
                    fit_split=args.fit_split,
                    evaluation_split=args.evaluation_split,
                )
            elif args.command == "select":
                from ..torch.selection import select_edit

                result = select_edit(
                    model,
                    dataset,
                    candidates=_read(args.candidates),
                    targets=_read(args.targets),
                    target_outputs=args.target_outputs,
                    protected_outputs=args.protected_outputs,
                    protection_tolerance=args.protection_tolerance,
                    provenance=args.provenance,
                    max_candidates=args.max_candidates,
                    selection_split=args.selection_split,
                    validation_split=args.validation_split,
                )
            elif args.command == "semantics":
                from ..torch.semantics import semantic_study
                from .semantics import TabulatedReference

                result = semantic_study(
                    model,
                    dataset,
                    reference=TabulatedReference(_read(args.reference)),
                    correspondence=_read(args.correspondence),
                    tolerance=args.tolerance,
                )
            elif args.command == "probe":
                from ..torch.probes import probe_study

                controls = {} if args.edits is None else _read(args.edits)
                result = probe_study(
                    model,
                    dataset,
                    feature=args.feature,
                    labels=_read(args.labels),
                    classes=args.classes,
                    provenance=args.provenance,
                    ridge=args.ridge,
                    refit_edits=args.refit_edits,
                    edits={
                        name: {
                            module: Intervention(**control)
                            for module, control in values.items()
                        }
                        for name, values in controls.items()
                    },
                    fit_split=args.fit_split,
                    evaluation_split=args.evaluation_split,
                )
            elif args.command == "fit-reduction":
                from ..torch.reduction import fit_reduction_study

                result = fit_reduction_study(
                    model,
                    dataset,
                    start_layer=args.start_layer,
                    rank=args.rank,
                    ridge=args.ridge,
                    fit_split=args.fit_split,
                    evaluation_split=args.evaluation_split,
                )
            elif args.command == "edges":
                from ..torch.edges import edge_study

                result = edge_study(
                    model,
                    dataset,
                    patches=_read(args.patches),
                    provenance=args.provenance,
                    split=args.split,
                )
            elif args.command == "reduce":
                from ..torch.reduction import reduction_study

                result = reduction_study(
                    model,
                    dataset,
                    start_layer=args.start_layer,
                    maps=_read(args.maps),
                    provenance=args.provenance,
                    split=args.split,
                )
            elif args.command == "suffix":
                from ..torch.suffix import suffix_study

                result = suffix_study(
                    model,
                    dataset,
                    start_layer=args.start_layer,
                    states=_read(args.states),
                    provenance=args.provenance,
                    split=args.split,
                )
            elif args.command == "coalitions":
                from ..torch.coalitions import coalition_study

                result = coalition_study(
                    model,
                    dataset,
                    modules=args.modules,
                    max_evaluations=args.max_evaluations,
                    background=(
                        None if args.background is None else _read(args.background)
                    ),
                    split=args.split,
                )
            elif args.command == "protect":
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
                    read_slots=(
                        None if args.read_slots is None else _read(args.read_slots)
                    ),
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
        if args.command == "replay":
            summary["replay_status"] = result["status"]
        print(json.dumps(summary))
        return 1 if args.command == "replay" and result["status"] == "mismatch" else 0
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
