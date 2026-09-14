"""Run with JAX_ENABLE_X64=1 python examples/research/nnx_observability.py OUTPUT.

Writes a finite numerical fixture and Obsidian export; no training or downloads.
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from nmn.nnx import ThreeNeuronYat
from nmn.nnx.research import collect_research_data
from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.native_export import export_native_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    model = ThreeNeuronYat.reference(dtype=jnp.float64)
    dataset = ResearchDataset(
        [
            ResearchSample(f"point-{i}", tuple(point), "evaluation", f"point-{i}")
            for i, point in enumerate(
                [(0.0, 0.0), (1.0, 1.0), (0.25, 0.75), (-0.5, 1.5)]
            )
        ],
        name="NNX reference observability",
        provenance="Designed arithmetic fixture; no semantic labels",
    )
    record = collect_research_data(
        model,
        dataset,
        split="evaluation",
        edits={
            "delete-h": {"h": {"gate": 0.0}},
            "replace-h": {"h": {"replacement": 0.25}},
        },
    )
    args.output.mkdir(parents=True)
    path = args.output / "observations.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    export_native_record(path, args.output / "obsidian")
    print(path)


if __name__ == "__main__":
    main()
