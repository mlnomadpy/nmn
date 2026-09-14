"""Bounded preimage proposals followed by actual native output measurements."""

import argparse
import json
from pathlib import Path

import torch

from nmn.research.native_export import export_native_record
from nmn.torch import Intervention, ThreeNeuronYat
from nmn.torch.preimage import search_preimages
from nmn.torch.research import collect_research_data, save_research_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    # Y reads (h,v). Hold v=1; optimize h within [0,1] toward feature value 1/2.
    search = search_preimages(
        model.y,
        [[1.0, 1.0]],
        [[0.5]],
        lower=[0.0, 1.0],
        upper=[1.0, 1.0],
        sample_ids=["reference"],
        provenance="Designed finite-bank target, not a learned erasure projection",
        max_steps=100,
        max_seconds=10.0,
        learning_rate=0.1,
    )
    selected_h = search["selected_inputs"][0][0]
    observations = collect_research_data(
        model,
        torch.ones((1, 2), dtype=torch.float64),
        sample_ids=["reference"],
        derivatives=False,
        edits={"selected-preimage": {"h": Intervention(replacement=selected_h)}},
        metadata={
            "preimage_module_sha256": search["module_sha256"],
            "mapping": "replace h; v remains fixed at 1 by solver bounds",
        },
    )
    args.output.mkdir(parents=True)
    for name, record in [("search", search), ("execution", observations)]:
        path = args.output / (name + ".json")
        save_research_data(record, path)
        export_native_record(path, args.output / (name + "-export"))
    print(
        json.dumps(
            {
                "selected_h": selected_h,
                "feature_residual": search["feature_residuals"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
