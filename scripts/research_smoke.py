"""Run the first-result workflow against an installed NMN, outside its checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    timings = []

    def run(*arguments: str, expected: int = 0) -> dict:
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", "nmn", "research", *arguments],
            cwd=output,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        timings.append(
            {
                "arguments": list(arguments),
                "seconds": time.perf_counter() - start,
                "exit_code": result.returncode,
            }
        )
        if result.returncode != expected:
            raise RuntimeError(
                f"{arguments}: expected {expected}, got {result.returncode}: {result.stderr}"
            )
        return json.loads(result.stdout)

    model = run("model")
    contract = run("contract")
    (output / "model.json").write_text(json.dumps(model))
    (output / "contract.json").write_text(json.dumps(contract))
    run("model", "model.json")
    run("contract", "contract.json")
    run("inspect", "--model", "model.json")
    edited = run("trace", "--gate", "0")
    assert edited["outputs"] == {"target": "1/2", "protected": "1", "leaky": "3/2"}
    restored = run("trace", "--gate", "0", "--replace-h", "1")
    assert restored["outputs"]["target"] == "4"
    assert run("compare", "--gate", "1/2")["output_deltas"]["target"] == "-11/5"
    run("verify")
    run("verify", "--leaky", expected=1)
    run("verify", "--model", "model.json", "--contract", "contract.json")
    assert (
        run("verify", "--contract", "contract.json", "--max-cases", "10", expected=3)[
            "cases_checked"
        ]
        == 10
    )
    for name, options in [
        ("fixed", []),
        ("configured", ["--model", "model.json"]),
        ("complete", ["--model", "model.json", "--contract", "contract.json"]),
        ("partial", ["--contract", "contract.json", "--max-cases", "10"]),
    ]:
        run("demo", *options, "--output", name)
        run("reproduce", name)
        run("export", name, "--output", f"vault/{name}")
        run("reproduce", f"vault/{name}")
    contract["target"]["minimum_decrease"] = "4"
    (output / "failed-contract.json").write_text(json.dumps(contract))
    failed = run("verify", "--contract", "failed-contract.json", expected=1)
    assert failed["protected_violations_observed"] == 0
    run("demo", "--contract", "failed-contract.json", "--output", "failed")
    assert run("reproduce", "failed")["verification_status"] == "counterexample-found"
    run("export", "failed", "--output", "vault/failed")
    (output / "timings.json").write_text(
        json.dumps(
            {
                "scope": "local installed CLI wall times; not a benchmark claim",
                "python": sys.version,
                "commands": timings,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(timings)} installed CLI commands; evidence: {output}")


if __name__ == "__main__":
    main()
