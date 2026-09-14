"""Backend-free reusable extraction preserves source linkage and input files."""

import hashlib
import json

import pytest

from nmn.research.components import extract_native_component, list_native_components


def test_maps_extraction_and_no_overwrite(tmp_path):
    source = tmp_path / "summary.json"
    maps = {"encoder": {"weight": [[1]], "bias": [0]}}
    raw = json.dumps({"schema": "nmn.reduction-study.v1", "maps": maps}).encode()
    source.write_bytes(raw)
    assert list_native_components(source)["components"] == ["maps"]
    output = tmp_path / "extracted"
    result = extract_native_component(source, "maps", output)
    assert json.loads((output / "data.json").read_text()) == maps
    assert result["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        result["data_sha256"]
        == hashlib.sha256((output / "data.json").read_bytes()).hexdigest()
    )
    assert result["recomputed"] is False
    with pytest.raises(FileExistsError):
        extract_native_component(source, "maps", output)
    with pytest.raises(ValueError, match="unavailable"):
        extract_native_component(source, "evaluation", tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    assert source.read_bytes() == raw
