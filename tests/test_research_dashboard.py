"""Portable evidence indexing preserves scope and rejects unsafe embedding."""

import json
import shutil

import pytest

from nmn.research.dashboard import build_dashboard


def test_statuses_portability_and_unavailable_inputs(tmp_path):
    sources = []
    statuses = ["certified-under-assumptions", "counterexample-found", "inconclusive"]
    for index, status in enumerate(statuses):
        path = tmp_path / f"{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "nmn.finite-contract-evidence.v1",
                    "status": status,
                    "cases_checked": 1,
                    "cases_total": 2,
                    "dataset": {"name": "</script><img src=x onerror=alert(1)>"},
                }
            )
        )
        sources.append(path)
    missing = tmp_path / "missing.json"
    result = build_dashboard([*sources, sources[0], missing], tmp_path / "report")
    assert result["records"] == 4 and result["unavailable"] == 1
    shutil.move(tmp_path / "report", tmp_path / "moved")
    report = tmp_path / "moved"
    catalog = json.loads((report / "catalog.json").read_text())
    assert [r["status"] for r in catalog["records"]] == statuses + ["unavailable"]
    for entry, source in zip(catalog["records"], sources):
        assert (report / entry["data"]).read_bytes() == source.read_bytes()
        assert (report / entry["note"]).is_file()
        assert entry["scope"] == "finite exact contract"
    html = (report / "index.html").read_text()
    assert "</script><img" not in html
    assert r"\u003c/script\u003e" in html
    with pytest.raises(ValueError, match="already exists"):
        build_dashboard(sources, report)


def test_bad_record_does_not_prevent_later_records(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"schema": "nmn.finite-contract-evidence.v1", "status": "invented"})
    )
    unknown = tmp_path / "unknown.json"
    unknown.write_text("{}")
    output = tmp_path / "report"
    result = build_dashboard([bad, unknown], output)
    assert result["unavailable"] == 2
    assert not list((output / "evidence").glob("*/data.json"))
