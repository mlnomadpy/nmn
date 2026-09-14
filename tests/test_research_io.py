"""Interrupted and competing publication must not leave partial final evidence."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from nmn.research import io


def test_complete_file_is_staged_before_publication(tmp_path, monkeypatch):
    destination = tmp_path / "record.json"
    original_link = io.os.link

    def check_link(source, target):
        assert not destination.exists()
        assert json.loads(source.read_text()) == {"value": [1, 2, 3]}
        original_link(source, target)

    monkeypatch.setattr(io.os, "link", check_link)
    io.write_json_exclusive({"value": [1, 2, 3]}, destination)
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_flush_and_invalid_json_leave_no_destination(tmp_path, monkeypatch):
    destination = tmp_path / "record.json"

    def fail(_):
        raise OSError("injected flush failure")

    monkeypatch.setattr(io.os, "fsync", fail)
    with pytest.raises(OSError, match="flush failure"):
        io.write_json_exclusive({"value": 1}, destination)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        io.write_json_exclusive({"value": float("nan")}, destination)
    assert not list(tmp_path.iterdir())


def test_competing_writers_preserve_one_complete_winner(tmp_path, monkeypatch):
    destination = tmp_path / "record.json"
    barrier = threading.Barrier(2)
    original_link = io.os.link

    def competing_link(source, target):
        barrier.wait(timeout=5)
        original_link(source, target)

    monkeypatch.setattr(io.os, "link", competing_link)

    def write(value):
        try:
            io.write_json_exclusive({"value": value}, destination)
            return value
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [1, 2]))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert json.loads(destination.read_text()) == {"value": winners[0]}
    assert list(tmp_path.iterdir()) == [destination]
