# Test suite organization

The suite is organized by the boundary it verifies:

- `test_<backend>/` — backend-specific unit and regression tests;
- `integration/` — numerical parity and cross-framework behavior;
- `benchmarks/` — small performance assertions that are safe in CI;
- `test_cli.py` — import-light command-line behavior;
- `test_collection_policy.py` — optional-backend and collection isolation;
- `test_workflow_policy.py` — CI/CD configuration invariants.

Exploratory programs, reports, and long-running benchmarks belong in the
repository-level `benchmarks/` directory. Files below `tests/` must be
deterministic, assertion-based, and safe to import during collection.

Useful commands:

```bash
python -m pytest -q -m "not slow"
python -m pytest tests/test_nnx -q
python -m pytest tests/integration -q
python -m pytest tests/test_workflow_policy.py tests/test_collection_policy.py -q
```

Tests for unavailable optional backends are skipped before importing that
backend. Keep new optional-backend imports inside their backend tree or guarded
with `pytest.importorskip`.
