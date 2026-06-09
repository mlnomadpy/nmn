"""Enable ``python -m nmn`` to run the CLI."""

from __future__ import annotations

from nmn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
