"""Create canonical fixtures for cross-platform CI artifact transfer."""

from __future__ import annotations

import argparse

from tests.conformance.oracle import write_dense_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    arguments = parser.parse_args()
    write_dense_fixture(arguments.output)


if __name__ == "__main__":
    main()
