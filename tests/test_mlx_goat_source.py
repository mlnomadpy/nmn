"""Dependency-light source checks for MLX GOAT compatibility."""

import ast
from pathlib import Path

GOAT_SOURCE = Path(__file__).parents[1] / "src" / "nmn" / "mlx" / "goat.py"


def test_goat_source_compiles_without_importing_mlx():
    source = GOAT_SOURCE.read_text(encoding="utf-8")
    compile(source, str(GOAT_SOURCE), "exec")


def test_goat_precision_path_avoids_newer_result_type_api():
    source = GOAT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GOAT_SOURCE))
    mlx_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "mx"
    }
    assert "result_type" not in mlx_attributes
