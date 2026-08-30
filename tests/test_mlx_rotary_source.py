"""Static regressions for MLX rotary paths that cannot run without Metal."""

import ast
from pathlib import Path


ROTARY_SOURCE = Path(__file__).parents[1] / "src" / "nmn" / "mlx" / "rotary.py"


def _rotary_call():
    source = ROTARY_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ROTARY_SOURCE))
    rotary_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RotaryYatAttention"
    )
    return next(
        node
        for node in rotary_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )


def test_rotary_call_mask_gate_uses_only_defined_decode_lengths():
    call = _rotary_call()
    loaded_names = {
        node.id
        for node in ast.walk(call)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    self_attributes = {
        node.attr
        for node in ast.walk(call)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }

    assert "decode" in {arg.arg for arg in call.args.kwonlyargs}
    assert "decode" in loaded_names
    assert "K_len" not in loaded_names
    assert "decode" not in self_attributes


def test_rotary_mask_gate_selects_cached_or_sequence_key_length():
    call = _rotary_call()
    conditional_assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(call)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.IfExp)
    }

    effective_mask = conditional_assignments["effective_mask"]
    assert ast.unparse(effective_mask) == "full_mask if decode else mask"

    attention_k_len = conditional_assignments["attention_k_len"]
    assert ast.unparse(attention_k_len) == "cache_new if decode else L"


def test_rotary_source_compiles_without_importing_mlx():
    source = ROTARY_SOURCE.read_text(encoding="utf-8")
    compile(source, str(ROTARY_SOURCE), "exec")
