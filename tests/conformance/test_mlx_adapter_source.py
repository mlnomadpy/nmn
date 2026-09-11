"""Static MLX adapter regressions for hosts without a usable Metal device."""

import ast
from pathlib import Path

ADAPTER_SOURCE = Path(__file__).parent / "adapters" / "mlx.py"


def _tree():
    return ast.parse(ADAPTER_SOURCE.read_text(encoding="utf-8"), ADAPTER_SOURCE)


def _method(name):
    adapter = next(
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == "MlxAdapter"
    )
    return next(
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_dense_uses_the_mlx_layer_dtype_keyword():
    method = _method("_layer")
    constructor = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "YatNMN"
    )
    keywords = {keyword.arg for keyword in constructor.keywords}
    assert "dtype" in keywords
    assert "param_dtype" not in keywords


def test_autodiff_does_not_use_unsupported_has_aux():
    has_aux_keywords = [
        keyword
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "has_aux"
    ]
    assert has_aux_keywords == []


def test_transpose_kernel_is_made_contiguous_before_mlx_conversion():
    method = _method("convolution_value_and_grad")
    assignments = {
        target.id: value
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        for value in [node.value]
    }
    kernel = assignments["kernel"]
    assert isinstance(kernel, ast.Call)
    assert ast.unparse(kernel.func) == "np.ascontiguousarray"


def test_compiled_stateful_paths_bundle_outputs_with_gradients():
    for method_name in (
        "dense_value_and_grad",
        "embedding_value_and_grad",
        "convolution_value_and_grad",
    ):
        method = _method(method_name)
        evaluator = next(
            node
            for node in method.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
        )
        returned_names = {
            node.id
            for statement in evaluator.body
            if isinstance(statement, ast.Return)
            for node in ast.walk(statement.value)
            if isinstance(node, ast.Name)
        }
        assert "gradients" in returned_names
        assert "output" in returned_names or any(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "layer"
            for statement in evaluator.body
            if isinstance(statement, ast.Return)
            for node in ast.walk(statement.value)
        )


def test_mlx_adapter_source_compiles_without_importing_mlx():
    source = ADAPTER_SOURCE.read_text(encoding="utf-8")
    compile(source, str(ADAPTER_SOURCE), "exec")
