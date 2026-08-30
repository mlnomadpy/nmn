"""Utilities for exporting native NMN TensorFlow modules as SavedModels."""

from __future__ import annotations

from os import PathLike
from typing import Any, Dict, Optional, Union

import tensorflow as tf

ExportPath = Union[str, bytes, PathLike]


def _named_spec(spec: tf.TensorSpec, name: str) -> tf.TensorSpec:
    if not isinstance(spec, tf.TensorSpec):
        raise TypeError(f"{name}_signature must be a tf.TensorSpec, got {type(spec)!r}")
    return tf.TensorSpec(spec.shape, spec.dtype, name=name)


def _outputs(result):
    if isinstance(result, tuple):
        return {"outputs": result[0], "kernel": result[1]}
    return {"outputs": result}


def _save(module: tf.Module, export_dir: ExportPath, functions: Dict[str, Any]) -> None:
    signatures = {
        name: function.get_concrete_function() for name, function in functions.items()
    }
    tf.saved_model.save(module, export_dir, signatures=signatures)


def export_single_input(
    module: tf.Module,
    export_dir: ExportPath,
    input_signature: tf.TensorSpec,
) -> None:
    """Export a one-input NMN module with a callable default signature."""
    inputs_spec = _named_spec(input_signature, "inputs")

    @tf.function(input_signature=[inputs_spec])
    def serving_default(inputs):
        return _outputs(module(inputs))

    _save(module, export_dir, {"serving_default": serving_default})


def export_embedding(
    module: tf.Module,
    export_dir: ExportPath,
    lookup_signature: tf.TensorSpec,
    attend_signature: tf.TensorSpec,
) -> None:
    """Export embedding lookup and YAT attend as separate signatures."""
    inputs_spec = _named_spec(lookup_signature, "inputs")
    query_spec = _named_spec(attend_signature, "query")

    @tf.function(input_signature=[inputs_spec])
    def serving_default(inputs):
        return {"outputs": module(inputs)}

    @tf.function(input_signature=[query_spec])
    def attend(query):
        return {"outputs": module.attend(query)}

    _save(
        module,
        export_dir,
        {"serving_default": serving_default, "attend": attend},
    )


def export_attention(
    module: tf.Module,
    export_dir: ExportPath,
    query_signature: tf.TensorSpec,
    *,
    key_signature: Optional[tf.TensorSpec] = None,
    value_signature: Optional[tf.TensorSpec] = None,
    mask_signature: Optional[tf.TensorSpec] = None,
) -> None:
    """Export self- or cross-attention with an inference-only signature."""
    if (key_signature is None) != (value_signature is None):
        raise ValueError(
            "key_signature and value_signature must either both be provided "
            "or both be omitted"
        )

    query_spec = _named_spec(query_signature, "query")
    mask_spec = (
        _named_spec(mask_signature, "mask") if mask_signature is not None else None
    )

    if key_signature is None:
        if mask_spec is None:

            @tf.function(input_signature=[query_spec])
            def serving_default(query):
                return {"outputs": module(query, training=False)}

        else:

            @tf.function(input_signature=[query_spec, mask_spec])
            def serving_default(query, mask):
                return {"outputs": module(query, mask=mask, training=False)}

    else:
        key_spec = _named_spec(key_signature, "key")
        value_spec = _named_spec(value_signature, "value")
        if mask_spec is None:

            @tf.function(input_signature=[query_spec, key_spec, value_spec])
            def serving_default(query, key, value):
                return {"outputs": module(query, key=key, value=value, training=False)}

        else:

            @tf.function(input_signature=[query_spec, key_spec, value_spec, mask_spec])
            def serving_default(query, key, value, mask):
                return {
                    "outputs": module(
                        query,
                        key=key,
                        value=value,
                        mask=mask,
                        training=False,
                    )
                }

    _save(module, export_dir, {"serving_default": serving_default})


class SingleInputSavedModelMixin:
    """Mixin implementing ``export`` for one-input TensorFlow modules."""

    def export(self, export_dir: ExportPath, input_signature: tf.TensorSpec) -> None:
        export_single_input(self, export_dir, input_signature)
