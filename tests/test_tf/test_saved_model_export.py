"""Callable SavedModel export tests for native TensorFlow NMN modules."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nmn.tf import MultiHeadYatAttention, YatConv2D, YatEmbed, YatNMN


def _assert_outputs_match(expected, restored, signature_name, **inputs):
    signature = restored.signatures[signature_name]
    eager = signature(**inputs)["outputs"]

    @tf.function
    def graph_call():
        return signature(**inputs)["outputs"]

    graph = graph_call()
    assert eager.shape == expected.shape
    assert eager.dtype == expected.dtype
    np.testing.assert_allclose(eager.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(graph.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


def _assert_variables_preserved(module, restored):
    variable_names = [
        name for name, value in vars(module).items() if isinstance(value, tf.Variable)
    ]
    assert variable_names
    for name in variable_names:
        expected = getattr(module, name)
        actual = getattr(restored, name)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_yat_nmn_exports_callable_saved_model(tmp_path):
    layer = YatNMN(features=4)
    inputs = tf.reshape(tf.linspace(-1.0, 1.0, 6), (2, 3))

    export_dir = tmp_path / "dense"
    # Exporting a fresh lazy-built module must trace and create its variables.
    layer.export(export_dir, tf.TensorSpec([None, 3], tf.float32))
    expected = layer(inputs)
    restored = tf.saved_model.load(export_dir)

    assert set(restored.signatures) == {"serving_default"}
    _assert_outputs_match(expected, restored, "serving_default", inputs=inputs)
    _assert_variables_preserved(layer, restored)


def test_yat_conv_exports_callable_saved_model(tmp_path):
    layer = YatConv2D(filters=4, kernel_size=2, groups=2)
    inputs = tf.reshape(tf.linspace(-0.5, 0.5, 64), (1, 4, 4, 4))
    expected = layer(inputs)

    export_dir = tmp_path / "conv"
    layer.export(export_dir, tf.TensorSpec([None, 4, 4, 4], tf.float32))
    restored = tf.saved_model.load(export_dir)

    _assert_outputs_match(expected, restored, "serving_default", inputs=inputs)
    _assert_variables_preserved(layer, restored)


def test_yat_embedding_exports_lookup_and_attend(tmp_path):
    layer = YatEmbed(num_embeddings=7, features=3)
    token_ids = tf.constant([[0, 2], [5, 1]], dtype=tf.int32)
    query = tf.reshape(tf.linspace(-0.4, 0.5, 6), (2, 3))
    expected_lookup = layer(token_ids)
    expected_attend = layer.attend(query)

    export_dir = tmp_path / "embed"
    layer.export(
        export_dir,
        tf.TensorSpec([None, None], tf.int32),
        tf.TensorSpec([None, 3], tf.float32),
    )
    restored = tf.saved_model.load(export_dir)

    assert set(restored.signatures) == {"serving_default", "attend"}
    _assert_outputs_match(
        expected_lookup, restored, "serving_default", inputs=token_ids
    )
    _assert_outputs_match(expected_attend, restored, "attend", query=query)
    _assert_variables_preserved(layer, restored)


def test_yat_attention_exports_cross_attention_signature(tmp_path):
    layer = MultiHeadYatAttention(embed_dim=4, num_heads=2, dropout=0.2)
    query = tf.reshape(tf.linspace(-0.7, 0.8, 8), (1, 2, 4))
    key = tf.reshape(tf.linspace(-0.6, 0.9, 12), (1, 3, 4))
    value = tf.reshape(tf.linspace(0.8, -0.5, 12), (1, 3, 4))
    expected = layer(query, key=key, value=value, training=False)

    export_dir = tmp_path / "attention"
    layer.export(
        export_dir,
        tf.TensorSpec([None, None, 4], tf.float32),
        key_signature=tf.TensorSpec([None, None, 4], tf.float32),
        value_signature=tf.TensorSpec([None, None, 4], tf.float32),
    )
    restored = tf.saved_model.load(export_dir)

    _assert_outputs_match(
        expected,
        restored,
        "serving_default",
        query=query,
        key=key,
        value=value,
    )
    _assert_variables_preserved(layer, restored)


def test_attention_export_rejects_partial_cross_attention_signature(tmp_path):
    layer = MultiHeadYatAttention(embed_dim=4, num_heads=2)
    with pytest.raises(ValueError, match="both be provided"):
        layer.export(
            tmp_path / "invalid",
            tf.TensorSpec([None, None, 4], tf.float32),
            key_signature=tf.TensorSpec([None, None, 4], tf.float32),
        )
