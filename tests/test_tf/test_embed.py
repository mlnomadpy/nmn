"""Tests for TensorFlow YatEmbed."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nmn.tf.embed import YatEmbed  # noqa: E402


class TestYatEmbed:
    def test_forward_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        tokens = tf.constant([0, 5, 10, 99])
        out = embed(tokens)
        assert out.shape == (4, 32)

    def test_forward_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        tokens = tf.constant([[0, 1, 2], [3, 4, 5]])
        out = embed(tokens)
        assert out.shape == (2, 3, 16)

    def test_attend_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = tf.random.normal((4, 32))
        out = embed.attend(query)
        assert out.shape == (4, 100)

    def test_attend_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        query = tf.random.normal((2, 10, 16))
        out = embed.attend(query)
        assert out.shape == (2, 10, 50)

    def test_attend_no_nan(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = tf.random.normal((4, 32))
        out = embed.attend(query).numpy()
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_attend_positive(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = tf.random.normal((4, 32))
        out = embed.attend(query).numpy()
        assert np.all(out >= 0)

    def test_constant_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, constant_alpha=True)
        assert embed._constant_alpha_value == pytest.approx(1.4142135, abs=1e-5)
        query = tf.random.normal((4, 16))
        out = embed.attend(query).numpy()
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, use_alpha=False)
        assert embed.alpha is None
        query = tf.random.normal((4, 16))
        out = embed.attend(query).numpy()
        assert not np.any(np.isnan(out))

    def test_spherical_mode(self):
        embed = YatEmbed(num_embeddings=50, features=16, spherical=True)
        query = tf.random.normal((4, 16))
        out = embed.attend(query)
        assert out.shape == (4, 50)
        assert not np.any(np.isnan(out.numpy()))

    def test_weight_normalized(self):
        embed = YatEmbed(num_embeddings=50, features=16, weight_normalized=True)
        norms = tf.sqrt(tf.reduce_sum(tf.square(embed.embedding), axis=1)).numpy()
        np.testing.assert_allclose(norms, np.ones(50), atol=1e-5)


def _tf_attend_value_and_grads(dtype, spherical, compiled, loss_scale=1.0):
    layer = YatEmbed(
        num_embeddings=2, features=2, epsilon=1.0, spherical=spherical, dtype=dtype
    )
    layer.embedding.assign(tf.constant([[-100.0, -99.0], [100.0, -99.0]], dtype=dtype))
    layer.alpha.assign(tf.constant([1.25], dtype=dtype))
    query = tf.Variable([[100.0, 100.0]], dtype=dtype)

    def evaluate():
        with tf.GradientTape() as tape:
            output = layer.attend(query)
            loss = tf.reduce_sum(tf.cast(output, tf.float32)) * loss_scale
        gradients = tape.gradient(loss, (query, layer.embedding, layer.alpha))
        return output, gradients

    return tf.function(evaluate)() if compiled else evaluate()


@pytest.mark.parametrize("dtype", [tf.float16, tf.bfloat16])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_low_precision_attend_matches_fp32_output_and_gradients(
    dtype, spherical, compiled
):
    expected_output, expected_grads = _tf_attend_value_and_grads(
        tf.float32, spherical, compiled
    )
    output, grads = _tf_attend_value_and_grads(dtype, spherical, compiled)

    assert output.dtype == dtype
    np.testing.assert_allclose(
        tf.cast(output, tf.float32).numpy(),
        expected_output.numpy(),
        rtol=1.5e-2,
        atol=2e-2,
    )
    for actual, expected in zip(grads, expected_grads):
        assert np.isfinite(tf.cast(actual, tf.float32).numpy()).all()
        np.testing.assert_allclose(
            tf.cast(actual, tf.float32).numpy(),
            expected.numpy(),
            rtol=2e-2,
            atol=2e-2,
        )


@pytest.mark.parametrize("spherical", [False, True])
def test_fp16_attend_saturates_output_and_gradients_and_preserves_nan(spherical):
    layer = YatEmbed(
        num_embeddings=2, features=2, spherical=spherical, dtype=tf.float16
    )
    layer.embedding.assign(
        tf.constant([[300.0, 300.0], [300.0, -300.0]], dtype=tf.float16)
    )
    query = tf.Variable([[300.0, 300.0]], dtype=tf.float16)

    @tf.function
    def evaluate(query_value):
        with tf.GradientTape() as tape:
            tape.watch(query_value)
            output = layer.attend(query_value)
            loss = tf.reduce_sum(tf.cast(output, tf.float32))
        return output, tape.gradient(loss, (query_value, layer.embedding, layer.alpha))

    output, gradients = evaluate(query)
    assert output[0, 0] == tf.constant(np.finfo(np.float16).max, tf.float16)
    assert all(
        np.isfinite(tf.cast(value, tf.float32).numpy()).all() for value in gradients
    )

    nan_output = layer.attend(tf.constant([[np.nan, 300.0]], tf.float16))
    assert np.isnan(nan_output.numpy()).all()


def test_fp16_attend_returning_gradients_saturate_against_fp32():
    spherical, loss_scale = False, 1e4
    _, expected_grads = _tf_attend_value_and_grads(
        tf.float32, spherical, True, loss_scale
    )
    _, grads = _tf_attend_value_and_grads(tf.float16, spherical, True, loss_scale)
    limits = np.finfo(np.float16)
    for actual, expected in zip(grads, expected_grads):
        clipped = np.clip(expected.numpy(), limits.min, limits.max).astype(np.float16)
        assert np.isfinite(actual.numpy()).all()
        np.testing.assert_allclose(actual.numpy(), clipped, rtol=2e-2, atol=32.0)
