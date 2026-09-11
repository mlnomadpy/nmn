"""Keras dense conformance adapter."""

from __future__ import annotations

import importlib.util

import numpy as np

from tests.conformance.oracle import (
    AttentionCase,
    AttentionResult,
    ConvolutionCase,
    DenseCase,
    DenseConfiguration,
    EmbeddingAttendCase,
    EmbeddingCase,
    LinearAttentionCase,
    LinearAttentionResult,
    OracleResult,
)


class KerasAdapter:
    @staticmethod
    def _layer(case: DenseCase, configuration: DenseConfiguration | None = None):
        import keras

        from nmn.keras import YatNMN

        configuration = configuration or DenseConfiguration()
        layer = YatNMN(
            units=case.kernel.shape[1],
            use_bias=configuration.use_bias,
            constant_bias=configuration.constant_bias,
            use_alpha=True,
            spherical=configuration.spherical,
            weight_normalized=configuration.weight_normalized,
            epsilon=float(case.epsilon),
            learnable_epsilon=configuration.learnable_epsilon,
            dtype="float32",
        )
        inputs = keras.ops.convert_to_tensor(case.inputs, dtype="float32")
        layer(inputs)
        layer.kernel.assign(case.kernel)
        if configuration.bias_mode == "learnable":
            layer.bias.assign(case.bias)
        layer.alpha.assign([float(case.alpha)])
        return layer

    @staticmethod
    def available() -> bool:
        if importlib.util.find_spec("keras") is None:
            return False
        try:
            import keras
        except (ImportError, ModuleNotFoundError):
            return False
        return keras.backend.backend() == "tensorflow"

    @staticmethod
    def dense(
        case: DenseCase,
        *,
        compiled: bool = False,
        configuration: DenseConfiguration | None = None,
    ) -> np.ndarray:
        import keras

        layer = KerasAdapter._layer(case, configuration)
        inputs = keras.ops.convert_to_tensor(case.inputs, dtype="float32")
        if compiled and keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            output = tf.function(layer)(inputs)
        elif compiled:
            output = keras.ops.convert_to_numpy(layer(inputs))
        else:
            output = layer(inputs)
        return np.asarray(output)

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import keras

        if keras.backend.backend() != "tensorflow":
            raise RuntimeError("Keras conformance gradients require TensorFlow backend")
        import tensorflow as tf

        layer = KerasAdapter._layer(case)
        inputs = tf.convert_to_tensor(case.inputs, dtype=tf.float32)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                output = layer(values)
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(
                loss,
                [values, layer.kernel, layer.bias, layer.alpha, layer.epsilon_param],
            )
            return output, gradients

        function = tf.function(evaluate) if compiled else evaluate
        output, values = function(inputs)
        raw_scale = tf.math.sigmoid(layer.epsilon_param.value)
        gradients = {
            "input": values[0],
            "kernel": values[1],
            "bias": values[2],
            "alpha": tf.squeeze(values[3]),
            "epsilon": tf.squeeze(values[4] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def embedding_value_and_grad(
        case: EmbeddingCase, *, compiled: bool = False
    ) -> OracleResult:
        import keras
        import tensorflow as tf

        from nmn.keras import YatEmbed

        layer = YatEmbed(case.embedding.shape[0], case.embedding.shape[1])
        layer.build()
        layer.embedding.assign(case.embedding)
        indices = tf.convert_to_tensor(case.indices, dtype=tf.int32)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                output = layer(values)
                loss = tf.reduce_sum(output * cotangent)
            return output, tape.gradient(loss, layer.embedding)

        if keras.backend.backend() != "tensorflow":
            raise RuntimeError("Keras conformance gradients require TensorFlow backend")
        output, gradient = (tf.function(evaluate) if compiled else evaluate)(indices)
        gradient = tf.convert_to_tensor(gradient)
        return OracleResult(np.asarray(output), {"embedding": np.asarray(gradient)})

    @staticmethod
    def embedding_attend_value_and_grad(
        case: EmbeddingAttendCase, *, compiled: bool = False
    ) -> OracleResult:
        import keras
        import tensorflow as tf

        from nmn.keras import YatEmbed

        if keras.backend.backend() != "tensorflow":
            raise RuntimeError("Keras conformance gradients require TensorFlow backend")
        layer = YatEmbed(
            case.embedding.shape[0],
            case.embedding.shape[1],
            epsilon=float(case.epsilon),
        )
        layer.build()
        layer.embedding.assign(case.embedding)
        layer.alpha.assign([float(case.alpha)])
        query = tf.convert_to_tensor(case.query, dtype=tf.float32)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                output = layer.attend(values)
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(loss, (values, layer.embedding, layer.alpha))
            return output, gradients

        output, values = (tf.function(evaluate) if compiled else evaluate)(query)
        gradients = dict(zip(("query", "embedding", "alpha"), values))
        gradients["alpha"] = tf.squeeze(gradients["alpha"])
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def convolution_value_and_grad(
        case: ConvolutionCase, *, transpose: bool, compiled: bool = False
    ) -> OracleResult:
        import keras
        import tensorflow as tf

        from nmn.keras import YatConv1D, YatConvTranspose1D

        if keras.backend.backend() != "tensorflow":
            raise RuntimeError("Keras conformance gradients require TensorFlow backend")
        layer_type = YatConvTranspose1D if transpose else YatConv1D
        layer = layer_type(
            filters=case.kernel.shape[-1],
            kernel_size=case.kernel.shape[0],
            padding="valid",
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            dtype="float32",
        )
        inputs = tf.convert_to_tensor(case.inputs, dtype=tf.float32)
        layer.build(inputs.shape)
        kernel = case.kernel[::-1].transpose(0, 2, 1) if transpose else case.kernel
        layer.kernel.assign(kernel)
        layer.bias.assign(case.bias)
        layer.alpha.assign([float(case.alpha)])
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                output = layer(values)
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(
                loss,
                (values, layer.kernel, layer.bias, layer.alpha, layer.epsilon_param),
            )
            return output, gradients

        output, values = (tf.function(evaluate) if compiled else evaluate)(inputs)
        raw_scale = tf.math.sigmoid(layer.epsilon_param.value)
        kernel_gradient = values[1]
        if transpose:
            kernel_gradient = tf.transpose(kernel_gradient, (0, 2, 1))[::-1]
        gradients = {
            "input": values[0],
            "kernel": kernel_gradient,
            "bias": values[2],
            "alpha": tf.squeeze(values[3]),
            "epsilon": tf.squeeze(values[4] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def attention_value_and_grad(
        case: AttentionCase, *, compiled: bool = False
    ) -> AttentionResult:
        import keras

        if keras.backend.backend() != "tensorflow":
            raise RuntimeError("Keras conformance gradients require TensorFlow backend")
        import tensorflow as tf

        from nmn.keras import yat_attention, yat_attention_weights

        mask = tf.convert_to_tensor(case.mask, dtype=tf.bool)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(query, key, value, alpha, epsilon):
            with tf.GradientTape() as tape:
                tape.watch((query, key, value, alpha, epsilon))
                weights = yat_attention_weights(
                    query, key, mask=mask, epsilon=epsilon, alpha=alpha
                )
                output = yat_attention(
                    query, key, value, mask=mask, epsilon=epsilon, alpha=alpha
                )
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(loss, (query, key, value, alpha, epsilon))
            return weights, output, gradients

        function = tf.function(evaluate) if compiled else evaluate
        operands = tuple(
            tf.convert_to_tensor(value, dtype=tf.float32)
            for value in (
                case.query,
                case.key,
                case.value,
                case.alpha,
                case.epsilon,
            )
        )
        weights, output, values = function(*operands)
        gradients = dict(zip(("query", "key", "value", "alpha", "epsilon"), values))
        return AttentionResult(
            np.asarray(weights),
            np.asarray(output),
            {name: np.asarray(item) for name, item in gradients.items()},
        )

    @staticmethod
    def linear_attention_value_and_grad(
        case: LinearAttentionCase, *, compiled: bool = False
    ) -> LinearAttentionResult:
        import keras

        if keras.backend.backend() != "tensorflow":
            raise RuntimeError("Keras conformance gradients require TensorFlow backend")
        import tensorflow as tf

        from nmn.keras import (
            maclaurin_features,
            maclaurin_yat_attention,
            radial_features,
            radial_yat_attention,
        )

        if case.key_padding_mask is not None:
            raise ValueError("Keras MAY/RAY exposes causal masking only")
        params = {
            name: (
                tf.convert_to_tensor(value, dtype=tf.float32)
                if isinstance(value, np.ndarray)
                else value
            )
            for name, value in case.projection.items()
        }
        if case.kind == "ray":
            params["b"] = params.pop("bias")
        feature_fn, attention_fn = (
            (maclaurin_features, maclaurin_yat_attention)
            if case.kind == "may"
            else (radial_features, radial_yat_attention)
        )
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(query, key, value):
            with tf.GradientTape() as tape:
                tape.watch((query, key, value))
                query_features = feature_fn(query, params)
                key_features = feature_fn(key, params)
                output = attention_fn(
                    query,
                    key,
                    value,
                    params,
                    causal=case.causal,
                    epsilon=float(case.epsilon),
                )
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(loss, (query, key, value))
            return query_features, key_features, output, gradients

        function = tf.function(evaluate) if compiled else evaluate
        operands = tuple(
            tf.convert_to_tensor(item, dtype=tf.float32)
            for item in (case.query, case.key, case.value)
        )
        query_features, key_features, output, gradients = function(*operands)
        return LinearAttentionResult(
            np.asarray(query_features),
            np.asarray(key_features),
            np.asarray(output),
            {
                name: np.asarray(value)
                for name, value in zip(("query", "key", "value"), gradients)
            },
        )
