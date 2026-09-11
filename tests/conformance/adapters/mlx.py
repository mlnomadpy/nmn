"""MLX dense conformance adapter."""

from __future__ import annotations

import numpy as np

from tests._isolated_backend import mlx_is_usable
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


class MlxAdapter:
    @staticmethod
    def _layer(case: DenseCase, configuration: DenseConfiguration | None = None):
        import mlx.core as mx

        from nmn.mlx import YatNMN

        configuration = configuration or DenseConfiguration()
        layer = YatNMN(
            features=case.kernel.shape[1],
            use_bias=configuration.use_bias,
            constant_bias=configuration.constant_bias,
            use_alpha=True,
            spherical=configuration.spherical,
            weight_normalized=configuration.weight_normalized,
            epsilon=float(case.epsilon),
            learnable_epsilon=configuration.learnable_epsilon,
            dtype=mx.float32,
        )
        layer.build(case.kernel.shape[0])
        layer.kernel = mx.array(case.kernel.T, dtype=mx.float32)
        if configuration.bias_mode == "learnable":
            layer.bias = mx.array(case.bias, dtype=mx.float32)
        layer.alpha = mx.array([float(case.alpha)], dtype=mx.float32)
        return layer

    @staticmethod
    def available() -> bool:
        return mlx_is_usable()

    @staticmethod
    def dense(
        case: DenseCase,
        *,
        compiled: bool = False,
        configuration: DenseConfiguration | None = None,
    ) -> np.ndarray:
        import mlx.core as mx

        layer = MlxAdapter._layer(case, configuration)
        function = mx.compile(layer, inputs=layer.state) if compiled else layer
        output = function(mx.array(case.inputs, dtype=mx.float32))
        mx.eval(output)
        return np.asarray(output)

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        layer = MlxAdapter._layer(case)
        inputs = mx.array(case.inputs, dtype=mx.float32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)

        def loss(model, values):
            output = model(values)
            return mx.sum(output * cotangent), output

        parameter_grad = nn.value_and_grad(layer, loss)
        input_grad = mx.grad(lambda values: loss(layer, values)[0])

        def evaluate(values):
            (value, output), parameter_values = parameter_grad(layer, values)
            del value
            raw_scale = mx.sigmoid(layer.epsilon_param)
            gradients = {
                "input": input_grad(values),
                "kernel": mx.transpose(parameter_values["kernel"]),
                "bias": parameter_values["bias"],
                "alpha": mx.squeeze(parameter_values["alpha"]),
                "epsilon": mx.squeeze(parameter_values["epsilon_param"] / raw_scale),
            }
            return output, gradients

        function = mx.compile(evaluate, inputs=layer.state) if compiled else evaluate
        output, gradients = function(inputs)
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def embedding_value_and_grad(
        case: EmbeddingCase, *, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        from nmn.mlx import YatEmbed

        layer = YatEmbed(case.embedding.shape[0], case.embedding.shape[1])
        layer.embedding = mx.array(case.embedding, dtype=mx.float32)
        indices = mx.array(case.indices, dtype=mx.int32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)
        gradient_fn = nn.value_and_grad(
            layer, lambda model, values: mx.sum(model(values) * cotangent)
        )

        def evaluate(values):
            _, gradients = gradient_fn(layer, values)
            return layer(values), gradients

        function = mx.compile(evaluate, inputs=layer.state) if compiled else evaluate
        output, gradients = function(indices)
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output), {"embedding": np.asarray(gradients["embedding"])}
        )

    @staticmethod
    def embedding_attend_value_and_grad(
        case: EmbeddingAttendCase, *, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        from nmn.mlx import YatEmbed

        layer = YatEmbed(
            case.embedding.shape[0],
            case.embedding.shape[1],
            epsilon=float(case.epsilon),
        )
        layer.embedding = mx.array(case.embedding, dtype=mx.float32)
        layer.alpha = mx.array([float(case.alpha)], dtype=mx.float32)
        query = mx.array(case.query, dtype=mx.float32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)
        parameter_fn = nn.value_and_grad(
            layer, lambda model, values: mx.sum(model.attend(values) * cotangent)
        )
        query_fn = mx.value_and_grad(
            lambda values: mx.sum(layer.attend(values) * cotangent)
        )

        def evaluate(values):
            _, parameter_gradients = parameter_fn(layer, values)
            _, query_gradient = query_fn(values)
            return layer.attend(values), parameter_gradients, query_gradient

        function = mx.compile(evaluate, inputs=layer.state) if compiled else evaluate
        output, parameter_gradients, query_gradient = function(query)
        gradients = {
            "query": query_gradient,
            "embedding": parameter_gradients["embedding"],
            "alpha": mx.squeeze(parameter_gradients["alpha"]),
        }
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def convolution_value_and_grad(
        case: ConvolutionCase, *, transpose: bool, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        from nmn.mlx import YatConv1D, YatConvTranspose1D

        layer_type = YatConvTranspose1D if transpose else YatConv1D
        layer = layer_type(
            filters=case.kernel.shape[-1],
            kernel_size=case.kernel.shape[0],
            padding="valid",
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            dtype=mx.float32,
        )
        layer.build(case.inputs.shape[-1])
        kernel = np.ascontiguousarray(
            case.kernel[::-1].transpose(2, 0, 1)
            if transpose
            else case.kernel.transpose(2, 0, 1)
        )
        layer.kernel = mx.array(kernel, dtype=mx.float32)
        layer.bias = mx.array(case.bias, dtype=mx.float32)
        layer.alpha = mx.array([float(case.alpha)], dtype=mx.float32)
        inputs = mx.array(case.inputs, dtype=mx.float32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)
        parameter_fn = nn.value_and_grad(
            layer, lambda model, values: mx.sum(model(values) * cotangent)
        )
        input_fn = mx.value_and_grad(lambda values: mx.sum(layer(values) * cotangent))

        def evaluate(values):
            _, parameter_gradients = parameter_fn(layer, values)
            _, input_gradient = input_fn(values)
            kernel_gradient = mx.transpose(parameter_gradients["kernel"], (1, 2, 0))
            if transpose:
                kernel_gradient = kernel_gradient[::-1]
            raw_scale = mx.sigmoid(layer.epsilon_param)
            gradients = {
                "input": input_gradient,
                "kernel": kernel_gradient,
                "bias": parameter_gradients["bias"],
                "alpha": mx.squeeze(parameter_gradients["alpha"]),
                "epsilon": mx.squeeze(parameter_gradients["epsilon_param"] / raw_scale),
            }
            return layer(values), gradients

        function = mx.compile(evaluate, inputs=layer.state) if compiled else evaluate
        output, gradients = function(inputs)
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def attention_value_and_grad(
        case: AttentionCase, *, compiled: bool = False
    ) -> AttentionResult:
        import mlx.core as mx

        from nmn.mlx import yat_attention, yat_attention_weights

        mask = mx.array(case.mask, dtype=mx.bool_)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)

        def loss(query, key, value, alpha, epsilon):
            output = yat_attention(
                query, key, value, mask=mask, epsilon=epsilon, alpha=alpha
            )
            return mx.sum(output * cotangent)

        gradient_fn = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))

        def evaluate(query, key, value, alpha, epsilon):
            _, gradients = gradient_fn(query, key, value, alpha, epsilon)
            weights = yat_attention_weights(
                query, key, mask=mask, epsilon=epsilon, alpha=alpha
            )
            output = yat_attention(
                query, key, value, mask=mask, epsilon=epsilon, alpha=alpha
            )
            return weights, output, gradients

        function = evaluate
        if compiled:
            function = mx.compile(function)
        operands = tuple(
            mx.array(value, dtype=mx.float32)
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
        mx.eval(weights, output, gradients)
        return AttentionResult(
            np.asarray(weights),
            np.asarray(output),
            {name: np.asarray(item) for name, item in gradients.items()},
        )

    @staticmethod
    def linear_attention_value_and_grad(
        case: LinearAttentionCase, *, compiled: bool = False
    ) -> LinearAttentionResult:
        import mlx.core as mx

        from nmn.mlx import (
            maclaurin_features,
            maclaurin_yat_attention,
            radial_features,
            radial_yat_attention,
        )

        if case.key_padding_mask is not None:
            raise ValueError("MLX MAY/RAY exposes causal masking only")
        params = {
            name: (
                mx.array(value, dtype=mx.float32)
                if isinstance(value, np.ndarray)
                else value
            )
            for name, value in case.projection.items()
        }
        feature_fn, attention_fn = (
            (maclaurin_features, maclaurin_yat_attention)
            if case.kind == "may"
            else (radial_features, radial_yat_attention)
        )
        cotangent = mx.array(case.cotangent, dtype=mx.float32)

        def loss(query, key, value):
            output = attention_fn(
                query,
                key,
                value,
                params,
                causal=case.causal,
                epsilon=float(case.epsilon),
            )
            return mx.sum(output * cotangent)

        gradient_fn = mx.value_and_grad(loss, argnums=(0, 1, 2))

        def evaluate(query, key, value):
            _, gradients = gradient_fn(query, key, value)
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
            return query_features, key_features, output, gradients

        function = evaluate
        if compiled:
            function = mx.compile(function)
        operands = tuple(
            mx.array(item, dtype=mx.float32)
            for item in (case.query, case.key, case.value)
        )
        query_features, key_features, output, gradients = function(*operands)
        mx.eval(query_features, key_features, output, gradients)
        return LinearAttentionResult(
            np.asarray(query_features),
            np.asarray(key_features),
            np.asarray(output),
            {
                name: np.asarray(value)
                for name, value in zip(("query", "key", "value"), gradients)
            },
        )
