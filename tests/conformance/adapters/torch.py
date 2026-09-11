"""PyTorch dense conformance adapter."""

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


class TorchAdapter:
    @staticmethod
    def _layer(case: DenseCase, configuration: DenseConfiguration | None = None):
        import torch

        from nmn.torch import YatNMN

        configuration = configuration or DenseConfiguration()
        layer = YatNMN(
            case.kernel.shape[0],
            case.kernel.shape[1],
            bias=configuration.use_bias,
            constant_bias=configuration.constant_bias,
            alpha=True,
            spherical=configuration.spherical,
            weight_normalized=configuration.weight_normalized,
            epsilon=float(case.epsilon),
            learnable_epsilon=configuration.learnable_epsilon,
            param_dtype=torch.float32,
        )
        with torch.no_grad():
            layer.weight.copy_(torch.asarray(case.kernel.T, dtype=torch.float32))
            if configuration.bias_mode == "learnable":
                layer.bias.copy_(torch.asarray(case.bias, dtype=torch.float32))
            layer.alpha.copy_(torch.asarray([float(case.alpha)], dtype=torch.float32))
        return layer

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("torch") is not None

    @staticmethod
    def dense(
        case: DenseCase,
        *,
        compiled: bool = False,
        configuration: DenseConfiguration | None = None,
    ) -> np.ndarray:
        import torch

        layer = TorchAdapter._layer(case, configuration)
        function = torch.compile(layer) if compiled else layer
        with torch.no_grad():
            output = function(torch.asarray(case.inputs, dtype=torch.float32))
        return output.detach().cpu().numpy()

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import torch

        layer = TorchAdapter._layer(case)
        inputs = torch.asarray(case.inputs, dtype=torch.float32).requires_grad_(True)
        function = torch.compile(layer) if compiled else layer
        output = function(inputs)
        loss = torch.sum(output * torch.asarray(case.cotangent, dtype=torch.float32))
        loss.backward()
        raw_scale = torch.sigmoid(layer.epsilon_param.detach())
        gradients = {
            "input": inputs.grad,
            "kernel": layer.weight.grad.T,
            "bias": layer.bias.grad,
            "alpha": layer.alpha.grad.squeeze(),
            "epsilon": (layer.epsilon_param.grad / raw_scale).squeeze(),
        }
        return OracleResult(
            output.detach().cpu().numpy(),
            {name: value.detach().cpu().numpy() for name, value in gradients.items()},
        )

    @staticmethod
    def embedding_value_and_grad(
        case: EmbeddingCase, *, compiled: bool = False
    ) -> OracleResult:
        import torch

        from nmn.torch import YatEmbed

        layer = YatEmbed(
            case.embedding.shape[0], case.embedding.shape[1], dtype=torch.float32
        )
        with torch.no_grad():
            layer.embedding.copy_(torch.asarray(case.embedding, dtype=torch.float32))
        indices = torch.asarray(case.indices, dtype=torch.long)
        cotangent = torch.asarray(case.cotangent, dtype=torch.float32)
        function = torch.compile(layer) if compiled else layer
        output = function(indices)
        torch.sum(output * cotangent).backward()
        return OracleResult(
            output.detach().cpu().numpy(),
            {"embedding": layer.embedding.grad.detach().cpu().numpy()},
        )

    @staticmethod
    def embedding_attend_value_and_grad(
        case: EmbeddingAttendCase, *, compiled: bool = False
    ) -> OracleResult:
        import torch

        from nmn.torch import YatEmbed

        layer = YatEmbed(
            case.embedding.shape[0],
            case.embedding.shape[1],
            use_alpha=True,
            epsilon=float(case.epsilon),
            dtype=torch.float32,
        )
        with torch.no_grad():
            layer.embedding.copy_(torch.asarray(case.embedding, dtype=torch.float32))
            layer.alpha.copy_(torch.asarray([float(case.alpha)], dtype=torch.float32))
        query = torch.asarray(case.query, dtype=torch.float32).requires_grad_(True)
        cotangent = torch.asarray(case.cotangent, dtype=torch.float32)
        function = torch.compile(layer.attend) if compiled else layer.attend
        output = function(query)
        torch.sum(output * cotangent).backward()
        return OracleResult(
            output.detach().cpu().numpy(),
            {
                "query": query.grad.detach().cpu().numpy(),
                "embedding": layer.embedding.grad.detach().cpu().numpy(),
                "alpha": layer.alpha.grad.squeeze().detach().cpu().numpy(),
            },
        )

    @staticmethod
    def convolution_value_and_grad(
        case: ConvolutionCase, *, transpose: bool, compiled: bool = False
    ) -> OracleResult:
        import torch

        from nmn.torch import YatConv1D, YatConvTranspose1D

        layer_type = YatConvTranspose1D if transpose else YatConv1D
        layer = layer_type(
            case.inputs.shape[-1],
            case.kernel.shape[-1],
            case.kernel.shape[0],
            padding=0,
            bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            param_dtype=torch.float32,
        )
        with torch.no_grad():
            if transpose:
                # JAX's canonical transpose-convolution convention reverses
                # the spatial axis relative to Torch's IOW kernel layout.
                weight = case.kernel[::-1].transpose(1, 2, 0).copy()
            else:
                # Canonical KWIO -> torch OIW.
                weight = case.kernel.transpose(2, 1, 0)
            layer.weight.copy_(torch.asarray(weight, dtype=torch.float32))
            layer.bias.copy_(torch.asarray(case.bias, dtype=torch.float32))
            layer.alpha.copy_(torch.asarray([float(case.alpha)], dtype=torch.float32))
        inputs = torch.asarray(case.inputs.transpose(0, 2, 1), dtype=torch.float32)
        inputs.requires_grad_(True)
        cotangent = torch.asarray(
            case.cotangent.transpose(0, 2, 1), dtype=torch.float32
        )
        function = torch.compile(layer) if compiled else layer
        output = function(inputs)
        torch.sum(output * cotangent).backward()
        raw_scale = torch.sigmoid(layer.epsilon_param.detach())
        kernel_gradient = layer.weight.grad
        if transpose:
            kernel_gradient = kernel_gradient.permute(2, 0, 1).flip(0)
        else:
            kernel_gradient = kernel_gradient.permute(2, 1, 0)
        gradients = {
            "input": inputs.grad.permute(0, 2, 1),
            "kernel": kernel_gradient,
            "bias": layer.bias.grad,
            "alpha": layer.alpha.grad.squeeze(),
            "epsilon": (layer.epsilon_param.grad / raw_scale).squeeze(),
        }
        return OracleResult(
            output.detach().permute(0, 2, 1).cpu().numpy(),
            {name: value.detach().cpu().numpy() for name, value in gradients.items()},
        )

    @staticmethod
    def attention_value_and_grad(
        case: AttentionCase, *, compiled: bool = False
    ) -> AttentionResult:
        import torch

        from nmn.torch.attention import yat_attention, yat_attention_weights

        query = torch.asarray(case.query, dtype=torch.float32).requires_grad_(True)
        key = torch.asarray(case.key, dtype=torch.float32).requires_grad_(True)
        value = torch.asarray(case.value, dtype=torch.float32).requires_grad_(True)
        alpha = torch.asarray(case.alpha, dtype=torch.float32).requires_grad_(True)
        epsilon = torch.asarray(case.epsilon, dtype=torch.float32).requires_grad_(True)
        mask = torch.asarray(case.mask, dtype=torch.bool)
        cotangent = torch.asarray(case.cotangent, dtype=torch.float32)

        def evaluate(q, k, v, a, e):
            weights = yat_attention_weights(
                q, k, mask=mask, training=False, epsilon=e, alpha=a
            )
            output = yat_attention(
                q, k, v, mask=mask, training=False, epsilon=e, alpha=a
            )
            return weights, output

        function = torch.compile(evaluate) if compiled else evaluate
        weights, output = function(query, key, value, alpha, epsilon)
        torch.sum(output * cotangent).backward()
        gradients = {
            "query": query.grad,
            "key": key.grad,
            "value": value.grad,
            "alpha": alpha.grad,
            "epsilon": epsilon.grad,
        }
        return AttentionResult(
            weights.detach().cpu().numpy(),
            output.detach().cpu().numpy(),
            {name: item.detach().cpu().numpy() for name, item in gradients.items()},
        )

    @staticmethod
    def linear_attention_value_and_grad(
        case: LinearAttentionCase, *, compiled: bool = False
    ) -> LinearAttentionResult:
        """Evaluate the public fixed-kernel MAY/RAY functions, not a module."""
        import torch

        from nmn.torch import (
            maclaurin_features,
            maclaurin_yat_attention,
            radial_features,
            radial_yat_attention,
        )

        if case.key_padding_mask is not None:
            raise ValueError("Torch MAY/RAY exposes causal masking only")
        params = {
            name: (
                torch.asarray(value, dtype=torch.float32)
                if isinstance(value, np.ndarray)
                else value
            )
            for name, value in case.projection.items()
        }
        if case.kind == "ray":
            # Torch's first implementation used ``b``; the other backends use
            # the clearer ``bias`` spelling retained by the canonical fixture.
            params["b"] = params.pop("bias")
        feature_fn, attention_fn = (
            (maclaurin_features, maclaurin_yat_attention)
            if case.kind == "may"
            else (radial_features, radial_yat_attention)
        )
        cotangent = torch.asarray(case.cotangent, dtype=torch.float32)

        def evaluate(query, key, value):
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
            return query_features, key_features, output

        function = torch.compile(evaluate) if compiled else evaluate
        query = torch.asarray(case.query, dtype=torch.float32).requires_grad_(True)
        key = torch.asarray(case.key, dtype=torch.float32).requires_grad_(True)
        value = torch.asarray(case.value, dtype=torch.float32).requires_grad_(True)
        query_features, key_features, output = function(query, key, value)
        torch.sum(output * cotangent).backward()
        return LinearAttentionResult(
            query_features.detach().cpu().numpy(),
            key_features.detach().cpu().numpy(),
            output.detach().cpu().numpy(),
            {
                "query": query.grad.detach().cpu().numpy(),
                "key": key.grad.detach().cpu().numpy(),
                "value": value.grad.detach().cpu().numpy(),
            },
        )
