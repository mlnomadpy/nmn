"""YAT convolution layers for Keras/TensorFlow."""

import threading
import weakref

from keras.src import constraints, initializers, regularizers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import backend, standardize_dtype
from keras.src.backend.common.backend_utils import compute_conv_transpose_output_shape
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.ops.operation_utils import compute_conv_output_shape
from keras.src.saving.object_registration import register_keras_serializable
from keras.src.saving.serialization_lib import deserialize_keras_object

from nmn._epsilon import (
    epsilon_parameter_dtype,
    inverse_softplus,
    validate_epsilon,
    validate_epsilon_for_dtype,
)

from ._yat_core import reduction_safe_upcast, yat_score


def _epsilon_weight_dtype(layer):
    dtype = epsilon_parameter_dtype(layer.variable_dtype)
    validate_epsilon_for_dtype(layer.epsilon, dtype)
    if backend() == "jax" and dtype == "float64":
        import jax

        if not jax.config.x64_enabled:
            raise ValueError(
                "float64 learnable epsilon requires jax_enable_x64=True; "
                "the JAX backend would otherwise store it as float32"
            )
    return dtype


def _epsilon_initializer(value):
    def initialize(shape, dtype=None):
        initialized = ops.full(shape, value, dtype=dtype)
        actual_dtype = standardize_dtype(initialized.dtype)
        expected_dtype = standardize_dtype(dtype)
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"learnable epsilon requested {expected_dtype} storage but "
                f"the backend created {actual_dtype}"
            )
        return initialized

    return initialize


def _reject_kernel_bank_expansion(bank_id, existing_filters, requested_filters):
    """Reject fixed-shape Keras variable expansion before mutating a bank."""
    raise ValueError(
        f"Kernel bank '{bank_id}' has {existing_filters} filters and cannot be "
        f"expanded in place to {requested_filters}. Keras variables have fixed "
        "shapes; create the first consumer with a sufficiently large "
        "kernel_bank_size. The existing bank was not modified."
    )


@register_keras_serializable(package="nmn", name="YatKernelBank")
class _KernelBankRef:
    """Serializable identity for a shared kernel without owning its Variable.

    Keras' object-sharing scope preserves one ref object when a Functional model
    containing multiple consumers is cloned or loaded.  Each consumer tracks the
    shared Variable exactly once; the process-local weak registry is only used
    to connect layers constructed directly by users.
    """

    def __init__(self, bank_id, signature, capacity):
        self.bank_id = bank_id
        self.signature = _freeze_signature(signature)
        self.capacity = int(capacity)
        self.variable = None
        # Registry lookup and Variable creation are separate operations.  This
        # per-reference lock keeps the first creation and every attachment
        # atomic without serializing unrelated kernel banks.
        self.lock = threading.Lock()

    def get_config(self):
        return {
            "bank_id": self.bank_id,
            "signature": self.signature,
            "capacity": self.capacity,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class _KernelBankSerializationMixin:
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        bank = config.get("kernel_bank")
        if isinstance(bank, dict):
            config["kernel_bank"] = deserialize_keras_object(bank)
        return cls(**config)


def _freeze_signature(value):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature(item) for item in value)
    return value


def _safe_kernel_initializer(initializer):
    """Initialize low-precision kernels through fp32 for JAX LAPACK safety."""

    def initialize(shape, dtype=None):
        dtype_name = standardize_dtype(dtype)
        if dtype_name in {"float16", "bfloat16"}:
            return ops.cast(initializer(shape, dtype="float32"), dtype_name)
        return initializer(shape, dtype=dtype)

    return initialize


def _validate_kernel_bank_config(layer):
    if (
        layer.tie_kernel_bank
        and layer.kernel_bank_size is not None
        and layer.kernel_bank_size < layer.filters
    ):
        raise ValueError(
            f"kernel_bank_size ({layer.kernel_bank_size}) must be greater than "
            f"or equal to filters ({layer.filters})."
        )


def _get_bank_ref(layer, signature, capacity):
    policy_signature = (
        "dtype_policy",
        backend(),
        layer.dtype_policy.name,
        standardize_dtype(layer.variable_dtype),
        standardize_dtype(layer.compute_dtype),
    )
    signature = tuple(signature) + (policy_signature,)
    supplied = layer._kernel_bank_ref
    if supplied is not None:
        if supplied.signature != _freeze_signature(signature):
            raise ValueError("Serialized kernel bank signature is incompatible.")
        if supplied.capacity < layer.filters:
            _reject_kernel_bank_expansion(
                supplied.bank_id, supplied.capacity, layer.filters
            )
        return supplied

    key = (layer.kernel_bank_id, tuple(signature))
    with type(layer)._KERNEL_BANKS_LOCK:
        bank = type(layer)._KERNEL_BANKS.get(key)
        if bank is None:
            bank = _KernelBankRef(layer.kernel_bank_id, signature, capacity)
            type(layer)._KERNEL_BANKS[key] = bank
        elif capacity > bank.capacity:
            _reject_kernel_bank_expansion(
                layer.kernel_bank_id, bank.capacity, capacity
            )
    layer._kernel_bank_ref = bank
    return bank


def _conv_output_shape(layer, input_shape):
    return compute_conv_output_shape(
        input_shape,
        layer.filters,
        tuple(layer.kernel_size),
        strides=tuple(layer.strides),
        padding=layer.padding,
        data_format=layer.data_format or "channels_last",
        dilation_rate=tuple(layer.dilation_rate),
    )


def _conv_transpose_output_shape(layer, input_shape):
    return tuple(
        compute_conv_transpose_output_shape(
            input_shape,
            tuple(layer.kernel_size),
            layer.filters,
            strides=tuple(layer.strides),
            padding=layer.padding,
            output_padding=layer.output_padding,
            data_format=layer.data_format or "channels_last",
            dilation_rate=tuple(layer.dilation_rate),
        )
    )


def _standardize_output_padding(output_padding, rank):
    if output_padding is None:
        return None
    if isinstance(output_padding, int):
        return (output_padding,) * rank
    output_padding = tuple(output_padding)
    if len(output_padding) != rank:
        raise ValueError(
            f"output_padding must have {rank} values, got {output_padding}"
        )
    return output_padding


def _to_channels_last(value, data_format):
    """Move a public channels-first tensor to TensorFlow's CPU-safe layout."""
    if data_format != "channels_first":
        return value
    rank = len(value.shape)
    return ops.transpose(value, (0,) + tuple(range(2, rank)) + (1,))


def _from_channels_last(value, data_format):
    """Restore a CPU-safe channels-last result to the public layout."""
    if data_format != "channels_first":
        return value
    rank = len(value.shape)
    return ops.transpose(value, (0, rank - 1) + tuple(range(1, rank - 1)))


def _channels_last_yat_score(layer, dot_prod_map, distance_sq_map):
    """Apply the complete CPU-safe YAT tail, then restore public layout."""
    return _from_channels_last(
        yat_score(layer, dot_prod_map, distance_sq_map, data_format="channels_last"),
        layer.data_format,
    )


def _build_forward_kernel(layer, input_dim):
    if not layer.tie_kernel_bank:
        layer.kernel = layer.add_weight(
            name="kernel",
            shape=tuple(layer.kernel_size)
            + (input_dim // layer.groups, layer.filters),
            initializer=_safe_kernel_initializer(layer.kernel_initializer),
            regularizer=layer.kernel_regularizer,
            constraint=layer.kernel_constraint,
            trainable=True,
        )
        return

    capacity = layer.kernel_bank_size or layer.filters
    signature = (
        "forward",
        tuple(layer.kernel_size),
        input_dim // layer.groups,
        layer.groups,
    )
    bank = _get_bank_ref(layer, signature, capacity)
    with bank.lock:
        if bank.variable is None:
            bank.variable = layer.add_weight(
                name="kernel",
                shape=tuple(layer.kernel_size)
                + (input_dim // layer.groups, bank.capacity),
                initializer=_safe_kernel_initializer(layer.kernel_initializer),
                regularizer=layer.kernel_regularizer,
                constraint=layer.kernel_constraint,
                trainable=True,
            )
        layer.kernel = bank.variable
    layer._kernel_slice = slice(0, layer.filters)


def _build_transpose_kernel(layer, input_dim):
    """Create or attach a fixed-capacity transpose-convolution kernel bank."""
    if not layer.tie_kernel_bank:
        layer.kernel = layer.add_weight(
            name="kernel",
            shape=tuple(layer.kernel_size) + (layer.filters, input_dim),
            initializer=_safe_kernel_initializer(layer.kernel_initializer),
            regularizer=layer.kernel_regularizer,
            constraint=layer.kernel_constraint,
            trainable=True,
        )
        return

    capacity = layer.kernel_bank_size or layer.filters
    signature = ("transpose", tuple(layer.kernel_size), input_dim)
    bank = _get_bank_ref(layer, signature, capacity)
    with bank.lock:
        if bank.variable is None:
            bank.variable = layer.add_weight(
                name="kernel",
                shape=tuple(layer.kernel_size) + (bank.capacity, input_dim),
                initializer=_safe_kernel_initializer(layer.kernel_initializer),
                regularizer=layer.kernel_regularizer,
                constraint=layer.kernel_constraint,
                trainable=True,
            )
        layer.kernel = bank.variable
    layer._kernel_slice = slice(0, layer.filters)


@keras_export("keras.layers.YatConv1D")
class YatConv1D(_KernelBankSerializationMixin, Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = weakref.WeakValueDictionary()
    _KERNEL_BANKS_LOCK = threading.Lock()

    """1D YAT convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved with the layer
    input to produce a tensor of outputs using the YAT  algorithm.
    YAT uses squared dot products divided by squared Euclidean distances plus epsilon.

    Note: This layer is activation-free. Any activation function should be applied
    as a separate layer after this layer.

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the number
            of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer, specifying the
            length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the
            stride length of the convolution. Defaults to 1.
        padding: One of `"valid"`, `"same"` or `"causal"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding with zeros
            evenly to the left/right or up/down of the input such that output has
            the same height/width dimension as the input. `"causal"` results in
            causal (dilated) convolutions, e.g. `output[t]` does not depend on
            `input[t+1:]`. Defaults to `"valid"`.
        data_format: A string, one of `channels_last` (default) or
            `channels_first`. The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch_size, steps, features)` while `channels_first` corresponds to
            inputs with shape `(batch_size, features, steps)`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution. Defaults to 1.
        groups: A positive integer specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters / groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        use_bias: Boolean, whether the layer uses a bias vector. Defaults to `True`.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to `True`.
        epsilon: Float, small constant added to denominator for numerical stability.
            Defaults to 1e-5.
        kernel_initializer: Initializer for the `kernel` weights matrix (see
            `keras.initializers`). Defaults to `"orthogonal"`.
        bias_initializer: Initializer for the bias vector (see
            `keras.initializers`). Defaults to `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix (see `keras.regularizers`).
        bias_regularizer: Regularizer function applied to the bias vector (see
            `keras.regularizers`).
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation") (see `keras.regularizers`).
        kernel_constraint: Constraint function applied to the kernel matrix (see
            `keras.constraints`).
        bias_constraint: Constraint function applied to the bias vector (see
            `keras.constraints`).

    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        use_bias=True,
        constant_bias=None,
        use_alpha=True,
        epsilon=1e-5,
        learnable_epsilon=False,
        weight_normalized=False,
        use_dropconnect=False,
        drop_rate=0.0,
        tie_kernel_bank=False,
        kernel_bank_size=None,
        kernel_bank_id="default",
        kernel_bank=None,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size,)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides,)
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate,)
        if any(stride != 1 for stride in self.strides) and any(
            dilation != 1 for dilation in self.dilation_rate
        ):
            raise ValueError(
                "`strides > 1` is incompatible with `dilation_rate > 1`."
            )
        self.groups = groups
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)
        self._kernel_bank_ref = kernel_bank
        _validate_kernel_bank_config(self)

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"Found `None`. Full input shape: {input_shape}"
            )
        
        input_dim = int(input_shape[channel_axis])
        
        if input_dim % self.groups != 0:
            raise ValueError(
                f"The number of input channels ({input_dim}) must be "
                f"divisible by the number of groups ({self.groups})."
            )
        
        if self.filters % self.groups != 0:
            raise ValueError(
                f"The number of filters ({self.filters}) must be "
                f"divisible by the number of groups ({self.groups})."
            )

        _build_forward_kernel(self, input_dim)

        # Bias: learnable parameter, or None if constant_bias is set / use_bias=False
        if self.use_bias and self._constant_bias_value is None:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        if self.use_alpha:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=_epsilon_initializer(raw_eps),
                dtype=_epsilon_weight_dtype(self),
                trainable=True,
            )
        else:
            self.epsilon_param = None

        # Apply build-time weight normalization (per filter, last axis)
        # Note: skipped when tie_kernel_bank to avoid mutating shared state.
        if self.weight_normalized and not self.tie_kernel_bank:
            reduce_axes = tuple(range(self.kernel.ndim - 1))
            kernel_norm = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=reduce_axes, keepdims=True)
            )
            self.kernel.assign(self.kernel / (kernel_norm + 1e-8))

        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        kernel = self.kernel
        # Slice shared bank if tying
        if self.tie_kernel_bank:
            kernel = kernel[..., self._kernel_slice]

        inputs = reduction_safe_upcast(inputs)
        kernel = reduction_safe_upcast(kernel)
        inputs = _to_channels_last(inputs, self.data_format)

        # DropConnect: random kernel mask during training
        if self.use_dropconnect and training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = ops.cast(
                ops.random.uniform(ops.shape(kernel), dtype=kernel.dtype) < keep_prob,
                kernel.dtype,
            )
            kernel = (kernel * mask) / keep_prob

        # Optional forward-time weight normalization (per filter, last axis)
        if self.weight_normalized:
            reduce_axes = tuple(range(kernel.ndim - 1))
            kernel = kernel / (
                ops.sqrt(ops.sum(ops.square(kernel), axis=reduce_axes, keepdims=True)) + 1e-8
            )

        # Keras' low-level conv op only accepts valid/same.  Causal Conv1D is
        # left padded by the effective dilated kernel size, then evaluated as
        # valid for both the dot product and patch norm.
        conv_inputs = inputs
        conv_padding = self.padding
        if self.padding == "causal":
            left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_width = ((0, 0), (left_pad, 0), (0, 0))
            conv_inputs = ops.pad(inputs, pad_width)
            conv_padding = "valid"

        # Compute standard convolution (dot product)
        dot_prod_map = ops.conv(
            conv_inputs,
            kernel,
            strides=self.strides,
            padding=conv_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = conv_inputs * conv_inputs

        # Create ones kernel for computing patch squared sums
        input_channels_per_group = kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (
            input_channels_per_group,
            self.groups,
        )
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=conv_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        patch_sq_sum_map = ops.repeat(
            patch_sq_sum_map_raw,
            self.filters // self.groups,
            axis=-1,
        )

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            kernel_sq_sum_per_filter = ops.sum(
                kernel ** 2, axis=tuple(range(kernel.ndim - 1))
            )

        # Reshape for broadcasting
        kernel_sq_sum_reshaped = ops.reshape(
            kernel_sq_sum_per_filter, (1, 1, -1)
        )

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return _channels_last_yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        return _conv_output_shape(self, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "use_bias": self.use_bias,
            "constant_bias": self.constant_bias,
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
            "learnable_epsilon": self.learnable_epsilon,
            "weight_normalized": self.weight_normalized,
            "use_dropconnect": self.use_dropconnect,
            "drop_rate": self.drop_rate,
            "tie_kernel_bank": self.tie_kernel_bank,
            "kernel_bank_size": self.kernel_bank_size,
            "kernel_bank_id": self.kernel_bank_id,
            "kernel_bank": self._kernel_bank_ref if self.tie_kernel_bank else None,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


@keras_export("keras.layers.YatConv2D")
class YatConv2D(_KernelBankSerializationMixin, Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = weakref.WeakValueDictionary()
    _KERNEL_BANKS_LOCK = threading.Lock()

    """2D YAT convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved with the layer
    input to produce a tensor of outputs using the YAT algorithm.

    Note: This layer is activation-free. Any activation function should be applied
    as a separate layer after this layer.

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the number
            of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single
            integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the height and width. Can be a single
            integer to specify the same value for all spatial dimensions.
            Defaults to `(1, 1)`.
        padding: one of `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding with zeros
            evenly to the left/right or up/down of the input such that output has
            the same height/width dimension as the input.
        data_format: A string, one of `channels_last` (default) or
            `channels_first`. The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape `(batch, channels, height, width)`.
        dilation_rate: an integer or tuple/list of 2 integers, specifying the
            dilation rate to use for dilated convolution. Can be a single integer
            to specify the same value for all spatial dimensions. Defaults to `(1, 1)`.
        groups: A positive integer specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters / groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        use_bias: Boolean, whether the layer uses a bias vector.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to `True`.
        epsilon: Float, small constant added to denominator for numerical stability.
            Defaults to 1e-5.
        kernel_initializer: Initializer for the `kernel` weights matrix (see
            `keras.initializers`). Defaults to `"orthogonal"`.
        bias_initializer: Initializer for the bias vector (see
            `keras.initializers`). Defaults to `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix (see `keras.regularizers`).
        bias_regularizer: Regularizer function applied to the bias vector (see
            `keras.regularizers`).
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation") (see `keras.regularizers`).
        kernel_constraint: Constraint function applied to the kernel matrix (see
            `keras.constraints`).
        bias_constraint: Constraint function applied to the bias vector (see
            `keras.constraints`).

    Input shape:
        4D tensor with shape: `(batch_size, rows, cols, channels)` if
        `data_format` is `"channels_last"` or 4D tensor with shape:
        `(batch_size, channels, rows, cols)` if `data_format` is
        `"channels_first"`.

    Output shape:
        4D tensor with shape: `(batch_size, new_rows, new_cols, filters)` if
        `data_format` is `"channels_last"` or 4D tensor with shape:
        `(batch_size, filters, new_rows, new_cols)` if `data_format` is
        `"channels_first"`. `rows` and `cols` values might have changed due to
        padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        use_bias=True,
        constant_bias=None,
        use_alpha=True,
        epsilon=1e-5,
        learnable_epsilon=False,
        weight_normalized=False,
        use_dropconnect=False,
        drop_rate=0.0,
        tie_kernel_bank=False,
        kernel_bank_size=None,
        kernel_bank_id="default",
        kernel_bank=None,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate, dilation_rate)
        self.groups = groups
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)
        self._kernel_bank_ref = kernel_bank
        _validate_kernel_bank_config(self)

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(ndim=4)
        self.supports_masking = True

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"Found `None`. Full input shape: {input_shape}"
            )
        
        input_dim = int(input_shape[channel_axis])
        
        if input_dim % self.groups != 0:
            raise ValueError(
                f"The number of input channels ({input_dim}) must be "
                f"divisible by the number of groups ({self.groups})."
            )
        
        if self.filters % self.groups != 0:
            raise ValueError(
                f"The number of filters ({self.filters}) must be "
                f"divisible by the number of groups ({self.groups})."
            )

        _build_forward_kernel(self, input_dim)

        # Bias: learnable parameter, or None if constant_bias is set / use_bias=False
        if self.use_bias and self._constant_bias_value is None:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        if self.use_alpha:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=_epsilon_initializer(raw_eps),
                dtype=_epsilon_weight_dtype(self),
                trainable=True,
            )
        else:
            self.epsilon_param = None

        # Apply build-time weight normalization (per filter, last axis)
        # Skipped when tie_kernel_bank to avoid mutating shared state.
        if self.weight_normalized and not self.tie_kernel_bank:
            reduce_axes = tuple(range(self.kernel.ndim - 1))
            kernel_norm = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=reduce_axes, keepdims=True)
            )
            self.kernel.assign(self.kernel / (kernel_norm + 1e-8))

        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        kernel = self.kernel
        # Slice shared bank if tying
        if self.tie_kernel_bank:
            kernel = kernel[..., self._kernel_slice]

        inputs = reduction_safe_upcast(inputs)
        kernel = reduction_safe_upcast(kernel)
        inputs = _to_channels_last(inputs, self.data_format)

        # DropConnect: random kernel mask during training
        if self.use_dropconnect and training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = ops.cast(
                ops.random.uniform(ops.shape(kernel), dtype=kernel.dtype) < keep_prob,
                kernel.dtype,
            )
            kernel = (kernel * mask) / keep_prob

        # Optional forward-time weight normalization (per filter, last axis)
        if self.weight_normalized:
            reduce_axes = tuple(range(kernel.ndim - 1))
            kernel = kernel / (
                ops.sqrt(ops.sum(ops.square(kernel), axis=reduce_axes, keepdims=True)) + 1e-8
            )

        # Compute standard convolution (dot product)
        dot_prod_map = ops.conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        input_channels_per_group = kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (
            input_channels_per_group,
            self.groups,
        )
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        patch_sq_sum_map = ops.repeat(
            patch_sq_sum_map_raw,
            self.filters // self.groups,
            axis=-1,
        )

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            kernel_sq_sum_per_filter = ops.sum(
                kernel ** 2, axis=tuple(range(kernel.ndim - 1))
            )

        # Reshape for broadcasting
        kernel_sq_sum_reshaped = ops.reshape(
            kernel_sq_sum_per_filter, (1, 1, 1, -1)
        )

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return _channels_last_yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        return _conv_output_shape(self, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "use_bias": self.use_bias,
            "constant_bias": self.constant_bias,
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
            "learnable_epsilon": self.learnable_epsilon,
            "weight_normalized": self.weight_normalized,
            "use_dropconnect": self.use_dropconnect,
            "drop_rate": self.drop_rate,
            "tie_kernel_bank": self.tie_kernel_bank,
            "kernel_bank_size": self.kernel_bank_size,
            "kernel_bank_id": self.kernel_bank_id,
            "kernel_bank": self._kernel_bank_ref if self.tie_kernel_bank else None,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


@keras_export("keras.layers.YatConv3D")
class YatConv3D(_KernelBankSerializationMixin, Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = weakref.WeakValueDictionary()
    _KERNEL_BANKS_LOCK = threading.Lock()

    """3D YAT convolution layer (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved with the layer
    input to produce a tensor of outputs using the YAT  algorithm.

    Note: This layer is activation-free. Any activation function should be applied
    as a separate layer after this layer.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
        strides: An integer or tuple/list of 3 integers. Defaults to `(1, 1, 1)`.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or `channels_first`.
        dilation_rate: an integer or tuple/list of 3 integers. Defaults to `(1, 1, 1)`.
        groups: A positive integer specifying the number of groups.
        use_bias: Boolean, whether the layer uses a bias vector.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to `True`.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-5.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output.
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        5D tensor with shape: `(batch_size, conv_dim1, conv_dim2, conv_dim3, channels)`

    Output shape:
        5D tensor with shape: `(batch_size, new_dim1, new_dim2, new_dim3, filters)`
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1, 1),
        groups=1,
        use_bias=True,
        constant_bias=None,
        use_alpha=True,
        epsilon=1e-5,
        learnable_epsilon=False,
        weight_normalized=False,
        use_dropconnect=False,
        drop_rate=0.0,
        tie_kernel_bank=False,
        kernel_bank_size=None,
        kernel_bank_id="default",
        kernel_bank=None,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides, strides)
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate, dilation_rate, dilation_rate)
        self.groups = groups
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)
        self._kernel_bank_ref = kernel_bank
        _validate_kernel_bank_config(self)

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(ndim=5)
        self.supports_masking = True

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"Found `None`. Full input shape: {input_shape}"
            )
        
        input_dim = int(input_shape[channel_axis])
        
        if input_dim % self.groups != 0:
            raise ValueError(
                f"The number of input channels ({input_dim}) must be "
                f"divisible by the number of groups ({self.groups})."
            )
        
        if self.filters % self.groups != 0:
            raise ValueError(
                f"The number of filters ({self.filters}) must be "
                f"divisible by the number of groups ({self.groups})."
            )

        _build_forward_kernel(self, input_dim)

        # Bias: learnable parameter, or None if constant_bias is set / use_bias=False
        if self.use_bias and self._constant_bias_value is None:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        if self.use_alpha:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=_epsilon_initializer(raw_eps),
                dtype=_epsilon_weight_dtype(self),
                trainable=True,
            )
        else:
            self.epsilon_param = None

        # Apply build-time weight normalization (per filter, last axis)
        # Skipped when tie_kernel_bank to avoid mutating shared state.
        if self.weight_normalized and not self.tie_kernel_bank:
            reduce_axes = tuple(range(self.kernel.ndim - 1))
            kernel_norm = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=reduce_axes, keepdims=True)
            )
            self.kernel.assign(self.kernel / (kernel_norm + 1e-8))

        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        kernel = self.kernel
        # Slice shared bank if tying
        if self.tie_kernel_bank:
            kernel = kernel[..., self._kernel_slice]

        inputs = reduction_safe_upcast(inputs)
        kernel = reduction_safe_upcast(kernel)
        inputs = _to_channels_last(inputs, self.data_format)

        # DropConnect: random kernel mask during training
        if self.use_dropconnect and training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = ops.cast(
                ops.random.uniform(ops.shape(kernel), dtype=kernel.dtype) < keep_prob,
                kernel.dtype,
            )
            kernel = (kernel * mask) / keep_prob

        # Optional forward-time weight normalization (per filter, last axis)
        if self.weight_normalized:
            reduce_axes = tuple(range(kernel.ndim - 1))
            kernel = kernel / (
                ops.sqrt(ops.sum(ops.square(kernel), axis=reduce_axes, keepdims=True)) + 1e-8
            )

        # Compute standard convolution (dot product)
        dot_prod_map = ops.conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        input_channels_per_group = kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (
            input_channels_per_group,
            self.groups,
        )
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        patch_sq_sum_map = ops.repeat(
            patch_sq_sum_map_raw,
            self.filters // self.groups,
            axis=-1,
        )

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            kernel_sq_sum_per_filter = ops.sum(
                kernel ** 2, axis=tuple(range(kernel.ndim - 1))
            )

        # Reshape for broadcasting
        kernel_sq_sum_reshaped = ops.reshape(
            kernel_sq_sum_per_filter, (1, 1, 1, 1, -1)
        )

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return _channels_last_yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        return _conv_output_shape(self, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "use_bias": self.use_bias,
            "constant_bias": self.constant_bias,
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
            "learnable_epsilon": self.learnable_epsilon,
            "weight_normalized": self.weight_normalized,
            "use_dropconnect": self.use_dropconnect,
            "drop_rate": self.drop_rate,
            "tie_kernel_bank": self.tie_kernel_bank,
            "kernel_bank_size": self.kernel_bank_size,
            "kernel_bank_id": self.kernel_bank_id,
            "kernel_bank": self._kernel_bank_ref if self.tie_kernel_bank else None,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


@keras_export("keras.layers.YatConvTranspose1D")
class YatConvTranspose1D(_KernelBankSerializationMixin, Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = weakref.WeakValueDictionary()
    _KERNEL_BANKS_LOCK = threading.Lock()

    """1D YAT transposed convolution layer (deconvolution).

    This layer creates a transposed convolution kernel using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of a single integer.
        strides: An integer or tuple/list of a single integer. Defaults to 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` or `channels_first`.
        dilation_rate: an integer or tuple/list of a single integer.
        output_padding: Optional integer or tuple/list of a single integer
            specifying the added output size along the spatial dimension.
        use_bias: Boolean, whether the layer uses a bias vector.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to `True`.
        epsilon: Float, small constant for numerical stability.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.

    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        use_bias=True,
        constant_bias=None,
        use_alpha=True,
        epsilon=1e-5,
        learnable_epsilon=False,
        weight_normalized=False,
        use_dropconnect=False,
        drop_rate=0.0,
        tie_kernel_bank=False,
        kernel_bank_size=None,
        kernel_bank_id="default",
        kernel_bank=None,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        output_padding=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size,)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides,)
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate,)
        self.output_padding = _standardize_output_padding(output_padding, 1)
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)
        self._kernel_bank_ref = kernel_bank
        _validate_kernel_bank_config(self)

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"Found `None`. Full input shape: {input_shape}"
            )

        input_dim = int(input_shape[channel_axis])

        _build_transpose_kernel(self, input_dim)

        # Bias: learnable parameter, or None if constant_bias is set / use_bias=False
        if self.use_bias and self._constant_bias_value is None:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        if self.use_alpha:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=_epsilon_initializer(raw_eps),
                dtype=_epsilon_weight_dtype(self),
                trainable=True,
            )
        else:
            self.epsilon_param = None

        # Apply build-time weight normalization (per filter)
        # Filter axis = len(kernel_size); reduce over all OTHER axes
        if self.weight_normalized:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(self.kernel.ndim) if i != filter_axis)
            kernel_norm = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=reduce_axes, keepdims=True)
            )
            self.kernel.assign(self.kernel / (kernel_norm + 1e-8))

        self.input_dim = input_dim
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        kernel = self.kernel
        # Slice shared bank if tying (transpose conv: filter axis is at len(kernel_size))
        if self.tie_kernel_bank:
            filter_axis = len(self.kernel_size)
            slicer = [slice(None)] * kernel.ndim
            slicer[filter_axis] = self._kernel_slice
            kernel = kernel[tuple(slicer)]

        inputs = reduction_safe_upcast(inputs)
        kernel = reduction_safe_upcast(kernel)
        inputs = _to_channels_last(inputs, self.data_format)

        # DropConnect: random kernel mask during training
        if self.use_dropconnect and training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = ops.cast(
                ops.random.uniform(ops.shape(kernel), dtype=kernel.dtype) < keep_prob,
                kernel.dtype,
            )
            kernel = (kernel * mask) / keep_prob

        # Optional forward-time weight normalization (per filter)
        if self.weight_normalized:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel = kernel / (
                ops.sqrt(ops.sum(ops.square(kernel), axis=reduce_axes, keepdims=True)) + 1e-8
            )

        # Compute transposed convolution (dot product)
        dot_prod_map = ops.conv_transpose(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input for YAT distance
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        ones_kernel_shape = tuple(self.kernel_size) + (1, self.input_dim)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            # Sum over all axes except the filter axis.
            # Transpose conv kernel shape: (*kernel_size, filters, in_dim)
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel_sq_sum_per_filter = ops.sum(kernel ** 2, axis=reduce_axes)

        kernel_sq_sum_reshaped = ops.reshape(
            kernel_sq_sum_per_filter, (1, 1, -1)
        )

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return _channels_last_yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        return _conv_transpose_output_shape(self, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "output_padding": self.output_padding,
            "use_bias": self.use_bias,
            "constant_bias": self.constant_bias,
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
            "learnable_epsilon": self.learnable_epsilon,
            "weight_normalized": self.weight_normalized,
            "use_dropconnect": self.use_dropconnect,
            "drop_rate": self.drop_rate,
            "tie_kernel_bank": self.tie_kernel_bank,
            "kernel_bank_size": self.kernel_bank_size,
            "kernel_bank_id": self.kernel_bank_id,
            "kernel_bank": self._kernel_bank_ref if self.tie_kernel_bank else None,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


@keras_export("keras.layers.YatConvTranspose2D")
class YatConvTranspose2D(_KernelBankSerializationMixin, Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = weakref.WeakValueDictionary()
    _KERNEL_BANKS_LOCK = threading.Lock()

    """2D YAT transposed convolution layer (deconvolution).

    This layer creates a transposed convolution kernel using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers.
        strides: An integer or tuple/list of 2 integers. Defaults to (1, 1).
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` or `channels_first`.
        dilation_rate: an integer or tuple/list of 2 integers.
        output_padding: Optional integer or tuple/list of 2 integers specifying
            the added output size along each spatial dimension.
        use_bias: Boolean, whether the layer uses a bias vector.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to `True`.
        epsilon: Float, small constant for numerical stability.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.

    Input shape:
        4D tensor with shape: `(batch_size, rows, cols, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, new_rows, new_cols, filters)`
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        use_bias=True,
        constant_bias=None,
        use_alpha=True,
        epsilon=1e-5,
        learnable_epsilon=False,
        weight_normalized=False,
        use_dropconnect=False,
        drop_rate=0.0,
        tie_kernel_bank=False,
        kernel_bank_size=None,
        kernel_bank_id="default",
        kernel_bank=None,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        output_padding=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate, dilation_rate)
        self.output_padding = _standardize_output_padding(output_padding, 2)
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)
        self._kernel_bank_ref = kernel_bank
        _validate_kernel_bank_config(self)

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(ndim=4)
        self.supports_masking = True

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"Found `None`. Full input shape: {input_shape}"
            )

        input_dim = int(input_shape[channel_axis])

        _build_transpose_kernel(self, input_dim)

        # Bias: learnable parameter, or None if constant_bias is set / use_bias=False
        if self.use_bias and self._constant_bias_value is None:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        if self.use_alpha:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=_epsilon_initializer(raw_eps),
                dtype=_epsilon_weight_dtype(self),
                trainable=True,
            )
        else:
            self.epsilon_param = None

        # Apply build-time weight normalization (per filter)
        # Skipped when tie_kernel_bank to avoid mutating shared state.
        if self.weight_normalized and not self.tie_kernel_bank:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(self.kernel.ndim) if i != filter_axis)
            kernel_norm = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=reduce_axes, keepdims=True)
            )
            self.kernel.assign(self.kernel / (kernel_norm + 1e-8))

        self.input_dim = input_dim
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        kernel = self.kernel
        # Slice shared bank if tying (transpose conv: filter axis is at len(kernel_size))
        if self.tie_kernel_bank:
            filter_axis = len(self.kernel_size)
            slicer = [slice(None)] * kernel.ndim
            slicer[filter_axis] = self._kernel_slice
            kernel = kernel[tuple(slicer)]

        inputs = reduction_safe_upcast(inputs)
        kernel = reduction_safe_upcast(kernel)
        inputs = _to_channels_last(inputs, self.data_format)

        # DropConnect: random kernel mask during training
        if self.use_dropconnect and training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = ops.cast(
                ops.random.uniform(ops.shape(kernel), dtype=kernel.dtype) < keep_prob,
                kernel.dtype,
            )
            kernel = (kernel * mask) / keep_prob

        # Optional forward-time weight normalization (per filter)
        if self.weight_normalized:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel = kernel / (
                ops.sqrt(ops.sum(ops.square(kernel), axis=reduce_axes, keepdims=True)) + 1e-8
            )

        # Compute transposed convolution (dot product)
        dot_prod_map = ops.conv_transpose(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input for YAT distance
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        ones_kernel_shape = tuple(self.kernel_size) + (1, self.input_dim)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            # Sum over all axes except the filter axis.
            # Transpose conv kernel shape: (*kernel_size, filters, in_dim)
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel_sq_sum_per_filter = ops.sum(kernel ** 2, axis=reduce_axes)

        kernel_sq_sum_reshaped = ops.reshape(
            kernel_sq_sum_per_filter, (1, 1, 1, -1)
        )

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return _channels_last_yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        return _conv_transpose_output_shape(self, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "output_padding": self.output_padding,
            "use_bias": self.use_bias,
            "constant_bias": self.constant_bias,
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
            "learnable_epsilon": self.learnable_epsilon,
            "weight_normalized": self.weight_normalized,
            "use_dropconnect": self.use_dropconnect,
            "drop_rate": self.drop_rate,
            "tie_kernel_bank": self.tie_kernel_bank,
            "kernel_bank_size": self.kernel_bank_size,
            "kernel_bank_id": self.kernel_bank_id,
            "kernel_bank": self._kernel_bank_ref if self.tie_kernel_bank else None,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


@keras_export("keras.layers.YatConvTranspose3D")
class YatConvTranspose3D(_KernelBankSerializationMixin, Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = weakref.WeakValueDictionary()
    _KERNEL_BANKS_LOCK = threading.Lock()

    """3D YAT transposed convolution layer (deconvolution).

    This layer creates a transposed convolution kernel using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 3 integers.
        strides: An integer or tuple/list of 3 integers. Defaults to (1, 1, 1).
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` or `channels_first`.
        dilation_rate: an integer or tuple/list of 3 integers.
        output_padding: Optional integer or tuple/list of 3 integers specifying
            the added output size along each spatial dimension.
        use_bias: Boolean, whether the layer uses a bias vector.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to `True`.
        epsilon: Float, small constant for numerical stability.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.

    Input shape:
        5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`

    Output shape:
        5D tensor with shape: `(batch_size, new_dim1, new_dim2, new_dim3, filters)`
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1, 1),
        use_bias=True,
        constant_bias=None,
        use_alpha=True,
        epsilon=1e-5,
        learnable_epsilon=False,
        weight_normalized=False,
        use_dropconnect=False,
        drop_rate=0.0,
        tie_kernel_bank=False,
        kernel_bank_size=None,
        kernel_bank_id="default",
        kernel_bank=None,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        output_padding=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides, strides)
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate, dilation_rate, dilation_rate)
        self.output_padding = _standardize_output_padding(output_padding, 3)
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)
        self._kernel_bank_ref = kernel_bank
        _validate_kernel_bank_config(self)

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(ndim=5)
        self.supports_masking = True

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"Found `None`. Full input shape: {input_shape}"
            )

        input_dim = int(input_shape[channel_axis])

        _build_transpose_kernel(self, input_dim)

        # Bias: learnable parameter, or None if constant_bias is set / use_bias=False
        if self.use_bias and self._constant_bias_value is None:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        if self.use_alpha:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=_epsilon_initializer(raw_eps),
                dtype=_epsilon_weight_dtype(self),
                trainable=True,
            )
        else:
            self.epsilon_param = None

        # Apply build-time weight normalization (per filter)
        # Skipped when tie_kernel_bank to avoid mutating shared state.
        if self.weight_normalized and not self.tie_kernel_bank:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(self.kernel.ndim) if i != filter_axis)
            kernel_norm = ops.sqrt(
                ops.sum(ops.square(self.kernel), axis=reduce_axes, keepdims=True)
            )
            self.kernel.assign(self.kernel / (kernel_norm + 1e-8))

        self.input_dim = input_dim
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        kernel = self.kernel
        # Slice shared bank if tying (transpose conv: filter axis is at len(kernel_size))
        if self.tie_kernel_bank:
            filter_axis = len(self.kernel_size)
            slicer = [slice(None)] * kernel.ndim
            slicer[filter_axis] = self._kernel_slice
            kernel = kernel[tuple(slicer)]

        inputs = reduction_safe_upcast(inputs)
        kernel = reduction_safe_upcast(kernel)
        inputs = _to_channels_last(inputs, self.data_format)

        # DropConnect: random kernel mask during training
        if self.use_dropconnect and training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = ops.cast(
                ops.random.uniform(ops.shape(kernel), dtype=kernel.dtype) < keep_prob,
                kernel.dtype,
            )
            kernel = (kernel * mask) / keep_prob

        # Optional forward-time weight normalization (per filter)
        if self.weight_normalized:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel = kernel / (
                ops.sqrt(ops.sum(ops.square(kernel), axis=reduce_axes, keepdims=True)) + 1e-8
            )

        # Compute transposed convolution (dot product)
        dot_prod_map = ops.conv_transpose(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input for YAT distance
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        ones_kernel_shape = tuple(self.kernel_size) + (1, self.input_dim)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )

        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter (1.0 if normalized).
        # Transpose conv kernel shape: (*kernel_size, filters, in_dim)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel_sq_sum_per_filter = ops.sum(kernel ** 2, axis=reduce_axes)

        kernel_sq_sum_reshaped = ops.reshape(
            kernel_sq_sum_per_filter, (1, 1, 1, 1, -1)
        )

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return _channels_last_yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        return _conv_transpose_output_shape(self, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "output_padding": self.output_padding,
            "use_bias": self.use_bias,
            "constant_bias": self.constant_bias,
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
            "learnable_epsilon": self.learnable_epsilon,
            "weight_normalized": self.weight_normalized,
            "use_dropconnect": self.use_dropconnect,
            "drop_rate": self.drop_rate,
            "tie_kernel_bank": self.tie_kernel_bank,
            "kernel_bank_size": self.kernel_bank_size,
            "kernel_bank_id": self.kernel_bank_id,
            "kernel_bank": self._kernel_bank_ref if self.tie_kernel_bank else None,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


# DEPRECATED: lowercase aliases. The canonical names are the uppercase
# variants (YatConv1D, YatConv2D, ...) — they match the names exported
# from every other backend (torch / nnx / linen / tf). The lowercase
# aliases are kept for backward compatibility and will be removed in a
# future minor release.
YatConv1d = YatConv1D
YatConv2d = YatConv2D
YatConv3d = YatConv3D
YatConvTranspose1d = YatConvTranspose1D
YatConvTranspose2d = YatConvTranspose2D
YatConvTranspose3d = YatConvTranspose3D
