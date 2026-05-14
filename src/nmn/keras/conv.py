"""YAT convolution layers for Keras/TensorFlow."""

import logging
import threading

from keras.src import activations, constraints, initializers, regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src import ops
import math

from ._yat_core import yat_score

logger = logging.getLogger(__name__)


@keras_export("keras.layers.YatConv1D")
class YatConv1D(Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = {}
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
        self.groups = groups
        self.use_alpha = use_alpha
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)

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

        # Kernel: standalone or from a shared bank
        if self.tie_kernel_bank:
            bank_filters = self.kernel_bank_size or self.filters
            bank_kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, bank_filters)
            bank_key = (
                self.kernel_bank_id,
                tuple(self.kernel_size),
                input_dim // self.groups,
                self.groups,
            )
            with type(self)._KERNEL_BANKS_LOCK:
                shared_kernel = type(self)._KERNEL_BANKS.get(bank_key)
                if shared_kernel is None:
                    self.kernel = self.add_weight(
                        name="kernel",
                        shape=bank_kernel_shape,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        trainable=True,
                    )
                    type(self)._KERNEL_BANKS[bank_key] = self.kernel
                else:
                    existing_filters = shared_kernel.shape[-1]
                    if bank_filters > existing_filters:
                        logger.info(
                            "Auto-expanding Keras kernel bank '%s': %d -> %d filters",
                            self.kernel_bank_id, existing_filters, bank_filters,
                        )
                        # Expand the shared kernel by sampling new tail filters
                        new_kernel_val = self.kernel_initializer(bank_kernel_shape, dtype=shared_kernel.dtype)
                        old_val = shared_kernel.numpy() if hasattr(shared_kernel, "numpy") else ops.convert_to_numpy(shared_kernel)
                        new_kernel_val = ops.convert_to_tensor(new_kernel_val)
                        # Splice old values into the front
                        import numpy as _np
                        new_arr = _np.array(new_kernel_val)
                        new_arr[..., :existing_filters] = _np.array(old_val)
                        shared_kernel.assign(ops.convert_to_tensor(new_arr))
                    self.kernel = shared_kernel
            self._kernel_slice = slice(0, self.filters)
        else:
            kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, self.filters)
            self.kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )

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
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=initializers.Constant(raw_eps),
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
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        input_channels_per_group = kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels_per_group, 1)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        channel_axis = 1 if self.data_format == "channels_first" else -1
        if self.groups > 1:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters // self.groups, axis=channel_axis)
        else:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=channel_axis)

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            kernel_sq_sum_per_filter = ops.sum(
                kernel ** 2, axis=tuple(range(kernel.ndim - 1))
            )

        # Reshape for broadcasting
        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, -1))

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            length = input_shape[2]
            if length is not None:
                if self.padding == "valid":
                    length = length - self.kernel_size[0] + 1
                elif self.padding == "causal":
                    length = length
                length = (length + self.strides[0] - 1) // self.strides[0]
            return (input_shape[0], self.filters, length)
        else:
            length = input_shape[1]
            if length is not None:
                if self.padding == "valid":
                    length = length - self.kernel_size[0] + 1
                elif self.padding == "causal":
                    length = length
                length = (length + self.strides[0] - 1) // self.strides[0]
            return (input_shape[0], length, self.filters)

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
class YatConv2D(Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = {}
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
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)

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

        # Kernel: standalone or from a shared bank
        if self.tie_kernel_bank:
            bank_filters = self.kernel_bank_size or self.filters
            bank_kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, bank_filters)
            bank_key = (
                self.kernel_bank_id,
                tuple(self.kernel_size),
                input_dim // self.groups,
                self.groups,
            )
            with type(self)._KERNEL_BANKS_LOCK:
                shared_kernel = type(self)._KERNEL_BANKS.get(bank_key)
                if shared_kernel is None:
                    self.kernel = self.add_weight(
                        name="kernel",
                        shape=bank_kernel_shape,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        trainable=True,
                    )
                    type(self)._KERNEL_BANKS[bank_key] = self.kernel
                else:
                    existing_filters = shared_kernel.shape[-1]
                    if bank_filters > existing_filters:
                        logger.info(
                            "Auto-expanding Keras kernel bank '%s': %d -> %d filters",
                            self.kernel_bank_id, existing_filters, bank_filters,
                        )
                        new_kernel_val = self.kernel_initializer(bank_kernel_shape, dtype=shared_kernel.dtype)
                        import numpy as _np
                        old_arr = _np.array(shared_kernel)
                        new_arr = _np.array(new_kernel_val)
                        new_arr[..., :existing_filters] = old_arr
                        shared_kernel.assign(ops.convert_to_tensor(new_arr))
                    self.kernel = shared_kernel
            self._kernel_slice = slice(0, self.filters)
        else:
            kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, self.filters)
            self.kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )

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
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=initializers.Constant(raw_eps),
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
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        input_channels_per_group = kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels_per_group, 1)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        channel_axis = 1 if self.data_format == "channels_first" else -1
        if self.groups > 1:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters // self.groups, axis=channel_axis)
        else:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=channel_axis)

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            kernel_sq_sum_per_filter = ops.sum(
                kernel ** 2, axis=tuple(range(kernel.ndim - 1))
            )

        # Reshape for broadcasting
        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, 1, -1))

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]

        if rows is not None:
            if self.padding == "valid":
                rows = rows - self.kernel_size[0] + 1
            rows = (rows + self.strides[0] - 1) // self.strides[0]
        
        if cols is not None:
            if self.padding == "valid":
                cols = cols - self.kernel_size[1] + 1
            cols = (cols + self.strides[1] - 1) // self.strides[1]

        if self.data_format == "channels_first":
            return (input_shape[0], self.filters, rows, cols)
        else:
            return (input_shape[0], rows, cols, self.filters)

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
class YatConv3D(Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = {}
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
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)

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

        # Kernel: standalone or from a shared bank
        if self.tie_kernel_bank:
            bank_filters = self.kernel_bank_size or self.filters
            bank_kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, bank_filters)
            bank_key = (
                self.kernel_bank_id,
                tuple(self.kernel_size),
                input_dim // self.groups,
                self.groups,
            )
            with type(self)._KERNEL_BANKS_LOCK:
                shared_kernel = type(self)._KERNEL_BANKS.get(bank_key)
                if shared_kernel is None:
                    self.kernel = self.add_weight(
                        name="kernel",
                        shape=bank_kernel_shape,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        trainable=True,
                    )
                    type(self)._KERNEL_BANKS[bank_key] = self.kernel
                else:
                    existing_filters = shared_kernel.shape[-1]
                    if bank_filters > existing_filters:
                        logger.info(
                            "Auto-expanding Keras kernel bank '%s': %d -> %d filters",
                            self.kernel_bank_id, existing_filters, bank_filters,
                        )
                        new_kernel_val = self.kernel_initializer(bank_kernel_shape, dtype=shared_kernel.dtype)
                        import numpy as _np
                        old_arr = _np.array(shared_kernel)
                        new_arr = _np.array(new_kernel_val)
                        new_arr[..., :existing_filters] = old_arr
                        shared_kernel.assign(ops.convert_to_tensor(new_arr))
                    self.kernel = shared_kernel
            self._kernel_slice = slice(0, self.filters)
        else:
            kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, self.filters)
            self.kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )

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
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=initializers.Constant(raw_eps),
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
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        input_channels_per_group = kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels_per_group, 1)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        channel_axis = 1 if self.data_format == "channels_first" else -1
        if self.groups > 1:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters // self.groups, axis=channel_axis)
        else:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=channel_axis)

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            kernel_sq_sum_per_filter = ops.sum(
                kernel ** 2, axis=tuple(range(kernel.ndim - 1))
            )

        # Reshape for broadcasting
        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1, 1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, 1, 1, -1))

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            dims = [input_shape[2], input_shape[3], input_shape[4]]
        else:
            dims = [input_shape[1], input_shape[2], input_shape[3]]

        new_dims = []
        for i, dim in enumerate(dims):
            if dim is not None:
                if self.padding == "valid":
                    dim = dim - self.kernel_size[i] + 1
                dim = (dim + self.strides[i] - 1) // self.strides[i]
            new_dims.append(dim)

        if self.data_format == "channels_first":
            return (input_shape[0], self.filters) + tuple(new_dims)
        else:
            return (input_shape[0],) + tuple(new_dims) + (self.filters,)

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
class YatConvTranspose1D(Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = {}
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
        self.use_alpha = use_alpha
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)

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

        # Kernel: standalone or from a shared bank
        # Transpose conv shape: (*kernel_size, filters, input_dim) — filter axis = len(kernel_size)
        if self.tie_kernel_bank:
            bank_filters = self.kernel_bank_size or self.filters
            bank_kernel_shape = tuple(self.kernel_size) + (bank_filters, input_dim)
            bank_key = (
                self.kernel_bank_id,
                tuple(self.kernel_size),
                input_dim,
                "transpose",
            )
            with type(self)._KERNEL_BANKS_LOCK:
                shared_kernel = type(self)._KERNEL_BANKS.get(bank_key)
                if shared_kernel is None:
                    self.kernel = self.add_weight(
                        name="kernel",
                        shape=bank_kernel_shape,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        trainable=True,
                    )
                    type(self)._KERNEL_BANKS[bank_key] = self.kernel
                else:
                    filter_axis = len(self.kernel_size)
                    existing_filters = shared_kernel.shape[filter_axis]
                    if bank_filters > existing_filters:
                        logger.info(
                            "Auto-expanding Keras kernel bank '%s': %d -> %d filters",
                            self.kernel_bank_id, existing_filters, bank_filters,
                        )
                        new_kernel_val = self.kernel_initializer(bank_kernel_shape, dtype=shared_kernel.dtype)
                        import numpy as _np
                        old_arr = _np.array(shared_kernel)
                        new_arr = _np.array(new_kernel_val)
                        # Splice along the filter axis
                        slicer = [slice(None)] * new_arr.ndim
                        slicer[filter_axis] = slice(0, existing_filters)
                        new_arr[tuple(slicer)] = old_arr
                        shared_kernel.assign(ops.convert_to_tensor(new_arr))
                    self.kernel = shared_kernel
            self._kernel_slice = slice(0, self.filters)
        else:
            kernel_shape = tuple(self.kernel_size) + (self.filters, input_dim)
            self.kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )

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
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=initializers.Constant(raw_eps),
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
            data_format=self.data_format,
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
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        channel_axis = 1 if self.data_format == "channels_first" else -1
        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=channel_axis)

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            # Sum over all axes except the filter axis.
            # Transpose conv kernel shape: (*kernel_size, filters, in_dim)
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel_sq_sum_per_filter = ops.sum(kernel ** 2, axis=reduce_axes)

        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, -1))

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            length = input_shape[2]
        else:
            length = input_shape[1]
        
        if length is not None:
            if self.padding == "same":
                length = length * self.strides[0]
            else:
                length = length * self.strides[0] + max(self.kernel_size[0] - self.strides[0], 0)
        
        if self.data_format == "channels_first":
            return (input_shape[0], self.filters, length)
        else:
            return (input_shape[0], length, self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
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
class YatConvTranspose2D(Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = {}
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
        self.use_alpha = use_alpha
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)

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

        # Kernel shape for transpose conv: (*kernel_size, filters, input_dim)
        kernel_shape = tuple(self.kernel_size) + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

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
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=initializers.Constant(raw_eps),
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
            data_format=self.data_format,
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
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        channel_axis = 1 if self.data_format == "channels_first" else -1
        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=channel_axis)

        # Compute kernel squared sum per filter (1.0 if normalized)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            # Sum over all axes except the filter axis.
            # Transpose conv kernel shape: (*kernel_size, filters, in_dim)
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel_sq_sum_per_filter = ops.sum(kernel ** 2, axis=reduce_axes)

        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, 1, -1))

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]

        if rows is not None:
            if self.padding == "same":
                rows = rows * self.strides[0]
            else:
                rows = rows * self.strides[0] + max(self.kernel_size[0] - self.strides[0], 0)
        
        if cols is not None:
            if self.padding == "same":
                cols = cols * self.strides[1]
            else:
                cols = cols * self.strides[1] + max(self.kernel_size[1] - self.strides[1], 0)

        if self.data_format == "channels_first":
            return (input_shape[0], self.filters, rows, cols)
        else:
            return (input_shape[0], rows, cols, self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
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
class YatConvTranspose3D(Layer):
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = {}
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
        self.use_alpha = use_alpha
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.weight_normalized = weight_normalized
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)

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

        # Kernel shape for transpose conv: (*kernel_size, filters, input_dim)
        kernel_shape = tuple(self.kernel_size) + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

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
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=initializers.Constant(raw_eps),
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
            data_format=self.data_format,
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
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        channel_axis = 1 if self.data_format == "channels_first" else -1
        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=channel_axis)

        # Compute kernel squared sum per filter (1.0 if normalized).
        # Transpose conv kernel shape: (*kernel_size, filters, in_dim)
        if self.weight_normalized:
            kernel_sq_sum_per_filter = ops.ones((self.filters,), dtype=kernel.dtype)
        else:
            filter_axis = len(self.kernel_size)
            reduce_axes = tuple(i for i in range(kernel.ndim) if i != filter_axis)
            kernel_sq_sum_per_filter = ops.sum(kernel ** 2, axis=reduce_axes)

        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1, 1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, 1, 1, -1))

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            dims = [input_shape[2], input_shape[3], input_shape[4]]
        else:
            dims = [input_shape[1], input_shape[2], input_shape[3]]

        new_dims = []
        for i, dim in enumerate(dims):
            if dim is not None:
                if self.padding == "same":
                    dim = dim * self.strides[i]
                else:
                    dim = dim * self.strides[i] + max(self.kernel_size[i] - self.strides[i], 0)
            new_dims.append(dim)

        if self.data_format == "channels_first":
            return (input_shape[0], self.filters) + tuple(new_dims)
        else:
            return (input_shape[0],) + tuple(new_dims) + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
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