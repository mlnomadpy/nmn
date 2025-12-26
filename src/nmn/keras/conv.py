"""YAT convolution layers for Keras/TensorFlow."""

from keras.src import activations, constraints, initializers, regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src import ops
import math


@keras_export("keras.layers.YatConv1D")
class YatConv1D(Layer):
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
        use_alpha=True,
        epsilon=1e-5,
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
        self.use_bias = use_bias
        self.use_alpha = use_alpha
        self.epsilon = epsilon

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

        kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, self.filters)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
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

        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Compute standard convolution (dot product)
        dot_prod_map = ops.conv(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs
        
        # Create ones kernel for computing patch squared sums
        input_channels_per_group = self.kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels_per_group, 1)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=self.kernel.dtype)
        
        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        if self.groups > 1:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters // self.groups, axis=-1)
        else:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter
        kernel_sq_sum_per_filter = ops.sum(self.kernel**2, axis=tuple(range(self.kernel.ndim - 1)))

        # Reshape for broadcasting
        kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, -1))

        # Compute YAT: squared distance
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map

        # YAT computation: (dot_product)^2 / (distance_squared + epsilon)
        outputs = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        # Apply alpha scaling
        if self.use_alpha and self.alpha is not None:
            scale = (ops.sqrt(ops.cast(self.filters, self.compute_dtype)) /
                    ops.log1p(ops.cast(self.filters, self.compute_dtype))) ** self.alpha
            outputs = outputs * scale

        return outputs

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
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
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
        use_alpha=True,
        epsilon=1e-5,
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
        self.use_bias = use_bias
        self.use_alpha = use_alpha
        self.epsilon = epsilon

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

        kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, self.filters)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
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

        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Compute standard convolution (dot product)
        dot_prod_map = ops.conv(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs
        
        # Create ones kernel for computing patch squared sums
        input_channels_per_group = self.kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels_per_group, 1)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=self.kernel.dtype)
        
        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        if self.groups > 1:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters // self.groups, axis=-1)
        else:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter
        kernel_sq_sum_per_filter = ops.sum(self.kernel**2, axis=tuple(range(self.kernel.ndim - 1)))

        # Reshape for broadcasting
        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, 1, -1))

        # Compute YAT: squared distance
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map

        # YAT computation: (dot_product)^2 / (distance_squared + epsilon)
        outputs = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        # Apply alpha scaling
        if self.use_alpha and self.alpha is not None:
            scale = (ops.sqrt(ops.cast(self.filters, self.compute_dtype)) /
                    ops.log1p(ops.cast(self.filters, self.compute_dtype))) ** self.alpha
            outputs = outputs * scale

        return outputs

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
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
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
        use_alpha=True,
        epsilon=1e-5,
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
        self.use_bias = use_bias
        self.use_alpha = use_alpha
        self.epsilon = epsilon

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

        kernel_shape = tuple(self.kernel_size) + (input_dim // self.groups, self.filters)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
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

        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Compute standard convolution (dot product)
        dot_prod_map = ops.conv(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input patches using convolution with ones
        inputs_squared = inputs * inputs
        
        # Create ones kernel for computing patch squared sums
        input_channels_per_group = self.kernel.shape[-2]
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels_per_group, 1)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=self.kernel.dtype)
        
        patch_sq_sum_map_raw = ops.conv(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Handle grouped convolution
        if self.groups > 1:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters // self.groups, axis=-1)
        else:
            patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter
        kernel_sq_sum_per_filter = ops.sum(self.kernel**2, axis=tuple(range(self.kernel.ndim - 1)))

        # Reshape for broadcasting
        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1, 1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, 1, 1, -1))

        # Compute YAT: squared distance
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map

        # YAT computation: (dot_product)^2 / (distance_squared + epsilon)
        outputs = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        # Apply alpha scaling
        if self.use_alpha and self.alpha is not None:
            scale = (ops.sqrt(ops.cast(self.filters, self.compute_dtype)) /
                    ops.log1p(ops.cast(self.filters, self.compute_dtype))) ** self.alpha
            outputs = outputs * scale

        return outputs

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
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
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
        use_alpha=True,
        epsilon=1e-5,
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
        self.use_bias = use_bias
        self.use_alpha = use_alpha
        self.epsilon = epsilon

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
        
        # Kernel shape for transpose conv: [kernel_size, filters, input_dim]
        kernel_shape = tuple(self.kernel_size) + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
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

        self.input_dim = input_dim
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Compute transposed convolution (dot product)
        dot_prod_map = ops.conv_transpose(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input for YAT distance
        inputs_squared = inputs * inputs
        
        # Create ones kernel for computing patch squared sums
        ones_kernel_shape = tuple(self.kernel_size) + (1, self.input_dim)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=self.kernel.dtype)
        
        patch_sq_sum_map_raw = ops.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter
        kernel_sq_sum_per_filter = ops.sum(self.kernel**2, axis=(0, 2))  # Sum over spatial and input channel dims

        kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, -1))

        # Compute YAT: squared distance
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map

        # YAT computation: (dot_product)^2 / (distance_squared + epsilon)
        outputs = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        # Apply alpha scaling
        if self.use_alpha and self.alpha is not None:
            scale = (ops.sqrt(ops.cast(self.filters, self.compute_dtype)) /
                    ops.log1p(ops.cast(self.filters, self.compute_dtype))) ** self.alpha
            outputs = outputs * scale

        return outputs

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
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
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
        use_alpha=True,
        epsilon=1e-5,
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
        self.use_bias = use_bias
        self.use_alpha = use_alpha
        self.epsilon = epsilon

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
        
        # Kernel shape for transpose conv: [h, w, filters, input_dim]
        kernel_shape = tuple(self.kernel_size) + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
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

        self.input_dim = input_dim
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Compute transposed convolution (dot product)
        dot_prod_map = ops.conv_transpose(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        # Compute squared input for YAT distance
        inputs_squared = inputs * inputs
        
        # Create ones kernel for computing patch squared sums
        ones_kernel_shape = tuple(self.kernel_size) + (1, self.input_dim)
        ones_kernel = ops.ones(ones_kernel_shape, dtype=self.kernel.dtype)
        
        patch_sq_sum_map_raw = ops.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        patch_sq_sum_map = ops.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum per filter
        kernel_sq_sum_per_filter = ops.sum(self.kernel**2, axis=(0, 1, 3))  # Sum over spatial and input channel dims

        if self.data_format == "channels_first":
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, -1, 1, 1))
        else:
            kernel_sq_sum_reshaped = ops.reshape(kernel_sq_sum_per_filter, (1, 1, 1, -1))

        # Compute YAT: squared distance
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map

        # YAT computation: (dot_product)^2 / (distance_squared + epsilon)
        outputs = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        # Apply alpha scaling
        if self.use_alpha and self.alpha is not None:
            scale = (ops.sqrt(ops.cast(self.filters, self.compute_dtype)) /
                    ops.log1p(ops.cast(self.filters, self.compute_dtype))) ** self.alpha
            outputs = outputs * scale

        return outputs

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
            "use_alpha": self.use_alpha,
            "epsilon": self.epsilon,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


# Aliases for backward compatibility
YatConv1d = YatConv1D
YatConv2d = YatConv2D
YatConv3d = YatConv3D
YatConvTranspose1d = YatConvTranspose1D
YatConvTranspose2d = YatConvTranspose2D