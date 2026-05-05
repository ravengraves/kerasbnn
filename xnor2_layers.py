from __future__ import annotations

import math
import numpy as np
import keras
from keras import ops

from kerasbnn.binary_layers import (
    BinaryAffineNorm,
    BinaryDropcut,
    QuantizedActivation,
    QuantizedConv2D,
    quantize_signed_unit_interval_ste,
    quantize_unit_interval_ste,
)
from kerasbnn.runtime_quant import (
    exact_xnor_alpha_bias_ops as quantized_xnor_alpha_bias_ops,
    fold_xnor_activation_compare_ops,
    quantize_to_fixed_grid_ops,
)


def _normalize_tuple(value, n, name):
    if isinstance(value, int):
        return (value,) * n
    if isinstance(value, (tuple, list)) and len(value) == n:
        return tuple(int(v) for v in value)
    raise ValueError(f"`{name}` must be an int or tuple/list of length {n}. Got: {value!r}")


def _normalize_padding(padding):
    padding = str(padding).lower()
    if padding not in {"valid", "same"}:
        raise ValueError(f"`padding` must be 'valid' or 'same'. Got: {padding!r}")
    return padding


def _normalize_data_format(data_format):
    if data_format is None:
        data_format = keras.config.image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_last", "channels_first"}:
        raise ValueError(
            "`data_format` must be 'channels_last' or 'channels_first'. "
            f"Got: {data_format!r}"
        )
    return data_format


def _normalize_mode(value, name, valid_values):
    value = str(value).lower()
    if value not in valid_values:
        valid = "', '".join(valid_values)
        raise ValueError(f"`{name}` must be one of '{valid}'. Got: {value!r}")
    return value


def _cast_scalar(value, dtype):
    return ops.cast(float(value), dtype)


@keras.saving.register_keras_serializable(package="XNOR2")
class ClipRangeConstraint(keras.constraints.Constraint):
    def __init__(self, min_value=-1.0, max_value=1.0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, w):
        return ops.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


@ops.custom_gradient
def binary_sign_clipped_ste(x):
    one = ops.cast(1.0, x.dtype)
    minus_one = ops.cast(-1.0, x.dtype)
    y = ops.where(x >= 0, one, minus_one)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        mask = ops.cast(ops.less_equal(ops.absolute(x), one), upstream.dtype)
        return upstream * mask

    return y, grad


@ops.custom_gradient
def binary_sign_clipped_gain_ste(x, gain):
    one = ops.cast(1.0, x.dtype)
    minus_one = ops.cast(-1.0, x.dtype)
    y = ops.where(x >= 0, one, minus_one)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        mask = ops.cast(ops.less_equal(ops.absolute(x), one), upstream.dtype)
        gain_term = ops.cast(ops.stop_gradient(gain), upstream.dtype)
        return upstream * mask * gain_term, ops.zeros_like(gain)

    return y, grad


@ops.custom_gradient
def binary_sign_triangular_ste(x):
    one = ops.cast(1.0, x.dtype)
    zero = ops.cast(0.0, x.dtype)
    minus_one = ops.cast(-1.0, x.dtype)
    y = ops.where(x >= 0, one, minus_one)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        slope = ops.maximum(zero, ops.cast(2.0, x.dtype) - ops.cast(2.0, x.dtype) * ops.absolute(x))
        return upstream * ops.cast(slope, upstream.dtype)

    return y, grad


@ops.custom_gradient
def binary_sign_triangular_gain_ste(x, gain):
    one = ops.cast(1.0, x.dtype)
    zero = ops.cast(0.0, x.dtype)
    minus_one = ops.cast(-1.0, x.dtype)
    y = ops.where(x >= 0, one, minus_one)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        slope = ops.maximum(zero, ops.cast(2.0, x.dtype) - ops.cast(2.0, x.dtype) * ops.absolute(x))
        gain_term = ops.cast(ops.stop_gradient(gain), upstream.dtype)
        return upstream * ops.cast(slope, upstream.dtype) * gain_term, ops.zeros_like(gain)

    return y, grad


@ops.custom_gradient
def binary_weight_sign_magnitude_aware_ste(x, scale):
    one = ops.cast(1.0, x.dtype)
    minus_one = ops.cast(-1.0, x.dtype)
    y = ops.where(x >= 0, one, minus_one)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        mask = ops.cast(ops.less_equal(ops.absolute(x), one), upstream.dtype)
        scale_term = ops.cast(ops.stop_gradient(scale), upstream.dtype)
        return upstream * mask * scale_term, ops.zeros_like(scale)

    return y, grad


@ops.custom_gradient
def binary_weight_sign_scaled_ste(x, scale, floor, gain):
    one = ops.cast(1.0, x.dtype)
    minus_one = ops.cast(-1.0, x.dtype)
    y = ops.where(x >= 0, one, minus_one)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        mask = ops.cast(ops.less_equal(ops.absolute(x), one), upstream.dtype)
        scale_term = ops.maximum(ops.cast(ops.stop_gradient(scale), upstream.dtype), ops.cast(ops.stop_gradient(floor), upstream.dtype))
        gain_term = ops.cast(ops.stop_gradient(gain), upstream.dtype)
        return upstream * mask * scale_term * gain_term, ops.zeros_like(scale), ops.zeros_like(floor), ops.zeros_like(gain)

    return y, grad


def apply_binary_activation_surrogate(x, mode: str = "triangular", gain: float = 1.0):
    mode = str(mode).lower()
    gain = _cast_scalar(gain, x.dtype)
    if mode == "triangular":
        return binary_sign_triangular_gain_ste(x, gain)
    if mode == "clipped":
        return binary_sign_clipped_gain_ste(x, gain)
    raise ValueError(f"Unsupported activation surrogate mode: {mode!r}")


def apply_binary_weight_surrogate(
    x,
    scale,
    *,
    mode: str = "magnitude_aware",
    floor: float = 0.0,
    gain: float = 1.0,
):
    mode = str(mode).lower()
    if mode == "magnitude_aware":
        return binary_weight_sign_scaled_ste(
            x,
            scale,
            _cast_scalar(floor, x.dtype),
            _cast_scalar(gain, x.dtype),
        )
    if mode == "clipped":
        return apply_binary_activation_surrogate(x, mode="clipped", gain=gain)
    raise ValueError(f"Unsupported weight surrogate mode: {mode!r}")


def _broadcast_channel_params(values, x, axis: int):
    axis = int(axis)
    if axis < 0:
        axis += len(x.shape)
    shape = [1] * len(x.shape)
    shape[axis] = int(values.shape[0])
    return ops.reshape(values, tuple(shape))


def _binary_sign_from_folded_compare(x, scale_i8, bias_i8, axis: int):
    x = quantize_to_fixed_grid_ops(ops.cast(x, "float32"))
    scale = _broadcast_channel_params(ops.cast(scale_i8, "float32"), x, axis)
    bias = _broadcast_channel_params(ops.cast(bias_i8, "float32"), x, axis)
    return ops.where(
        scale * x + bias >= _cast_scalar(0.0, x.dtype),
        _cast_scalar(1.0, x.dtype),
        _cast_scalar(-1.0, x.dtype),
    )


def _apply_xnor_affine_float(outputs, alpha, bias=None, data_format="channels_last"):
    outputs = ops.cast(outputs, "float32")
    alpha = ops.cast(alpha, outputs.dtype)
    if len(outputs.shape) == 4:
        channels = int(alpha.shape[0] if len(alpha.shape) == 1 else alpha.shape[-1])
        if data_format == "channels_last":
            alpha = ops.reshape(alpha, (1, 1, 1, channels))
        else:
            alpha = ops.reshape(alpha, (1, channels, 1, 1))
    outputs = outputs * alpha
    if bias is not None:
        bias = ops.cast(bias, outputs.dtype)
        if len(outputs.shape) == 4:
            channels = int(bias.shape[0] if len(bias.shape) == 1 else bias.shape[-1])
            if data_format == "channels_last":
                bias = ops.reshape(bias, (1, 1, 1, channels))
            else:
                bias = ops.reshape(bias, (1, channels, 1, 1))
        outputs = outputs + bias
    return quantize_to_fixed_grid_ops(outputs)


@keras.saving.register_keras_serializable(package="XNOR2")
class ThermometerBinaryInput(keras.layers.Layer):
    def __init__(
        self,
        levels=4,
        min_value=0.0,
        max_value=1.0,
        data_format="channels_last",
        clip_inputs=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.levels = int(levels)
        if self.levels <= 0:
            raise ValueError(f"`levels` must be >= 1. Got: {levels!r}")
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        if self.max_value <= self.min_value:
            raise ValueError("`max_value` must be greater than `min_value`.")
        self.data_format = _normalize_data_format(data_format)
        self.clip_inputs = bool(clip_inputs)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self._rank = len(input_shape)
        if self.data_format != "channels_last":
            raise ValueError("ThermometerBinaryInput currently supports channels_last only.")
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Input channels must be defined.")
        self._channels = int(channels)
        thresholds = ((np.arange(self.levels, dtype=np.float32) + 1.0) / (self.levels + 1.0)).reshape(
            (1,) * self._rank + (self.levels,)
        )
        self._thresholds = thresholds
        super().build(input_shape)

    def call(self, inputs):
        x = ops.cast(inputs, "float32")
        x = (x - _cast_scalar(self.min_value, x.dtype)) / _cast_scalar(self.max_value - self.min_value, x.dtype)
        if self.clip_inputs:
            x = ops.clip(x, 0.0, 1.0)
        thresholds = ops.cast(ops.convert_to_tensor(self._thresholds), x.dtype)
        bits = binary_sign_clipped_ste(ops.expand_dims(x, axis=-1) - thresholds + _cast_scalar(1e-6, x.dtype))
        return ops.concatenate([bits[..., idx, :] for idx in range(self._channels)], axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if input_shape[-1] is not None:
            input_shape[-1] = int(input_shape[-1]) * self.levels
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "levels": self.levels,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "data_format": self.data_format,
                "clip_inputs": self.clip_inputs,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="XNOR2")
class QuantizedInput(keras.layers.Layer):
    def __init__(
        self,
        levels=16,
        min_value=0.0,
        max_value=1.0,
        signed=True,
        data_format="channels_last",
        clip_inputs=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.levels = int(levels)
        if self.levels <= 1:
            raise ValueError(f"`levels` must be > 1. Got: {levels!r}")
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        if self.max_value <= self.min_value:
            raise ValueError("`max_value` must be greater than `min_value`.")
        self.signed = bool(signed)
        self.data_format = _normalize_data_format(data_format)
        self.clip_inputs = bool(clip_inputs)

    def call(self, inputs):
        x = ops.cast(inputs, "float32")
        x = (x - _cast_scalar(self.min_value, x.dtype)) / _cast_scalar(self.max_value - self.min_value, x.dtype)
        if self.clip_inputs:
            x = ops.clip(x, 0.0, 1.0)
        if self.signed:
            x = x * _cast_scalar(2.0, x.dtype) - _cast_scalar(1.0, x.dtype)
            return quantize_signed_unit_interval_ste(x, levels=self.levels)
        return quantize_unit_interval_ste(x, levels=self.levels)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "levels": self.levels,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "signed": self.signed,
                "data_format": self.data_format,
                "clip_inputs": self.clip_inputs,
            }
        )
        return config


class _XNORPreNormMixin:
    def _build_norm(self, input_shape):
        if not self.normalize:
            self.affine_norm = None
            self.gamma = None
            self.beta = None
            self.moving_mean = None
            self.moving_variance = None
            return
        input_shape = tuple(input_shape)
        axis = len(input_shape) - 1 if self.data_format == "channels_last" else 1
        channels = input_shape[axis]
        if channels is None:
            raise ValueError("Input channels must be defined for normalization.")
        self._norm_axis = axis
        self._norm_channels = int(channels)
        self._reduce_axes = tuple(i for i in range(len(input_shape)) if i != axis)
        self._broadcast_shape = [1] * len(input_shape)
        self._broadcast_shape[axis] = self._norm_channels
        if self.binary_batchnorm:
            self.affine_norm = BinaryAffineNorm(
                levels=self.norm_levels,
                threshold_limit=self.norm_threshold_limit,
                use_bias=True,
                output_mode="raw",
                feature_axis=axis,
                data_format=self.data_format,
                momentum=self.momentum,
                epsilon=self.epsilon,
                use_running_stats=True,
                quantize_input=self.bn_quantize_input,
                quantize_centered=self.bn_quantize_centered,
                quantize_normalized=self.bn_quantize_normalized,
                quantize_output=self.bn_quantize_output,
                quantize_eval_stats=self.bn_quantize_eval_stats,
                eval_use_batch_stats=self.bn_eval_use_batch_stats,
                force_full_quant_eval=self.bn_force_full_quant_eval,
                quantize_affine_params=self.bn_quantize_affine_params,
                name="affine_norm",
            )
            self.affine_norm.build(input_shape)
            self.affine_norm.norm_scale.assign(
                np.full((self._norm_channels,), self.norm_gamma_init, dtype=np.float32)
            )
            self.gamma = None
            self.beta = None
            self.moving_mean = None
            self.moving_variance = None
            return
        self.affine_norm = None
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self._norm_channels,),
            initializer=keras.initializers.Constant(self.norm_gamma_init),
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self._norm_channels,),
            initializer="zeros",
            trainable=True,
        )
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(self._norm_channels,),
            initializer="zeros",
            trainable=False,
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(self._norm_channels,),
            initializer="ones",
            trainable=False,
        )

    def _apply_norm(self, x, training=None):
        if not self.normalize:
            return x
        if self.affine_norm is not None:
            return self.affine_norm(x, training=training)
        eps = _cast_scalar(self.epsilon, x.dtype)
        if training is False:
            mean = ops.cast(self.moving_mean, x.dtype)
            variance = ops.cast(self.moving_variance, x.dtype)
        else:
            mean = ops.mean(x, axis=self._reduce_axes)
            centered = x - ops.reshape(mean, self._broadcast_shape)
            variance = ops.mean(ops.square(centered), axis=self._reduce_axes)
            if training is not False:
                momentum = ops.cast(self.momentum, self.moving_mean.dtype)
                one_minus = ops.cast(1.0 - self.momentum, self.moving_mean.dtype)
                self.moving_mean.assign(momentum * self.moving_mean + one_minus * ops.cast(mean, self.moving_mean.dtype))
                self.moving_variance.assign(
                    momentum * self.moving_variance + one_minus * ops.cast(variance, self.moving_variance.dtype)
                )
        mean = ops.reshape(ops.cast(mean, x.dtype), self._broadcast_shape)
        variance = ops.reshape(ops.cast(variance, x.dtype), self._broadcast_shape)
        gamma = ops.reshape(ops.cast(self.gamma, x.dtype), self._broadcast_shape)
        beta = ops.reshape(ops.cast(self.beta, x.dtype), self._broadcast_shape)
        x = (x - mean) / ops.sqrt(variance + eps)
        return gamma * x + beta

    def _build_rprelu(self, input_shape):
        if not self.use_rprelu:
            self.rprelu_gamma = None
            self.rprelu_beta = None
            self.rprelu_zeta = None
            return
        input_shape = tuple(input_shape)
        axis = len(input_shape) - 1 if self.data_format == "channels_last" else 1
        channels = input_shape[axis]
        if channels is None:
            raise ValueError("Input channels must be defined for RPReLU.")
        self._rprelu_axis = axis
        self._rprelu_channels = int(channels)
        self._rprelu_broadcast_shape = [1] * len(input_shape)
        self._rprelu_broadcast_shape[axis] = self._rprelu_channels
        self.rprelu_gamma = self.add_weight(
            name="rprelu_gamma",
            shape=(self._rprelu_channels,),
            initializer=keras.initializers.Constant(self.rprelu_init_gamma),
            trainable=True,
        )
        self.rprelu_beta = self.add_weight(
            name="rprelu_beta",
            shape=(self._rprelu_channels,),
            initializer=keras.initializers.Constant(self.rprelu_init_beta),
            trainable=True,
            constraint=keras.constraints.NonNeg(),
        )
        self.rprelu_zeta = self.add_weight(
            name="rprelu_zeta",
            shape=(self._rprelu_channels,),
            initializer=keras.initializers.Constant(self.rprelu_init_zeta),
            trainable=True,
        )

    def _apply_rprelu(self, x):
        if not self.use_rprelu:
            return x
        gamma = ops.reshape(ops.cast(self.rprelu_gamma, x.dtype), self._rprelu_broadcast_shape)
        beta = ops.reshape(ops.cast(self.rprelu_beta, x.dtype), self._rprelu_broadcast_shape)
        zeta = ops.reshape(ops.cast(self.rprelu_zeta, x.dtype), self._rprelu_broadcast_shape)
        centered = x - gamma
        return ops.where(centered > 0, centered + zeta, beta * centered + zeta)


@keras.saving.register_keras_serializable(package="XNOR2")
class XNORBinaryConv2D(keras.layers.Layer, _XNORPreNormMixin):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        use_bias=True,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=False,
        norm_levels=128,
        norm_threshold_limit=1.0,
        momentum=0.95,
        epsilon=1e-4,
        kernel_initializer="glorot_uniform",
        activation_surrogate="triangular",
        activation_surrogate_gain=1.0,
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.0,
        weight_surrogate_gain=1.0,
        norm_gamma_init=0.25,
        bn_quantize_input=False,
        bn_quantize_centered=False,
        bn_quantize_normalized=False,
        bn_quantize_output=False,
        bn_quantize_eval_stats=True,
        bn_eval_use_batch_stats=False,
        bn_force_full_quant_eval=False,
        bn_quantize_affine_params=True,
        use_rprelu=False,
        rprelu_init_beta=0.25,
        rprelu_init_gamma=0.0,
        rprelu_init_zeta=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = _normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = _normalize_tuple(strides, 2, "strides")
        self.padding = _normalize_padding(padding)
        self.data_format = _normalize_data_format(data_format)
        self.dilation_rate = _normalize_tuple(dilation_rate, 2, "dilation_rate")
        self.use_bias = bool(use_bias)
        self.output_mode = _normalize_mode(output_mode, "output_mode", {"raw", "bipolar"})
        self.normalize = bool(normalize)
        self.binary_batchnorm = bool(binary_batchnorm)
        self.norm_levels = int(norm_levels)
        self.norm_threshold_limit = float(norm_threshold_limit)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.activation_surrogate = str(activation_surrogate).lower()
        self.activation_surrogate_gain = float(activation_surrogate_gain)
        self.weight_surrogate = str(weight_surrogate).lower()
        self.weight_surrogate_floor = float(weight_surrogate_floor)
        self.weight_surrogate_gain = float(weight_surrogate_gain)
        self.norm_gamma_init = float(norm_gamma_init)
        self.bn_quantize_input = bool(bn_quantize_input)
        self.bn_quantize_centered = bool(bn_quantize_centered)
        self.bn_quantize_normalized = bool(bn_quantize_normalized)
        self.bn_quantize_output = bool(bn_quantize_output)
        self.bn_quantize_eval_stats = bool(bn_quantize_eval_stats)
        self.bn_eval_use_batch_stats = bool(bn_eval_use_batch_stats)
        self.bn_force_full_quant_eval = bool(bn_force_full_quant_eval)
        self.bn_quantize_affine_params = bool(bn_quantize_affine_params)
        self.use_rprelu = bool(use_rprelu)
        self.rprelu_init_beta = float(rprelu_init_beta)
        self.rprelu_init_gamma = float(rprelu_init_gamma)
        self.rprelu_init_zeta = float(rprelu_init_zeta)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_constraint = keras.constraints.Constraint()

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        if self.data_format == "channels_last":
            in_channels = input_shape[-1]
        else:
            in_channels = input_shape[1]
        if in_channels is None:
            raise ValueError("The channel dimension of the inputs must be defined.")
        self._input_channels = int(in_channels)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(*self.kernel_size, self._input_channels, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
            constraint=ClipRangeConstraint(-1.0, 1.0),
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        self._build_norm(input_shape)
        self._build_rprelu(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        hist = getattr(inputs, "_keras_history", None)
        if hist is not None and not hasattr(self, "_input_is_binary_source"):
            src_layer = hist[0]
            self._input_is_binary_source = (src_layer.__class__.__name__ == "ThermometerBinaryInput")
        if training is False:
            x = ops.cast(inputs, "float32")
            feature_axis = -1 if self.data_format == "channels_last" else 1
            if getattr(self, "_input_is_binary_source", False):
                x_b = ops.where(
                    x >= _cast_scalar(0.0, x.dtype),
                    _cast_scalar(1.0, x.dtype),
                    _cast_scalar(-1.0, x.dtype),
                )
            else:
                act_scale_i8, act_bias_i8 = fold_xnor_activation_compare_ops(self, dtype="float32")
                x_b = _binary_sign_from_folded_compare(x, act_scale_i8, act_bias_i8, axis=feature_axis)
            kernel_eval = ops.cast(self.kernel, "float32")
            alpha_eval, bias_eval = quantized_xnor_alpha_bias_ops(self, kernel_eval, reduce_axes=(0, 1, 2))
            w_b = ops.where(
                kernel_eval >= _cast_scalar(0.0, kernel_eval.dtype),
                _cast_scalar(1.0, kernel_eval.dtype),
                _cast_scalar(-1.0, kernel_eval.dtype),
            )
        else:
            x = ops.cast(inputs, self.compute_dtype)
            kernel = ops.cast(self.kernel, x.dtype)
            x = self._apply_norm(x, training=training)
            x = self._apply_rprelu(x)
            x_b = apply_binary_activation_surrogate(
                x,
                mode=self.activation_surrogate,
                gain=self.activation_surrogate_gain,
            )
            alpha = ops.mean(ops.absolute(kernel), axis=(0, 1, 2), keepdims=True)
            bias = ops.cast(self.bias, x_b.dtype) if self.bias is not None else None
            w_b = apply_binary_weight_surrogate(
                kernel,
                alpha,
                mode=self.weight_surrogate,
                floor=self.weight_surrogate_floor,
                gain=self.weight_surrogate_gain,
            )
        if training is False:
            x_b = ops.cast(x_b, "float32")
            w_b = ops.cast(w_b, "float32")
        else:
            kernel = ops.cast(kernel, x_b.dtype)
            w_b = ops.cast(w_b, x_b.dtype)
        outputs = ops.conv(
            x_b,
            w_b,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if training is False:
            outputs = _apply_xnor_affine_float(outputs, alpha_eval, bias_eval, data_format=self.data_format)
        else:
            alpha = ops.cast(alpha, x_b.dtype)
            outputs = outputs * ops.cast(alpha, outputs.dtype)
            if bias is not None:
                bias = ops.cast(bias, outputs.dtype)
                if self.data_format == "channels_last":
                    bias = ops.reshape(bias, (1, 1, 1, self.filters))
                else:
                    bias = ops.reshape(bias, (1, self.filters, 1, 1))
                outputs = outputs + bias
        if self.output_mode == "bipolar":
            if training is False:
                outputs = ops.where(
                    outputs >= _cast_scalar(0.0, outputs.dtype),
                    _cast_scalar(1.0, outputs.dtype),
                    _cast_scalar(-1.0, outputs.dtype),
                )
            else:
                outputs = apply_binary_activation_surrogate(
                    outputs,
                    mode=self.activation_surrogate,
                    gain=self.activation_surrogate_gain,
                )
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if self.data_format == "channels_last":
            h, w = input_shape[1], input_shape[2]
        else:
            h, w = input_shape[2], input_shape[3]

        def out_dim(size, kernel, stride, dilation, padding):
            if size is None:
                return None
            if padding == "same":
                return math.ceil(size / stride)
            effective_kernel = dilation * (kernel - 1) + 1
            return math.floor((size - effective_kernel) / stride) + 1

        oh = out_dim(h, self.kernel_size[0], self.strides[0], self.dilation_rate[0], self.padding)
        ow = out_dim(w, self.kernel_size[1], self.strides[1], self.dilation_rate[1], self.padding)
        if self.data_format == "channels_last":
            return (input_shape[0], oh, ow, self.filters)
        return (input_shape[0], self.filters, oh, ow)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "output_mode": self.output_mode,
                "normalize": self.normalize,
                "binary_batchnorm": self.binary_batchnorm,
                "norm_levels": self.norm_levels,
                "norm_threshold_limit": self.norm_threshold_limit,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "activation_surrogate": self.activation_surrogate,
                "activation_surrogate_gain": self.activation_surrogate_gain,
                "weight_surrogate": self.weight_surrogate,
                "weight_surrogate_floor": self.weight_surrogate_floor,
                "weight_surrogate_gain": self.weight_surrogate_gain,
                "norm_gamma_init": self.norm_gamma_init,
                "bn_quantize_input": self.bn_quantize_input,
                "bn_quantize_centered": self.bn_quantize_centered,
                "bn_quantize_normalized": self.bn_quantize_normalized,
                "bn_quantize_output": self.bn_quantize_output,
                "bn_quantize_eval_stats": self.bn_quantize_eval_stats,
                "bn_eval_use_batch_stats": self.bn_eval_use_batch_stats,
                "bn_force_full_quant_eval": self.bn_force_full_quant_eval,
                "bn_quantize_affine_params": self.bn_quantize_affine_params,
                "use_rprelu": self.use_rprelu,
                "rprelu_init_beta": self.rprelu_init_beta,
                "rprelu_init_gamma": self.rprelu_init_gamma,
                "rprelu_init_zeta": self.rprelu_init_zeta,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="XNOR2")
class XNORBinaryDense(keras.layers.Layer, _XNORPreNormMixin):
    def __init__(
        self,
        units,
        use_bias=True,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=False,
        norm_levels=128,
        norm_threshold_limit=1.0,
        momentum=0.95,
        epsilon=1e-4,
        kernel_initializer="glorot_uniform",
        activation_surrogate="triangular",
        activation_surrogate_gain=1.0,
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.0,
        weight_surrogate_gain=1.0,
        norm_gamma_init=0.25,
        bn_quantize_input=False,
        bn_quantize_centered=False,
        bn_quantize_normalized=False,
        bn_quantize_output=False,
        bn_quantize_eval_stats=True,
        bn_eval_use_batch_stats=False,
        bn_force_full_quant_eval=False,
        bn_quantize_affine_params=True,
        use_rprelu=False,
        rprelu_init_beta=0.25,
        rprelu_init_gamma=0.0,
        rprelu_init_zeta=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.use_bias = bool(use_bias)
        self.output_mode = _normalize_mode(output_mode, "output_mode", {"raw", "bipolar"})
        self.normalize = bool(normalize)
        self.binary_batchnorm = bool(binary_batchnorm)
        self.norm_levels = int(norm_levels)
        self.norm_threshold_limit = float(norm_threshold_limit)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.activation_surrogate = str(activation_surrogate).lower()
        self.activation_surrogate_gain = float(activation_surrogate_gain)
        self.weight_surrogate = str(weight_surrogate).lower()
        self.weight_surrogate_floor = float(weight_surrogate_floor)
        self.weight_surrogate_gain = float(weight_surrogate_gain)
        self.norm_gamma_init = float(norm_gamma_init)
        self.bn_quantize_input = bool(bn_quantize_input)
        self.bn_quantize_centered = bool(bn_quantize_centered)
        self.bn_quantize_normalized = bool(bn_quantize_normalized)
        self.bn_quantize_output = bool(bn_quantize_output)
        self.bn_quantize_eval_stats = bool(bn_quantize_eval_stats)
        self.bn_eval_use_batch_stats = bool(bn_eval_use_batch_stats)
        self.bn_force_full_quant_eval = bool(bn_force_full_quant_eval)
        self.bn_quantize_affine_params = bool(bn_quantize_affine_params)
        self.use_rprelu = bool(use_rprelu)
        self.rprelu_init_beta = float(rprelu_init_beta)
        self.rprelu_init_gamma = float(rprelu_init_gamma)
        self.rprelu_init_zeta = float(rprelu_init_zeta)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.data_format = "channels_last"

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        in_dim = input_shape[-1]
        if in_dim is None:
            raise ValueError("The last input dimension must be defined.")
        self._input_dim = int(in_dim)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self._input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            constraint=ClipRangeConstraint(-1.0, 1.0),
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        self._build_norm(input_shape)
        self._build_rprelu(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        hist = getattr(inputs, "_keras_history", None)
        if hist is not None and not hasattr(self, "_input_is_binary_source"):
            src_layer = hist[0]
            self._input_is_binary_source = (src_layer.__class__.__name__ == "ThermometerBinaryInput")
        if training is False:
            x = ops.cast(inputs, "float32")
            if getattr(self, "_input_is_binary_source", False):
                x_b = ops.where(
                    x >= _cast_scalar(0.0, x.dtype),
                    _cast_scalar(1.0, x.dtype),
                    _cast_scalar(-1.0, x.dtype),
                )
            else:
                act_scale_i8, act_bias_i8 = fold_xnor_activation_compare_ops(self, dtype="float32")
                x_b = _binary_sign_from_folded_compare(x, act_scale_i8, act_bias_i8, axis=-1)
            kernel_eval = ops.cast(self.kernel, "float32")
            alpha_eval, bias_eval = quantized_xnor_alpha_bias_ops(self, kernel_eval, reduce_axes=0)
            w_b = ops.where(
                kernel_eval >= _cast_scalar(0.0, kernel_eval.dtype),
                _cast_scalar(1.0, kernel_eval.dtype),
                _cast_scalar(-1.0, kernel_eval.dtype),
            )
        else:
            x = ops.cast(inputs, self.compute_dtype)
            x = self._apply_norm(x, training=training)
            x = self._apply_rprelu(x)
            x_b = apply_binary_activation_surrogate(
                x,
                mode=self.activation_surrogate,
                gain=self.activation_surrogate_gain,
            )
            kernel = ops.cast(self.kernel, x_b.dtype)
            alpha = ops.mean(ops.absolute(kernel), axis=0, keepdims=True)
            bias = ops.cast(self.bias, x_b.dtype) if self.bias is not None else None
            w_b = apply_binary_weight_surrogate(
                kernel,
                alpha,
                mode=self.weight_surrogate,
                floor=self.weight_surrogate_floor,
                gain=self.weight_surrogate_gain,
            )
        if training is False:
            x_b = ops.cast(x_b, "float32")
            w_b = ops.cast(w_b, "float32")
            outputs = ops.matmul(x_b, w_b)
            outputs = _apply_xnor_affine_float(outputs, alpha_eval, bias_eval, data_format=self.data_format)
        else:
            w_b = ops.cast(w_b, x_b.dtype)
            alpha = ops.cast(alpha, x_b.dtype)
            outputs = ops.matmul(x_b, w_b * ops.cast(alpha, w_b.dtype))
            if bias is not None:
                outputs = outputs + ops.cast(bias, outputs.dtype)
        if self.output_mode == "bipolar":
            if training is False:
                outputs = ops.where(
                    outputs >= _cast_scalar(0.0, outputs.dtype),
                    _cast_scalar(1.0, outputs.dtype),
                    _cast_scalar(-1.0, outputs.dtype),
                )
            else:
                outputs = apply_binary_activation_surrogate(
                    outputs,
                    mode=self.activation_surrogate,
                    gain=self.activation_surrogate_gain,
                )
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] = self.units
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "output_mode": self.output_mode,
                "normalize": self.normalize,
                "binary_batchnorm": self.binary_batchnorm,
                "norm_levels": self.norm_levels,
                "norm_threshold_limit": self.norm_threshold_limit,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "activation_surrogate": self.activation_surrogate,
                "activation_surrogate_gain": self.activation_surrogate_gain,
                "weight_surrogate": self.weight_surrogate,
                "weight_surrogate_floor": self.weight_surrogate_floor,
                "weight_surrogate_gain": self.weight_surrogate_gain,
                "norm_gamma_init": self.norm_gamma_init,
                "bn_quantize_input": self.bn_quantize_input,
                "bn_quantize_centered": self.bn_quantize_centered,
                "bn_quantize_normalized": self.bn_quantize_normalized,
                "bn_quantize_output": self.bn_quantize_output,
                "bn_quantize_eval_stats": self.bn_quantize_eval_stats,
                "bn_eval_use_batch_stats": self.bn_eval_use_batch_stats,
                "bn_force_full_quant_eval": self.bn_force_full_quant_eval,
                "bn_quantize_affine_params": self.bn_quantize_affine_params,
                "use_rprelu": self.use_rprelu,
                "rprelu_init_beta": self.rprelu_init_beta,
                "rprelu_init_gamma": self.rprelu_init_gamma,
                "rprelu_init_zeta": self.rprelu_init_zeta,
            }
        )
        return config


def quantized_first_conv(
    x,
    filters: int,
    kernel_size: int,
    *,
    levels: int = 16,
    output_levels: int = 0,
    strides: int = 1,
    name: str,
):
    x = QuantizedConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        data_format="channels_last",
        levels=levels,
        binarize_inputs=False,
        integer_output=False,
        integer_kernel=True,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    if int(output_levels) > 1:
        return QuantizedActivation(levels=output_levels, signed=True, name=f"{name}_act")(x)
    return keras.layers.Activation("linear", dtype="float32", name=f"{name}_act")(x)
