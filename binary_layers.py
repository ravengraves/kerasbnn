import math
from numbers import Real
import numpy as np
import keras
from keras import ops

from kerasbnn.runtime_quant import (
    fake_quantize_signed_output_exact_ops,
    quantize_signed_int_grid_ops,
    quantize_signed_unit_grid_ops,
    quantize_signed_unit_interval_exact_ops,
    quantize_to_fixed_grid_ops,
    quantize_unit_interval_exact_ops,
)


@keras.saving.register_keras_serializable(package="BinaryLayers")
class ClipConstraint(keras.constraints.Constraint):
    def __init__(self, clip_value=1.0):
        self.clip_value = float(clip_value)

    def __call__(self, w):
        clip_value = ops.cast(self.clip_value, w.dtype)
        return ops.clip(w, -clip_value, clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}


@keras.saving.register_keras_serializable(package="BinaryLayers")
class ClipRangeConstraint(keras.constraints.Constraint):
    def __init__(self, min_value=-1.0, max_value=1.0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, w):
        min_value = ops.cast(self.min_value, w.dtype)
        max_value = ops.cast(self.max_value, w.dtype)
        return ops.clip(w, min_value, max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


def _normalize_score_mode(value):
    value = str(value).lower()
    if value not in {"scaled_sign", "xnor_popcount"}:
        raise ValueError(
            "`score_mode` must be 'scaled_sign' or 'xnor_popcount'. "
            f"Got: {value!r}"
        )
    return value


def _normalize_feature_axis(axis):
    if axis is None:
        return None
    return int(axis)


def _normalize_mode(value, name, valid_values):
    value = str(value).lower()
    if value not in valid_values:
        valid = "', '".join(valid_values)
        raise ValueError(f"`{name}` must be one of '{valid}'. Got: {value!r}")
    return value


def _normalize_tuple(value, n, name):
    if isinstance(value, int):
        return (value,) * n
    if isinstance(value, (tuple, list)) and len(value) == n:
        return tuple(int(v) for v in value)
    raise ValueError(f"`{name}` must be an int or a tuple/list of length {n}. Got: {value!r}")


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


def _conv_bias_shape(filters, data_format):
    if data_format == "channels_last":
        return (1, 1, 1, filters)
    return (1, filters, 1, 1)


def _resolve_feature_axis(feature_axis, rank, data_format):
    if feature_axis is None:
        if data_format == "channels_first" and rank > 2:
            return 1
        return rank - 1
    axis = int(feature_axis)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"feature_axis {feature_axis!r} is out of range for rank {rank}")
    return axis


def _cast_scalar(value, dtype):
    return ops.cast(float(value), dtype)


@ops.custom_gradient

def binary_sign_ste(x):
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
def binary_step_ste(x):
    zero = ops.cast(0.0, x.dtype)
    one = ops.cast(1.0, x.dtype)
    y = ops.where(x > zero, one, zero)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        mask = ops.cast(ops.less_equal(ops.absolute(x), one), upstream.dtype)
        return upstream * mask

    return y, grad


def binary_threshold_ste(x, threshold=0.5):
    threshold = ops.cast(threshold, x.dtype)
    return binary_step_ste(x - threshold)


def binary_sign_threshold_ste(x, threshold=0.0):
    threshold = ops.cast(threshold, x.dtype)
    return binary_sign_ste(x - threshold)


def quantize_unit_interval_ste(x, levels=128):
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    zero = ops.cast(0.0, x.dtype)
    levels_f = ops.cast(float(levels), x.dtype)
    max_unit = ops.cast((levels - 1) / levels, x.dtype)
    clipped = ops.clip(x, zero, max_unit)
    quantized = ops.floor(clipped * levels_f) / levels_f
    return clipped + ops.stop_gradient(quantized - clipped)


def quantize_signed_unit_interval_ste(x, levels=128):
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    levels_f = ops.cast(float(levels), x.dtype)
    min_unit = ops.cast(-1.0, x.dtype)
    max_unit = ops.cast((levels - 1) / levels, x.dtype)
    clipped = ops.clip(x, min_unit, max_unit)
    quantized = ops.floor(clipped * levels_f) / levels_f
    return clipped + ops.stop_gradient(quantized - clipped)


def quantize_signed_int_ste(x, levels=128):
    """Quantize `[-1, 1]` floats to a symmetric signed integer grid with STE.

    This matches the Keras symmetric int8 convention more closely than a full
    two's-complement `[-128, 127]` mapping: for `levels=128`, the effective
    signed range is `[-127, 127]`, i.e. `qmax = levels - 1`.

    Forward:
        q = clip(round(clip(x, -1, 1) * qmax), -qmax, qmax)
    Backward:
        straight-through on the smooth scaled value `clip(x, -1, 1) * qmax`

    This keeps trainable weights in `[-1, 1]` while making later fixed-point
    export straightforward: multiply by `qmax`, round, then store as signed
    integers.
    """
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    qmax = levels - 1
    qmax_f = ops.cast(float(qmax), x.dtype)
    min_unit = ops.cast(-1.0, x.dtype)
    max_unit = ops.cast(1.0, x.dtype)
    clipped = ops.clip(x, min_unit, max_unit)
    scaled = clipped * qmax_f
    quantized = ops.clip(ops.round(scaled), -qmax_f, qmax_f)
    return scaled + ops.stop_gradient(quantized - scaled)


def fake_quantize_signed_output_ste(x, levels=128, scale=1.0):
    """Quantize to a signed integer grid in forward, keep unit-scale STE backward.

    For `levels=128`, values are quantized onto `[-127, 127]` integer bins.
    The forward path returns multiples of `scale`, while the backward path acts
    like an identity through the clipped pre-rounded values.
    """
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    qmax = levels - 1
    qmax_f = ops.cast(float(qmax), x.dtype)
    scale_f = ops.cast(float(scale), x.dtype)
    scaled = x / scale_f
    clipped = ops.clip(scaled, -qmax_f, qmax_f)
    quantized = ops.clip(ops.round(clipped), -qmax_f, qmax_f)
    return (clipped + ops.stop_gradient(quantized - clipped)) * scale_f


def quantize_symmetric_range_ste(x, max_abs=1.0, levels=128):
    """Quantize floats to a symmetric finite range with STE.

    The forward path clips to `[-max_abs, max_abs]` and rounds onto a signed
    grid with `levels - 1` positive bins. The backward path uses the smooth
    clipped value, which makes it suitable for learned hardware thresholds.
    """
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    max_abs_t = ops.cast(float(max_abs), x.dtype)
    qmax_t = ops.cast(float(levels - 1), x.dtype)
    clipped = ops.clip(x, -max_abs_t, max_abs_t)
    step = ops.maximum(max_abs_t / qmax_t, ops.cast(1e-6, x.dtype))
    quantized = ops.round(clipped / step) * step
    quantized = ops.clip(quantized, -max_abs_t, max_abs_t)
    return clipped + ops.stop_gradient(quantized - clipped)


def quantize_positive_range_ste(x, max_value=1.0, levels=128):
    """Quantize nonnegative floats to a finite range with STE."""
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    max_value_t = ops.cast(float(max_value), x.dtype)
    qmax_t = ops.cast(float(levels - 1), x.dtype)
    zero = ops.cast(0.0, x.dtype)
    clipped = ops.clip(x, zero, max_value_t)
    step = ops.maximum(max_value_t / qmax_t, ops.cast(1e-6, x.dtype))
    quantized = ops.round(clipped / step) * step
    quantized = ops.clip(quantized, zero, max_value_t)
    return clipped + ops.stop_gradient(quantized - clipped)


def round_ste(x):
    rounded = ops.round(x)
    return x + ops.stop_gradient(rounded - x)


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryInput(keras.layers.Layer):
    """Thermometer-binarize float inputs into {-1, +1} feature channels."""

    def __init__(
        self,
        levels=1,
        min_value=0.0,
        max_value=1.0,
        feature_axis=None,
        data_format=None,
        clip_inputs=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.levels = int(levels)
        if self.levels <= 0:
            raise ValueError(f"`levels` must be >= 1. Got: {levels!r}")
        self.min_value = min_value
        self.max_value = max_value
        self.feature_axis = _normalize_feature_axis(feature_axis)
        self.data_format = _normalize_data_format(data_format)
        self.clip_inputs = bool(clip_inputs)

    @staticmethod
    def _range_array(value, feature_dim, name):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((feature_dim,), float(arr), dtype=np.float32)
        elif arr.size != feature_dim:
            raise ValueError(f"`{name}` must be scalar or have length {feature_dim}. Got shape {arr.shape!r}")
        return arr.reshape(feature_dim)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        rank = len(input_shape)
        axis = _resolve_feature_axis(self.feature_axis, rank, self.data_format)
        feature_dim = input_shape[axis]
        if feature_dim is None:
            raise ValueError("The input feature dimension must be defined.")
        self._rank = rank
        self._feature_axis = axis
        self._feature_dim = int(feature_dim)
        min_arr = self._range_array(self.min_value, self._feature_dim, "min_value")
        max_arr = self._range_array(self.max_value, self._feature_dim, "max_value")
        if np.any(max_arr <= min_arr):
            raise ValueError("Each `max_value` entry must be greater than `min_value`.")
        self._min_arr = min_arr
        self._max_arr = max_arr
        self._range_shape = (1,) * (rank - 1) + (self._feature_dim,)
        if axis != rank - 1:
            self._perm = tuple(i for i in range(rank) if i != axis) + (axis,)
            inv = [0] * rank
            for i, src in enumerate(self._perm):
                inv[src] = i
            self._inv_perm = tuple(inv)
        else:
            self._perm = None
            self._inv_perm = None
        self._thresholds = ((np.arange(self.levels, dtype=np.float32) + 1.0) / (self.levels + 1.0)).reshape((1,) * rank + (self.levels,))
        super().build(input_shape)

    def call(self, inputs):
        x = ops.cast(inputs, self.compute_dtype)
        if self._perm is not None:
            x = ops.transpose(x, self._perm)
        min_arr = ops.cast(ops.reshape(ops.convert_to_tensor(self._min_arr), self._range_shape), x.dtype)
        max_arr = ops.cast(ops.reshape(ops.convert_to_tensor(self._max_arr), self._range_shape), x.dtype)
        scale = ops.maximum(max_arr - min_arr, ops.cast(1e-6, x.dtype))
        x = (x - min_arr) / scale
        if self.clip_inputs:
            x = ops.clip(x, 0.0, 1.0)
        thresholds = ops.cast(ops.convert_to_tensor(self._thresholds), x.dtype)
        bits = binary_step_ste(ops.expand_dims(x, axis=-1) - thresholds + ops.cast(1e-6, x.dtype))
        bits = bits * ops.cast(2.0, bits.dtype) - ops.cast(1.0, bits.dtype)
        outputs = ops.concatenate([bits[..., idx, :] for idx in range(self._feature_dim)], axis=-1)
        if self._inv_perm is not None:
            outputs = ops.transpose(outputs, self._inv_perm)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        axis = self._feature_axis if hasattr(self, "_feature_axis") else _resolve_feature_axis(self.feature_axis, len(input_shape), self.data_format)
        if input_shape[axis] is not None:
            input_shape[axis] = int(input_shape[axis]) * self.levels
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "levels": self.levels,
                "min_value": np.asarray(self.min_value).tolist() if np.ndim(self.min_value) else float(self.min_value),
                "max_value": np.asarray(self.max_value).tolist() if np.ndim(self.max_value) else float(self.max_value),
                "feature_axis": self.feature_axis,
                "data_format": self.data_format,
                "clip_inputs": self.clip_inputs,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryAffineNorm(keras.layers.Layer):
    """Quantized affine normalization for binary-friendly thresholds.

    Forward:
        training:
            x_n = (x - mean_batch) / sqrt(var_batch + eps)
        inference/export:
            x_n = (x - mean_q) * inv_std_q
        y = scale_q * (x_n + bias_q - threshold_q)
        out = sign(y) when `output_mode="bipolar"`

    The working hardware-friendly path keeps the reduction-heavy normalization
    math in float during training, while fake-quantizing the learned affine
    parameters and the inference-time EMA `mean` / `inv_std` values onto the
    same 8-bit-style grid used for export. This preserves training stability
    while making validation and export follow the fixed-point inference path.

    Export recipe for C/FPGA:
    1. Use `moving_mean` / `moving_variance` gathered during training.
    2. Quantize `norm_scale`, `norm_threshold`, and optional `bias` with the
       same rules as the forward pass.
    3. Precompute:
          inv_std = 1 / sqrt(moving_variance + eps)
          fused_scale = scale_q * inv_std
          fused_offset = scale_q * ((bias_q - threshold_q) - moving_mean * inv_std)
       or equivalently keep the normalized form if your runtime already has a
       reciprocal-sqrt implementation.
    4. For bipolar output, the runtime op is just:
          out = sign(fused_scale * x + fused_offset)
       so the layer can be reduced to a per-channel signed scale plus a
       per-channel comparator threshold.
    """

    def __init__(
        self,
        levels=128,
        threshold_limit=1.0,
        use_bias=True,
        output_mode="bipolar",
        feature_axis=None,
        data_format=None,
        momentum=0.95,
        epsilon=1e-4,
        use_running_stats=True,
        quantize_input=False,
        quantize_centered=False,
        quantize_normalized=False,
        quantize_output=False,
        quantize_eval_stats=True,
        eval_use_batch_stats=False,
        force_full_quant_eval=False,
        quantize_affine_params=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.levels = int(levels)
        if self.levels <= 1:
            raise ValueError(f"`levels` must be > 1. Got: {levels!r}")
        self.threshold_limit = float(threshold_limit)
        if self.threshold_limit <= 0.0:
            raise ValueError(f"`threshold_limit` must be > 0. Got: {threshold_limit!r}")
        self.use_bias = bool(use_bias)
        self.output_mode = _normalize_mode(output_mode, "output_mode", {"raw", "bipolar"})
        self.feature_axis = _normalize_feature_axis(feature_axis)
        self.data_format = _normalize_data_format(data_format)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.use_running_stats = bool(use_running_stats)
        self.quantize_input = bool(quantize_input)
        self.quantize_centered = bool(quantize_centered)
        self.quantize_normalized = bool(quantize_normalized)
        self.quantize_output = bool(quantize_output)
        self.quantize_eval_stats = bool(quantize_eval_stats)
        self.eval_use_batch_stats = bool(eval_use_batch_stats)
        self.force_full_quant_eval = bool(force_full_quant_eval)
        self.quantize_affine_params = bool(quantize_affine_params)
        self.scale_constraint = ClipConstraint(1.0)
        self.threshold_constraint = ClipRangeConstraint(-self.threshold_limit, self.threshold_limit)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        rank = len(input_shape)
        axis = _resolve_feature_axis(self.feature_axis, rank, self.data_format)
        feature_dim = input_shape[axis]
        if feature_dim is None:
            raise ValueError("The feature dimension must be defined.")
        self._rank = rank
        self._feature_axis = axis
        self._feature_dim = int(feature_dim)
        shape = (self._feature_dim,)
        self._reduce_axes = tuple(i for i in range(rank) if i != axis)
        self.norm_scale = self.add_weight(
            name="norm_scale",
            shape=shape,
            initializer="ones",
            trainable=True,
            constraint=self.scale_constraint,
        )
        self.norm_threshold = self.add_weight(
            name="norm_threshold",
            shape=shape,
            initializer="zeros",
            trainable=True,
            constraint=self.threshold_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=shape,
                initializer="zeros",
                trainable=True,
                constraint=self.threshold_constraint,
            )
        else:
            self.bias = None
        if self.use_running_stats:
            self.moving_mean = self.add_weight(
                name="moving_mean",
                shape=shape,
                initializer="zeros",
                trainable=False,
            )
            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=shape,
                initializer="ones",
                trainable=False,
            )
        else:
            self.moving_mean = None
            self.moving_variance = None
        self._broadcast_shape = [1] * rank
        self._broadcast_shape[axis] = self._feature_dim
        # The binary YOLO path feeds normalized XNOR-popcount features here, so
        # the pre-norm signal is expected to stay near [-1, 1]. We keep all
        # forward operands on an 8-bit grid around that range to match the
        # eventual fixed-point runtime while still using float math on GPU.
        self._input_limit = max(1.0, self.threshold_limit)
        self._center_limit = max(2.0, self._input_limit * 2.0)
        self._variance_limit = self._center_limit * self._center_limit
        self._inv_std_limit = 8.0
        self._normalized_limit = max(2.0, self.threshold_limit)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = ops.cast(inputs, "float32" if training is False else self.compute_dtype)
        eval_full_quant = bool(self.force_full_quant_eval and training is False)
        quantize_input = self.quantize_input or eval_full_quant
        quantize_centered = self.quantize_centered or eval_full_quant
        quantize_normalized = self.quantize_normalized or eval_full_quant
        quantize_output = self.quantize_output or eval_full_quant

        if quantize_input:
            x = quantize_symmetric_range_ste(x, max_abs=self._input_limit, levels=self.levels)
        if self.use_running_stats:
            use_running = (training is False) and (not self.eval_use_batch_stats)
            if use_running:
                mean = ops.cast(self.moving_mean, x.dtype)
                variance = ops.cast(self.moving_variance, x.dtype)
                if self.quantize_eval_stats or eval_full_quant:
                    mean = quantize_symmetric_range_ste(
                        mean,
                        max_abs=self._input_limit,
                        levels=self.levels,
                    )
                    inv_std = quantize_positive_range_ste(
                        ops.cast(1.0, x.dtype) / ops.sqrt(variance + ops.cast(self.epsilon, x.dtype)),
                        max_value=self._inv_std_limit,
                        levels=self.levels,
                    )
                else:
                    inv_std = ops.cast(1.0, x.dtype) / ops.sqrt(variance + ops.cast(self.epsilon, x.dtype))
            else:
                # Training-time stats stabilize the pre-sign distribution. At
                # inference/export time these are replaced by the EMA values.
                mean = ops.mean(x, axis=self._reduce_axes)
                centered = x - ops.reshape(mean, self._broadcast_shape)
                if quantize_centered:
                    centered = quantize_symmetric_range_ste(
                        centered,
                        max_abs=self._center_limit,
                        levels=self.levels,
                    )
                variance = ops.mean(ops.square(centered), axis=self._reduce_axes)
                momentum = ops.cast(self.momentum, self.moving_mean.dtype)
                one_minus = ops.cast(1.0 - self.momentum, self.moving_mean.dtype)
                self.moving_mean.assign(momentum * self.moving_mean + one_minus * ops.cast(mean, self.moving_mean.dtype))
                self.moving_variance.assign(momentum * self.moving_variance + one_minus * ops.cast(variance, self.moving_variance.dtype))
            mean = ops.reshape(ops.cast(mean, x.dtype), self._broadcast_shape)
            centered = x - mean
            if quantize_centered:
                centered = quantize_symmetric_range_ste(
                    centered,
                    max_abs=self._center_limit,
                    levels=self.levels,
                )
            if use_running:
                inv_std = ops.reshape(ops.cast(inv_std, x.dtype), self._broadcast_shape)
                x = centered * inv_std
            else:
                variance = ops.reshape(ops.cast(variance, x.dtype), self._broadcast_shape)
                x = centered / ops.sqrt(variance + ops.cast(self.epsilon, x.dtype))
            if quantize_normalized:
                x = quantize_symmetric_range_ste(
                    x,
                    max_abs=self._normalized_limit,
                    levels=self.levels,
                )
        # The learned affine terms are quantized every forward pass so the train
        # graph matches the future fixed-point path as closely as possible.
        if self.quantize_affine_params:
            scale = quantize_symmetric_range_ste(
                ops.cast(self.norm_scale, x.dtype),
                max_abs=1.0,
                levels=self.levels,
            )
            threshold = quantize_symmetric_range_ste(
                ops.cast(self.norm_threshold, x.dtype),
                max_abs=self.threshold_limit,
                levels=self.levels,
            )
        else:
            scale = ops.cast(self.norm_scale, x.dtype)
            threshold = ops.cast(self.norm_threshold, x.dtype)
        scale = ops.reshape(scale, self._broadcast_shape)
        threshold = ops.reshape(threshold, self._broadcast_shape)
        x = x - threshold
        if self.use_bias:
            if self.quantize_affine_params:
                bias = quantize_symmetric_range_ste(
                    ops.cast(self.bias, x.dtype),
                    max_abs=self.threshold_limit,
                    levels=self.levels,
                )
            else:
                bias = ops.cast(self.bias, x.dtype)
            x = x + ops.reshape(bias, self._broadcast_shape)
        if quantize_output:
            x = quantize_symmetric_range_ste(
                x,
                max_abs=self._normalized_limit,
                levels=self.levels,
            )
        x = scale * x
        if quantize_output:
            x = quantize_symmetric_range_ste(
                x,
                max_abs=self._normalized_limit,
                levels=self.levels,
            )
        if self.output_mode == "bipolar":
            # In hardware this is the only nonlinear step left after folding:
            # compare the affine-normalized channel against zero.
            x = binary_sign_ste(x)
        return x

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "levels": self.levels,
                "threshold_limit": self.threshold_limit,
                "use_bias": self.use_bias,
                "output_mode": self.output_mode,
                "feature_axis": self.feature_axis,
                "data_format": self.data_format,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_running_stats": self.use_running_stats,
                "quantize_input": self.quantize_input,
                "quantize_centered": self.quantize_centered,
                "quantize_normalized": self.quantize_normalized,
                "quantize_output": self.quantize_output,
                "quantize_eval_stats": self.quantize_eval_stats,
                "eval_use_batch_stats": self.eval_use_batch_stats,
                "force_full_quant_eval": self.force_full_quant_eval,
                "quantize_affine_params": self.quantize_affine_params,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryConv2D(keras.layers.Layer):
    """Backend-agnostic binary 2D convolution using Keras ops.

    Forward pass:
        x_b = sign(x) with STE gradients
        w_b = sign(w) with STE gradients
        y = conv(x_b, alpha * w_b)
        out = sign(y - threshold) when `output_mode="bipolar"`
    where alpha is the per-filter mean absolute kernel magnitude.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        use_bias=True,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        score_mode="scaled_sign",
        input_threshold=0.5,
        output_mode="raw",
        output_levels=128,
        use_batchnorm=False,
        bn_momentum=0.95,
        bn_quantize_input=False,
        bn_quantize_centered=False,
        bn_quantize_normalized=False,
        bn_quantize_output=False,
        bn_quantize_eval_stats=True,
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
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.score_mode = _normalize_score_mode(score_mode)
        self.input_threshold = float(input_threshold)
        self.output_mode = _normalize_mode(output_mode, "output_mode", {"raw", "bipolar"})
        self.output_levels = int(output_levels)
        self.use_batchnorm = bool(use_batchnorm)
        self.bn_momentum = float(bn_momentum)
        self.bn_quantize_input = bool(bn_quantize_input)
        self.bn_quantize_centered = bool(bn_quantize_centered)
        self.bn_quantize_normalized = bool(bn_quantize_normalized)
        self.bn_quantize_output = bool(bn_quantize_output)
        self.bn_quantize_eval_stats = bool(bn_quantize_eval_stats)
        if self.output_levels <= 1:
            raise ValueError(f"`output_levels` must be > 1. Got: {output_levels!r}")
        self.kernel_constraint = ClipConstraint(1.0)
        self.bias_constraint = None
        self.affine_norm = None

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        if self.data_format == "channels_last":
            in_channels = input_shape[-1]
        else:
            in_channels = input_shape[1]
        if in_channels is None:
            raise ValueError("The channel dimension of the inputs must be defined.")

        kernel_shape = self.kernel_size + (int(in_channels), self.filters)
        self._kernel_elements = int(self.kernel_size[0] * self.kernel_size[1] * int(in_channels))
        if self.output_mode == "bipolar" and self.score_mode == "xnor_popcount":
            threshold_limit = math.sqrt(float(self._kernel_elements))
        else:
            threshold_limit = float((self.output_levels - 1) / self.output_levels)
        self._threshold_limit = threshold_limit
        self.bias_constraint = ClipRangeConstraint(-threshold_limit, threshold_limit)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            constraint=self.kernel_constraint,
        )
        if self.use_batchnorm:
            self.bias = None
            self.affine_norm = BinaryAffineNorm(
                levels=self.output_levels,
                threshold_limit=threshold_limit,
                use_bias=self.use_bias,
                output_mode=self.output_mode,
                data_format=self.data_format,
                momentum=self.bn_momentum,
                quantize_input=self.bn_quantize_input,
                quantize_centered=self.bn_quantize_centered,
                quantize_normalized=self.bn_quantize_normalized,
                quantize_output=self.bn_quantize_output,
                quantize_eval_stats=self.bn_quantize_eval_stats,
                name="affine_norm",
            )
            self.affine_norm.build(self.compute_output_shape(input_shape))
        elif self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                constraint=self.bias_constraint if self.output_mode == "bipolar" else None,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        inputs = ops.cast(inputs, self.compute_dtype)
        kernel = ops.convert_to_tensor(self.kernel)
        if self.score_mode == "scaled_sign":
            x_b = binary_sign_ste(inputs)
            w_b = binary_sign_ste(kernel)
            reduce_axes = (0, 1, 2)
            alpha = ops.mean(ops.absolute(kernel), axis=reduce_axes, keepdims=True)
            kernel_eff = w_b * ops.cast(alpha, w_b.dtype)
            outputs = ops.conv(
                x_b,
                kernel_eff,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )
        else:
            x_b = binary_sign_threshold_ste(inputs, threshold=self.input_threshold)
            w_b = binary_sign_ste(kernel)
            outputs = ops.conv(
                x_b,
                w_b,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )
            if self.output_mode == "bipolar":
                kernel_elems = ops.cast(math.sqrt(float(self._kernel_elements)), outputs.dtype)
                outputs = outputs / kernel_elems
            else:
                kernel_elems = ops.cast(float(self._kernel_elements), outputs.dtype)
                outputs = outputs / kernel_elems
        if self.bias is not None:
            bias = ops.reshape(ops.convert_to_tensor(self.bias), _conv_bias_shape(self.filters, self.data_format))
            bias = ops.cast(bias, outputs.dtype)
            if self.output_mode == "bipolar" and not self.use_batchnorm:
                if self.score_mode != "xnor_popcount":
                    bias = quantize_signed_unit_interval_ste(bias, levels=self.output_levels)
                outputs = binary_sign_threshold_ste(outputs, threshold=bias)
            else:
                outputs = outputs + bias
        if self.use_batchnorm:
            outputs = self.affine_norm(outputs, training=training)
        elif self.output_mode == "bipolar" and not self.use_bias:
            outputs = binary_sign_ste(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
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
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "score_mode": self.score_mode,
                "input_threshold": self.input_threshold,
                "output_mode": self.output_mode,
                "output_levels": self.output_levels,
                "use_batchnorm": self.use_batchnorm,
                "bn_momentum": self.bn_momentum,
                "bn_quantize_input": self.bn_quantize_input,
                "bn_quantize_centered": self.bn_quantize_centered,
                "bn_quantize_normalized": self.bn_quantize_normalized,
                "bn_quantize_output": self.bn_quantize_output,
                "bn_quantize_eval_stats": self.bn_quantize_eval_stats,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryDense(keras.layers.Layer):
    """Binary dense layer with either scaled-sign or XNOR-popcount scores."""

    def __init__(
        self,
        units,
        use_bias=True,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        score_mode="scaled_sign",
        input_threshold=0.5,
        output_mode="raw",
        output_levels=128,
        raw_normalize=True,
        use_batchnorm=False,
        bn_momentum=0.95,
        bn_quantize_input=False,
        bn_quantize_centered=False,
        bn_quantize_normalized=False,
        bn_quantize_output=False,
        bn_quantize_eval_stats=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.use_bias = bool(use_bias)
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.score_mode = _normalize_score_mode(score_mode)
        self.input_threshold = float(input_threshold)
        self.output_mode = _normalize_mode(output_mode, "output_mode", {"raw", "bipolar"})
        self.output_levels = int(output_levels)
        self.raw_normalize = bool(raw_normalize)
        self.use_batchnorm = bool(use_batchnorm)
        self.bn_momentum = float(bn_momentum)
        self.bn_quantize_input = bool(bn_quantize_input)
        self.bn_quantize_centered = bool(bn_quantize_centered)
        self.bn_quantize_normalized = bool(bn_quantize_normalized)
        self.bn_quantize_output = bool(bn_quantize_output)
        self.bn_quantize_eval_stats = bool(bn_quantize_eval_stats)
        if self.output_levels <= 1:
            raise ValueError(f"`output_levels` must be > 1. Got: {output_levels!r}")
        self.kernel_constraint = ClipConstraint(1.0)
        self.bias_constraint = None
        self.affine_norm = None

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        in_dim = input_shape[-1]
        if in_dim is None:
            raise ValueError("The last input dimension must be defined.")
        self._input_dim = int(in_dim)
        if self.output_mode == "bipolar" and self.score_mode == "xnor_popcount":
            threshold_limit = math.sqrt(float(self._input_dim))
        else:
            threshold_limit = float((self.output_levels - 1) / self.output_levels)
        self._threshold_limit = threshold_limit
        self.bias_constraint = ClipRangeConstraint(-threshold_limit, threshold_limit)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self._input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            constraint=self.kernel_constraint,
        )
        if self.use_batchnorm:
            self.bias = None
            self.affine_norm = BinaryAffineNorm(
                levels=self.output_levels,
                threshold_limit=threshold_limit,
                use_bias=self.use_bias,
                output_mode=self.output_mode,
                feature_axis=-1,
                data_format="channels_last",
                momentum=self.bn_momentum,
                quantize_input=self.bn_quantize_input,
                quantize_centered=self.bn_quantize_centered,
                quantize_normalized=self.bn_quantize_normalized,
                quantize_output=self.bn_quantize_output,
                quantize_eval_stats=self.bn_quantize_eval_stats,
                name="affine_norm",
            )
            self.affine_norm.build(self.compute_output_shape(input_shape))
        elif self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
                constraint=self.bias_constraint if self.output_mode == "bipolar" else None,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        inputs = ops.cast(inputs, self.compute_dtype)
        kernel = ops.convert_to_tensor(self.kernel)
        if self.score_mode == "scaled_sign":
            x_b = binary_sign_ste(inputs)
            w_b = binary_sign_ste(kernel)
            alpha = ops.mean(ops.absolute(kernel), axis=0, keepdims=True)
            outputs = ops.matmul(x_b, w_b * ops.cast(alpha, w_b.dtype))
        else:
            x_b = binary_sign_threshold_ste(inputs, threshold=self.input_threshold)
            w_b = binary_sign_ste(kernel)
            outputs = ops.matmul(x_b, w_b)
            if self.output_mode == "bipolar":
                input_dim = ops.cast(math.sqrt(float(self._input_dim)), outputs.dtype)
                outputs = outputs / input_dim
            elif self.raw_normalize:
                input_dim = ops.cast(float(self._input_dim), outputs.dtype)
                outputs = outputs / input_dim
        if self.bias is not None:
            bias = ops.cast(ops.convert_to_tensor(self.bias), outputs.dtype)
            if self.output_mode == "bipolar" and not self.use_batchnorm:
                if self.score_mode != "xnor_popcount":
                    bias = quantize_signed_unit_interval_ste(bias, levels=self.output_levels)
                outputs = binary_sign_threshold_ste(outputs, threshold=bias)
            else:
                outputs = outputs + bias
        if self.use_batchnorm:
            outputs = self.affine_norm(outputs, training=training)
        elif self.output_mode == "bipolar" and not self.use_bias:
            outputs = binary_sign_ste(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
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
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "score_mode": self.score_mode,
                "input_threshold": self.input_threshold,
                "output_mode": self.output_mode,
                "output_levels": self.output_levels,
                "raw_normalize": self.raw_normalize,
                "use_batchnorm": self.use_batchnorm,
                "bn_momentum": self.bn_momentum,
                "bn_quantize_input": self.bn_quantize_input,
                "bn_quantize_centered": self.bn_quantize_centered,
                "bn_quantize_normalized": self.bn_quantize_normalized,
                "bn_quantize_output": self.bn_quantize_output,
                "bn_quantize_eval_stats": self.bn_quantize_eval_stats,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryOutput(BinaryDense):
    """Binary output projection with raw bipolar dot-product logits.

    This is intended for classifier heads that should stay hardware-friendly:
    inputs and weights are binarized to -1/+1, multiplied, and summed, but the
    layer does not apply a post-score threshold. By default it returns the raw
    accumulator-like logits rather than the normalized [-1, 1] score used by
    `BinaryDense(score_mode="xnor_popcount")`.
    """

    def __init__(
        self,
        units,
        use_bias=True,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        input_threshold=0.0,
        normalize=False,
        **kwargs,
    ):
        super().__init__(
            units=units,
            use_bias=use_bias,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            score_mode="xnor_popcount",
            input_threshold=input_threshold,
            output_mode="raw",
            raw_normalize=normalize,
            **kwargs,
        )
        self.normalize = bool(normalize)

    def get_config(self):
        config = super().get_config()
        config.pop("score_mode", None)
        config.pop("raw_normalize", None)
        config.update({"normalize": self.normalize})
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class QuantizedDense(keras.layers.Layer):
    """Dense layer with signed STE-quantized weights stored in `[-1, 1]`.

    When `integer_output=False`, weights are quantized onto a signed unit grid.

    When `integer_output=True`, the forward path uses a Keras-like symmetric
    signed integer view of the weights and outputs. For `levels=128`, that
    means:
    - stored weights remain floats in `[-1, 1]`
    - export-time integer values are `round(weight * 127)`
    - forward outputs are requantized onto the same signed grid, scaled by
      `output_scale`

    This is intended to make future fixed-point export simple and explicit.
    """

    def __init__(
        self,
        units,
        use_bias=True,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        levels=128,
        input_threshold=0.0,
        binarize_inputs=True,
        integer_output=False,
        integer_kernel=False,
        output_scale=1.0,
        quantize_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.use_bias = bool(use_bias)
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.levels = int(levels)
        if self.levels <= 1:
            raise ValueError(f"`levels` must be > 1. Got: {levels!r}")
        self.input_threshold = float(input_threshold)
        self.binarize_inputs = bool(binarize_inputs)
        self.integer_output = bool(integer_output)
        self.integer_kernel = bool(integer_kernel)
        self.output_scale = float(output_scale)
        if self.output_scale <= 0.0:
            raise ValueError(f"`output_scale` must be > 0. Got: {output_scale!r}")
        self.quantize_bias = bool(quantize_bias)
        self._kernel_max = 1.0
        self.kernel_constraint = ClipRangeConstraint(-1.0, self._kernel_max)

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
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        inputs = ops.cast(inputs, "float32" if training is False else self.compute_dtype)
        if training is False:
            if self.binarize_inputs:
                threshold = ops.cast(self.input_threshold, inputs.dtype)
                inputs_eff = ops.where(inputs >= threshold, _cast_scalar(1.0, inputs.dtype), _cast_scalar(-1.0, inputs.dtype))
            else:
                inputs_eff = quantize_to_fixed_grid_ops(inputs)
            kernel = ops.convert_to_tensor(self.kernel)
            if self.integer_kernel:
                kernel_eff = quantize_signed_int_grid_ops(kernel, levels=self.levels)
                outputs = ops.matmul(ops.cast(inputs_eff, "float32"), ops.cast(kernel_eff, "float32"))
                outputs = quantize_to_fixed_grid_ops(outputs)
            else:
                kernel_eff = quantize_signed_unit_grid_ops(kernel, levels=self.levels)
                outputs = ops.matmul(ops.cast(inputs_eff, "float32"), ops.cast(kernel_eff, "float32"))
                outputs = quantize_to_fixed_grid_ops(outputs / _cast_scalar(self.levels, "float32"))
            if self.use_bias:
                bias = ops.cast(ops.convert_to_tensor(self.bias), "float32")
                if self.quantize_bias and self.integer_output:
                    bias = fake_quantize_signed_output_exact_ops(bias, levels=self.levels, scale=self.output_scale)
                bias = quantize_to_fixed_grid_ops(bias)
                outputs = outputs + bias
            if self.integer_output:
                outputs = fake_quantize_signed_output_exact_ops(outputs, levels=self.levels, scale=self.output_scale)
            if self.activation is not None:
                outputs = self.activation(outputs)
            return outputs
        if self.binarize_inputs:
            inputs_eff = binary_sign_threshold_ste(inputs, threshold=self.input_threshold)
        else:
            inputs_eff = inputs
        kernel = ops.convert_to_tensor(self.kernel)
        if self.integer_kernel:
            kernel_eff = quantize_signed_int_ste(kernel, levels=self.levels)
        else:
            kernel_eff = quantize_signed_unit_interval_ste(kernel, levels=self.levels)
        outputs = ops.matmul(inputs_eff, kernel_eff)
        if self.use_bias:
            bias = ops.cast(ops.convert_to_tensor(self.bias), outputs.dtype)
            if self.quantize_bias and self.integer_output:
                bias = fake_quantize_signed_output_ste(bias, levels=self.levels, scale=self.output_scale)
            outputs = outputs + bias
        if self.integer_output:
            outputs = fake_quantize_signed_output_ste(outputs, levels=self.levels, scale=self.output_scale)
        if self.activation is not None:
            outputs = self.activation(outputs)
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
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "levels": self.levels,
                "input_threshold": self.input_threshold,
                "binarize_inputs": self.binarize_inputs,
                "integer_output": self.integer_output,
                "integer_kernel": self.integer_kernel,
                "output_scale": self.output_scale,
                "quantize_bias": self.quantize_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class QuantizedConv2D(keras.layers.Layer):
    """Conv2D with STE-quantized weights and optional requantized outputs."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="same",
        data_format=None,
        dilation_rate=1,
        use_bias=True,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        levels=128,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=False,
        integer_kernel=True,
        output_scale=1.0,
        quantize_bias=True,
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
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.levels = int(levels)
        if self.levels <= 1:
            raise ValueError(f"`levels` must be > 1. Got: {levels!r}")
        self.input_threshold = float(input_threshold)
        self.binarize_inputs = bool(binarize_inputs)
        self.integer_output = bool(integer_output)
        self.integer_kernel = bool(integer_kernel)
        self.output_scale = float(output_scale)
        if self.output_scale <= 0.0:
            raise ValueError(f"`output_scale` must be > 0. Got: {output_scale!r}")
        self.quantize_bias = bool(quantize_bias)
        self.kernel_constraint = ClipRangeConstraint(-1.0, 1.0)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        if self.data_format == "channels_last":
            in_channels = input_shape[-1]
        else:
            in_channels = input_shape[1]
        if in_channels is None:
            raise ValueError("The input channel dimension must be defined.")
        self._input_channels = int(in_channels)
        kernel_shape = (*self.kernel_size, self._input_channels, self.filters)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        inputs = ops.cast(inputs, "float32" if training is False else self.compute_dtype)
        if training is False:
            if self.binarize_inputs:
                threshold = ops.cast(self.input_threshold, inputs.dtype)
                inputs_eff = ops.where(inputs >= threshold, _cast_scalar(1.0, inputs.dtype), _cast_scalar(-1.0, inputs.dtype))
            else:
                inputs_eff = quantize_to_fixed_grid_ops(inputs)
            kernel = ops.convert_to_tensor(self.kernel)
            if self.integer_kernel:
                kernel_eff = quantize_signed_int_grid_ops(kernel, levels=self.levels)
                outputs = ops.conv(
                    ops.cast(inputs_eff, "float32"),
                    ops.cast(kernel_eff, "float32"),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
                outputs = quantize_to_fixed_grid_ops(outputs)
            else:
                kernel_eff = quantize_signed_unit_grid_ops(kernel, levels=self.levels)
                outputs = ops.conv(
                    ops.cast(inputs_eff, "float32"),
                    ops.cast(kernel_eff, "float32"),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
                outputs = quantize_to_fixed_grid_ops(outputs / _cast_scalar(self.levels, "float32"))
            if self.use_bias:
                bias = ops.cast(ops.convert_to_tensor(self.bias), "float32")
                bias = ops.reshape(bias, _conv_bias_shape(self.filters, self.data_format))
                if self.quantize_bias and self.integer_output:
                    bias = fake_quantize_signed_output_exact_ops(
                        bias,
                        levels=self.levels,
                        scale=self.output_scale,
                    )
                bias = quantize_to_fixed_grid_ops(bias)
                outputs = outputs + bias
            if self.integer_output:
                outputs = fake_quantize_signed_output_exact_ops(
                    outputs,
                    levels=self.levels,
                    scale=self.output_scale,
                )
            if self.activation is not None:
                outputs = self.activation(outputs)
            return outputs
        if self.binarize_inputs:
            inputs_eff = binary_sign_threshold_ste(inputs, threshold=self.input_threshold)
        else:
            inputs_eff = inputs
        kernel = ops.convert_to_tensor(self.kernel)
        if self.integer_kernel:
            kernel_eff = quantize_signed_int_ste(kernel, levels=self.levels)
        else:
            kernel_eff = quantize_signed_unit_interval_ste(kernel, levels=self.levels)
        outputs = ops.conv(
            inputs_eff,
            kernel_eff,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            bias = ops.cast(ops.convert_to_tensor(self.bias), outputs.dtype)
            bias = ops.reshape(bias, _conv_bias_shape(self.filters, self.data_format))
            if self.quantize_bias and self.integer_output:
                bias = fake_quantize_signed_output_ste(
                    bias,
                    levels=self.levels,
                    scale=self.output_scale,
                )
            outputs = outputs + bias
        if self.integer_output:
            outputs = fake_quantize_signed_output_ste(
                outputs,
                levels=self.levels,
                scale=self.output_scale,
            )
        if self.activation is not None:
            outputs = self.activation(outputs)
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
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "levels": self.levels,
                "input_threshold": self.input_threshold,
                "binarize_inputs": self.binarize_inputs,
                "integer_output": self.integer_output,
                "integer_kernel": self.integer_kernel,
                "output_scale": self.output_scale,
                "quantize_bias": self.quantize_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class QuantizedOutput(QuantizedDense):
    """Compatibility shim for QuantizedDense with output requantization enabled."""

    def __init__(
        self,
        units,
        use_bias=True,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        levels=128,
        input_threshold=0.0,
        binarize_inputs=True,
        integer_kernel=True,
        output_scale=1.0,
        quantize_bias=True,
        **kwargs,
    ):
        super().__init__(
            units=units,
            use_bias=use_bias,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            levels=levels,
            input_threshold=input_threshold,
            binarize_inputs=binarize_inputs,
            integer_output=True,
            integer_kernel=integer_kernel,
            output_scale=output_scale,
            quantize_bias=quantize_bias,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="BinaryLayers")
class QuantizedActivation(keras.layers.Layer):
    """STE activation quantizer for fixed-point-friendly hidden states."""

    def __init__(self, levels=128, signed=True, **kwargs):
        super().__init__(**kwargs)
        self.levels = int(levels)
        if self.levels <= 1:
            raise ValueError(f"`levels` must be > 1. Got: {levels!r}")
        self.signed = bool(signed)

    def call(self, inputs, training=None):
        inputs = ops.cast(inputs, "float32" if training is False else self.compute_dtype)
        if training is False:
            inputs = quantize_to_fixed_grid_ops(inputs)
            if self.signed:
                return quantize_signed_unit_interval_exact_ops(inputs, levels=self.levels)
            return quantize_unit_interval_exact_ops(inputs, levels=self.levels)
        if self.signed:
            return quantize_signed_unit_interval_ste(inputs, levels=self.levels)
        return quantize_unit_interval_ste(inputs, levels=self.levels)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"levels": self.levels, "signed": self.signed})
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryCutout(keras.layers.Layer):
    """Randomly replace rectangular regions with binary values during training."""

    def __init__(
        self,
        factor=0.15,
        size_range=(0.08, 0.16),
        num_boxes=1,
        method="noise",
        data_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factor = float(factor)
        self.size_range = self._normalize_size_range(size_range)
        self.num_boxes = int(num_boxes)
        if self.num_boxes <= 0:
            raise ValueError(f"`num_boxes` must be >= 1. Got: {num_boxes!r}")
        self.method = _normalize_mode(method, "method", {"black", "white", "both", "noise"})
        self.data_format = _normalize_data_format(data_format)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    @staticmethod
    def _normalize_axis_range(value, name):
        if isinstance(value, Real):
            lo = hi = float(value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            lo, hi = (float(v) for v in value)
        else:
            raise ValueError(f"`{name}` must be a number or a length-2 tuple/list. Got: {value!r}")
        if lo < 0.0 or hi < 0.0:
            raise ValueError(f"`{name}` values must be non-negative. Got: {value!r}")
        if hi < lo:
            raise ValueError(f"`{name}` max must be >= min. Got: {value!r}")
        return (lo, hi)

    @classmethod
    def _normalize_size_range(cls, value):
        if isinstance(value, Real):
            axis_range = cls._normalize_axis_range(value, "size_range")
            return (axis_range, axis_range)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if all(isinstance(v, Real) for v in value):
                axis_range = cls._normalize_axis_range(value, "size_range")
                return (axis_range, axis_range)
            height_range = cls._normalize_axis_range(value[0], "size_range[0]")
            width_range = cls._normalize_axis_range(value[1], "size_range[1]")
            return (height_range, width_range)
        raise ValueError(
            "`size_range` must be a number, a length-2 float range, or "
            "((h_min, h_max), (w_min, w_max)). "
            f"Got: {value!r}"
        )

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError("BinaryCutout expects rank-4 image or feature map inputs.")
        if self.data_format == "channels_last":
            height = input_shape[1]
            width = input_shape[2]
        else:
            height = input_shape[2]
            width = input_shape[3]
        if height is None or width is None:
            raise ValueError("BinaryCutout requires statically known spatial dimensions.")
        self._height = int(height)
        self._width = int(width)
        height_range, width_range = self.size_range
        self._min_height = max(1, int(round(self._height * height_range[0])))
        self._max_height = min(self._height, max(self._min_height, int(round(self._height * height_range[1]))))
        self._min_width = max(1, int(round(self._width * width_range[0])))
        self._max_width = min(self._width, max(self._min_width, int(round(self._width * width_range[1]))))
        super().build(input_shape)

    def _make_box_mask(self, batch, dtype):
        box_height = keras.random.randint(
            (batch, 1, 1, 1),
            minval=self._min_height,
            maxval=self._max_height + 1,
            dtype="int32",
            seed=self.seed_generator,
        )
        box_width = keras.random.randint(
            (batch, 1, 1, 1),
            minval=self._min_width,
            maxval=self._max_width + 1,
            dtype="int32",
            seed=self.seed_generator,
        )
        top = keras.random.randint(
            (batch, 1, 1, 1),
            minval=0,
            maxval=self._height,
            dtype="int32",
            seed=self.seed_generator,
        )
        left = keras.random.randint(
            (batch, 1, 1, 1),
            minval=0,
            maxval=self._width,
            dtype="int32",
            seed=self.seed_generator,
        )
        bottom = ops.minimum(top + box_height, self._height)
        right = ops.minimum(left + box_width, self._width)
        rows = ops.arange(self._height, dtype="int32")
        cols = ops.arange(self._width, dtype="int32")
        if self.data_format == "channels_last":
            rows = ops.reshape(rows, (1, self._height, 1, 1))
            cols = ops.reshape(cols, (1, 1, self._width, 1))
        else:
            rows = ops.reshape(rows, (1, 1, self._height, 1))
            cols = ops.reshape(cols, (1, 1, 1, self._width))
        mask = ops.logical_and(rows >= top, rows < bottom) & ops.logical_and(cols >= left, cols < right)
        return ops.cast(mask, dtype)

    def _make_fill(self, inputs):
        dtype = inputs.dtype
        bipolar = ops.cast(ops.min(inputs) < 0.0, dtype)
        black = -ops.ones_like(inputs) * bipolar
        white = ops.ones_like(inputs)
        if self.method == "black":
            return black
        if self.method == "white":
            return white
        if self.method == "both":
            batch = ops.shape(inputs)[0]
            pick = keras.random.randint(
                (batch, 1, 1, 1),
                minval=0,
                maxval=2,
                dtype="int32",
                seed=self.seed_generator,
            )
            pick = ops.cast(pick, dtype)
            return pick * (white - black) + black
        noise = keras.random.randint(
            ops.shape(inputs),
            minval=0,
            maxval=2,
            dtype="int32",
            seed=self.seed_generator,
        )
        noise = ops.cast(noise, dtype)
        return noise * (white - black) + black

    def call(self, inputs, training=None):
        if training is False or self.factor <= 0.0:
            return inputs

        outputs = ops.cast(inputs, self.compute_dtype)
        batch = ops.shape(outputs)[0]
        one = ops.cast(1.0, outputs.dtype)
        for _ in range(self.num_boxes):
            apply_mask = ops.cast(
                keras.random.uniform((batch, 1, 1, 1), 0.0, 1.0, seed=self.seed_generator) < self.factor,
                outputs.dtype,
            )
            box_mask = self._make_box_mask(batch, outputs.dtype)
            replace_mask = box_mask * apply_mask
            fill = self._make_fill(outputs)
            outputs = outputs * (one - replace_mask) + fill * replace_mask
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "size_range": [list(self.size_range[0]), list(self.size_range[1])],
                "num_boxes": self.num_boxes,
                "method": self.method,
                "data_format": self.data_format,
                "seed": self.seed,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryGreyout(BinaryCutout):
    """Backward-compatible alias for BinaryCutout."""

    def __init__(
        self,
        factor=0.15,
        min_frac=0.08,
        max_frac=0.16,
        mode="noise",
        num_boxes=1,
        data_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(
            factor=factor,
            size_range=(min_frac, max_frac),
            num_boxes=num_boxes,
            method=mode,
            data_format=data_format,
            seed=seed,
            **kwargs,
        )
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)
        self.mode = _normalize_mode(mode, "mode", {"black", "white", "both", "noise"})

    def get_config(self):
        config = super().get_config()
        config.pop("size_range", None)
        config.pop("method", None)
        config.update(
            {
                "min_frac": self.min_frac,
                "max_frac": self.max_frac,
                "mode": self.mode,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryDropcut(BinaryCutout):
    """Alias for BinaryCutout using the requested Dropcut naming."""

    def __init__(
        self,
        dropcut_probability=0.25,
        size_range=(0.05, 0.20),
        num_boxes=1,
        method="noise",
        data_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(
            factor=dropcut_probability,
            size_range=size_range,
            num_boxes=num_boxes,
            method=method,
            data_format=data_format,
            seed=seed,
            **kwargs,
        )
        self.dropcut_probability = float(dropcut_probability)

    def get_config(self):
        config = super().get_config()
        config.pop("factor", None)
        config.update({"dropcut_probability": self.dropcut_probability})
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinarySoftmax(keras.layers.Layer):
    """Straight-through hardmax with a normal softmax gradient."""

    def __init__(self, axis=-1, epsilon=1.0 / 128.0, logit_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        if int(axis) != -1:
            raise ValueError("BinarySoftmax currently supports only axis=-1.")
        self.axis = int(axis)
        self.epsilon = float(epsilon)
        self.logit_scale = float(logit_scale)

    def call(self, inputs, training=None):
        inputs = ops.cast(inputs, self.compute_dtype)
        logits = inputs * ops.cast(self.logit_scale, inputs.dtype)
        soft = ops.softmax(logits, axis=self.axis)
        num_classes = soft.shape[-1]
        if num_classes is None:
            raise ValueError("BinarySoftmax requires a statically known class dimension.")
        hard = ops.one_hot(ops.argmax(logits, axis=self.axis), num_classes)
        hard = ops.cast(hard, soft.dtype)
        if self.epsilon > 0.0 and num_classes > 1:
            off_value = self.epsilon / float(num_classes - 1)
            on_value = 1.0 - self.epsilon
            hard = hard * ops.cast(on_value, hard.dtype) + (1.0 - hard) * ops.cast(off_value, hard.dtype)
        if training is False:
            return hard
        # Keras/JAX may trace training with `training=None`; keep the train path
        # identical to ordinary softmax unless inference is explicit.
        return soft

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "epsilon": self.epsilon,
                "logit_scale": self.logit_scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="BinaryLayers")
class BinaryAveragePooling2D(keras.layers.Layer):
    """Average pooling over binarized activations, then binarize the pooled result.

    This keeps the activation stream binary after pooling while still using
    the average-pooling window in the forward pass.
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pool_size = _normalize_tuple(pool_size, 2, "pool_size")
        self.strides = _normalize_tuple(strides if strides is not None else pool_size, 2, "strides")
        self.padding = _normalize_padding(padding)
        self.data_format = _normalize_data_format(data_format)

    def call(self, inputs):
        inputs = ops.cast(inputs, self.compute_dtype)
        x_b = binary_sign_ste(inputs)
        pooled = ops.average_pool(
            x_b,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return binary_sign_ste(pooled)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if self.data_format == "channels_last":
            h, w, c = input_shape[1], input_shape[2], input_shape[3]
        else:
            c, h, w = input_shape[1], input_shape[2], input_shape[3]

        def out_dim(size, window, stride, padding):
            if size is None:
                return None
            if padding == "same":
                return math.ceil(size / stride)
            return math.floor((size - window) / stride) + 1

        oh = out_dim(h, self.pool_size[0], self.strides[0], self.padding)
        ow = out_dim(w, self.pool_size[1], self.strides[1], self.padding)
        if self.data_format == "channels_last":
            return (input_shape[0], oh, ow, c)
        return (input_shape[0], c, oh, ow)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
            }
        )
        return config
