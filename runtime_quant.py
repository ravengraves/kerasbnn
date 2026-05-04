from __future__ import annotations

import numpy as np
from keras import ops


FX_SHIFT = 12
FX_ONE = 1 << FX_SHIFT


def _cast_scalar(value, dtype):
    return ops.cast(float(value), dtype)


def _round_to_int_float_ops(x):
    x = ops.cast(x, "float32")
    half = _cast_scalar(0.5, x.dtype)
    return ops.where(
        x >= _cast_scalar(0.0, x.dtype),
        ops.floor(x + half),
        -ops.floor(-x + half),
    )


def _broadcast_channel(values, x, axis: int):
    rank = len(x.shape)
    if axis < 0:
        axis += rank
    shape = [1] * rank
    shape[axis] = int(values.shape[0])
    return ops.reshape(values, tuple(shape))


def quantize_symmetric_range_np(x: np.ndarray, max_abs: float, levels: int) -> np.ndarray:
    qmax = float(levels - 1)
    step = max(max_abs / qmax, 1e-6)
    clipped = np.clip(np.asarray(x, dtype=np.float32), -max_abs, max_abs)
    return np.clip(np.rint(clipped / step) * step, -max_abs, max_abs).astype(np.float32)


def quantize_positive_range_np(x: np.ndarray, max_value: float, levels: int) -> np.ndarray:
    qmax = float(levels - 1)
    step = max(max_value / qmax, 1e-6)
    clipped = np.clip(np.asarray(x, dtype=np.float32), 0.0, max_value)
    return np.clip(np.rint(clipped / step) * step, 0.0, max_value).astype(np.float32)


def fake_quantize_signed_output_np(x: np.ndarray, levels: int, scale: float) -> np.ndarray:
    qmax = levels - 1
    clipped = np.clip(np.asarray(x, dtype=np.float32) / scale, -qmax, qmax)
    return (np.clip(np.rint(clipped), -qmax, qmax) * scale).astype(np.float32)


def quantize_to_fixed_grid_np(x: np.ndarray | float) -> np.ndarray:
    return (np.rint(np.asarray(x, dtype=np.float64) * FX_ONE) / FX_ONE).astype(np.float32)


def quantize_to_fixed_grid_ops(x):
    x = ops.cast(x, "float32")
    return _round_to_int_float_ops(x * _cast_scalar(FX_ONE, x.dtype)) / _cast_scalar(FX_ONE, x.dtype)


def quantize_unit_interval_exact_ops(x, levels=128):
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    x = ops.cast(x, "float32")
    zero = _cast_scalar(0.0, x.dtype)
    levels_f = _cast_scalar(levels, x.dtype)
    max_unit = _cast_scalar((levels - 1) / levels, x.dtype)
    clipped = ops.clip(x, zero, max_unit)
    return ops.floor(clipped * levels_f) / levels_f


def quantize_signed_unit_interval_exact_ops(x, levels=128):
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    x = ops.cast(x, "float32")
    levels_f = _cast_scalar(levels, x.dtype)
    min_unit = _cast_scalar(-1.0, x.dtype)
    max_unit = _cast_scalar((levels - 1) / levels, x.dtype)
    clipped = ops.clip(x, min_unit, max_unit)
    return ops.floor(clipped * levels_f) / levels_f


def quantize_signed_int_grid_ops(x, levels=128):
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    x = ops.cast(x, "float32")
    qmax_f = _cast_scalar(levels - 1, x.dtype)
    clipped = ops.clip(x, _cast_scalar(-1.0, x.dtype), _cast_scalar(1.0, x.dtype))
    return ops.clip(_round_to_int_float_ops(clipped * qmax_f), -qmax_f, qmax_f)


def quantize_signed_unit_grid_ops(x, levels=128):
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    x = ops.cast(x, "float32")
    levels_f = _cast_scalar(levels, x.dtype)
    clipped = ops.clip(x, _cast_scalar(-1.0, x.dtype), _cast_scalar((levels - 1) / levels, x.dtype))
    return ops.floor(clipped * levels_f)


def fake_quantize_signed_output_exact_ops(x, levels=128, scale=1.0):
    levels = int(levels)
    if levels <= 1:
        raise ValueError(f"`levels` must be > 1. Got: {levels}")
    x = ops.cast(x, "float32")
    qmax_f = _cast_scalar(levels - 1, x.dtype)
    scale_f = _cast_scalar(scale, x.dtype)
    scaled = x / scale_f
    clipped = ops.clip(scaled, -qmax_f, qmax_f)
    return _round_to_int_float_ops(clipped) * scale_f


def quantize_symmetric_range_ops(x, max_abs: float, levels: int):
    x = ops.cast(x, "float32")
    qmax = _cast_scalar(levels - 1, x.dtype)
    max_abs_t = _cast_scalar(max_abs, x.dtype)
    step = ops.maximum(max_abs_t / qmax, _cast_scalar(1e-6, x.dtype))
    clipped = ops.clip(x, -max_abs_t, max_abs_t)
    quantized = ops.round(clipped / step) * step
    return ops.clip(quantized, -max_abs_t, max_abs_t)


def quantize_positive_range_ops(x, max_value: float, levels: int):
    x = ops.cast(x, "float32")
    qmax = _cast_scalar(levels - 1, x.dtype)
    max_value_t = _cast_scalar(max_value, x.dtype)
    step = ops.maximum(max_value_t / qmax, _cast_scalar(1e-6, x.dtype))
    clipped = ops.clip(x, 0.0, max_value_t)
    quantized = ops.round(clipped / step) * step
    return ops.clip(quantized, 0.0, max_value_t)


def fold_binary_affine_float_np(layer) -> tuple[np.ndarray, np.ndarray]:
    levels = int(layer.levels)
    scale = np.asarray(layer.norm_scale.numpy(), dtype=np.float32)
    threshold = np.asarray(layer.norm_threshold.numpy(), dtype=np.float32)
    bias = np.asarray(layer.bias.numpy(), dtype=np.float32) if layer.bias is not None else np.zeros_like(scale)
    if layer.quantize_affine_params:
        scale = quantize_symmetric_range_np(scale, 1.0, levels)
        threshold = quantize_symmetric_range_np(threshold, layer.threshold_limit, levels)
        bias = quantize_symmetric_range_np(bias, layer.threshold_limit, levels)

    if layer.use_running_stats:
        mean = np.asarray(layer.moving_mean.numpy(), dtype=np.float32)
        variance = np.asarray(layer.moving_variance.numpy(), dtype=np.float32)
        if layer.quantize_eval_stats or layer.force_full_quant_eval:
            mean = quantize_symmetric_range_np(mean, layer._input_limit, levels)
            inv_std = quantize_positive_range_np(1.0 / np.sqrt(variance + layer.epsilon), layer._inv_std_limit, levels)
        else:
            inv_std = (1.0 / np.sqrt(variance + layer.epsilon)).astype(np.float32)
    else:
        mean = np.zeros_like(scale)
        inv_std = np.ones_like(scale)

    mul = scale * inv_std
    add = scale * (bias - threshold - mean * inv_std)
    return mul.astype(np.float32), add.astype(np.float32)


def fold_binary_affine_float_ops(layer, dtype="float32"):
    levels = int(layer.levels)
    dtype = str(dtype)
    scale = ops.cast(layer.norm_scale, dtype)
    threshold = ops.cast(layer.norm_threshold, dtype)
    if layer.bias is not None:
        bias = ops.cast(layer.bias, dtype)
    else:
        bias = ops.zeros_like(scale)
    if layer.quantize_affine_params:
        scale = quantize_symmetric_range_ops(scale, 1.0, levels)
        threshold = quantize_symmetric_range_ops(threshold, layer.threshold_limit, levels)
        bias = quantize_symmetric_range_ops(bias, layer.threshold_limit, levels)

    if layer.use_running_stats:
        mean = ops.cast(layer.moving_mean, dtype)
        variance = ops.cast(layer.moving_variance, dtype)
        if layer.quantize_eval_stats or layer.force_full_quant_eval:
            mean = quantize_symmetric_range_ops(mean, layer._input_limit, levels)
            inv_std = quantize_positive_range_ops(
                _cast_scalar(1.0, variance.dtype) / ops.sqrt(variance + _cast_scalar(layer.epsilon, variance.dtype)),
                layer._inv_std_limit,
                levels,
            )
        else:
            inv_std = _cast_scalar(1.0, variance.dtype) / ops.sqrt(variance + _cast_scalar(layer.epsilon, variance.dtype))
    else:
        mean = ops.zeros_like(scale)
        inv_std = ops.ones_like(scale)

    mul = scale * inv_std
    add = scale * (bias - threshold - mean * inv_std)
    return ops.cast(mul, dtype), ops.cast(add, dtype)


def fold_rprelu_sign_threshold_np(gamma: np.ndarray, beta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    zeta = np.asarray(zeta, dtype=np.float32)
    threshold = np.empty_like(gamma)
    small_beta = np.abs(beta) < 1e-6
    neg_branch = zeta >= 0.0
    threshold[neg_branch & ~small_beta] = gamma[neg_branch & ~small_beta] - (
        zeta[neg_branch & ~small_beta] / beta[neg_branch & ~small_beta]
    )
    threshold[~neg_branch] = gamma[~neg_branch] - zeta[~neg_branch]
    always_positive = small_beta & (zeta >= 0.0)
    threshold[always_positive] = 0.0
    threshold[small_beta & (zeta < 0.0)] = gamma[small_beta & (zeta < 0.0)] - zeta[small_beta & (zeta < 0.0)]
    return threshold


def fold_rprelu_sign_threshold_ops(gamma, beta, zeta):
    gamma = ops.cast(gamma, "float32")
    beta = ops.cast(beta, "float32")
    zeta = ops.cast(zeta, "float32")
    small_beta = ops.less(ops.absolute(beta), _cast_scalar(1e-6, beta.dtype))
    neg_branch = ops.greater_equal(zeta, _cast_scalar(0.0, zeta.dtype))
    safe_beta = ops.where(small_beta, ops.ones_like(beta), beta)
    neg_threshold = gamma - zeta / safe_beta
    pos_threshold = gamma - zeta
    threshold = ops.where(ops.logical_and(neg_branch, ops.logical_not(small_beta)), neg_threshold, pos_threshold)
    always_positive = ops.logical_and(small_beta, neg_branch)
    return ops.where(always_positive, ops.zeros_like(threshold), threshold)


def quantize_compare_pair_i8_np(scale: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(scale, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    q_scale = np.zeros(scale.shape, dtype=np.int8)
    q_bias = np.zeros(bias.shape, dtype=np.int8)
    for idx in np.ndindex(scale.shape):
        s = float(scale[idx])
        b = float(bias[idx])
        max_abs = max(abs(s), abs(b), 1e-9)
        k = 127.0 / max_abs
        qs = int(np.clip(np.rint(s * k), -127, 127))
        qb = int(np.clip(np.rint(b * k), -127, 127))
        if qs == 0 and qb == 0:
            qb = 1 if b >= 0.0 else -1
        q_scale[idx] = np.int8(qs)
        q_bias[idx] = np.int8(qb)
    return q_scale, q_bias


def quantize_compare_pair_i8_ops(scale, bias):
    scale = ops.cast(scale, "float32")
    bias = ops.cast(bias, "float32")
    max_abs = ops.maximum(
        ops.maximum(ops.absolute(scale), ops.absolute(bias)),
        _cast_scalar(1e-9, scale.dtype),
    )
    gain = _cast_scalar(127.0, scale.dtype) / max_abs
    q_scale = ops.clip(ops.round(scale * gain), _cast_scalar(-127.0, scale.dtype), _cast_scalar(127.0, scale.dtype))
    q_bias = ops.clip(ops.round(bias * gain), _cast_scalar(-127.0, scale.dtype), _cast_scalar(127.0, scale.dtype))
    both_zero = ops.logical_and(
        ops.equal(q_scale, _cast_scalar(0.0, q_scale.dtype)),
        ops.equal(q_bias, _cast_scalar(0.0, q_bias.dtype)),
    )
    q_bias = ops.where(
        both_zero,
        ops.where(
            ops.greater_equal(bias, _cast_scalar(0.0, bias.dtype)),
            _cast_scalar(1.0, q_bias.dtype),
            _cast_scalar(-1.0, q_bias.dtype),
        ),
        q_bias,
    )
    return ops.cast(q_scale, "int8"), ops.cast(q_bias, "int8")


def fold_xnor_activation_compare_np(layer) -> tuple[np.ndarray, np.ndarray]:
    if getattr(layer, "affine_norm", None) is not None:
        mul, add = fold_binary_affine_float_np(layer.affine_norm)
    else:
        channels = int(layer.kernel.shape[-2])
        mul = np.ones((channels,), dtype=np.float32)
        add = np.zeros((channels,), dtype=np.float32)

    if getattr(layer, "use_rprelu", False):
        gamma = np.asarray(layer.rprelu_gamma.numpy(), dtype=np.float32)
        beta = np.asarray(layer.rprelu_beta.numpy(), dtype=np.float32)
        zeta = np.asarray(layer.rprelu_zeta.numpy(), dtype=np.float32)
        threshold = fold_rprelu_sign_threshold_np(gamma, beta, zeta)
        always_positive = (np.abs(beta) < 1e-6) & (zeta >= 0.0)
        cmp_scale = mul.copy()
        cmp_bias = add - threshold
        cmp_scale[always_positive] = 0.0
        cmp_bias[always_positive] = 1.0
    else:
        cmp_scale = mul
        cmp_bias = add

    return quantize_compare_pair_i8_np(cmp_scale, cmp_bias)


def fold_xnor_activation_compare_ops(layer, dtype="float32"):
    dtype = str(dtype)
    if getattr(layer, "affine_norm", None) is not None:
        mul, add = fold_binary_affine_float_ops(layer.affine_norm, dtype=dtype)
    else:
        channels = int(layer.kernel.shape[-2])
        mul = ops.ones((channels,), dtype=dtype)
        add = ops.zeros((channels,), dtype=dtype)

    if getattr(layer, "use_rprelu", False):
        gamma = ops.cast(layer.rprelu_gamma, dtype)
        beta = ops.cast(layer.rprelu_beta, dtype)
        zeta = ops.cast(layer.rprelu_zeta, dtype)
        threshold = fold_rprelu_sign_threshold_ops(gamma, beta, zeta)
        always_positive = ops.logical_and(
            ops.less(ops.absolute(beta), _cast_scalar(1e-6, beta.dtype)),
            ops.greater_equal(zeta, _cast_scalar(0.0, zeta.dtype)),
        )
        cmp_scale = ops.where(always_positive, ops.zeros_like(mul), mul)
        cmp_bias = ops.where(always_positive, ops.ones_like(add), add - threshold)
    else:
        cmp_scale = mul
        cmp_bias = add

    return quantize_compare_pair_i8_ops(cmp_scale, cmp_bias)


def exact_binary_sign_from_compare_ops(x, scale_i8, bias_i8, axis: int = -1):
    x = ops.cast(x, "float32")
    x_q = ops.cast(_round_to_int_float_ops(x * _cast_scalar(FX_ONE, x.dtype)), "int32")
    scale = _broadcast_channel(ops.cast(scale_i8, "int32"), x_q, axis)
    bias = _broadcast_channel(ops.cast(bias_i8, "int32"), x_q, axis)
    score = scale * x_q + bias * ops.cast(FX_ONE, "int32")
    one = _cast_scalar(1.0, x.dtype)
    minus_one = _cast_scalar(-1.0, x.dtype)
    return ops.where(score >= ops.cast(0, "int32"), one, minus_one)


def exact_xnor_alpha_bias_ops(layer, kernel, reduce_axes):
    kernel = ops.cast(kernel, "float32")
    alpha = ops.mean(ops.absolute(kernel), axis=reduce_axes, keepdims=True)
    alpha = quantize_to_fixed_grid_ops(alpha)
    if layer.bias is None:
        bias = None
    else:
        bias = quantize_to_fixed_grid_ops(ops.cast(layer.bias, "float32"))
    return alpha, bias


def exact_xnor_alpha_bias_q_ops(layer, kernel, reduce_axes):
    kernel = ops.cast(kernel, "float32")
    alpha = ops.mean(ops.absolute(kernel), axis=reduce_axes, keepdims=True)
    alpha_q = ops.cast(_round_to_int_float_ops(alpha * _cast_scalar(FX_ONE, alpha.dtype)), "int32")
    if layer.bias is None:
        bias_q = None
    else:
        bias = ops.cast(layer.bias, "float32")
        bias_q = ops.cast(_round_to_int_float_ops(bias * _cast_scalar(FX_ONE, bias.dtype)), "int32")
    return alpha_q, bias_q


def exact_xnor_apply_affine_q_ops(dot, alpha_q, bias_q=None, data_format="channels_last"):
    dot = ops.cast(dot, "float32")
    alpha = ops.cast(alpha_q, "float32") / _cast_scalar(FX_ONE, dot.dtype)
    if len(dot.shape) == 4:
        channels = int(alpha.shape[0] if len(alpha.shape) == 1 else alpha.shape[-1])
        if data_format == "channels_last":
            alpha = ops.reshape(alpha, (1, 1, 1, channels))
        else:
            alpha = ops.reshape(alpha, (1, channels, 1, 1))
    outputs = dot * alpha
    if bias_q is not None:
        bias = ops.cast(bias_q, "float32") / _cast_scalar(FX_ONE, dot.dtype)
        if len(dot.shape) == 4:
            channels = int(bias.shape[0] if len(bias.shape) == 1 else bias.shape[-1])
            if data_format == "channels_last":
                bias = ops.reshape(bias, (1, 1, 1, channels))
            else:
                bias = ops.reshape(bias, (1, channels, 1, 1))
        outputs = outputs + bias
    return quantize_to_fixed_grid_ops(outputs)


__all__ = [
    "FX_SHIFT",
    "FX_ONE",
    "exact_xnor_alpha_bias_q_ops",
    "exact_xnor_apply_affine_q_ops",
    "exact_binary_sign_from_compare_ops",
    "exact_xnor_alpha_bias_ops",
    "fake_quantize_signed_output_np",
    "fold_binary_affine_float_np",
    "fold_binary_affine_float_ops",
    "fold_rprelu_sign_threshold_np",
    "fold_rprelu_sign_threshold_ops",
    "fold_xnor_activation_compare_np",
    "fold_xnor_activation_compare_ops",
    "quantize_compare_pair_i8_np",
    "quantize_compare_pair_i8_ops",
    "quantize_positive_range_np",
    "quantize_positive_range_ops",
    "quantize_symmetric_range_np",
    "quantize_symmetric_range_ops",
    "quantize_to_fixed_grid_np",
    "quantize_to_fixed_grid_ops",
]
