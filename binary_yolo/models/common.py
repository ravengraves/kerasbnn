from __future__ import annotations

from dataclasses import dataclass
import math

import keras
from keras import layers

from kerasbnn import (
    BinaryAffineNorm,
    BinaryDropcut,
    QuantizedActivation,
    QuantizedConv2D,
    QuantizedDense,
    QuantizedInput,
    ThermometerBinaryInput,
    XNORBinaryConv2D,
    quantized_first_conv,
)


FIXED_ACTIVATION_LEVELS = 128
FIXED_OUTPUT_LEVELS = 128
FIXED_BN_MOMENTUM = 0.8
FIXED_NORM_GAMMA_INIT = 0.25
FIXED_HEAD_PRE_FILTERS = 256
FIXED_HEAD_CHANNELS = 64
FIXED_HEAD_PROJ_LEVELS = 128
FIXED_OBJ_CLS_HIDDEN = 128
FIXED_OBJ_REMAP_HIDDEN = 64
FIXED_BBOX_OUTPUT_SCALE = 2.0
FIXED_OBJ_OUTPUT_SCALE = 64.0
FIXED_CLS_OUTPUT_SCALE = 32.0
FIXED_EXTRA_HEAD_BLOCKS = 4


@dataclass(frozen=True)
class BinaryInputConfig:
    thermo_levels: int = 4
    input_levels: int = 16
    input_min_value: float = 0.0
    input_max_value: float = 1.0
    input_signed: bool = False
    quantized_input: bool = False
    input_dropcut: float = 0.0
    input_dropcut_size: tuple[float, float] = (0.05, 0.30)
    input_dropcut_method: str = "both"
    input_dropcut_boxes: int = 1


@keras.saving.register_keras_serializable(package="BinaryYOLO")
class SpaceToDepth(layers.Layer):
    def __init__(self, block_size: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError(f"`block_size` must be > 0. Got: {block_size!r}")

    def call(self, inputs):
        shape = keras.ops.shape(inputs)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        block = self.block_size
        x = keras.ops.reshape(
            inputs,
            (batch, height // block, block, width // block, block, channels),
        )
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return keras.ops.reshape(
            x,
            (batch, height // block, width // block, channels * block * block),
        )

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        block = self.block_size
        out_height = None if height is None else height // block
        out_width = None if width is None else width // block
        out_channels = None if channels is None else channels * block * block
        return (batch, out_height, out_width, out_channels)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config


def _scaled_channels(base: int, mult: float) -> int:
    value = max(8, int(math.ceil(float(base) * float(mult))))
    return int(((value + 7) // 8) * 8)


def make_binary_input(image, cfg: BinaryInputConfig):
    if cfg.quantized_input:
        x = QuantizedInput(
            levels=cfg.input_levels,
            min_value=cfg.input_min_value,
            max_value=cfg.input_max_value,
            signed=cfg.input_signed,
            data_format="channels_last",
            name="quantized_input",
        )(image)
    else:
        x = ThermometerBinaryInput(
            levels=cfg.thermo_levels,
            min_value=cfg.input_min_value,
            max_value=cfg.input_max_value,
            data_format="channels_last",
            name="binary_input",
        )(image)
    if cfg.input_dropcut > 0.0:
        x = BinaryDropcut(
            dropcut_probability=cfg.input_dropcut,
            size_range=cfg.input_dropcut_size,
            num_boxes=cfg.input_dropcut_boxes,
            method=cfg.input_dropcut_method,
            data_format="channels_last",
            name="input_dropcut",
        )(x)
    return x


def xnor_conv(x, filters: int, kernel_size: int, *, name: str, strides: int = 1):
    return XNORBinaryConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=FIXED_ACTIVATION_LEVELS,
        momentum=FIXED_BN_MOMENTUM,
        epsilon=1e-4,
        kernel_initializer="glorot_uniform",
        activation_surrogate="triangular",
        activation_surrogate_gain=1.0,
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        weight_surrogate_gain=1.0,
        norm_gamma_init=FIXED_NORM_GAMMA_INIT,
        bn_quantize_input=False,
        bn_quantize_centered=False,
        bn_quantize_normalized=False,
        bn_quantize_output=False,
        bn_quantize_eval_stats=False,
        bn_eval_use_batch_stats=False,
        bn_force_full_quant_eval=False,
        bn_quantize_affine_params=True,
        use_rprelu=True,
        rprelu_init_beta=0.25,
        rprelu_init_gamma=0.0,
        rprelu_init_zeta=0.0,
        name=name,
    )(x)


def qstem_first_conv(x, filters: int, kernel_size: int, *, input_levels: int, name: str, strides: int = 1):
    return quantized_first_conv(
        x,
        filters,
        kernel_size,
        levels=input_levels,
        output_levels=0,
        strides=strides,
        name=name,
    )


def stem_conv(x, filters: int, kernel_size: int, *, cfg: BinaryInputConfig, name: str, strides: int = 1):
    if cfg.quantized_input:
        return qstem_first_conv(
            x,
            filters,
            kernel_size,
            input_levels=cfg.input_levels,
            name=name,
            strides=strides,
        )
    return xnor_conv(x, filters, kernel_size, name=name, strides=strides)


def xnor_fire_block(x, squeeze_filters: int, expand1_filters: int, expand3_filters: int, *, name: str):
    squeeze = xnor_conv(x, squeeze_filters, 1, name=f"{name}_sq")
    expand1 = xnor_conv(squeeze, expand1_filters, 1, name=f"{name}_e1")
    expand3 = xnor_conv(squeeze, expand3_filters, 3, name=f"{name}_e3")
    return layers.Concatenate(name=f"{name}_out")([expand1, expand3])


def head_output_prep(x, *, name: str):
    levels = max(FIXED_HEAD_PROJ_LEVELS, FIXED_OUTPUT_LEVELS)
    x = BinaryAffineNorm(
        levels=levels,
        threshold_limit=float((levels - 1) / levels),
        use_bias=True,
        output_mode="raw",
        feature_axis=-1,
        data_format="channels_last",
        momentum=FIXED_BN_MOMENTUM,
        epsilon=1e-4,
        use_running_stats=True,
        quantize_input=False,
        quantize_centered=False,
        quantize_normalized=False,
        quantize_output=False,
        quantize_eval_stats=False,
        eval_use_batch_stats=False,
        force_full_quant_eval=False,
        quantize_affine_params=True,
        name=f"{name}_norm",
    )(x)
    return QuantizedActivation(levels=levels, signed=True, name=f"{name}_quant")(x)


def prediction_head(x, num_classes: int, *, name: str):
    x = xnor_conv(x, FIXED_HEAD_PRE_FILTERS, 3, name=f"{name}_pre")
    x = xnor_conv(x, 3 * FIXED_HEAD_CHANNELS, 1, name=f"{name}_proj")
    x = QuantizedActivation(levels=FIXED_HEAD_PROJ_LEVELS, signed=True, name=f"{name}_proj_quant")(x)
    h, w = int(x.shape[1]), int(x.shape[2])
    x = layers.Reshape((h, w, 3, FIXED_HEAD_CHANNELS), name=f"{name}_reshape")(x)
    bbox_x = head_output_prep(x, name=f"{name}_bbox_prep")

    bbox = QuantizedDense(
        4,
        levels=FIXED_OUTPUT_LEVELS,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=True,
        integer_kernel=True,
        output_scale=FIXED_BBOX_OUTPUT_SCALE,
        quantize_bias=True,
        name=f"{name}_bbox",
    )(bbox_x)

    obj_feat = QuantizedDense(
        FIXED_OBJ_CLS_HIDDEN,
        levels=FIXED_HEAD_PROJ_LEVELS,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=False,
        name=f"{name}_obj_hidden",
    )(x)
    obj_feat = QuantizedActivation(levels=FIXED_HEAD_PROJ_LEVELS, signed=True, name=f"{name}_obj_hidden_act")(obj_feat)
    obj_feat = QuantizedDense(
        FIXED_OBJ_REMAP_HIDDEN,
        levels=FIXED_HEAD_PROJ_LEVELS,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=False,
        name=f"{name}_obj_remap",
    )(obj_feat)
    obj_feat = QuantizedActivation(levels=FIXED_HEAD_PROJ_LEVELS, signed=True, name=f"{name}_obj_remap_act")(obj_feat)
    obj_feat = head_output_prep(obj_feat, name=f"{name}_obj_prep")

    cls_feat = QuantizedDense(
        FIXED_OBJ_CLS_HIDDEN,
        levels=FIXED_HEAD_PROJ_LEVELS,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=False,
        name=f"{name}_cls_hidden",
    )(x)
    cls_feat = QuantizedActivation(levels=FIXED_HEAD_PROJ_LEVELS, signed=True, name=f"{name}_cls_hidden_act")(cls_feat)
    cls_feat = head_output_prep(cls_feat, name=f"{name}_cls_prep")

    obj = QuantizedDense(
        1,
        levels=FIXED_OUTPUT_LEVELS,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=True,
        integer_kernel=True,
        output_scale=FIXED_OBJ_OUTPUT_SCALE,
        quantize_bias=True,
        name=f"{name}_obj",
    )(obj_feat)
    cls = QuantizedDense(
        num_classes,
        levels=FIXED_OUTPUT_LEVELS,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=True,
        integer_kernel=True,
        output_scale=FIXED_CLS_OUTPUT_SCALE,
        quantize_bias=True,
        name=f"{name}_cls",
    )(cls_feat)
    pred = layers.Concatenate(axis=-1, name=f"{name}_concat")([bbox, obj, cls])
    return layers.Activation("linear", dtype="float32", name=name)(pred)
