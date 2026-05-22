from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
from keras import layers, ops

from kerasbnn import (
    BinaryAffineNorm,
    BinaryConv2D,
    BinaryDropcut,
    BinaryInput,
    QuantizedActivation,
    QuantizedConv2D,
    QuantizedDense,
)


@keras.saving.register_keras_serializable(package="BinaryYOLO")
class SpaceToDepth(layers.Layer):
    def __init__(self, block_size: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError(f"`block_size` must be > 0. Got: {block_size!r}")

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        block = self.block_size
        x = ops.reshape(
            inputs,
            (batch, height // block, block, width // block, block, channels),
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return ops.reshape(
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


@dataclass(frozen=True)
class TinierBinaryConfig:
    thermo_levels: int = 4
    input_min_value: float = 0.0
    input_max_value: float = 1.0
    float_input: bool = False
    input_dropcut: float = 0.0
    input_dropcut_size: tuple[float, float] = (0.05, 0.20)
    input_dropcut_method: str = "both"
    input_dropcut_boxes: int = 1
    activation_levels: int = 128
    output_levels: int = 128
    binary_batchnorm: bool = True
    use_bn_in_fire: bool = True
    bn_momentum: float = 0.95
    bn_quantize_input: bool = True
    bn_quantize_centered: bool = False
    bn_quantize_normalized: bool = False
    bn_quantize_output: bool = True
    bn_quantize_eval_stats: bool = True
    head_pre_filters: int = 256
    head_channels: int = 64
    head_proj_levels: int = 128
    obj_cls_hidden: int = 128
    obj_remap_hidden: int = 64
    late_quantized_from: int = 0
    late_conv_levels: int = 128
    late_act_levels: int = 16
    bbox_output_scale: float = 2.0
    obj_output_scale: float = 64.0
    cls_output_scale: float = 32.0
    logit_scale: float = 32.0
    obj_bias_init: float = 0.0


def _float_conv_bn_lrelu(
    x,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    *,
    use_bn: bool = True,
    name: str,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=not use_bn,
        name=f"{name}_conv",
    )(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.LeakyReLU(negative_slope=0.1, name=f"{name}_lrelu")(x)


def _binary_conv(
    x,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    *,
    cfg: TinierBinaryConfig,
    use_bn: bool | None = None,
    raw_output: bool = False,
    name: str,
):
    if use_bn is None:
        use_bn = cfg.binary_batchnorm
    return BinaryConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        score_mode="xnor_popcount",
        input_threshold=0.0,
        output_mode="raw" if raw_output else "bipolar",
        output_levels=cfg.output_levels if raw_output else cfg.activation_levels,
        use_batchnorm=use_bn,
        bn_momentum=cfg.bn_momentum,
        bn_quantize_input=cfg.bn_quantize_input,
        bn_quantize_centered=cfg.bn_quantize_centered,
        bn_quantize_normalized=cfg.bn_quantize_normalized,
        bn_quantize_output=cfg.bn_quantize_output,
        bn_quantize_eval_stats=cfg.bn_quantize_eval_stats,
        name=name,
    )(x)


def _float_fire_block(
    x,
    squeeze_filters: int,
    expand1_filters: int,
    expand3_filters: int,
    *,
    use_bn_in_fire: bool,
    name: str,
):
    squeeze = _float_conv_bn_lrelu(
        x,
        squeeze_filters,
        1,
        use_bn=use_bn_in_fire,
        name=f"{name}_sq",
    )
    expand1 = _float_conv_bn_lrelu(
        squeeze,
        expand1_filters,
        1,
        use_bn=use_bn_in_fire,
        name=f"{name}_e1",
    )
    expand3 = _float_conv_bn_lrelu(
        squeeze,
        expand3_filters,
        3,
        use_bn=use_bn_in_fire,
        name=f"{name}_e3",
    )
    return layers.Concatenate(name=f"{name}_out")([expand1, expand3])


def _binary_fire_block(
    x,
    squeeze_filters: int,
    expand1_filters: int,
    expand3_filters: int,
    *,
    cfg: TinierBinaryConfig,
    name: str,
):
    squeeze = _binary_conv(
        x,
        squeeze_filters,
        1,
        cfg=cfg,
        use_bn=cfg.use_bn_in_fire,
        name=f"{name}_sq",
    )
    expand1 = _binary_conv(
        squeeze,
        expand1_filters,
        1,
        cfg=cfg,
        use_bn=cfg.use_bn_in_fire,
        name=f"{name}_e1",
    )
    expand3 = _binary_conv(
        squeeze,
        expand3_filters,
        3,
        cfg=cfg,
        use_bn=cfg.use_bn_in_fire,
        name=f"{name}_e3",
    )
    return layers.Concatenate(name=f"{name}_out")([expand1, expand3])


def _quantized_conv(
    x,
    filters: int,
    kernel_size: int,
    *,
    cfg: TinierBinaryConfig,
    strides: int = 1,
    name: str,
):
    x = QuantizedConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        data_format="channels_last",
        levels=cfg.late_conv_levels,
        binarize_inputs=False,
        integer_output=False,
        integer_kernel=True,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = BinaryAffineNorm(
        levels=cfg.output_levels,
        output_mode="raw",
        data_format="channels_last",
        use_bias=True,
        momentum=cfg.bn_momentum,
        quantize_input=False,
        quantize_centered=cfg.bn_quantize_centered,
        quantize_normalized=cfg.bn_quantize_normalized,
        quantize_output=False,
        quantize_eval_stats=cfg.bn_quantize_eval_stats,
        name=f"{name}_affine",
    )(x)
    return QuantizedActivation(levels=cfg.late_act_levels, signed=True, name=f"{name}_act")(x)


def _quantized_fire_block(
    x,
    squeeze_filters: int,
    expand1_filters: int,
    expand3_filters: int,
    *,
    cfg: TinierBinaryConfig,
    name: str,
):
    squeeze = _quantized_conv(
        x,
        squeeze_filters,
        1,
        cfg=cfg,
        name=f"{name}_sq",
    )
    expand1 = _quantized_conv(
        squeeze,
        expand1_filters,
        1,
        cfg=cfg,
        name=f"{name}_e1",
    )
    expand3 = _quantized_conv(
        squeeze,
        expand3_filters,
        3,
        cfg=cfg,
        name=f"{name}_e3",
    )
    return layers.Concatenate(name=f"{name}_out")([expand1, expand3])


def _prediction_conv_float(
    x,
    num_classes: int,
    name: str,
    *,
    head_pre_filters: int = 256,
    head_channels: int = 64,
    obj_cls_hidden: int = 128,
    obj_remap_hidden: int = 64,
    obj_bias_init: float = 0.0,
):
    if head_pre_filters <= 0 or head_channels <= 0:
        filters = 3 * (num_classes + 5)
        bias_init = np.zeros((filters,), np.float32)
        if obj_bias_init != 0.0:
            step = num_classes + 5
            for anchor_idx in range(3):
                bias_init[anchor_idx * step + 4] = float(obj_bias_init)
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation=None,
            bias_initializer=keras.initializers.Constant(bias_init),
            name=f"{name}_conv",
        )(x)
        h, w = int(x.shape[1]), int(x.shape[2])
        x = layers.Reshape((h, w, 3, num_classes + 5), name=f"{name}_reshape")(x)
        return layers.Activation("linear", dtype="float32", name=name)(x)

    x = _float_conv_bn_lrelu(x, head_pre_filters, 3, use_bn=True, name=f"{name}_pre")
    x = _float_conv_bn_lrelu(x, 3 * head_channels, 1, use_bn=True, name=f"{name}_proj")
    h, w = int(x.shape[1]), int(x.shape[2])
    x = layers.Reshape((h, w, 3, head_channels), name=f"{name}_reshape")(x)

    bbox = layers.Dense(4, name=f"{name}_bbox")(x)

    obj_feat = layers.Dense(obj_cls_hidden, name=f"{name}_obj_hidden")(x)
    obj_feat = layers.LeakyReLU(negative_slope=0.1, name=f"{name}_obj_hidden_lrelu")(obj_feat)
    if obj_remap_hidden > 0:
        obj_feat = layers.Dense(obj_remap_hidden, name=f"{name}_obj_remap")(obj_feat)
        obj_feat = layers.LeakyReLU(negative_slope=0.1, name=f"{name}_obj_remap_lrelu")(obj_feat)

    cls_feat = layers.Dense(obj_cls_hidden, name=f"{name}_cls_hidden")(x)
    cls_feat = layers.LeakyReLU(negative_slope=0.1, name=f"{name}_cls_hidden_lrelu")(cls_feat)

    obj = layers.Dense(
        1,
        bias_initializer=keras.initializers.Constant(float(obj_bias_init)),
        name=f"{name}_obj",
    )(obj_feat)
    cls = layers.Dense(num_classes, name=f"{name}_cls")(cls_feat)

    x = layers.Concatenate(axis=-1, name=f"{name}_concat")([bbox, obj, cls])
    return layers.Activation("linear", dtype="float32", name=name)(x)


def _prediction_conv_binary(x, num_classes: int, cfg: TinierBinaryConfig, name: str):
    x = _binary_conv(
        x,
        cfg.head_pre_filters,
        3,
        cfg=cfg,
        use_bn=cfg.binary_batchnorm,
        name=f"{name}_pre",
    )
    x = _binary_conv(
        x,
        3 * cfg.head_channels,
        1,
        cfg=cfg,
        use_bn=False,
        raw_output=True,
        name=f"{name}_proj",
    )
    if cfg.binary_batchnorm:
        x = BinaryAffineNorm(
            levels=cfg.output_levels,
            output_mode="raw",
            data_format="channels_last",
            use_bias=True,
            momentum=cfg.bn_momentum,
            quantize_input=cfg.bn_quantize_input,
            quantize_centered=cfg.bn_quantize_centered,
            quantize_normalized=cfg.bn_quantize_normalized,
            quantize_output=cfg.bn_quantize_output,
            quantize_eval_stats=cfg.bn_quantize_eval_stats,
            name=f"{name}_proj_bn",
        )(x)
    x = QuantizedActivation(levels=cfg.head_proj_levels, signed=True, name=f"{name}_proj_quant")(x)
    h, w = int(x.shape[1]), int(x.shape[2])
    x = layers.Reshape((h, w, 3, cfg.head_channels), name=f"{name}_reshape")(x)

    bbox = QuantizedDense(
        4,
        levels=cfg.output_levels,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=True,
        integer_kernel=True,
        output_scale=cfg.bbox_output_scale,
        quantize_bias=True,
        name=f"{name}_bbox",
    )(x)

    obj_feat = QuantizedDense(
        cfg.obj_cls_hidden,
        levels=cfg.head_proj_levels,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=False,
        name=f"{name}_obj_hidden",
    )(x)
    obj_feat = QuantizedActivation(levels=cfg.head_proj_levels, signed=True, name=f"{name}_obj_hidden_act")(obj_feat)
    if cfg.obj_remap_hidden:
        obj_feat = QuantizedDense(
            cfg.obj_remap_hidden,
            levels=cfg.head_proj_levels,
            input_threshold=0.0,
            binarize_inputs=False,
            integer_output=False,
            name=f"{name}_obj_remap",
        )(obj_feat)
        obj_feat = QuantizedActivation(levels=cfg.head_proj_levels, signed=True, name=f"{name}_obj_remap_act")(obj_feat)
    cls_feat = QuantizedDense(
        cfg.obj_cls_hidden,
        levels=cfg.head_proj_levels,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=False,
        name=f"{name}_cls_hidden",
    )(x)
    cls_feat = QuantizedActivation(levels=cfg.head_proj_levels, signed=True, name=f"{name}_cls_hidden_act")(cls_feat)

    obj = QuantizedDense(
        1,
        levels=cfg.output_levels,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=True,
        integer_kernel=True,
        output_scale=cfg.obj_output_scale,
        quantize_bias=True,
        name=f"{name}_obj",
    )(obj_feat)
    cls = QuantizedDense(
        num_classes,
        levels=cfg.output_levels,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=True,
        integer_kernel=True,
        output_scale=cfg.cls_output_scale,
        quantize_bias=True,
        name=f"{name}_cls",
    )(cls_feat)
    x = layers.Concatenate(axis=-1, name=f"{name}_concat")([bbox, obj, cls])
    return layers.Activation("linear", dtype="float32", name=name)(x)


def build_float_tinier_yolov3(
    num_classes: int,
    input_shape: tuple[int, int, int] = (416, 416, 3),
    *,
    use_bn_in_fire: bool = False,
    head_pre_filters: int = 256,
    head_channels: int = 64,
    obj_cls_hidden: int = 128,
    obj_remap_hidden: int = 64,
    obj_bias_init: float = 0.0,
):
    image = keras.Input(shape=input_shape, name="image")

    x = _float_conv_bn_lrelu(image, 16, 3, name="conv1")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool1")(x)

    x = _float_conv_bn_lrelu(x, 32, 3, name="conv2")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool2")(x)

    x = _float_conv_bn_lrelu(x, 64, 3, name="conv3")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool3")(x)

    x = _float_conv_bn_lrelu(x, 128, 3, name="conv4")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool4")(x)

    c5 = _float_conv_bn_lrelu(x, 256, 3, name="conv5")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool5")(c5)

    dense = x
    for idx in range(1, 5):
        fire = _float_fire_block(
            dense,
            64,
            256,
            256,
            use_bn_in_fire=use_bn_in_fire,
            name=f"fire{idx}",
        )
        dense = layers.Concatenate(name=f"dense{idx}")([fire, dense])
    f5 = _float_fire_block(
        dense,
        64,
        256,
        256,
        use_bn_in_fire=use_bn_in_fire,
        name="fire5",
    )

    passthrough = _float_conv_bn_lrelu(c5, 64, 1, name="passthrough_reduce")
    passthrough = SpaceToDepth(block_size=2, name="passthrough_reorg")(passthrough)
    head1_in = layers.Concatenate(name="head1_in")([passthrough, f5])

    f6 = _float_fire_block(
        head1_in,
        64,
        256,
        256,
        use_bn_in_fire=use_bn_in_fire,
        name="fire6",
    )
    head1_feat = layers.Concatenate(name="head1_dense")([f6, head1_in])
    pred_13 = _prediction_conv_float(
        head1_feat,
        num_classes,
        "pred_13",
        head_pre_filters=head_pre_filters,
        head_channels=head_channels,
        obj_cls_hidden=obj_cls_hidden,
        obj_remap_hidden=obj_remap_hidden,
        obj_bias_init=obj_bias_init,
    )

    up = _float_conv_bn_lrelu(f6, 128, 1, name="up_reduce")
    up = layers.UpSampling2D(size=2, interpolation="nearest", name="upsample")(up)
    head2_in = layers.Concatenate(name="head2_in")([up, c5])

    f7 = _float_fire_block(
        head2_in,
        32,
        128,
        128,
        use_bn_in_fire=use_bn_in_fire,
        name="fire7",
    )
    dense7 = layers.Concatenate(name="dense7")([f7, head2_in])
    f8 = _float_fire_block(
        dense7,
        32,
        128,
        128,
        use_bn_in_fire=use_bn_in_fire,
        name="fire8",
    )
    head2_feat = layers.Concatenate(name="dense8")([f8, dense7])
    pred_26 = _prediction_conv_float(
        head2_feat,
        num_classes,
        "pred_26",
        head_pre_filters=head_pre_filters,
        head_channels=head_channels,
        obj_cls_hidden=obj_cls_hidden,
        obj_remap_hidden=obj_remap_hidden,
        obj_bias_init=obj_bias_init,
    )

    return keras.Model(
        image,
        {"pred_13": pred_13, "pred_26": pred_26},
        name="tinier_yolov3_float",
    )


def build_binary_tinier_yolov3(
    num_classes: int,
    input_shape: tuple[int, int, int] = (416, 416, 3),
    *,
    cfg: TinierBinaryConfig | None = None,
):
    cfg = cfg or TinierBinaryConfig()
    image = keras.Input(shape=input_shape, name="image")

    if cfg.float_input:
        x = layers.Rescaling(2.0, offset=-1.0, name="float_input_scale")(image)
    else:
        x = BinaryInput(
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

    x = _binary_conv(x, 16, 3, cfg=cfg, name="conv1")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool1")(x)

    x = _binary_conv(x, 32, 3, cfg=cfg, name="conv2")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool2")(x)

    x = _binary_conv(x, 64, 3, cfg=cfg, name="conv3")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool3")(x)

    x = _binary_conv(x, 128, 3, cfg=cfg, name="conv4")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool4")(x)

    c5 = _binary_conv(x, 256, 3, cfg=cfg, name="conv5")
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="pool5")(c5)

    def fire_block(idx, x, sq, e1, e3):
        if cfg.late_quantized_from and idx >= cfg.late_quantized_from:
            return _quantized_fire_block(x, sq, e1, e3, cfg=cfg, name=f"fire{idx}")
        return _binary_fire_block(x, sq, e1, e3, cfg=cfg, name=f"fire{idx}")

    dense = x
    for idx in range(1, 5):
        fire = fire_block(idx, dense, 64, 256, 256)
        dense = layers.Concatenate(name=f"dense{idx}")([fire, dense])
    f5 = fire_block(5, dense, 64, 256, 256)

    if cfg.late_quantized_from and cfg.late_quantized_from <= 6:
        passthrough = _quantized_conv(c5, 64, 1, cfg=cfg, name="passthrough_reduce")
    else:
        passthrough = _binary_conv(c5, 64, 1, cfg=cfg, name="passthrough_reduce")
    passthrough = SpaceToDepth(block_size=2, name="passthrough_reorg")(passthrough)
    head1_in = layers.Concatenate(name="head1_in")([passthrough, f5])

    f6 = fire_block(6, head1_in, 64, 256, 256)
    head1_feat = layers.Concatenate(name="head1_dense")([f6, head1_in])
    pred_13 = _prediction_conv_binary(head1_feat, num_classes, cfg, "pred_13")

    if cfg.late_quantized_from and cfg.late_quantized_from <= 7:
        up = _quantized_conv(f6, 128, 1, cfg=cfg, name="up_reduce")
    else:
        up = _binary_conv(f6, 128, 1, cfg=cfg, name="up_reduce")
    up = layers.UpSampling2D(size=2, interpolation="nearest", name="upsample")(up)
    head2_in = layers.Concatenate(name="head2_in")([up, c5])

    f7 = fire_block(7, head2_in, 32, 128, 128)
    dense7 = layers.Concatenate(name="dense7")([f7, head2_in])
    f8 = fire_block(8, dense7, 32, 128, 128)
    head2_feat = layers.Concatenate(name="dense8")([f8, dense7])
    pred_26 = _prediction_conv_binary(head2_feat, num_classes, cfg, "pred_26")

    return keras.Model(
        image,
        {"pred_13": pred_13, "pred_26": pred_26},
        name="tinier_yolov3_binary",
    )
