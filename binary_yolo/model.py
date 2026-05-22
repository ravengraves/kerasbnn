from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import layers

from kerasbnn import (
    BinaryConv2D,
    BinaryDense,
    BinaryDropcut,
    BinaryInput,
    QuantizedActivation,
    QuantizedDense,
)


keras.config.set_dtype_policy("mixed_float16")


@dataclass(frozen=True)
class ModelConfig:
    input_size: int | tuple[int, int] = 256
    thermo_levels: int = 4
    input_min_value: float = 0.0
    input_max_value: float = 1.0
    float_input: bool = False
    activation_levels: int = 128
    # Deprecated architecture knobs retained for compatibility with old
    # saved configs/search code. The live model now uses fixed power-of-two
    # channels and 4x4 kernels for the production path.
    width_scale: float = 1.0
    channel_round_mode: str = "power2"
    main_kernel_size: int = 4
    stem_filters: int = 32
    stage7_filters: int = 1024
    input_dropcut: float = 0.1
    input_dropcut_boxes: int = 1
    input_dropcut_size: tuple[float, float] = (0.05, 0.20)
    input_dropcut_method: str = "both"
    input_dropcut2: float = 0.0
    input_dropcut2_method: str = "noise"
    feature_dropcut: float = 0.0
    feature_dropcut_boxes: int = 2
    feature_dropcut_size: tuple[float, float] = (0.02, 0.06)
    feature_dropcut_stage: str = "none"
    dropout: float = 0.0
    dropout_stage: str = "none"
    float_backbone: bool = False
    float_neck: bool = False
    float_head_convs: bool = False
    float_batchnorm: bool = False
    binary_batchnorm: bool = True
    bn_momentum: float = 0.95
    bn_quantize_input: bool = True
    bn_quantize_centered: bool = False
    bn_quantize_normalized: bool = False
    bn_quantize_output: bool = True
    bn_quantize_eval_stats: bool = True
    head_pre_filters: int = 256
    head_channels: int = 64
    head_proj_raw: bool = True
    head_proj_levels: int = 8
    obj_cls_hidden: int = 128
    obj_remap_hidden: int = 64
    share_obj_cls_hidden: bool = False
    float_obj_cls_hidden: bool = False
    float_output_heads: bool = False
    quant_levels: int = 8
    output_levels: int = 128
    bbox_output_scale: float = 2.0
    obj_output_scale: float = 64.0
    cls_output_scale: float = 32.0


# Fixed production architecture:
# - power-of-two channels for hardware friendliness
# - 4x4 kernels on the backbone/neck/head conv path
BACKBONE_FILTERS = (32, 64, 128, 256, 512, 1024, 1024, 512)
NECK_P5_FILTERS = 256
NECK_P4_FILTERS = 256
NECK_P3_FILTERS = 128
HEAD_PRE_FILTERS = 256
HEAD_CHANNELS = 64
OBJ_CLS_HIDDEN = 128
OBJ_REMAP_HIDDEN = 64
MAIN_KERNEL_SIZE = 4


def _input_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, tuple):
        return int(size[0]), int(size[1])
    return int(size), int(size)


def _bconv(
    x,
    filters: int,
    k: int,
    s: int,
    name: str,
    levels: int = 128,
    threshold: float = 0.0,
    raw_output: bool = False,
    batchnorm: bool = False,
    bn_quantize_input: bool = False,
    bn_quantize_centered: bool = False,
    bn_quantize_normalized: bool = True,
    bn_quantize_output: bool = True,
    bn_quantize_eval_stats: bool = True,
    bn_momentum: float = 0.95,
):
    return BinaryConv2D(
        filters,
        k,
        strides=s,
        padding="same",
        score_mode="xnor_popcount",
        input_threshold=threshold,
        output_mode="raw" if raw_output else "bipolar",
        output_levels=levels,
        use_batchnorm=batchnorm,
        bn_momentum=bn_momentum,
        bn_quantize_input=bn_quantize_input,
        bn_quantize_centered=bn_quantize_centered,
        bn_quantize_normalized=bn_quantize_normalized,
        bn_quantize_output=bn_quantize_output,
        bn_quantize_eval_stats=bn_quantize_eval_stats,
        name=f"{name}_conv",
    )(x)


def _fconv(x, filters: int, k: int, s: int, name: str, activation: str | None = "relu", batchnorm: bool = False):
    x = layers.Conv2D(
        filters,
        k,
        strides=s,
        padding="same",
        use_bias=not batchnorm,
        activation=None,
        name=f"{name}_conv",
    )(x)
    if batchnorm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    if activation:
        x = layers.Activation(activation, name=f"{name}_act")(x)
    return x


def _regularize(x, name: str, cfg: ModelConfig):
    if cfg.feature_dropcut and cfg.feature_dropcut_stage == name:
        x = BinaryDropcut(
            dropcut_probability=cfg.feature_dropcut,
            size_range=cfg.feature_dropcut_size,
            num_boxes=cfg.feature_dropcut_boxes,
            method="both",
            name=f"{name}_dropcut",
        )(x)
    if cfg.dropout and cfg.dropout_stage in {name, "all_heads"}:
        x = layers.Dropout(cfg.dropout, name=f"{name}_dropout")(x)
    return x


def _bdense_act(x, units: int, name: str, cfg: ModelConfig):
    return BinaryDense(
        units,
        score_mode="xnor_popcount",
        input_threshold=0.0,
        output_mode="bipolar",
        output_levels=cfg.activation_levels,
        use_batchnorm=cfg.binary_batchnorm,
        bn_momentum=cfg.bn_momentum,
        bn_quantize_input=cfg.bn_quantize_input,
        bn_quantize_centered=cfg.bn_quantize_centered,
        bn_quantize_normalized=cfg.bn_quantize_normalized,
        bn_quantize_output=cfg.bn_quantize_output,
        bn_quantize_eval_stats=cfg.bn_quantize_eval_stats,
        name=f"{name}_dense",
    )(x)


def _qdense_act(x, units: int, name: str, cfg: ModelConfig):
    x = QuantizedDense(
        units,
        levels=cfg.quant_levels,
        input_threshold=0.0,
        binarize_inputs=False,
        integer_output=False,
        name=f"{name}_dense",
    )(x)
    return QuantizedActivation(levels=cfg.quant_levels, signed=True, name=f"{name}_act")(x)


def _fdense_act(x, units: int, name: str, batchnorm: bool = False):
    x = layers.Dense(units, use_bias=not batchnorm, name=f"{name}_dense")(x)
    if batchnorm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_act")(x)


def _head(x, name: str, classes: int, cfg: ModelConfig):
    if cfg.float_head_convs:
        x = _fconv(x, HEAD_PRE_FILTERS, MAIN_KERNEL_SIZE, 1, f"{name}_pre", batchnorm=cfg.float_batchnorm)
    else:
        x = _bconv(
            x,
            HEAD_PRE_FILTERS,
            MAIN_KERNEL_SIZE,
            1,
            f"{name}_pre",
            cfg.activation_levels,
            batchnorm=cfg.binary_batchnorm,
            bn_quantize_input=cfg.bn_quantize_input,
            bn_quantize_centered=cfg.bn_quantize_centered,
            bn_quantize_normalized=cfg.bn_quantize_normalized,
            bn_quantize_output=cfg.bn_quantize_output,
            bn_quantize_eval_stats=cfg.bn_quantize_eval_stats,
            bn_momentum=cfg.bn_momentum,
        )
    x = _regularize(x, name, cfg)
    if cfg.float_head_convs:
        x = _fconv(x, 3 * HEAD_CHANNELS, 1, 1, f"{name}_proj", activation=None, batchnorm=cfg.float_batchnorm)
    else:
        x = _bconv(
            x,
            3 * HEAD_CHANNELS,
            1,
            1,
            f"{name}_proj",
            cfg.activation_levels,
            raw_output=cfg.head_proj_raw,
            batchnorm=cfg.binary_batchnorm and not cfg.head_proj_raw,
            bn_quantize_input=cfg.bn_quantize_input,
            bn_quantize_centered=cfg.bn_quantize_centered,
            bn_quantize_normalized=cfg.bn_quantize_normalized,
            bn_quantize_output=cfg.bn_quantize_output,
            bn_quantize_eval_stats=cfg.bn_quantize_eval_stats,
            bn_momentum=cfg.bn_momentum,
        )
    if not cfg.float_head_convs and cfg.head_proj_raw and cfg.binary_batchnorm:
        x = layers.BatchNormalization(name=f"{name}_proj_bn")(x)
    if not cfg.float_head_convs and cfg.head_proj_raw and cfg.head_proj_levels:
        x = QuantizedActivation(levels=cfg.head_proj_levels, signed=True, name=f"{name}_proj_quant")(x)
    h, w = int(x.shape[1]), int(x.shape[2])
    x = layers.Reshape((h, w, 3, HEAD_CHANNELS), name=f"{name}_reshape")(x)
    head_binarize = (not cfg.head_proj_raw) and (not cfg.float_head_convs)
    if cfg.float_output_heads:
        bbox = layers.Dense(4, use_bias=True, name=f"{name}_bbox")(x)
    else:
        bbox = QuantizedDense(
            4,
            levels=cfg.output_levels,
            input_threshold=0.0,
            binarize_inputs=head_binarize,
            integer_output=True,
            integer_kernel=True,
            output_scale=cfg.bbox_output_scale,
            quantize_bias=True,
            name=f"{name}_bbox",
        )(x)
    if head_binarize:
        feat_fn = _bdense_act
    elif cfg.float_obj_cls_hidden:
        feat_fn = lambda tensor, units, name, _: _fdense_act(tensor, units, name, batchnorm=cfg.float_batchnorm)
    else:
        feat_fn = _qdense_act
    obj_feat = feat_fn(x, OBJ_CLS_HIDDEN, f"{name}_obj_hidden", cfg)
    if OBJ_REMAP_HIDDEN:
        obj_feat = feat_fn(obj_feat, OBJ_REMAP_HIDDEN, f"{name}_obj_remap", cfg)
    if cfg.share_obj_cls_hidden:
        cls_feat = obj_feat
    else:
        cls_feat = feat_fn(x, OBJ_CLS_HIDDEN, f"{name}_cls_hidden", cfg)
    if cfg.float_output_heads:
        obj = layers.Dense(1, use_bias=True, name=f"{name}_obj")(obj_feat)
        cls = layers.Dense(classes, use_bias=True, name=f"{name}_cls")(cls_feat)
    else:
        obj = QuantizedDense(
            1,
            levels=cfg.output_levels,
            input_threshold=0.0,
            binarize_inputs=head_binarize,
            integer_output=True,
            integer_kernel=True,
            output_scale=cfg.obj_output_scale,
            quantize_bias=True,
            name=f"{name}_obj",
        )(obj_feat)
        cls = QuantizedDense(
            classes,
            levels=cfg.output_levels,
            input_threshold=0.0,
            binarize_inputs=head_binarize,
            integer_output=True,
            integer_kernel=True,
            output_scale=cfg.cls_output_scale,
            quantize_bias=True,
            name=f"{name}_cls",
        )(cls_feat)
    x = layers.Concatenate(axis=-1, name=f"{name}_logits")([bbox, obj, cls])
    return layers.Activation("linear", dtype="float32", name=name)(x)


def build_binary_yolov3_small(num_classes: int, cfg: ModelConfig):
    chans = list(BACKBONE_FILTERS)
    input_h, input_w = _input_hw(cfg.input_size)
    x_in = keras.Input((input_h, input_w, 3), name="image")
    if cfg.float_input:
        x = layers.Rescaling(2.0, offset=-1.0, name="float_input_scale")(x_in)
    else:
        x = BinaryInput(
            levels=cfg.thermo_levels,
            min_value=cfg.input_min_value,
            max_value=cfg.input_max_value,
            data_format="channels_last",
            name="binary_input",
        )(x_in)
    if cfg.input_dropcut:
        x = BinaryDropcut(
            dropcut_probability=cfg.input_dropcut,
            size_range=cfg.input_dropcut_size,
            num_boxes=cfg.input_dropcut_boxes,
            method=cfg.input_dropcut_method,
            name="input_dropcut",
        )(x)
    if cfg.input_dropcut2:
        x = BinaryDropcut(
            dropcut_probability=cfg.input_dropcut2,
            size_range=cfg.input_dropcut_size,
            num_boxes=cfg.input_dropcut_boxes,
            method=cfg.input_dropcut2_method,
            name="input_dropcut2",
        )(x)
    backbone_fn = (
        (lambda t, f, k, s, n: _fconv(t, f, k, s, n, batchnorm=cfg.float_batchnorm))
        if cfg.float_backbone
        else (
            lambda t, f, k, s, n: _bconv(
                t,
                f,
                k,
                s,
                n,
                cfg.activation_levels,
                batchnorm=cfg.binary_batchnorm,
                bn_quantize_input=cfg.bn_quantize_input,
                bn_quantize_centered=cfg.bn_quantize_centered,
                bn_quantize_normalized=cfg.bn_quantize_normalized,
                bn_quantize_output=cfg.bn_quantize_output,
                bn_quantize_eval_stats=cfg.bn_quantize_eval_stats,
                bn_momentum=cfg.bn_momentum,
            )
        )
    )
    for i, (filters, stride) in enumerate(zip(chans[:3], (1, 2, 2)), start=1):
        x = backbone_fn(x, filters, MAIN_KERNEL_SIZE, stride, f"stage{i}")
    p3_route = backbone_fn(x, chans[3], MAIN_KERNEL_SIZE, 2, "stage4")
    p4_route = backbone_fn(p3_route, chans[4], MAIN_KERNEL_SIZE, 2, "stage5")
    x = backbone_fn(p4_route, chans[5], MAIN_KERNEL_SIZE, 2, "stage6")
    x = backbone_fn(x, chans[6], MAIN_KERNEL_SIZE, 1, "stage7")
    x = backbone_fn(x, chans[7], 1, 1, "stage8")
    neck_fn = (
        (lambda t, f, k, s, n: _fconv(t, f, k, s, n, batchnorm=cfg.float_batchnorm))
        if cfg.float_neck
        else (
            lambda t, f, k, s, n: _bconv(
                t,
                f,
                k,
                s,
                n,
                cfg.activation_levels,
                batchnorm=cfg.binary_batchnorm,
                bn_quantize_input=cfg.bn_quantize_input,
                bn_quantize_centered=cfg.bn_quantize_centered,
                bn_quantize_normalized=cfg.bn_quantize_normalized,
                bn_quantize_output=cfg.bn_quantize_output,
                bn_quantize_eval_stats=cfg.bn_quantize_eval_stats,
                bn_momentum=cfg.bn_momentum,
            )
        )
    )
    p5_feat = neck_fn(x, NECK_P5_FILTERS, MAIN_KERNEL_SIZE, 1, "p5_feat")
    p5 = _head(p5_feat, "p5", num_classes, cfg)
    p4_up = layers.UpSampling2D(2, interpolation="nearest", name="up4")(neck_fn(x, NECK_P4_FILTERS, 1, 1, "up4_reduce"))
    p4_feat = neck_fn(layers.Concatenate(name="p4_concat")([p4_up, p4_route]), NECK_P4_FILTERS, MAIN_KERNEL_SIZE, 1, "p4_feat")
    p4 = _head(p4_feat, "p4", num_classes, cfg)
    p3_up = layers.UpSampling2D(2, interpolation="nearest", name="up3")(neck_fn(p4_feat, NECK_P3_FILTERS, 1, 1, "up3_reduce"))
    p3_feat = neck_fn(layers.Concatenate(name="p3_concat")([p3_up, p3_route]), NECK_P3_FILTERS, MAIN_KERNEL_SIZE, 1, "p3_feat")
    p3 = _head(p3_feat, "p3", num_classes, cfg)
    return keras.Model(x_in, {"p3": p3, "p4": p4, "p5": p5}, name="binary_yolov3_small")
