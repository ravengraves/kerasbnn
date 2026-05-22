from __future__ import annotations

from binary_yolo.model import ModelConfig, build_binary_yolov3_small


def build_float_yolov3_small_reference(num_classes: int, input_size: int = 224, width_scale: float = 1.0):
    cfg = ModelConfig(
        input_size=input_size,
        thermo_levels=4,
        width_scale=width_scale,
        stem_filters=32 if width_scale <= 1.0 else 48,
        input_dropcut=0.0,
        float_input=True,
        float_backbone=True,
        float_neck=True,
        float_head_convs=True,
        float_batchnorm=True,
        float_obj_cls_hidden=True,
        float_output_heads=True,
        head_pre_filters=256 if width_scale <= 1.0 else 384,
        head_channels=96 if width_scale <= 1.0 else 128,
        obj_cls_hidden=256 if width_scale <= 1.0 else 384,
        obj_remap_hidden=128 if width_scale <= 1.0 else 192,
        head_proj_raw=False,
        head_proj_levels=0,
    )
    return build_binary_yolov3_small(num_classes, cfg), cfg
