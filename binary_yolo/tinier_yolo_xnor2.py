from __future__ import annotations

from dataclasses import dataclass

from binary_yolo.models.tinier_yolo import TinierYOLOConfig, build_binary_tinier_yolo
from binary_yolo.models.yolov3s import YoloV3SConfig, build_binary_yolov3s
from binary_yolo.tiniermodel import build_float_tinier_yolov3


@dataclass(frozen=True)
class TinierXNOR2Config:
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
    tinyv3_backbone: bool = False


def _to_tinier_cfg(cfg: TinierXNOR2Config) -> TinierYOLOConfig:
    return TinierYOLOConfig(
        thermo_levels=cfg.thermo_levels,
        input_levels=cfg.input_levels,
        input_min_value=cfg.input_min_value,
        input_max_value=cfg.input_max_value,
        input_signed=cfg.input_signed,
        quantized_input=cfg.quantized_input,
        input_dropcut=cfg.input_dropcut,
        input_dropcut_size=cfg.input_dropcut_size,
        input_dropcut_method=cfg.input_dropcut_method,
        input_dropcut_boxes=cfg.input_dropcut_boxes,
    )


def _to_yolov3s_cfg(cfg: TinierXNOR2Config) -> YoloV3SConfig:
    return YoloV3SConfig(
        thermo_levels=cfg.thermo_levels,
        input_levels=cfg.input_levels,
        input_min_value=cfg.input_min_value,
        input_max_value=cfg.input_max_value,
        input_signed=cfg.input_signed,
        quantized_input=cfg.quantized_input,
        input_dropcut=cfg.input_dropcut,
        input_dropcut_size=cfg.input_dropcut_size,
        input_dropcut_method=cfg.input_dropcut_method,
        input_dropcut_boxes=cfg.input_dropcut_boxes,
    )


def build_binary_tinier_yolov3_xnor2(
    num_classes: int,
    input_shape: tuple[int, int, int] = (416, 416, 3),
    *,
    cfg: TinierXNOR2Config | None = None,
):
    cfg = cfg or TinierXNOR2Config()
    if cfg.tinyv3_backbone:
        return build_binary_yolov3s(num_classes, input_shape=input_shape, cfg=_to_yolov3s_cfg(cfg))
    return build_binary_tinier_yolo(num_classes, input_shape=input_shape, cfg=_to_tinier_cfg(cfg))


def build_float_tinier_yolov3_xnor2(
    num_classes: int,
    input_shape: tuple[int, int, int] = (416, 416, 3),
    *,
    cfg: TinierXNOR2Config | None = None,
    use_bn_in_fire: bool | None = None,
    head_pre_filters: int | None = None,
    head_channels: int | None = None,
    obj_cls_hidden: int | None = None,
    obj_remap_hidden: int | None = None,
    obj_bias_init: float | None = None,
):
    del cfg
    return build_float_tinier_yolov3(
        num_classes,
        input_shape=input_shape,
        use_bn_in_fire=True if use_bn_in_fire is None else bool(use_bn_in_fire),
        head_pre_filters=256 if head_pre_filters is None else int(head_pre_filters),
        head_channels=64 if head_channels is None else int(head_channels),
        obj_cls_hidden=128 if obj_cls_hidden is None else int(obj_cls_hidden),
        obj_remap_hidden=64 if obj_remap_hidden is None else int(obj_remap_hidden),
        obj_bias_init=0.0 if obj_bias_init is None else float(obj_bias_init),
    )
