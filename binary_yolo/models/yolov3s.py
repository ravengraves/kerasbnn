from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import layers

from binary_yolo.models.common import (
    BinaryInputConfig,
    make_binary_input,
    prediction_head,
    stem_conv,
    xnor_conv,
    _scaled_channels,
)


YOLOV3S_WIDTH_MULT = 1.5


@dataclass(frozen=True)
class YoloV3SConfig(BinaryInputConfig):
    pass


def build_binary_yolov3s(
    num_classes: int,
    input_shape: tuple[int, int, int] = (416, 416, 3),
    *,
    cfg: YoloV3SConfig | None = None,
):
    cfg = cfg or YoloV3SConfig()
    image = keras.Input(shape=input_shape, name="image")
    x = make_binary_input(image, cfg)

    stem16 = _scaled_channels(16, YOLOV3S_WIDTH_MULT)
    stem32 = _scaled_channels(32, YOLOV3S_WIDTH_MULT)
    stem64 = _scaled_channels(64, YOLOV3S_WIDTH_MULT)
    stem128 = _scaled_channels(128, YOLOV3S_WIDTH_MULT)
    stem256 = _scaled_channels(256, YOLOV3S_WIDTH_MULT)
    stem512 = _scaled_channels(512, YOLOV3S_WIDTH_MULT)
    stem1024 = _scaled_channels(1024, YOLOV3S_WIDTH_MULT)

    x = stem_conv(x, stem16, 3, cfg=cfg, name="conv1", strides=2)

    x = xnor_conv(x, stem32, 3, name="conv2", strides=2)

    x = xnor_conv(x, stem64, 3, name="conv3", strides=2)

    x = xnor_conv(x, stem128, 3, name="conv4", strides=2)

    c5 = xnor_conv(x, stem256, 3, name="conv5")
    x = xnor_conv(c5, stem512, 3, name="conv6", strides=2)

    x = xnor_conv(x, stem1024, 3, name="conv7")
    route = xnor_conv(x, stem256, 1, name="route1")
    head1_feat = xnor_conv(route, stem512, 3, name="conv9")
    pred_13 = prediction_head(head1_feat, num_classes, name="pred_13")

    up = xnor_conv(route, stem128, 1, name="up_reduce")
    up = layers.UpSampling2D(size=2, interpolation="nearest", name="upsample")(up)
    head2_in = layers.Concatenate(name="head2_in")([up, c5])
    head2_feat = xnor_conv(head2_in, stem256, 3, name="conv11")
    pred_26 = prediction_head(head2_feat, num_classes, name="pred_26")

    return keras.Model(
        image,
        {"pred_13": pred_13, "pred_26": pred_26},
        name="tiny_yolov3_xnor2",
    )
