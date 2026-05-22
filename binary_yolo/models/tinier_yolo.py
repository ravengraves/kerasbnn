from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import layers

from binary_yolo.models.common import (
    BinaryInputConfig,
    FIXED_EXTRA_HEAD_BLOCKS,
    SpaceToDepth,
    make_binary_input,
    prediction_head,
    stem_conv,
    xnor_conv,
    xnor_fire_block,
)


@dataclass(frozen=True)
class TinierYOLOConfig(BinaryInputConfig):
    pass


def build_binary_tinier_yolo(
    num_classes: int,
    input_shape: tuple[int, int, int] = (416, 416, 3),
    *,
    cfg: TinierYOLOConfig | None = None,
):
    cfg = cfg or TinierYOLOConfig()
    image = keras.Input(shape=input_shape, name="image")
    x = make_binary_input(image, cfg)

    x = stem_conv(x, 16, 3, cfg=cfg, name="conv1", strides=2)

    x = xnor_conv(x, 32, 3, name="conv2", strides=2)

    x = xnor_conv(x, 64, 3, name="conv3", strides=2)

    x = xnor_conv(x, 128, 3, name="conv4", strides=2)

    c5 = xnor_conv(x, 256, 3, name="conv5")
    x = xnor_conv(c5, 256, 3, name="down5", strides=2)

    dense = x
    for idx in range(1, 5):
        fire = xnor_fire_block(dense, 64, 256, 256, name=f"fire{idx}")
        dense = layers.Concatenate(name=f"dense{idx}")([fire, dense])
    f5 = xnor_fire_block(dense, 64, 256, 256, name="fire5")

    passthrough = xnor_conv(c5, 64, 1, name="passthrough_reduce")
    passthrough = SpaceToDepth(block_size=2, name="passthrough_reorg")(passthrough)
    head1_in = layers.Concatenate(name="head1_in")([passthrough, f5])

    f6 = xnor_fire_block(head1_in, 64, 256, 256, name="fire6")
    head1_feat = layers.Concatenate(name="head1_dense")([f6, head1_in])

    extra_head1_blocks = (FIXED_EXTRA_HEAD_BLOCKS + 1) // 2
    extra_head2_blocks = FIXED_EXTRA_HEAD_BLOCKS // 2
    for offset in range(extra_head1_blocks):
        block_idx = 9 + offset
        fire = xnor_fire_block(head1_feat, 64, 256, 256, name=f"fire{block_idx}")
        head1_feat = layers.Concatenate(name=f"dense{block_idx}")([fire, head1_feat])
    pred_13 = prediction_head(head1_feat, num_classes, name="pred_13")

    up = xnor_conv(f6, 128, 1, name="up_reduce")
    up = layers.UpSampling2D(size=2, interpolation="nearest", name="upsample")(up)
    head2_in = layers.Concatenate(name="head2_in")([up, c5])

    f7 = xnor_fire_block(head2_in, 32, 128, 128, name="fire7")
    dense7 = layers.Concatenate(name="dense7")([f7, head2_in])
    f8 = xnor_fire_block(dense7, 32, 128, 128, name="fire8")
    head2_feat = layers.Concatenate(name="dense8")([f8, dense7])

    for offset in range(extra_head2_blocks):
        block_idx = 9 + extra_head1_blocks + offset
        fire = xnor_fire_block(head2_feat, 32, 128, 128, name=f"fire{block_idx}")
        head2_feat = layers.Concatenate(name=f"dense{block_idx}")([fire, head2_feat])
    pred_26 = prediction_head(head2_feat, num_classes, name="pred_26")

    return keras.Model(
        image,
        {"pred_13": pred_13, "pred_26": pred_26},
        name="tinier_yolov3_xnor2",
    )
