from __future__ import annotations

import sys

import binary_yolo.train_tinier as base
from binary_yolo.tinier_yolo_xnor2 import (
    TinierXNOR2Config,
    build_binary_tinier_yolov3_xnor2,
    build_float_tinier_yolov3_xnor2,
)


base.TinierBinaryConfig = TinierXNOR2Config
base.build_binary_tinier_yolov3 = build_binary_tinier_yolov3_xnor2
base.build_float_tinier_yolov3 = build_float_tinier_yolov3_xnor2


_base_parse_args = base.parse_args


def _flag_present(*names: str) -> bool:
    argv = sys.argv[1:]
    return any(name in argv for name in names)


def _parse_args_with_xnor2_defaults():
    args = _base_parse_args()
    if not _flag_present("--noobj-weight"):
        args.noobj_weight = 0.02
    if not _flag_present("--hard-negative-min"):
        args.hard_negative_min = 64
    if not _flag_present("--score-threshold"):
        args.score_threshold = 0.001
    if not _flag_present("--objectness-threshold"):
        args.objectness_threshold = 0.001
    if not _flag_present("--bn-recalibrate-before-val", "--no-bn-recalibrate-before-val"):
        args.bn_recalibrate_before_val = True
    if not _flag_present("--box-iou-aux-gain"):
        args.box_iou_aux_gain = 0.5
    if not _flag_present("--box-iou-aux-mode"):
        args.box_iou_aux_mode = "iou"
    return args


base.parse_args = _parse_args_with_xnor2_defaults


if __name__ == "__main__":
    base.main()
