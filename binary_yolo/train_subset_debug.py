from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from binary_yolo.model import ModelConfig, build_binary_yolov3_small
from binary_yolo.train_fruits import (
    TrainConfig,
    _artifact_path,
    _input_hw,
    _optimizer,
    _resolve_input_size_arg,
    _save_json,
    _subset_samples,
)
from binary_yolo.yad2k_core import build_targets, fit_anchors, load_dataset_splits, make_losses, map50

import keras


class RepeatedArrayDataset(keras.utils.PyDataset):
    def __init__(self, images, targets, batch_size: int, repeat: int):
        super().__init__()
        self.images = images
        self.targets = targets
        self.batch_size = int(batch_size)
        self.repeat = max(1, int(repeat))
        self.base_count = int(images.shape[0])
        self.total_count = self.base_count * self.repeat

    def __len__(self):
        return max(1, math.ceil(self.total_count / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = min(start + self.batch_size, self.total_count)
        batch_ids = np.arange(start, stop, dtype=np.int64) % self.base_count
        batch_x = self.images[batch_ids]
        batch_y = {name: values[batch_ids] for name, values in self.targets.items()}
        return batch_x, batch_y


class SubsetMapCallback(keras.callbacks.Callback):
    def __init__(self, images, samples, anchors, input_size: int, num_classes: int, batch_size: int, bbox_output_scale: float, obj_output_scale: float, cls_output_scale: float, logit_scale: float, score_threshold: float, objectness_threshold: float, class_mode: str, box_mode: str):
        super().__init__()
        self.images = images
        self.samples = samples
        self.anchors = anchors
        self.input_size = input_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.bbox_output_scale = bbox_output_scale
        self.obj_output_scale = obj_output_scale
        self.cls_output_scale = cls_output_scale
        self.logit_scale = logit_scale
        self.score_threshold = score_threshold
        self.objectness_threshold = objectness_threshold
        self.class_mode = class_mode
        self.box_mode = box_mode

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        preds = self.model.predict(self.images, batch_size=self.batch_size, verbose=0)
        logs["val_map50"] = map50(
            self.samples,
            preds,
            self.anchors,
            self.input_size,
            self.num_classes,
            score=self.score_threshold,
            objectness=self.objectness_threshold,
            bbox_output_scale=self.bbox_output_scale,
            obj_output_scale=self.obj_output_scale,
            cls_output_scale=self.cls_output_scale,
            logit_scale=self.logit_scale,
            class_mode=self.class_mode,
            box_mode=self.box_mode,
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-format", default="auto")
    ap.add_argument("--dataset-split", default="official")
    ap.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--input-width", type=int, default=0)
    ap.add_argument("--input-height", type=int, default=0)
    ap.add_argument("--train-sample-fraction", type=float, default=0.05)
    ap.add_argument("--train-sample-count", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--positive-top-k", type=int, default=3)
    ap.add_argument("--optimizer", default="adam", choices=("adam", "adamw", "rmsprop", "sgd"))
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--schedule", default="constant", choices=("constant", "cosine"))
    ap.add_argument("--class-mode", default="softmax", choices=("softmax", "bce"))
    ap.add_argument("--box-mode", default="linear", choices=("linear", "yolo"))
    ap.add_argument("--obj-gain", type=float, default=4.0)
    ap.add_argument("--obj-pos-gain", type=float, default=8.0)
    ap.add_argument("--noobj-weight", type=float, default=0.04)
    ap.add_argument("--cls-gain", type=float, default=8.0)
    ap.add_argument("--box-gain", type=float, default=5.0)
    ap.add_argument("--ignore-iou", type=float, default=0.5)
    ap.add_argument("--bbox-output-scale", type=float, default=2.0)
    ap.add_argument("--obj-output-scale", type=float, default=64.0)
    ap.add_argument("--cls-output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--rescore-objectness", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--hard-negative-ratio", type=float, default=0.0)
    ap.add_argument("--hard-negative-min", type=int, default=1024)
    ap.add_argument("--conf-neg-normalizer", default="pos", choices=("neg", "pos", "none"))
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--objectness-threshold", type=float, default=0.25)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--float-input", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-backbone", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-neck", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-head-convs", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-batchnorm", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--binary-batchnorm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-momentum", type=float, default=0.95)
    ap.add_argument("--bn-quantize-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-centered", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-normalized", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-eval-stats", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--float-obj-cls-hidden", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-output-heads", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--head-proj-raw", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--head-proj-levels", type=int, default=8)
    ap.add_argument("--input-dropcut", type=float, default=0.0)
    ap.add_argument("--p3-loss-weight", type=float, default=1.0)
    ap.add_argument("--p4-loss-weight", type=float, default=1.0)
    ap.add_argument("--p5-loss-weight", type=float, default=1.0)
    ap.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--auto-scale-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--clipnorm", type=float, default=1.0)
    ap.add_argument("--dtype-policy", default="mixed_float16", choices=("mixed_float16", "float32"))
    ap.add_argument("--run-dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    input_size = _resolve_input_size_arg(args)
    keras.config.set_dtype_policy(args.dtype_policy)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    train_samples, _, classes = load_dataset_splits(
        args.dataset,
        dataset_format=args.dataset_format,
        split_mode=args.dataset_split,
        seed=args.seed,
        include_difficult=args.include_difficult,
        input_size=input_size,
    )
    train_samples = _subset_samples(train_samples, args.train_sample_fraction, args.seed)
    if args.train_sample_count > 0:
        train_samples = train_samples[: args.train_sample_count]
    anchors = fit_anchors(train_samples, input_size, seed=args.seed)

    val_images, val_targets = build_targets(train_samples, len(classes), anchors, input_size, args.thermo_levels, augment=False, positive_top_k=args.positive_top_k)
    train_images, train_targets = build_targets(train_samples, len(classes), anchors, input_size, args.thermo_levels, augment=False, positive_top_k=args.positive_top_k)
    train_dataset = RepeatedArrayDataset(train_images, train_targets, args.batch_size, args.repeat)

    model_cfg = ModelConfig(
        input_size=input_size,
        thermo_levels=args.thermo_levels,
        input_dropcut=args.input_dropcut,
        float_input=args.float_input,
        float_backbone=args.float_backbone,
        float_neck=args.float_neck,
        float_head_convs=args.float_head_convs,
        float_batchnorm=args.float_batchnorm,
        binary_batchnorm=args.binary_batchnorm,
        bn_momentum=args.bn_momentum,
        bn_quantize_input=args.bn_quantize_input,
        bn_quantize_centered=args.bn_quantize_centered,
        bn_quantize_normalized=args.bn_quantize_normalized,
        bn_quantize_output=args.bn_quantize_output,
        bn_quantize_eval_stats=args.bn_quantize_eval_stats,
        float_obj_cls_hidden=args.float_obj_cls_hidden,
        float_output_heads=args.float_output_heads,
        bbox_output_scale=args.bbox_output_scale,
        obj_output_scale=args.obj_output_scale,
        cls_output_scale=args.cls_output_scale,
        head_proj_raw=args.head_proj_raw,
        head_proj_levels=args.head_proj_levels,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        schedule=args.schedule,
        obj_gain=args.obj_gain,
        obj_pos_gain=args.obj_pos_gain,
        noobj_weight=args.noobj_weight,
        cls_gain=args.cls_gain,
        box_gain=args.box_gain,
        class_mode=args.class_mode,
        ignore_iou=args.ignore_iou,
        bbox_output_scale=args.bbox_output_scale,
        obj_output_scale=args.obj_output_scale,
        cls_output_scale=args.cls_output_scale,
        logit_scale=args.logit_scale,
        rescore_objectness=args.rescore_objectness,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_negative_min=args.hard_negative_min,
        conf_neg_normalizer=args.conf_neg_normalizer,
        score_threshold=args.score_threshold,
        objectness_threshold=args.objectness_threshold,
        train_augment=False,
        jit_compile=args.jit_compile,
        auto_scale_loss=args.auto_scale_loss,
        clipnorm=args.clipnorm,
    )

    model = build_binary_yolov3_small(len(classes), model_cfg)
    steps = len(train_dataset)
    model.compile(
        optimizer=_optimizer(train_cfg, steps),
        loss=make_losses(
            anchors,
            model_cfg.input_size,
            len(classes),
            obj_gain=train_cfg.obj_gain,
            obj_pos_gain=train_cfg.obj_pos_gain,
            noobj_weight=train_cfg.noobj_weight,
            cls_neg_weight=train_cfg.cls_neg_weight,
            cls_gain=train_cfg.cls_gain,
            class_mode=train_cfg.class_mode,
            box_gain=train_cfg.box_gain,
            bbox_output_scale=train_cfg.bbox_output_scale,
            obj_output_scale=train_cfg.obj_output_scale,
            cls_output_scale=train_cfg.cls_output_scale,
            logit_scale=train_cfg.logit_scale,
            ignore_iou=train_cfg.ignore_iou,
            rescore_objectness=train_cfg.rescore_objectness,
            hard_negative_ratio=train_cfg.hard_negative_ratio,
            hard_negative_min=train_cfg.hard_negative_min,
            conf_neg_normalizer=train_cfg.conf_neg_normalizer,
            box_mode=args.box_mode,
        ),
        loss_weights={"p3": args.p3_loss_weight, "p4": args.p4_loss_weight, "p5": args.p5_loss_weight},
        jit_compile=args.jit_compile,
        steps_per_execution=max(1, min(steps, 32)),
        auto_scale_loss=args.auto_scale_loss,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        verbose=2,
        validation_data=(val_images, val_targets),
        callbacks=[
            SubsetMapCallback(
                val_images,
                train_samples,
                anchors,
                input_size,
                len(classes),
                args.batch_size,
                train_cfg.bbox_output_scale,
                train_cfg.obj_output_scale,
                train_cfg.cls_output_scale,
                train_cfg.logit_scale,
                train_cfg.score_threshold,
                train_cfg.objectness_threshold,
                train_cfg.class_mode,
                args.box_mode,
            ),
            keras.callbacks.CSVLogger(_artifact_path(run_dir, "voc", "history.csv")),
            keras.callbacks.ModelCheckpoint(
                _artifact_path(run_dir, "voc", "best.weights.h5"),
                monitor="val_map50",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ],
    ).history
    input_h, input_w = _input_hw(input_size)
    summary = {
        "dataset": {"train_samples": len(train_samples), "repeat": args.repeat, "positive_top_k": args.positive_top_k},
        "model": asdict(model_cfg),
        "train": asdict(train_cfg) | {"p3_loss_weight": args.p3_loss_weight, "p4_loss_weight": args.p4_loss_weight, "p5_loss_weight": args.p5_loss_weight},
        "input_hw": [input_h, input_w],
        "best_val_map50": float(max(history.get("val_map50", [0.0]))),
        "final_val_map50": float(history.get("val_map50", [0.0])[-1] if history.get("val_map50") else 0.0),
        "best_val_loss": float(min(history["val_loss"])),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_loss": float(history["loss"][-1]),
    }
    _save_json(run_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
