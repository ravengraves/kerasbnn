from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import keras
import numpy as np

from binary_yolo.dataset import DetectionPyDataset
from binary_yolo.train_fruits import (
    _artifact_path,
    _optimizer,
    _resolve_input_size_arg,
    _save_json,
    _subset_samples,
    TrainConfig,
)
from binary_yolo.train_tinier import (
    BinaryBNRecalibrationCallback,
    EpochMetadataCallback,
    GlobalBestMapWeightsCallback,
    GlobalBestWeightsCallback,
    MeanAPCallback,
    TeacherDistillBurnInCallback,
    TINIER_ANCHOR_MASKS,
    TINIER_HEADS,
    TINIER_STRIDES,
    _append_teacher_targets,
    _build_cfg,
    _build_model,
    _build_teacher,
    _build_train_cfg,
    _distilled_losses,
    _effective_distill_args,
    _load_bbox_priors_csv,
    _save_bbox_priors_csv,
    _teacher_preds_to_student_targets,
)
import binary_yolo.yad2k_core as yad2k_core


class ShuffledArrayDataset(keras.utils.PyDataset):
    def __init__(self, images, targets, batch_size: int, shuffle: bool = True, seed: int = 7, repeat: int = 1):
        super().__init__()
        self.images = images
        self.targets = targets
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.repeat = max(1, int(repeat))
        self.rng = np.random.default_rng(int(seed))
        self.base_count = int(images.shape[0])
        self.order = np.arange(self.base_count * self.repeat, dtype=np.int32) % self.base_count
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return max(1, math.ceil(int(self.order.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = min(start + self.batch_size, int(self.order.shape[0]))
        batch_ids = self.order[start:stop]
        return self.images[batch_ids], {name: values[batch_ids] for name, values in self.targets.items()}

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


class EpochMetadataCallback(keras.callbacks.Callback):
    def __init__(self, restart_cycle: int, epochs_run_offset: int = 0):
        super().__init__()
        self.restart_cycle = int(restart_cycle)
        self.epochs_run_offset = int(epochs_run_offset)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        logs["restart_cycle"] = self.restart_cycle
        logs["epochs_run_total"] = self.epochs_run_offset + int(epoch) + 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-format", default="voc")
    ap.add_argument("--dataset-split", default="official")
    ap.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.0)
    ap.add_argument("--test-sample-fraction", type=float, default=1.0)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--input-width", type=int, default=0)
    ap.add_argument("--input-height", type=int, default=0)
    ap.add_argument("--train-sample-fraction", type=float, default=1.0)
    ap.add_argument("--val-sample-fraction", type=float, default=1.0)
    ap.add_argument("--train-sample-count", type=int, default=0)
    ap.add_argument("--subset-train-as-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outer-epochs", type=int, default=8)
    ap.add_argument("--student-inner-epochs", type=int, default=4)
    ap.add_argument("--restart-cycles", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--teacher-batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--positive-top-k", type=int, default=3)
    ap.add_argument("--optimizer", default="adam", choices=("adam", "adamw", "rmsprop", "sgd"))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--schedule", default="constant", choices=("constant", "cosine", "cosine_restart", "exp"))
    ap.add_argument("--class-mode", default="softmax", choices=("softmax", "bce"))
    ap.add_argument("--box-mode", default="linear", choices=("linear", "linear_float", "yolo"))
    ap.add_argument("--obj-gain", type=float, default=4.0)
    ap.add_argument("--obj-pos-gain", type=float, default=8.0)
    ap.add_argument("--noobj-weight", type=float, default=0.02)
    ap.add_argument("--cls-gain", type=float, default=8.0)
    ap.add_argument("--cls-neg-weight", type=float, default=0.0)
    ap.add_argument("--box-gain", type=float, default=5.0)
    ap.add_argument("--box-iou-aux-gain", type=float, default=0.0)
    ap.add_argument("--box-iou-aux-mode", default="iou", choices=("iou", "diou"))
    ap.add_argument("--ignore-iou", type=float, default=0.5)
    ap.add_argument("--bbox-output-scale", type=float, default=2.0)
    ap.add_argument("--obj-output-scale", type=float, default=64.0)
    ap.add_argument("--cls-output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--rescore-objectness", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--hard-negative-ratio", type=float, default=0.0)
    ap.add_argument("--hard-negative-min", type=int, default=256)
    ap.add_argument("--conf-neg-normalizer", default="pos", choices=("neg", "pos", "none"))
    ap.add_argument("--score-threshold", type=float, default=0.05)
    ap.add_argument("--objectness-threshold", type=float, default=0.05)
    ap.add_argument("--val-map-every", type=int, default=1)
    ap.add_argument("--validation-freq", type=int, default=1)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--input-levels", type=int, default=16)
    ap.add_argument("--input-signed", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-input", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-stem-first-layer", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--quantized-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--input-dropcut", type=float, default=0.0)
    ap.add_argument("--input-dropcut-size-min", type=float, default=0.05)
    ap.add_argument("--input-dropcut-size-max", type=float, default=0.20)
    ap.add_argument("--input-dropcut-method", default="both", choices=("black", "white", "both", "noise"))
    ap.add_argument("--input-dropcut-boxes", type=int, default=1)
    ap.add_argument("--early-quantized-layers", type=int, default=1)
    ap.add_argument("--activation-surrogate", default="triangular", choices=("clipped", "triangular"))
    ap.add_argument("--activation-surrogate-gain", type=float, default=1.0)
    ap.add_argument("--weight-surrogate", default="magnitude_aware", choices=("clipped", "magnitude_aware"))
    ap.add_argument("--weight-surrogate-floor", type=float, default=0.0)
    ap.add_argument("--weight-surrogate-gain", type=float, default=1.0)
    ap.add_argument("--norm-gamma-init", type=float, default=0.25)
    ap.add_argument("--use-rprelu", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--rprelu-init-beta", type=float, default=0.25)
    ap.add_argument("--rprelu-init-gamma", type=float, default=0.0)
    ap.add_argument("--rprelu-init-zeta", type=float, default=0.0)
    ap.add_argument("--binary-kernel-init-scale", type=float, default=0.0)
    ap.add_argument("--activation-levels", type=int, default=128)
    ap.add_argument("--output-levels", type=int, default=128)
    ap.add_argument("--binary-batchnorm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-bn-in-fire", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-momentum", type=float, default=0.95)
    ap.add_argument("--bn-quantize-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-centered", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-normalized", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-eval-stats", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-eval-use-batch-stats", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-force-full-quant-eval", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-affine-params", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-recalibrate-before-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-recalibrate-batches", type=int, default=0)
    ap.add_argument("--stem-output-levels", type=int, default=0)
    ap.add_argument("--head-pre-filters", type=int, default=256)
    ap.add_argument("--head-channels", type=int, default=64)
    ap.add_argument("--head-proj-levels", type=int, default=128)
    ap.add_argument("--obj-cls-hidden", type=int, default=128)
    ap.add_argument("--obj-remap-hidden", type=int, default=64)
    ap.add_argument("--obj-bias-init", type=float, default=0.0)
    ap.add_argument("--backbone-width-mult", type=float, default=1.0)
    ap.add_argument("--float-prediction-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-neck-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--direct-quantized-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--late-quantized-from", type=int, default=0)
    ap.add_argument("--late-conv-levels", type=int, default=128)
    ap.add_argument("--late-act-levels", type=int, default=16)
    ap.add_argument("--teacher-weights", default="")
    ap.add_argument("--teacher-run-dir", default="")
    ap.add_argument("--teacher-checkpoint-monitor", default="val_map50", choices=("val_loss", "val_map50"))
    ap.add_argument("--teacher-use-bn-in-fire", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--teacher-distill", type=float, default=0.001)
    ap.add_argument("--teacher-bbox-distill", type=float, default=0.05)
    ap.add_argument("--teacher-obj-distill", type=float, default=0.1)
    ap.add_argument("--teacher-cls-distill", type=float, default=1.0)
    ap.add_argument("--teacher-distill-burnin-epochs", type=int, default=8)
    ap.add_argument("--teacher-distill-mode", default="decoded_dense", choices=("decoded_dense", "topk_align"))
    ap.add_argument("--teacher-align-top-k-teacher", type=int, default=16)
    ap.add_argument("--teacher-align-top-k-student", type=int, default=64)
    ap.add_argument("--teacher-align-student-pull", type=float, default=0.0)
    ap.add_argument("--distill-score-threshold", type=float, default=0.25)
    ap.add_argument("--loader-workers", type=int, default=2)
    ap.add_argument("--train-augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--augment-start-strength", type=float, default=0.2)
    ap.add_argument("--augment-ramp-epochs", type=int, default=100)
    ap.add_argument("--augment-constant-strength", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--advanced-augmentations", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--advanced-augment-prob", type=float, default=0.05)
    ap.add_argument("--advanced-augment-ramp-epochs", type=int, default=0)
    ap.add_argument("--cycle-early-stop-patience-epochs", type=int, default=64)
    ap.add_argument("--cycle-early-stop-monitor", default="val_map50", choices=("val_loss", "val_map50"))
    ap.add_argument("--dtype-policy", default="mixed_float16", choices=("mixed_float16", "float32"))
    ap.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--auto-scale-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--clipnorm", type=float, default=1.0)
    ap.add_argument("--warm-start-weights", default="")
    ap.add_argument("--warm-start-source-model", default="auto", choices=("auto", "binary", "float"))
    ap.add_argument("--restart-reload-monitor", default="val_map50", choices=("val_loss", "val_map50"))
    ap.add_argument("--run-dir", required=True)
    return ap.parse_args()


def _teacher_weights_path(args) -> Path:
    if args.teacher_run_dir:
        suffix = "best_map.weights.h5" if args.teacher_checkpoint_monitor == "val_map50" else "best.weights.h5"
        return Path(args.teacher_run_dir) / f"model.voc.{suffix}"
    return Path(args.teacher_weights)


def _restart_reload_path(run_dir: Path, monitor: str) -> Path:
    suffix = "best_map.weights.h5" if monitor == "val_map50" else "best.weights.h5"
    return _artifact_path(run_dir, "voc", suffix)


def _prepare_split_samples(args, input_size):
    train_samples, val_samples, classes = yad2k_core.load_dataset_splits(
        args.dataset,
        dataset_format=args.dataset_format,
        split_mode=args.dataset_split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        include_difficult=args.include_difficult,
        input_size=input_size,
    )
    if args.test_fraction > 0.0:
        merged = list(train_samples) + list(val_samples)
        train_samples, val_samples, _ = yad2k_core.split_samples_three_way(
            merged,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    train_samples = _subset_samples(train_samples, args.train_sample_fraction, args.seed)
    val_samples = _subset_samples(val_samples, args.val_sample_fraction, args.seed + 1)
    if args.train_sample_count > 0:
        train_samples = train_samples[: args.train_sample_count]
    if args.subset_train_as_val:
        val_samples = list(train_samples)
    return train_samples, val_samples, classes


def _csv_logger(path: Path, append: bool):
    return keras.callbacks.CSVLogger(path, append=append)


def _history_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _history_float(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        if stripped.lower() in {"na", "nan", "none", "null"}:
            return default
        value = stripped
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def main():
    args = parse_args()
    epochs_per_restart = int(args.outer_epochs) * int(args.student_inner_epochs)
    args.epochs = int(args.restart_cycles) * epochs_per_restart
    input_size = _resolve_input_size_arg(args)
    keras.config.set_dtype_policy(args.dtype_policy)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    train_samples, val_samples, classes = _prepare_split_samples(args, input_size)

    anchors = yad2k_core.fit_anchors(train_samples, input_size, n=6, seed=args.seed)
    train_cfg = _build_train_cfg(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_bbox_priors_csv(run_dir / "bbox_priors.csv", anchors)

    teacher = None
    teacher_spec = None
    teacher_path = _teacher_weights_path(args)
    teacher_anchors = anchors
    if float(args.teacher_distill) > 0.0:
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher weights not found: {teacher_path}")
        args.teacher_weights = str(teacher_path)
        teacher, teacher_spec = _build_teacher(len(classes), input_size, args)
        teacher_run_dir = str(getattr(args, "teacher_run_dir", "") or "")
        if teacher_run_dir:
            teacher_anchor_path = Path(teacher_run_dir) / "bbox_priors.csv"
            if teacher_anchor_path.exists():
                teacher_anchors = _load_bbox_priors_csv(teacher_anchor_path)
                print(f"Loaded teacher anchors from {teacher_anchor_path}")
            else:
                print(f"Teacher anchors not found at {teacher_anchor_path}; using student anchors for distillation")

    val_images, val_targets = yad2k_core.build_targets(
        val_samples,
        len(classes),
        anchors,
        input_size,
        args.thermo_levels,
        augment=False,
        positive_top_k=args.positive_top_k,
        num_workers=args.loader_workers,
        advanced_augmentations=False,
        advanced_augment_prob=args.advanced_augment_prob,
        heads=TINIER_HEADS,
        strides=TINIER_STRIDES,
        anchor_masks=TINIER_ANCHOR_MASKS,
    )

    history_path = _artifact_path(run_dir, "voc", "history.csv")
    best_weights = _artifact_path(run_dir, "voc", "best.weights.h5")
    restart_reload_weights = _restart_reload_path(run_dir, args.restart_reload_monitor)
    steps = max(1, math.ceil(len(train_samples) / args.batch_size))
    teacher_distill_scale_var = keras.Variable(1.0, dtype="float32", trainable=False, name="teacher_distill_scale")

    def _build_compiled_student(loss_args):
        student = _build_model(len(classes), input_size, args)
        base_losses = yad2k_core.make_losses(
            anchors,
            input_size,
            len(classes),
            obj_gain=train_cfg.obj_gain,
            obj_pos_gain=train_cfg.obj_pos_gain,
            noobj_weight=train_cfg.noobj_weight,
            cls_neg_weight=train_cfg.cls_neg_weight,
            cls_gain=train_cfg.cls_gain,
            class_mode=train_cfg.class_mode,
            box_gain=train_cfg.box_gain,
            box_iou_aux_gain=train_cfg.box_iou_aux_gain,
            box_iou_aux_mode=train_cfg.box_iou_aux_mode,
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
            heads=TINIER_HEADS,
            strides=TINIER_STRIDES,
            anchor_masks=TINIER_ANCHOR_MASKS,
        )
        losses = _distilled_losses(
            base_losses,
            len(classes),
            train_cfg,
            loss_args,
            anchors,
            input_size,
            teacher_distill_scale_var=teacher_distill_scale_var,
        )
        student.compile(
            optimizer=_optimizer(train_cfg, steps),
            loss=losses,
            jit_compile=train_cfg.jit_compile,
            steps_per_execution=max(1, min(steps, 128)),
            auto_scale_loss=train_cfg.auto_scale_loss,
        )
        return student

    model = None
    for restart_idx in range(args.restart_cycles):
        cycle_args = _effective_distill_args(args, restart_idx)
        model = _build_compiled_student(cycle_args)
        if restart_idx == 0 and args.warm_start_weights:
            warm_path = Path(args.warm_start_weights)
            if not warm_path.exists():
                raise FileNotFoundError(f"Warm-start weights not found: {warm_path}")
            model.load_weights(warm_path, skip_mismatch=False)
            print(f"Warm-started matching weights from {warm_path}")
        elif restart_reload_weights.exists():
            model.load_weights(restart_reload_weights, skip_mismatch=False)
            print(
                f"[restart {restart_idx + 1}/{args.restart_cycles}] "
                f"Reloaded {args.restart_reload_monitor} weights from {restart_reload_weights}"
            )

        map_callback = MeanAPCallback(
            val_images,
            val_samples,
            anchors,
            input_size,
            len(classes),
            train_cfg.batch_size,
            train_cfg.bbox_output_scale,
            train_cfg.obj_output_scale,
            train_cfg.cls_output_scale,
            train_cfg.logit_scale,
            train_cfg.score_threshold,
            train_cfg.objectness_threshold,
            train_cfg.class_mode,
            args.box_mode,
            args.val_map_every,
        )
        cycle_best_metric = float("-inf") if args.cycle_early_stop_monitor == "val_map50" else float("inf")
        cycle_best_epoch = -1

        for outer_idx in range(args.outer_epochs):
            if teacher is not None and float(getattr(cycle_args, "teacher_distill", 0.0)) > 0.0:
                teacher.load_weights(teacher_path, skip_mismatch=False)
                print(
                    f"[restart {restart_idx + 1}/{args.restart_cycles} "
                    f"outer {outer_idx + 1}/{args.outer_epochs}] Loaded teacher weights from {teacher_path}"
                )

            global_outer_idx = restart_idx * args.outer_epochs + outer_idx
            epoch_cursor = global_outer_idx * args.student_inner_epochs
            if args.train_augment and args.augment_constant_strength:
                augment_progress = 1.0
            elif args.train_augment:
                ramp_progress = min(1.0, float(epoch_cursor) / float(max(1, args.augment_ramp_epochs)))
                augment_progress = float(args.augment_start_strength) + (1.0 - float(args.augment_start_strength)) * ramp_progress
            else:
                augment_progress = 1.0
            advanced_ramp_epochs = int(args.advanced_augment_ramp_epochs) if int(args.advanced_augment_ramp_epochs) > 0 else int(args.augment_ramp_epochs)
            advanced_augment_prob = float(args.advanced_augment_prob) * min(1.0, float(epoch_cursor) / float(max(1, advanced_ramp_epochs)))

            train_images, train_targets = yad2k_core.build_targets(
                train_samples,
                len(classes),
                anchors,
                input_size,
                args.thermo_levels,
                augment=args.train_augment,
                augment_progress=augment_progress,
                positive_top_k=args.positive_top_k,
                num_workers=args.loader_workers,
                advanced_augmentations=args.advanced_augmentations,
                advanced_augment_prob=advanced_augment_prob,
                heads=TINIER_HEADS,
                strides=TINIER_STRIDES,
                anchor_masks=TINIER_ANCHOR_MASKS,
            )
            if teacher is not None and float(getattr(cycle_args, "teacher_distill", 0.0)) > 0.0:
                train_teacher = teacher.predict(train_images, batch_size=args.teacher_batch_size, verbose=0)
                val_teacher = teacher.predict(val_images, batch_size=args.teacher_batch_size, verbose=0)
                train_teacher = _teacher_preds_to_student_targets(
                    train_teacher,
                    teacher_anchors,
                    anchors,
                    input_size,
                    train_cfg,
                    teacher_spec,
                    args.box_mode,
                    distill_score_threshold=float(args.distill_score_threshold),
                )
                val_teacher = _teacher_preds_to_student_targets(
                    val_teacher,
                    teacher_anchors,
                    anchors,
                    input_size,
                    train_cfg,
                    teacher_spec,
                    args.box_mode,
                    distill_score_threshold=float(args.distill_score_threshold),
                )
                augmented_train_targets = _append_teacher_targets(train_targets, train_teacher)
                augmented_val_targets = _append_teacher_targets(val_targets, val_teacher)
            else:
                augmented_train_targets = train_targets
                augmented_val_targets = val_targets

            train_data = ShuffledArrayDataset(
                train_images,
                augmented_train_targets,
                args.batch_size,
                shuffle=True,
                seed=args.seed + global_outer_idx,
                repeat=args.repeat,
            )

            initial_epoch = epoch_cursor
            final_epoch = epoch_cursor + args.student_inner_epochs
            history_rows = _history_rows(history_path)
            callbacks = [
                map_callback,
                EpochMetadataCallback(restart_idx + 1, 0),
                _csv_logger(history_path, append=(restart_idx > 0 or global_outer_idx > 0)),
                GlobalBestWeightsCallback(
                    best_weights,
                    monitor="val_loss",
                    mode="min",
                    initial_best=min(
                        (_history_float(r.get("val_loss"), float("inf")) for r in history_rows),
                        default=float("inf"),
                    ),
                    verbose=1,
                ),
                GlobalBestMapWeightsCallback(
                    _artifact_path(run_dir, "voc", "best_map.weights.h5"),
                    initial_best=max(
                        (_history_float(r.get("val_map50"), float("-inf")) for r in history_rows),
                        default=float("-inf"),
                    ),
                    verbose=1,
                ),
            ]
            if args.bn_recalibrate_before_val:
                callbacks.insert(
                    0,
                    BinaryBNRecalibrationCallback(
                        train_images,
                        batch_size=args.batch_size,
                        max_batches=int(args.bn_recalibrate_batches),
                        verbose=1,
                    ),
                )
            if float(getattr(cycle_args, "teacher_distill", 0.0)) > 0.0:
                callbacks.insert(
                    1,
                    TeacherDistillBurnInCallback(
                        teacher_distill_scale_var,
                        int(getattr(cycle_args, "teacher_distill_burnin_epochs", 0)),
                    ),
                )
            model.fit(
                train_data,
                validation_data=(val_images, augmented_val_targets),
                initial_epoch=initial_epoch,
                epochs=final_epoch,
                validation_freq=max(1, int(args.validation_freq)),
                verbose=2,
                callbacks=callbacks,
            )
            model.save(_artifact_path(run_dir, "voc", "last.keras"))

            cycle_rows = [
                r
                for r in _history_rows(history_path)
                if int(_history_float(r.get("restart_cycle"), 0)) == (restart_idx + 1)
            ]
            if cycle_rows:
                if args.cycle_early_stop_monitor == "val_map50":
                    best_cycle_row = max(
                        cycle_rows,
                        key=lambda r: _history_float(r.get("val_map50"), float("-inf")),
                    )
                    best_cycle_metric = _history_float(best_cycle_row.get("val_map50"), float("-inf"))
                    metric_label = "val_map50"
                    improved = best_cycle_metric > cycle_best_metric
                else:
                    best_cycle_row = min(
                        cycle_rows,
                        key=lambda r: _history_float(r.get("val_loss"), float("inf")),
                    )
                    best_cycle_metric = _history_float(best_cycle_row.get("val_loss"), float("inf"))
                    metric_label = "val_loss"
                    improved = best_cycle_metric < cycle_best_metric
                best_cycle_epoch = int(_history_float(best_cycle_row.get("epoch"), -1))
                if improved:
                    cycle_best_metric = best_cycle_metric
                    cycle_best_epoch = best_cycle_epoch

                latest_cycle_epoch = int(_history_float(cycle_rows[-1].get("epoch"), -1))
                patience_epochs = int(args.cycle_early_stop_patience_epochs)
                if patience_epochs > 0 and cycle_best_epoch >= 0 and (latest_cycle_epoch - cycle_best_epoch) >= patience_epochs:
                    print(
                        f"[restart {restart_idx + 1}/{args.restart_cycles}] "
                        f"Early-stopping cycle after {latest_cycle_epoch - cycle_best_epoch} stagnant epochs "
                        f"(best {metric_label}={cycle_best_metric:.6f} at epoch {cycle_best_epoch})"
                    )
                    break

    if restart_reload_weights.exists():
        model.load_weights(restart_reload_weights)

    rows = _history_rows(history_path)
    best_val_map50 = max((_history_float(r.get("val_map50"), 0.0) for r in rows), default=0.0)
    final_val_map50 = _history_float(rows[-1].get("val_map50"), 0.0) if rows else 0.0
    best_val_loss = min((_history_float(r.get("val_loss"), float("inf")) for r in rows), default=float("inf"))
    final_val_loss = _history_float(rows[-1].get("val_loss"), 0.0) if rows else 0.0
    final_train_loss = _history_float(rows[-1].get("loss"), 0.0) if rows else 0.0

    summary = {
        "dataset": args.dataset,
        "dataset_format": args.dataset_format,
        "dataset_split": args.dataset_split,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "classes": classes,
        "anchors": anchors.tolist(),
        "input_size": input_size if isinstance(input_size, int) else list(input_size),
        "teacher_weights": str(teacher_path) if teacher is not None else "",
        "restart_cycles": args.restart_cycles,
        "outer_epochs": args.outer_epochs,
        "student_inner_epochs": args.student_inner_epochs,
        "repeat": args.repeat,
        "restart_reload_monitor": args.restart_reload_monitor,
        "teacher_checkpoint_monitor": args.teacher_checkpoint_monitor,
        "best_val_map50": best_val_map50,
        "final_val_map50": final_val_map50,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
        "train_cfg": train_cfg.__dict__,
        "tinier_cfg": None if args.float_model else _build_cfg(args).__dict__,
        "augment_start_strength": float(args.augment_start_strength),
        "augment_ramp_epochs": int(args.augment_ramp_epochs),
        "augment_constant_strength": bool(args.augment_constant_strength),
        "advanced_augmentations": bool(args.advanced_augmentations),
        "advanced_augment_prob": float(args.advanced_augment_prob),
        "advanced_augment_ramp_epochs": int(args.advanced_augment_ramp_epochs),
        "cycle_early_stop_patience_epochs": int(args.cycle_early_stop_patience_epochs),
        "cycle_early_stop_monitor": args.cycle_early_stop_monitor,
    }
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
