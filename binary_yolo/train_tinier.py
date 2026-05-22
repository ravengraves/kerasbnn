from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict, fields
from pathlib import Path

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import keras
from keras import ops
import numpy as np

from binary_yolo.dataset import CachedDetectionDataset, DetectionPyDataset
from binary_yolo.tiniermodel import (
    TinierBinaryConfig,
    build_binary_tinier_yolov3,
    build_float_tinier_yolov3,
)
from binary_yolo.train_fruits import (
    TrainConfig,
    _artifact_path,
    _input_hw,
    _optimizer,
    _resolve_input_size_arg,
    _save_json,
    _subset_samples,
)
import binary_yolo.yad2k_core as yad2k_core


TINIER_HEADS = ("pred_26", "pred_13")
TINIER_STRIDES = (16, 32)
TINIER_ANCHOR_MASKS = ([0, 1, 2], [3, 4, 5])

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


class MeanAPCallback(keras.callbacks.Callback):
    def __init__(
        self,
        data,
        samples,
        anchors,
        input_size: int | tuple[int, int],
        num_classes: int,
        batch_size: int,
        bbox_output_scale: float,
        obj_output_scale: float,
        cls_output_scale: float,
        logit_scale: float,
        score_threshold: float,
        objectness_threshold: float,
        class_mode: str,
        box_mode: str,
        every: int = 1,
    ):
        super().__init__()
        self.data = data
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
        self.every = max(1, int(every))
        self._last_map50 = None
        self.last_predict_seconds = 0.0
        self.last_map_seconds = 0.0
        self.last_total_seconds = 0.0
        self.last_ran = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if ((int(epoch) + 1) % self.every) != 0:
            self.last_predict_seconds = 0.0
            self.last_map_seconds = 0.0
            self.last_total_seconds = 0.0
            self.last_ran = False
            logs["val_map50"] = float("nan")
            return
        started = time.perf_counter()
        predict_started = time.perf_counter()
        preds = self.model.predict(self.data, batch_size=self.batch_size, verbose=0)
        self.last_predict_seconds = time.perf_counter() - predict_started
        map_started = time.perf_counter()
        self._last_map50 = float(
            yad2k_core.map50(
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
                heads=TINIER_HEADS,
                anchor_masks=TINIER_ANCHOR_MASKS,
            )
        )
        self.last_map_seconds = time.perf_counter() - map_started
        self.last_total_seconds = time.perf_counter() - started
        self.last_ran = True
        logs["val_map50"] = self._last_map50


class EpochTimingCallback(keras.callbacks.Callback):
    def __init__(self, train_data, val_data, map_callback: MeanAPCallback):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.map_callback = map_callback
        self._epoch_started = 0.0
        self._val_started = 0.0
        self._val_seconds = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_started = time.perf_counter()
        self._val_seconds = 0.0
        if hasattr(self.train_data, "consume_profile"):
            self.train_data.consume_profile()
        if hasattr(self.val_data, "consume_profile"):
            self.val_data.consume_profile()

    def on_test_begin(self, logs=None):
        self._val_started = time.perf_counter()

    def on_test_end(self, logs=None):
        self._val_seconds += time.perf_counter() - self._val_started

    def on_epoch_end(self, epoch, logs=None):
        epoch_seconds = time.perf_counter() - self._epoch_started
        train_profile = self.train_data.consume_profile() if hasattr(self.train_data, "consume_profile") else {}
        val_profile = self.val_data.consume_profile() if hasattr(self.val_data, "consume_profile") else {}
        logs = logs or {}
        print(
            "[timing] "
            f"epoch={epoch} total={epoch_seconds:.2f}s "
            f"val_fit={self._val_seconds:.2f}s "
            f"map_total={self.map_callback.last_total_seconds:.2f}s "
            f"map_predict={self.map_callback.last_predict_seconds:.2f}s "
            f"map_post={self.map_callback.last_map_seconds:.2f}s "
            f"train_get={train_profile.get('getitem_seconds', 0.0):.2f}s/{train_profile.get('getitem_calls', 0)} "
            f"train_refresh_full={train_profile.get('refresh_full_seconds', 0.0):.2f}s/{train_profile.get('refresh_full_calls', 0)} "
            f"train_refresh_partial={train_profile.get('refresh_partial_seconds', 0.0):.2f}s/{train_profile.get('refresh_partial_calls', 0)} "
            f"train_refresh_items={train_profile.get('refresh_partial_items', 0)} "
            f"val_get={val_profile.get('getitem_seconds', 0.0):.2f}s/{val_profile.get('getitem_calls', 0)}"
        )


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


class BinaryBNRecalibrationCallback(keras.callbacks.Callback):
    def __init__(self, calibration_data, batch_size: int, max_batches: int = 0, verbose: int = 1):
        super().__init__()
        self.calibration_data = calibration_data
        self.batch_size = max(1, int(batch_size))
        self.max_batches = max(0, int(max_batches))
        self.verbose = int(verbose)

    @staticmethod
    def _flatten_layers(model):
        flatten = getattr(model, "_flatten_layers", None)
        if flatten is None:
            return list(getattr(model, "layers", []))
        return list(flatten(include_self=False, recursive=True))

    def _collect_affine_norm_layers(self):
        layers = []
        seen = set()
        for layer in self._flatten_layers(self.model):
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            if (
                hasattr(layer, "use_running_stats")
                and hasattr(layer, "moving_mean")
                and hasattr(layer, "moving_variance")
                and hasattr(layer, "momentum")
            ):
                if bool(getattr(layer, "use_running_stats")):
                    layers.append(layer)
        return layers

    def _iter_calibration_batches(self):
        data = self.calibration_data
        if data is None:
            return
        if hasattr(data, "images") and hasattr(data, "base_count"):
            images = data.images[: int(data.base_count)]
            total = images.shape[0]
            for start in range(0, total, self.batch_size):
                yield images[start : start + self.batch_size]
            return
        if isinstance(data, np.ndarray):
            total = int(data.shape[0])
            for start in range(0, total, self.batch_size):
                yield data[start : start + self.batch_size]
            return
        if isinstance(data, (tuple, list)) and len(data) >= 1 and isinstance(data[0], np.ndarray):
            images = data[0]
            total = int(images.shape[0])
            for start in range(0, total, self.batch_size):
                yield images[start : start + self.batch_size]
            return
        length = len(data)
        for index in range(length):
            batch = data[index]
            if isinstance(batch, (tuple, list)):
                yield batch[0]
            else:
                yield batch

    def on_test_begin(self, logs=None):
        layers = self._collect_affine_norm_layers()
        if not layers:
            return
        original_momenta = [float(layer.momentum) for layer in layers]
        batch_count = 0
        for batch in self._iter_calibration_batches() or ():
            batch_count += 1
            momentum = 0.0 if batch_count <= 1 else float(batch_count - 1) / float(batch_count)
            for layer in layers:
                layer.momentum = momentum
            _ = self.model(batch, training=True)
            if self.max_batches and batch_count >= self.max_batches:
                break
        for layer, momentum in zip(layers, original_momenta):
            layer.momentum = momentum
        if self.verbose:
            print(f"[bn-recalibrate] updated running stats using {batch_count} calibration batches")


class GlobalBestWeightsCallback(keras.callbacks.Callback):
    def __init__(self, filepath: Path, monitor: str = "val_loss", mode: str = "min", initial_best: float | None = None, verbose: int = 1):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.verbose = int(verbose)
        if self.mode not in {"min", "max"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if initial_best is None:
            initial_best = float("inf") if self.mode == "min" else float("-inf")
        self.best = float(initial_best)

    def _is_improvement(self, value: float) -> bool:
        return value < self.best if self.mode == "min" else value > self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        raw = logs.get(self.monitor)
        if raw is None:
            return
        current = float(raw)
        if not self._is_improvement(current):
            return
        self.best = current
        self.model.save_weights(self.filepath)
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: global best {self.monitor} improved to {current:.6f}, saving to {self.filepath}")


class GlobalBestMapWeightsCallback(keras.callbacks.Callback):
    def __init__(self, filepath: Path, initial_best: float | None = None, verbose: int = 1):
        super().__init__()
        self.filepath = Path(filepath)
        self.verbose = int(verbose)
        if initial_best is None:
            initial_best = float("-inf")
        self.best = float(initial_best)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        raw = logs.get("val_map50")
        if raw is None:
            return
        current = _history_float(raw, float("-inf"))
        if not math.isfinite(current):
            return
        if current <= self.best:
            return
        self.best = current
        self.model.save_weights(self.filepath)
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: global best val_map50 improved to {current:.6f}, saving to {self.filepath}")


class TeacherDistillBurnInCallback(keras.callbacks.Callback):
    def __init__(self, scale_var, burnin_epochs: int):
        super().__init__()
        self.scale_var = scale_var
        self.burnin_epochs = int(burnin_epochs)

    def _scale_for_epoch(self, epoch: int) -> float:
        if self.burnin_epochs <= 0:
            return 1.0
        return 1.0 if int(epoch) < self.burnin_epochs else 0.0

    def on_train_begin(self, logs=None):
        self.scale_var.assign(self._scale_for_epoch(0))

    def on_epoch_begin(self, epoch, logs=None):
        self.scale_var.assign(self._scale_for_epoch(epoch))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="none", choices=("none", "teacher_float_voc256"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-format", default="auto")
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
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--restart-cycles", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--subset-train-as-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--positive-top-k", type=int, default=3)
    ap.add_argument("--optimizer", default="adam", choices=("adam", "adamw", "rmsprop", "sgd"))
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--schedule", default="constant", choices=("constant", "cosine", "cosine_restart", "exp"))
    ap.add_argument("--class-mode", default="softmax", choices=("softmax", "bce"))
    ap.add_argument("--box-mode", default="linear", choices=("linear", "linear_float", "yolo"))
    ap.add_argument("--obj-gain", type=float, default=4.0)
    ap.add_argument("--obj-pos-gain", type=float, default=8.0)
    ap.add_argument("--noobj-weight", type=float, default=0.04)
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
    ap.add_argument("--loss-logit-scale", type=float, default=0.0)
    ap.add_argument("--rescore-objectness", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--hard-negative-ratio", type=float, default=0.0)
    ap.add_argument("--hard-negative-min", type=int, default=1024)
    ap.add_argument("--conf-neg-normalizer", default="pos", choices=("neg", "pos", "none"))
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--objectness-threshold", type=float, default=0.25)
    ap.add_argument("--val-map-every", type=int, default=1)
    ap.add_argument("--validation-freq", type=int, default=1)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--input-levels", type=int, default=16)
    ap.add_argument("--input-signed", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--quantized-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--input-dropcut", type=float, default=0.0)
    ap.add_argument("--input-dropcut-size-min", type=float, default=0.05)
    ap.add_argument("--input-dropcut-size-max", type=float, default=0.20)
    ap.add_argument("--input-dropcut-method", default="both", choices=("black", "white", "both", "noise"))
    ap.add_argument("--input-dropcut-boxes", type=int, default=1)
    ap.add_argument("--use-bn-in-fire", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-momentum", type=float, default=0.8)
    ap.add_argument("--bn-recalibrate-before-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-recalibrate-batches", type=int, default=0)
    ap.add_argument("--head-pre-filters", type=int, default=256)
    ap.add_argument("--head-channels", type=int, default=64)
    ap.add_argument("--obj-cls-hidden", type=int, default=128)
    ap.add_argument("--obj-remap-hidden", type=int, default=64)
    ap.add_argument("--obj-bias-init", type=float, default=0.0)
    ap.add_argument("--train-augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--augment-start-strength", type=float, default=0.2)
    ap.add_argument("--augment-ramp-epochs", type=int, default=100)
    ap.add_argument("--augment-constant-strength", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--advanced-augmentations", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--advanced-augment-prob", type=float, default=0.05)
    ap.add_argument("--advanced-augment-ramp-epochs", type=int, default=0)
    ap.add_argument("--teacher-weights", default="")
    ap.add_argument("--teacher-run-dir", default="")
    ap.add_argument("--teacher-checkpoint-monitor", default="val_map50", choices=("val_loss", "val_map50"))
    ap.add_argument("--teacher-use-bn-in-fire", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--teacher-distill", type=float, default=0.001)
    ap.add_argument("--teacher-bbox-distill", type=float, default=0.05)
    ap.add_argument("--teacher-obj-distill", type=float, default=0.1)
    ap.add_argument("--teacher-cls-distill", type=float, default=1.0)
    ap.add_argument("--teacher-distill-burnin-epochs", type=int, default=8)
    ap.add_argument("--distill-score-threshold", type=float, default=0.25)
    ap.add_argument("--teacher-distill-mode", default="decoded_dense", choices=("decoded_dense", "topk_align"))
    ap.add_argument("--teacher-align-top-k-teacher", type=int, default=16)
    ap.add_argument("--teacher-align-top-k-student", type=int, default=64)
    ap.add_argument("--teacher-align-student-pull", type=float, default=0.0)
    ap.add_argument("--loader-workers", type=int, default=2)
    ap.add_argument("--loader-queue-size", type=int, default=64)
    ap.add_argument("--loader-multiprocessing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-cache-refresh-epochs", type=int, default=0)
    ap.add_argument("--train-cache-refresh-divisor", type=int, default=33)
    ap.add_argument("--backup-and-restore", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--auto-scale-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--clipnorm", type=float, default=1.0)
    ap.add_argument("--dtype-policy", default="mixed_float16", choices=("mixed_float16", "float32"))
    ap.add_argument("--warm-start-weights", default="")
    ap.add_argument("--warm-start-source-model", default="auto", choices=("auto", "binary", "float"))
    ap.add_argument("--restart-reload-monitor", default="val_map50", choices=("val_loss", "val_map50"))
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    _apply_preset(args)
    return args


def _apply_preset(args):
    if args.preset == "teacher_float_voc256":
        # Match the legacy float teacher recipe that reaches >0.95 mAP@50 on
        # the 5% VOC fit test. Keep the simple head and float32 training
        # semantics; the newer widened float head and mixed-float16 path were
        # the main regressions.
        args.float_model = True
        args.dtype_policy = "float32"
        args.input_size = 256
        args.input_width = 0
        args.input_height = 0
        args.use_bn_in_fire = False
        # The historical >0.95 subset-fit teacher used the legacy subset path,
        # which effectively meant no train augmentation. Keep that behavior in
        # this preset so the smoke test remains a reliable training-health
        # check. Full teacher runs should pass their augmentation settings
        # explicitly rather than relying on this preset.
        args.train_augment = False
        args.augment_start_strength = 0.2
        args.augment_ramp_epochs = 100
        args.epochs = 16
        args.batch_size = 8
        args.optimizer = "adam"
        args.lr = 5e-4
        args.schedule = "constant"
        args.class_mode = "softmax"
        args.box_mode = "yolo"
        args.logit_scale = 1.0
        args.loss_logit_scale = 1.0
        args.obj_gain = 4.0
        args.obj_pos_gain = 8.0
        args.noobj_weight = 0.04
        args.cls_gain = 8.0
        args.box_gain = 5.0
        args.box_iou_aux_gain = 0.0
        args.hard_negative_min = 1024
        args.conf_neg_normalizer = "pos"
        args.positive_top_k = 1
        args.score_threshold = 0.05
        args.objectness_threshold = 0.05
        args.clipnorm = 1.0
        args.val_map_every = 1
        args.head_pre_filters = 0
        args.head_channels = 0
        args.obj_cls_hidden = 0
        args.obj_remap_hidden = 0
        args.loader_workers = max(args.loader_workers, 2)
        args.loader_queue_size = max(args.loader_queue_size, 64)
        args.loader_multiprocessing = False
        args.backup_and_restore = False
        # Standalone teacher training should never inherit student
        # distillation defaults.
        args.teacher_distill = 0.0
        args.teacher_bbox_distill = 0.0
        args.teacher_obj_distill = 0.0
        args.teacher_cls_distill = 0.0


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
        train_samples, val_samples, test_samples = yad2k_core.split_samples_three_way(
            merged,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    else:
        test_samples = []
    train_samples = _subset_samples(train_samples, args.train_sample_fraction, args.seed)
    val_samples = _subset_samples(val_samples, args.val_sample_fraction, args.seed + 1)
    test_samples = _subset_samples(test_samples, args.test_sample_fraction, args.seed + 2) if test_samples else []
    if args.train_sample_count > 0:
        train_samples = train_samples[: args.train_sample_count]
    if args.subset_train_as_val:
        val_samples = list(train_samples)
    return train_samples, val_samples, test_samples, classes


def _build_cfg(args) -> TinierBinaryConfig:
    kwargs = dict(
        thermo_levels=args.thermo_levels,
        input_levels=args.input_levels,
        input_signed=args.input_signed,
        float_input=getattr(args, "float_input", False),
        float_stem_first_layer=getattr(args, "float_stem_first_layer", False),
        quantized_input=args.quantized_input,
        input_dropcut=args.input_dropcut,
        input_dropcut_size=(args.input_dropcut_size_min, args.input_dropcut_size_max),
        input_dropcut_method=args.input_dropcut_method,
        input_dropcut_boxes=args.input_dropcut_boxes,
        early_quantized_layers=getattr(args, "early_quantized_layers", 1),
        activation_surrogate=getattr(args, "activation_surrogate", "triangular"),
        activation_surrogate_gain=getattr(args, "activation_surrogate_gain", 1.0),
        weight_surrogate=getattr(args, "weight_surrogate", "magnitude_aware"),
        weight_surrogate_floor=getattr(args, "weight_surrogate_floor", 0.25),
        weight_surrogate_gain=getattr(args, "weight_surrogate_gain", 1.0),
        norm_gamma_init=getattr(args, "norm_gamma_init", 0.25),
        use_rprelu=getattr(args, "use_rprelu", True),
        rprelu_init_beta=getattr(args, "rprelu_init_beta", 0.25),
        rprelu_init_gamma=getattr(args, "rprelu_init_gamma", 0.0),
        rprelu_init_zeta=getattr(args, "rprelu_init_zeta", 0.0),
        binary_kernel_init_scale=getattr(args, "binary_kernel_init_scale", 0.0),
        activation_levels=getattr(args, "activation_levels", 128),
        output_levels=getattr(args, "output_levels", 128),
        binary_batchnorm=getattr(args, "binary_batchnorm", True),
        use_bn_in_fire=args.use_bn_in_fire,
        bn_momentum=args.bn_momentum,
        bn_quantize_input=getattr(args, "bn_quantize_input", False),
        bn_quantize_centered=getattr(args, "bn_quantize_centered", False),
        bn_quantize_normalized=getattr(args, "bn_quantize_normalized", False),
        bn_quantize_output=getattr(args, "bn_quantize_output", False),
        bn_quantize_eval_stats=getattr(args, "bn_quantize_eval_stats", False),
        bn_eval_use_batch_stats=getattr(args, "bn_eval_use_batch_stats", False),
        bn_force_full_quant_eval=getattr(args, "bn_force_full_quant_eval", False),
        bn_quantize_affine_params=getattr(args, "bn_quantize_affine_params", True),
        stem_output_levels=getattr(args, "stem_output_levels", 0),
        head_pre_filters=args.head_pre_filters,
        head_channels=args.head_channels,
        head_proj_levels=getattr(args, "head_proj_levels", 128),
        obj_cls_hidden=args.obj_cls_hidden,
        obj_remap_hidden=args.obj_remap_hidden,
        obj_bias_init=args.obj_bias_init,
        backbone_width_mult=getattr(args, "backbone_width_mult", 1.0),
        tinyv3_backbone=getattr(args, "tinyv3_backbone", False),
        float_neck_head=getattr(args, "float_neck_head", False),
        direct_quantized_head=getattr(args, "direct_quantized_head", False),
        late_quantized_from=getattr(args, "late_quantized_from", 0),
        late_conv_levels=getattr(args, "late_conv_levels", 128),
        late_act_levels=getattr(args, "late_act_levels", 16),
        bbox_output_scale=args.bbox_output_scale,
        obj_output_scale=args.obj_output_scale,
        cls_output_scale=args.cls_output_scale,
        logit_scale=args.logit_scale,
    )
    supported = {f.name for f in fields(TinierBinaryConfig)}
    return TinierBinaryConfig(**{k: v for k, v in kwargs.items() if k in supported})


def _build_model(num_classes: int, input_size, args):
    input_shape = (*_input_hw(input_size), 3)
    if args.float_model:
        return build_float_tinier_yolov3(
            num_classes,
            input_shape=input_shape,
            use_bn_in_fire=args.use_bn_in_fire,
            head_pre_filters=args.head_pre_filters,
            head_channels=args.head_channels,
            obj_cls_hidden=args.obj_cls_hidden,
            obj_remap_hidden=args.obj_remap_hidden,
            obj_bias_init=args.obj_bias_init,
        )
    return build_binary_tinier_yolov3(
        num_classes,
        input_shape=input_shape,
        cfg=_build_cfg(args),
    )


def _transfer_float_to_binary_tinier(binary_model, weights_path: Path, num_classes: int, input_size, args):
    cfg = _build_cfg(args)
    source_use_bn = "floatbn" in weights_path.as_posix()
    source = build_float_tinier_yolov3(
        num_classes,
        input_shape=(*_input_hw(input_size), 3),
        use_bn_in_fire=source_use_bn,
        head_pre_filters=args.head_pre_filters,
        head_channels=args.head_channels,
        obj_cls_hidden=args.obj_cls_hidden,
        obj_remap_hidden=args.obj_remap_hidden,
    )
    source.load_weights(weights_path, skip_mismatch=True)
    copied = []

    def _normalized_kernel(src_kernel, dst_shape):
        kernel = np.asarray(src_kernel, dtype=np.float32)
        if kernel.shape != dst_shape:
            if kernel.ndim == 4 and dst_shape[2] == kernel.shape[2] * cfg.thermo_levels:
                kernel = np.repeat(kernel, cfg.thermo_levels, axis=2)
            else:
                raise ValueError(f"Cannot map kernel shape {kernel.shape} -> {dst_shape}")
        scale = max(float(np.max(np.abs(kernel))), 1e-6)
        return np.clip(kernel / scale, -1.0, 1.0).astype(np.float32)

    def _bn_to_affine(bn_weights):
        gamma, beta, moving_mean, moving_variance = [np.asarray(w, dtype=np.float32) for w in bn_weights]
        gamma_safe = np.where(np.abs(gamma) > 1e-3, gamma, np.where(gamma >= 0.0, 1e-3, -1e-3)).astype(np.float32)
        norm_scale = np.sign(gamma_safe).astype(np.float32)
        norm_threshold = np.zeros_like(beta, dtype=np.float32)
        bias = (beta / gamma_safe).astype(np.float32)
        moving_variance = np.maximum(moving_variance.astype(np.float32), 1e-4)
        return [norm_scale, norm_threshold, bias, moving_mean.astype(np.float32), moving_variance]

    def _bias_to_affine(bias):
        bias = np.asarray(bias, dtype=np.float32)
        return [
            np.ones_like(bias, dtype=np.float32),
            np.zeros_like(bias, dtype=np.float32),
            bias.astype(np.float32),
            np.zeros_like(bias, dtype=np.float32),
            np.ones_like(bias, dtype=np.float32),
        ]

    for dst in binary_model.layers:
        try:
            src_conv = source.get_layer(f"{dst.name}_conv")
        except ValueError:
            continue
        if not hasattr(dst, "weights") or not dst.weights:
            continue
        dst_weights = dst.get_weights()
        if not dst_weights:
            continue
        try:
            kernel = _normalized_kernel(src_conv.get_weights()[0], dst_weights[0].shape)
        except ValueError:
            continue
        new_weights = [kernel]
        if len(dst_weights) >= 6:
            try:
                src_bn = source.get_layer(f"{dst.name}_bn")
                new_weights.extend(_bn_to_affine(src_bn.get_weights()))
            except ValueError:
                src_conv_weights = src_conv.get_weights()
                if len(src_conv_weights) >= 2 and src_conv_weights[1].shape == dst_weights[3].shape:
                    new_weights.extend(_bias_to_affine(src_conv_weights[1]))
                else:
                    new_weights.extend(dst_weights[1:6])
        elif len(dst_weights) > 1:
            new_weights.extend(dst_weights[1:])
        dst.set_weights(new_weights)
        copied.append(dst.name)

    print(
        f"Warm-started {len(copied)} Tinier binary layers from float source: "
        f"{copied[:16]}{' ...' if len(copied) > 16 else ''}"
    )


def _build_train_cfg(args) -> TrainConfig:
    return TrainConfig(
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
        box_iou_aux_gain=args.box_iou_aux_gain,
        box_iou_aux_mode=args.box_iou_aux_mode,
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
        positive_top_k=args.positive_top_k,
        box_mode=args.box_mode,
        score_threshold=args.score_threshold,
        objectness_threshold=args.objectness_threshold,
        train_augment=args.train_augment,
        jit_compile=args.jit_compile,
        auto_scale_loss=args.auto_scale_loss,
        clipnorm=args.clipnorm,
    )


def _compile(model, anchors, num_classes: int, input_size, train_cfg: TrainConfig, steps: int, box_mode: str):
    model.compile(
        optimizer=_optimizer(train_cfg, steps),
        loss=yad2k_core.make_losses(
            anchors,
            input_size,
            num_classes,
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
            box_mode=box_mode,
            heads=TINIER_HEADS,
            strides=TINIER_STRIDES,
            anchor_masks=TINIER_ANCHOR_MASKS,
        ),
        jit_compile=train_cfg.jit_compile,
        steps_per_execution=max(1, min(steps, 64)),
        auto_scale_loss=train_cfg.auto_scale_loss,
    )


def _build_teacher(num_classes: int, input_size, args):
    spec = _resolve_teacher_spec(input_size, args)
    teacher = build_float_tinier_yolov3(
        num_classes,
        input_shape=(*_input_hw(spec["input_size"]), 3),
        use_bn_in_fire=spec["use_bn_in_fire"],
        head_pre_filters=spec["head_pre_filters"],
        head_channels=spec["head_channels"],
        obj_cls_hidden=spec["obj_cls_hidden"],
        obj_remap_hidden=spec["obj_remap_hidden"],
    )
    teacher.load_weights(args.teacher_weights, skip_mismatch=False)
    return teacher, spec


def _teacher_spec_path(run_dir: Path) -> Path:
    return run_dir / "teacher_spec.json"


def _teacher_weights_path(args) -> Path:
    if getattr(args, "teacher_run_dir", ""):
        suffix = "best_map.weights.h5" if getattr(args, "teacher_checkpoint_monitor", "val_map50") == "val_map50" else "best.weights.h5"
        return Path(args.teacher_run_dir) / f"model.voc.{suffix}"
    return Path(args.teacher_weights)


def _restart_reload_path(run_dir: Path, dataset_tag: str, monitor: str) -> Path:
    suffix = "best_map.weights.h5" if monitor == "val_map50" else "best.weights.h5"
    return _artifact_path(run_dir, dataset_tag, suffix)


def _normalize_input_size_value(input_size):
    if isinstance(input_size, (tuple, list)):
        return [int(input_size[0]), int(input_size[1])]
    return int(input_size)


def _teacher_spec_from_args(input_size, train_cfg: TrainConfig | None, args):
    return {
        "input_size": _normalize_input_size_value(input_size),
        "use_bn_in_fire": bool(getattr(args, "use_bn_in_fire", getattr(args, "teacher_use_bn_in_fire", False))),
        "head_pre_filters": int(getattr(args, "head_pre_filters", 256)),
        "head_channels": int(getattr(args, "head_channels", 64)),
        "obj_cls_hidden": int(getattr(args, "obj_cls_hidden", 128)),
        "obj_remap_hidden": int(getattr(args, "obj_remap_hidden", 64)),
        "box_mode": str(getattr(args, "box_mode", "linear")),
        "class_mode": str(getattr(args, "class_mode", "softmax")),
        "logit_scale": float(getattr(args, "logit_scale", 32.0)),
        "loss_logit_scale": float(getattr(args, "loss_logit_scale", getattr(args, "logit_scale", 32.0))),
        "bbox_output_scale": float(getattr(args, "bbox_output_scale", getattr(train_cfg, "bbox_output_scale", 2.0) if train_cfg else 2.0)),
        "obj_output_scale": float(getattr(args, "obj_output_scale", getattr(train_cfg, "obj_output_scale", 64.0) if train_cfg else 64.0)),
        "cls_output_scale": float(getattr(args, "cls_output_scale", getattr(train_cfg, "cls_output_scale", 32.0) if train_cfg else 32.0)),
        "dtype_policy": str(getattr(args, "dtype_policy", "float32")),
    }


def _resolve_teacher_spec(input_size, args):
    teacher_run_dir = str(getattr(args, "teacher_run_dir", "") or "")
    if teacher_run_dir:
        spec_path = _teacher_spec_path(Path(teacher_run_dir))
        if not spec_path.exists():
            raise FileNotFoundError(f"Teacher spec not found: {spec_path}")
        spec = json.loads(spec_path.read_text())
        spec["input_size"] = spec.get("input_size", _normalize_input_size_value(input_size))
        return spec
    return _teacher_spec_from_args(input_size, None, args)


def _teacher_head_spec(head: str):
    idx = TINIER_HEADS.index(head)
    mask = np.asarray(TINIER_ANCHOR_MASKS[idx], np.int32)
    stride = int(TINIER_STRIDES[idx])
    return mask, stride


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x, dtype=np.float32)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-6)


def _int8_sat_np(x: np.ndarray, output_scale: float) -> np.ndarray:
    x = np.asarray(x, np.float32) / np.float32(output_scale)
    return np.clip(x, -127.0, 127.0).astype(np.float32)


def _uint01_np(x: np.ndarray, output_scale: float) -> np.ndarray:
    x = _int8_sat_np(x, output_scale)
    return ((x + 127.0) / 254.0).astype(np.float32)


def _piecewise_ratio_from_u_np(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, np.float32)
    return np.where(u <= 0.5, 2.0 * u, 1.0 + 4.0 * (u - 0.5)).astype(np.float32)


def _piecewise_u_from_ratio_np(ratio: np.ndarray) -> np.ndarray:
    ratio = np.asarray(ratio, np.float32)
    return np.where(ratio <= 1.0, 0.5 * ratio, 0.5 + 0.25 * (ratio - 1.0)).astype(np.float32)


def _decode_teacher_head(pred: np.ndarray, head: str, anchors: np.ndarray, input_size, teacher_spec: dict, distill_score_threshold: float = 0.25):
    pred = np.asarray(pred, np.float32)
    h, w = pred.shape[1:3]
    input_h, input_w = _input_hw(input_size)
    input_wh = np.asarray([input_w, input_h], np.float32)
    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    grid_xy = np.stack([gx, gy], axis=-1).reshape(1, h, w, 1, 2)
    grid_wh = np.asarray([w, h], np.float32).reshape(1, 1, 1, 1, 2)
    mask, _ = _teacher_head_spec(head)
    anchor_wh = anchors[mask].astype(np.float32).reshape(1, 1, 1, len(mask), 2) / input_wh
    box_mode = str(teacher_spec["box_mode"])
    logit_scale = float(teacher_spec["logit_scale"])
    class_mode = str(teacher_spec["class_mode"])

    if box_mode == "yolo":
        xy = (_sigmoid_np(pred[..., :2]) + grid_xy) / grid_wh
        wh = np.exp(np.clip(pred[..., 2:4], -6.0, 6.0).astype(np.float32)) * anchor_wh
        obj_logits = pred[..., 4:5] / logit_scale
        cls_logits = pred[..., 5:] / logit_scale
    elif box_mode == "linear_float":
        xy_u = np.clip(pred[..., :2], 0.0, 1.0).astype(np.float32)
        wh_u = np.clip(pred[..., 2:4], 0.0, 1.0).astype(np.float32)
        xy = (xy_u + grid_xy) / grid_wh
        wh = _piecewise_ratio_from_u_np(wh_u) * anchor_wh
        obj_logits = pred[..., 4:5] / logit_scale
        cls_logits = pred[..., 5:] / logit_scale
    else:
        bbox_output_scale = float(teacher_spec["bbox_output_scale"])
        obj_output_scale = float(teacher_spec["obj_output_scale"])
        cls_output_scale = float(teacher_spec["cls_output_scale"])
        xy = (_uint01_np(pred[..., :2], bbox_output_scale) + grid_xy) / grid_wh
        wh = _piecewise_ratio_from_u_np(_uint01_np(pred[..., 2:4], bbox_output_scale)) * anchor_wh
        obj_logits = _int8_sat_np(pred[..., 4:5], obj_output_scale) / logit_scale
        cls_logits = _int8_sat_np(pred[..., 5:], cls_output_scale) / logit_scale

    obj_conf = _sigmoid_np(obj_logits)
    if class_mode == "softmax":
        cls_prob = _softmax_np(cls_logits)
        cls_conf = np.max(cls_prob, axis=-1, keepdims=True)
    else:
        cls_prob = _sigmoid_np(cls_logits)
        cls_conf = np.max(cls_prob, axis=-1, keepdims=True)
    # For bbox/class, trust teacher objectness more than score so class
    # uncertainty does not suppress otherwise useful localization targets.
    active = (obj_conf >= float(distill_score_threshold)).astype(np.float32)
    bbox_cls_weight = (active * obj_conf).astype(np.float32)
    obj_weight = active.astype(np.float32)
    return (
        xy.astype(np.float32),
        wh.astype(np.float32),
        obj_conf.astype(np.float32),
        cls_prob.astype(np.float32),
        bbox_cls_weight,
        obj_weight,
    )


def _encode_teacher_for_student(head: str, pred: np.ndarray, teacher_anchors: np.ndarray, student_anchors: np.ndarray, input_size, train_cfg: TrainConfig, teacher_spec: dict, student_box_mode: str, distill_score_threshold: float = 0.25):
    del student_anchors, train_cfg, student_box_mode
    xy, wh, obj_prob, cls_prob, bbox_cls_weight, obj_weight = _decode_teacher_head(
        pred,
        head,
        teacher_anchors,
        input_size,
        teacher_spec,
        distill_score_threshold=distill_score_threshold,
    )
    bbox = np.concatenate([xy, wh], axis=-1).astype(np.float32)
    return np.concatenate([bbox, obj_prob.astype(np.float32), cls_prob.astype(np.float32), bbox_cls_weight, obj_weight], axis=-1)


def _teacher_preds_to_student_targets(teacher_preds: dict[str, np.ndarray], teacher_anchors: np.ndarray, student_anchors: np.ndarray, input_size, train_cfg: TrainConfig, teacher_spec: dict, student_box_mode: str, distill_score_threshold: float = 0.25):
    converted = {}
    for head, pred in teacher_preds.items():
        converted[head] = _encode_teacher_for_student(head, pred, teacher_anchors, student_anchors, input_size, train_cfg, teacher_spec, student_box_mode, distill_score_threshold=distill_score_threshold)
    return converted


def _append_teacher_targets(base_targets: dict[str, np.ndarray], teacher_preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    augmented = {}
    for head, values in base_targets.items():
        augmented[head] = np.concatenate([values, teacher_preds[head].astype(np.float32)], axis=-1)
    return augmented


def _effective_distill_args(args, restart_idx: int):
    if int(getattr(args, "teacher_distill_burnin_epochs", 0)) <= 0:
        return args
    if int(restart_idx) == 0:
        return args
    effective = argparse.Namespace(**vars(args))
    effective.teacher_distill = 0.0
    effective.teacher_bbox_distill = 0.0
    effective.teacher_obj_distill = 0.0
    effective.teacher_cls_distill = 0.0
    return effective


def _xywh_to_xyxy_ops(boxes):
    half_wh = boxes[..., 2:4] * 0.5
    mins = boxes[..., :2] - half_wh
    maxs = boxes[..., :2] + half_wh
    return ops.concatenate([mins, maxs], axis=-1)


def _pairwise_iou_xywh_ops(student_boxes, teacher_boxes, eps: float = 1e-6):
    student_xyxy = _xywh_to_xyxy_ops(student_boxes)
    teacher_xyxy = _xywh_to_xyxy_ops(teacher_boxes)
    student_xyxy = ops.expand_dims(student_xyxy, axis=2)
    teacher_xyxy = ops.expand_dims(teacher_xyxy, axis=1)
    inter_mins = ops.maximum(student_xyxy[..., :2], teacher_xyxy[..., :2])
    inter_maxs = ops.minimum(student_xyxy[..., 2:4], teacher_xyxy[..., 2:4])
    inter_wh = ops.maximum(inter_maxs - inter_mins, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    student_area = (student_xyxy[..., 2] - student_xyxy[..., 0]) * (student_xyxy[..., 3] - student_xyxy[..., 1])
    teacher_area = (teacher_xyxy[..., 2] - teacher_xyxy[..., 0]) * (teacher_xyxy[..., 3] - teacher_xyxy[..., 1])
    union = ops.maximum(student_area + teacher_area - inter_area, eps)
    return inter_area / union


def _pairwise_soft_class_ce_ops(student_cls, teacher_cls, class_mode: str, eps: float = 1e-6):
    teacher = ops.expand_dims(teacher_cls, axis=1)
    student = ops.expand_dims(student_cls, axis=2)
    if class_mode == "softmax":
        return -ops.sum(teacher * ops.log(ops.clip(student, eps, 1.0)), axis=-1)
    student = ops.clip(student, eps, 1.0 - eps)
    return -ops.sum(
        teacher * ops.log(student) + (1.0 - teacher) * ops.log(1.0 - student),
        axis=-1,
    )


def _pairwise_soft_binary_ce_ops(student_obj, teacher_obj, eps: float = 1e-6):
    student = ops.clip(ops.expand_dims(student_obj, axis=2), eps, 1.0 - eps)
    teacher = ops.expand_dims(teacher_obj, axis=1)
    return -(
        teacher * ops.log(student)
        + (1.0 - teacher) * ops.log(1.0 - student)
    )[..., 0]


def _take_topk_ops(values, scores, k: int):
    if k <= 0:
        return values, scores
    total = int(values.shape[1])
    if k >= total:
        return values, scores
    _, indices = ops.top_k(scores[..., 0], k=k, sorted=True)
    value_idx = ops.expand_dims(indices, axis=-1)
    value_idx = ops.repeat(value_idx, values.shape[-1], axis=-1)
    score_idx = ops.expand_dims(indices, axis=-1)
    return (
        ops.take_along_axis(values, value_idx, axis=1),
        ops.take_along_axis(scores, score_idx, axis=1),
    )


def _teacher_topk_alignment_loss_ops(
    student_boxes,
    student_obj,
    student_cls,
    teacher_boxes,
    teacher_obj,
    teacher_cls,
    *,
    class_mode: str,
    bbox_gain: float,
    obj_gain: float,
    cls_gain: float,
    teacher_top_k: int,
    student_top_k: int,
    student_pull: float,
    eps: float = 1e-6,
):
    teacher_score = ops.stop_gradient(teacher_obj * ops.max(teacher_cls, axis=-1, keepdims=True))
    student_score = ops.stop_gradient(student_obj)
    teacher_values = ops.concatenate([teacher_boxes, teacher_obj, teacher_cls], axis=-1)
    student_values = ops.concatenate([student_boxes, student_obj, student_cls], axis=-1)
    teacher_values, teacher_score = _take_topk_ops(teacher_values, teacher_score, teacher_top_k)
    student_values, student_score = _take_topk_ops(student_values, student_score, student_top_k)
    box_dim = 4
    obj_dim = 1
    teacher_boxes = teacher_values[..., :box_dim]
    teacher_obj = teacher_values[..., box_dim : box_dim + obj_dim]
    teacher_cls = teacher_values[..., box_dim + obj_dim :]
    student_boxes = student_values[..., :box_dim]
    student_obj = student_values[..., box_dim : box_dim + obj_dim]
    student_cls = student_values[..., box_dim + obj_dim :]
    pair_iou = _pairwise_iou_xywh_ops(student_boxes, teacher_boxes, eps=eps)
    pair_cls = _pairwise_soft_class_ce_ops(student_cls, teacher_cls, class_mode=class_mode, eps=eps)
    pair_obj = _pairwise_soft_binary_ce_ops(student_obj, teacher_obj, eps=eps)
    cost = (
        float(bbox_gain) * (1.0 - pair_iou)
        + float(cls_gain) * pair_cls
        + float(obj_gain) * pair_obj
    )
    teacher_match_cost = ops.min(cost, axis=1)
    teacher_w = ops.squeeze(teacher_score, axis=-1)
    teacher_loss = ops.sum(teacher_w * teacher_match_cost) / ops.maximum(ops.sum(teacher_w), eps)
    if float(student_pull) <= 0.0:
        return teacher_loss
    student_match_cost = ops.min(cost, axis=2)
    student_w = ops.squeeze(student_score, axis=-1)
    student_loss = ops.sum(student_w * student_match_cost) / ops.maximum(ops.sum(student_w), eps)
    return teacher_loss + float(student_pull) * student_loss


def _distilled_losses(base_losses: dict[str, callable], num_classes: int, train_cfg: TrainConfig, args, anchors: np.ndarray, input_size, teacher_distill_scale_var=None):
    teacher_weight = float(args.teacher_distill)
    base_channels = 5 + num_classes + 5 * yad2k_core.MAX_TRUE_BOXES
    teacher_channels = 5 + num_classes
    weight_channels = 2
    eps = 1e-6
    bbox_delta = 0.125
    input_h, input_w = _input_hw(input_size)
    input_wh = np.asarray([input_w, input_h], np.float32)

    def _smooth_l1(x):
        abs_x = ops.absolute(x)
        delta_t = ops.cast(bbox_delta, abs_x.dtype)
        quadratic = 0.5 * ops.square(x)
        linear = delta_t * (abs_x - 0.5 * delta_t)
        return ops.where(abs_x <= delta_t, quadratic, linear)

    def _soft_binary_ce(student_prob, teacher_prob):
        student_prob = ops.clip(student_prob, eps, 1.0 - eps)
        return -(
            teacher_prob * ops.log(student_prob)
            + (1.0 - teacher_prob) * ops.log(1.0 - student_prob)
        )

    def _soft_class_ce(student_prob, teacher_prob):
        if train_cfg.class_mode == "softmax":
            student_log_prob = ops.log(ops.clip(student_prob, eps, 1.0))
            return -ops.sum(teacher_prob * student_log_prob, axis=-1, keepdims=True)
        student_prob = ops.clip(student_prob, eps, 1.0 - eps)
        return -(
            teacher_prob * ops.log(student_prob)
            + (1.0 - teacher_prob) * ops.log(1.0 - student_prob)
        )

    def wrap(head: str, base_loss):
        mask, stride = _teacher_head_spec(head)
        grid_h, grid_w = yad2k_core._grid_hw((input_h, input_w), stride)
        gy, gx = np.meshgrid(np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32), indexing="ij")
        grid_xy = np.stack([gx, gy], axis=-1).reshape(1, grid_h, grid_w, 1, 2)
        grid_s = np.asarray([grid_w, grid_h], np.float32).reshape(1, 1, 1, 1, 2)
        anchor_wh = (anchors[np.asarray(mask, np.int32)] / input_wh).reshape(1, 1, 1, len(mask), 2).astype(np.float32)

        def loss(y_true, y_pred):
            base_true = y_true[..., :base_channels]
            total = base_loss(base_true, y_pred)
            if teacher_weight <= 0.0:
                return total
            if teacher_distill_scale_var is not None:
                active_teacher_weight = teacher_weight * ops.cast(teacher_distill_scale_var, "float32")
            else:
                active_teacher_weight = teacher_weight
            pos_mask = ops.stop_gradient(base_true[..., 4:5])
            teacher = y_true[..., base_channels : base_channels + teacher_channels]
            bbox_cls_weight = y_true[..., base_channels + teacher_channels : base_channels + teacher_channels + 1]
            obj_weight = y_true[..., base_channels + teacher_channels + 1 : base_channels + teacher_channels + weight_channels]
            teacher = ops.stop_gradient(teacher)
            bbox_cls_weight = ops.stop_gradient(bbox_cls_weight) * pos_mask
            obj_weight = ops.stop_gradient(obj_weight)
            if args.box_mode == "yolo":
                xy_p = (ops.sigmoid(y_pred[..., :2]) + ops.cast(grid_xy, "float32")) / ops.cast(grid_s, "float32")
                wh_p = ops.exp(ops.clip(y_pred[..., 2:4], -6.0, 6.0)) * ops.cast(anchor_wh, "float32")
            elif args.box_mode == "linear_float":
                xy_p = (ops.clip(y_pred[..., :2], 0.0, 1.0) + ops.cast(grid_xy, "float32")) / ops.cast(grid_s, "float32")
                wh_p = ops.cast(yad2k_core._piecewise_ratio_from_u_ops(ops.clip(y_pred[..., 2:4], 0.0, 1.0)), "float32") * ops.cast(anchor_wh, "float32")
            else:
                xy_p = (ops.cast(yad2k_core._uint01_ops(y_pred[..., :2], train_cfg.bbox_output_scale), "float32") + ops.cast(grid_xy, "float32")) / ops.cast(grid_s, "float32")
                wh_p = ops.cast(yad2k_core._piecewise_ratio_from_u_ops(yad2k_core._uint01_ops(y_pred[..., 2:4], train_cfg.bbox_output_scale)), "float32") * ops.cast(anchor_wh, "float32")
            bbox_s = ops.concatenate([xy_p, wh_p], axis=-1)
            bbox_t = teacher[..., :4]
            obj_s = ops.sigmoid(ops.clip(y_pred[..., 4:5] / train_cfg.obj_output_scale, -127.0, 127.0) / train_cfg.logit_scale)
            obj_t = teacher[..., 4:5]
            cls_logits = ops.clip(y_pred[..., 5:] / train_cfg.cls_output_scale, -127.0, 127.0) / train_cfg.logit_scale
            if train_cfg.class_mode == "softmax":
                cls_s = ops.softmax(cls_logits, axis=-1)
            else:
                cls_s = ops.sigmoid(cls_logits)
            cls_t = teacher[..., 5:]
            if getattr(args, "teacher_distill_mode", "decoded_dense") == "topk_align":
                student_boxes = ops.reshape(bbox_s, (ops.shape(bbox_s)[0], -1, 4))
                student_obj = ops.reshape(obj_s, (ops.shape(obj_s)[0], -1, 1))
                student_cls = ops.reshape(cls_s, (ops.shape(cls_s)[0], -1, num_classes))
                teacher_boxes = ops.reshape(bbox_t, (ops.shape(bbox_t)[0], -1, 4))
                teacher_obj = ops.reshape(obj_t, (ops.shape(obj_t)[0], -1, 1))
                teacher_cls = ops.reshape(cls_t, (ops.shape(cls_t)[0], -1, num_classes))
                distill = _teacher_topk_alignment_loss_ops(
                    student_boxes,
                    student_obj,
                    student_cls,
                    teacher_boxes,
                    teacher_obj,
                    teacher_cls,
                    class_mode=train_cfg.class_mode,
                    bbox_gain=float(args.teacher_bbox_distill),
                    obj_gain=float(args.teacher_obj_distill),
                    cls_gain=float(args.teacher_cls_distill),
                    teacher_top_k=int(getattr(args, "teacher_align_top_k_teacher", 16)),
                    student_top_k=int(getattr(args, "teacher_align_top_k_student", 64)),
                    student_pull=float(getattr(args, "teacher_align_student_pull", 0.0)),
                    eps=eps,
                )
            else:
                distill = (
                    float(args.teacher_bbox_distill) * ops.mean(bbox_cls_weight * _smooth_l1(bbox_s - bbox_t))
                    + float(args.teacher_obj_distill) * ops.mean(obj_weight * _soft_binary_ce(obj_s, obj_t))
                    + float(args.teacher_cls_distill) * ops.mean(bbox_cls_weight * _soft_class_ce(cls_s, cls_t))
                )
            return total + active_teacher_weight * distill

        return loss

    return {head: wrap(head, loss) for head, loss in base_losses.items()}


def _callbacks(
    run_dir: Path,
    dataset_tag: str,
    train_cfg: TrainConfig,
    map_data,
    val_samples,
    anchors,
    input_size,
    num_classes: int,
    train_data,
    val_data,
    calibration_data=None,
    teacher_distill_scale_var=None,
    teacher_distill_enabled: bool = False,
    teacher_distill_burnin_epochs: int = 0,
    *,
    append: bool = False,
    restart_cycle: int = 1,
    epochs_run_offset: int = 0,
):
    history_path = _artifact_path(run_dir, dataset_tag, "history.csv")
    best_weights_path = _artifact_path(run_dir, dataset_tag, "best.weights.h5")
    best_map_weights_path = _artifact_path(run_dir, dataset_tag, "best_map.weights.h5")
    rows = _history_rows(history_path)
    best_val_loss = min((_history_float(r.get("val_loss"), float("inf")) for r in rows), default=float("inf"))
    best_val_map50 = max((_history_float(r.get("val_map50"), float("-inf")) for r in rows), default=float("-inf"))
    map_callback = MeanAPCallback(
        map_data,
        val_samples,
        anchors,
        input_size,
        num_classes,
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
    callbacks = [
        map_callback,
        EpochTimingCallback(train_data, val_data, map_callback),
        EpochMetadataCallback(restart_cycle, epochs_run_offset),
        keras.callbacks.CSVLogger(history_path, append=append),
        GlobalBestWeightsCallback(
            best_weights_path,
            monitor="val_loss",
            mode="min",
            initial_best=best_val_loss,
            verbose=1,
        ),
        GlobalBestMapWeightsCallback(
            best_map_weights_path,
            initial_best=best_val_map50,
            verbose=1,
        ),
    ]
    if args.bn_recalibrate_before_val:
        callbacks.insert(
            0,
            BinaryBNRecalibrationCallback(
                calibration_data if calibration_data is not None else train_data,
                batch_size=train_cfg.batch_size,
                max_batches=int(args.bn_recalibrate_batches),
                verbose=1,
            ),
        )
    if args.backup_and_restore:
        callbacks.append(keras.callbacks.BackupAndRestore(run_dir / f"model.{dataset_tag}.backup"))
    if teacher_distill_scale_var is not None and teacher_distill_enabled:
        callbacks.append(
            TeacherDistillBurnInCallback(
                teacher_distill_scale_var,
                int(teacher_distill_burnin_epochs),
            )
        )
    return callbacks


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


def _save_bbox_priors_csv(path: Path, anchors: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anchor_index", "width", "height", "area"])
        for idx, (width, height) in enumerate(np.asarray(anchors, np.float32), start=1):
            writer.writerow([idx, float(width), float(height), float(width * height)])


def _load_bbox_priors_csv(path: Path) -> np.ndarray:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    anchors = [[float(row["width"]), float(row["height"])] for row in rows]
    return np.asarray(anchors, np.float32)


def main():
    global args
    args = parse_args()
    epochs_per_restart = int(args.epochs)
    input_size = _resolve_input_size_arg(args)
    keras.config.set_dtype_policy(args.dtype_policy)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    train_samples, val_samples, test_samples, classes = _prepare_split_samples(args, input_size)

    anchors = yad2k_core.fit_anchors(train_samples, input_size, n=6, seed=args.seed)

    train_cfg = _build_train_cfg(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_bbox_priors_csv(run_dir / "bbox_priors.csv", anchors)

    dataset_tag = "voc" if args.dataset_format in ("voc", "auto") else "dataset"

    teacher_spec_payload = _teacher_spec_from_args(input_size, train_cfg, args)
    _save_json(_teacher_spec_path(run_dir), teacher_spec_payload)

    teacher = None
    teacher_spec = None
    teacher_anchors = anchors
    teacher_distill_scale_var = keras.Variable(1.0, dtype="float32", trainable=False, name="teacher_distill_scale")
    teacher_path = Path(args.teacher_weights) if args.teacher_weights else None
    if args.teacher_run_dir:
        teacher_path = _teacher_weights_path(args)
        args.teacher_weights = str(teacher_path)
    if args.teacher_weights:
        teacher_path = Path(args.teacher_weights)
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher weights not found: {teacher_path}")
        teacher, teacher_spec = _build_teacher(len(classes), input_size, args)
        if args.teacher_run_dir:
            teacher_anchor_path = Path(args.teacher_run_dir) / "bbox_priors.csv"
            if teacher_anchor_path.exists():
                teacher_anchors = _load_bbox_priors_csv(teacher_anchor_path)
                print(f"Loaded teacher anchors from {teacher_anchor_path}")
            else:
                print(f"Teacher anchors not found at {teacher_anchor_path}; using student anchors for distillation")
    else:
        args.teacher_distill = 0.0
        args.teacher_bbox_distill = 0.0
        args.teacher_obj_distill = 0.0
        args.teacher_cls_distill = 0.0

    image_dtype = np.float32 if args.float_model else np.float16

    if args.subset_train_as_val:
        val_images, val_targets = yad2k_core.build_targets(
            val_samples,
            len(classes),
            anchors,
            input_size,
            args.thermo_levels,
            augment=False,
            positive_top_k=args.positive_top_k,
            image_dtype=image_dtype,
            advanced_augmentations=False,
            advanced_augment_prob=args.advanced_augment_prob,
            heads=TINIER_HEADS,
            strides=TINIER_STRIDES,
            anchor_masks=TINIER_ANCHOR_MASKS,
        )
        train_images, train_targets = yad2k_core.build_targets(
            train_samples,
            len(classes),
            anchors,
            input_size,
            args.thermo_levels,
            augment=train_cfg.train_augment,
            positive_top_k=args.positive_top_k,
            image_dtype=image_dtype,
            advanced_augmentations=args.advanced_augmentations,
            advanced_augment_prob=0.0,
            heads=TINIER_HEADS,
            strides=TINIER_STRIDES,
            anchor_masks=TINIER_ANCHOR_MASKS,
        )
        train_targets_with_teacher = None
        val_targets_with_teacher = None
        if teacher is not None:
            train_teacher = teacher.predict(train_images, batch_size=args.batch_size, verbose=0)
            val_teacher = teacher.predict(val_images, batch_size=args.batch_size, verbose=0)
            train_teacher = _teacher_preds_to_student_targets(
                train_teacher,
                teacher_anchors,
                anchors,
                input_size,
                train_cfg,
                teacher_spec,
                args.box_mode,
                distill_score_threshold=float(getattr(args, "distill_score_threshold", 0.25)),
            )
            val_teacher = _teacher_preds_to_student_targets(
                val_teacher,
                teacher_anchors,
                anchors,
                input_size,
                train_cfg,
                teacher_spec,
                args.box_mode,
                distill_score_threshold=float(getattr(args, "distill_score_threshold", 0.25)),
            )
            train_targets_with_teacher = _append_teacher_targets(train_targets, train_teacher)
            val_targets_with_teacher = _append_teacher_targets(val_targets, val_teacher)
        train_data = None
        val_data = None
        map_data = val_images
        steps = max(1, math.ceil((int(train_images.shape[0]) * max(1, int(args.repeat))) / int(args.batch_size)))
    else:
        if args.train_cache_refresh_epochs > 0:
            train_data = CachedDetectionDataset(
                train_samples,
                len(classes),
                anchors,
                input_size,
                thermo_levels=args.thermo_levels,
                augment=args.train_augment,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
                positive_top_k=args.positive_top_k,
                rebuild_every=args.train_cache_refresh_epochs,
                refresh_divisor=args.train_cache_refresh_divisor,
                build_workers=args.loader_workers,
                image_dtype=image_dtype,
                augment_start_strength=args.augment_start_strength,
                augment_ramp_epochs=args.augment_ramp_epochs,
                augment_constant_strength=args.augment_constant_strength,
                advanced_augmentations=args.advanced_augmentations,
                advanced_augment_prob=args.advanced_augment_prob,
                advanced_augment_ramp_epochs=args.advanced_augment_ramp_epochs,
                heads=TINIER_HEADS,
                strides=TINIER_STRIDES,
                anchor_masks=TINIER_ANCHOR_MASKS,
            )
        else:
            train_data = DetectionPyDataset(
                train_samples,
                len(classes),
                anchors,
                input_size,
                thermo_levels=args.thermo_levels,
                augment=args.train_augment,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
                positive_top_k=args.positive_top_k,
                workers=args.loader_workers,
                use_multiprocessing=args.loader_multiprocessing,
                max_queue_size=args.loader_queue_size,
                image_dtype=image_dtype,
                augment_start_strength=args.augment_start_strength,
                augment_ramp_epochs=args.augment_ramp_epochs,
                augment_constant_strength=args.augment_constant_strength,
                advanced_augmentations=args.advanced_augmentations,
                advanced_augment_prob=args.advanced_augment_prob,
                advanced_augment_ramp_epochs=args.advanced_augment_ramp_epochs,
                heads=TINIER_HEADS,
                strides=TINIER_STRIDES,
                anchor_masks=TINIER_ANCHOR_MASKS,
            )
        val_data = CachedDetectionDataset(
            val_samples,
            len(classes),
            anchors,
            input_size,
            thermo_levels=args.thermo_levels,
            augment=False,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 1,
            positive_top_k=args.positive_top_k,
            rebuild_every=0,
            refresh_divisor=0,
            build_workers=args.loader_workers,
            image_dtype=image_dtype,
            advanced_augmentations=False,
            advanced_augment_prob=args.advanced_augment_prob,
            heads=TINIER_HEADS,
            strides=TINIER_STRIDES,
            anchor_masks=TINIER_ANCHOR_MASKS,
        )
        map_data = val_data
        steps = len(train_data)

    loss_logit_scale = float(args.loss_logit_scale) if float(args.loss_logit_scale) > 0.0 else float(train_cfg.logit_scale)
    best_weights = _artifact_path(run_dir, dataset_tag, "best.weights.h5")
    restart_reload_weights = _restart_reload_path(run_dir, dataset_tag, args.restart_reload_monitor)
    history_path = _artifact_path(run_dir, dataset_tag, "history.csv")
    model = None
    for restart_idx in range(args.restart_cycles):
        cycle_args = _effective_distill_args(args, restart_idx)
        model = _build_model(len(classes), input_size, args)
        if restart_idx == 0 and args.warm_start_weights:
            warm_path = Path(args.warm_start_weights)
            if warm_path.exists():
                source_kind = args.warm_start_source_model
                if source_kind == "auto":
                    source_kind = "float" if (not args.float_model and "tinier_float" in warm_path.as_posix()) else "binary"
                if source_kind == "float" and not args.float_model:
                    _transfer_float_to_binary_tinier(model, warm_path, len(classes), input_size, args)
                else:
                    model.load_weights(warm_path, skip_mismatch=False)
                    print(f"Warm-started matching weights from {warm_path}")
            else:
                print(f"Warm-start weights not found: {warm_path}")
        elif restart_reload_weights.exists():
            model.load_weights(restart_reload_weights, skip_mismatch=False)
            print(
                f"[restart {restart_idx + 1}/{args.restart_cycles}] "
                f"Reloaded {args.restart_reload_monitor} weights from {restart_reload_weights}"
            )

        losses = yad2k_core.make_losses(
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
            logit_scale=loss_logit_scale,
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
            losses,
            len(classes),
            train_cfg,
            cycle_args,
            anchors,
            input_size,
            teacher_distill_scale_var=teacher_distill_scale_var,
        )
        model.compile(
            optimizer=_optimizer(train_cfg, steps),
            loss=losses,
            jit_compile=train_cfg.jit_compile,
            steps_per_execution=max(1, min(steps, 128)),
            auto_scale_loss=train_cfg.auto_scale_loss,
        )
        epoch_offset = restart_idx * epochs_per_restart
        if args.subset_train_as_val:
            use_teacher_targets = bool(float(getattr(cycle_args, "teacher_distill", 0.0)) > 0.0 and teacher is not None and train_targets_with_teacher is not None)
            cycle_train_targets = train_targets_with_teacher if use_teacher_targets else train_targets
            cycle_val_targets = val_targets_with_teacher if use_teacher_targets else val_targets
            train_data = RepeatedArrayDataset(train_images, cycle_train_targets, args.batch_size, args.repeat)
            val_data = (val_images, cycle_val_targets)
            calibration_data = train_images
        else:
            calibration_data = None
        model.fit(
            train_data,
            validation_data=val_data,
            initial_epoch=epoch_offset,
            epochs=epoch_offset + epochs_per_restart,
            validation_freq=max(1, int(args.validation_freq)),
            verbose=2,
            callbacks=_callbacks(
                run_dir,
                dataset_tag,
                train_cfg,
                map_data,
                val_samples,
                anchors,
                input_size,
                len(classes),
                train_data,
                val_data,
                calibration_data,
                teacher_distill_scale_var,
                teacher_distill_enabled=bool(float(getattr(cycle_args, "teacher_distill", 0.0)) > 0.0),
                teacher_distill_burnin_epochs=int(getattr(cycle_args, "teacher_distill_burnin_epochs", 0)),
                append=(restart_idx > 0),
                restart_cycle=restart_idx + 1,
                epochs_run_offset=0,
            ),
        )

    if restart_reload_weights.exists():
        model.load_weights(restart_reload_weights)
    model.save(_artifact_path(run_dir, dataset_tag, "last.keras"))

    if test_samples:
        test_data = CachedDetectionDataset(
            test_samples,
            len(classes),
            anchors,
            input_size,
            thermo_levels=args.thermo_levels,
            augment=False,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 2,
            positive_top_k=args.positive_top_k,
            rebuild_every=0,
            refresh_divisor=0,
            build_workers=args.loader_workers,
            image_dtype=image_dtype,
            advanced_augmentations=False,
            advanced_augment_prob=args.advanced_augment_prob,
            heads=TINIER_HEADS,
            strides=TINIER_STRIDES,
            anchor_masks=TINIER_ANCHOR_MASKS,
        )
        test_eval = model.evaluate(test_data, verbose=0, return_dict=True)
        test_preds = model.predict(test_data, verbose=0)
        test_metrics = {
            "test_map50": float(
                yad2k_core.map50(
                    test_samples,
                    test_preds,
                    anchors,
                    input_size,
                    len(classes),
                    score=train_cfg.score_threshold,
                    objectness=train_cfg.objectness_threshold,
                    bbox_output_scale=train_cfg.bbox_output_scale,
                    obj_output_scale=train_cfg.obj_output_scale,
                    cls_output_scale=train_cfg.cls_output_scale,
                    logit_scale=train_cfg.logit_scale,
                    class_mode=train_cfg.class_mode,
                    box_mode=args.box_mode,
                    heads=TINIER_HEADS,
                    anchor_masks=TINIER_ANCHOR_MASKS,
                )
            ),
            "test_loss": float(test_eval.get("loss", 0.0)),
        }
    else:
        test_metrics = {}

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
        "test_count": len(test_samples),
        "classes": classes,
        "anchors": anchors.tolist(),
        "input_size": input_size if isinstance(input_size, int) else list(input_size),
        "float_model": bool(args.float_model),
        "tinier_cfg": None if args.float_model else asdict(_build_cfg(args)),
        "train_cfg": asdict(train_cfg),
        "teacher_spec": teacher_spec_payload,
        "loss_logit_scale": loss_logit_scale,
        "restart_cycles": int(args.restart_cycles),
        "restart_reload_monitor": str(args.restart_reload_monitor),
        "teacher_checkpoint_monitor": str(args.teacher_checkpoint_monitor),
        "epochs_per_restart": int(epochs_per_restart),
        "augment_start_strength": float(args.augment_start_strength),
        "augment_ramp_epochs": int(args.augment_ramp_epochs),
        "augment_constant_strength": bool(args.augment_constant_strength),
        "advanced_augmentations": bool(args.advanced_augmentations),
        "advanced_augment_prob": float(args.advanced_augment_prob),
        "advanced_augment_ramp_epochs": int(args.advanced_augment_ramp_epochs),
        "best_val_map50": best_val_map50,
        "final_val_map50": final_val_map50,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
    }
    summary.update(test_metrics)
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
