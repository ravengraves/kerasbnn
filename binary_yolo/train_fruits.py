from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re

os.environ.setdefault("KERAS_BACKEND", "jax")

try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import keras
import jax
import numpy as np

from binary_yolo.dataset import DetectionPyDataset
from binary_yolo.model import ModelConfig, build_binary_yolov3_small
from binary_yolo.yad2k_core import ANCHOR_MASKS, STRIDES, _detect_dataset_format, _xyxy_to_cxcywh, build_targets, draw_boxes, find_sample, fit_anchors, load_dataset_splits, make_losses, map50, predict_boxes, split_samples, split_samples_three_way

keras.config.set_dtype_policy("mixed_float16")


_BASE_HARD_NEGATIVE_MIN = 3584
_BASE_HARD_NEGATIVE_SIZE = 224
_BASE_OUTPUT_BOXES = sum((_BASE_HARD_NEGATIVE_SIZE // stride) ** 2 * len(mask) for stride, mask in zip(STRIDES, ANCHOR_MASKS))
_HARD_NEGATIVE_BOX_SCALE = _BASE_HARD_NEGATIVE_MIN / _BASE_OUTPUT_BOXES


@dataclass(frozen=True)
class TrainConfig:
    name: str = "baseline"
    epochs: int = 256
    batch_size: int = 8
    lr: float = 5e-4
    optimizer: str = "adam"
    schedule: str = "cosine"
    cosine_first_decay_epochs: float = 16.0
    cosine_t_mul: float = 2.0
    cosine_m_mul: float = 1.0
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = False
    decay_rate: float = 0.9
    decay_epochs: float = 12.0
    plateau_factor: float = 0.5
    plateau_patience: int = 6
    plateau_min_lr: float = 1e-4
    early_patience: int = 0
    weight_decay: float = 1e-4
    clipnorm: float = 1.0
    rmsprop_rho: float = 0.9
    auto_scale_loss: bool = True
    jit_compile: bool = True
    obj_gain: float = 4.0
    obj_pos_gain: float = 8.0
    noobj_weight: float = 0.04
    cls_neg_weight: float = 0.0
    cls_gain: float = 8.0
    class_mode: str = "softmax"
    box_gain: float = 5.0
    box_iou_aux_gain: float = 0.0
    box_iou_aux_mode: str = "iou"
    bbox_output_scale: float = 2.0
    obj_output_scale: float = 32.0
    cls_output_scale: float = 32.0
    logit_scale: float = 32.0
    ignore_iou: float = 0.5
    rescore_objectness: bool = False
    hard_negative_ratio: float = 0.0
    hard_negative_min: int = 0
    hard_negative_min_p3: int = 0
    hard_negative_min_p4: int = 0
    hard_negative_min_p5: int = 0
    conf_neg_normalizer: str = "pos"
    positive_top_k: int = 3
    box_mode: str = "linear"
    score_threshold: float = 0.25
    objectness_threshold: float = 0.25
    overfit_sample: str = "fruit0"
    overfit_epochs: int = 8
    overfit_batch_size: int = 128
    overfit_repeat: int = 1024
    overfit_lr: float = 5e-4
    checkpoint_monitor: str = "val_map50"
    checkpoint_mode: str = "max"
    train_augment: bool = True
    augment_start_strength: float = 0.2
    augment_ramp_epochs: int = 100
    loader_workers: int = 8
    loader_queue_size: int = 32
    loader_multiprocessing: bool = True


def _safe_tag(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "dataset"


def _resolve_dataset_tag(dataset_root: str | Path, dataset_format: str) -> str:
    return _detect_dataset_format(Path(dataset_root)) if dataset_format == "auto" else dataset_format


def _artifact_stem(dataset_tag: str) -> str:
    return f"model.{_safe_tag(dataset_tag)}"


def _resolve_run_dir(run_dir: str | Path, dataset_tag: str) -> Path:
    run_dir = Path(run_dir)
    if str(run_dir) == "runs/binary_yolo_detection":
        return Path(f"runs/{_artifact_stem(dataset_tag)}.binary_yolo_detection")
    return run_dir


def _artifact_path(run_dir: Path, dataset_tag: str, suffix: str) -> Path:
    return run_dir / f"{_artifact_stem(dataset_tag)}.{suffix}"


def _input_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, tuple):
        return int(size[0]), int(size[1])
    side = int(size)
    return side, side


def _input_wh(size: int | tuple[int, int]) -> tuple[int, int]:
    h, w = _input_hw(size)
    return w, h


def _resolve_input_size_arg(args) -> int | tuple[int, int]:
    if getattr(args, "input_height", 0) and getattr(args, "input_width", 0):
        return int(args.input_height), int(args.input_width)
    return int(args.input_size)


def parse_args():
    ap = argparse.ArgumentParser(description="Train or search binary YOLOv3-small detection runs.")
    ap.add_argument("--dataset", default="fruits_detection")
    ap.add_argument("--dataset-format", default="auto", choices=["auto", "fruits", "voc", "coco"])
    ap.add_argument("--dataset-split", default="auto", choices=["auto", "official", "random"])
    ap.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--run-dir", default="runs/binary_yolo_detection")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--input-width", type=int, default=0)
    ap.add_argument("--input-height", type=int, default=0)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.0)
    ap.add_argument("--train-sample-fraction", type=float, default=1.0)
    ap.add_argument("--val-sample-fraction", type=float, default=1.0)
    ap.add_argument("--test-sample-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--search-set", default="default", choices=["default", "long_lr", "long_exp", "long_plateau", "long_model", "dropcut_modes", "objmine_quick", "objmine_refine", "voc_subset_diag", "voc_subset_mining"])
    ap.add_argument("--config-name", default="")
    ap.add_argument("--epochs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "rmsprop", "sgd"])
    ap.add_argument("--schedule", default="cosine", choices=["constant", "cosine", "cosine_restart", "exp", "plateau"])
    ap.add_argument("--cosine-first-decay-epochs", type=float, default=16.0)
    ap.add_argument("--cosine-t-mul", type=float, default=2.0)
    ap.add_argument("--cosine-m-mul", type=float, default=1.0)
    ap.add_argument("--quant-levels", type=int, default=8)
    ap.add_argument("--output-levels", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--dropout-stage", default="none")
    ap.add_argument("--input-dropcut", type=float, default=0.1)
    ap.add_argument("--input-dropcut-boxes", type=int, default=1)
    ap.add_argument("--input-dropcut-size-min", type=float, default=0.05)
    ap.add_argument("--input-dropcut-size-max", type=float, default=0.20)
    ap.add_argument("--input-dropcut-method", default="both", choices=["black", "white", "both", "noise"])
    ap.add_argument("--input-dropcut2", type=float, default=0.0)
    ap.add_argument("--input-dropcut2-method", default="noise", choices=["black", "white", "both", "noise"])
    ap.add_argument("--feature-dropcut", type=float, default=0.0)
    ap.add_argument("--feature-dropcut-stage", default="none")
    ap.add_argument("--clipnorm", type=float, default=1.0)
    ap.add_argument("--rmsprop-rho", type=float, default=0.9)
    ap.add_argument("--sgd-momentum", type=float, default=0.9)
    ap.add_argument("--sgd-nesterov", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--obj-gain", type=float, default=4.0)
    ap.add_argument("--obj-pos-gain", type=float, default=8.0)
    ap.add_argument("--noobj-weight", type=float, default=0.04)
    ap.add_argument("--cls-neg-weight", type=float, default=0.0)
    ap.add_argument("--cls-gain", type=float, default=8.0)
    ap.add_argument("--box-gain", type=float, default=5.0)
    ap.add_argument("--class-mode", choices=["bce", "softmax"], default="softmax")
    ap.add_argument("--bbox-output-scale", type=float, default=2.0)
    ap.add_argument("--obj-output-scale", type=float, default=64.0)
    ap.add_argument("--cls-output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--ignore-iou", type=float, default=0.5)
    ap.add_argument("--rescore-objectness", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--hard-negative-ratio", type=float, default=0.0)
    ap.add_argument("--hard-negative-min", type=int, default=0, help="0 auto-scales from the detector output box count")
    ap.add_argument("--hard-negative-min-p3", type=int, default=0)
    ap.add_argument("--hard-negative-min-p4", type=int, default=0)
    ap.add_argument("--hard-negative-min-p5", type=int, default=0)
    ap.add_argument("--conf-neg-normalizer", choices=["neg", "pos", "none"], default="pos")
    ap.add_argument("--positive-top-k", type=int, default=3)
    ap.add_argument("--box-mode", choices=["linear", "yolo"], default="linear")
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--objectness-threshold", type=float, default=0.25)
    ap.add_argument("--head-proj-raw", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--head-proj-levels", type=int, default=8)
    ap.add_argument("--binary-batchnorm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-momentum", type=float, default=0.95)
    ap.add_argument("--bn-quantize-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-centered", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-normalized", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-eval-stats", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overfit-epochs", type=int, default=8)
    ap.add_argument("--overfit-batch-size", type=int, default=128)
    ap.add_argument("--overfit-repeat", type=int, default=1024)
    ap.add_argument("--overfit-lr", type=float, default=5e-4)
    ap.add_argument("--train-augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--augment-start-strength", type=float, default=0.2)
    ap.add_argument("--augment-ramp-epochs", type=int, default=100)
    ap.add_argument("--loader-workers", type=int, default=8)
    ap.add_argument("--loader-queue-size", type=int, default=32)
    ap.add_argument("--loader-multiprocessing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warm-start-weights", default="")
    return ap.parse_args()


def _save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _output_box_count(input_size: int | tuple[int, int]) -> int:
    h, w = _input_hw(input_size)
    return sum(((h + stride - 1) // stride) * ((w + stride - 1) // stride) * len(mask) for stride, mask in zip(STRIDES, ANCHOR_MASKS))


def _resolve_train_cfg(train_cfg: TrainConfig, input_size: int | tuple[int, int]) -> TrainConfig:
    hard_negative_min = train_cfg.hard_negative_min
    if hard_negative_min <= 0:
        hard_negative_min = max(1, int(round(_HARD_NEGATIVE_BOX_SCALE * _output_box_count(input_size))))
    scale_mins = []
    for explicit in (
        train_cfg.hard_negative_min_p3,
        train_cfg.hard_negative_min_p4,
        train_cfg.hard_negative_min_p5,
    ):
        scale_mins.append(hard_negative_min if explicit <= 0 else explicit)
    return replace(
        train_cfg,
        hard_negative_min=hard_negative_min,
        hard_negative_min_p3=scale_mins[0],
        hard_negative_min_p4=scale_mins[1],
        hard_negative_min_p5=scale_mins[2],
    )


def _subset_samples(samples, fraction: float, seed: int):
    fraction = float(fraction)
    if fraction <= 0.0 or fraction >= 1.0:
        return list(samples)
    count = max(1, int(round(len(samples) * fraction)))
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(samples), size=count, replace=False))
    return [samples[int(i)] for i in ids]


def _prepare_split_samples(args):
    input_size = _resolve_input_size_arg(args)
    train_samples, val_samples, classes = load_dataset_splits(
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
        train_samples, val_samples, test_samples = split_samples_three_way(
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
    return train_samples, val_samples, test_samples, classes


def _repeat_data(data, repeats: int):
    if repeats <= 1:
        return data
    images, targets = data
    images = np.repeat(images, repeats, axis=0)
    targets = {name: np.repeat(values, repeats, axis=0) for name, values in targets.items()}
    return images, targets


def _to_device(data):
    images, targets = data
    try:
        images = jax.device_put(images)
        targets = {name: jax.device_put(values) for name, values in targets.items()}
    except Exception:
        pass
    return images, targets


def _optimizer(cfg: TrainConfig, steps: int):
    if cfg.schedule == "cosine":
        lr = keras.optimizers.schedules.CosineDecay(cfg.lr, cfg.epochs * steps)
    elif cfg.schedule == "cosine_restart":
        lr = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=cfg.lr,
            first_decay_steps=max(1, int(cfg.cosine_first_decay_epochs * steps)),
            t_mul=cfg.cosine_t_mul,
            m_mul=cfg.cosine_m_mul,
        )
    elif cfg.schedule == "exp":
        lr = keras.optimizers.schedules.ExponentialDecay(cfg.lr, max(1, int(cfg.decay_epochs * steps)), cfg.decay_rate, staircase=True)
    else:
        lr = cfg.lr
    kw = {"learning_rate": lr}
    if cfg.clipnorm:
        kw["clipnorm"] = cfg.clipnorm
    if cfg.optimizer == "adam":
        return keras.optimizers.Adam(**kw)
    if cfg.optimizer == "adamw":
        return keras.optimizers.AdamW(weight_decay=cfg.weight_decay, **kw)
    if cfg.optimizer == "sgd":
        return keras.optimizers.SGD(momentum=cfg.sgd_momentum, nesterov=cfg.sgd_nesterov, **kw)
    return keras.optimizers.RMSprop(rho=cfg.rmsprop_rho, momentum=0.0, **kw)


def _configure_distribution():
    try:
        devices = keras.distribution.list_devices()
    except Exception:
        return []
    try:
        jax_devices = list(jax.devices())
    except Exception:
        jax_devices = []
    gpu_devices = [dev for dev in devices if dev.startswith("gpu:")]
    if gpu_devices and jax_devices:
        gpu_kinds = [dev.device_kind for dev in jax_devices if dev.platform == "gpu"]
        if gpu_kinds and len(set(gpu_kinds)) > 1:
            return ["gpu:0"]
    if len(devices) > 1:
        keras.distribution.set_distribution(keras.distribution.DataParallel(devices=devices))
    return devices


def _compile(model, anchors, num_classes: int, model_cfg: ModelConfig, train_cfg: TrainConfig, steps: int):
    model.compile(
        optimizer=_optimizer(train_cfg, steps),
        loss=make_losses(
            anchors,
            model_cfg.input_size,
            num_classes,
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
            hard_negative_mins=(
                train_cfg.hard_negative_min_p3,
                train_cfg.hard_negative_min_p4,
                train_cfg.hard_negative_min_p5,
            ),
            conf_neg_normalizer=train_cfg.conf_neg_normalizer,
            box_mode=train_cfg.box_mode,
        ),
        jit_compile=train_cfg.jit_compile,
        steps_per_execution=max(1, min(steps, 32)),
        auto_scale_loss=train_cfg.auto_scale_loss,
    )


class MeanAPCallback(keras.callbacks.Callback):
    def __init__(self, data, samples, anchors, input_size: int, num_classes: int, batch_size: int, bbox_output_scale: float, obj_output_scale: float, cls_output_scale: float, logit_scale: float, score_threshold: float, objectness_threshold: float, class_mode: str, box_mode: str):
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

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        preds = self.model.predict(self.data, batch_size=self.batch_size, verbose=0)
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


def _callbacks(run_dir: Path, dataset_tag: str, train_cfg: TrainConfig, val_images, val_samples, anchors, model_cfg: ModelConfig, num_classes: int):
    callbacks = [
        MeanAPCallback(
            val_images,
            val_samples,
            anchors,
            model_cfg.input_size,
            num_classes,
            train_cfg.batch_size,
            train_cfg.bbox_output_scale,
            train_cfg.obj_output_scale,
            train_cfg.cls_output_scale,
            train_cfg.logit_scale,
            train_cfg.score_threshold,
            train_cfg.objectness_threshold,
            train_cfg.class_mode,
            train_cfg.box_mode,
        ),
        keras.callbacks.CSVLogger(_artifact_path(run_dir, dataset_tag, "history.csv")),
        keras.callbacks.ModelCheckpoint(
            _artifact_path(run_dir, dataset_tag, "best.weights.h5"),
            monitor=train_cfg.checkpoint_monitor,
            mode=train_cfg.checkpoint_mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.BackupAndRestore(run_dir / f"{_artifact_stem(dataset_tag)}.backup"),
    ]
    if train_cfg.schedule == "plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=train_cfg.plateau_factor,
                patience=train_cfg.plateau_patience,
                min_lr=train_cfg.plateau_min_lr,
                verbose=1,
            )
        )
    if train_cfg.early_patience:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=train_cfg.checkpoint_monitor,
                mode=train_cfg.checkpoint_mode,
                patience=train_cfg.early_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks


def _fit(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    anchors,
    classes,
    train_data,
    val_data,
    val_samples,
    run_dir: Path,
    dataset_tag: str,
    warm_start_weights: str | Path | None = None,
):
    model = build_binary_yolov3_small(len(classes), model_cfg)
    if warm_start_weights:
        warm_start_path = Path(warm_start_weights)
        if warm_start_path.exists():
            model.load_weights(warm_start_path, skip_mismatch=True)
            print(f"Warm-started matching weights from {warm_start_path} (skipped mismatched tensors)")
        else:
            print(f"Warm-start weights not found: {warm_start_path}")
    _compile(model, anchors, len(classes), model_cfg, train_cfg, max(1, len(train_data)))
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=train_cfg.epochs,
        verbose=2,
        callbacks=_callbacks(run_dir, dataset_tag, train_cfg, val_data, val_samples, anchors, model_cfg, len(classes)),
    )
    best_weights = _artifact_path(run_dir, dataset_tag, "best.weights.h5")
    if best_weights.exists():
        model.load_weights(best_weights)
    model.save(_artifact_path(run_dir, dataset_tag, "last.keras"))
    return model, history.history


def _summarize(model, image, anchors, classes, run_dir: Path, name: str, score: float, bbox_output_scale: float, obj_output_scale: float, cls_output_scale: float, logit_scale: float, objectness_threshold: float, class_mode: str, box_mode: str, gt_boxes=None, gt_classes=None):
    preds = predict_boxes(
        model.predict(image[None, ...], verbose=0),
        anchors,
        image.shape[:2],
        score=score,
        objectness=objectness_threshold,
        bbox_output_scale=bbox_output_scale,
        obj_output_scale=obj_output_scale,
        cls_output_scale=cls_output_scale,
        logit_scale=logit_scale,
        class_mode=class_mode,
        box_mode=box_mode,
    )
    draw_boxes(image, run_dir / f"{name}_pred.png", preds[0], preds[2], classes, scores=preds[1], color="red")
    if gt_boxes is not None:
        draw_boxes(image, run_dir / f"{name}_gt.png", gt_boxes, gt_classes, classes, color="lime")
    return preds


def run_once(
    samples,
    classes,
    anchors,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    run_dir: Path,
    dataset_tag: str,
    dataset_root: str | Path,
    dataset_split: str,
    val_fraction: float,
    seed: int,
    skip_overfit: bool,
    skip_full: bool,
    warm_start_weights: str | Path | None = None,
    train_samples=None,
    val_samples=None,
    test_samples=None,
):
    train_cfg = _resolve_train_cfg(train_cfg, model_cfg.input_size)
    if train_samples is None or val_samples is None:
        train_samples, val_samples = split_samples(samples, val_fraction=val_fraction, seed=seed)
    train_data = DetectionPyDataset(
        train_samples,
        len(classes),
        anchors,
        model_cfg.input_size,
        model_cfg.thermo_levels,
        augment=train_cfg.train_augment,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        seed=seed,
        positive_top_k=train_cfg.positive_top_k,
        workers=train_cfg.loader_workers,
        use_multiprocessing=train_cfg.loader_multiprocessing,
        max_queue_size=train_cfg.loader_queue_size,
        augment_start_strength=train_cfg.augment_start_strength,
        augment_ramp_epochs=train_cfg.augment_ramp_epochs,
    )
    val_data = DetectionPyDataset(
        val_samples,
        len(classes),
        anchors,
        model_cfg.input_size,
        model_cfg.thermo_levels,
        augment=False,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        seed=seed,
        positive_top_k=train_cfg.positive_top_k,
        workers=train_cfg.loader_workers,
        use_multiprocessing=train_cfg.loader_multiprocessing,
        max_queue_size=train_cfg.loader_queue_size,
    )
    test_data = None
    if test_samples:
        test_data = DetectionPyDataset(
            test_samples,
            len(classes),
            anchors,
            model_cfg.input_size,
            model_cfg.thermo_levels,
            augment=False,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            seed=seed,
            positive_top_k=train_cfg.positive_top_k,
            workers=train_cfg.loader_workers,
            use_multiprocessing=train_cfg.loader_multiprocessing,
            max_queue_size=train_cfg.loader_queue_size,
            augment_start_strength=train_cfg.augment_start_strength,
            augment_ramp_epochs=train_cfg.augment_ramp_epochs,
        )
    out = {
        "dataset": {
            "path": str(dataset_root),
            "format": dataset_tag,
            "split": dataset_split,
            "train_samples": int(len(train_samples)),
            "val_samples": int(len(val_samples)),
            "test_samples": int(len(test_samples or [])),
        },
        "classes": classes,
        "anchors": anchors.tolist(),
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "warm_start_weights": str(warm_start_weights or ""),
    }
    if not skip_overfit:
        try:
            sample = find_sample(samples, train_cfg.overfit_sample)
        except StopIteration:
            sample = train_samples[0]
        overfit_val = build_targets(
            [sample],
            len(classes),
            anchors,
            model_cfg.input_size,
            model_cfg.thermo_levels,
            augment=False,
            positive_top_k=train_cfg.positive_top_k,
        )
        overfit_data = _repeat_data(overfit_val, train_cfg.overfit_repeat)
        overfit_val = _to_device(overfit_val)
        overfit_data = _to_device(overfit_data)
        gt = _xyxy_to_cxcywh(sample.boxes_xyxy, sample.size)
        gt_xyxy = np.column_stack((gt[:, 0] - gt[:, 2] * 0.5, gt[:, 1] - gt[:, 3] * 0.5, gt[:, 0] + gt[:, 2] * 0.5, gt[:, 1] + gt[:, 3] * 0.5))
        model = build_binary_yolov3_small(len(classes), model_cfg)
        _compile(
            model,
            anchors,
            len(classes),
            model_cfg,
            replace(train_cfg, epochs=train_cfg.overfit_epochs, batch_size=train_cfg.overfit_batch_size, lr=train_cfg.overfit_lr),
            max(1, math.ceil(len(overfit_data[0]) / train_cfg.overfit_batch_size)),
        )
        hist = model.fit(
            *overfit_data,
            epochs=train_cfg.overfit_epochs,
            batch_size=train_cfg.overfit_batch_size,
            verbose=2,
            validation_data=overfit_val,
            callbacks=[
                MeanAPCallback(
                    overfit_val[0],
                    [sample],
                    anchors,
                    model_cfg.input_size,
                    len(classes),
                    1,
                    train_cfg.bbox_output_scale,
                    train_cfg.obj_output_scale,
                    train_cfg.cls_output_scale,
                    train_cfg.logit_scale,
                    train_cfg.score_threshold,
                    train_cfg.objectness_threshold,
                    train_cfg.class_mode,
                    train_cfg.box_mode,
                ),
                keras.callbacks.CSVLogger(run_dir / "overfit_history.csv"),
            ],
        ).history
        preds = _summarize(
            model,
            overfit_val[0][0],
            anchors,
            classes,
            run_dir,
            "overfit",
            train_cfg.score_threshold,
            train_cfg.bbox_output_scale,
            train_cfg.obj_output_scale,
            train_cfg.cls_output_scale,
            train_cfg.logit_scale,
            train_cfg.objectness_threshold,
            train_cfg.class_mode,
            train_cfg.box_mode,
            gt_xyxy,
            sample.class_ids,
        )
        out["overfit"] = {
            "final_loss": float(hist["loss"][-1]),
            "final_val_loss": float(hist["val_loss"][-1]),
            "best_val_map50": float(max(hist.get("val_map50", [0.0]))),
            "final_val_map50": float(hist.get("val_map50", [0.0])[-1] if hist.get("val_map50") else 0.0),
            "num_predictions": int(len(preds[0])),
        }
        model.save(_artifact_path(run_dir, dataset_tag, "overfit.keras"))
    if not skip_full:
        model, hist = _fit(
            model_cfg,
            train_cfg,
            anchors,
            classes,
            train_data,
            val_data,
            val_samples,
            run_dir,
            dataset_tag,
            warm_start_weights=warm_start_weights,
        )
        image0, boxes0, class_ids0, sample = val_data.sample_at(0)
        gt = _xyxy_to_cxcywh(boxes0, _input_wh(model_cfg.input_size))
        gt_xyxy = np.column_stack((gt[:, 0] - gt[:, 2] * 0.5, gt[:, 1] - gt[:, 3] * 0.5, gt[:, 0] + gt[:, 2] * 0.5, gt[:, 1] + gt[:, 3] * 0.5))
        preds = _summarize(
            model,
            image0,
            anchors,
            classes,
            run_dir,
            "val0",
            train_cfg.score_threshold,
            train_cfg.bbox_output_scale,
            train_cfg.obj_output_scale,
            train_cfg.cls_output_scale,
            train_cfg.logit_scale,
            train_cfg.objectness_threshold,
            train_cfg.class_mode,
            train_cfg.box_mode,
            gt_xyxy,
            class_ids0,
        )
        out["full"] = {
            "best_val_loss": float(min(hist["val_loss"])),
            "final_val_loss": float(hist["val_loss"][-1]),
            "final_loss": float(hist["loss"][-1]),
            "best_val_map50": float(max(hist.get("val_map50", [0.0]))),
            "final_val_map50": float(hist.get("val_map50", [0.0])[-1] if hist.get("val_map50") else 0.0),
            "num_predictions_on_val0": int(len(preds[0])),
        }
        if test_data is not None and test_samples:
            test_eval = model.evaluate(test_data, verbose=0, return_dict=True)
            test_preds = model.predict(test_data, verbose=0)
            test_map = map50(
                test_samples,
                test_preds,
                anchors,
                model_cfg.input_size,
                len(classes),
                score=train_cfg.score_threshold,
                objectness=train_cfg.objectness_threshold,
                bbox_output_scale=train_cfg.bbox_output_scale,
                obj_output_scale=train_cfg.obj_output_scale,
                cls_output_scale=train_cfg.cls_output_scale,
                logit_scale=train_cfg.logit_scale,
                class_mode=train_cfg.class_mode,
                box_mode=train_cfg.box_mode,
            )
            image0, boxes0, class_ids0, _ = test_data.sample_at(0)
            gt = _xyxy_to_cxcywh(boxes0, _input_wh(model_cfg.input_size))
            gt_xyxy = np.column_stack((gt[:, 0] - gt[:, 2] * 0.5, gt[:, 1] - gt[:, 3] * 0.5, gt[:, 0] + gt[:, 2] * 0.5, gt[:, 1] + gt[:, 3] * 0.5))
            test0_preds = _summarize(
                model,
                image0,
                anchors,
                classes,
                run_dir,
                "test0",
                train_cfg.score_threshold,
                train_cfg.bbox_output_scale,
                train_cfg.obj_output_scale,
                train_cfg.cls_output_scale,
                train_cfg.logit_scale,
                train_cfg.objectness_threshold,
                train_cfg.class_mode,
                train_cfg.box_mode,
                gt_xyxy,
                class_ids0,
            )
            out["test"] = {
                "loss": float(test_eval["loss"]),
                "map50": float(test_map),
                "num_predictions_on_test0": int(len(test0_preds[0])),
            }
    _save_json(run_dir / "summary.json", out)
    return out


def search_configs(input_size: int, search_set: str = "default"):
    base = ModelConfig(
        input_size=input_size,
        thermo_levels=4,
        width_scale=1.25,
        quant_levels=8,
        output_levels=128,
        binary_batchnorm=True,
        bn_momentum=0.95,
        bn_quantize_input=True,
        bn_quantize_centered=False,
        bn_quantize_normalized=False,
        bn_quantize_output=True,
        bn_quantize_eval_stats=True,
        head_proj_raw=True,
        head_proj_levels=8,
        bbox_output_scale=2.0,
        obj_output_scale=32.0,
        cls_output_scale=32.0,
    )
    if search_set == "long_lr":
        return [
            (
                "wide_rmsprop_lr4e4_e96",
                replace(base, width_scale=1.25),
                TrainConfig(name="wide_rmsprop_lr4e4_e96", epochs=96, batch_size=16, optimizer="rmsprop", lr=4e-4, schedule="constant", clipnorm=0.5, noobj_weight=0.15),
            ),
            (
                "wide_rmsprop_lr1e3_e96",
                replace(base, width_scale=1.25),
                TrainConfig(name="wide_rmsprop_lr1e3_e96", epochs=96, batch_size=16, optimizer="rmsprop", lr=1e-3, schedule="constant", clipnorm=0.5, noobj_weight=0.15),
            ),
        ]
    if search_set == "long_exp":
        return [
            (
                "wide_rmsprop_exp_soft_e96",
                replace(base, width_scale=1.25),
                TrainConfig(name="wide_rmsprop_exp_soft_e96", epochs=96, batch_size=16, optimizer="rmsprop", lr=8e-4, schedule="exp", decay_rate=0.78, decay_epochs=24, clipnorm=0.5, noobj_weight=0.15),
            ),
            (
                "wide_rmsprop_exp_mid_e96",
                replace(base, width_scale=1.25),
                TrainConfig(name="wide_rmsprop_exp_mid_e96", epochs=96, batch_size=16, optimizer="rmsprop", lr=1e-3, schedule="exp", decay_rate=0.72, decay_epochs=24, clipnorm=0.5, noobj_weight=0.15),
            ),
        ]
    if search_set == "long_plateau":
        return [
            (
                "wide_rmsprop_dropcut10_thermo224_e96",
                replace(base, input_size=224, thermo_levels=4, width_scale=1.25, input_dropcut=0.10, input_dropcut_boxes=3, input_dropcut_size=(0.05, 0.2)),
                TrainConfig(
                    name="wide_rmsprop_dropcut10_thermo224_e96",
                    epochs=96,
                    batch_size=16,
                    optimizer="rmsprop",
                    lr=1e-3,
                    schedule="plateau",
                    plateau_factor=0.4,
                    plateau_patience=4,
                    plateau_min_lr=8e-5,
                    early_patience=14,
                    clipnorm=0.5,
                    noobj_weight=0.15,
                ),
            ),
            (
                "wide_rmsprop_dropcut25_thermo224_e96",
                replace(base, input_size=224, thermo_levels=4, width_scale=1.25, input_dropcut=0.25, input_dropcut_boxes=3, input_dropcut_size=(0.05, 0.2)),
                TrainConfig(
                    name="wide_rmsprop_dropcut25_thermo224_e96",
                    epochs=96,
                    batch_size=16,
                    optimizer="rmsprop",
                    lr=1e-3,
                    schedule="plateau",
                    plateau_factor=0.4,
                    plateau_patience=4,
                    plateau_min_lr=8e-5,
                    early_patience=14,
                    clipnorm=0.5,
                    noobj_weight=0.15,
                ),
            ),
        ]
    if search_set == "long_model":
        return [
            (
                "dropcut25_thermo224_long_map",
                replace(base, input_size=224, thermo_levels=4, width_scale=1.25, input_dropcut=0.25, input_dropcut_boxes=3, input_dropcut_size=(0.05, 0.2)),
                TrainConfig(
                    name="dropcut25_thermo224_long_map",
                    epochs=128,
                    batch_size=16,
                    optimizer="rmsprop",
                    lr=1e-3,
                    schedule="plateau",
                    plateau_factor=0.4,
                    plateau_patience=4,
                    plateau_min_lr=8e-5,
                    early_patience=24,
                    clipnorm=0.5,
                    noobj_weight=0.15,
                ),
            ),
            (
                "dropcut25_stem32_thermo224_long_map",
                replace(base, input_size=224, thermo_levels=4, width_scale=1.25, stem_filters=32, input_dropcut=0.25, input_dropcut_boxes=3, input_dropcut_size=(0.05, 0.2)),
                TrainConfig(
                    name="dropcut25_stem32_thermo224_long_map",
                    epochs=128,
                    batch_size=16,
                    optimizer="rmsprop",
                    lr=1e-3,
                    schedule="plateau",
                    plateau_factor=0.4,
                    plateau_patience=4,
                    plateau_min_lr=8e-5,
                    early_patience=24,
                    clipnorm=0.5,
                    noobj_weight=0.15,
                ),
            ),
        ]
    if search_set == "dropcut_modes":
        train = TrainConfig(
            epochs=32,
            batch_size=16,
            optimizer="adam",
            lr=5e-4,
            schedule="constant",
            clipnorm=0.5,
            obj_gain=3.5,
            obj_pos_gain=7.0,
            noobj_weight=0.04,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            logit_scale=32.0,
            ignore_iou=0.5,
            rescore_objectness=False,
            hard_negative_min=0,
            conf_neg_normalizer="pos",
            objectness_threshold=0.05,
            score_threshold=0.2,
            checkpoint_monitor="val_map50",
            checkpoint_mode="max",
        )
        model = replace(base, input_size=224, thermo_levels=4, width_scale=1.25, quant_levels=8, head_proj_raw=True, head_proj_levels=8, input_dropcut_boxes=3, input_dropcut_size=(0.05, 0.2))
        return [
            (
                "dropcut_both25_thermo4_raw4",
                replace(model, input_dropcut=0.25, input_dropcut_method="both"),
                replace(train, name="dropcut_both25_thermo4_raw4"),
            ),
            (
                "dropcut_noise25_thermo4_raw4",
                replace(model, input_dropcut=0.25, input_dropcut_method="noise"),
                replace(train, name="dropcut_noise25_thermo4_raw4"),
            ),
            (
                "dropcut_both12_noise12_thermo4_raw4",
                replace(model, input_dropcut=0.125, input_dropcut_method="both", input_dropcut2=0.125, input_dropcut2_method="noise"),
                replace(train, name="dropcut_both12_noise12_thermo4_raw4"),
            ),
        ]
    if search_set == "objmine_quick":
        train = TrainConfig(
            epochs=12,
            batch_size=16,
            optimizer="adam",
            lr=5e-4,
            schedule="constant",
            clipnorm=0.5,
            obj_gain=3.5,
            obj_pos_gain=7.0,
            noobj_weight=0.04,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            logit_scale=32.0,
            ignore_iou=0.5,
            rescore_objectness=False,
            hard_negative_min=0,
            conf_neg_normalizer="pos",
            objectness_threshold=0.05,
            score_threshold=0.2,
            checkpoint_monitor="val_map50",
            checkpoint_mode="max",
        )
        model = replace(
            base,
            input_size=224,
            thermo_levels=4,
            width_scale=1.25,
            quant_levels=8,
            output_levels=128,
            head_proj_raw=True,
            head_proj_levels=8,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            input_dropcut=0.25,
            input_dropcut_method="both",
            input_dropcut_boxes=3,
            input_dropcut_size=(0.05, 0.2),
        )
        return [
            (
                "objmine_base",
                model,
                replace(train, name="objmine_base"),
            ),
            (
                "objmine_hn1024_pos5x2",
                model,
                replace(train, name="objmine_hn1024_pos5x2", hard_negative_min=1024, obj_gain=2.5, obj_pos_gain=5.0),
            ),
            (
                "objmine_hn3072_pos7x3p5_noobj004",
                model,
                replace(train, name="objmine_hn3072_pos7x3p5_noobj004", hard_negative_min=3072, obj_gain=3.5, obj_pos_gain=7.0, noobj_weight=0.04),
            ),
            (
                "objmine_hn1536_pos6x4_noobj0035",
                model,
                replace(train, name="objmine_hn1536_pos6x4_noobj0035", hard_negative_min=1536, obj_gain=4.0, obj_pos_gain=6.0, noobj_weight=0.035),
            ),
        ]
    if search_set == "objmine_refine":
        train = TrainConfig(
            epochs=16,
            batch_size=16,
            optimizer="adam",
            lr=5e-4,
            schedule="constant",
            clipnorm=0.5,
            obj_gain=3.5,
            obj_pos_gain=7.0,
            noobj_weight=0.04,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            logit_scale=32.0,
            ignore_iou=0.5,
            rescore_objectness=False,
            hard_negative_min=0,
            conf_neg_normalizer="pos",
            objectness_threshold=0.05,
            score_threshold=0.2,
            checkpoint_monitor="val_map50",
            checkpoint_mode="max",
        )
        model = replace(
            base,
            input_size=224,
            thermo_levels=4,
            width_scale=1.25,
            quant_levels=8,
            output_levels=128,
            head_proj_raw=True,
            head_proj_levels=8,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            input_dropcut=0.25,
            input_dropcut_method="both",
            input_dropcut_boxes=3,
            input_dropcut_size=(0.05, 0.2),
        )
        return [
            ("objmine_refine_hn2560", model, replace(train, name="objmine_refine_hn2560", hard_negative_min=2560)),
            ("objmine_refine_hn3072", model, replace(train, name="objmine_refine_hn3072", hard_negative_min=3072)),
            ("objmine_refine_hn3584", model, replace(train, name="objmine_refine_hn3584", hard_negative_min=3584)),
            ("objmine_refine_hn4096", model, replace(train, name="objmine_refine_hn4096", hard_negative_min=4096)),
            ("objmine_refine_hn5120", model, replace(train, name="objmine_refine_hn5120", hard_negative_min=5120)),
        ]
    if search_set == "voc_subset_diag":
        train = TrainConfig(
            epochs=64,
            batch_size=8,
            optimizer="adam",
            lr=5e-4,
            schedule="cosine",
            clipnorm=0.5,
            obj_gain=3.5,
            obj_pos_gain=7.0,
            noobj_weight=0.04,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            logit_scale=32.0,
            ignore_iou=0.5,
            rescore_objectness=False,
            hard_negative_min=0,
            conf_neg_normalizer="pos",
            objectness_threshold=0.05,
            score_threshold=0.2,
            checkpoint_monitor="val_map50",
            checkpoint_mode="max",
            train_augment=True,
            loader_workers=4,
            loader_queue_size=32,
            loader_multiprocessing=False,
        )
        model = replace(
            base,
            input_size=input_size,
            thermo_levels=4,
            width_scale=1.25,
            quant_levels=8,
            output_levels=128,
            head_proj_raw=True,
            head_proj_levels=8,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            input_dropcut=0.25,
            input_dropcut_method="both",
        )
        return [
            ("voc_diag_base", model, replace(train, name="voc_diag_base")),
            ("voc_diag_nodropcut", replace(model, input_dropcut=0.0), replace(train, name="voc_diag_nodropcut")),
            (
                "voc_diag_wide_nodropcut",
                replace(model, input_dropcut=0.0, width_scale=1.5, head_pre_filters=256, head_channels=48, obj_cls_hidden=96),
                replace(train, name="voc_diag_wide_nodropcut"),
            ),
            (
                "voc_diag_wide_binaryhead_nodropcut",
                replace(model, input_dropcut=0.0, width_scale=1.5, head_pre_filters=256, head_channels=48, obj_cls_hidden=96, head_proj_raw=False, head_proj_levels=0),
                replace(train, name="voc_diag_wide_binaryhead_nodropcut"),
            ),
            (
                "voc_diag_wide_nodropcut_noaug",
                replace(model, input_dropcut=0.0, width_scale=1.5, head_pre_filters=256, head_channels=48, obj_cls_hidden=96),
                replace(train, name="voc_diag_wide_nodropcut_noaug", train_augment=False),
            ),
        ]
    if search_set == "voc_subset_mining":
        train = TrainConfig(
            epochs=24,
            batch_size=8,
            optimizer="adam",
            lr=5e-4,
            schedule="cosine",
            clipnorm=0.5,
            obj_gain=3.5,
            obj_pos_gain=7.0,
            noobj_weight=0.04,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            logit_scale=32.0,
            ignore_iou=0.5,
            rescore_objectness=False,
            hard_negative_ratio=0.0,
            hard_negative_min=0,
            conf_neg_normalizer="pos",
            objectness_threshold=0.05,
            score_threshold=0.2,
            checkpoint_monitor="val_map50",
            checkpoint_mode="max",
            train_augment=True,
            loader_workers=4,
            loader_queue_size=32,
            loader_multiprocessing=False,
        )
        model = replace(
            base,
            input_size=input_size,
            thermo_levels=4,
            width_scale=1.25,
            quant_levels=8,
            output_levels=128,
            head_proj_raw=True,
            head_proj_levels=8,
            bbox_output_scale=2.0,
            obj_output_scale=32.0,
            cls_output_scale=32.0,
            input_dropcut=0.0,
        )
        return [
            ("vocmine_auto_pos", model, replace(train, name="vocmine_auto_pos")),
            ("vocmine_hn1024_pos", model, replace(train, name="vocmine_hn1024_pos", hard_negative_min=1024)),
            ("vocmine_hn2048_pos", model, replace(train, name="vocmine_hn2048_pos", hard_negative_min=2048)),
            ("vocmine_ratio32_pos", model, replace(train, name="vocmine_ratio32_pos", hard_negative_min=0, hard_negative_ratio=32.0)),
            ("vocmine_ratio64_pos", model, replace(train, name="vocmine_ratio64_pos", hard_negative_min=0, hard_negative_ratio=64.0)),
            ("vocmine_ratio64_none", model, replace(train, name="vocmine_ratio64_none", hard_negative_min=0, hard_negative_ratio=64.0, conf_neg_normalizer="none", noobj_weight=0.02)),
        ]
    model_grid = {
        "base": base,
        "base_reg": replace(base, input_dropcut=0.08, input_dropcut_boxes=2, input_dropcut_size=(0.03, 0.08), dropout=0.05, dropout_stage="all_heads"),
        "wide": replace(base, width_scale=1.25),
        "wide_reg_stable": replace(base, width_scale=1.5, input_dropcut=0.06, feature_dropcut=0.05, feature_dropcut_stage="p4", dropout=0.08, dropout_stage="all_heads"),
    }
    opt_grid = {
        "rmsprop_const": TrainConfig(epochs=32, batch_size=16, optimizer="rmsprop", lr=8e-4, schedule="constant", clipnorm=0.5, noobj_weight=0.15),
        "adamw_cosine": TrainConfig(epochs=32, batch_size=16, optimizer="adamw", lr=4e-4, schedule="cosine", clipnorm=1.0, noobj_weight=0.10),
    }
    return [
        (f"{model_name}_{opt_name}", model_cfg, replace(train_cfg, name=f"{model_name}_{opt_name}", batch_size=12 if model_name == "wide_reg_stable" else train_cfg.batch_size))
        for model_name, model_cfg in model_grid.items()
        for opt_name, train_cfg in opt_grid.items()
    ]


def run_search(args, samples, classes, anchors):
    dataset_tag = _resolve_dataset_tag(args.dataset, args.dataset_format)
    root = _resolve_run_dir(args.run_dir, dataset_tag)
    root.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, test_samples, _ = _prepare_split_samples(args)
    jsonl = root / "grid_search.jsonl"
    rows = []
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    done = {row["name"] for row in rows}
    for name, model_cfg, train_cfg in search_configs(args.input_size, args.search_set):
        if name in done and (root / name / "summary.json").exists():
            continue
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running search config: {name}")
        row = run_once(
            samples,
            classes,
            anchors,
            model_cfg,
            train_cfg,
            run_dir,
            dataset_tag,
            args.dataset,
            args.dataset_split,
            args.val_fraction,
            args.seed,
            True,
            False,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
        )
        rows.append(
            {
                "name": name,
                "best_val_loss": row["full"]["best_val_loss"],
                "final_val_loss": row["full"]["final_val_loss"],
                "final_loss": row["full"]["final_loss"],
                "best_val_map50": row["full"].get("best_val_map50", 0.0),
                "final_val_map50": row["full"].get("final_val_map50", 0.0),
                "num_predictions_on_val0": row["full"]["num_predictions_on_val0"],
                "model": row["model"],
                "train": row["train"],
            }
        )
        with jsonl.open("a") as fp:
            fp.write(json.dumps(rows[-1], sort_keys=True) + "\n")
    rows.sort(key=lambda row: (-row.get("best_val_map50", 0.0), row["best_val_loss"]))
    _save_json(root / "grid_search_summary.json", {"best": rows[0], "results": rows})
    return rows


def collect_search_results(run_dir: str | Path):
    root = Path(run_dir)
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        row = json.loads(summary_path.read_text())
        if "full" not in row:
            continue
        rows.append(
            {
                "name": summary_path.parent.name,
                "best_val_loss": row["full"]["best_val_loss"],
                "final_val_loss": row["full"]["final_val_loss"],
                "final_loss": row["full"]["final_loss"],
                "best_val_map50": row["full"].get("best_val_map50", 0.0),
                "final_val_map50": row["full"].get("final_val_map50", 0.0),
                "num_predictions_on_val0": row["full"]["num_predictions_on_val0"],
                "model": row["model"],
                "train": row["train"],
            }
        )
    rows.sort(key=lambda row: (-row.get("best_val_map50", 0.0), row["best_val_loss"]))
    if rows:
        (root / "grid_search.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        _save_json(root / "grid_search_summary.json", {"best": rows[0], "results": rows})
    return rows


def run_named_search_config(args, samples, classes, anchors):
    dataset_tag = _resolve_dataset_tag(args.dataset, args.dataset_format)
    matches = [(name, model_cfg, train_cfg) for name, model_cfg, train_cfg in search_configs(args.input_size, args.search_set) if name == args.config_name]
    if not matches:
        raise KeyError(f"Unknown config: {args.config_name}")
    name, model_cfg, train_cfg = matches[0]
    run_dir = _resolve_run_dir(args.run_dir, dataset_tag) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, test_samples, _ = _prepare_split_samples(args)
    return run_once(
        samples,
        classes,
        anchors,
        model_cfg,
        train_cfg,
        run_dir,
        dataset_tag,
        args.dataset,
        args.dataset_split,
        args.val_fraction,
        args.seed,
        args.skip_overfit,
        args.skip_full,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )


def main():
    args = parse_args()
    input_size = _resolve_input_size_arg(args)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    dataset_tag = _resolve_dataset_tag(args.dataset, args.dataset_format)
    devices = _configure_distribution()
    print("Keras devices:", devices)
    train_samples, val_samples, test_samples, split_classes = _prepare_split_samples(args)
    classes = split_classes
    samples = train_samples + val_samples + test_samples
    anchors = fit_anchors(train_samples, input_size, seed=args.seed)
    if args.search:
        if args.config_name:
            run_named_search_config(args, samples, classes, anchors)
            collect_search_results(_resolve_run_dir(args.run_dir, dataset_tag))
            return
        run_search(args, samples, classes, anchors)
        return
    model_cfg = ModelConfig(
        input_size=input_size,
        thermo_levels=args.thermo_levels,
        quant_levels=args.quant_levels,
        output_levels=args.output_levels,
        binary_batchnorm=args.binary_batchnorm,
        bn_momentum=args.bn_momentum,
        bn_quantize_input=args.bn_quantize_input,
        bn_quantize_centered=args.bn_quantize_centered,
        bn_quantize_normalized=args.bn_quantize_normalized,
        bn_quantize_output=args.bn_quantize_output,
        bn_quantize_eval_stats=args.bn_quantize_eval_stats,
        bbox_output_scale=args.bbox_output_scale,
        obj_output_scale=args.obj_output_scale,
        cls_output_scale=args.cls_output_scale,
        head_proj_raw=args.head_proj_raw,
        head_proj_levels=args.head_proj_levels,
        input_dropcut=args.input_dropcut,
        input_dropcut_boxes=args.input_dropcut_boxes,
        input_dropcut_size=(args.input_dropcut_size_min, args.input_dropcut_size_max),
        input_dropcut_method=args.input_dropcut_method,
        input_dropcut2=args.input_dropcut2,
        input_dropcut2_method=args.input_dropcut2_method,
        feature_dropcut=args.feature_dropcut,
        feature_dropcut_stage=args.feature_dropcut_stage,
        dropout=args.dropout,
        dropout_stage=args.dropout_stage,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        schedule=args.schedule,
        cosine_first_decay_epochs=args.cosine_first_decay_epochs,
        cosine_t_mul=args.cosine_t_mul,
        cosine_m_mul=args.cosine_m_mul,
        clipnorm=args.clipnorm,
        rmsprop_rho=args.rmsprop_rho,
        sgd_momentum=args.sgd_momentum,
        sgd_nesterov=args.sgd_nesterov,
        obj_gain=args.obj_gain,
        obj_pos_gain=args.obj_pos_gain,
        noobj_weight=args.noobj_weight,
        cls_neg_weight=args.cls_neg_weight,
        cls_gain=args.cls_gain,
        class_mode=args.class_mode,
        box_gain=args.box_gain,
        bbox_output_scale=args.bbox_output_scale,
        obj_output_scale=args.obj_output_scale,
        cls_output_scale=args.cls_output_scale,
        logit_scale=args.logit_scale,
        ignore_iou=args.ignore_iou,
        rescore_objectness=args.rescore_objectness,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_negative_min=args.hard_negative_min,
        hard_negative_min_p3=args.hard_negative_min_p3,
        hard_negative_min_p4=args.hard_negative_min_p4,
        hard_negative_min_p5=args.hard_negative_min_p5,
        conf_neg_normalizer=args.conf_neg_normalizer,
        positive_top_k=args.positive_top_k,
        box_mode=args.box_mode,
        score_threshold=args.score_threshold,
        objectness_threshold=args.objectness_threshold,
        overfit_epochs=args.overfit_epochs,
        overfit_batch_size=args.overfit_batch_size,
        overfit_repeat=args.overfit_repeat,
        overfit_lr=args.overfit_lr,
        train_augment=args.train_augment,
        augment_start_strength=args.augment_start_strength,
        augment_ramp_epochs=args.augment_ramp_epochs,
        loader_workers=args.loader_workers,
        loader_queue_size=args.loader_queue_size,
        loader_multiprocessing=args.loader_multiprocessing,
    )
    run_dir = _resolve_run_dir(args.run_dir, dataset_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_once(
        samples,
        classes,
        anchors,
        model_cfg,
        train_cfg,
        run_dir,
        dataset_tag,
        args.dataset,
        args.dataset_split,
        args.val_fraction,
        args.seed,
        args.skip_overfit,
        args.skip_full,
        warm_start_weights=args.warm_start_weights,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )


if __name__ == "__main__":
    main()
