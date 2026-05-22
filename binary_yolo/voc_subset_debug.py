from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

ROOT = Path(__file__).resolve().parents[1]
BITBNN = ROOT / "bitbnn"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BITBNN) not in sys.path:
    sys.path.insert(0, str(BITBNN))

import jax
import keras
import numpy as np

import binary_layers  # noqa: F401
from binary_yolo.debug_objectness import OBJ_FEAT_LAYERS, _head_report
from binary_yolo.manual_debug import _activation_stats, _assign_model_state, _device_to_host, _make_debug_functions, _summarize_grads
from binary_yolo.model import ModelConfig, build_binary_yolov3_small
from binary_yolo.train_fruits import TrainConfig, _optimizer, _subset_samples
from binary_yolo.yad2k_core import HEADS, build_targets, fit_anchors, load_dataset_splits, make_losses, map50

keras.config.set_dtype_policy("mixed_float16")


def _batch_slice(targets: dict[str, np.ndarray], start: int, stop: int):
    return {head: value[start:stop] for head, value in targets.items()}


def _evaluate_loss(eval_step, trainable, non_trainable, images: np.ndarray, targets: dict[str, np.ndarray], batch_size: int):
    losses = []
    head_rows = []
    next_non_trainable = non_trainable
    for start in range(0, len(images), batch_size):
        stop = start + batch_size
        x = jax.device_put(images[start:stop])
        y = {head: jax.device_put(value[start:stop]) for head, value in targets.items()}
        next_non_trainable, loss, head_losses = eval_step(trainable, next_non_trainable, x, y)
        losses.append(float(jax.block_until_ready(loss)))
        head_rows.append({head: float(jax.block_until_ready(value)) for head, value in head_losses.items()})
    mean_head = {head: float(np.mean([row[head] for row in head_rows])) for head in HEADS}
    return next_non_trainable, float(np.mean(losses)), mean_head


def _head_probe(model, images: np.ndarray, targets: dict[str, np.ndarray], anchors: np.ndarray, cfg: TrainConfig, model_cfg: ModelConfig, num_classes: int, probe_limit: int):
    probe_images = images[:probe_limit]
    probe_targets = {head: value[:probe_limit] for head, value in targets.items()}
    probe_model = keras.Model(
        model.input,
        {
            **{f"{head}_pred": model.output[head] for head in HEADS},
            **{f"{head}_feat": model.get_layer(name).output for head, name in OBJ_FEAT_LAYERS.items()},
            **{f"{head}_pre": model.get_layer(f"{head}_pre_conv").output for head in HEADS},
            **{f"{head}_proj": model.get_layer(f"{head}_proj_conv").output for head in HEADS},
        },
    )
    out = probe_model.predict(probe_images, batch_size=min(16, len(probe_images)), verbose=0)
    report = {}
    for head_idx, head in enumerate(HEADS):
        report[head] = _head_report(
            head,
            probe_targets[head],
            out[f"{head}_pred"],
            out[f"{head}_feat"],
            anchors[head_idx * 3 : (head_idx + 1) * 3],
            num_classes,
            cfg,
            model_cfg.input_size,
        )
        report[head]["pre_activation"] = _activation_stats({f"{head}_pre": out[f"{head}_pre"]})[f"{head}_pre"]
        report[head]["proj_activation"] = _activation_stats({f"{head}_proj": out[f"{head}_proj"]})[f"{head}_proj"]
    return report


def _force_objectness_on(preds: dict[str, np.ndarray], output_scale: float):
    forced = {}
    high = np.float32(127.0 * output_scale)
    for head, value in preds.items():
        arr = np.array(value, copy=True)
        arr[..., 4] = high
        forced[head] = arr
    return forced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="VOCdevkit/VOC2012")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-fraction", type=float, default=0.05)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--probe-limit", type=int, default=32)
    ap.add_argument("--optimizer", choices=["adam", "adamw", "rmsprop", "sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--clipnorm", type=float, default=0.5)
    ap.add_argument("--sgd-momentum", type=float, default=0.9)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--width-scale", type=float, default=1.25)
    ap.add_argument("--stem-filters", type=int, default=0)
    ap.add_argument("--float-head-convs", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--head-pre-filters", type=int, default=192)
    ap.add_argument("--head-channels", type=int, default=32)
    ap.add_argument("--obj-hidden", type=int, default=64)
    ap.add_argument("--obj-remap-hidden", type=int, default=0)
    ap.add_argument("--share-obj-cls-hidden", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-obj-cls-hidden", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--quant-levels", type=int, default=8)
    ap.add_argument("--head-proj-levels", type=int, default=8)
    ap.add_argument("--output-levels", type=int, default=128)
    ap.add_argument("--head-output-scale", type=float, default=2.0)
    ap.add_argument("--input-dropcut", type=float, default=0.0)
    ap.add_argument("--input-dropcut-method", default="both")
    ap.add_argument("--obj-gain", type=float, default=3.5)
    ap.add_argument("--obj-pos-gain", type=float, default=7.0)
    ap.add_argument("--noobj-weight", type=float, default=0.04)
    ap.add_argument("--cls-neg-weight", type=float, default=0.0)
    ap.add_argument("--ignore-iou", type=float, default=0.5)
    ap.add_argument("--hard-negative-ratio", type=float, default=0.0)
    ap.add_argument("--hard-negative-min", type=int, default=1024)
    ap.add_argument("--conf-neg-normalizer", choices=["neg", "pos", "none"], default="pos")
    ap.add_argument("--rescore-objectness", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--score-threshold", type=float, default=0.2)
    ap.add_argument("--metric-score-threshold", type=float, default=0.05)
    ap.add_argument("--objectness-threshold", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, classes = load_dataset_splits(
        args.dataset,
        dataset_format="voc",
        split_mode="official",
        seed=args.seed,
        include_difficult=False,
        input_size=224,
    )
    train_samples = _subset_samples(train_samples, args.train_fraction, args.seed)
    val_samples = _subset_samples(val_samples, args.val_fraction, args.seed + 1)
    anchors = fit_anchors(train_samples, 224, seed=args.seed)

    model_cfg = ModelConfig(
        input_size=224,
        thermo_levels=args.thermo_levels,
        width_scale=args.width_scale,
        stem_filters=args.stem_filters,
        float_head_convs=args.float_head_convs,
        input_dropcut=args.input_dropcut,
        input_dropcut_method=args.input_dropcut_method,
        head_pre_filters=args.head_pre_filters,
        head_channels=args.head_channels,
        head_proj_raw=True,
        head_proj_levels=args.head_proj_levels,
        obj_cls_hidden=args.obj_hidden,
        obj_remap_hidden=args.obj_remap_hidden,
        share_obj_cls_hidden=args.share_obj_cls_hidden,
        float_obj_cls_hidden=args.float_obj_cls_hidden,
        quant_levels=args.quant_levels,
        output_levels=args.output_levels,
        head_output_scale=args.head_output_scale,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        schedule=args.schedule,
        clipnorm=args.clipnorm,
        sgd_momentum=args.sgd_momentum,
        obj_gain=args.obj_gain,
        obj_pos_gain=args.obj_pos_gain,
        noobj_weight=args.noobj_weight,
        cls_neg_weight=args.cls_neg_weight,
        output_scale=args.head_output_scale,
        logit_scale=32.0,
        ignore_iou=args.ignore_iou,
        rescore_objectness=args.rescore_objectness,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_negative_min=args.hard_negative_min,
        conf_neg_normalizer=args.conf_neg_normalizer,
        objectness_threshold=args.objectness_threshold,
        score_threshold=args.score_threshold,
        auto_scale_loss=True,
    )

    train_images, train_targets = build_targets(train_samples, len(classes), anchors, model_cfg.input_size, model_cfg.thermo_levels, augment=False)
    val_images, val_targets = build_targets(val_samples, len(classes), anchors, model_cfg.input_size, model_cfg.thermo_levels, augment=False)

    model = build_binary_yolov3_small(len(classes), model_cfg)
    _ = model(train_images[:1], training=False)
    steps = max(1, math.ceil(len(train_images) / args.batch_size))
    optimizer = _optimizer(train_cfg, steps)
    if train_cfg.auto_scale_loss:
        optimizer = keras.optimizers.LossScaleOptimizer(optimizer)
    optimizer.build(model.trainable_variables)
    loss_fns = make_losses(
        anchors,
        model_cfg.input_size,
        len(classes),
        obj_gain=train_cfg.obj_gain,
        obj_pos_gain=train_cfg.obj_pos_gain,
        noobj_weight=train_cfg.noobj_weight,
        cls_neg_weight=train_cfg.cls_neg_weight,
        box_gain=train_cfg.box_gain,
        output_scale=train_cfg.output_scale,
        logit_scale=train_cfg.logit_scale,
        ignore_iou=train_cfg.ignore_iou,
        rescore_objectness=train_cfg.rescore_objectness,
        hard_negative_ratio=train_cfg.hard_negative_ratio,
        hard_negative_min=train_cfg.hard_negative_min,
        conf_neg_normalizer=train_cfg.conf_neg_normalizer,
    )
    train_step, eval_step = _make_debug_functions(model, loss_fns, optimizer)
    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    opt_vars = [v.value for v in optimizer.variables]
    var_paths = [v.path for v in model.trainable_variables]

    summary = {
        "dataset": {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "classes": classes,
        },
        "anchors": anchors.tolist(),
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "metric_score_threshold": args.metric_score_threshold,
        "epochs": [],
    }
    (run_dir / "meta.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "epochs"}, indent=2))
    print(f"devices={[str(device) for device in jax.devices()]} train={len(train_samples)} val={len(val_samples)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        batch_losses = []
        batch_head_losses = []
        batch_grad_norms = []
        batch_global_norms = []
        for start in range(0, len(train_images), args.batch_size):
            stop = start + args.batch_size
            xb = jax.device_put(train_images[start:stop])
            yb = {head: jax.device_put(value[start:stop]) for head, value in train_targets.items()}
            trainable, non_trainable, opt_vars, loss, head_losses, grad_norms, global_grad = train_step(trainable, non_trainable, opt_vars, xb, yb)
            batch_losses.append(float(jax.block_until_ready(loss)))
            batch_head_losses.append({head: float(jax.block_until_ready(value)) for head, value in head_losses.items()})
            batch_grad_norms.append(np.asarray(_device_to_host(grad_norms), np.float32))
            batch_global_norms.append(float(jax.block_until_ready(global_grad)))

        non_trainable, val_loss, val_head_losses = _evaluate_loss(eval_step, trainable, non_trainable, val_images, val_targets, args.batch_size)
        non_trainable, train_eval_loss, train_eval_head_losses = _evaluate_loss(eval_step, trainable, non_trainable, train_images, train_targets, args.batch_size)
        _assign_model_state(model, trainable, non_trainable)
        train_preds = model.predict(train_images, batch_size=args.batch_size, verbose=0)
        val_preds = model.predict(val_images, batch_size=args.batch_size, verbose=0)
        train_map = float(map50(train_samples, train_preds, anchors, model_cfg.input_size, len(classes), score=args.metric_score_threshold, objectness=train_cfg.objectness_threshold, output_scale=train_cfg.output_scale, logit_scale=train_cfg.logit_scale))
        val_map = float(map50(val_samples, val_preds, anchors, model_cfg.input_size, len(classes), score=args.metric_score_threshold, objectness=train_cfg.objectness_threshold, output_scale=train_cfg.output_scale, logit_scale=train_cfg.logit_scale))
        train_map_cls_only = float(map50(train_samples, _force_objectness_on(train_preds, train_cfg.output_scale), anchors, model_cfg.input_size, len(classes), score=args.metric_score_threshold, objectness=0.0, output_scale=train_cfg.output_scale, logit_scale=train_cfg.logit_scale))
        val_map_cls_only = float(map50(val_samples, _force_objectness_on(val_preds, train_cfg.output_scale), anchors, model_cfg.input_size, len(classes), score=args.metric_score_threshold, objectness=0.0, output_scale=train_cfg.output_scale, logit_scale=train_cfg.logit_scale))
        probe = _head_probe(model, train_images, train_targets, anchors, train_cfg, model_cfg, len(classes), min(args.probe_limit, len(train_images)))
        report = {
            "epoch": epoch,
            "epoch_seconds": float(time.time() - t0),
            "train_loss": float(np.mean(batch_losses)),
            "train_eval_loss": train_eval_loss,
            "val_loss": val_loss,
            "train_head_losses": {head: float(np.mean([row[head] for row in batch_head_losses])) for head in HEADS},
            "train_eval_head_losses": train_eval_head_losses,
            "val_head_losses": val_head_losses,
            "train_map50": train_map,
            "val_map50": val_map,
            "train_map50_cls_only": train_map_cls_only,
            "val_map50_cls_only": val_map_cls_only,
            "gradients": _summarize_grads(var_paths, batch_grad_norms, batch_global_norms),
            "probe": probe,
        }
        summary["epochs"].append(report)
        (run_dir / f"epoch_{epoch:02d}.json").write_text(json.dumps(report, indent=2))
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={report['train_loss']:.3f} "
            f"train_eval_loss={report['train_eval_loss']:.3f} "
            f"val_loss={report['val_loss']:.3f} "
            f"train_map50={train_map:.4f} "
            f"val_map50={val_map:.4f} "
            f"train_map50_cls={train_map_cls_only:.4f} "
            f"grad={report['gradients']['global_mean']:.3f} "
            f"time={report['epoch_seconds']:.1f}s"
        )
        for head in HEADS:
            row = probe[head]
            print(
                f"  {head}: pos={row['pos_count']} neg={row['neg_count']} "
                f"obj_pos={row['obj_prob_pos_mean']:.3f} obj_neg={row['obj_prob_neg_mean']:.3f} "
                f"grad_pos={row['obj_grad_pos_mean']:.6f} grad_neg={row['obj_grad_neg_mean']:.6f} "
                f"sim={row.get('hard_neg_hamming_sim_mean', 0.0):.3f}"
            )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
