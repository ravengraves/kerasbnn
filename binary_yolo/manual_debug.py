from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
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
import jax.numpy as jnp
import keras
import numpy as np

import binary_layers  # noqa: F401
from binary_yolo.debug_overfit import _cxcywh_to_xyxy, _decode_entry, _iou_xyxy, _target_locations, _top_candidates
from binary_yolo.model import ModelConfig, build_binary_yolov3_small
from binary_yolo.train_fruits import TrainConfig, _optimizer
from binary_yolo.yad2k_core import HEADS, build_targets, find_sample, fit_anchors, load_dataset, make_losses, map50, predict_boxes, split_samples

keras.config.set_dtype_policy("mixed_float16")


PROBE_LAYERS = [
    "stage3_conv",
    "stage4_conv",
    "stage5_conv",
    "p5_feat_conv",
    "p5_pre_conv",
    "p5_proj_conv",
    "p4_feat_conv",
    "p4_pre_conv",
    "p4_proj_conv",
    "p3_feat_conv",
    "p3_pre_conv",
    "p3_proj_conv",
]


def _repeat_data(data, repeats: int):
    images, targets = data
    return np.repeat(images, repeats, axis=0), {head: np.repeat(value, repeats, axis=0) for head, value in targets.items()}


def _device_to_host(tree):
    return jax.device_get(tree)


def _assign_model_state(model, trainable, non_trainable):
    for ref, value in zip(model.trainable_variables, _device_to_host(trainable)):
        ref.assign(value)
    for ref, value in zip(model.non_trainable_variables, _device_to_host(non_trainable)):
        ref.assign(value)


def _group_for_path(path: str) -> str:
    prefix = path.split("/")[0]
    if prefix.startswith(("stage", "up3_reduce", "up4_reduce")):
        return "backbone"
    for head in HEADS:
        if prefix.startswith(f"{head}_feat"):
            return f"{head}_feat"
        if prefix.startswith((f"{head}_pre", f"{head}_proj")):
            return f"{head}_shared"
        if prefix.startswith(f"{head}_bbox"):
            return f"{head}_bbox"
        if prefix.startswith(f"{head}_obj"):
            return f"{head}_obj"
        if prefix.startswith(f"{head}_cls"):
            return f"{head}_cls"
    return prefix


def _activation_stats(outputs: dict[str, np.ndarray]) -> dict[str, dict]:
    stats = {}
    for name, arr in outputs.items():
        arr = np.asarray(arr)
        flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
        stats[name] = {
            "shape": list(arr.shape),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "spatial_channel_std": float(np.mean(np.std(flat, axis=0))),
            "unique_rows": int(len(np.unique(flat, axis=0))),
            "plus_one_frac": float(np.mean(flat > 0.0)),
        }
    return stats


def _raw_head_stats(preds: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for head, arr in preds.items():
        flat = np.asarray(arr).reshape(-1, arr.shape[-1]).astype(np.float32)
        out[head] = {
            "shape": list(arr.shape),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "unique_rows": int(len(np.unique(flat.astype(np.int32), axis=0))),
            "obj_unique": int(len(np.unique(flat[:, 4].astype(np.int32)))),
            "bbox_mean_abs": float(np.mean(np.abs(flat[:, :4]))),
            "obj_mean_abs": float(np.mean(np.abs(flat[:, 4]))),
            "cls_mean_abs": float(np.mean(np.abs(flat[:, 5:]))),
        }
    return out


def _prediction_debug(
    preds: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    sample,
    classes: list[str],
    anchors: np.ndarray,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
):
    target_rows = _target_locations(targets)
    topk_rows, all_rows = _top_candidates(preds, anchors, model_cfg.input_size, cfg.output_scale, cfg.logit_scale, topk=20)
    gt_boxes = [_cxcywh_to_xyxy(np.asarray(row["target"][:4], np.float32)) for row in target_rows]
    gt_classes = [row["target_class"] for row in target_rows]
    for row in target_rows:
        raw = preds[row["head"]][0, row["grid_y"], row["grid_x"], row["anchor_local"]]
        row["pred"] = _decode_entry(
            raw,
            row["grid_y"],
            row["grid_x"],
            row["head"],
            row["anchor_id"],
            anchors,
            model_cfg.input_size,
            cfg.output_scale,
            cfg.logit_scale,
        )
        row["iou_to_target"] = _iou_xyxy(np.asarray(row["pred"]["box_xyxy"], np.float32), np.asarray(row["target_box_xyxy"], np.float32))
    best_iou_any = []
    matching_rows = []
    background_rows = []
    for gt_idx, gt in enumerate(gt_boxes):
        best = max(all_rows, key=lambda row: _iou_xyxy(np.asarray(row["box_xyxy"], np.float32), gt))
        best_iou_any.append(
            {
                "gt_index": gt_idx,
                "gt_class": gt_classes[gt_idx],
                "best_iou": float(_iou_xyxy(np.asarray(best["box_xyxy"], np.float32), gt)),
                "best_score": float(best["score"]),
                "best_obj_prob": float(best["obj_prob"]),
                "best_cls_id": int(best["cls_id"]),
                "best_cls_score": float(best["cls_score"]),
                "head": best["head"],
                "grid_y": int(best["grid_y"]),
                "grid_x": int(best["grid_x"]),
                "anchor_id": int(best["anchor_id"]),
            }
        )
    for row in all_rows:
        if not gt_boxes:
            continue
        ious = [_iou_xyxy(np.asarray(row["box_xyxy"], np.float32), gt) for gt in gt_boxes]
        best_gt = int(np.argmax(ious))
        best_iou = float(max(ious))
        packed = {
            "head": row["head"],
            "grid_y": int(row["grid_y"]),
            "grid_x": int(row["grid_x"]),
            "anchor_id": int(row["anchor_id"]),
            "score": float(row["score"]),
            "obj_prob": float(row["obj_prob"]),
            "cls_id": int(row["cls_id"]),
            "cls_score": float(row["cls_score"]),
            "best_iou": best_iou,
            "best_gt": best_gt,
            "best_gt_class": gt_classes[best_gt],
            "class_match": bool(row["cls_id"] == gt_classes[best_gt]),
            "box_xyxy": row["box_xyxy"],
        }
        if best_iou >= 0.5 and packed["class_match"]:
            matching_rows.append(packed)
        if best_iou < 0.1:
            background_rows.append(packed)
    matching_rows.sort(key=lambda row: row["score"], reverse=True)
    background_rows.sort(key=lambda row: row["score"], reverse=True)
    preds_default = predict_boxes(
        preds,
        anchors,
        model_cfg.input_size,
        score=cfg.score_threshold,
        objectness=cfg.objectness_threshold,
        output_scale=cfg.output_scale,
        logit_scale=cfg.logit_scale,
    )
    preds_loose = predict_boxes(
        preds,
        anchors,
        model_cfg.input_size,
        score=0.01,
        objectness=0.0,
        output_scale=cfg.output_scale,
        logit_scale=cfg.logit_scale,
    )
    positive_scores = [float(row["pred"]["score"]) for row in target_rows]
    positive_objs = [float(row["pred"]["obj_prob"]) for row in target_rows]
    positive_cls = [float(row["pred"]["cls_score"]) for row in target_rows]
    return {
        "map50_default": float(
            map50(
                [sample],
                preds,
                anchors,
                model_cfg.input_size,
                len(classes),
                score=cfg.score_threshold,
                objectness=cfg.objectness_threshold,
                output_scale=cfg.output_scale,
                logit_scale=cfg.logit_scale,
            )
        ),
        "map50_loose": float(
            map50(
                [sample],
                preds,
                anchors,
                model_cfg.input_size,
                len(classes),
                score=0.01,
                objectness=0.0,
                output_scale=cfg.output_scale,
                logit_scale=cfg.logit_scale,
            )
        ),
        "prediction_count_default": int(len(preds_default[0])),
        "prediction_count_loose": int(len(preds_loose[0])),
        "positive_score_mean": float(np.mean(positive_scores)) if positive_scores else 0.0,
        "positive_obj_mean": float(np.mean(positive_objs)) if positive_objs else 0.0,
        "positive_cls_mean": float(np.mean(positive_cls)) if positive_cls else 0.0,
        "best_matching_score": float(matching_rows[0]["score"]) if matching_rows else 0.0,
        "best_background_score": float(background_rows[0]["score"]) if background_rows else 0.0,
        "target_rows": target_rows,
        "best_iou_any": best_iou_any,
        "matching_rows": matching_rows[:20],
        "background_rows": background_rows[:20],
        "topk_rows": topk_rows,
        "raw_head_stats": _raw_head_stats(preds),
    }


def _summarize_grads(var_paths: list[str], batch_grad_norms: list[np.ndarray], batch_global_norms: list[float]) -> dict:
    grad_arr = np.stack(batch_grad_norms)
    mean_by_var = grad_arr.mean(axis=0)
    groups = defaultdict(list)
    for path, value in zip(var_paths, mean_by_var):
        groups[_group_for_path(path)].append(float(value))
    top_vars = sorted(zip(var_paths, mean_by_var.tolist()), key=lambda item: item[1], reverse=True)[:15]
    return {
        "global_mean": float(np.mean(batch_global_norms)),
        "global_max": float(np.max(batch_global_norms)),
        "groups": {name: {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "count": len(vals)} for name, vals in sorted(groups.items())},
        "top_vars": [{"path": path, "grad_norm": float(value)} for path, value in top_vars],
    }


def _make_debug_functions(model, loss_fns, optimizer):
    def _loss_and_aux(trainable, non_trainable, x, y, training):
        preds, non_trainable = model.stateless_call(trainable, non_trainable, x, training=training)
        head_losses = {head: loss_fns[head](y[head], preds[head]) for head in HEADS}
        total = sum(head_losses.values())
        return total, (non_trainable, preds, head_losses)

    grad_fn = jax.value_and_grad(_loss_and_aux, argnums=0, has_aux=True)

    @jax.jit
    def train_step(trainable, non_trainable, opt_vars, x, y):
        (loss, (non_trainable, _, head_losses)), grads = grad_fn(trainable, non_trainable, x, y, True)
        grad_norms = tuple(jnp.linalg.norm(jnp.ravel(jnp.asarray(g, dtype=jnp.float32))) for g in grads)
        global_grad = jnp.sqrt(sum(jnp.sum(jnp.square(jnp.asarray(g, dtype=jnp.float32))) for g in grads))
        trainable, opt_vars = optimizer.stateless_apply(opt_vars, grads, trainable)
        return trainable, non_trainable, opt_vars, loss, head_losses, grad_norms, global_grad

    @jax.jit
    def eval_step(trainable, non_trainable, x, y):
        loss, (non_trainable, _, head_losses) = _loss_and_aux(trainable, non_trainable, x, y, False)
        return non_trainable, loss, head_losses

    return train_step, eval_step


def _log_epoch(epoch: int, epochs: int, report: dict):
    print(
        f"epoch {epoch}/{epochs} "
        f"loss={report['train_loss']:.3f} "
        f"val_loss={report['val_loss']:.3f} "
        f"map50={report['prediction']['map50_default']:.4f} "
        f"map50_loose={report['prediction']['map50_loose']:.4f} "
        f"preds={report['prediction']['prediction_count_default']} "
        f"preds_loose={report['prediction']['prediction_count_loose']} "
        f"grad={report['gradients']['global_mean']:.3f} "
        f"time={report['epoch_seconds']:.1f}s"
    )
    for row in report["prediction"]["target_rows"]:
        pred = row["pred"]
        print(
            f"  gt head={row['head']} cls={row['target_class']} "
            f"iou={row['iou_to_target']:.3f} obj={pred['obj_prob']:.3f} "
            f"cls_id={pred['cls_id']} cls={pred['cls_score']:.3f} score={pred['score']:.3f}"
        )
    print(
        "  best_match="
        f"{report['prediction']['best_matching_score']:.3f} "
        "best_bg="
        f"{report['prediction']['best_background_score']:.3f} "
        "p3_unique="
        f"{report['prediction']['raw_head_stats']['p3']['unique_rows']} "
        "p4_unique="
        f"{report['prediction']['raw_head_stats']['p4']['unique_rows']} "
        "p5_unique="
        f"{report['prediction']['raw_head_stats']['p5']['unique_rows']}"
    )
    print(
        "  probe "
        + " ".join(
            f"{name}:{stats['unique_rows']}/{stats['spatial_channel_std']:.3f}"
            for name, stats in report["probe"].items()
            if name in {"p3_pre_conv", "p4_pre_conv", "p5_pre_conv", "p3_proj_conv", "p4_proj_conv", "p5_proj_conv"}
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fruits_detection")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", default="fruit0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clipnorm", type=float, default=0.0)
    ap.add_argument("--thermo-levels", type=int, default=1)
    ap.add_argument("--auto-scale-loss", action="store_true", default=True)
    ap.add_argument("--no-auto-scale-loss", dest="auto_scale_loss", action="store_false")
    ap.add_argument("--obj-gain", type=float, default=1.0)
    ap.add_argument("--obj-pos-gain", type=float, default=1.0)
    ap.add_argument("--noobj-weight", type=float, default=0.2)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--width-scale", type=float, default=1.0)
    ap.add_argument("--stem-filters", type=int, default=0)
    ap.add_argument("--obj-hidden", type=int, default=64)
    ap.add_argument("--obj-remap-hidden", type=int, default=0)
    ap.add_argument("--quant-levels", type=int, default=128)
    ap.add_argument("--output-levels", type=int, default=128)
    ap.add_argument("--output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--score-threshold", type=float, default=0.2)
    ap.add_argument("--objectness-threshold", type=float, default=0.05)
    ap.add_argument("--ignore-iou", type=float, default=0.5)
    ap.add_argument("--no-rescore-objectness", action="store_true")
    ap.add_argument("--hard-negative-ratio", type=float, default=0.0)
    ap.add_argument("--hard-negative-min", type=int, default=0)
    ap.add_argument("--conf-neg-normalizer", choices=["neg", "pos", "none"], default="neg")
    ap.add_argument("--optimizer", choices=["adam", "adamw", "rmsprop"], default="adam")
    ap.add_argument("--head-proj-raw", action="store_true")
    ap.add_argument("--head-proj-levels", type=int, default=0)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    samples, classes = load_dataset(args.dataset)
    train_samples, _ = split_samples(samples, val_fraction=0.1, seed=args.seed)
    sample = find_sample(samples, args.sample)
    model_cfg = ModelConfig(
        input_size=args.input_size,
        thermo_levels=args.thermo_levels,
        width_scale=args.width_scale,
        stem_filters=args.stem_filters,
        quant_levels=args.quant_levels,
        output_levels=args.output_levels,
        head_output_scale=args.output_scale,
        head_proj_raw=args.head_proj_raw,
        head_proj_levels=args.head_proj_levels,
        obj_cls_hidden=args.obj_hidden,
        obj_remap_hidden=args.obj_remap_hidden,
    )
    train_cfg = TrainConfig(
        optimizer=args.optimizer,
        lr=args.lr,
        overfit_lr=args.lr,
        overfit_epochs=args.epochs,
        overfit_repeat=args.repeat,
        overfit_batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        objectness_threshold=args.objectness_threshold,
        ignore_iou=args.ignore_iou,
        rescore_objectness=not args.no_rescore_objectness,
        clipnorm=args.clipnorm,
        auto_scale_loss=args.auto_scale_loss,
        obj_gain=args.obj_gain,
        obj_pos_gain=args.obj_pos_gain,
        noobj_weight=args.noobj_weight,
        output_scale=args.output_scale,
        logit_scale=args.logit_scale,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_negative_min=args.hard_negative_min,
        conf_neg_normalizer=args.conf_neg_normalizer,
    )
    anchors = fit_anchors(train_samples, model_cfg.input_size, seed=args.seed)
    val_images, val_targets = build_targets([sample], len(classes), anchors, model_cfg.input_size, model_cfg.thermo_levels, augment=False)
    train_images, train_targets = _repeat_data((val_images, val_targets), args.repeat)
    model = build_binary_yolov3_small(len(classes), model_cfg)
    _ = model(val_images[:1], training=False)
    probe_model = keras.Model(model.input, {name: model.get_layer(name).output for name in PROBE_LAYERS})
    steps = max(1, math.ceil(len(train_images) / args.batch_size))
    opt_cfg = replace(train_cfg, epochs=args.epochs, batch_size=args.batch_size)
    optimizer = _optimizer(opt_cfg, steps)
    if train_cfg.auto_scale_loss:
        optimizer = keras.optimizers.LossScaleOptimizer(optimizer)
    optimizer.build(model.trainable_variables)
    loss_fns = make_losses(
        anchors,
        model_cfg.input_size,
        len(classes),
        noobj_weight=train_cfg.noobj_weight,
        cls_neg_weight=train_cfg.cls_neg_weight,
        obj_gain=train_cfg.obj_gain,
        obj_pos_gain=train_cfg.obj_pos_gain,
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
    reports = []

    meta = {
        "devices": [str(device) for device in jax.devices()],
        "sample": args.sample,
        "classes": classes,
        "anchors": anchors.tolist(),
        "model": model_cfg.__dict__,
        "train": {
            "epochs": args.epochs,
            "repeat": args.repeat,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "thermo_levels": args.thermo_levels,
            "optimizer": args.optimizer,
            "clipnorm": args.clipnorm,
            "obj_hidden": args.obj_hidden,
            "obj_remap_hidden": args.obj_remap_hidden,
            "obj_gain": args.obj_gain,
            "obj_pos_gain": args.obj_pos_gain,
            "noobj_weight": args.noobj_weight,
            "output_scale": args.output_scale,
            "logit_scale": args.logit_scale,
            "hard_negative_ratio": args.hard_negative_ratio,
            "hard_negative_min": args.hard_negative_min,
            "conf_neg_normalizer": args.conf_neg_normalizer,
            "auto_scale_loss": args.auto_scale_loss,
            "score_threshold": args.score_threshold,
            "objectness_threshold": args.objectness_threshold,
            "ignore_iou": args.ignore_iou,
            "rescore_objectness": not args.no_rescore_objectness,
            "head_proj_raw": args.head_proj_raw,
            "head_proj_levels": args.head_proj_levels,
            "quant_levels": args.quant_levels,
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"devices={meta['devices']} sample={args.sample} repeat={args.repeat} batch_size={args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        batch_losses = []
        batch_head_losses = []
        batch_grad_norms = []
        batch_global_norms = []
        for start in range(0, len(train_images), args.batch_size):
            end = start + args.batch_size
            xb = train_images[start:end]
            yb = {head: value[start:end] for head, value in train_targets.items()}
            trainable, non_trainable, opt_vars, loss, head_losses, grad_norms, global_grad = train_step(trainable, non_trainable, opt_vars, xb, yb)
            loss = float(jax.block_until_ready(loss))
            head_losses = {head: float(jax.block_until_ready(value)) for head, value in head_losses.items()}
            grad_norms = np.asarray(_device_to_host(grad_norms), np.float32)
            global_grad = float(jax.block_until_ready(global_grad))
            batch_losses.append(loss)
            batch_head_losses.append(head_losses)
            batch_grad_norms.append(grad_norms)
            batch_global_norms.append(global_grad)
        non_trainable_eval, val_loss, val_head_losses = eval_step(trainable, non_trainable, val_images, val_targets)
        non_trainable = non_trainable_eval
        val_loss = float(jax.block_until_ready(val_loss))
        val_head_losses = {head: float(jax.block_until_ready(value)) for head, value in val_head_losses.items()}
        _assign_model_state(model, trainable, non_trainable)
        preds = {head: np.asarray(value) for head, value in model(val_images, training=False).items()}
        probe = {name: np.asarray(value) for name, value in probe_model(val_images, training=False).items()}
        prediction = _prediction_debug(preds, val_targets, sample, classes, anchors, train_cfg, model_cfg)
        prediction["topk_rows"] = prediction["topk_rows"][: args.topk]
        report = {
            "epoch": epoch,
            "epoch_seconds": float(time.time() - t0),
            "train_loss": float(np.mean(batch_losses)),
            "train_head_losses": {head: float(np.mean([row[head] for row in batch_head_losses])) for head in HEADS},
            "val_loss": val_loss,
            "val_head_losses": val_head_losses,
            "gradients": _summarize_grads(var_paths, batch_grad_norms, batch_global_norms),
            "prediction": prediction,
            "probe": _activation_stats(probe),
        }
        reports.append(report)
        (run_dir / f"epoch_{epoch:02d}.json").write_text(json.dumps(report, indent=2))
        _log_epoch(epoch, args.epochs, report)

    summary = {
        "meta": meta,
        "epochs": [
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
                "map50_default": row["prediction"]["map50_default"],
                "map50_loose": row["prediction"]["map50_loose"],
                "prediction_count_default": row["prediction"]["prediction_count_default"],
                "prediction_count_loose": row["prediction"]["prediction_count_loose"],
                "positive_score_mean": row["prediction"]["positive_score_mean"],
                "positive_obj_mean": row["prediction"]["positive_obj_mean"],
                "best_matching_score": row["prediction"]["best_matching_score"],
                "best_background_score": row["prediction"]["best_background_score"],
                "p3_unique": row["prediction"]["raw_head_stats"]["p3"]["unique_rows"],
                "p4_unique": row["prediction"]["raw_head_stats"]["p4"]["unique_rows"],
                "p5_unique": row["prediction"]["raw_head_stats"]["p5"]["unique_rows"],
                "grad_global_mean": row["gradients"]["global_mean"],
            }
            for row in reports
        ]
    }
    (run_dir / "manual_debug_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
