from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import sys
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
from binary_yolo.manual_debug import _assign_model_state, _device_to_host, _make_debug_functions
from binary_yolo.model import ModelConfig, build_binary_yolov3_small
from binary_yolo.train_fruits import TrainConfig, _optimizer
from binary_yolo.yad2k_core import ANCHOR_MASKS, HEADS, MAX_TRUE_BOXES, STRIDES, build_targets, find_sample, fit_anchors, load_dataset, make_losses, split_samples

keras.config.set_dtype_policy("mixed_float16")

OBJ_FEAT_LAYERS = {head: f"{head}_obj_hidden_act" for head in HEADS}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _u01(x: np.ndarray, output_scale: float) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32) / float(output_scale), -127.0, 127.0)
    return (x + 127.0) / 254.0


def _ratio_from_u(u: np.ndarray) -> np.ndarray:
    return np.where(u <= 0.5, 2.0 * u, 1.0 + 4.0 * (u - 0.5)).astype(np.float32)


def _pairwise_iou_xyxy_np(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    a = a[:, :, None, :]
    b = b[:, None, :, :]
    tl = np.maximum(a[..., :2], b[..., :2])
    br = np.minimum(a[..., 2:], b[..., 2:])
    inter_wh = np.maximum(br - tl, 0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = np.maximum(a[..., 2] - a[..., 0], 0.0) * np.maximum(a[..., 3] - a[..., 1], 0.0)
    area_b = np.maximum(b[..., 2] - b[..., 0], 0.0) * np.maximum(b[..., 3] - b[..., 1], 0.0)
    iou = inter / np.maximum(area_a + area_b - inter, 1e-6)
    return iou * valid[:, None, :]


def _decode_for_loss(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray, input_size: int, stride: int, num_classes: int, cfg: TrainConfig):
    grid = input_size // stride
    gy, gx = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32), indexing="ij")
    grid_xy = np.stack([gx, gy], axis=-1).reshape(1, grid, grid, 1, 2)
    grid_s = np.asarray([grid, grid], np.float32).reshape(1, 1, 1, 1, 2)
    anchor_wh = (anchors / input_size).reshape(1, 1, 1, len(anchors), 2)
    target_channels = 5 + num_classes

    target = y_true[..., :target_channels]
    gt_pack = y_true[:, 0, 0, 0, target_channels:].reshape(-1, MAX_TRUE_BOXES, 5)
    gt_boxes, gt_valid = gt_pack[..., :4], gt_pack[..., 4]
    obj = target[..., 4:5]
    box = target[..., :4]

    xy_p = _u01(y_pred[..., :2], cfg.output_scale)
    pred_xy = (xy_p + grid_xy) / grid_s
    pred_wh = _ratio_from_u(_u01(y_pred[..., 2:4], cfg.output_scale)) * anchor_wh
    pred_xyxy = np.concatenate([pred_xy - pred_wh * 0.5, pred_xy + pred_wh * 0.5], axis=-1)

    target_xyxy = np.concatenate([box[..., :2] - box[..., 2:4] * 0.5, box[..., :2] + box[..., 2:4] * 0.5], axis=-1)
    gt_xyxy = np.concatenate([gt_boxes[..., :2] - gt_boxes[..., 2:4] * 0.5, gt_boxes[..., :2] + gt_boxes[..., 2:4] * 0.5], axis=-1)
    pair_iou = _pairwise_iou_xyxy_np(pred_xyxy.reshape(pred_xyxy.shape[0], -1, 4), gt_xyxy, gt_valid)
    best_iou = pair_iou.max(axis=-1, keepdims=True).reshape(y_pred.shape[0], grid, grid, len(anchors), 1)
    ignore_mask = (best_iou < cfg.ignore_iou).astype(np.float32)

    tl = np.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    br = np.minimum(pred_xyxy[..., 2:4], target_xyxy[..., 2:4])
    inter = np.prod(np.maximum(br - tl, 0.0), axis=-1, keepdims=True)
    area_p = np.prod(np.maximum(pred_xyxy[..., 2:4] - pred_xyxy[..., :2], 0.0), axis=-1, keepdims=True)
    area_t = np.prod(np.maximum(target_xyxy[..., 2:4] - target_xyxy[..., :2], 0.0), axis=-1, keepdims=True)
    assigned_iou = inter / np.maximum(area_p + area_t - inter, 1e-6)

    obj_target = obj * (assigned_iou if cfg.rescore_objectness else 1.0)
    obj_logits = y_pred[..., 4:5].astype(np.float32) / float(cfg.output_scale * cfg.logit_scale)
    obj_prob = _sigmoid(obj_logits)
    cls = target[..., 5:5 + num_classes]
    cls_logits = y_pred[..., 5:5 + num_classes].astype(np.float32) / float(cfg.output_scale * cfg.logit_scale)
    cls_prob = _sigmoid(cls_logits)
    pos_count = max(float(obj.sum()), 1.0)
    neg_mask = (1.0 - obj) * ignore_mask
    neg_loss = np.maximum(obj_logits, 0.0) + np.log1p(np.exp(-np.abs(obj_logits)))
    if cfg.hard_negative_min > 0:
        flat_loss = np.where(neg_mask > 0.0, neg_loss, -1e9).reshape(-1)
        k = max(1, min(int(cfg.hard_negative_min), int(flat_loss.shape[0])))
        thresh = np.partition(flat_loss, -k)[-k]
        neg_mask = neg_mask * (neg_loss >= thresh).astype(np.float32)
    if cfg.conf_neg_normalizer == "pos":
        neg_count = pos_count
    elif cfg.conf_neg_normalizer == "none":
        neg_count = 1.0
    else:
        neg_count = max(float(neg_mask.sum()), 1.0)
    obj_grad = (
        cfg.obj_gain * cfg.obj_pos_gain * obj * (obj_prob - obj_target) / pos_count
        + cfg.obj_gain * cfg.noobj_weight * neg_mask * obj_prob / neg_count
    )
    return {
        "obj": obj[..., 0],
        "ignore_mask": ignore_mask[..., 0],
        "neg_mask": neg_mask[..., 0],
        "obj_target": obj_target[..., 0],
        "obj_logits": obj_logits[..., 0],
        "obj_prob": obj_prob[..., 0],
        "obj_grad": obj_grad[..., 0],
        "best_iou": best_iou[..., 0],
        "assigned_iou": assigned_iou[..., 0],
        "cls": cls,
        "cls_prob": cls_prob,
    }


def _feature_similarity(feat: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray, obj_prob: np.ndarray, topk: int = 16):
    rows = feat.reshape(-1, feat.shape[-1]).astype(np.int8)
    pos_flat = pos_mask.reshape(-1) > 0.5
    neg_flat = neg_mask.reshape(-1) > 0.5
    prob_flat = obj_prob.reshape(-1)
    if not np.any(pos_flat) or not np.any(neg_flat):
        return {}
    pos_rows = rows[pos_flat]
    neg_idx = np.flatnonzero(neg_flat)
    neg_idx = neg_idx[np.argsort(prob_flat[neg_idx])[::-1][:topk]]
    hard_neg = rows[neg_idx]
    sims = []
    for row in hard_neg:
        match = np.max(np.mean(row[None, :] == pos_rows, axis=-1))
        sims.append(float(match))
    return {
        "pos_unique": int(len(np.unique(pos_rows, axis=0))),
        "hard_neg_topk": int(len(hard_neg)),
        "hard_neg_hamming_sim_mean": float(np.mean(sims)) if sims else 0.0,
        "hard_neg_hamming_sim_max": float(np.max(sims)) if sims else 0.0,
    }


def _head_report(head: str, y_true: np.ndarray, y_pred: np.ndarray, feat: np.ndarray, anchors: np.ndarray, num_classes: int, cfg: TrainConfig, input_size: int):
    stats = _decode_for_loss(y_true, y_pred, anchors, input_size, STRIDES[HEADS.index(head)], num_classes, cfg)
    pos = stats["obj"] > 0.5
    neg = stats["neg_mask"] > 0.5
    ign = (stats["obj"] < 0.5) & (stats["ignore_mask"] < 0.5)
    report = {
        "pos_count": int(pos.sum()),
        "neg_count": int(neg.sum()),
        "ignored_count": int(ign.sum()),
        "assigned_iou_pos_mean": float(stats["assigned_iou"][pos].mean()) if np.any(pos) else 0.0,
        "assigned_iou_pos_max": float(stats["assigned_iou"][pos].max()) if np.any(pos) else 0.0,
        "obj_target_pos_mean": float(stats["obj_target"][pos].mean()) if np.any(pos) else 0.0,
        "obj_prob_pos_mean": float(stats["obj_prob"][pos].mean()) if np.any(pos) else 0.0,
        "obj_prob_neg_mean": float(stats["obj_prob"][neg].mean()) if np.any(neg) else 0.0,
        "obj_prob_neg_max": float(stats["obj_prob"][neg].max()) if np.any(neg) else 0.0,
        "obj_prob_ignored_mean": float(stats["obj_prob"][ign].mean()) if np.any(ign) else 0.0,
        "obj_grad_pos_mean": float(np.abs(stats["obj_grad"][pos]).mean()) if np.any(pos) else 0.0,
        "obj_grad_neg_mean": float(np.abs(stats["obj_grad"][neg]).mean()) if np.any(neg) else 0.0,
        "obj_grad_neg_max": float(np.abs(stats["obj_grad"][neg]).max()) if np.any(neg) else 0.0,
        "best_iou_neg_max": float(stats["best_iou"][neg].max()) if np.any(neg) else 0.0,
        "cls_true_prob_pos_mean": float(np.sum(stats["cls_prob"][pos] * stats["cls"][pos], axis=-1).mean()) if np.any(pos) else 0.0,
        "cls_top1_acc_pos": float(np.mean(np.argmax(stats["cls_prob"][pos], axis=-1) == np.argmax(stats["cls"][pos], axis=-1))) if np.any(pos) else 0.0,
    }
    report.update(_feature_similarity(feat, pos, neg, stats["obj_prob"]))
    return report


def _positive_assignments(targets: dict[str, np.ndarray], classes: list[str]):
    rows = []
    for head, mask in zip(HEADS, ANCHOR_MASKS):
        num_classes = targets[head].shape[-1] - 5 - 5 * MAX_TRUE_BOXES
        target = targets[head][..., : 5 + num_classes]
        batch, gy, gx, a = np.where(target[..., 4] > 0.5)
        for b, y, x, a_local in zip(batch, gy, gx, a):
            row = target[b, y, x, a_local]
            rows.append(
                {
                    "image_index": int(b),
                    "head": head,
                    "grid_y": int(y),
                    "grid_x": int(x),
                    "anchor_id": int(mask[a_local]),
                    "class_id": int(np.argmax(row[5:])),
                    "class_name": classes[int(np.argmax(row[5:]))],
                    "box": row[:4].astype(np.float32).tolist(),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fruits_detection")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--samples", nargs="+", default=["fruit0", "fruit1"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--optimizer", choices=["adam", "adamw", "rmsprop"], default="adam")
    ap.add_argument("--obj-gain", type=float, default=3.0)
    ap.add_argument("--obj-pos-gain", type=float, default=6.0)
    ap.add_argument("--noobj-weight", type=float, default=0.05)
    ap.add_argument("--ignore-iou", type=float, default=0.5)
    ap.add_argument("--no-rescore-objectness", action="store_true")
    ap.add_argument("--hard-negative-min", type=int, default=0)
    ap.add_argument("--conf-neg-normalizer", choices=["neg", "pos", "none"], default="neg")
    ap.add_argument("--head-proj-raw", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    samples, classes = load_dataset(args.dataset)
    train_samples, _ = split_samples(samples, val_fraction=0.1, seed=args.seed)
    chosen = [find_sample(samples, name) for name in args.samples]
    model_cfg = ModelConfig(head_proj_raw=args.head_proj_raw)
    cfg = TrainConfig(
        optimizer=args.optimizer,
        lr=args.lr,
        overfit_lr=args.lr,
        overfit_epochs=args.epochs,
        overfit_repeat=args.repeat,
        overfit_batch_size=args.batch_size,
        obj_gain=args.obj_gain,
        obj_pos_gain=args.obj_pos_gain,
        noobj_weight=args.noobj_weight,
        ignore_iou=args.ignore_iou,
        rescore_objectness=not args.no_rescore_objectness,
        hard_negative_min=args.hard_negative_min,
        conf_neg_normalizer=args.conf_neg_normalizer,
        auto_scale_loss=False,
    )

    anchors = fit_anchors(train_samples, model_cfg.input_size, seed=args.seed)
    base_images, base_targets = build_targets(chosen, len(classes), anchors, model_cfg.input_size, model_cfg.thermo_levels, augment=False)
    images = np.repeat(base_images, args.repeat, axis=0)
    targets = {head: np.repeat(value, args.repeat, axis=0) for head, value in base_targets.items()}

    model = build_binary_yolov3_small(len(classes), model_cfg)
    _ = model(base_images[:1], training=False)
    probe = keras.Model(
        model.input,
        {
            **{f"{head}_pred": model.output[head] for head in HEADS},
            **{f"{head}_feat": model.get_layer(name).output for head, name in OBJ_FEAT_LAYERS.items()},
        },
    )
    steps = max(1, math.ceil(len(images) / args.batch_size))
    opt = _optimizer(replace(cfg, epochs=args.epochs, batch_size=args.batch_size), steps)
    opt.build(model.trainable_variables)
    loss_fns = make_losses(
        anchors,
        model_cfg.input_size,
        len(classes),
        obj_gain=cfg.obj_gain,
        obj_pos_gain=cfg.obj_pos_gain,
        noobj_weight=cfg.noobj_weight,
        cls_neg_weight=cfg.cls_neg_weight,
        box_gain=cfg.box_gain,
        output_scale=cfg.output_scale,
        logit_scale=cfg.logit_scale,
        ignore_iou=cfg.ignore_iou,
        rescore_objectness=cfg.rescore_objectness,
        hard_negative_min=cfg.hard_negative_min,
        conf_neg_normalizer=cfg.conf_neg_normalizer,
    )
    train_step, eval_step = _make_debug_functions(model, loss_fns, opt)

    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    opt_vars = [v.value for v in opt.variables]

    summary = {
        "samples": args.samples,
        "classes": classes,
        "anchors": anchors.tolist(),
        "config": {
            "epochs": args.epochs,
            "repeat": args.repeat,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "obj_gain": args.obj_gain,
            "obj_pos_gain": args.obj_pos_gain,
            "noobj_weight": args.noobj_weight,
            "ignore_iou": args.ignore_iou,
            "rescore_objectness": cfg.rescore_objectness,
            "hard_negative_min": args.hard_negative_min,
            "conf_neg_normalizer": args.conf_neg_normalizer,
            "head_proj_raw": args.head_proj_raw,
        },
        "positive_assignments": _positive_assignments(base_targets, classes),
        "epochs": [],
    }

    batch_x = jax.device_put(images[: args.batch_size])
    batch_y = {head: jax.device_put(value[: args.batch_size]) for head, value in targets.items()}
    eval_x = jax.device_put(base_images)
    eval_y = {head: jax.device_put(value[: len(base_images)]) for head, value in base_targets.items()}

    for epoch in range(1, args.epochs + 1):
        for start in range(0, len(images), args.batch_size):
            x = jax.device_put(images[start : start + args.batch_size])
            y = {head: jax.device_put(value[start : start + args.batch_size]) for head, value in targets.items()}
            trainable, non_trainable, opt_vars, *_ = train_step(trainable, non_trainable, opt_vars, x, y)

        eval_non_trainable, val_loss, head_losses = eval_step(trainable, non_trainable, eval_x, eval_y)
        _assign_model_state(model, trainable, eval_non_trainable)
        probe_out = probe.predict(base_images, verbose=0)
        report = {
            "epoch": epoch,
            "val_loss": float(np.array(_device_to_host(val_loss))),
            "head_losses": {head: float(np.array(_device_to_host(value))) for head, value in head_losses.items()},
            "heads": {},
        }
        for head in HEADS:
            pred = probe_out[f"{head}_pred"]
            feat = probe_out[f"{head}_feat"]
            decoded = _decode_for_loss(
                base_targets[head],
                pred,
                anchors[ANCHOR_MASKS[HEADS.index(head)]],
                model_cfg.input_size,
                STRIDES[HEADS.index(head)],
                len(classes),
                cfg,
            )
            report["heads"][head] = _head_report(
                head,
                base_targets[head],
                pred,
                feat,
                anchors[ANCHOR_MASKS[HEADS.index(head)]],
                len(classes),
                cfg,
                model_cfg.input_size,
            )
            report["heads"][head]["obj_feature_similarity"] = _feature_similarity(
                feat,
                decoded["obj"] > 0.5,
                decoded["neg_mask"] > 0.5,
                decoded["obj_prob"],
            )
        summary["epochs"].append(report)
        (run_dir / f"epoch_{epoch:02d}.json").write_text(json.dumps(report, indent=2))
        print(
            f"epoch {epoch}/{args.epochs} val_loss={report['val_loss']:.3f} "
            + " ".join(
                f"{head}:pos={report['heads'][head]['pos_count']} neg={report['heads'][head]['neg_count']} "
                f"obj_pos={report['heads'][head]['obj_prob_pos_mean']:.3f} obj_neg={report['heads'][head]['obj_prob_neg_mean']:.3f} "
                f"grad_neg={report['heads'][head]['obj_grad_neg_mean']:.6f} sim={report['heads'][head]['obj_feature_similarity'].get('hard_neg_hamming_sim_mean', 0.0):.3f}"
                for head in HEADS
            )
        )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
