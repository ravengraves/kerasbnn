from __future__ import annotations

import argparse
import json
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

import keras
import numpy as np

import binary_layers  # noqa: F401
from binary_yolo.model import ModelConfig
from binary_yolo.yad2k_core import ANCHOR_MASKS, HEADS, MAX_TRUE_BOXES, STRIDES, build_targets, find_sample, fit_anchors, load_dataset, map50, predict_boxes, split_samples


def _cfg_from_summary(summary: dict) -> tuple[ModelConfig, dict]:
    return ModelConfig(**summary["model"]), summary["train"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _int8_sat(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.asarray(x, np.float16) / np.float16(scale), np.float16(-127.0), np.float16(127.0)).astype(np.float32)


def _u01(x: np.ndarray, scale: float) -> np.ndarray:
    return (_int8_sat(x, scale) + 127.0) / 254.0


def _ratio_from_u(u: np.ndarray) -> np.ndarray:
    return np.where(u <= 0.5, 2.0 * u, 1.0 + 4.0 * (u - 0.5)).astype(np.float32)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    tl = np.maximum(a[:2], b[:2])
    br = np.minimum(a[2:], b[2:])
    inter = np.prod(np.maximum(br - tl, 0.0))
    area_a = np.prod(np.maximum(a[2:] - a[:2], 0.0))
    area_b = np.prod(np.maximum(b[2:] - b[:2], 0.0))
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    return np.array([box[0] - box[2] * 0.5, box[1] - box[3] * 0.5, box[0] + box[2] * 0.5, box[1] + box[3] * 0.5], np.float32)


def _decode_entry(raw: np.ndarray, grid_y: int, grid_x: int, head: str, anchor_id: int, anchors: np.ndarray, input_size: int, output_scale: float, logit_scale: float):
    grid_h = input_size // STRIDES[HEADS.index(head)]
    grid_w = grid_h
    xy = (_u01(raw[:2], output_scale) + np.asarray([grid_x, grid_y], np.float32)) / np.asarray([grid_w, grid_h], np.float32)
    ratio = _ratio_from_u(_u01(raw[2:4], output_scale))
    wh = ratio * anchors[anchor_id] / float(input_size)
    obj_logits = _int8_sat(raw[4:5], output_scale) / float(logit_scale)
    cls_logits = _int8_sat(raw[5:], output_scale) / float(logit_scale)
    obj_prob = float(_sigmoid(obj_logits)[0])
    cls_prob = _sigmoid(cls_logits)
    cls_id = int(np.argmax(cls_prob))
    cls_score = float(cls_prob[cls_id])
    return {
        "xy": xy.tolist(),
        "wh": wh.tolist(),
        "box_xyxy": _cxcywh_to_xyxy(np.concatenate([xy, wh])).tolist(),
        "obj_logits": obj_logits.tolist(),
        "obj_prob": obj_prob,
        "cls_logits": cls_logits.tolist(),
        "cls_prob": cls_prob.tolist(),
        "cls_id": cls_id,
        "cls_score": cls_score,
        "score": obj_prob * cls_score,
    }


def _target_locations(targets: dict[str, np.ndarray]):
    rows = []
    for head, mask in zip(HEADS, ANCHOR_MASKS):
        num_classes = targets[head].shape[-1] - 5 - 5 * MAX_TRUE_BOXES
        tgt = targets[head][0, ..., : 5 + num_classes]
        ys, xs, anchors_local = np.where(tgt[..., 4] > 0.5)
        for y, x, a_local in zip(ys, xs, anchors_local):
            anchor_id = mask[a_local]
            row = tgt[y, x, a_local]
            rows.append(
                {
                    "head": head,
                    "grid_y": int(y),
                    "grid_x": int(x),
                    "anchor_local": int(a_local),
                    "anchor_id": int(anchor_id),
                    "target": row.tolist(),
                    "target_box_xyxy": _cxcywh_to_xyxy(row[:4]).tolist(),
                    "target_class": int(np.argmax(row[5:])),
                }
            )
    return rows


def _top_candidates(preds: dict[str, np.ndarray], anchors: np.ndarray, input_size: int, output_scale: float, logit_scale: float, topk: int):
    rows = []
    for head, mask in zip(HEADS, ANCHOR_MASKS):
        pred = preds[head][0]
        for y in range(pred.shape[0]):
            for x in range(pred.shape[1]):
                for a_local, anchor_id in enumerate(mask):
                    raw = pred[y, x, a_local]
                    decoded = _decode_entry(raw, y, x, head, anchor_id, anchors, input_size, output_scale, logit_scale)
                    rows.append(
                        {
                            "head": head,
                            "grid_y": y,
                            "grid_x": x,
                            "anchor_local": a_local,
                            "anchor_id": anchor_id,
                            "raw": raw.astype(np.float32).tolist(),
                            **decoded,
                        }
                    )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:topk], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="fruits_detection")
    ap.add_argument("--sample", default="")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    model_cfg, train_cfg = _cfg_from_summary(summary)
    sample_name = args.sample or train_cfg["overfit_sample"]
    output_scale = float(train_cfg["output_scale"])
    logit_scale = float(train_cfg["logit_scale"])

    samples, classes = load_dataset(args.dataset)
    anchors = fit_anchors(split_samples(samples, 0.1, 7)[0], model_cfg.input_size, seed=7)
    sample = find_sample(samples, sample_name)
    images, targets = build_targets([sample], len(classes), anchors, model_cfg.input_size, model_cfg.thermo_levels, augment=False)
    model = keras.models.load_model(run_dir / "overfit.keras", compile=False)
    preds = model.predict(images, verbose=0)

    target_rows = _target_locations(targets)
    topk_rows, all_rows = _top_candidates(preds, anchors, model_cfg.input_size, output_scale, logit_scale, args.topk)
    gt_boxes = [_cxcywh_to_xyxy(np.asarray(row["target"][:4], np.float32)) for row in target_rows]
    gt_classes = [row["target_class"] for row in target_rows]

    for row in target_rows:
        head_pred = preds[row["head"]][0, row["grid_y"], row["grid_x"], row["anchor_local"]]
        row["pred"] = _decode_entry(head_pred, row["grid_y"], row["grid_x"], row["head"], row["anchor_id"], anchors, model_cfg.input_size, output_scale, logit_scale)
        row["iou_to_target"] = _iou_xyxy(np.asarray(row["pred"]["box_xyxy"], np.float32), np.asarray(row["target_box_xyxy"], np.float32))

    for row in topk_rows:
        if gt_boxes:
            ious = [_iou_xyxy(np.asarray(row["box_xyxy"], np.float32), gt) for gt in gt_boxes]
            row["best_iou"] = float(max(ious))
            row["best_gt"] = int(np.argmax(ious))
            row["best_gt_class"] = gt_classes[row["best_gt"]]
        else:
            row["best_iou"] = 0.0
            row["best_gt"] = -1
            row["best_gt_class"] = -1

    best_iou_any = []
    for gt_idx, gt in enumerate(gt_boxes):
        best = max(all_rows, key=lambda row: _iou_xyxy(np.asarray(row["box_xyxy"], np.float32), gt))
        best_iou_any.append(
            {
                "gt_index": gt_idx,
                "gt_class": gt_classes[gt_idx],
                "gt_box": gt.tolist(),
                "best_iou": _iou_xyxy(np.asarray(best["box_xyxy"], np.float32), gt),
                "best_score": best["score"],
                "best_obj_prob": best["obj_prob"],
                "best_cls_id": best["cls_id"],
                "best_cls_score": best["cls_score"],
                "head": best["head"],
                "grid_y": best["grid_y"],
                "grid_x": best["grid_x"],
                "anchor_id": best["anchor_id"],
                "box_xyxy": best["box_xyxy"],
            }
        )

    good_rows = []
    for row in all_rows:
        if not gt_boxes:
            continue
        ious = [_iou_xyxy(np.asarray(row["box_xyxy"], np.float32), gt) for gt in gt_boxes]
        best_gt = int(np.argmax(ious))
        best_iou = float(max(ious))
        if best_iou < 0.5:
            continue
        good_rows.append(
            {
                "head": row["head"],
                "grid_y": row["grid_y"],
                "grid_x": row["grid_x"],
                "anchor_id": row["anchor_id"],
                "score": row["score"],
                "obj_prob": row["obj_prob"],
                "cls_id": row["cls_id"],
                "cls_score": row["cls_score"],
                "best_iou": best_iou,
                "best_gt": best_gt,
                "best_gt_class": gt_classes[best_gt],
                "class_match": bool(row["cls_id"] == gt_classes[best_gt]),
                "box_xyxy": row["box_xyxy"],
            }
        )
    good_rows.sort(key=lambda row: row["score"], reverse=True)

    decoded = predict_boxes(preds, anchors, model_cfg.input_size, score=0.01, output_scale=output_scale, logit_scale=logit_scale)
    report = {
        "run_dir": str(run_dir),
        "sample": sample_name,
        "classes": classes,
        "anchors": anchors.tolist(),
        "train_cfg": train_cfg,
        "map50_score001": map50([sample], preds, anchors, model_cfg.input_size, len(classes), score=0.01, output_scale=output_scale, logit_scale=logit_scale),
        "target_rows": target_rows,
        "topk_rows": topk_rows,
        "best_iou_any": best_iou_any,
        "good_rows": good_rows[:100],
        "decoded_prediction_count_score001": int(len(decoded[0])),
        "decoded_top_scores_score001": decoded[1][:10].tolist() if len(decoded[1]) else [],
        "raw_stats": {
            head: {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "bbox_mean_abs": float(np.mean(np.abs(arr[..., :4]))),
                "obj_mean_abs": float(np.mean(np.abs(arr[..., 4:5]))),
                "cls_mean_abs": float(np.mean(np.abs(arr[..., 5:]))),
            }
            for head, arr in preds.items()
        },
    }
    out_path = run_dir / "debug_overfit.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
