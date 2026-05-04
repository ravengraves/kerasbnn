from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")

sys.path.insert(0, "/home/claude/source/yolovxnor")

from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2
from binary_yolo.train_tinier import _prepare_split_samples
from kerasbnn.eval_fruit_codegen_parity import (
    DEFAULT_RUN_DIR,
    _build_images,
    _load_anchors_csv,
    _manual_load_named_weights,
    _map50,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Compare Keras train-mode and eval-mode outputs on the best fruits XNOR2 model.")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--dataset", default="fruits_detection")
    ap.add_argument("--dataset-format", default="auto")
    ap.add_argument("--dataset-split", default="auto")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--input-levels", type=int, default=16)
    ap.add_argument("--quantized-input", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tinyv3-backbone", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.0)
    ap.add_argument("--train-sample-fraction", type=float, default=1.0)
    ap.add_argument("--val-sample-fraction", type=float, default=1.0)
    ap.add_argument("--test-sample-fraction", type=float, default=1.0)
    ap.add_argument("--subset-train-as-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--score-threshold", type=float, default=0.001)
    ap.add_argument("--objectness-threshold", type=float, default=0.001)
    ap.add_argument("--bbox-output-scale", type=float, default=2.0)
    ap.add_argument("--obj-output-scale", type=float, default=64.0)
    ap.add_argument("--cls-output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--class-mode", default="softmax", choices=("softmax", "bce"))
    ap.add_argument("--box-mode", default="linear", choices=("linear", "yolo"))
    ap.add_argument("--max-val-samples", type=int, default=0)
    return ap.parse_args()


def _predict_named(model, images: np.ndarray, batch_size: int, training: bool) -> dict[str, np.ndarray]:
    names = list(model.output_names)
    accum = {name: [] for name in names}
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        outputs = model(batch, training=training)
        if isinstance(outputs, dict):
            for name in names:
                accum[name].append(np.asarray(outputs[name]))
        elif isinstance(outputs, (list, tuple)):
            for name, value in zip(names, outputs):
                accum[name].append(np.asarray(value))
        else:
            accum[names[0]].append(np.asarray(outputs))
    return {name: np.concatenate(values, axis=0) for name, values in accum.items()}


def _build_model(num_classes: int, input_size: int, thermo_levels: int, input_levels: int, quantized_input: bool, tinyv3_backbone: bool):
    return build_binary_tinier_yolov3_xnor2(
        num_classes,
        input_shape=(input_size, input_size, 3),
        cfg=TinierXNOR2Config(
            thermo_levels=thermo_levels,
            input_levels=input_levels,
            quantized_input=quantized_input,
            tinyv3_backbone=tinyv3_backbone,
        ),
    )


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    weights_path = run_dir / "model.voc.best_map.weights.h5"
    anchors_path = run_dir / "bbox_priors.csv"
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if not anchors_path.exists():
        raise FileNotFoundError(anchors_path)

    split_args = argparse.Namespace(
        dataset=args.dataset,
        dataset_format=args.dataset_format,
        dataset_split=args.dataset_split,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        include_difficult=args.include_difficult,
        train_sample_fraction=args.train_sample_fraction,
        val_sample_fraction=args.val_sample_fraction,
        test_sample_fraction=args.test_sample_fraction,
        train_sample_count=0,
        subset_train_as_val=args.subset_train_as_val,
    )
    train_samples, val_samples, _, classes = _prepare_split_samples(split_args, args.input_size)
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]
    val_images = _build_images(val_samples, args.input_size, args.thermo_levels)
    anchors = _load_anchors_csv(anchors_path)

    eval_model = _build_model(
        len(classes),
        args.input_size,
        args.thermo_levels,
        args.input_levels,
        args.quantized_input,
        args.tinyv3_backbone,
    )
    train_model = _build_model(
        len(classes),
        args.input_size,
        args.thermo_levels,
        args.input_levels,
        args.quantized_input,
        args.tinyv3_backbone,
    )
    _manual_load_named_weights(eval_model, weights_path)
    _manual_load_named_weights(train_model, weights_path)

    eval_preds = _predict_named(eval_model, val_images, args.batch_size, training=False)
    train_preds = _predict_named(train_model, val_images, args.batch_size, training=True)

    eval_map50 = _map50(val_samples, eval_preds, anchors, args, len(classes))
    train_map50 = _map50(val_samples, train_preds, anchors, args, len(classes))

    pred_13_diff = np.abs(np.asarray(train_preds["pred_13"]) - np.asarray(eval_preds["pred_13"]))
    pred_26_diff = np.abs(np.asarray(train_preds["pred_26"]) - np.asarray(eval_preds["pred_26"]))

    result = {
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "classes": len(classes),
        },
        "eval_map50": eval_map50,
        "train_map50": train_map50,
        "map50_delta": train_map50 - eval_map50,
        "pred_13_max_abs_diff": float(np.max(pred_13_diff)),
        "pred_13_mean_abs_diff": float(np.mean(pred_13_diff)),
        "pred_26_max_abs_diff": float(np.max(pred_26_diff)),
        "pred_26_mean_abs_diff": float(np.mean(pred_26_diff)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
