from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")

import keras
import numpy as np

sys.path.insert(0, "/home/claude/source/yolovxnor")

from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2
from binary_yolo.train_tinier import _prepare_split_samples
from kerasbnn.benchmark_fruit_codegen import _map50, _predict_named
from kerasbnn.debug_export_layer_parity import (
    DEFAULT_RUN_DIR,
    _build_images,
    _collect_c_tensors,
    _load_anchors_csv,
    _manual_load_named_weights,
)
from kerasbnn.eval_fruit_codegen_parity import _predict_c_runtime
from kerasbnn.runtime_quant import fold_xnor_activation_compare_np
from kerasbnn.xnor2_layers import XNORBinaryConv2D


def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate exported XNOR activation compare tensors using train images.")
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
    ap.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--train-sample-fraction", type=float, default=1.0)
    ap.add_argument("--val-sample-fraction", type=float, default=1.0)
    ap.add_argument("--test-sample-fraction", type=float, default=1.0)
    ap.add_argument("--subset-train-as-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--calib-train-images", type=int, default=16)
    ap.add_argument("--eval-val-images", type=int, default=0)
    return ap.parse_args()


def _target_bits_from_keras(layer, x: np.ndarray) -> np.ndarray:
    hist = getattr(layer.input, "_keras_history", None)
    src_is_thermo = False
    if hist is not None:
        src_is_thermo = (hist[0].__class__.__name__ == "ThermometerBinaryInput")
    if src_is_thermo:
        return np.where(np.asarray(x, dtype=np.float32) >= 0.0, 1, -1).astype(np.int8)
    scale_f, bias_f = fold_xnor_activation_compare_np(layer)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4:
        scale = scale_f.reshape(1, 1, 1, -1)
        bias = bias_f.reshape(1, 1, 1, -1)
    elif x.ndim == 2:
        scale = scale_f.reshape(1, -1)
        bias = bias_f.reshape(1, -1)
    else:
        scale = scale_f
        bias = bias_f
    return np.where(scale * x + bias >= 0.0, 1, -1).astype(np.int8)


def _best_threshold_bits(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    pos_prefix = np.cumsum(ys > 0)
    neg_prefix = np.cumsum(ys < 0)
    total_pos = int(pos_prefix[-1]) if pos_prefix.size else 0
    total_neg = int(neg_prefix[-1]) if neg_prefix.size else 0
    best_err = x.size + 1
    best_thr = 0.0
    for i in range(xs.size + 1):
        pos_left = int(pos_prefix[i - 1]) if i > 0 else 0
        neg_left = int(neg_prefix[i - 1]) if i > 0 else 0
        err = pos_left + (total_neg - neg_left)
        if err < best_err:
            best_err = err
            if i == 0:
                best_thr = float(xs[0] - 1.0)
            elif i == xs.size:
                best_thr = float(xs[-1] + 1.0)
            else:
                best_thr = float(0.5 * (xs[i - 1] + xs[i]))
    return best_thr


def _fit_compare_pair_channel(c_input: np.ndarray, target_bits: np.ndarray, base_sign: float) -> tuple[int, int]:
    x = np.asarray(c_input, dtype=np.float32).reshape(-1)
    y = np.asarray(target_bits, dtype=np.int8).reshape(-1)
    pos = int(np.sum(y > 0))
    neg = int(np.sum(y < 0))
    if pos == 0:
        return 0, -1
    if neg == 0:
        return 0, 1
    sign = 1 if base_sign >= 0.0 else -1
    z = sign * x
    thr = _best_threshold_bits(z, y)
    best = None
    for r in range(1, 128):
        b = int(np.clip(np.rint(-r * thr), -127, 127))
        pred = np.where(r * z + b >= 0.0, 1, -1).astype(np.int8)
        err = int(np.sum(pred != y))
        margin = float(np.mean(np.abs(r * z + b)))
        key = (err, -margin, abs(b))
        if best is None or key < best[0]:
            best = (key, sign * r, b)
    assert best is not None
    return int(best[1]), int(best[2])


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    weights_path = run_dir / "model.voc.best_map.weights.h5"
    anchors = _load_anchors_csv(run_dir / "bbox_priors.csv")

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
    calib_train = train_samples[: args.calib_train_images]
    eval_val = val_samples if args.eval_val_images <= 0 else val_samples[: args.eval_val_images]
    train_images = _build_images(calib_train, args.input_size, args.thermo_levels)
    val_images = _build_images(eval_val, args.input_size, args.thermo_levels)

    model = build_binary_tinier_yolov3_xnor2(
        len(classes),
        input_shape=(args.input_size, args.input_size, 3),
        cfg=TinierXNOR2Config(
            thermo_levels=args.thermo_levels,
            input_levels=args.input_levels,
            quantized_input=args.quantized_input,
            tinyv3_backbone=args.tinyv3_backbone,
        ),
    )
    _manual_load_named_weights(model, weights_path)

    xnor_layers = [layer for layer in model.layers if isinstance(layer, XNORBinaryConv2D)]
    input_sources = []
    seen = set()
    for layer in xnor_layers:
        src = layer.input._keras_history[0]
        if src.name not in seen:
            input_sources.append(src.output)
            seen.add(src.name)
    src_names = [t._keras_history[0].name for t in input_sources]
    submodel = keras.Model(model.input, input_sources)
    keras_inputs = submodel(train_images, training=False)
    if not isinstance(keras_inputs, (list, tuple)):
        keras_inputs = [keras_inputs]
    keras_map = {name: np.asarray(val) for name, val in zip(src_names, keras_inputs)}

    c_inputs_accum: dict[str, list[np.ndarray]] = {name: [] for name in src_names}
    with tempfile.TemporaryDirectory(prefix="kerasbnn_compare_calib_") as td:
        for i, image in enumerate(train_images):
            c_tensors, _ = _collect_c_tensors(model, image, Path(td), model_name=f"cmpcal_{i}")
            for name in src_names:
                c_inputs_accum[name].append(np.asarray(c_tensors[name], dtype=np.float32))
    c_input_map = {name: np.stack(vals, axis=0) for name, vals in c_inputs_accum.items()}

    overrides: dict[str, np.ndarray] = {}
    stats = []
    for layer in xnor_layers:
        src_name = layer.input._keras_history[0].name
        k_in = keras_map[src_name]
        c_in = c_input_map[src_name]
        target = _target_bits_from_keras(layer, k_in)
        base_scale, _ = fold_xnor_activation_compare_np(layer)
        channels = k_in.shape[-1]
        act_scale = np.zeros((channels,), dtype=np.int8)
        act_bias = np.zeros((channels,), dtype=np.int8)
        total_err = 0
        total_count = 0
        for c in range(channels):
            s, b = _fit_compare_pair_channel(c_in[..., c], target[..., c], float(base_scale[c]))
            act_scale[c] = np.int8(s)
            act_bias[c] = np.int8(b)
            pred = np.where(s * c_in[..., c] + b >= 0.0, 1, -1).astype(np.int8)
            total_err += int(np.sum(pred != target[..., c]))
            total_count += int(pred.size)
        overrides[f"cmpcal_export_{layer.name}_act_scale_i8"] = act_scale
        overrides[f"cmpcal_export_{layer.name}_act_bias_i8"] = act_bias
        stats.append({"layer": layer.name, "disagree_rate": total_err / max(total_count, 1), "channels": channels})

    keras_preds = _predict_named(model, val_images, batch_size=8, training=False)
    keras_map50 = _map50(
        eval_val,
        keras_preds,
        anchors,
        argparse.Namespace(
            input_size=args.input_size,
            score_threshold=0.001,
            objectness_threshold=0.001,
            bbox_output_scale=2.0,
            obj_output_scale=64.0,
            cls_output_scale=32.0,
            logit_scale=32.0,
            class_mode="softmax",
            box_mode="linear",
        ),
        len(classes),
    )
    with tempfile.TemporaryDirectory(prefix="kerasbnn_compare_calib_eval_") as td:
        c_preds, _ = _predict_c_runtime(model, val_images, Path(td), model_name="cmpcal_export", array_overrides=overrides)
    c_map50 = _map50(
        eval_val,
        c_preds,
        anchors,
        argparse.Namespace(
            input_size=args.input_size,
            score_threshold=0.001,
            objectness_threshold=0.001,
            bbox_output_scale=2.0,
            obj_output_scale=64.0,
            cls_output_scale=32.0,
            logit_scale=32.0,
            class_mode="softmax",
            box_mode="linear",
        ),
        len(classes),
    )
    stats.sort(key=lambda r: r["disagree_rate"], reverse=True)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "calib_train_images": len(calib_train),
                "eval_val_images": len(eval_val),
                "keras_map50": float(keras_map50),
                "c_map50": float(c_map50),
                "map50_delta": float(c_map50 - keras_map50),
                "top_layer_disagree_after_fit": stats[:15],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
