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

import numpy as np

sys.path.insert(0, "/home/claude/source/yolovxnor")

from kerasbnn.debug_export_layer_parity import (
    DEFAULT_RUN_DIR,
    _build_images,
    _collect_c_tensors,
    _h5_arrays,
    _load_anchors_csv,
    _manual_load_named_weights,
)
from kerasbnn.benchmark_fruit_codegen import _map50, _predict_named
from kerasbnn.export_c import FX_ONE
from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2
from binary_yolo.train_tinier import _prepare_split_samples
from kerasbnn.xnor2_layers import XNORBinaryConv2D


def parse_args():
    ap = argparse.ArgumentParser(description="Fit per-channel affine corrections for exported XNOR alpha/bias arrays.")
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
    ap.add_argument("--calib-images", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=1)
    return ap.parse_args()


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x, y: [N, C]
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_center = x - x_mean
    y_center = y - y_mean
    denom = np.mean(x_center * x_center, axis=0)
    numer = np.mean(x_center * y_center, axis=0)
    a = np.ones_like(x_mean, dtype=np.float32)
    mask = denom > 1e-8
    a[mask] = numer[mask] / denom[mask]
    b = y_mean - a * x_mean
    return a.astype(np.float32), b.astype(np.float32)


def _to_fx(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * FX_ONE).astype(np.int32)


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
    val_samples = val_samples[: args.calib_images]
    images = _build_images(val_samples, args.input_size, args.thermo_levels)

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

    keras_layers = [layer for layer in model.layers if isinstance(layer, XNORBinaryConv2D)]
    submodel = None
    if keras_layers:
        import keras  # local import to keep env init order simple
        submodel = keras.Model(model.input, [layer.output for layer in keras_layers])
    layer_names = [layer.name for layer in keras_layers]
    keras_outs = submodel(images, training=False)
    if not isinstance(keras_outs, (list, tuple)):
        keras_outs = [keras_outs]
    keras_map = {name: np.asarray(value) for name, value in zip(layer_names, keras_outs)}

    array_overrides: dict[str, np.ndarray] = {}
    model_name = "calib_model"
    for _ in range(args.iterations):
        c_accum: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
        with tempfile.TemporaryDirectory(prefix="kerasbnn_xnor_calib_") as td:
            for i, image in enumerate(images):
                c_tensors, _ = _collect_c_tensors(model, image, Path(td), model_name=f"{model_name}_{i}")
                for name in layer_names:
                    c_accum[name].append(np.asarray(c_tensors[name], dtype=np.float32))
        new_overrides = {}
        for layer in keras_layers:
            name = layer.name
            y_c = np.stack(c_accum[name], axis=0)
            y_k = np.asarray(keras_map[name], dtype=np.float32)
            c_flat = y_c.reshape(-1, y_c.shape[-1])
            k_flat = y_k.reshape(-1, y_k.shape[-1])
            a, b = _fit_affine(c_flat, k_flat)
            kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
            alpha0 = np.mean(np.abs(kernel), axis=(0, 1, 2)).astype(np.float32)
            if layer.bias is None:
                bias0 = np.zeros_like(alpha0, dtype=np.float32)
            else:
                bias0 = np.asarray(layer.bias.numpy(), dtype=np.float32)
            alpha_new = a * alpha0
            bias_new = a * bias0 + b
            new_overrides[f"{model_name}_{name}_alpha_q"] = _to_fx(alpha_new)
            new_overrides[f"{model_name}_{name}_bias_q"] = _to_fx(bias_new)
        array_overrides = new_overrides

    # Final eval with calibrated arrays.
    with tempfile.TemporaryDirectory(prefix="kerasbnn_xnor_calib_eval_") as td:
        # reuse benchmark helper pattern without importing large script
        from kerasbnn.eval_fruit_codegen_parity import _predict_c_runtime
        keras_preds = _predict_named(model, images, batch_size=8, training=False)
        keras_map50 = _map50(
            val_samples,
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
        # monkeypatch export through direct call
        from kerasbnn import eval_fruit_codegen_parity as parity_mod
        orig = parity_mod.export_model_to_c
        try:
            from kerasbnn.export_c import export_model_to_c as export_with_overrides

            def _wrapped_export(model_, output_dir, model_name="kerasbnn_model"):
                return export_with_overrides(model_, output_dir, model_name=model_name, array_overrides=array_overrides)

            parity_mod.export_model_to_c = _wrapped_export
            c_preds, _ = _predict_c_runtime(model, images, Path(td), model_name=model_name)
        finally:
            parity_mod.export_model_to_c = orig
        c_map50 = _map50(
            val_samples,
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

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "calib_images": len(images),
                "keras_map50": float(keras_map50),
                "c_map50": float(c_map50),
                "map50_delta": float(c_map50 - keras_map50),
                "override_count": len(array_overrides),
                "sample_override_names": sorted(array_overrides.keys())[:10],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
