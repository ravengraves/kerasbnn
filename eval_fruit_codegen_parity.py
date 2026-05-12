from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")

import h5py
import numpy as np

sys.path.insert(0, "/home/claude/source/yolovxnor")

from kerasbnn.generated_runtime import predict_bundle_outputs
from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2
from binary_yolo.train_tinier import _prepare_split_samples
import binary_yolo.yad2k_core as yad2k_core
from kerasbnn.xnor2_layers import XNORBinaryConv2D, XNORBinaryDense


REPO_ROOT = Path("/home/claude/source/yolovxnor")
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "tinier_xnor2_fruits_gpu1_r7_256_thermo4_advaugramp_bs8_lr5e5_cycle512_warmr6"


def parse_args():
    ap = argparse.ArgumentParser(description="Compare Keras vs exported C runtime on the best fruits XNOR2 model.")
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
    ap.add_argument("--keep-export-dir", default="")
    return ap.parse_args()


def _load_anchors_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row.get("w", row.get("width"))
            h = row.get("h", row.get("height"))
            rows.append((float(w), float(h)))
    if not rows:
        raise ValueError(f"No anchors found in {path}")
    return np.asarray(rows, np.float32)


def _build_images(samples, input_size: int, thermo_levels: int):
    if not samples:
        return np.zeros((0, input_size, input_size, 3), np.float32)
    return np.stack(
        [yad2k_core._prepare_sample(sample, input_size, thermo_levels, train=False)[0] for sample in samples]
    ).astype(np.float32)


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


def _h5_arrays(vars_group: h5py.Group) -> list[np.ndarray]:
    arrays = []
    for key in sorted(vars_group.keys(), key=lambda k: int(k)):
        arrays.append(np.asarray(vars_group[key]))
    return arrays


def _manual_load_named_weights(model, weights_path: Path) -> None:
    with h5py.File(weights_path, "r") as f:
        layers_group = f["layers"]
        for _, group in layers_group.items():
            if not isinstance(group, h5py.Group):
                continue
            vars_group = group.get("vars")
            if vars_group is None:
                continue
            layer_name = vars_group.attrs.get("name")
            if isinstance(layer_name, bytes):
                layer_name = layer_name.decode("utf-8")
            if not layer_name:
                continue
            arrays = _h5_arrays(vars_group)
            try:
                layer = model.get_layer(layer_name)
            except Exception:
                layer = None
            if layer is not None:
                if isinstance(layer, XNORBinaryConv2D):
                    idx = 0
                    layer.kernel.assign(arrays[idx]); idx += 1
                    if layer.bias is not None:
                        layer.bias.assign(arrays[idx]); idx += 1
                    if layer.use_rprelu:
                        layer.rprelu_gamma.assign(arrays[idx]); idx += 1
                        layer.rprelu_beta.assign(arrays[idx]); idx += 1
                        layer.rprelu_zeta.assign(arrays[idx]); idx += 1
                elif isinstance(layer, XNORBinaryDense):
                    idx = 0
                    layer.kernel.assign(arrays[idx]); idx += 1
                    if layer.bias is not None:
                        layer.bias.assign(arrays[idx]); idx += 1
                    if layer.use_rprelu:
                        layer.rprelu_gamma.assign(arrays[idx]); idx += 1
                        layer.rprelu_beta.assign(arrays[idx]); idx += 1
                        layer.rprelu_zeta.assign(arrays[idx]); idx += 1
                elif arrays:
                    layer.set_weights(arrays)

            affine_group = group.get("affine_norm")
            if affine_group is not None and layer is not None and getattr(layer, "affine_norm", None) is not None:
                affine_vars = affine_group.get("vars")
                if affine_vars is not None:
                    layer.affine_norm.set_weights(_h5_arrays(affine_vars))


def _predict_c_runtime(
    model,
    images: np.ndarray,
    export_dir: Path,
    model_name: str,
) -> tuple[dict[str, np.ndarray], dict]:
    return predict_bundle_outputs(
        model,
        images=images,
        output_dir=export_dir,
        model_name=model_name,
        repo_root=REPO_ROOT,
    )


def _map50(samples, preds: dict[str, np.ndarray], anchors: np.ndarray, args, num_classes: int) -> float:
    return float(
        yad2k_core.map50(
            samples,
            preds,
            anchors,
            args.input_size,
            num_classes,
            score=args.score_threshold,
            objectness=args.objectness_threshold,
            bbox_output_scale=args.bbox_output_scale,
            obj_output_scale=args.obj_output_scale,
            cls_output_scale=args.cls_output_scale,
            logit_scale=args.logit_scale,
            class_mode=args.class_mode,
            box_mode=args.box_mode,
            heads=("pred_26", "pred_13"),
            anchor_masks=([0, 1, 2], [3, 4, 5]),
        )
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

    keras_preds = _predict_named(model, val_images, args.batch_size, training=False)
    keras_map50 = _map50(val_samples, keras_preds, anchors, args, len(classes))

    if args.keep_export_dir:
        export_dir = Path(args.keep_export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        c_preds, meta = _predict_c_runtime(
            model,
            val_images,
            export_dir,
            model_name="fruits_xnor2_export",
        )
    else:
        with tempfile.TemporaryDirectory(prefix="kerasbnn_fruit_export_") as td:
            c_preds, meta = _predict_c_runtime(
                model,
                val_images,
                Path(td),
                model_name="fruits_xnor2_export",
            )

    c_map50 = _map50(val_samples, c_preds, anchors, args, len(classes))

    pred_13_diff = np.abs(c_preds["pred_13"] - np.asarray(keras_preds["pred_13"]))
    pred_26_diff = np.abs(c_preds["pred_26"] - np.asarray(keras_preds["pred_26"]))

    result = {
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "classes": len(classes),
        },
        "keras_map50": keras_map50,
        "c_map50": c_map50,
        "map50_delta": c_map50 - keras_map50,
        "pred_13_max_abs_diff": float(np.max(pred_13_diff)),
        "pred_13_mean_abs_diff": float(np.mean(pred_13_diff)),
        "pred_26_max_abs_diff": float(np.max(pred_26_diff)),
        "pred_26_mean_abs_diff": float(np.mean(pred_26_diff)),
        "export_outputs": [meta["tensors"][idx]["name"] for idx in meta["outputs"]],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
