from __future__ import annotations

import argparse
import json
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
from kerasbnn.debug_export_layer_parity import (
    _build_images,
    _collect_c_tensors,
    _manual_load_named_weights,
    _expected_xnor_binary_input_np,
)
from kerasbnn.xnor2_layers import XNORBinaryConv2D, XNORBinaryDense


REPO_ROOT = Path("/home/claude/source/yolovxnor")
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "tinier_xnor2_fruits_gpu1_r11_256_thermo4_binarys2_advaugramp_bs8_cycle512_warmr10_lr5e5"


def parse_args():
    ap = argparse.ArgumentParser(description="Trace Keras vs C layer-by-layer divergence for the best fruits XNOR2 model.")
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
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument(
        "--layers",
        nargs="*",
        default=[
            "binary_input",
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "conv5",
            "down5",
            "fire1_sq",
            "fire1_e1",
            "fire1_e3",
            "fire2_sq",
            "fire2_e1",
            "fire2_e3",
            "fire3_sq",
            "fire3_e1",
            "fire3_e3",
            "fire4_sq",
            "fire4_e1",
            "fire4_e3",
        ],
    )
    return ap.parse_args()


def _mad(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def _rel_mad(a, b, ref):
    return _mad(a, b) / (float(np.mean(np.abs(np.asarray(ref, dtype=np.float32)))) + 1e-8)


def _build_model(args, num_classes: int):
    return build_binary_tinier_yolov3_xnor2(
        num_classes,
        input_shape=(args.input_size, args.input_size, 3),
        cfg=TinierXNOR2Config(
            thermo_levels=args.thermo_levels,
            input_levels=args.input_levels,
            quantized_input=args.quantized_input,
            tinyv3_backbone=args.tinyv3_backbone,
        ),
    )


def _collect_keras_outputs(model, image: np.ndarray, layer_names: list[str]):
    layers = [model.get_layer(name) for name in layer_names]
    sub = keras.Model(model.input, [layer.output for layer in layers])
    outs = sub(image[None, ...], training=False)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    return {name: np.asarray(value)[0] for name, value in zip(layer_names, outs)}


def _standalone_eval(layer, source_value: np.ndarray):
    inp = keras.Input(shape=tuple(source_value.shape))
    out = layer(inp, training=False)
    tmp = keras.Model(inp, out)
    return np.asarray(tmp(source_value[None, ...], training=False))[0]


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    weights_path = run_dir / "model.voc.best_map.weights.h5"

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
        val_sample_count=0,
        test_sample_count=0,
        subset_train_as_val=args.subset_train_as_val,
    )
    _, val_samples, _, classes = _prepare_split_samples(split_args, args.input_size)
    image = _build_images([val_samples[args.image_index]], args.input_size, args.thermo_levels)[0]

    model = _build_model(args, len(classes))
    _manual_load_named_weights(model, weights_path)
    source_names = []
    for name in args.layers:
        layer = model.get_layer(name)
        if isinstance(layer, (XNORBinaryConv2D, XNORBinaryDense)):
            src_name = layer.input._keras_history[0].name
            if src_name not in args.layers:
                source_names.append(src_name)
    collect_names = []
    seen = set()
    for name in list(args.layers) + source_names:
        if name not in seen:
            seen.add(name)
            collect_names.append(name)

    keras_map = _collect_keras_outputs(model, image, collect_names)
    with tempfile.TemporaryDirectory(prefix="kerasbnn_layer_seq_") as td:
        c_tensors, _ = _collect_c_tensors(model, image, Path(td), "layer_seq")

    rows = []
    prev_diff = None
    for name in args.layers:
        layer = model.get_layer(name)
        k_out = keras_map[name]
        c_out = np.asarray(c_tensors[name], dtype=np.float32)
        row = {
            "layer": name,
            "type": layer.__class__.__name__,
            "output_mean_abs_diff": _mad(k_out, c_out),
            "output_relative_diff": _rel_mad(k_out, c_out, k_out),
        }

        if isinstance(layer, (XNORBinaryConv2D, XNORBinaryDense)):
            src_name = layer.input._keras_history[0].name
            k_src = keras_map[src_name]
            c_src = np.asarray(c_tensors[src_name], dtype=np.float32)
            expected_k = _expected_xnor_binary_input_np(layer, k_src)
            expected_c = _expected_xnor_binary_input_np(layer, c_src)
            standalone_k = _standalone_eval(layer, k_src)
            standalone_c = _standalone_eval(layer, c_src)
            row.update(
                {
                    "source": src_name,
                    "source_mean_abs_diff": _mad(k_src, c_src),
                    "source_relative_diff": _rel_mad(k_src, c_src, k_src),
                    "input_bit_disagree": float(np.mean(expected_k != expected_c)),
                    "same_input_keras_vs_standalone_mad": _mad(k_out, standalone_k),
                    "same_input_keras_vs_standalone_rel": _rel_mad(k_out, standalone_k, k_out),
                    "c_vs_standalone_on_c_input_mad": _mad(c_out, standalone_c),
                    "c_vs_standalone_on_c_input_rel": _rel_mad(c_out, standalone_c, c_out),
                    "standalone_output_from_c_input_vs_from_k_input_mad": _mad(standalone_c, standalone_k),
                    "standalone_output_from_c_input_vs_from_k_input_rel": _rel_mad(standalone_c, standalone_k, standalone_k),
                }
            )
        rows.append(row)

    print(json.dumps({"run_dir": str(run_dir), "image_index": args.image_index, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
