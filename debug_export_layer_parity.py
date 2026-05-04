from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")

import keras
import h5py
import numpy as np

sys.path.insert(0, "/home/claude/source/yolovxnor")

from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2
from binary_yolo.train_tinier import _prepare_split_samples
import binary_yolo.yad2k_core as yad2k_core
from kerasbnn.export_c import FX_ONE, export_model_to_c
from kerasbnn.runtime_quant import fold_xnor_activation_compare_np
from kerasbnn.xnor2_layers import XNORBinaryConv2D, XNORBinaryDense


REPO_ROOT = Path("/home/claude/source/yolovxnor")
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "tinier_xnor2_fruits_gpu1_r11_256_thermo4_binarys2_advaugramp_bs8_cycle512_warmr10_lr5e5"


def parse_args():
    ap = argparse.ArgumentParser(description="Compare exported C runtime intermediate tensors against Keras layer outputs.")
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
    ap.add_argument("--max-images", type=int, default=4)
    ap.add_argument("--top", type=int, default=30)
    return ap.parse_args()


def _load_anchors_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row.get("w", row.get("width"))), float(row.get("h", row.get("height")))))
    return np.asarray(rows, np.float32)


def _build_images(samples, input_size: int, thermo_levels: int):
    if not samples:
        return np.zeros((0, input_size, input_size, 3), np.float32)
    return np.stack(
        [yad2k_core._prepare_sample(sample, input_size, thermo_levels, train=False)[0] for sample in samples]
    ).astype(np.float32)


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


def _compile_debug_export(output_dir: Path, model_name: str, arena_bytes: int) -> Path:
    wrapper = output_dir / f"{model_name}_debug_wrapper.c"
    wrapper.write_text(
        textwrap.dedent(
            f"""
            #include <string.h>
            #include "kerasbnn.h"
            #include "{model_name}_export.h"

            int run_model_debug(const float *input, int tensor_id, void *out, size_t out_bytes) {{
                static unsigned char arena[{arena_bytes}];
                kbnn_context ctx;
                int rc = kbnn_context_init(&ctx, &{model_name}, arena, sizeof(arena));
                if (rc != 0) return rc;
                kbnn_context_set_input_f32(&ctx, 0, input);
                rc = kbnn_model_run(&ctx);
                if (rc == 0) {{
                    const kbnn_tensor_desc *desc = &{model_name}.tensors[tensor_id];
                    if (out != NULL && out_bytes >= desc->bytes && desc->bytes > 0) {{
                        memcpy(out, kbnn_context_tensor_ptr(&ctx, tensor_id), desc->bytes);
                    }}
                }}
                kbnn_context_release(&ctx);
                return rc;
            }}
            """
        ).strip()
        + "\n"
    )
    lib_path = output_dir / f"lib{model_name}_debug.so"
    cmd = [
        "clang",
        "-shared",
        "-fPIC",
        "-Ofast",
        "-march=native",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-ffast-math",
        "-fno-math-errno",
        "-funroll-loops",
        "-fomit-frame-pointer",
        f"-I{REPO_ROOT}",
        f"-I{REPO_ROOT / 'kerasbnn'}",
        str(wrapper),
        str(REPO_ROOT / "kerasbnn" / "kerasbnn.c"),
        "-lm",
        "-o",
        str(lib_path),
    ]
    subprocess.run(cmd, check=True)
    return lib_path


def _decode_packed_binary(buf: np.ndarray, shape: tuple[int, ...], words_per_point: int) -> np.ndarray:
    spatial = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
    channels = int(shape[-1])
    flat = np.empty((spatial, channels), dtype=np.float32)
    words = buf.reshape(spatial, words_per_point)
    for s in range(spatial):
        for c in range(channels):
            bit = (int(words[s, c // 64]) >> (c & 63)) & 1
            flat[s, c] = 1.0 if bit else -1.0
    return flat.reshape(shape)


def _expected_xnor_binary_input_np(layer, x: np.ndarray) -> np.ndarray:
    hist = getattr(layer.input, "_keras_history", None)
    src_is_thermo = False
    if hist is not None:
        src_is_thermo = (hist[0].__class__.__name__ == "ThermometerBinaryInput")
    if src_is_thermo:
        return np.where(np.asarray(x, dtype=np.float32) >= 0.0, 1.0, -1.0).astype(np.float32)
    scale_i8, bias_i8 = fold_xnor_activation_compare_np(layer)
    x = np.asarray(x, dtype=np.float32)
    x_q = np.rint(x * FX_ONE) / FX_ONE
    if x_q.ndim == 4:
        scale = scale_i8.reshape(1, 1, 1, -1).astype(np.float32)
        bias = bias_i8.reshape(1, 1, 1, -1).astype(np.float32)
    elif x_q.ndim == 2:
        scale = scale_i8.reshape(1, -1).astype(np.float32)
        bias = bias_i8.reshape(1, -1).astype(np.float32)
    else:
        scale = scale_i8.astype(np.float32)
        bias = bias_i8.astype(np.float32)
    return np.where(scale * x_q + bias >= 0.0, 1.0, -1.0).astype(np.float32)


def _collect_c_tensors(
    model,
    image: np.ndarray,
    export_dir: Path,
    model_name: str,
    array_overrides: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    outputs = export_model_to_c(model, export_dir, model_name=model_name, array_overrides=array_overrides)
    meta = json.loads(outputs["json"].read_text())
    lib_path = _compile_debug_export(export_dir, model_name, int(meta["arena_bytes"]))
    dll = ctypes.CDLL(str(lib_path))
    dll.run_model_debug.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    dll.run_model_debug.restype = ctypes.c_int

    tensors = {}
    input_ptr = image.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    for tensor_id, desc in enumerate(meta["tensors"]):
        nbytes = int(desc["bytes"])
        if nbytes <= 0:
            continue
        raw = np.zeros((nbytes,), dtype=np.uint8)
        rc = dll.run_model_debug(input_ptr, tensor_id, raw.ctypes.data_as(ctypes.c_void_p), nbytes)
        if rc != 0:
            raise RuntimeError(f"C runtime returned error code {rc} for tensor {tensor_id}")
        shape = tuple(desc["shape"])
        if desc["kind"] == "KBNN_TENSOR_I32":
            tensors[desc["name"]] = raw.view(np.int32).astype(np.float32).reshape(shape) / FX_ONE
        elif desc["kind"] == "KBNN_TENSOR_PACKED_BINARY":
            tensors[desc["name"]] = _decode_packed_binary(raw.view(np.uint64), shape, int(desc["words_per_point"]))
    return tensors, meta


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
        subset_train_as_val=args.subset_train_as_val,
    )
    _, val_samples, _, classes = _prepare_split_samples(split_args, args.input_size)
    val_samples = val_samples[: args.max_images]
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

    tensor_names = {layer.name for layer in model.layers}
    keras_layers = [layer for layer in model.layers if layer.name in tensor_names]
    submodel = keras.Model(model.input, [layer.output for layer in keras_layers])
    layer_names = [layer.name for layer in keras_layers]

    keras_outs = submodel(images, training=False)
    if not isinstance(keras_outs, (list, tuple)):
        keras_outs = [keras_outs]
    keras_map = {name: np.asarray(value) for name, value in zip(layer_names, keras_outs)}

    agg = {}
    with tempfile.TemporaryDirectory(prefix="kerasbnn_layer_parity_") as td:
        for image in images:
            c_tensors, meta = _collect_c_tensors(model, image, Path(td), model_name="layer_dbg")
            for name, c_value in c_tensors.items():
                if name not in keras_map:
                    continue
                k_value = keras_map[name][0 if keras_map[name].shape[0] == 1 else 0]
                # Choose matching sample by image index later via accumulation.
        # rerun accumulation sample-aware
    agg = {}
    with tempfile.TemporaryDirectory(prefix="kerasbnn_layer_parity_") as td:
        for idx, image in enumerate(images):
            c_tensors, meta = _collect_c_tensors(model, image, Path(td), model_name=f"layer_dbg_{idx}")
            for name, c_value in c_tensors.items():
                if name not in keras_map:
                    continue
                k_value = np.asarray(keras_map[name][idx])
                if c_value.shape != k_value.shape:
                    continue
                diff = np.abs(c_value - k_value)
                item = agg.setdefault(
                    name,
                    {
                        "kind": next(t["kind"] for t in meta["tensors"] if t["name"] == name),
                        "shape": list(c_value.shape),
                        "sum_abs": 0.0,
                        "count": 0,
                        "max_abs": 0.0,
                        "mean_k_abs": 0.0,
                    },
                )
                item["sum_abs"] += float(np.sum(diff))
                item["count"] += int(diff.size)
                item["max_abs"] = max(item["max_abs"], float(np.max(diff)))
                item["mean_k_abs"] += float(np.mean(np.abs(k_value)))

    rows = []
    packed_rows = []
    layer_map = {layer.name: layer for layer in model.layers}
    for name, item in agg.items():
        mean_abs = item["sum_abs"] / max(item["count"], 1)
        mean_k_abs = item["mean_k_abs"] / max(len(images), 1)
        rel = mean_abs / max(mean_k_abs, 1e-8)
        rows.append(
            {
                "name": name,
                "kind": item["kind"],
                "shape": item["shape"],
                "mean_abs_diff": mean_abs,
                "max_abs_diff": item["max_abs"],
                "mean_abs_ref": mean_k_abs,
                "relative_mean_diff": rel,
            }
        )

    with tempfile.TemporaryDirectory(prefix="kerasbnn_layer_parity_packed_") as td:
        for idx, image in enumerate(images):
            c_tensors, meta = _collect_c_tensors(model, image, Path(td), model_name=f"layer_dbg_packed_{idx}")
            for op in meta["ops"]:
                if op["kind"] != "KBNN_OP_XNOR_CONV2D":
                    continue
                layer_name = op["name"]
                packed_name = f"{layer_name}_packed"
                input_id = int(op["input"])
                input_name = meta["tensors"][input_id]["name"]
                if packed_name not in c_tensors or input_name not in keras_map or layer_name not in layer_map:
                    continue
                layer = layer_map[layer_name]
                k_input = np.asarray(keras_map[input_name][idx])
                expected = _expected_xnor_binary_input_np(layer, k_input)
                got = np.asarray(c_tensors[packed_name], dtype=np.float32)
                if expected.shape != got.shape:
                    continue
                disagree = float(np.mean(expected != got))
                packed_rows.append(
                    {
                        "layer": layer_name,
                        "input_name": input_name,
                        "disagree_rate": disagree,
                        "shape": list(got.shape),
                    }
                )

    rows.sort(key=lambda r: (r["relative_mean_diff"], r["mean_abs_diff"]), reverse=True)
    packed_rows.sort(key=lambda r: r["disagree_rate"], reverse=True)
    print(json.dumps({"run_dir": str(run_dir), "top": rows[: args.top], "packed_top": packed_rows[: args.top]}, indent=2))


if __name__ == "__main__":
    main()
