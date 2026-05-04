from __future__ import annotations

import ctypes
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
import numpy as np

sys.path.insert(0, "/home/claude/source/yolovxnor")

from kerasbnn.export_c import FX_ONE, export_model_to_c  # noqa: E402
from kerasbnn.runtime_quant import fold_xnor_activation_compare_np, quantize_to_fixed_grid_np  # noqa: E402
from kerasbnn.xnor2_layers import XNORBinaryConv2D, XNORBinaryDense  # noqa: E402
from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2  # noqa: E402


REPO_ROOT = Path("/home/claude/source/yolovxnor")


def _compile_shared_export(output_dir: Path, model_name: str, arena_bytes: int, outputs: list[int], counts: list[int]) -> Path:
    out_args = ", ".join([f"int32_t *out{i}" for i in range(len(outputs))])
    copies = "\n".join(
        [
            f"        memcpy(out{i}, kbnn_context_get_output_i32(&ctx, {tensor_id}), sizeof(int32_t) * {count});"
            for i, (tensor_id, count) in enumerate(zip(outputs, counts))
        ]
    )
    wrapper = output_dir / f"{model_name}_wrapper.c"
    wrapper.write_text(
        textwrap.dedent(
            f"""
            #include <string.h>
            #include "kerasbnn.h"
            #include "{model_name}_export.h"

            int run_model(const float *input, {out_args}) {{
                static unsigned char arena[{arena_bytes}];
                kbnn_context ctx;
                int rc = kbnn_context_init(&ctx, &{model_name}, arena, sizeof(arena));
                if (rc != 0) return rc;
                kbnn_context_set_input_f32(&ctx, 0, input);
                rc = kbnn_model_run(&ctx);
                if (rc == 0) {{
            {copies}
                }}
                kbnn_context_release(&ctx);
                return rc;
            }}
            """
        ).strip()
        + "\n"
    )
    lib_path = output_dir / f"lib{model_name}.so"
    cmd = [
        "clang",
        "-shared",
        "-fPIC",
        "-Ofast",
        "-std=c11",
        "-march=native",
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


def _run_exported_model(model, image: np.ndarray, model_name: str) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="kerasbnn_runtime_") as td:
        output_dir = Path(td)
        export_model_to_c(model, output_dir, model_name=model_name)
        meta = json.loads((output_dir / f"{model_name}_export.json").read_text())
        outputs = [int(v) for v in meta["outputs"]]
        counts = [int(np.prod(meta["tensors"][tensor_id]["shape"])) for tensor_id in outputs]
        lib_path = _compile_shared_export(output_dir, model_name, int(meta["arena_bytes"]), outputs, counts)
        dll = ctypes.CDLL(str(lib_path))
        dll.run_model.argtypes = [ctypes.POINTER(ctypes.c_float)] + [ctypes.POINTER(ctypes.c_int32)] * len(outputs)
        dll.run_model.restype = ctypes.c_int
        out_bufs = [np.zeros((count,), dtype=np.int32) for count in counts]
        args = [image.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))] + [
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) for buf in out_bufs
        ]
        rc = dll.run_model(*args)
        if rc != 0:
            raise RuntimeError(f"C runtime returned error code {rc}")
        out_arrays = []
        for tensor_id, buf in zip(outputs, out_bufs):
            shape = tuple(meta["tensors"][tensor_id]["shape"])
            out_arrays.append((buf.astype(np.float32) / FX_ONE).reshape(shape))
        return out_arrays


def _manual_known_xnor(layer: XNORBinaryConv2D, x: np.ndarray) -> np.ndarray:
    kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
    bias = quantize_to_fixed_grid_np(np.asarray(layer.bias.numpy(), dtype=np.float32))
    scale_i8, bias_i8 = fold_xnor_activation_compare_np(layer)
    prepared = (
        x.astype(np.float32) * scale_i8.reshape((1, 1, 1, -1)).astype(np.float32)
        + bias_i8.reshape((1, 1, 1, -1)).astype(np.float32)
    )
    input_sign = prepared >= 0.0
    weight_sign = kernel >= 0.0
    alpha = quantize_to_fixed_grid_np(np.mean(np.abs(kernel), axis=(0, 1, 2)))
    out = np.empty((x.shape[0], x.shape[1], x.shape[2], kernel.shape[-1]), dtype=np.float32)
    for b in range(x.shape[0]):
        for y in range(x.shape[1]):
            for x_idx in range(x.shape[2]):
                for oc in range(kernel.shape[-1]):
                    dot = 0
                    for ic in range(kernel.shape[2]):
                        dot += 1 if input_sign[b, y, x_idx, ic] == weight_sign[0, 0, ic, oc] else -1
                    out[b, y, x_idx, oc] = dot * alpha[oc] + bias[oc]
    return out


def _manual_known_xnor_dense(layer: XNORBinaryDense, x: np.ndarray) -> np.ndarray:
    kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
    bias = quantize_to_fixed_grid_np(np.asarray(layer.bias.numpy(), dtype=np.float32))
    scale_i8, bias_i8 = fold_xnor_activation_compare_np(layer)
    prepared = x.astype(np.float32) * scale_i8.reshape((1, -1)).astype(np.float32) + bias_i8.reshape((1, -1)).astype(np.float32)
    input_sign = np.where(prepared >= 0.0, 1.0, -1.0)
    weight_sign = np.where(kernel >= 0.0, 1.0, -1.0)
    alpha = quantize_to_fixed_grid_np(np.mean(np.abs(kernel), axis=0))
    return input_sign @ (weight_sign * alpha[None, :]) + bias[None, :]


def test_known_single_xnor_layer():
    inp = keras.Input(shape=(2, 2, 3), name="input")
    out = XNORBinaryConv2D(
        1,
        1,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="xnor",
    )(inp)
    model = keras.Model(inp, out)
    layer = model.get_layer("xnor")
    layer.kernel.assign(np.array([[[[0.4], [-0.7], [0.2]]]], dtype=np.float32))
    layer.bias.assign(np.array([0.15], dtype=np.float32))
    layer.affine_norm.norm_scale.assign(np.array([0.50, -0.75, 0.25], dtype=np.float32))
    layer.affine_norm.norm_threshold.assign(np.array([0.10, -0.20, 0.05], dtype=np.float32))
    layer.affine_norm.bias.assign(np.array([0.00, 0.15, -0.10], dtype=np.float32))
    layer.affine_norm.moving_mean.assign(np.array([0.05, -0.10, 0.20], dtype=np.float32))
    layer.affine_norm.moving_variance.assign(np.array([0.25, 0.50, 0.75], dtype=np.float32))
    layer.rprelu_gamma.assign(np.array([0.10, -0.05, 0.20], dtype=np.float32))
    layer.rprelu_beta.assign(np.array([0.50, 0.25, 0.75], dtype=np.float32))
    layer.rprelu_zeta.assign(np.array([-0.02, 0.15, -0.10], dtype=np.float32))

    image = np.array(
        [[[[0.25, -0.50, 0.75], [0.10, 0.20, -0.30]], [[0.40, -0.20, 0.10], [0.00, 0.50, -0.40]]]],
        dtype=np.float32,
    )
    keras_out = np.asarray(model(image, training=False))
    manual_out = _manual_known_xnor(layer, image)
    c_out = _run_exported_model(model, image, model_name="test_known_xnor")[0]

    if not np.allclose(keras_out, manual_out, atol=1e-6):
        raise AssertionError(f"Manual known-layer reference mismatch:\nkeras={keras_out}\nmanual={manual_out}")
    if not np.allclose(c_out, manual_out, atol=2.0 / FX_ONE):
        raise AssertionError(f"C known-layer reference mismatch:\nmanual={manual_out}\nc={c_out}")


def test_random_single_xnor_layer():
    rng = np.random.default_rng(7)
    inp = keras.Input(shape=(8, 8, 5), name="input")
    out = XNORBinaryConv2D(
        2,
        3,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="xnor_random",
    )(inp)
    model = keras.Model(inp, out)
    layer = model.get_layer("xnor_random")
    layer.kernel.assign(rng.uniform(-1.0, 1.0, size=layer.kernel.shape).astype(np.float32))
    layer.bias.assign(rng.normal(scale=0.2, size=layer.bias.shape).astype(np.float32))
    layer.affine_norm.norm_scale.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_scale.shape).astype(np.float32))
    layer.affine_norm.norm_threshold.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_threshold.shape).astype(np.float32))
    layer.affine_norm.bias.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.bias.shape).astype(np.float32))
    layer.affine_norm.moving_mean.assign(rng.normal(scale=0.5, size=layer.affine_norm.moving_mean.shape).astype(np.float32))
    layer.affine_norm.moving_variance.assign(rng.uniform(0.05, 2.0, size=layer.affine_norm.moving_variance.shape).astype(np.float32))
    layer.rprelu_gamma.assign(rng.normal(scale=0.3, size=layer.rprelu_gamma.shape).astype(np.float32))
    layer.rprelu_beta.assign(rng.uniform(0.0, 1.0, size=layer.rprelu_beta.shape).astype(np.float32))
    layer.rprelu_zeta.assign(rng.normal(scale=0.3, size=layer.rprelu_zeta.shape).astype(np.float32))

    image = rng.normal(size=(1, 8, 8, 5)).astype(np.float32)
    keras_out = np.asarray(model(image, training=False))
    c_out = _run_exported_model(model, image, model_name="test_random_xnor")[0]
    max_err = float(np.max(np.abs(c_out - keras_out)))
    if max_err > 8.0 / FX_ONE:
        raise AssertionError(f"Random single-layer C mismatch: max_err={max_err}")


def test_known_single_xnor_dense():
    inp = keras.Input(shape=(4,), name="input")
    out = XNORBinaryDense(
        3,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="xnor_dense",
    )(inp)
    model = keras.Model(inp, out)
    layer = model.get_layer("xnor_dense")
    layer.kernel.assign(
        np.array(
            [
                [0.25, -0.6, 0.8],
                [-0.4, 0.3, -0.7],
                [0.9, -0.1, 0.2],
                [-0.2, 0.5, -0.3],
            ],
            dtype=np.float32,
        )
    )
    layer.bias.assign(np.array([0.1, -0.2, 0.05], dtype=np.float32))
    layer.affine_norm.norm_scale.assign(np.array([0.50, -0.75, 0.25, 0.33], dtype=np.float32))
    layer.affine_norm.norm_threshold.assign(np.array([0.10, -0.20, 0.05, -0.15], dtype=np.float32))
    layer.affine_norm.bias.assign(np.array([0.00, 0.15, -0.10, 0.05], dtype=np.float32))
    layer.affine_norm.moving_mean.assign(np.array([0.05, -0.10, 0.20, 0.12], dtype=np.float32))
    layer.affine_norm.moving_variance.assign(np.array([0.25, 0.50, 0.75, 0.90], dtype=np.float32))
    layer.rprelu_gamma.assign(np.array([0.10, -0.05, 0.20, 0.02], dtype=np.float32))
    layer.rprelu_beta.assign(np.array([0.50, 0.25, 0.75, 0.4], dtype=np.float32))
    layer.rprelu_zeta.assign(np.array([-0.02, 0.15, -0.10, 0.07], dtype=np.float32))

    image = np.array([[0.25, -0.50, 0.75, 0.10]], dtype=np.float32)
    keras_out = np.asarray(model(image, training=False))
    manual_out = _manual_known_xnor_dense(layer, image)
    c_out = _run_exported_model(model, image, model_name="test_known_xnor_dense")[0]

    if not np.allclose(keras_out, manual_out, atol=1e-6):
        raise AssertionError(f"Manual known dense reference mismatch:\nkeras={keras_out}\nmanual={manual_out}")
    if not np.allclose(c_out, manual_out, atol=2.0 / FX_ONE):
        raise AssertionError(f"C known dense reference mismatch:\nmanual={manual_out}\nc={c_out}")


def test_random_single_xnor_dense():
    rng = np.random.default_rng(27)
    inp = keras.Input(shape=(13,), name="input")
    out = XNORBinaryDense(
        7,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="xnor_dense_random",
    )(inp)
    model = keras.Model(inp, out)
    layer = model.get_layer("xnor_dense_random")
    layer.kernel.assign(rng.uniform(-1.0, 1.0, size=layer.kernel.shape).astype(np.float32))
    layer.bias.assign(rng.normal(scale=0.2, size=layer.bias.shape).astype(np.float32))
    layer.affine_norm.norm_scale.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_scale.shape).astype(np.float32))
    layer.affine_norm.norm_threshold.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_threshold.shape).astype(np.float32))
    layer.affine_norm.bias.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.bias.shape).astype(np.float32))
    layer.affine_norm.moving_mean.assign(rng.normal(scale=0.5, size=layer.affine_norm.moving_mean.shape).astype(np.float32))
    layer.affine_norm.moving_variance.assign(rng.uniform(0.05, 2.0, size=layer.affine_norm.moving_variance.shape).astype(np.float32))
    layer.rprelu_gamma.assign(rng.normal(scale=0.3, size=layer.rprelu_gamma.shape).astype(np.float32))
    layer.rprelu_beta.assign(rng.uniform(0.0, 1.0, size=layer.rprelu_beta.shape).astype(np.float32))
    layer.rprelu_zeta.assign(rng.normal(scale=0.3, size=layer.rprelu_zeta.shape).astype(np.float32))

    image = rng.normal(size=(1, 13)).astype(np.float32)
    keras_out = np.asarray(model(image, training=False))
    c_out = _run_exported_model(model, image, model_name="test_random_xnor_dense")[0]
    max_err = float(np.max(np.abs(c_out - keras_out)))
    if max_err > 8.0 / FX_ONE:
        raise AssertionError(f"Random single dense C mismatch: max_err={max_err}")


def test_full_image_tinier_yolo_xnor2():
    rng = np.random.default_rng(11)
    model = build_binary_tinier_yolov3_xnor2(4, input_shape=(64, 64, 3), cfg=TinierXNOR2Config())
    image = rng.uniform(0.0, 1.0, size=(1, 64, 64, 3)).astype(np.float32)
    keras_outs = model(image, training=False)
    keras_pred_13 = np.asarray(keras_outs["pred_13"])
    keras_pred_26 = np.asarray(keras_outs["pred_26"])
    c_pred_13, c_pred_26 = _run_exported_model(model, image, model_name="test_full_tinier")

    def assert_quantized_close(name: str, c_out: np.ndarray, k_out: np.ndarray):
        per_anchor = k_out.shape[-1]
        if per_anchor < 6:
            raise AssertionError(f"Unexpected YOLO channel width for {name}: {per_anchor}")
        diff = np.abs(c_out - k_out)
        bbox_ok = np.all(diff[..., :4] <= 4.0 + 1e-6)
        obj_ok = np.all(diff[..., 4:5] <= 64.0 + 1e-6)
        cls_ok = np.all(diff[..., 5:] <= 32.0 + 1e-6)
        if not (bbox_ok and obj_ok and cls_ok):
            raise AssertionError(
                f"{name} full-image mismatch exceeded one output quantum:"
                f" max_bbox={float(np.max(diff[..., :4]))},"
                f" max_obj={float(np.max(diff[..., 4:5]))},"
                f" max_cls={float(np.max(diff[..., 5:]))}"
            )

    assert_quantized_close("pred_13", c_pred_13, keras_pred_13)
    assert_quantized_close("pred_26", c_pred_26, keras_pred_26)


def main():
    test_known_single_xnor_layer()
    print("known single-layer runtime test passed")
    test_random_single_xnor_layer()
    print("random single-layer runtime test passed")
    test_known_single_xnor_dense()
    print("known single dense runtime test passed")
    test_random_single_xnor_dense()
    print("random single dense runtime test passed")
    test_full_image_tinier_yolo_xnor2()
    print("full-image runtime test passed")


if __name__ == "__main__":
    main()
