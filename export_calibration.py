from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")

import keras
import numpy as np

from kerasbnn.debug_export_layer_parity import _compile_debug_export
from kerasbnn.export_c import FX_ONE, export_model_to_c
from kerasbnn.runtime_quant import fold_xnor_activation_compare_np
from kerasbnn.xnor2_layers import XNORBinaryConv2D


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _value_noise_2d(rng: np.random.Generator, h: int, w: int, grid_h: int, grid_w: int) -> np.ndarray:
    gh = max(int(grid_h), 1)
    gw = max(int(grid_w), 1)
    grid = rng.random((gh + 1, gw + 1), dtype=np.float32)
    ys = np.linspace(0.0, gh, h, endpoint=False, dtype=np.float32)
    xs = np.linspace(0.0, gw, w, endpoint=False, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, gh)
    x1 = np.minimum(x0 + 1, gw)
    fy = _fade(ys - y0)
    fx = _fade(xs - x0)

    out = np.empty((h, w), dtype=np.float32)
    for iy in range(h):
        g00 = grid[y0[iy], x0]
        g01 = grid[y0[iy], x1]
        g10 = grid[y1[iy], x0]
        g11 = grid[y1[iy], x1]
        top = g00 + (g01 - g00) * fx
        bot = g10 + (g11 - g10) * fx
        out[iy] = top + (bot - top) * fy[iy]
    return out


def generate_perlin_calibration_images(
    count: int,
    input_size: int,
    *,
    channels: int = 3,
    seed: int = 43,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    images = []
    dynamic = float(max_value - min_value)
    for _ in range(int(count)):
        img = np.zeros((input_size, input_size), dtype=np.float32)
        amp = 1.0
        amp_sum = 0.0
        base = int(rng.choice([2, 3, 4, 5, 6, 8]))
        for octave in range(4):
            freq = base * (2 ** octave)
            gh = max(1, input_size // freq)
            gw = max(1, input_size // freq)
            img += amp * _value_noise_2d(rng, input_size, input_size, gh, gw)
            amp_sum += amp
            amp *= 0.5
        img /= max(amp_sum, 1e-6)
        gamma = float(rng.uniform(0.7, 1.4))
        contrast = float(rng.uniform(0.75, 1.35))
        bias = float(rng.uniform(-0.12, 0.12))
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, gamma)
        img = np.clip((img - 0.5) * contrast + 0.5 + bias, 0.0, 1.0)
        image = np.repeat(img[..., None], channels, axis=-1)
        image = min_value + dynamic * image
        images.append(image.astype(np.float32))
    return np.stack(images, axis=0)


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


def _collect_named_c_tensors(
    model,
    images: np.ndarray,
    tensor_names: list[str],
    *,
    model_name: str,
    array_overrides: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    with tempfile.TemporaryDirectory(prefix="kerasbnn_iter_compare_") as td:
        output_dir = Path(td)
        outputs = export_model_to_c(model, output_dir, model_name=model_name, array_overrides=array_overrides)
        meta = json.loads(outputs["json"].read_text())
        lib_path = _compile_debug_export(output_dir, model_name, int(meta["arena_bytes"]))
        dll = ctypes.CDLL(str(lib_path))
        dll.run_model_debug.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        dll.run_model_debug.restype = ctypes.c_int

        desc_by_name = {desc["name"]: (idx, desc) for idx, desc in enumerate(meta["tensors"])}
        accum: dict[str, list[np.ndarray]] = {name: [] for name in tensor_names}
        for image in images:
            input_ptr = image.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            for name in tensor_names:
                tensor_id, desc = desc_by_name[name]
                nbytes = int(desc["bytes"])
                raw = np.zeros((nbytes,), dtype=np.uint8)
                rc = dll.run_model_debug(input_ptr, tensor_id, raw.ctypes.data_as(ctypes.c_void_p), nbytes)
                if rc != 0:
                    raise RuntimeError(f"C runtime returned error code {rc} for tensor {tensor_id} ({name})")
                shape = tuple(desc["shape"])
                if desc["kind"] == "KBNN_TENSOR_I32":
                    value = raw.view(np.int32).astype(np.float32).reshape(shape) / FX_ONE
                elif desc["kind"] == "KBNN_TENSOR_PACKED_BINARY":
                    value = _decode_packed_binary(raw.view(np.uint64), shape, int(desc["words_per_point"]))
                else:
                    raise ValueError(f"Unsupported tensor kind for calibration: {desc['kind']}")
                accum[name].append(value)
    return {name: np.stack(values, axis=0) for name, values in accum.items()}, meta


def build_xnor_compare_overrides(model, calib_images: np.ndarray, *, model_name: str) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    xnor_layers = [layer for layer in model.layers if isinstance(layer, XNORBinaryConv2D)]
    if not xnor_layers:
        return {}, []

    input_sources = []
    groups: list[tuple[str, list[XNORBinaryConv2D]]] = []
    group_layers: dict[str, list[XNORBinaryConv2D]] = {}
    seen = set()
    for layer in xnor_layers:
        src = layer.input._keras_history[0]
        if src.name not in seen:
            input_sources.append(src.output)
            seen.add(src.name)
            group_layers[src.name] = []
            groups.append((src.name, group_layers[src.name]))
        group_layers[src.name].append(layer)
    src_names = [t._keras_history[0].name for t in input_sources]
    submodel = keras.Model(model.input, input_sources)
    keras_inputs = submodel(calib_images, training=False)
    if not isinstance(keras_inputs, (list, tuple)):
        keras_inputs = [keras_inputs]
    keras_map = {name: np.asarray(val) for name, val in zip(src_names, keras_inputs)}

    overrides: dict[str, np.ndarray] = {}
    stats: list[dict[str, float]] = []
    for block_index, (src_name, block_layers) in enumerate(groups):
        c_input_map, _ = _collect_named_c_tensors(
            model,
            calib_images,
            [src_name],
            model_name=f"{model_name}_iterblk{block_index}",
            array_overrides=_remap_override_model_name(
                overrides,
                src_model_name=model_name,
                dst_model_name=f"{model_name}_iterblk{block_index}",
            ),
        )
        c_in = c_input_map[src_name]
        k_in = keras_map[src_name]
        for layer in block_layers:
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
            overrides[f"{model_name}_{layer.name}_act_scale_i8"] = act_scale
            overrides[f"{model_name}_{layer.name}_act_bias_i8"] = act_bias
            stats.append(
                {
                    "layer": layer.name,
                    "source": src_name,
                    "block_index": float(block_index),
                    "disagree_rate": total_err / max(total_count, 1),
                    "channels": float(channels),
                }
            )

    stats.sort(key=lambda r: r["disagree_rate"], reverse=True)
    return overrides, stats


def _predict_named(model, images: np.ndarray, batch_size: int = 8) -> dict[str, np.ndarray]:
    names = list(model.output_names)
    accum = {name: [] for name in names}
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        outputs = model(batch, training=False)
        if isinstance(outputs, dict):
            for name in names:
                accum[name].append(np.asarray(outputs[name]))
        elif isinstance(outputs, (list, tuple)):
            for name, value in zip(names, outputs):
                accum[name].append(np.asarray(value))
        else:
            accum[names[0]].append(np.asarray(outputs))
    return {name: np.concatenate(values, axis=0) for name, values in accum.items()}


def _mean_output_error(ref_preds: dict[str, np.ndarray], test_preds: dict[str, np.ndarray]) -> float:
    names = sorted(ref_preds.keys())
    if not names:
        return 0.0
    total = 0.0
    for name in names:
        total += float(np.mean(np.abs(np.asarray(test_preds[name]) - np.asarray(ref_preds[name]))))
    return total / float(len(names))


def _evaluate_export_output_error(
    model,
    eval_images: np.ndarray,
    *,
    model_name: str,
    array_overrides: dict[str, np.ndarray] | None,
) -> float:
    from kerasbnn.eval_fruit_codegen_parity import _predict_c_runtime

    keras_preds = _predict_named(model, eval_images)
    with tempfile.TemporaryDirectory(prefix="kerasbnn_perlin_eval_") as td:
        c_preds, _ = _predict_c_runtime(
            model,
            eval_images,
            Path(td),
            model_name=model_name,
            array_overrides=array_overrides,
        )
    return _mean_output_error(keras_preds, c_preds)


def _remap_override_model_name(
    overrides: dict[str, np.ndarray],
    *,
    src_model_name: str,
    dst_model_name: str,
) -> dict[str, np.ndarray]:
    if src_model_name == dst_model_name:
        return dict(overrides)
    prefix = f"{src_model_name}_"
    remapped: dict[str, np.ndarray] = {}
    for name, values in overrides.items():
        if name.startswith(prefix):
            remapped[f"{dst_model_name}_{name[len(prefix):]}"] = values
        else:
            remapped[name] = values
    return remapped


def resolve_export_calibration_mode(
    mode: str,
    *,
    dataset: str = "",
    checkpoint: str = "",
    model_name: str = "",
) -> str:
    mode = str(mode).lower()
    if mode != "auto":
        return mode
    haystack = " ".join([dataset, checkpoint, model_name]).lower()
    if "fruit" in haystack:
        return "perlin_compare"
    return "none"


def build_export_array_overrides(
    model,
    *,
    model_name: str,
    mode: str = "auto",
    dataset: str = "",
    checkpoint: str = "",
    calib_images: int = 4,
    seed: int = 43,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict]:
    resolved = resolve_export_calibration_mode(mode, dataset=dataset, checkpoint=checkpoint, model_name=model_name)
    if resolved == "none":
        return {}, {"mode": "none"}
    if resolved != "perlin_compare":
        raise ValueError(f"Unsupported export calibration mode: {resolved!r}")
    input_shape = tuple(model.input_shape[1:])
    input_size = int(input_shape[0])
    channels = int(input_shape[-1])
    images = generate_perlin_calibration_images(
        calib_images,
        input_size,
        channels=channels,
        seed=seed,
        min_value=min_value,
        max_value=max_value,
    )
    overrides, stats = build_xnor_compare_overrides(model, images, model_name=model_name)

    eval_images = generate_perlin_calibration_images(
        max(4, int(calib_images) // 2),
        input_size,
        channels=channels,
        seed=seed + 1009,
        min_value=min_value,
        max_value=max_value,
    )
    baseline_error = _evaluate_export_output_error(
        model,
        eval_images,
        model_name=f"{model_name}_perlin_base",
        array_overrides=None,
    )
    calibrated_error = _evaluate_export_output_error(
        model,
        eval_images,
        model_name=f"{model_name}_perlin_cal",
        array_overrides=_remap_override_model_name(
            overrides,
            src_model_name=model_name,
            dst_model_name=f"{model_name}_perlin_cal",
        ),
    )
    use_calibrated = calibrated_error < baseline_error
    selected_overrides = overrides if use_calibrated else {}
    selected_mode = resolved if use_calibrated else "none"
    return selected_overrides, {
        "mode": selected_mode,
        "requested_mode": resolved,
        "selected": "calibrated" if use_calibrated else "baseline",
        "calib_images": int(calib_images),
        "eval_images": int(eval_images.shape[0]),
        "seed": int(seed),
        "baseline_output_error": float(baseline_error),
        "calibrated_output_error": float(calibrated_error),
        "top_layer_disagree_after_fit": stats[:15],
    }


__all__ = [
    "build_export_array_overrides",
    "build_xnor_compare_overrides",
    "generate_perlin_calibration_images",
    "resolve_export_calibration_mode",
]
