from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binary_yolo.model import ModelConfig, build_binary_yolov3_small
from binary_yolo.yad2k_core import VOC_CLASSES


FILE_MAGIC = b"BYOLOv1\0"
FILE_VERSION = 5
CONV_OC_BLOCK = 16

RECORD_BINARY_CONV = 1
RECORD_RAW_PROJ = 2
RECORD_QDENSE = 3
RECORD_QDENSE_OUT = 4

CMP_MODE_GE = 0
CMP_MODE_LE = 1
CMP_MODE_CONST_POS = 2
CMP_MODE_CONST_NEG = 3

RAW_PROJ_MODE_INC = 0
RAW_PROJ_MODE_DEC = 1
RAW_PROJ_MODE_CONST_MIN = 2
RAW_PROJ_MODE_CONST_MAX = 3

DEFAULT_WEIGHTS = ROOT / "runs" / "voc2012_320_warm111_strongaug_dropcut10_matcharch_gpu1" / "model.voc.best.weights.h5"
DEFAULT_OUTPUT = ROOT / "binary_yolo" / "default_voc320_xnor.weights.bin"
DEFAULT_SUMMARY = ROOT / "runs" / "voc2012_320_warm111_strongaug_dropcut10_matcharch_gpu1" / "summary.json"
DEFAULT_CFG = ModelConfig()
DEFAULT_NUM_CLASSES = 20
DEFAULT_ANCHORS = np.asarray(
    [
        [15.35999870300293, 27.95179557800293],
        [45.44000244140625, 39.51806640625],
        [26.2399959564209, 70.82667541503906],
        [49.280006408691406, 132.26666259765625],
        [85.12000274658203, 82.77333068847656],
        [97.2800064086914, 195.08023071289062],
        [193.15313720703125, 131.52957153320312],
        [168.23849487304688, 247.46665954589844],
        [284.15997314453125, 274.69879150390625],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RecordSpec:
    idx: int
    kind: int
    layer_name: str
    aux_layer_name: str | None = None


EXPORT_SPECS = [
    RecordSpec(0, RECORD_BINARY_CONV, "stage1_conv"),
    RecordSpec(1, RECORD_BINARY_CONV, "stage2_conv"),
    RecordSpec(2, RECORD_BINARY_CONV, "stage3_conv"),
    RecordSpec(3, RECORD_BINARY_CONV, "stage4_conv"),
    RecordSpec(4, RECORD_BINARY_CONV, "stage5_conv"),
    RecordSpec(5, RECORD_BINARY_CONV, "stage6_conv"),
    RecordSpec(6, RECORD_BINARY_CONV, "stage7_conv"),
    RecordSpec(7, RECORD_BINARY_CONV, "stage8_conv"),
    RecordSpec(8, RECORD_BINARY_CONV, "p5_feat_conv"),
    RecordSpec(9, RECORD_BINARY_CONV, "up4_reduce_conv"),
    RecordSpec(10, RECORD_BINARY_CONV, "p4_feat_conv"),
    RecordSpec(11, RECORD_BINARY_CONV, "up3_reduce_conv"),
    RecordSpec(12, RECORD_BINARY_CONV, "p3_feat_conv"),
    RecordSpec(13, RECORD_BINARY_CONV, "p3_pre_conv"),
    RecordSpec(14, RECORD_BINARY_CONV, "p4_pre_conv"),
    RecordSpec(15, RECORD_BINARY_CONV, "p5_pre_conv"),
    RecordSpec(16, RECORD_RAW_PROJ, "p3_proj_conv", "p3_proj_bn"),
    RecordSpec(17, RECORD_RAW_PROJ, "p4_proj_conv", "p4_proj_bn"),
    RecordSpec(18, RECORD_RAW_PROJ, "p5_proj_conv", "p5_proj_bn"),
    RecordSpec(19, RECORD_QDENSE, "p3_obj_hidden_dense"),
    RecordSpec(20, RECORD_QDENSE, "p3_obj_remap_dense"),
    RecordSpec(21, RECORD_QDENSE, "p3_cls_hidden_dense"),
    RecordSpec(22, RECORD_QDENSE_OUT, "p3_bbox"),
    RecordSpec(23, RECORD_QDENSE_OUT, "p3_obj"),
    RecordSpec(24, RECORD_QDENSE_OUT, "p3_cls"),
    RecordSpec(25, RECORD_QDENSE, "p4_obj_hidden_dense"),
    RecordSpec(26, RECORD_QDENSE, "p4_obj_remap_dense"),
    RecordSpec(27, RECORD_QDENSE, "p4_cls_hidden_dense"),
    RecordSpec(28, RECORD_QDENSE_OUT, "p4_bbox"),
    RecordSpec(29, RECORD_QDENSE_OUT, "p4_obj"),
    RecordSpec(30, RECORD_QDENSE_OUT, "p4_cls"),
    RecordSpec(31, RECORD_QDENSE, "p5_obj_hidden_dense"),
    RecordSpec(32, RECORD_QDENSE, "p5_obj_remap_dense"),
    RecordSpec(33, RECORD_QDENSE, "p5_cls_hidden_dense"),
    RecordSpec(34, RECORD_QDENSE_OUT, "p5_bbox"),
    RecordSpec(35, RECORD_QDENSE_OUT, "p5_obj"),
    RecordSpec(36, RECORD_QDENSE_OUT, "p5_cls"),
]


def _record_name(kind: int) -> str:
    return {
        RECORD_BINARY_CONV: "binary_conv",
        RECORD_RAW_PROJ: "raw_proj",
        RECORD_QDENSE: "qdense",
        RECORD_QDENSE_OUT: "qdense_out",
    }[kind]


def _clip_signed_unit(x: np.ndarray | float, levels: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    max_unit = np.float32((levels - 1) / levels)
    return np.clip(x, -1.0, max_unit).astype(np.float32)


def _signed_unit_codes(x: np.ndarray | float, levels: int) -> np.ndarray:
    clipped = _clip_signed_unit(x, levels)
    return np.floor(clipped * np.float32(levels)).astype(np.int16)


def _signed_int_codes(x: np.ndarray | float, levels: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    qmax = np.float32(levels - 1)
    clipped = np.clip(x, -1.0, 1.0).astype(np.float32)
    return np.clip(np.rint(clipped * qmax), -qmax, qmax).astype(np.int16)


def _quantize_symmetric_range(x: np.ndarray | float, max_abs: float, levels: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    qmax = np.float32(levels - 1)
    max_abs = np.float32(max_abs)
    clipped = np.clip(x, -max_abs, max_abs).astype(np.float32)
    step = max(max_abs / qmax, np.float32(1e-6))
    quantized = np.rint(clipped / step).astype(np.float32) * step
    return np.clip(quantized, -max_abs, max_abs).astype(np.float32)


def _quantize_positive_range(x: np.ndarray | float, max_value: float, levels: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    qmax = np.float32(levels - 1)
    max_value = np.float32(max_value)
    clipped = np.clip(x, 0.0, max_value).astype(np.float32)
    step = max(max_value / qmax, np.float32(1e-6))
    quantized = np.rint(clipped / step).astype(np.float32) * step
    return np.clip(quantized, 0.0, max_value).astype(np.float32)


def _f16(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def _quantize_symmetric_range_f16(x: np.ndarray | float, max_abs: float, levels: int) -> np.ndarray:
    x = _f16(x)
    qmax = np.float16(levels - 1)
    max_abs16 = np.float16(max_abs)
    clipped = _f16(np.clip(x, -max_abs16, max_abs16))
    step = np.float16(max(float(max_abs16 / qmax), 1e-6))
    scaled = _f16(clipped / step)
    quantized = _f16(np.rint(scaled).astype(np.float32) * np.float32(step))
    return _f16(np.clip(quantized, -max_abs16, max_abs16))


def _quantize_positive_range_f16(x: np.ndarray | float, max_value: float, levels: int) -> np.ndarray:
    x = _f16(x)
    qmax = np.float16(levels - 1)
    max_value16 = np.float16(max_value)
    clipped = _f16(np.clip(x, 0.0, max_value16))
    step = np.float16(max(float(max_value16 / qmax), 1e-6))
    scaled = _f16(clipped / step)
    quantized = _f16(np.rint(scaled).astype(np.float32) * np.float32(step))
    return _f16(np.clip(quantized, 0.0, max_value16))


def _quantize_symmetric_code_f16(x: np.ndarray | float, max_abs: float, levels: int) -> np.ndarray:
    x = _f16(x)
    qmax = np.float16(levels - 1)
    max_abs16 = np.float16(max_abs)
    clipped = _f16(np.clip(x, -max_abs16, max_abs16))
    step = np.float16(max(float(max_abs16 / qmax), 1e-6))
    scaled = _f16(clipped / step)
    return np.clip(np.rint(scaled).astype(np.int16), -(levels - 1), levels - 1)


def _quantize_positive_code_f16(x: np.ndarray | float, max_value: float, levels: int) -> np.ndarray:
    x = _f16(x)
    qmax = np.float16(levels - 1)
    max_value16 = np.float16(max_value)
    clipped = _f16(np.clip(x, 0.0, max_value16))
    step = np.float16(max(float(max_value16 / qmax), 1e-6))
    scaled = _f16(clipped / step)
    return np.clip(np.rint(scaled).astype(np.int16), 0, levels - 1)


def _symmetric_step_f16(max_abs: float, levels: int) -> np.float32:
    qmax = np.float16(levels - 1)
    max_abs16 = np.float16(max_abs)
    return _f16(np.float16(max(float(max_abs16 / qmax), 1e-6)))


def _positive_step_f16(max_value: float, levels: int) -> np.float32:
    qmax = np.float16(levels - 1)
    max_value16 = np.float16(max_value)
    return _f16(np.float16(max(float(max_value16 / qmax), 1e-6)))


def _pack_binary_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 4:
        raise ValueError(f"Expected a rank-4 conv kernel. Got shape {kernel.shape!r}")
    k_h, k_w, in_c, out_c = kernel.shape
    words = (int(in_c) + 63) // 64
    packed = np.zeros((out_c, k_h, k_w, words), dtype=np.uint64)
    signs = kernel >= 0.0
    for out_idx in range(out_c):
        for ky in range(k_h):
            for kx in range(k_w):
                for in_idx in range(in_c):
                    if signs[ky, kx, in_idx, out_idx]:
                        packed[out_idx, ky, kx, in_idx // 64] |= np.uint64(1) << np.uint64(in_idx & 63)
    return packed.reshape(-1)


def _pack_binary_kernel_blocked(kernel: np.ndarray, block: int = CONV_OC_BLOCK) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 4:
        raise ValueError(f"Expected a rank-4 conv kernel. Got shape {kernel.shape!r}")
    k_h, k_w, in_c, out_c = kernel.shape
    words = (int(in_c) + 63) // 64
    out_blocks = (int(out_c) + block - 1) // block
    packed = np.zeros((out_blocks, k_h, k_w, words, block), dtype=np.uint64)
    signs = kernel >= 0.0
    for out_idx in range(out_c):
        block_idx = out_idx // block
        lane = out_idx % block
        for ky in range(k_h):
            for kx in range(k_w):
                for in_idx in range(in_c):
                    if signs[ky, kx, in_idx, out_idx]:
                        packed[block_idx, ky, kx, in_idx // 64, lane] |= np.uint64(1) << np.uint64(in_idx & 63)
    return packed.reshape(-1)


def _export_binary_affine_comparators(layer) -> tuple[np.ndarray, np.ndarray]:
    aff = layer.affine_norm
    raw_extent = int(layer._kernel_elements)
    raw_values = np.arange(-raw_extent, raw_extent + 1, dtype=np.int16)
    threshold_limit = np.float32(math.sqrt(float(raw_extent)))
    probes = np.repeat((raw_values.astype(np.float32) / threshold_limit).reshape(1, 1, -1, 1), int(layer.filters), axis=-1)
    outputs = np.asarray(aff(probes, training=False), dtype=np.float32).reshape(raw_values.shape[0], int(layer.filters))
    active = outputs >= 0.0

    filters = int(layer.filters)
    packed_filters = ((filters + CONV_OC_BLOCK - 1) // CONV_OC_BLOCK) * CONV_OC_BLOCK
    cmp_mode = np.full((packed_filters,), CMP_MODE_CONST_NEG, dtype=np.uint8)
    cmp_threshold = np.zeros((packed_filters,), dtype=np.int16)

    for out_idx in range(filters):
        signs = active[:, out_idx]
        if np.all(signs):
            cmp_mode[out_idx] = CMP_MODE_CONST_POS
            continue
        if not np.any(signs):
            cmp_mode[out_idx] = CMP_MODE_CONST_NEG
            continue
        if (not signs[0]) and signs[-1]:
            threshold = int(raw_values[int(np.argmax(signs))])
            cmp_mode[out_idx] = CMP_MODE_GE
            cmp_threshold[out_idx] = threshold
            if not np.array_equal(signs, raw_values >= threshold):
                raise ValueError(f"{layer.name}: non-threshold GE response in channel {out_idx}")
            continue
        if signs[0] and (not signs[-1]):
            threshold = int(raw_values[int(len(raw_values) - 1 - np.argmax(signs[::-1]))])
            cmp_mode[out_idx] = CMP_MODE_LE
            cmp_threshold[out_idx] = threshold
            if not np.array_equal(signs, raw_values <= threshold):
                raise ValueError(f"{layer.name}: non-threshold LE response in channel {out_idx}")
            continue
        raise ValueError(f"{layer.name}: unsupported BinaryAffineNorm response in channel {out_idx}")
    return cmp_mode, cmp_threshold


def _export_binary_conv(spec: RecordSpec, layer) -> bytes:
    packed_kernel = _pack_binary_kernel_blocked(layer.kernel.numpy())
    cmp_mode, cmp_threshold = _export_binary_affine_comparators(layer)
    blob = bytearray()
    blob.extend(packed_kernel.astype(np.uint64).tobytes())
    blob.extend(cmp_mode.tobytes())
    blob.extend(cmp_threshold.tobytes())
    return bytes(blob)


def _export_raw_proj(spec: RecordSpec, conv_layer, bn_layer) -> bytes:
    packed_kernel = _pack_binary_kernel(conv_layer.kernel.numpy())
    raw_extent = int(conv_layer._kernel_elements)
    raw_values = np.arange(-raw_extent, raw_extent + 1, dtype=np.float32)[:, None]
    conv_bias = conv_layer.bias.numpy().astype(np.float32)[None, :]
    gamma = bn_layer.gamma.numpy().astype(np.float32)[None, :]
    beta = bn_layer.beta.numpy().astype(np.float32)[None, :]
    mean = bn_layer.moving_mean.numpy().astype(np.float32)[None, :]
    variance = bn_layer.moving_variance.numpy().astype(np.float32)[None, :]
    denom = np.sqrt(variance + np.float32(bn_layer.epsilon))
    projected = _f16(raw_values / np.float32(raw_extent) + conv_bias)
    normalized = _f16(gamma * _f16(projected - mean) / denom + beta)
    lut = _signed_unit_codes(normalized, levels=8).astype(np.int16).T.copy()
    raw_domain = np.arange(-raw_extent, raw_extent + 1, dtype=np.int16)
    mode = np.empty((conv_layer.filters,), dtype=np.uint8)
    thresholds = np.zeros((conv_layer.filters, 15), dtype=np.int16)

    for out_idx in range(int(conv_layer.filters)):
        codes = lut[out_idx]
        dif = np.diff(codes)
        if np.all(codes == -8):
            mode[out_idx] = RAW_PROJ_MODE_CONST_MIN
            continue
        if np.all(codes == 7):
            mode[out_idx] = RAW_PROJ_MODE_CONST_MAX
            continue
        if np.all(dif >= 0):
            mode[out_idx] = RAW_PROJ_MODE_INC
            for step_idx, target_code in enumerate(range(-7, 8)):
                hits = np.flatnonzero(codes >= target_code)
                thresholds[out_idx, step_idx] = raw_domain[hits[0]] if hits.size else np.int16(raw_extent + 1)
            recon = -8 + np.sum(raw_domain[:, None] >= thresholds[out_idx][None, :], axis=1)
        elif np.all(dif <= 0):
            mode[out_idx] = RAW_PROJ_MODE_DEC
            for step_idx, target_code in enumerate(range(6, -9, -1)):
                hits = np.flatnonzero(codes <= target_code)
                thresholds[out_idx, step_idx] = raw_domain[hits[0]] if hits.size else np.int16(raw_extent + 1)
            recon = 7 - np.sum(raw_domain[:, None] >= thresholds[out_idx][None, :], axis=1)
        else:
            raise ValueError(f"{conv_layer.name}: raw projection output is not monotonic for channel {out_idx}")
        if not np.array_equal(recon.astype(np.int16), codes):
            raise ValueError(f"{conv_layer.name}: failed to reconstruct compact raw projection for channel {out_idx}")

    blob = bytearray()
    blob.extend(packed_kernel.astype(np.uint64).tobytes())
    blob.extend(mode.tobytes())
    blob.extend(thresholds.reshape(-1).tobytes())
    return bytes(blob)


def _export_qdense(spec: RecordSpec, layer) -> bytes:
    weights = _signed_unit_codes(layer.kernel.numpy(), levels=int(layer.levels)).astype(np.int8).T.copy()
    bias = layer.bias.numpy().astype(np.float32).copy()
    blob = bytearray()
    blob.extend(weights.reshape(-1).tobytes())
    blob.extend(bias.tobytes())
    return bytes(blob)


def _export_qdense_out(spec: RecordSpec, layer) -> bytes:
    levels = int(layer.levels)
    weights = _signed_int_codes(layer.kernel.numpy(), levels=levels).astype(np.int8).T.copy()
    qmax = levels - 1
    bias_q = np.clip(np.rint(layer.bias.numpy().astype(np.float32) / np.float32(layer.output_scale)), -qmax, qmax).astype(np.int8)
    blob = bytearray()
    blob.extend(weights.reshape(-1).tobytes())
    blob.extend(bias_q.tobytes())
    return bytes(blob)


def _export_record(spec: RecordSpec, model) -> tuple[bytes, dict]:
    layer = model.get_layer(spec.layer_name)
    if spec.kind == RECORD_BINARY_CONV:
        blob = _export_binary_conv(spec, layer)
    elif spec.kind == RECORD_RAW_PROJ:
        aux = model.get_layer(spec.aux_layer_name)
        blob = _export_raw_proj(spec, layer, aux)
    elif spec.kind == RECORD_QDENSE:
        blob = _export_qdense(spec, layer)
    elif spec.kind == RECORD_QDENSE_OUT:
        blob = _export_qdense_out(spec, layer)
    else:
        raise ValueError(f"Unsupported record kind: {spec.kind}")
    info = {
        "idx": spec.idx,
        "kind": _record_name(spec.kind),
        "layer_name": spec.layer_name,
        "aux_layer_name": spec.aux_layer_name,
        "blob_bytes": len(blob),
    }
    return blob, info


def _write_record(handle, spec: RecordSpec, blob: bytes):
    handle.write(struct.pack("<IIII", spec.idx, spec.kind, len(blob), 0))
    handle.write(blob)


def parse_args():
    ap = argparse.ArgumentParser(description="Export the hardcoded binary VOC-320 YOLO model into a C-native packed weight file.")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the Keras weights file to export.")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to the packed binary output file.")
    ap.add_argument(
        "--manifest",
        default="",
        help="Optional JSON manifest path. Defaults to `<output>.json`.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(output_path.suffix + ".json")

    model = build_binary_yolov3_small(DEFAULT_NUM_CLASSES, DEFAULT_CFG)
    model.load_weights(weights_path)

    blobs = []
    manifest_records = []
    for spec in EXPORT_SPECS:
        blob, info = _export_record(spec, model)
        blobs.append((spec, blob))
        manifest_records.append(info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(struct.pack("<8sIII", FILE_MAGIC, FILE_VERSION, len(blobs), 0))
        for spec, blob in blobs:
            _write_record(handle, spec, blob)

    summary_payload = json.loads(DEFAULT_SUMMARY.read_text()) if DEFAULT_SUMMARY.exists() else {}
    manifest = {
        "file_magic": FILE_MAGIC.decode("ascii", errors="ignore"),
        "file_version": FILE_VERSION,
        "weights_source": str(weights_path),
        "summary_source": str(DEFAULT_SUMMARY),
        "output_path": str(output_path),
        "default_model_config": asdict(DEFAULT_CFG),
        "classes": VOC_CLASSES,
        "anchors": DEFAULT_ANCHORS.tolist(),
        "summary": {
            "best_val_map50": summary_payload.get("full", {}).get("best_val_map50"),
            "test_map50": summary_payload.get("test", {}).get("map50"),
        },
        "records": manifest_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {output_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
