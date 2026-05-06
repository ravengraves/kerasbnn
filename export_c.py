"""Generic C11 export for supported `kerasbnn` / Keras binary models.

The active lowering path supports:
- `InputLayer`
- `Activation` / `Reshape` / `Flatten` pass-through aliases
- `ThermometerBinaryInput`
- `QuantizedInput`
- `QuantizedActivation`
- `QuantizedConv2D`
- `QuantizedDense`
- fused `QuantizedActivation -> QuantizedDense`
- `XNORBinaryConv2D`
- `XNORBinaryDense`
- `BinaryAffineNorm`
- `SpaceToDepth`
- `Concatenate`
- `UpSampling2D`
- `Add`

The exporter writes packed/quantized weight headers in hex/integer form plus
JSON metadata. `export_model_bundle()` can also emit generic C11 wrapper/example
sources so the generated model can be compiled against `kerasbnn.c`.

Typical Python usage:

```python
from kerasbnn.export_c import export_model_bundle

artifacts = export_model_bundle(model, "out_dir", model_name="my_bnn")
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from kerasbnn.runtime_quant import (
    FX_ONE,
    fake_quantize_signed_output_np as _fake_quantize_signed_output,
    fold_binary_affine_float_ops as _fold_binary_affine_float_ops,
    fold_xnor_activation_compare_ops as _fold_xnor_activation_compare_ops,
)


def _shape_no_batch(tensor) -> tuple[int, ...]:
    shape = tuple(tensor.shape)
    if len(shape) <= 1:
        return tuple(int(v) for v in shape if v is not None)
    dims = []
    for dim in shape[1:]:
        if dim is None:
            raise ValueError(f"Dynamic non-batch dimension is not supported: {shape!r}")
        dims.append(int(dim))
    return tuple(dims)


def _fmt_f32(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return f"{value:.1f}f"
    return f"{value:.9g}f"


def _iter_inputs(layer) -> list[Any]:
    inputs = layer.input
    if isinstance(inputs, (list, tuple)):
        return list(inputs)
    return [inputs]


def _source_layer_name(tensor) -> str:
    return tensor._keras_history[0].name


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _quantize_signed_unit_grid(x: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, (levels - 1) / levels)
    return np.floor(clipped * levels).astype(np.int16)


def _quantize_signed_int_grid(x: np.ndarray, levels: int) -> np.ndarray:
    qmax = levels - 1
    clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return np.clip(np.rint(clipped * qmax), -qmax, qmax).astype(np.int16)


def _to_fx(x: np.ndarray | float) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * FX_ONE).astype(np.int32)


def _element_count(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _pack_binary_kernel(kernel_sign: np.ndarray) -> np.ndarray:
    # kernel_sign expected bool array [out_c, kh, kw, in_c]
    out_c, kh, kw, in_c = kernel_sign.shape
    words = (in_c + 63) // 64
    packed = np.zeros((out_c, kh, kw, words), dtype=np.uint64)
    for oc in range(out_c):
        for ky in range(kh):
            for kx in range(kw):
                bits = kernel_sign[oc, ky, kx]
                for c in range(in_c):
                    if bits[c]:
                        packed[oc, ky, kx, c // 64] |= np.uint64(1) << np.uint64(c & 63)
    return packed.reshape(-1)


def _pack_binary_dense_kernel(kernel_sign: np.ndarray) -> np.ndarray:
    # kernel_sign expected bool array [in_dim, out_dim]
    in_dim, out_dim = kernel_sign.shape
    words = (in_dim + 63) // 64
    packed = np.zeros((out_dim, words), dtype=np.uint64)
    for oc in range(out_dim):
        bits = kernel_sign[:, oc]
        for c in range(in_dim):
            if bits[c]:
                packed[oc, c // 64] |= np.uint64(1) << np.uint64(c & 63)
    return packed.reshape(-1)


def _transpose_qconv_kernel(kernel: np.ndarray) -> np.ndarray:
    # Keras [kh, kw, in_c, out_c] -> runtime [out_c, kh, kw, in_c]
    return np.transpose(kernel, (3, 0, 1, 2)).reshape(-1)


def _transpose_qdense_kernel(kernel: np.ndarray) -> np.ndarray:
    # Keras [in_dim, out_dim] -> runtime [out_dim, in_dim]
    return np.transpose(kernel, (1, 0)).reshape(-1)


def _fold_binary_affine(layer) -> tuple[np.ndarray, np.ndarray]:
    dtype = str(getattr(layer, "compute_dtype", "float32"))
    mul, add = _fold_binary_affine_float_ops(layer, dtype=dtype)
    mul = np.asarray(mul, dtype=np.float32)
    add = np.asarray(add, dtype=np.float32)
    return _to_fx(mul), _to_fx(add)


def _fold_xnor_activation_compare(layer) -> tuple[np.ndarray, np.ndarray]:
    dtype = str(getattr(layer, "compute_dtype", "float32"))
    scale_i8, bias_i8 = _fold_xnor_activation_compare_ops(layer, dtype=dtype)
    return np.asarray(scale_i8, dtype=np.int8), np.asarray(bias_i8, dtype=np.int8)


@dataclass
class ArrayDef:
    name: str
    c_type: str
    values: list[str]


@dataclass
class TensorDef:
    name: str
    kind: str
    rank: int
    shape: tuple[int, ...]
    words_per_point: int = 0
    offset_bytes: int = 0
    bytes: int = 0


@dataclass
class OpDef:
    kind: str
    name: str
    param_type: str
    param_name: str
    param_lines: list[str]
    json_payload: dict[str, Any] = field(default_factory=dict)


class ExportBuilder:
    def __init__(self, model_name: str, array_overrides: dict[str, np.ndarray] | None = None):
        self.model_name = model_name
        self.array_overrides = dict(array_overrides or {})
        self.arrays: list[ArrayDef] = []
        self.tensors: list[TensorDef] = []
        self.ops: list[OpDef] = []
        self.tensor_by_layer: dict[str, int] = {}
        self.output_tensor_ids: list[int] = []
        self.arena_offset = 0

    def add_array(self, name: str, c_type: str, values: np.ndarray | list[Any]) -> str:
        if name in self.array_overrides:
            values = self.array_overrides[name]
        arr = np.asarray(values).reshape(-1)
        rendered: list[str] = []
        if c_type == "uint64_t":
            rendered = [f"0x{int(v) & ((1 << 64) - 1):016x}ull" for v in arr]
        elif c_type in {"int8_t", "int16_t", "int32_t"}:
            rendered = [str(int(v)) for v in arr]
        elif c_type == "float":
            rendered = [_fmt_f32(v) for v in arr]
        else:
            raise ValueError(f"Unsupported C array type: {c_type}")
        self.arrays.append(ArrayDef(name=name, c_type=c_type, values=rendered))
        return name

    def add_tensor(self, name: str, kind: str, shape: tuple[int, ...], *, words_per_point: int = 0, alias_of: int | None = None) -> int:
        rank = len(shape)
        if alias_of is not None:
            src = self.tensors[alias_of]
            tensor = TensorDef(
                name=name,
                kind=src.kind,
                rank=rank,
                shape=shape,
                words_per_point=src.words_per_point if words_per_point == 0 else words_per_point,
                offset_bytes=src.offset_bytes,
                bytes=src.bytes,
            )
        else:
            if kind == "KBNN_TENSOR_EXTERNAL_F32":
                nbytes = 0
                offset = 0
            elif kind == "KBNN_TENSOR_I32":
                nbytes = int(np.prod(shape, dtype=np.int64)) * 4
                offset = _align_up(self.arena_offset, 64)
                self.arena_offset = offset + nbytes
            elif kind == "KBNN_TENSOR_PACKED_BINARY":
                spatial = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
                words = words_per_point or int(math.ceil(shape[-1] / 64.0))
                nbytes = spatial * words * 8
                offset = _align_up(self.arena_offset, 64)
                self.arena_offset = offset + nbytes
                words_per_point = words
            else:
                raise ValueError(f"Unsupported tensor kind: {kind}")
            tensor = TensorDef(
                name=name,
                kind=kind,
                rank=rank,
                shape=shape,
                words_per_point=words_per_point,
                offset_bytes=offset,
                bytes=nbytes,
            )
        idx = len(self.tensors)
        self.tensors.append(tensor)
        return idx

    def add_op(self, kind: str, name: str, param_type: str, param_lines: list[str], *, json_payload: dict[str, Any]) -> None:
        param_name = f"{self.model_name}_{name}_params"
        self.ops.append(
            OpDef(
                kind=kind,
                name=name,
                param_type=param_type,
                param_name=param_name,
                param_lines=param_lines,
                json_payload=json_payload,
            )
        )


def _emit_shape(shape: tuple[int, ...]) -> str:
    values = list(shape) + [0] * (5 - len(shape))
    return "{" + ", ".join(str(int(v)) for v in values[:5]) + "}"


def _tensor_kind_for_alias(builder: ExportBuilder, tensor_id: int) -> str:
    return builder.tensors[tensor_id].kind


def lower_model(model, model_name: str = "kerasbnn_model", array_overrides: dict[str, np.ndarray] | None = None) -> ExportBuilder:
    builder = ExportBuilder(model_name, array_overrides=array_overrides)
    consumer_count: dict[str, int] = {}
    fused_quant_act_for_dense: dict[str, Any] = {}
    skipped_quant_act_names: set[str] = set()
    for layer in model.layers:
        for tensor in _iter_inputs(layer):
            src_name = _source_layer_name(tensor)
            consumer_count[src_name] = consumer_count.get(src_name, 0) + 1
    for layer in model.layers:
        if layer.__class__.__name__ != "QuantizedDense":
            continue
        input_tensors = _iter_inputs(layer)
        if len(input_tensors) != 1:
            continue
        src_layer = input_tensors[0]._keras_history[0]
        if src_layer.__class__.__name__ != "QuantizedActivation":
            continue
        if consumer_count.get(src_layer.name, 0) != 1:
            continue
        fused_quant_act_for_dense[layer.name] = src_layer
        skipped_quant_act_names.add(src_layer.name)
    for layer in model.layers:
        class_name = layer.__class__.__name__
        if class_name == "InputLayer":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_EXTERNAL_F32", _shape_no_batch(layer.output))
            builder.tensor_by_layer[layer.name] = tensor_id
            continue

        input_tensors = _iter_inputs(layer)
        output_shape = _shape_no_batch(layer.output)
        if class_name == "QuantizedActivation" and layer.name in skipped_quant_act_names:
            continue
        if class_name == "QuantizedDense" and layer.name in fused_quant_act_for_dense:
            fused_act = fused_quant_act_for_dense[layer.name]
            fused_input_tensor = _iter_inputs(fused_act)[0]
            input_ids = [builder.tensor_by_layer[_source_layer_name(fused_input_tensor)]]
        else:
            input_ids = [builder.tensor_by_layer[_source_layer_name(t)] for t in input_tensors]

        if class_name in {"Activation", "Reshape", "Flatten"}:
            if class_name == "Activation":
                act_name = getattr(getattr(layer, "activation", None), "__name__", "")
                if act_name not in {"linear", ""}:
                    raise NotImplementedError(f"Only linear Activation aliases are supported. Layer: {layer.name}")
            if class_name in {"Reshape", "Flatten"}:
                input_shape = builder.tensors[input_ids[0]].shape
                if _element_count(input_shape) != _element_count(output_shape):
                    raise NotImplementedError(
                        f"{class_name} export requires equal element count. "
                        f"Layer: {layer.name} input={input_shape} output={output_shape}"
                    )
            tensor_id = builder.add_tensor(layer.name, _tensor_kind_for_alias(builder, input_ids[0]), output_shape, alias_of=input_ids[0])
            builder.tensor_by_layer[layer.name] = tensor_id
            continue

        if class_name == "ThermometerBinaryInput":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_PACKED_BINARY", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            builder.add_op(
                "KBNN_OP_THERMO_INPUT",
                layer.name,
                "kbnn_thermo_input_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".height = {output_shape[0]}",
                    f".width = {output_shape[1]}",
                    f".channels = {int(output_shape[2] // layer.levels)}",
                    f".levels = {int(layer.levels)}",
                    f".min_value = {_fmt_f32(layer.min_value)}",
                    f".max_value = {_fmt_f32(layer.max_value)}",
                ],
                json_payload={"input": input_ids[0], "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "QuantizedInput":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            builder.add_op(
                "KBNN_OP_QUANTIZED_INPUT",
                layer.name,
                "kbnn_quantized_input_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".rank = {len(output_shape)}",
                    f".shape = {_emit_shape(output_shape)}",
                    f".levels = {int(layer.levels)}",
                    f".min_value = {_fmt_f32(layer.min_value)}",
                    f".max_value = {_fmt_f32(layer.max_value)}",
                    f".signed_output = {1 if layer.signed else 0}",
                ],
                json_payload={"input": input_ids[0], "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "QuantizedConv2D":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
            if layer.integer_kernel:
                w_q = _quantize_signed_int_grid(kernel, layer.levels)
                weight_divisor = 1
            else:
                w_q = _quantize_signed_unit_grid(kernel, layer.levels)
                weight_divisor = int(layer.levels)
            w_name = builder.add_array(f"{model_name}_{layer.name}_weights", "int16_t", _transpose_qconv_kernel(w_q))
            if layer.bias is not None:
                bias = np.asarray(layer.bias.numpy(), dtype=np.float32)
                if layer.quantize_bias and layer.integer_output:
                    bias = _fake_quantize_signed_output(bias, layer.levels, layer.output_scale)
                b_name = builder.add_array(f"{model_name}_{layer.name}_bias_q", "int32_t", _to_fx(bias))
            else:
                b_name = "NULL"
            in_shape = builder.tensors[input_ids[0]].shape
            input_is_packed = builder.tensors[input_ids[0]].kind == "KBNN_TENSOR_PACKED_BINARY"
            stride_h, stride_w = layer.strides
            kernel_h, kernel_w = layer.kernel_size
            if layer.padding == "same":
                pad_h = max((output_shape[0] - 1) * stride_h + kernel_h - in_shape[0], 0)
                pad_w = max((output_shape[1] - 1) * stride_w + kernel_w - in_shape[1], 0)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
            else:
                pad_top = pad_bottom = pad_left = pad_right = 0
            builder.add_op(
                "KBNN_OP_QCONV2D",
                layer.name,
                "kbnn_qconv2d_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".in_h = {in_shape[0]}",
                    f".in_w = {in_shape[1]}",
                    f".in_c = {in_shape[2]}",
                    f".out_h = {output_shape[0]}",
                    f".out_w = {output_shape[1]}",
                    f".out_c = {output_shape[2]}",
                    f".kernel_h = {kernel_h}",
                    f".kernel_w = {kernel_w}",
                    f".stride_h = {stride_h}",
                    f".stride_w = {stride_w}",
                    f".pad_top = {pad_top}",
                    f".pad_bottom = {pad_bottom}",
                    f".pad_left = {pad_left}",
                    f".pad_right = {pad_right}",
                    f".input_is_packed_binary = {1 if input_is_packed else 0}",
                    f".weight_divisor = {weight_divisor}",
                    f".levels = {int(layer.levels)}",
                    f".integer_output = {1 if layer.integer_output else 0}",
                    f".output_scale_q = {int(round(float(layer.output_scale) * FX_ONE))}",
                    f".weights = {w_name}",
                    f".bias_q = {b_name}",
                ],
                json_payload={
                    "input": input_ids[0],
                    "output": tensor_id,
                    "shape": output_shape,
                    "in_shape": list(in_shape),
                    "kernel": [kernel_h, kernel_w],
                    "stride": [stride_h, stride_w],
                    "out_channels": int(output_shape[2]),
                    "input_is_packed_binary": bool(input_is_packed),
                    "weight_divisor": int(weight_divisor),
                },
            )
            continue

        if class_name == "XNORBinaryConv2D":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            scratch_shape = (*_shape_no_batch(input_tensors[0]),)
            scratch_id = builder.add_tensor(f"{layer.name}_packed", "KBNN_TENSOR_PACKED_BINARY", scratch_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
            alpha = np.mean(np.abs(kernel), axis=(0, 1, 2))
            bias = np.asarray(layer.bias.numpy(), dtype=np.float32) if layer.bias is not None else np.zeros((layer.filters,), dtype=np.float32)
            sign_bits = np.transpose(kernel >= 0.0, (3, 0, 1, 2))
            packed = _pack_binary_kernel(sign_bits)
            packed_name = builder.add_array(f"{model_name}_{layer.name}_weights_bits", "uint64_t", packed)
            alpha_name = builder.add_array(f"{model_name}_{layer.name}_alpha_q", "int32_t", _to_fx(alpha))
            bias_name = builder.add_array(f"{model_name}_{layer.name}_bias_q", "int32_t", _to_fx(bias))
            act_scale_i8, act_bias_i8 = _fold_xnor_activation_compare(layer)
            act_scale_name = builder.add_array(f"{model_name}_{layer.name}_act_scale_i8", "int8_t", act_scale_i8)
            act_bias_name = builder.add_array(f"{model_name}_{layer.name}_act_bias_i8", "int8_t", act_bias_i8)
            in_shape = builder.tensors[input_ids[0]].shape
            stride_h, stride_w = layer.strides
            kernel_h, kernel_w = layer.kernel_size
            if layer.padding == "same":
                pad_h = max((output_shape[0] - 1) * stride_h + kernel_h - in_shape[0], 0)
                pad_w = max((output_shape[1] - 1) * stride_w + kernel_w - in_shape[1], 0)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
            else:
                pad_top = pad_bottom = pad_left = pad_right = 0
            builder.add_op(
                "KBNN_OP_XNOR_CONV2D",
                layer.name,
                "kbnn_xnor_conv2d_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".scratch_id = {scratch_id}",
                    f".in_h = {in_shape[0]}",
                    f".in_w = {in_shape[1]}",
                    f".in_c = {in_shape[2]}",
                    f".out_h = {output_shape[0]}",
                    f".out_w = {output_shape[1]}",
                    f".out_c = {output_shape[2]}",
                    f".kernel_h = {kernel_h}",
                    f".kernel_w = {kernel_w}",
                    f".stride_h = {stride_h}",
                    f".stride_w = {stride_w}",
                    f".pad_top = {pad_top}",
                    f".pad_bottom = {pad_bottom}",
                    f".pad_left = {pad_left}",
                    f".pad_right = {pad_right}",
                    f".channel_words = {int(math.ceil(in_shape[2] / 64.0))}",
                    f".last_word_mask = 0x{int((1 << (in_shape[2] % 64)) - 1 if (in_shape[2] % 64) else (2**64 - 1)):016x}ull",
                    f".weights_bits = {packed_name}",
                    f".alpha_q = {alpha_name}",
                    f".bias_q = {bias_name}",
                    f".act_scale_i8 = {act_scale_name}",
                    f".act_bias_i8 = {act_bias_name}",
                    f".input_cast_fp16 = 0",
                ],
                json_payload={
                    "input": input_ids[0],
                    "output": tensor_id,
                    "shape": output_shape,
                    "in_shape": list(in_shape),
                    "kernel": [kernel_h, kernel_w],
                    "stride": [stride_h, stride_w],
                    "out_channels": int(output_shape[2]),
                    "channel_words": int(math.ceil(in_shape[2] / 64.0)),
                    "scratch_id": int(scratch_id),
                },
            )
            continue

        if class_name == "XNORBinaryDense":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            in_shape = builder.tensors[input_ids[0]].shape
            in_dim = int(in_shape[-1])
            outer_count = int(np.prod(in_shape[:-1], dtype=np.int64)) if len(in_shape) > 1 else 1
            scratch_shape = (outer_count, in_dim)
            scratch_id = builder.add_tensor(f"{layer.name}_packed", "KBNN_TENSOR_PACKED_BINARY", scratch_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
            alpha = np.mean(np.abs(kernel), axis=0)
            bias = np.asarray(layer.bias.numpy(), dtype=np.float32) if layer.bias is not None else np.zeros((layer.units,), dtype=np.float32)
            sign_bits = kernel >= 0.0
            packed = _pack_binary_dense_kernel(sign_bits)
            packed_name = builder.add_array(f"{model_name}_{layer.name}_weights_bits", "uint64_t", packed)
            alpha_name = builder.add_array(f"{model_name}_{layer.name}_alpha_q", "int32_t", _to_fx(alpha))
            bias_name = builder.add_array(f"{model_name}_{layer.name}_bias_q", "int32_t", _to_fx(bias))
            act_scale_i8, act_bias_i8 = _fold_xnor_activation_compare(layer)
            act_scale_name = builder.add_array(f"{model_name}_{layer.name}_act_scale_i8", "int8_t", act_scale_i8)
            act_bias_name = builder.add_array(f"{model_name}_{layer.name}_act_bias_i8", "int8_t", act_bias_i8)
            builder.add_op(
                "KBNN_OP_XNOR_DENSE",
                layer.name,
                "kbnn_xnor_dense_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".scratch_id = {scratch_id}",
                    f".outer_count = {outer_count}",
                    f".in_dim = {in_dim}",
                    f".out_dim = {int(output_shape[-1])}",
                    f".channel_words = {int(math.ceil(in_dim / 64.0))}",
                    f".last_word_mask = 0x{int((1 << (in_dim % 64)) - 1 if (in_dim % 64) else (2**64 - 1)):016x}ull",
                    f".weights_bits = {packed_name}",
                    f".alpha_q = {alpha_name}",
                    f".bias_q = {bias_name}",
                    f".act_scale_i8 = {act_scale_name}",
                    f".act_bias_i8 = {act_bias_name}",
                    f".input_cast_fp16 = 0",
                ],
                json_payload={
                    "input": input_ids[0],
                    "output": tensor_id,
                    "shape": output_shape,
                    "outer_count": int(outer_count),
                    "in_dim": int(in_dim),
                    "out_dim": int(output_shape[-1]),
                    "channel_words": int(math.ceil(in_dim / 64.0)),
                },
            )
            continue

        if class_name == "MaxPooling2D":
            raise NotImplementedError(
                f"MaxPooling2D export is no longer supported for kerasbnn BNN models. "
                f"Replace layer {layer.name!r} with stride-2 binary convolutions."
            )

        if class_name == "SpaceToDepth":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            in_shape = builder.tensors[input_ids[0]].shape
            builder.add_op(
                "KBNN_OP_SPACE_TO_DEPTH",
                layer.name,
                "kbnn_space_to_depth_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".in_h = {in_shape[0]}",
                    f".in_w = {in_shape[1]}",
                    f".channels = {in_shape[2]}",
                    f".block_size = {int(layer.block_size)}",
                ],
                json_payload={"input": input_ids[0], "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "Concatenate":
            axis = int(layer.axis)
            if axis < 0:
                axis += len(output_shape) + 1
            axis -= 1
            if axis != len(output_shape) - 1:
                raise NotImplementedError(f"Only channel-last concatenation is supported. Layer: {layer.name}")
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            ids_name = builder.add_array(f"{model_name}_{layer.name}_input_ids", "int32_t", np.asarray(input_ids, dtype=np.int32))
            builder.add_op(
                "KBNN_OP_CONCAT",
                layer.name,
                "kbnn_concat_params",
                [
                    f".output_id = {tensor_id}",
                    f".input_count = {len(input_ids)}",
                    f".input_ids = {ids_name}",
                    f".rank = {len(output_shape)}",
                    f".output_shape = {_emit_shape(output_shape)}",
                    f".axis = {axis}",
                ],
                json_payload={"inputs": input_ids, "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "Add":
            if len(input_ids) < 2:
                raise NotImplementedError(f"Add export requires at least two inputs. Layer: {layer.name}")
            input_meta = [builder.tensors[idx] for idx in input_ids]
            base_shape = tuple(input_meta[0].shape)
            if any(tuple(t.shape) != base_shape for t in input_meta[1:]) or tuple(output_shape) != base_shape:
                raise NotImplementedError(f"Add export requires identical input/output shapes. Layer: {layer.name}")
            if any(t.kind != "KBNN_TENSOR_I32" for t in input_meta):
                raise NotImplementedError(f"Add export only supports int32 tensors. Layer: {layer.name}")
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            ids_name = builder.add_array(f"{model_name}_{layer.name}_input_ids", "int32_t", np.asarray(input_ids, dtype=np.int32))
            builder.add_op(
                "KBNN_OP_ADD",
                layer.name,
                "kbnn_add_params",
                [
                    f".output_id = {tensor_id}",
                    f".input_count = {len(input_ids)}",
                    f".input_ids = {ids_name}",
                    f".rank = {len(output_shape)}",
                    f".shape = {_emit_shape(output_shape)}",
                ],
                json_payload={"inputs": input_ids, "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "UpSampling2D":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            in_shape = builder.tensors[input_ids[0]].shape
            scale_h, scale_w = layer.size
            builder.add_op(
                "KBNN_OP_UPSAMPLE2D",
                layer.name,
                "kbnn_upsample2d_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".in_h = {in_shape[0]}",
                    f".in_w = {in_shape[1]}",
                    f".channels = {in_shape[2]}",
                    f".scale_h = {scale_h}",
                    f".scale_w = {scale_w}",
                ],
                json_payload={"input": input_ids[0], "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "BinaryAffineNorm":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            mul_q, add_q = _fold_binary_affine(layer)
            mul_name = builder.add_array(f"{model_name}_{layer.name}_mul_q", "int32_t", mul_q)
            add_name = builder.add_array(f"{model_name}_{layer.name}_add_q", "int32_t", add_q)
            builder.add_op(
                "KBNN_OP_BINARY_AFFINE_NORM",
                layer.name,
                "kbnn_binary_affine_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".rank = {len(output_shape)}",
                    f".shape = {_emit_shape(output_shape)}",
                    f".mul_q = {mul_name}",
                    f".add_q = {add_name}",
                ],
                json_payload={"input": input_ids[0], "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "QuantizedActivation":
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            builder.add_op(
                "KBNN_OP_QUANT_ACT",
                layer.name,
                "kbnn_quant_act_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".rank = {len(output_shape)}",
                    f".shape = {_emit_shape(output_shape)}",
                    f".levels = {int(layer.levels)}",
                    f".signed_mode = {1 if layer.signed else 0}",
                ],
                json_payload={"input": input_ids[0], "output": tensor_id, "shape": output_shape},
            )
            continue

        if class_name == "QuantizedDense":
            fused_act = fused_quant_act_for_dense.get(layer.name)
            tensor_id = builder.add_tensor(layer.name, "KBNN_TENSOR_I32", output_shape)
            builder.tensor_by_layer[layer.name] = tensor_id
            kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
            if layer.integer_kernel:
                w_q = _quantize_signed_int_grid(kernel, layer.levels)
                weight_divisor = 1
            else:
                w_q = _quantize_signed_unit_grid(kernel, layer.levels)
                weight_divisor = int(layer.levels)
            w_name = builder.add_array(f"{model_name}_{layer.name}_weights", "int16_t", _transpose_qdense_kernel(w_q))
            if layer.bias is not None:
                bias = np.asarray(layer.bias.numpy(), dtype=np.float32)
                if layer.quantize_bias and layer.integer_output:
                    bias = _fake_quantize_signed_output(bias, layer.levels, layer.output_scale)
                b_name = builder.add_array(f"{model_name}_{layer.name}_bias_q", "int32_t", _to_fx(bias))
            else:
                b_name = "NULL"
            in_shape = builder.tensors[input_ids[0]].shape
            outer_count = int(np.prod(in_shape[:-1], dtype=np.int64)) if len(in_shape) > 1 else 1
            if fused_act is not None:
                builder.add_op(
                    "KBNN_OP_QACT_QDENSE",
                    layer.name,
                    "kbnn_qact_qdense_params",
                    [
                        f".input_id = {input_ids[0]}",
                        f".output_id = {tensor_id}",
                        f".outer_count = {outer_count}",
                        f".in_dim = {in_shape[-1]}",
                        f".out_dim = {output_shape[-1]}",
                        f".act_levels = {int(fused_act.levels)}",
                        f".act_signed_mode = {1 if fused_act.signed else 0}",
                        f".weight_divisor = {weight_divisor}",
                        f".levels = {int(layer.levels)}",
                        f".integer_output = {1 if layer.integer_output else 0}",
                        f".output_scale_q = {int(round(float(layer.output_scale) * FX_ONE))}",
                        f".weights = {w_name}",
                        f".bias_q = {b_name}",
                    ],
                    json_payload={
                        "input": input_ids[0],
                        "output": tensor_id,
                        "shape": output_shape,
                        "outer_count": int(outer_count),
                        "in_dim": int(in_shape[-1]),
                        "out_dim": int(output_shape[-1]),
                        "act_levels": int(fused_act.levels),
                        "weight_divisor": int(weight_divisor),
                    },
                )
                continue
            builder.add_op(
                "KBNN_OP_QDENSE",
                layer.name,
                "kbnn_qdense_params",
                [
                    f".input_id = {input_ids[0]}",
                    f".output_id = {tensor_id}",
                    f".outer_count = {outer_count}",
                    f".in_dim = {in_shape[-1]}",
                    f".out_dim = {output_shape[-1]}",
                    f".weight_divisor = {weight_divisor}",
                    f".levels = {int(layer.levels)}",
                    f".integer_output = {1 if layer.integer_output else 0}",
                    f".output_scale_q = {int(round(float(layer.output_scale) * FX_ONE))}",
                    f".weights = {w_name}",
                    f".bias_q = {b_name}",
                ],
                json_payload={
                    "input": input_ids[0],
                    "output": tensor_id,
                    "shape": output_shape,
                    "outer_count": int(outer_count),
                    "in_dim": int(in_shape[-1]),
                    "out_dim": int(output_shape[-1]),
                    "weight_divisor": int(weight_divisor),
                },
            )
            continue

        raise NotImplementedError(f"Unsupported layer for C export: {layer.name} ({class_name})")

    builder.output_tensor_ids = [builder.tensor_by_layer[_source_layer_name(t)] for t in model.outputs]
    return builder


def _render_array(array: ArrayDef) -> str:
    lines = [f"static const {array.c_type} {array.name}[] __attribute__((aligned(64))) = {{"]
    row = []
    for idx, value in enumerate(array.values):
        row.append(value)
        if len(row) == 8 or idx == len(array.values) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};")
    return "\n".join(lines)


def _render_header(builder: ExportBuilder) -> str:
    guard = f"{builder.model_name.upper()}_EXPORT_H".replace("-", "_")
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "kerasbnn.h"',
        "",
    ]
    for array in builder.arrays:
        lines.append(_render_array(array))
        lines.append("")
    for op in builder.ops:
        lines.append(f"static const {op.param_type} {op.param_name} = {{")
        for line in op.param_lines:
            lines.append(f"    {line},")
        lines.append("};")
        lines.append("")
    lines.append(f"static const kbnn_tensor_desc {builder.model_name}_tensors[] = {{")
    for tensor in builder.tensors:
        lines.append(
            "    {"
            f'.name = "{tensor.name}", '
            f".kind = {tensor.kind}, "
            f".rank = {tensor.rank}, "
            f".shape = {_emit_shape(tensor.shape)}, "
            f".words_per_point = {tensor.words_per_point}, "
            f".offset_bytes = {tensor.offset_bytes}, "
            f".bytes = {tensor.bytes}"
            "},"
        )
    lines.append("};")
    lines.append("")
    lines.append(f"static const kbnn_op {builder.model_name}_ops[] = {{")
    for op in builder.ops:
        lines.append(
            "    {"
            f".kind = {op.kind}, "
            f'.name = "{op.name}", '
            f".params = &{op.param_name}"
            "},"
        )
    lines.append("};")
    lines.append("")
    output_tensor_ids = []
    for out_tensor in model_outputs(builder):
        output_tensor_ids.append(out_tensor)
    lines.append(
        f"static const int32_t {builder.model_name}_output_ids[] = "
        "{"
        + ", ".join(str(v) for v in output_tensor_ids)
        + "};"
    )
    lines.append("")
    lines.append(f"static const kbnn_model {builder.model_name} = {{")
    lines.append(f'    .name = "{builder.model_name}",')
    lines.append(f"    .tensors = {builder.model_name}_tensors,")
    lines.append(f"    .tensor_count = {len(builder.tensors)},")
    lines.append(f"    .ops = {builder.model_name}_ops,")
    lines.append(f"    .op_count = {len(builder.ops)},")
    lines.append(f"    .output_ids = {builder.model_name}_output_ids,")
    lines.append(f"    .output_count = {len(output_tensor_ids)},")
    lines.append(f"    .arena_bytes = {builder.arena_offset},")
    lines.append("};")
    lines.append("")
    lines.append(f"#endif  /* {guard} */")
    return "\n".join(lines)


def model_outputs(builder: ExportBuilder) -> list[int]:
    if builder.output_tensor_ids:
        return list(builder.output_tensor_ids)
    return [len(builder.tensors) - 1]


def model_inputs(builder: ExportBuilder) -> list[int]:
    return [idx for idx, tensor in enumerate(builder.tensors) if tensor.kind == "KBNN_TENSOR_EXTERNAL_F32"]


def _render_json(builder: ExportBuilder) -> dict[str, Any]:
    return {
        "model_name": builder.model_name,
        "arena_bytes": builder.arena_offset,
        "tensors": [
            {
                "name": t.name,
                "kind": t.kind,
                "rank": t.rank,
                "shape": list(t.shape),
                "words_per_point": t.words_per_point,
                "offset_bytes": t.offset_bytes,
                "bytes": t.bytes,
            }
            for t in builder.tensors
        ],
        "ops": [
            {
                "kind": op.kind,
                "name": op.name,
                **op.json_payload,
            }
            for op in builder.ops
        ],
        "inputs": model_inputs(builder),
        "outputs": model_outputs(builder),
    }


def _render_runtime_source(builder: ExportBuilder) -> str:
    input_ids = model_inputs(builder)
    output_ids = model_outputs(builder)
    return "\n".join(
        [
            '#include "kerasbnn.h"',
            f'#include "{builder.model_name}_export.h"',
            "",
            f"static const int32_t {builder.model_name}_input_ids[] = "
            + "{"
            + ", ".join(str(v) for v in input_ids)
            + "};",
            "",
            f"int32_t {builder.model_name}_input_count(void) {{ return {len(input_ids)}; }}",
            f"int32_t {builder.model_name}_output_count(void) {{ return {len(output_ids)}; }}",
            f"size_t {builder.model_name}_arena_bytes(void) {{ return {builder.arena_offset}; }}",
            "",
            f"int {builder.model_name}_init(kbnn_context *ctx, void *arena, size_t arena_bytes) {{",
            f"    return kbnn_context_init(ctx, &{builder.model_name}, arena, arena_bytes);",
            "}",
            "",
            f"int32_t {builder.model_name}_input_tensor_id(int32_t index) {{",
            f"    if (index < 0 || index >= {len(input_ids)}) return -1;",
            f"    return {builder.model_name}_input_ids[index];",
            "}",
            "",
            f"int32_t {builder.model_name}_output_tensor_id(int32_t index) {{",
            f"    if (index < 0 || index >= {len(output_ids)}) return -1;",
            f"    return {builder.model_name}.output_ids[index];",
            "}",
            "",
            f"void {builder.model_name}_set_input_f32(kbnn_context *ctx, int32_t input_index, const float *input) {{",
            f"    if (input_index < 0 || input_index >= {len(input_ids)}) return;",
            f"    kbnn_context_set_input_f32(ctx, {builder.model_name}_input_ids[input_index], input);",
            "}",
            "",
            f"const int32_t *{builder.model_name}_output_i32(const kbnn_context *ctx, int32_t output_index) {{",
            f"    if (output_index < 0 || output_index >= {len(output_ids)}) return (const int32_t *)0;",
            f"    return kbnn_context_get_output_i32(ctx, {builder.model_name}.output_ids[output_index]);",
            "}",
            "",
            f"int {builder.model_name}_run(kbnn_context *ctx) {{",
            "    return kbnn_model_run(ctx);",
            "}",
            "",
        ]
    )


def _render_example_main(builder: ExportBuilder) -> str:
    input_ids = model_inputs(builder)
    input_tensors = [builder.tensors[idx] for idx in input_ids]
    lines = [
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        '#include "kerasbnn.h"',
        f'#include "{builder.model_name}_export.h"',
        "",
        f"int {builder.model_name}_init(kbnn_context *ctx, void *arena, size_t arena_bytes);",
        f"int32_t {builder.model_name}_output_count(void);",
        f"size_t {builder.model_name}_arena_bytes(void);",
        f"void {builder.model_name}_set_input_f32(kbnn_context *ctx, int32_t input_index, const float *input);",
        f"const int32_t *{builder.model_name}_output_i32(const kbnn_context *ctx, int32_t output_index);",
        f"int32_t {builder.model_name}_output_tensor_id(int32_t index);",
        f"int {builder.model_name}_run(kbnn_context *ctx);",
        "",
        "int main(void) {",
        f"    uint8_t *arena = (uint8_t *)calloc(1, {builder.model_name}_arena_bytes());",
        "    if (!arena) return 1;",
        "    kbnn_context ctx;",
        f"    if ({builder.model_name}_init(&ctx, arena, {builder.model_name}_arena_bytes()) != 0) {{",
        "        free(arena);",
        "        return 1;",
        "    }",
    ]
    for idx, tensor in enumerate(input_tensors):
        elems = _element_count(tensor.shape)
        lines.extend(
            [
                f"    float *input_{idx} = (float *)calloc({elems}, sizeof(float));",
                f"    if (!input_{idx}) {{ kbnn_context_release(&ctx); free(arena); return 1; }}",
                f"    {builder.model_name}_set_input_f32(&ctx, {idx}, input_{idx});",
            ]
        )
    lines.extend(
        [
            f"    if ({builder.model_name}_run(&ctx) != 0) {{",
            "        kbnn_context_release(&ctx);",
        ]
    )
    for idx, _ in enumerate(input_tensors):
        lines.append(f"        free(input_{idx});")
    lines.extend(
        [
            "        free(arena);",
            "        return 1;",
            "    }",
            f"    for (int32_t i = 0; i < {builder.model_name}_output_count(); ++i) {{",
            f"        const int32_t *out = {builder.model_name}_output_i32(&ctx, i);",
            f'        printf(\"output[%d] tensor_id=%d first_q12=%d\\n\", (int)i, (int){builder.model_name}_output_tensor_id(i), out ? (int)out[0] : 0);',
            "    }",
            "    kbnn_context_release(&ctx);",
        ]
    )
    for idx, _ in enumerate(input_tensors):
        lines.append(f"    free(input_{idx});")
    lines.extend(["    free(arena);", "    return 0;", "}", ""])
    return "\n".join(lines)


def export_model_to_c(
    model,
    output_dir: str | Path,
    model_name: str = "kerasbnn_model",
    array_overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    """Write the core export artifacts for a supported Keras BNN model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = lower_model(model, model_name=model_name, array_overrides=array_overrides)
    header_path = output_dir / f"{model_name}_export.h"
    json_path = output_dir / f"{model_name}_export.json"
    header_path.write_text(_render_header(builder) + "\n")
    json_path.write_text(json.dumps(_render_json(builder), indent=2, sort_keys=True) + "\n")
    return {"header": header_path, "json": json_path}


def export_model_bundle(
    model,
    output_dir: str | Path,
    model_name: str = "kerasbnn_model",
    array_overrides: dict[str, np.ndarray] | None = None,
    *,
    emit_runtime_c: bool = True,
    emit_example_main_c: bool = True,
) -> dict[str, Path]:
    """Write a complete C11-ready export bundle for a supported Keras BNN model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = lower_model(model, model_name=model_name, array_overrides=array_overrides)
    header_path = output_dir / f"{model_name}_export.h"
    json_path = output_dir / f"{model_name}_export.json"
    header_path.write_text(_render_header(builder) + "\n")
    json_path.write_text(json.dumps(_render_json(builder), indent=2, sort_keys=True) + "\n")
    artifacts = {"header": header_path, "json": json_path}
    if emit_runtime_c:
        runtime_path = output_dir / f"{model_name}_runtime.c"
        runtime_path.write_text(_render_runtime_source(builder))
        artifacts["runtime_c"] = runtime_path
    if emit_example_main_c:
        example_path = output_dir / f"{model_name}_example_main.c"
        example_path.write_text(_render_example_main(builder))
        artifacts["example_main_c"] = example_path
    return artifacts
