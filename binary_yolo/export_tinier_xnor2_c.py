"""Export a `tinier_yolo_xnor2` checkpoint as a generic `kerasbnn` C bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kerasbnn.export_c import export_model_bundle

from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2
from binary_yolo.yad2k_core import ANCHOR_MASKS, load_dataset


def _load_anchors_csv(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row.get("w", row.get("width"))
            h = row.get("h", row.get("height"))
            rows.append((float(w), float(h)))
    if not rows:
        raise ValueError(f"No anchors found in {path}")
    return rows


def _infer_anchors_csv(checkpoint: Path, explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    inferred = checkpoint.parent / "bbox_priors.csv"
    if inferred.exists():
        return inferred
    return None


def _escape_c_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _default_class_names(num_classes: int) -> list[str]:
    return [f"class_{idx}" for idx in range(num_classes)]


def _fmt_c_float(value: float) -> str:
    value = float(value)
    if float(value).is_integer():
        return f"{value:.1f}f"
    return f"{value:.9g}f"


def _load_class_names(dataset: str, dataset_format: str, num_classes: int) -> list[str]:
    if not dataset:
        return _default_class_names(num_classes)
    root = Path(dataset)
    if not root.exists():
        return _default_class_names(num_classes)
    _, classes = load_dataset(root, dataset_format=dataset_format)
    if len(classes) != num_classes:
        return _default_class_names(num_classes)
    return list(classes)


def _ordered_yolo_heads(meta: dict) -> list[dict]:
    output_ids = [int(v) for v in meta["outputs"]]
    if len(output_ids) > len(ANCHOR_MASKS):
        raise ValueError(f"Unsupported detector output count {len(output_ids)} for YOLO example config.")
    heads = []
    for output_index, tensor_id in enumerate(output_ids):
        tensor = meta["tensors"][tensor_id]
        shape = list(tensor["shape"])
        if len(shape) != 4:
            raise ValueError(f"YOLO example expects rank-4 outputs, got {shape!r} for tensor {tensor['name']}")
        heads.append(
            {
                "name": tensor["name"],
                "grid_h": int(shape[0]),
                "grid_w": int(shape[1]),
                "anchors_per_cell": int(shape[2]),
                "output_index": output_index,
            }
        )
    # Apply anchor masks from highest-resolution head to lowest-resolution head
    # so exported C decoding matches the native Python/Keras detector path even
    # when model output order differs from decode order.
    heads.sort(key=lambda head: (-head["grid_h"] * head["grid_w"], -head["grid_h"], -head["grid_w"], head["output_index"]))
    for head_idx, head in enumerate(heads):
        head["anchor_mask"] = list(ANCHOR_MASKS[head_idx])
    return heads


def _emit_yolo_config_header(
    output_dir: Path,
    model_name: str,
    meta: dict,
    input_size: int,
    anchors: list[tuple[float, float]],
    class_names: list[str],
    *,
    bbox_output_scale: float,
    obj_output_scale: float,
    cls_output_scale: float,
    logit_scale: float,
    score_threshold: float,
    objectness_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_mode: str,
    box_mode: str,
) -> Path:
    heads = _ordered_yolo_heads(meta)

    mode_class = {
        "bce": "KBNN_YOLO_CLASS_BCE",
        "softmax": "KBNN_YOLO_CLASS_SOFTMAX",
    }[class_mode]
    mode_box = {
        "linear": "KBNN_YOLO_BOX_LINEAR",
        "linear_float": "KBNN_YOLO_BOX_LINEAR_FLOAT",
        "yolo": "KBNN_YOLO_BOX_YOLO",
    }[box_mode]

    define_values = [
        (f"{model_name.upper()}_YOLO_INPUT_W", str(input_size)),
        (f"{model_name.upper()}_YOLO_INPUT_H", str(input_size)),
        (f"{model_name.upper()}_YOLO_NUM_CLASSES", str(len(class_names))),
        (f"{model_name.upper()}_YOLO_HEAD_COUNT", str(len(heads))),
        (f"{model_name.upper()}_YOLO_MAX_DETECTIONS", str(max_detections)),
        (f"{model_name.upper()}_YOLO_SCRATCH_DETECTIONS", "4096"),
        (f"{model_name.upper()}_YOLO_BBOX_OUTPUT_SCALE", _fmt_c_float(bbox_output_scale)),
        (f"{model_name.upper()}_YOLO_OBJ_OUTPUT_SCALE", _fmt_c_float(obj_output_scale)),
        (f"{model_name.upper()}_YOLO_CLS_OUTPUT_SCALE", _fmt_c_float(cls_output_scale)),
        (f"{model_name.upper()}_YOLO_LOGIT_SCALE", _fmt_c_float(logit_scale)),
        (f"{model_name.upper()}_YOLO_SCORE_THRESHOLD", _fmt_c_float(score_threshold)),
        (f"{model_name.upper()}_YOLO_OBJECTNESS_THRESHOLD", _fmt_c_float(objectness_threshold)),
        (f"{model_name.upper()}_YOLO_NMS_IOU_THRESHOLD", _fmt_c_float(iou_threshold)),
        (f"{model_name.upper()}_YOLO_PER_CLASS_NMS", "1"),
        (f"{model_name.upper()}_YOLO_CLASS_MODE", mode_class),
        (f"{model_name.upper()}_YOLO_BOX_MODE", mode_box),
    ]

    lines = [
        f"#ifndef {model_name.upper()}_YOLO_CONFIG_H",
        f"#define {model_name.upper()}_YOLO_CONFIG_H",
        "",
        '#include <stdint.h>',
        '#include "yolo_postprocess.h"',
        "",
        f"static const float {model_name}_yolo_anchors[] = {{",
    ]
    for name, value in define_values:
        lines.extend(
            [
                f"#ifndef {name}",
                f"#define {name} {value}",
                "#endif",
            ]
        )
    lines.append("")
    for aw, ah in anchors:
        lines.append(f"    {_fmt_c_float(aw)}, {_fmt_c_float(ah)},")
    lines.append("};")
    lines.append("")
    for idx, head in enumerate(heads):
        mask_values = ", ".join(str(v) for v in head["anchor_mask"][: head["anchors_per_cell"]])
        lines.append(
            f"static const int32_t {model_name}_yolo_head_{idx}_anchor_indices[] = {{{mask_values}}};"
        )
    lines.append("")
    lines.append(f"static const int32_t {model_name}_yolo_head_grid_h[] = " + "{" + ", ".join(str(head["grid_h"]) for head in heads) + "};")
    lines.append(f"static const int32_t {model_name}_yolo_head_grid_w[] = " + "{" + ", ".join(str(head["grid_w"]) for head in heads) + "};")
    lines.append(f"static const int32_t {model_name}_yolo_head_anchor_count[] = " + "{" + ", ".join(str(head["anchors_per_cell"]) for head in heads) + "};")
    lines.append(f"static const int32_t {model_name}_yolo_head_output_indices[] = " + "{" + ", ".join(str(head["output_index"]) for head in heads) + "};")
    lines.append(f"static const char *{model_name}_yolo_output_names[] = " + "{")
    for head in heads:
        lines.append(f'    "{_escape_c_string(head["name"])}",')
    lines.append("};")
    lines.append("")
    lines.append(f"static const char *{model_name}_yolo_class_names[] = " + "{")
    for class_name in class_names:
        lines.append(f'    "{_escape_c_string(class_name)}",')
    lines.append("};")
    lines.extend(
        [
            "",
            f"static const kbnn_yolo_postprocess_params {model_name}_yolo_params = {{",
            f"    .input_w = {model_name.upper()}_YOLO_INPUT_W,",
            f"    .input_h = {model_name.upper()}_YOLO_INPUT_H,",
            f"    .num_classes = {model_name.upper()}_YOLO_NUM_CLASSES,",
            f"    .bbox_output_scale = {model_name.upper()}_YOLO_BBOX_OUTPUT_SCALE,",
            f"    .obj_output_scale = {model_name.upper()}_YOLO_OBJ_OUTPUT_SCALE,",
            f"    .cls_output_scale = {model_name.upper()}_YOLO_CLS_OUTPUT_SCALE,",
            f"    .logit_scale = {model_name.upper()}_YOLO_LOGIT_SCALE,",
            f"    .score_threshold = {model_name.upper()}_YOLO_SCORE_THRESHOLD,",
            f"    .objectness_threshold = {model_name.upper()}_YOLO_OBJECTNESS_THRESHOLD,",
            f"    .nms_iou_threshold = {model_name.upper()}_YOLO_NMS_IOU_THRESHOLD,",
            f"    .max_detections = {model_name.upper()}_YOLO_MAX_DETECTIONS,",
            f"    .per_class_nms = {model_name.upper()}_YOLO_PER_CLASS_NMS,",
            f"    .class_mode = {model_name.upper()}_YOLO_CLASS_MODE,",
            f"    .box_mode = {model_name.upper()}_YOLO_BOX_MODE,",
            "};",
            "",
            "#endif",
            "",
        ]
    )
    path = output_dir / f"{model_name}_yolo_config.h"
    path.write_text("\n".join(lines))
    return path


def _emit_yolo_demo_c(output_dir: Path, model_name: str, head_count: int) -> Path:
    lines = [
        "/*",
        " * Example detector demo for a `kerasbnn` YOLO-style export.",
        " *",
        " * Build example:",
        f" *   clang -Ofast -march=native -std=c11 -Wall -Wextra -pedantic \\",
        f" *     -I../../kerasbnn -I../../kerasbnn/third_party/stb \\",
        f" *     {model_name}_runtime.c {model_name}_detector.c {model_name}_yolo_demo.c \\",
        " *     ../../kerasbnn/kerasbnn.c ../../kerasbnn/yolo_postprocess.c -lm -o yolo_demo",
        " */",
        "",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "",
        "#define STB_IMAGE_IMPLEMENTATION",
        '#include "stb_image.h"',
        f'#include "{model_name}_detector.h"',
        "",
        "int main(int argc, char **argv) {",
        "    int src_w = 0;",
        "    int src_h = 0;",
        "    int src_c = 0;",
        "    uint8_t *arena = NULL;",
        "    uint8_t *image = NULL;",
        f"    {model_name}_detector detector;",
        f"    kbnn_detection detections[{model_name.upper()}_YOLO_MAX_DETECTIONS];",
        "    int32_t det_count;",
        "    int i;",
        "    if (argc != 2) {",
        '        fprintf(stderr, "usage: %s image_path\\n", argv[0]);',
        "        return 1;",
        "    }",
        '    image = (uint8_t *) stbi_load(argv[1], &src_w, &src_h, &src_c, 3);',
        "    if (image == NULL) {",
        '        fprintf(stderr, "failed to load image: %s\\n", argv[1]);',
        "        return 1;",
        "    }",
        "    (void)src_c;",
        "    arena = (uint8_t *) calloc(1u, " + f"{model_name}_detector_arena_bytes()" + ");",
        "    if (arena == NULL) {",
        "        stbi_image_free(image);",
        "        free(arena);",
        "        return 1;",
        "    }",
        f"    if ({model_name}_detector_init(&detector, arena, {model_name}_detector_arena_bytes()) != 0) {{",
        "        stbi_image_free(image);",
        "        free(arena);",
        "        return 1;",
        "    }",
        f"    det_count = {model_name}_detector_run_rgb8_resize(",
        "        &detector,",
        "        image,",
        "        (int32_t)src_w,",
        "        (int32_t)src_h,",
        "        0,",
        "        detections,",
        f"        {model_name.upper()}_YOLO_MAX_DETECTIONS",
        "    );",
        "    stbi_image_free(image);",
        "    if (det_count < 0) {",
        f"        {model_name}_detector_release(&detector);",
        "        free(arena);",
        "        return 1;",
        "    }",
            '    printf("detections=%d\\n", (int)det_count);',
            "    for (i = 0; i < det_count; ++i) {",
            "        const kbnn_detection *det = &detections[i];",
            "        const float x1 = (det->cx - 0.5f * det->w) * (float)src_w;",
            "        const float y1 = (det->cy - 0.5f * det->h) * (float)src_h;",
            "        const float x2 = (det->cx + 0.5f * det->w) * (float)src_w;",
            "        const float y2 = (det->cy + 0.5f * det->h) * (float)src_h;",
            f"        const char *name = {model_name}_detector_class_name(det->class_id);",
            '        printf("%s score=%.4f box=[%.1f %.1f %.1f %.1f]\\n", name, det->score, x1, y1, x2, y2);',
            "    }",
            f"    {model_name}_detector_release(&detector);",
            "    free(arena);",
            "    return 0;",
            "}",
            "",
    ]
    path = output_dir / f"{model_name}_yolo_demo.c"
    path.write_text("\n".join(lines))
    return path


def _emit_yolo_detector_h(output_dir: Path, model_name: str) -> Path:
    guard = f"{model_name.upper()}_DETECTOR_H"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        '#include "kerasbnn.h"',
        '#include "yolo_postprocess.h"',
        f'#include "{model_name}_yolo_config.h"',
        "",
        "/*",
        " * Minimal deployment API for the exported YOLO detector.",
        " *",
        " * Input contract:",
        " * - `run_f32` expects a contiguous float32 HWC RGB image in [0, 1]",
        " * - `run_rgb8` expects a contiguous uint8 HWC RGB image at model size",
        " * - `run_rgb8_resize` accepts arbitrary RGB size and resizes internally",
        " *",
        " * Runtime contract:",
        " * - caller owns the arena buffer",
        " * - caller owns the output detection buffer",
        " * - returned value is detection count on success, negative on error",
        " */",
        "",
        "typedef struct {",
        "    kbnn_context ctx;",
        f"    kbnn_yolo_head heads[{model_name.upper()}_YOLO_HEAD_COUNT];",
        f"    kbnn_detection scratch[{model_name.upper()}_YOLO_SCRATCH_DETECTIONS];",
        f"    int32_t keep_indices[{model_name.upper()}_YOLO_SCRATCH_DETECTIONS];",
        f"    float input_rgb_f32[{model_name.upper()}_YOLO_INPUT_W * {model_name.upper()}_YOLO_INPUT_H * 3];",
        f"}} {model_name}_detector;",
        "",
        f"size_t {model_name}_detector_arena_bytes(void);",
        f"int32_t {model_name}_detector_input_w(void);",
        f"int32_t {model_name}_detector_input_h(void);",
        f"int32_t {model_name}_detector_num_classes(void);",
        f'const char *{model_name}_detector_class_name(int32_t class_id);',
        f"const kbnn_yolo_postprocess_params *{model_name}_detector_default_params(void);",
        f"int {model_name}_detector_init({model_name}_detector *detector, void *arena, size_t arena_bytes);",
        f"void {model_name}_detector_release({model_name}_detector *detector);",
        f"int32_t {model_name}_detector_run_f32(",
        f"    {model_name}_detector *detector,",
        "    const float *input_rgb,",
        "    kbnn_detection *out_detections,",
        "    int32_t out_capacity",
        ");",
        f"int32_t {model_name}_detector_run_f32_with_params(",
        f"    {model_name}_detector *detector,",
        "    const float *input_rgb,",
        "    const kbnn_yolo_postprocess_params *params,",
        "    kbnn_detection *out_detections,",
        "    int32_t out_capacity",
        ");",
        f"int32_t {model_name}_detector_run_rgb8(",
        f"    {model_name}_detector *detector,",
        "    const uint8_t *input_rgb_u8,",
        "    kbnn_detection *out_detections,",
        "    int32_t out_capacity",
        ");",
        f"int32_t {model_name}_detector_run_rgb8_with_params(",
        f"    {model_name}_detector *detector,",
        "    const uint8_t *input_rgb_u8,",
        "    const kbnn_yolo_postprocess_params *params,",
        "    kbnn_detection *out_detections,",
        "    int32_t out_capacity",
        ");",
        f"int32_t {model_name}_detector_run_rgb8_resize(",
        f"    {model_name}_detector *detector,",
        "    const uint8_t *input_rgb_u8,",
        "    int32_t src_w,",
        "    int32_t src_h,",
        "    int32_t src_stride_bytes,",
        "    kbnn_detection *out_detections,",
        "    int32_t out_capacity",
        ");",
        f"int32_t {model_name}_detector_run_rgb8_resize_with_params(",
        f"    {model_name}_detector *detector,",
        "    const uint8_t *input_rgb_u8,",
        "    int32_t src_w,",
        "    int32_t src_h,",
        "    int32_t src_stride_bytes,",
        "    const kbnn_yolo_postprocess_params *params,",
        "    kbnn_detection *out_detections,",
        "    int32_t out_capacity",
        ");",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    path = output_dir / f"{model_name}_detector.h"
    path.write_text("\n".join(lines))
    return path


def _emit_yolo_detector_c(output_dir: Path, model_name: str, head_count: int, first_input_op: dict | None) -> Path:
    direct_thermo = bool(first_input_op and first_input_op.get("kind") == "KBNN_OP_THERMO_INPUT")
    thermo_param_name = f"{model_name}_{first_input_op['name']}_params" if direct_thermo else ""
    thermo_output_id = int(first_input_op["output"]) if direct_thermo else -1
    run_after_input = 1 if direct_thermo else 0
    lines = [
        f'#include "{model_name}_detector.h"',
        f'#include "{model_name}_export.h"',
        "",
    ]
    if not direct_thermo:
        lines.extend(
            [
                "static void kbnn_detector_u8_to_f32(const uint8_t *src, float *dst, size_t count) {",
                "    size_t i;",
                "    for (i = 0; i < count; ++i) {",
                "        dst[i] = (float)src[i] * (1.0f / 255.0f);",
                "    }",
                "}",
                "",
                "static void kbnn_detector_resize_rgb8_to_f32(",
                "    const uint8_t *src,",
                "    int32_t src_w,",
                "    int32_t src_h,",
                "    int32_t src_stride_bytes,",
                "    float *dst,",
                "    int32_t dst_w,",
                "    int32_t dst_h",
                ") {",
                "    int32_t y;",
                "    if (src_stride_bytes <= 0) {",
                "        src_stride_bytes = src_w * 3;",
                "    }",
                "    for (y = 0; y < dst_h; ++y) {",
                "        int32_t x;",
                "        const int32_t sy = (y * src_h) / dst_h;",
                "        for (x = 0; x < dst_w; ++x) {",
                "            const int32_t sx = (x * src_w) / dst_w;",
                "            const uint8_t *pixel = src + (size_t)sy * (size_t)src_stride_bytes + (size_t)sx * 3u;",
                "            float *out = dst + ((size_t)y * (size_t)dst_w + (size_t)x) * 3u;",
                "            out[0] = (float)pixel[0] * (1.0f / 255.0f);",
                "            out[1] = (float)pixel[1] * (1.0f / 255.0f);",
                "            out[2] = (float)pixel[2] * (1.0f / 255.0f);",
                "        }",
                "    }",
                "}",
                "",
            ]
        )
    lines.extend(
        [
            f"int {model_name}_init(kbnn_context *ctx, void *arena, size_t arena_bytes);",
            f"size_t {model_name}_arena_bytes(void);",
            f"void {model_name}_set_input_f32(kbnn_context *ctx, int32_t input_index, const float *input);",
            f"const int32_t *{model_name}_output_i32(const kbnn_context *ctx, int32_t output_index);",
            f"int {model_name}_run(kbnn_context *ctx);",
            "int kbnn_model_run_from(kbnn_context *ctx, int32_t op_start);",
            "",
            f"size_t {model_name}_detector_arena_bytes(void) {{",
            f"    return {model_name}_arena_bytes();",
            "}",
            "",
            f"int32_t {model_name}_detector_input_w(void) {{",
            f"    return {model_name.upper()}_YOLO_INPUT_W;",
            "}",
            "",
            f"int32_t {model_name}_detector_input_h(void) {{",
            f"    return {model_name.upper()}_YOLO_INPUT_H;",
            "}",
            "",
            f"int32_t {model_name}_detector_num_classes(void) {{",
            f"    return {model_name.upper()}_YOLO_NUM_CLASSES;",
            "}",
            "",
            f"const char *{model_name}_detector_class_name(int32_t class_id) {{",
            f"    if (class_id < 0 || class_id >= {model_name.upper()}_YOLO_NUM_CLASSES) {{",
            '        return "unknown";',
            "    }",
            f"    return {model_name}_yolo_class_names[class_id];",
            "}",
            "",
            f"const kbnn_yolo_postprocess_params *{model_name}_detector_default_params(void) {{",
            f"    return &{model_name}_yolo_params;",
            "}",
            "",
            f"int {model_name}_detector_init({model_name}_detector *detector, void *arena, size_t arena_bytes) {{",
            "    if (detector == NULL || arena == NULL) {",
            "        return -1;",
            "    }",
        ]
    )
    for idx in range(head_count):
        lines.extend(
            [
                f"    detector->heads[{idx}].pred_q12 = NULL;",
                f"    detector->heads[{idx}].grid_h = {model_name}_yolo_head_grid_h[{idx}];",
                f"    detector->heads[{idx}].grid_w = {model_name}_yolo_head_grid_w[{idx}];",
                f"    detector->heads[{idx}].anchors_per_cell = {model_name}_yolo_head_anchor_count[{idx}];",
                f"    detector->heads[{idx}].anchor_indices = {model_name}_yolo_head_{idx}_anchor_indices;",
            ]
        )
    lines.extend(
        [
            f"    return {model_name}_init(&detector->ctx, arena, arena_bytes);",
            "}",
            "",
            f"void {model_name}_detector_release({model_name}_detector *detector) {{",
            "    if (detector == NULL) {",
            "        return;",
            "    }",
            "    kbnn_context_release(&detector->ctx);",
            "}",
            "",
            f"int32_t {model_name}_detector_run_f32_with_params(",
            f"    {model_name}_detector *detector,",
            "    const float *input_rgb,",
            "    const kbnn_yolo_postprocess_params *params,",
            "    kbnn_detection *out_detections,",
            "    int32_t out_capacity",
            ") {",
            "    int32_t i;",
            "    if (detector == NULL || input_rgb == NULL || out_detections == NULL || out_capacity <= 0) {",
            "        return -1;",
            "    }",
            f"    {model_name}_set_input_f32(&detector->ctx, 0, input_rgb);",
            f"    if ({model_name}_run(&detector->ctx) != 0) {{",
            "        return -2;",
            "    }",
            f"    for (i = 0; i < {head_count}; ++i) {{",
            f"        detector->heads[i].pred_q12 = {model_name}_output_i32(&detector->ctx, {model_name}_yolo_head_output_indices[i]);",
            "    }",
            "    return kbnn_yolo_postprocess(",
            "        detector->heads,",
            f"        {head_count},",
            f"        {model_name}_yolo_anchors,",
            f"        (int32_t)(sizeof({model_name}_yolo_anchors) / sizeof({model_name}_yolo_anchors[0]) / 2u),",
            f"        (params != NULL) ? params : &{model_name}_yolo_params,",
            "        detector->scratch,",
            f"        {model_name.upper()}_YOLO_SCRATCH_DETECTIONS,",
            "        detector->keep_indices,",
            f"        {model_name.upper()}_YOLO_SCRATCH_DETECTIONS,",
            "        out_detections,",
            "        out_capacity",
            "    );",
            "}",
            "",
            f"int32_t {model_name}_detector_run_rgb8_with_params(",
            f"    {model_name}_detector *detector,",
            "    const uint8_t *input_rgb_u8,",
            "    const kbnn_yolo_postprocess_params *params,",
            "    kbnn_detection *out_detections,",
            "    int32_t out_capacity",
            ") {",
            "    int32_t i;",
            "    if (detector == NULL || input_rgb_u8 == NULL || out_detections == NULL || out_capacity <= 0) {",
            "        return -1;",
            "    }",
        ]
    )
    if direct_thermo:
        lines.extend(
            [
                f"    kbnn_pack_thermo_input_u8(&{thermo_param_name}, input_rgb_u8, {model_name.upper()}_YOLO_INPUT_W * 3,",
                f"        (uint64_t *)kbnn_context_tensor_ptr(&detector->ctx, {thermo_output_id}));",
                f"    if (kbnn_model_run_from(&detector->ctx, {run_after_input}) != 0) {{",
                "        return -2;",
                "    }",
            ]
        )
    else:
        lines.extend(
            [
                "    kbnn_detector_u8_to_f32(",
                "        input_rgb_u8,",
                "        detector->input_rgb_f32,",
                f"        (size_t){model_name.upper()}_YOLO_INPUT_W * (size_t){model_name.upper()}_YOLO_INPUT_H * 3u",
                "    );",
                f"    return {model_name}_detector_run_f32_with_params(",
                "        detector,",
                "        detector->input_rgb_f32,",
                "        params,",
                "        out_detections,",
                "        out_capacity",
                "    );",
                "}",
                "",
            ]
        )
    if direct_thermo:
        lines.extend(
            [
                f"    for (i = 0; i < {head_count}; ++i) {{",
                f"        detector->heads[i].pred_q12 = {model_name}_output_i32(&detector->ctx, {model_name}_yolo_head_output_indices[i]);",
                "    }",
                "    return kbnn_yolo_postprocess(",
                "        detector->heads,",
                f"        {head_count},",
                f"        {model_name}_yolo_anchors,",
                f"        (int32_t)(sizeof({model_name}_yolo_anchors) / sizeof({model_name}_yolo_anchors[0]) / 2u),",
                f"        (params != NULL) ? params : &{model_name}_yolo_params,",
                "        detector->scratch,",
                f"        {model_name.upper()}_YOLO_SCRATCH_DETECTIONS,",
                "        detector->keep_indices,",
                f"        {model_name.upper()}_YOLO_SCRATCH_DETECTIONS,",
                "        out_detections,",
                "        out_capacity",
                "    );",
                "}",
                "",
            ]
        )

    lines.extend(
        [
            f"int32_t {model_name}_detector_run_rgb8(",
            f"    {model_name}_detector *detector,",
            "    const uint8_t *input_rgb_u8,",
            "    kbnn_detection *out_detections,",
            "    int32_t out_capacity",
            ") {",
            f"    return {model_name}_detector_run_rgb8_with_params(",
            "        detector,",
            "        input_rgb_u8,",
            f"        &{model_name}_yolo_params,",
            "        out_detections,",
            "        out_capacity",
            "    );",
            "}",
            "",
            f"int32_t {model_name}_detector_run_rgb8_resize_with_params(",
            f"    {model_name}_detector *detector,",
            "    const uint8_t *input_rgb_u8,",
            "    int32_t src_w,",
            "    int32_t src_h,",
            "    int32_t src_stride_bytes,",
            "    const kbnn_yolo_postprocess_params *params,",
            "    kbnn_detection *out_detections,",
            "    int32_t out_capacity",
            ") {",
            "    int32_t i;",
            "    if (detector == NULL || input_rgb_u8 == NULL || src_w <= 0 || src_h <= 0 || out_detections == NULL || out_capacity <= 0) {",
            "        return -1;",
            "    }",
        ]
    )
    if direct_thermo:
        lines.extend(
            [
                f"    kbnn_pack_thermo_input_u8_resize(&{thermo_param_name}, input_rgb_u8, src_w, src_h, src_stride_bytes,",
                f"        (uint64_t *)kbnn_context_tensor_ptr(&detector->ctx, {thermo_output_id}));",
                f"    if (kbnn_model_run_from(&detector->ctx, {run_after_input}) != 0) {{",
                    "        return -2;",
                "    }",
                f"    for (i = 0; i < {head_count}; ++i) {{",
                f"        detector->heads[i].pred_q12 = {model_name}_output_i32(&detector->ctx, {model_name}_yolo_head_output_indices[i]);",
                "    }",
                "    return kbnn_yolo_postprocess(",
                "        detector->heads,",
                f"        {head_count},",
                f"        {model_name}_yolo_anchors,",
                f"        (int32_t)(sizeof({model_name}_yolo_anchors) / sizeof({model_name}_yolo_anchors[0]) / 2u),",
                f"        (params != NULL) ? params : &{model_name}_yolo_params,",
                "        detector->scratch,",
                f"        {model_name.upper()}_YOLO_SCRATCH_DETECTIONS,",
                "        detector->keep_indices,",
                f"        {model_name.upper()}_YOLO_SCRATCH_DETECTIONS,",
                "        out_detections,",
                "        out_capacity",
                "    );",
                "}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "    kbnn_detector_resize_rgb8_to_f32(",
                "        input_rgb_u8,",
                "        src_w,",
                "        src_h,",
                "        src_stride_bytes,",
                "        detector->input_rgb_f32,",
                f"        {model_name.upper()}_YOLO_INPUT_W,",
                f"        {model_name.upper()}_YOLO_INPUT_H",
                "    );",
                f"    return {model_name}_detector_run_f32_with_params(",
                "        detector,",
                "        detector->input_rgb_f32,",
                "        params,",
                "        out_detections,",
                "        out_capacity",
                "    );",
                "}",
                "",
            ]
        )
    lines.extend(
        [
            f"int32_t {model_name}_detector_run_rgb8_resize(",
            f"    {model_name}_detector *detector,",
            "    const uint8_t *input_rgb_u8,",
            "    int32_t src_w,",
            "    int32_t src_h,",
            "    int32_t src_stride_bytes,",
            "    kbnn_detection *out_detections,",
            "    int32_t out_capacity",
            ") {",
            f"    return {model_name}_detector_run_rgb8_resize_with_params(",
            "        detector,",
            "        input_rgb_u8,",
            "        src_w,",
            "        src_h,",
            "        src_stride_bytes,",
            f"        &{model_name}_yolo_params,",
            "        out_detections,",
            "        out_capacity",
            "    );",
            "}",
            "",
            f"int32_t {model_name}_detector_run_f32(",
            f"    {model_name}_detector *detector,",
            "    const float *input_rgb,",
            "    kbnn_detection *out_detections,",
            "    int32_t out_capacity",
            ") {",
            f"    return {model_name}_detector_run_f32_with_params(",
            "        detector,",
            "        input_rgb,",
            f"        &{model_name}_yolo_params,",
            "        out_detections,",
            "        out_capacity",
            "    );",
            "}",
            "",
        ]
    )
    path = output_dir / f"{model_name}_detector.c"
    path.write_text("\n".join(lines))
    return path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="tinier_xnor2_export")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--input-levels", type=int, default=16)
    ap.add_argument("--quantized-input", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tinyv3-backbone", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--input-dropcut", type=float, default=0.0)
    ap.add_argument("--input-dropcut-size-min", type=float, default=0.05)
    ap.add_argument("--input-dropcut-size-max", type=float, default=0.30)
    ap.add_argument("--input-dropcut-method", default="both", choices=("black", "white", "both", "noise"))
    ap.add_argument("--input-dropcut-boxes", type=int, default=1)
    ap.add_argument("--skip-mismatch", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--dataset-format", default="auto")
    ap.add_argument("--anchors-csv", default="")
    ap.add_argument("--box-mode", default="linear", choices=("linear", "linear_float", "yolo"))
    ap.add_argument("--class-mode", default="softmax", choices=("softmax", "bce"))
    ap.add_argument("--bbox-output-scale", type=float, default=2.0)
    ap.add_argument("--obj-output-scale", type=float, default=64.0)
    ap.add_argument("--cls-output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--score-threshold", type=float, default=0.001)
    ap.add_argument("--objectness-threshold", type=float, default=0.001)
    ap.add_argument("--iou-threshold", type=float, default=0.35)
    ap.add_argument("--max-detections", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    cfg = TinierXNOR2Config(
        thermo_levels=args.thermo_levels,
        input_levels=args.input_levels,
        quantized_input=args.quantized_input,
        input_dropcut=args.input_dropcut,
        input_dropcut_size=(args.input_dropcut_size_min, args.input_dropcut_size_max),
        input_dropcut_method=args.input_dropcut_method,
        input_dropcut_boxes=args.input_dropcut_boxes,
        tinyv3_backbone=args.tinyv3_backbone,
    )
    model = build_binary_tinier_yolov3_xnor2(
        args.num_classes,
        input_shape=(args.input_size, args.input_size, 3),
        cfg=cfg,
    )
    model.load_weights(checkpoint_path, skip_mismatch=bool(args.skip_mismatch))
    output_dir = Path(args.output_dir)
    outputs = export_model_bundle(
        model,
        output_dir,
        model_name=args.model_name,
    )
    meta = json.loads(outputs["json"].read_text())
    anchors_path = _infer_anchors_csv(checkpoint_path, args.anchors_csv)
    if anchors_path is not None:
        anchors = _load_anchors_csv(anchors_path)
        class_names = _load_class_names(args.dataset, args.dataset_format, args.num_classes)
        outputs["yolo_config_h"] = _emit_yolo_config_header(
            output_dir,
            args.model_name,
            meta,
            args.input_size,
            anchors,
            class_names,
            bbox_output_scale=args.bbox_output_scale,
            obj_output_scale=args.obj_output_scale,
            cls_output_scale=args.cls_output_scale,
            logit_scale=args.logit_scale,
            score_threshold=args.score_threshold,
            objectness_threshold=args.objectness_threshold,
            iou_threshold=args.iou_threshold,
            max_detections=args.max_detections,
            class_mode=args.class_mode,
            box_mode=args.box_mode,
        )
        outputs["yolo_demo_c"] = _emit_yolo_demo_c(output_dir, args.model_name, len(meta["outputs"]))
        outputs["detector_h"] = _emit_yolo_detector_h(output_dir, args.model_name)
        outputs["detector_c"] = _emit_yolo_detector_c(
            output_dir,
            args.model_name,
            len(meta["outputs"]),
            meta["ops"][0] if meta.get("ops") else None,
        )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    if anchors_path is not None:
        print(f"anchors_csv: {anchors_path}")


if __name__ == "__main__":
    main()
