from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")

sys.path.insert(0, "/home/claude/source/yolovxnor")

from kerasbnn.export_c import export_model_to_c
from kerasbnn.export_calibration import build_export_array_overrides
from kerasbnn.eval_fruit_codegen_parity import (
    DEFAULT_RUN_DIR,
    _build_images,
    _compile_shared_export,
    _load_anchors_csv,
    _manual_load_named_weights,
    _map50,
    _predict_c_runtime,
    _predict_named,
)
from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2
from binary_yolo.train_tinier import _prepare_split_samples


REPO_ROOT = Path("/home/claude/source/yolovxnor")


def _op_target_summary(op: dict) -> dict:
    summary = {
        "name": op.get("name"),
        "kind": op.get("kind"),
        "shape": op.get("shape"),
    }
    for key in (
        "in_shape",
        "kernel",
        "stride",
        "out_channels",
        "channel_words",
        "outer_count",
        "in_dim",
        "out_dim",
        "levels",
        "act_levels",
        "weight_divisor",
        "input_is_packed_binary",
    ):
        if key in op:
            summary[key] = op[key]
    return summary


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark exported C runtime for the best fruits XNOR2 model.")
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
    ap.add_argument("--subset-train-as-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--score-threshold", type=float, default=0.001)
    ap.add_argument("--objectness-threshold", type=float, default=0.001)
    ap.add_argument("--bbox-output-scale", type=float, default=2.0)
    ap.add_argument("--obj-output-scale", type=float, default=64.0)
    ap.add_argument("--cls-output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--class-mode", default="softmax", choices=("softmax", "bce"))
    ap.add_argument("--box-mode", default="linear", choices=("linear", "yolo"))
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--single-image-iterations", type=int, default=200)
    ap.add_argument("--val-iterations", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cc", default="clang")
    ap.add_argument("--calibrate-export", default="auto", choices=("auto", "none", "perlin_compare"))
    ap.add_argument("--calibrate-images", type=int, default=4)
    ap.add_argument("--calibrate-seed", type=int, default=43)
    return ap.parse_args()


def _build_model(args, num_classes: int):
    cfg = TinierXNOR2Config(
        thermo_levels=args.thermo_levels,
        input_levels=args.input_levels,
        quantized_input=args.quantized_input,
        tinyv3_backbone=args.tinyv3_backbone,
    )
    return build_binary_tinier_yolov3_xnor2(
        num_classes,
        input_shape=(args.input_size, args.input_size, 3),
        cfg=cfg,
    )


def _compile_bench_export(
    output_dir: Path,
    model_name: str,
    arena_bytes: int,
    output_ids: list[int],
    counts: list[int],
    input_numel: int,
    cc: str,
    *,
    enable_profile: bool = False,
) -> Path:
    out_args = ", ".join([f"int32_t *out{i}" for i in range(len(output_ids))])
    copies = "\n".join(
        [
            f"        memcpy(out{i}, kbnn_context_get_output_i32(&ctx, {tensor_id}), sizeof(int32_t) * {count});"
            for i, (tensor_id, count) in enumerate(zip(output_ids, counts))
        ]
    )
    wrapper = output_dir / f"{model_name}_bench_wrapper.c"
    wrapper.write_text(
        textwrap.dedent(
            f"""
            #define _POSIX_C_SOURCE 199309L
            #include <stdint.h>
            #include <stdio.h>
            #include <string.h>
            #include <time.h>
            #include "kerasbnn.h"
            #include "{model_name}_export.h"

            static double kbnn_now_seconds(void) {{
                struct timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
            }}

            int run_model(const float *input, {out_args}) {{
                static unsigned char arena[{arena_bytes}] __attribute__((aligned(64)));
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

            double bench_single_image(const float *input, int iterations, int32_t *checksum) {{
                static unsigned char arena[{arena_bytes}] __attribute__((aligned(64)));
                kbnn_context ctx;
                int rc = kbnn_context_init(&ctx, &{model_name}, arena, sizeof(arena));
                if (rc != 0) return -1.0;
                double start = kbnn_now_seconds();
                int32_t sum = 0;
                for (int i = 0; i < iterations; ++i) {{
                    kbnn_context_set_input_f32(&ctx, 0, input);
                    rc = kbnn_model_run(&ctx);
                    if (rc != 0) {{
                        kbnn_context_release(&ctx);
                        return -2.0;
                    }}
                    const int32_t *out0 = kbnn_context_get_output_i32(&ctx, {output_ids[0]});
                    sum += out0[0];
                }}
                double end = kbnn_now_seconds();
                kbnn_context_release(&ctx);
                *checksum = sum;
                return end - start;
            }}

            double bench_image_set(const float *inputs, int image_count, int iterations, int32_t *checksum) {{
                static unsigned char arena[{arena_bytes}] __attribute__((aligned(64)));
                kbnn_context ctx;
                int rc = kbnn_context_init(&ctx, &{model_name}, arena, sizeof(arena));
                if (rc != 0) return -1.0;
                double start = kbnn_now_seconds();
                int32_t sum = 0;
                for (int it = 0; it < iterations; ++it) {{
                    for (int img = 0; img < image_count; ++img) {{
                        const float *input = inputs + (size_t)img * (size_t){input_numel};
                        kbnn_context_set_input_f32(&ctx, 0, input);
                        rc = kbnn_model_run(&ctx);
                        if (rc != 0) {{
                            kbnn_context_release(&ctx);
                            return -2.0;
                        }}
                        const int32_t *out0 = kbnn_context_get_output_i32(&ctx, {output_ids[0]});
                        sum += out0[0];
                    }}
                }}
                double end = kbnn_now_seconds();
                kbnn_context_release(&ctx);
                *checksum = sum;
                return end - start;
            }}
            """
        ).strip()
        + "\n"
    )
    lib_suffix = "_bench_profile" if enable_profile else "_bench"
    lib_path = output_dir / f"lib{model_name}{lib_suffix}.so"
    cmd = [
        cc,
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
    if enable_profile:
        cmd.insert(1, "-DKBNN_ENABLE_PROFILE")
    subprocess.run(cmd, check=True)
    return lib_path


class KbnnProfile(ctypes.Structure):
    _fields_ = [
        ("total_ns", ctypes.c_uint64),
        ("op_ns", ctypes.c_uint64 * 512),
        ("op_calls", ctypes.c_uint64 * 512),
        ("thermo_input_ns", ctypes.c_uint64),
        ("quantized_input_ns", ctypes.c_uint64),
        ("qconv2d_ns", ctypes.c_uint64),
        ("xnor_conv_prep_ns", ctypes.c_uint64),
        ("xnor_conv_dot_ns", ctypes.c_uint64),
        ("xnor_dense_prep_ns", ctypes.c_uint64),
        ("xnor_dense_dot_ns", ctypes.c_uint64),
        ("space_to_depth_ns", ctypes.c_uint64),
        ("concat_ns", ctypes.c_uint64),
        ("upsample2d_ns", ctypes.c_uint64),
        ("binary_affine_ns", ctypes.c_uint64),
        ("quant_act_ns", ctypes.c_uint64),
        ("qdense_ns", ctypes.c_uint64),
        ("qact_qdense_ns", ctypes.c_uint64),
    ]


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
        test_sample_fraction=1.0,
        train_sample_count=0,
        subset_train_as_val=args.subset_train_as_val,
    )
    _, val_samples, _, classes = _prepare_split_samples(split_args, args.input_size)
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]
    val_images = _build_images(val_samples, args.input_size, args.thermo_levels)

    model = _build_model(args, len(classes))
    _manual_load_named_weights(model, weights_path)

    keras_preds = _predict_named(model, val_images, args.batch_size, training=False)
    keras_map50 = _map50(val_samples, keras_preds, anchors, args, len(classes))
    array_overrides, calibration_meta = build_export_array_overrides(
        model,
        model_name="fruit_bench_model",
        mode=args.calibrate_export,
        dataset=args.dataset,
        checkpoint=str(weights_path),
        calib_images=args.calibrate_images,
        seed=args.calibrate_seed,
    )

    with tempfile.TemporaryDirectory(prefix="kerasbnn_fruit_bench_") as td:
        export_dir = Path(td)
        outputs = export_model_to_c(
            model,
            export_dir,
            model_name="fruit_bench_model",
            array_overrides=array_overrides,
        )
        meta = json.loads(outputs["json"].read_text())
        output_ids = [int(v) for v in meta["outputs"]]
        counts = [int(np.prod(meta["tensors"][tensor_id]["shape"])) for tensor_id in output_ids]
        input_numel = int(np.prod(meta["tensors"][0]["shape"]))

        lib_path = _compile_bench_export(
            export_dir,
            "fruit_bench_model",
            int(meta["arena_bytes"]),
            output_ids,
            counts,
            input_numel,
            args.cc,
        )
        profile_lib_path = _compile_bench_export(
            export_dir,
            "fruit_bench_model",
            int(meta["arena_bytes"]),
            output_ids,
            counts,
            input_numel,
            args.cc,
            enable_profile=True,
        )

        c_preds, _ = _predict_c_runtime(
            model,
            val_images,
            export_dir,
            "fruit_bench_model",
            array_overrides=array_overrides,
        )
        c_map50 = _map50(val_samples, c_preds, anchors, args, len(classes))

        dll = ctypes.CDLL(str(lib_path))
        profile_dll = ctypes.CDLL(str(profile_lib_path))
        dll.bench_single_image.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.bench_single_image.restype = ctypes.c_double
        dll.bench_image_set.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.bench_image_set.restype = ctypes.c_double
        profile_dll.bench_single_image.argtypes = dll.bench_single_image.argtypes
        profile_dll.bench_single_image.restype = ctypes.c_double
        profile_dll.bench_image_set.argtypes = dll.bench_image_set.argtypes
        profile_dll.bench_image_set.restype = ctypes.c_double

        first_image = np.ascontiguousarray(val_images[:1].reshape(-1).astype(np.float32))
        all_images = np.ascontiguousarray(val_images.astype(np.float32).reshape(-1))

        checksum_single = ctypes.c_int32(0)
        checksum_set = ctypes.c_int32(0)
        single_seconds = dll.bench_single_image(
            first_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(args.single_image_iterations),
            ctypes.byref(checksum_single),
        )
        set_seconds = dll.bench_image_set(
            all_images.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(len(val_images)),
            int(args.val_iterations),
            ctypes.byref(checksum_set),
        )
        profile_wrapper = export_dir / "fruit_bench_model_profile_wrapper.c"
        profile_wrapper.write_text(
            textwrap.dedent(
                f"""
                #include <string.h>
                #include "kerasbnn.h"
                #include "fruit_bench_model_export.h"

                int bench_profile_single_image(const float *input, int iterations, kbnn_profile *profile_out, int32_t *checksum) {{
                    static unsigned char arena[{int(meta["arena_bytes"])}] __attribute__((aligned(64)));
                    kbnn_context ctx;
                    int rc = kbnn_context_init(&ctx, &fruit_bench_model, arena, sizeof(arena));
                    if (rc != 0) return rc;
                    kbnn_context_reset_profile(&ctx);
                    int32_t sum = 0;
                    for (int i = 0; i < iterations; ++i) {{
                        kbnn_context_set_input_f32(&ctx, 0, input);
                        rc = kbnn_model_run(&ctx);
                        if (rc != 0) {{
                            kbnn_context_release(&ctx);
                            return rc;
                        }}
                        sum += kbnn_context_get_output_i32(&ctx, {output_ids[0]})[0];
                    }}
                    if (profile_out != NULL) {{
                        *profile_out = *kbnn_context_get_profile(&ctx);
                    }}
                    if (checksum != NULL) {{
                        *checksum = sum;
                    }}
                    kbnn_context_release(&ctx);
                    return 0;
                }}
                """
            ).strip()
            + "\n"
        )
        profile_helper = export_dir / "libfruit_bench_model_profile_helper.so"
        subprocess.run(
            [
                args.cc,
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
                "-DKBNN_ENABLE_PROFILE",
                f"-I{REPO_ROOT}",
                f"-I{REPO_ROOT / 'kerasbnn'}",
                str(profile_wrapper),
                str(REPO_ROOT / "kerasbnn" / "kerasbnn.c"),
                "-lm",
                "-o",
                str(profile_helper),
            ],
            check=True,
        )
        profile_helper_dll = ctypes.CDLL(str(profile_helper))
        profile_helper_dll.bench_profile_single_image.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(KbnnProfile),
            ctypes.POINTER(ctypes.c_int32),
        ]
        profile_helper_dll.bench_profile_single_image.restype = ctypes.c_int

        profile = KbnnProfile()
        checksum_profile = ctypes.c_int32(0)
        rc = profile_helper_dll.bench_profile_single_image(
            first_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(args.single_image_iterations),
            ctypes.byref(profile),
            ctypes.byref(checksum_profile),
        )
        if rc != 0:
            raise RuntimeError(f"profile benchmark returned error code {rc}")

    single_ms = (single_seconds * 1000.0) / max(1, args.single_image_iterations)
    total_images = max(1, len(val_images) * args.val_iterations)
    set_ms_per_image = (set_seconds * 1000.0) / total_images
    images_per_second = total_images / max(set_seconds, 1e-9)
    op_profiles = []
    for idx, op in enumerate(meta["ops"]):
        op_ns = int(profile.op_ns[idx])
        if op_ns <= 0:
            continue
        op_profiles.append(
            {
                "index": idx,
                "name": op["name"],
                "kind": op["kind"],
                "total_ms": op_ns / 1e6,
                "per_call_us": (op_ns / max(1, int(profile.op_calls[idx]))) / 1e3,
                "calls": int(profile.op_calls[idx]),
                "share": op_ns / max(1, int(profile.total_ns)),
                "target": _op_target_summary(op),
            }
        )
    op_profiles.sort(key=lambda x: x["total_ms"], reverse=True)
    profile_summary = {
        "total_ms": profile.total_ns / 1e6,
        "xnor_conv_prep_ms": profile.xnor_conv_prep_ns / 1e6,
        "xnor_conv_dot_ms": profile.xnor_conv_dot_ns / 1e6,
        "xnor_dense_prep_ms": profile.xnor_dense_prep_ns / 1e6,
        "xnor_dense_dot_ms": profile.xnor_dense_dot_ns / 1e6,
        "qconv2d_ms": profile.qconv2d_ns / 1e6,
        "qdense_ms": profile.qdense_ns / 1e6,
        "binary_affine_ms": profile.binary_affine_ns / 1e6,
        "quant_act_ms": profile.quant_act_ns / 1e6,
        "concat_ms": profile.concat_ns / 1e6,
        "qact_qdense_ms": profile.qact_qdense_ns / 1e6,
        "space_to_depth_ms": profile.space_to_depth_ns / 1e6,
        "upsample2d_ms": profile.upsample2d_ns / 1e6,
        "thermo_input_ms": profile.thermo_input_ns / 1e6,
        "quantized_input_ms": profile.quantized_input_ns / 1e6,
        "layer_targets": [_op_target_summary(op) for op in meta["ops"]],
        "top_ops": op_profiles[:16],
    }

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "weights": str(weights_path),
                "compiler": args.cc,
                "calibration": calibration_meta,
                "override_count": len(array_overrides),
                "val_samples": len(val_images),
                "keras_map50": float(keras_map50),
                "c_map50": float(c_map50),
                "map50_delta": float(c_map50 - keras_map50),
                "single_image_iterations": int(args.single_image_iterations),
                "single_image_total_seconds": float(single_seconds),
                "single_image_ms": float(single_ms),
                "single_image_checksum": int(checksum_single.value),
                "single_image_profile_checksum": int(checksum_profile.value),
                "val_iterations": int(args.val_iterations),
                "val_total_seconds": float(set_seconds),
                "val_ms_per_image": float(set_ms_per_image),
                "val_images_per_second": float(images_per_second),
                "val_checksum": int(checksum_set.value),
                "profile": profile_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
