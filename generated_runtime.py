"""Helpers for compiling and invoking generic `kerasbnn` export bundles.

This module is intentionally thin:
- the model is lowered through `export_model_bundle(...)`
- generated `{model_name}_runtime.c` remains the public runtime entry point
- this helper only builds a small ctypes wrapper around that generated code

It is used by the fruits parity/benchmark tests so they exercise the same
generic export path intended for wider reuse.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np

from kerasbnn.export_c import FX_ONE, export_model_bundle


def export_bundle_meta(
    model,
    output_dir: Path,
    model_name: str,
    *,
    emit_example_main_c: bool = False,
) -> tuple[dict[str, Path], dict]:
    artifacts = export_model_bundle(
        model,
        output_dir,
        model_name=model_name,
        emit_example_main_c=emit_example_main_c,
    )
    meta = json.loads(artifacts["json"].read_text())
    return artifacts, meta


def compile_bundle_runner(
    output_dir: Path,
    model_name: str,
    arena_bytes: int,
    counts: list[int],
    *,
    repo_root: Path,
    cc: str = "clang",
) -> Path:
    out_args = ", ".join([f"int32_t *out{i}" for i in range(len(counts))])
    copies = "\n".join(
        [
            f"        memcpy(out{i}, {model_name}_output_i32(&ctx, {i}), sizeof(int32_t) * {count});"
            for i, count in enumerate(counts)
        ]
    )
    wrapper = output_dir / f"{model_name}_wrapper.c"
    wrapper.write_text(
        textwrap.dedent(
            f"""
            #include <string.h>
            #include "kerasbnn.h"
            #include "{model_name}_export.h"

            int {model_name}_init(kbnn_context *ctx, void *arena, size_t arena_bytes);
            void {model_name}_set_input_f32(kbnn_context *ctx, int32_t input_index, const float *input);
            const int32_t *{model_name}_output_i32(const kbnn_context *ctx, int32_t output_index);
            int {model_name}_run(kbnn_context *ctx);

            int run_model(const float *input, {out_args}) {{
                static unsigned char arena[{arena_bytes}];
                kbnn_context ctx;
                int rc = {model_name}_init(&ctx, arena, sizeof(arena));
                if (rc != 0) return rc;
                {model_name}_set_input_f32(&ctx, 0, input);
                rc = {model_name}_run(&ctx);
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
        f"-I{repo_root}",
        f"-I{repo_root / 'kerasbnn'}",
        str(wrapper),
        str(output_dir / f"{model_name}_runtime.c"),
        str(repo_root / "kerasbnn" / "kerasbnn.c"),
        "-lm",
        "-o",
        str(lib_path),
    ]
    subprocess.run(cmd, check=True)
    return lib_path


def predict_bundle_outputs(
    model,
    images: np.ndarray,
    output_dir: Path,
    model_name: str,
    *,
    repo_root: Path,
    cc: str = "clang",
) -> tuple[dict[str, np.ndarray], dict]:
    _, meta = export_bundle_meta(
        model,
        output_dir,
        model_name=model_name,
        emit_example_main_c=False,
    )
    output_ids = [int(v) for v in meta["outputs"]]
    counts = [int(np.prod(meta["tensors"][tensor_id]["shape"])) for tensor_id in output_ids]
    names = [meta["tensors"][tensor_id]["name"] for tensor_id in output_ids]
    lib_path = compile_bundle_runner(
        output_dir,
        model_name,
        int(meta["arena_bytes"]),
        counts,
        repo_root=repo_root,
        cc=cc,
    )
    dll = ctypes.CDLL(str(lib_path))
    dll.run_model.argtypes = [ctypes.POINTER(ctypes.c_float)] + [ctypes.POINTER(ctypes.c_int32)] * len(output_ids)
    dll.run_model.restype = ctypes.c_int
    accum: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for image in images:
        out_bufs = [np.zeros((count,), dtype=np.int32) for count in counts]
        args = [image.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))] + [
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) for buf in out_bufs
        ]
        rc = dll.run_model(*args)
        if rc != 0:
            raise RuntimeError(f"C runtime returned error code {rc}")
        for name, tensor_id, buf in zip(names, output_ids, out_bufs):
            shape = tuple(meta["tensors"][tensor_id]["shape"])
            accum[name].append((buf.astype(np.float32) / FX_ONE).reshape(shape))
    return {name: np.stack(values, axis=0) for name, values in accum.items()}, meta
