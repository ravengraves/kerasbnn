"""CLI for exporting supported serialized Keras BNN models to portable C11.

Supported lowered layer families:
- `InputLayer`
- `Activation` / `Reshape` / `Flatten`
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

Unsupported layers raise `NotImplementedError` during export.

Usage:
    python -m kerasbnn.export_codegen \
        --model path/to/model.keras \
        --output-dir out/export \
        --model-name my_bnn

Artifacts:
- `{model_name}_export.h`: packed hex/int weight header plus static model tables
- `{model_name}_export.json`: tensor/op/input/output metadata
- `{model_name}_runtime.c`: generic C11 wrapper helpers
- `{model_name}_example_main.c`: standalone example program using zero inputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras

from kerasbnn.compat import get_compat_custom_objects
from kerasbnn.export_c import export_model_bundle


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export a supported serialized Keras BNN model to C11 artifacts.")
    ap.add_argument("--model", required=True, help="Path to a serialized .keras or .h5 Keras model.")
    ap.add_argument("--output-dir", required=True, help="Directory that will receive generated export files.")
    ap.add_argument("--model-name", default="", help="Optional C symbol prefix. Defaults to the model filename stem.")
    ap.add_argument("--no-runtime-c", action=argparse.BooleanOptionalAction, default=False, help="Skip the generated runtime wrapper C source.")
    ap.add_argument("--no-example-main-c", action=argparse.BooleanOptionalAction, default=False, help="Skip the generated example main C source.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    model_name = args.model_name or model_path.stem.replace("-", "_")
    model = keras.models.load_model(
        model_path,
        custom_objects=get_compat_custom_objects(),
        compile=False,
    )
    artifacts = export_model_bundle(
        model,
        Path(args.output_dir),
        model_name=model_name,
        emit_runtime_c=not args.no_runtime_c,
        emit_example_main_c=not args.no_example_main_c,
    )
    for label, path in artifacts.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
