# kerasbnn

`kerasbnn` is the shared Python module for the repository's Keras-based binary
and quantized neural network layers.

It contains:
- base binary/quantized layers used by the older binary YOLO models
- XNOR2-specific layers used by the current `tinier_yolo_xnor2` models
- reusable building blocks intended to support future export tooling

Primary modules:
- `kerasbnn.layers`
  - trainable binary/XNOR layers such as `BinaryConv2D`, `BinaryDense`,
    `XNORBinaryConv2D`, and `XNORBinaryDense`
- `kerasbnn.normalization`
  - normalization layers such as `BinaryAffineNorm`
- `kerasbnn.quantization`
  - quantized activations/projections such as `QuantizedActivation`,
    `QuantizedConv2D`, `QuantizedDense`, and `quantized_first_conv`
- `kerasbnn.inputs`
  - input transforms such as `BinaryInput`, `ThermometerBinaryInput`,
    and `QuantizedInput`
- `kerasbnn.augment`
  - augmentation/regularization layers such as `BinaryCutout`,
    `BinaryGreyout`, and `BinaryDropcut`
- `kerasbnn.surrogates`
  - STE and surrogate-gradient helper functions
- `kerasbnn.constraints`
  - weight clipping constraints

Compatibility modules kept for older imports:
- `kerasbnn.binary_layers`
- `kerasbnn.xnor2_layers`

Package intent:
- keep the Keras training layers in one stable import path
- let model code share the same layer implementations
- make later export tools easier to build around a single package

Compatibility:
- `bitbnn.binary_layers` now re-exports from `kerasbnn.binary_layers`
- `binary_yolo.xnor2_layers` now re-exports from `kerasbnn.xnor2_layers`
- temporary compatibility packages `xnornet` and `kbnn` also re-export from
  `kerasbnn`
- `kerasbnn.compat.get_compat_custom_objects()` provides a legacy-name mapping
  for loading older serialized models under the new package name
- `kerasbnn.compat.remap_serialized_keras_config()` can normalize older
  serialized Keras configs before deserialization

Recommended direct imports:

```python
from kerasbnn import BinaryAffineNorm, QuantizedDense
from kerasbnn import XNORBinaryConv2D, ThermometerBinaryInput
```

Or by category:

```python
from kerasbnn.normalization import BinaryAffineNorm
from kerasbnn.quantization import QuantizedDense
from kerasbnn.layers import XNORBinaryConv2D
from kerasbnn.inputs import ThermometerBinaryInput
```

For serialized model compatibility:

```python
from kerasbnn.compat import get_compat_custom_objects

model = keras.models.load_model(path, custom_objects=get_compat_custom_objects())
```

For raw serialized configs:

```python
from kerasbnn.compat import remap_serialized_keras_config

config = remap_serialized_keras_config(config)
```

## C Export

`kerasbnn` also provides a generic C11 export path for supported binary /
quantized Keras models.

Supported lowered layer families:
- `InputLayer`
- `Activation`, `Reshape`, `Flatten`
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

Python API:

```python
from kerasbnn import export_model_bundle

artifacts = export_model_bundle(model, "out/export", model_name="my_bnn")
```

This writes:
- `my_bnn_export.h`
- `my_bnn_export.json`
- `my_bnn_runtime.c`
- `my_bnn_example_main.c`

CLI for serialized `.keras` / `.h5` models:

```bash
python -m kerasbnn.export_codegen \
  --model path/to/model.keras \
  --output-dir out/export \
  --model-name my_bnn
```

The generated header stores packed binary weights as hex `uint64_t` arrays and
quantized tensors as integer arrays. The runtime/example sources compile
against `kerasbnn.c` / `kerasbnn.h`.

## Detector Utilities

For YOLO-style detectors, `kerasbnn` also ships optional detector-side C
utilities:

- `kerasbnn/yolo_postprocess.h`
- `kerasbnn/yolo_postprocess.c`

These are intentionally separate from the generic exporter/runtime:

- `kerasbnn.c` remains model-agnostic
- `yolo_postprocess.c` only decodes raw detector head tensors and runs NMS

Supported detector output layout:

- each head is an `int32` Q12 tensor from `kbnn_context_get_output_i32()`
- shape: `[grid_h, grid_w, anchors_per_cell, 5 + num_classes]`
- decode modes:
  - `KBNN_YOLO_BOX_LINEAR`
  - `KBNN_YOLO_BOX_LINEAR_FLOAT`
  - `KBNN_YOLO_BOX_YOLO`
- class score modes:
  - `KBNN_YOLO_CLASS_BCE`
  - `KBNN_YOLO_CLASS_SOFTMAX`

`binary_yolo/export_tinier_xnor2_c.py` can layer a detector-specific example on
top of the generic export bundle when anchors are available. It emits:

- `{model_name}_yolo_config.h`
- `{model_name}_yolo_demo.c`

The generated demo uses the vendored stb image loader at:

- `kerasbnn/third_party/stb/stb_image.h`

and can be compiled alongside:

- `{model_name}_runtime.c`
- `kerasbnn.c`
- `yolo_postprocess.c`

The demo intentionally keeps preprocessing simple:

- RGB load through stb
- direct resize to the model input size
- float `[0, 1]` normalization
- detector decode + NMS through `yolo_postprocess.c`
