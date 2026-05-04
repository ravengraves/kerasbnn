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
