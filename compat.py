"""Compatibility helpers for renamed/legacy kerasbnn layer identifiers."""

from __future__ import annotations

import keras

from kerasbnn.binary_layers import (
    BinaryInput,
    BinaryAffineNorm,
    BinaryConv2D,
    BinaryDense,
    BinaryOutput,
    QuantizedDense,
    QuantizedConv2D,
    QuantizedOutput,
    QuantizedActivation,
    BinaryCutout,
    BinaryGreyout,
    BinaryDropcut,
    BinarySoftmax,
    BinaryAveragePooling2D,
    ClipConstraint,
    ClipRangeConstraint,
)
from kerasbnn.xnor2_layers import (
    ClipRangeConstraint as XNORClipRangeConstraint,
    ThermometerBinaryInput,
    QuantizedInput,
    XNORBinaryConv2D,
    XNORBinaryDense,
)


_CANONICAL_LAYER_MAP = {
    "ClipConstraint": ClipConstraint,
    "ClipRangeConstraint": ClipRangeConstraint,
    "BinaryInput": BinaryInput,
    "BinaryAffineNorm": BinaryAffineNorm,
    "BinaryConv2D": BinaryConv2D,
    "BinaryDense": BinaryDense,
    "BinaryOutput": BinaryOutput,
    "QuantizedDense": QuantizedDense,
    "QuantizedConv2D": QuantizedConv2D,
    "QuantizedOutput": QuantizedOutput,
    "QuantizedActivation": QuantizedActivation,
    "BinaryCutout": BinaryCutout,
    "BinaryGreyout": BinaryGreyout,
    "BinaryDropcut": BinaryDropcut,
    "BinarySoftmax": BinarySoftmax,
    "BinaryAveragePooling2D": BinaryAveragePooling2D,
    "ThermometerBinaryInput": ThermometerBinaryInput,
    "QuantizedInput": QuantizedInput,
    "XNORBinaryConv2D": XNORBinaryConv2D,
    "XNORBinaryDense": XNORBinaryDense,
    "XNORClipRangeConstraint": XNORClipRangeConstraint,
}


def _module_aliases(module_name: str) -> dict[str, object]:
    aliases = {}
    for name, cls in _CANONICAL_LAYER_MAP.items():
        aliases[f"{module_name}.{name}"] = cls
        aliases[f"{module_name}>{name}"] = cls
    return aliases


_LEGACY_NAME_MAP: dict[str, object] = {
    **_CANONICAL_LAYER_MAP,
    "BinaryLayers>ClipConstraint": ClipConstraint,
    "BinaryLayers>ClipRangeConstraint": ClipRangeConstraint,
    "BinaryLayers>BinaryInput": BinaryInput,
    "BinaryLayers>BinaryAffineNorm": BinaryAffineNorm,
    "BinaryLayers>BinaryConv2D": BinaryConv2D,
    "BinaryLayers>BinaryDense": BinaryDense,
    "BinaryLayers>BinaryOutput": BinaryOutput,
    "BinaryLayers>QuantizedDense": QuantizedDense,
    "BinaryLayers>QuantizedConv2D": QuantizedConv2D,
    "BinaryLayers>QuantizedOutput": QuantizedOutput,
    "BinaryLayers>QuantizedActivation": QuantizedActivation,
    "BinaryLayers>BinaryCutout": BinaryCutout,
    "BinaryLayers>BinaryGreyout": BinaryGreyout,
    "BinaryLayers>BinaryDropcut": BinaryDropcut,
    "BinaryLayers>BinarySoftmax": BinarySoftmax,
    "BinaryLayers>BinaryAveragePooling2D": BinaryAveragePooling2D,
    "XNOR2>ClipRangeConstraint": XNORClipRangeConstraint,
    "XNOR2>ThermometerBinaryInput": ThermometerBinaryInput,
    "XNOR2>QuantizedInput": QuantizedInput,
    "XNOR2>XNORBinaryConv2D": XNORBinaryConv2D,
    "XNOR2>XNORBinaryDense": XNORBinaryDense,
    **_module_aliases("binary_layers"),
    **_module_aliases("bitbnn.binary_layers"),
    **_module_aliases("kerasbnn.binary_layers"),
    **_module_aliases("xnornet.binary_layers"),
    **_module_aliases("kbnn.binary_layers"),
    **_module_aliases("binary_yolo.xnor2_layers"),
    **_module_aliases("kerasbnn.xnor2_layers"),
    **_module_aliases("xnornet.xnor2_layers"),
    **_module_aliases("kbnn.xnor2_layers"),
}


def remap_legacy_layer_name(name: str) -> str:
    """Map a legacy serialized identifier to the current friendly class name."""

    if name in _CANONICAL_LAYER_MAP:
        return name
    target = _LEGACY_NAME_MAP.get(str(name))
    if target is None:
        return str(name)
    return str(target.__name__)


def remap_serialized_keras_config(config):
    """Recursively normalize legacy layer identifiers inside a Keras config tree."""

    if isinstance(config, list):
        return [remap_serialized_keras_config(item) for item in config]
    if not isinstance(config, dict):
        return config

    remapped = {key: remap_serialized_keras_config(value) for key, value in config.items()}
    class_name = remapped.get("class_name")
    if isinstance(class_name, str):
        remapped["class_name"] = remap_legacy_layer_name(class_name)
    registered_name = remapped.get("registered_name")
    if isinstance(registered_name, str):
        remapped["registered_name"] = remap_legacy_layer_name(registered_name)
    module_name = remapped.get("module")
    if isinstance(module_name, str):
        remapped["module"] = module_name.replace("bitbnn", "kerasbnn").replace("xnornet", "kerasbnn").replace("kbnn", "kerasbnn")
    return remapped


def get_compat_custom_objects() -> dict[str, object]:
    """Return custom_objects aliases covering current and legacy layer names."""

    return dict(_LEGACY_NAME_MAP)


def register_compat_custom_objects() -> dict[str, object]:
    """Register current and legacy aliases into Keras custom objects."""

    custom_objects = keras.saving.get_custom_objects()
    custom_objects.update(get_compat_custom_objects())
    return custom_objects


__all__ = [
    "get_compat_custom_objects",
    "register_compat_custom_objects",
    "remap_legacy_layer_name",
    "remap_serialized_keras_config",
]
