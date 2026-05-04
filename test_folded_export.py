from __future__ import annotations

import sys

import numpy as np
import keras

sys.path.insert(0, "/home/claude/source/yolovxnor")

from kerasbnn.export_c import _fold_xnor_activation_compare  # noqa: E402
from kerasbnn.xnor2_layers import XNORBinaryConv2D, XNORBinaryDense  # noqa: E402
from binary_yolo.tinier_yolo_xnor2 import TinierXNOR2Config, build_binary_tinier_yolov3_xnor2  # noqa: E402


def _folded_sign(layer: XNORBinaryConv2D, raw_input: np.ndarray) -> np.ndarray:
    scale_i8, bias_i8 = _fold_xnor_activation_compare(layer)
    scale = scale_i8.astype(np.float32)
    bias = bias_i8.astype(np.float32)
    return (raw_input * scale.reshape((1,) * (raw_input.ndim - 1) + (-1,)) + bias.reshape((1,) * (raw_input.ndim - 1) + (-1,))) >= 0.0


def _keras_sign(layer: XNORBinaryConv2D, raw_input: np.ndarray) -> np.ndarray:
    x = raw_input.astype(np.float32)
    x = np.asarray(layer._apply_norm(x, training=False))
    x = np.asarray(layer._apply_rprelu(x))
    return x >= 0.0


def test_known_single_dense_layer():
    layer = XNORBinaryDense(
        3,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="test_dense_known",
    )
    x = np.array([[0.25, -0.5, 0.75, 0.1]], dtype=np.float32)
    layer.build(x.shape)
    layer.kernel.assign(
        np.array(
            [
                [0.25, -0.6, 0.8],
                [-0.4, 0.3, -0.7],
                [0.9, -0.1, 0.2],
                [-0.2, 0.5, -0.3],
            ],
            dtype=np.float32,
        )
    )
    layer.bias.assign(np.array([0.1, -0.2, 0.05], dtype=np.float32))
    layer.affine_norm.norm_scale.assign(np.array([0.50, -0.75, 0.25, 0.33], dtype=np.float32))
    layer.affine_norm.norm_threshold.assign(np.array([0.10, -0.20, 0.05, -0.15], dtype=np.float32))
    layer.affine_norm.bias.assign(np.array([0.00, 0.15, -0.10, 0.05], dtype=np.float32))
    layer.affine_norm.moving_mean.assign(np.array([0.05, -0.10, 0.20, 0.12], dtype=np.float32))
    layer.affine_norm.moving_variance.assign(np.array([0.25, 0.50, 0.75, 0.90], dtype=np.float32))
    layer.rprelu_gamma.assign(np.array([0.10, -0.05, 0.20, 0.02], dtype=np.float32))
    layer.rprelu_beta.assign(np.array([0.50, 0.25, 0.75, 0.4], dtype=np.float32))
    layer.rprelu_zeta.assign(np.array([-0.02, 0.15, -0.10, 0.07], dtype=np.float32))

    expected = _keras_sign(layer, x)
    actual = _folded_sign(layer, x)
    if not np.array_equal(expected, actual):
        raise AssertionError(f"Known single dense folded sign mismatch:\nexpected={expected}\nactual={actual}")


def test_random_single_dense_layer():
    rng = np.random.default_rng(17)
    layer = XNORBinaryDense(
        7,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="test_dense_random",
    )
    x = rng.normal(size=(5, 13)).astype(np.float32)
    layer.build(x.shape)
    layer.kernel.assign(rng.uniform(-1.0, 1.0, size=layer.kernel.shape).astype(np.float32))
    layer.bias.assign(rng.normal(scale=0.2, size=layer.bias.shape).astype(np.float32))
    layer.affine_norm.norm_scale.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_scale.shape).astype(np.float32))
    layer.affine_norm.norm_threshold.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_threshold.shape).astype(np.float32))
    layer.affine_norm.bias.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.bias.shape).astype(np.float32))
    layer.affine_norm.moving_mean.assign(rng.normal(scale=0.5, size=layer.affine_norm.moving_mean.shape).astype(np.float32))
    layer.affine_norm.moving_variance.assign(rng.uniform(0.05, 2.0, size=layer.affine_norm.moving_variance.shape).astype(np.float32))
    layer.rprelu_gamma.assign(rng.normal(scale=0.3, size=layer.rprelu_gamma.shape).astype(np.float32))
    layer.rprelu_beta.assign(rng.uniform(0.0, 1.0, size=layer.rprelu_beta.shape).astype(np.float32))
    layer.rprelu_zeta.assign(rng.normal(scale=0.3, size=layer.rprelu_zeta.shape).astype(np.float32))

    expected = _keras_sign(layer, x)
    actual = _folded_sign(layer, x)
    if not np.array_equal(expected, actual):
        raise AssertionError("Random single dense folded sign mismatch")


def test_known_single_layer():
    layer = XNORBinaryConv2D(
        1,
        1,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="test_known",
    )
    x = np.array(
        [[[[0.25, -0.50, 0.75], [0.10, 0.20, -0.30]]]],
        dtype=np.float32,
    )
    layer.build(x.shape)
    layer.kernel.assign(np.array([[[[0.4], [-0.7], [0.2]]]], dtype=np.float32))
    layer.bias.assign(np.array([0.15], dtype=np.float32))
    layer.affine_norm.norm_scale.assign(np.array([0.50, -0.75, 0.25], dtype=np.float32))
    layer.affine_norm.norm_threshold.assign(np.array([0.10, -0.20, 0.05], dtype=np.float32))
    layer.affine_norm.bias.assign(np.array([0.00, 0.15, -0.10], dtype=np.float32))
    layer.affine_norm.moving_mean.assign(np.array([0.05, -0.10, 0.20], dtype=np.float32))
    layer.affine_norm.moving_variance.assign(np.array([0.25, 0.50, 0.75], dtype=np.float32))
    layer.rprelu_gamma.assign(np.array([0.10, -0.05, 0.20], dtype=np.float32))
    layer.rprelu_beta.assign(np.array([0.50, 0.25, 0.75], dtype=np.float32))
    layer.rprelu_zeta.assign(np.array([-0.02, 0.15, -0.10], dtype=np.float32))

    expected = _keras_sign(layer, x)
    actual = _folded_sign(layer, x)
    if not np.array_equal(expected, actual):
        raise AssertionError(f"Known single-layer folded sign mismatch:\nexpected={expected}\nactual={actual}")


def test_random_single_layer():
    rng = np.random.default_rng(7)
    layer = XNORBinaryConv2D(
        2,
        3,
        output_mode="raw",
        normalize=True,
        binary_batchnorm=True,
        norm_levels=128,
        momentum=0.8,
        epsilon=1e-4,
        activation_surrogate="triangular",
        weight_surrogate="magnitude_aware",
        weight_surrogate_floor=0.25,
        use_rprelu=True,
        name="test_random",
    )
    x = rng.normal(size=(2, 8, 8, 5)).astype(np.float32)
    layer.build(x.shape)
    layer.kernel.assign(rng.uniform(-1.0, 1.0, size=layer.kernel.shape).astype(np.float32))
    layer.bias.assign(rng.normal(scale=0.2, size=layer.bias.shape).astype(np.float32))
    layer.affine_norm.norm_scale.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_scale.shape).astype(np.float32))
    layer.affine_norm.norm_threshold.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.norm_threshold.shape).astype(np.float32))
    layer.affine_norm.bias.assign(rng.uniform(-1.0, 1.0, size=layer.affine_norm.bias.shape).astype(np.float32))
    layer.affine_norm.moving_mean.assign(rng.normal(scale=0.5, size=layer.affine_norm.moving_mean.shape).astype(np.float32))
    layer.affine_norm.moving_variance.assign(rng.uniform(0.05, 2.0, size=layer.affine_norm.moving_variance.shape).astype(np.float32))
    layer.rprelu_gamma.assign(rng.normal(scale=0.3, size=layer.rprelu_gamma.shape).astype(np.float32))
    layer.rprelu_beta.assign(rng.uniform(0.0, 1.0, size=layer.rprelu_beta.shape).astype(np.float32))
    layer.rprelu_zeta.assign(rng.normal(scale=0.3, size=layer.rprelu_zeta.shape).astype(np.float32))

    expected = _keras_sign(layer, x)
    actual = _folded_sign(layer, x)
    if not np.array_equal(expected, actual):
        raise AssertionError("Random single-layer folded sign mismatch")


def test_full_image_model():
    rng = np.random.default_rng(11)
    model = build_binary_tinier_yolov3_xnor2(4, input_shape=(64, 64, 3), cfg=TinierXNOR2Config())
    image = rng.uniform(0.0, 1.0, size=(1, 64, 64, 3)).astype(np.float32)
    xnor_layers = [layer for layer in model.layers if isinstance(layer, XNORBinaryConv2D)]
    parent_layers = [layer.input._keras_history[0] for layer in xnor_layers]
    probe = keras.Model(inputs=model.input, outputs=[parent.output for parent in parent_layers])
    raw_inputs = probe(image, training=False)
    if not isinstance(raw_inputs, (list, tuple)):
        raw_inputs = [raw_inputs]

    for layer, raw_input in zip(xnor_layers, raw_inputs):
        raw_input = np.asarray(raw_input)
        expected = _keras_sign(layer, raw_input)
        actual = _folded_sign(layer, raw_input)
        if not np.array_equal(expected, actual):
            mismatch = np.count_nonzero(expected != actual)
            raise AssertionError(f"Full-image folded sign mismatch in {layer.name}: {mismatch} bits differ")


def main():
    test_known_single_layer()
    print("known single-layer test passed")
    test_random_single_layer()
    print("random single-layer test passed")
    test_known_single_dense_layer()
    print("known single dense test passed")
    test_random_single_dense_layer()
    print("random single dense test passed")
    test_full_image_model()
    print("full-image model test passed")


if __name__ == "__main__":
    main()
