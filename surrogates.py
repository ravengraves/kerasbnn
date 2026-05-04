"""STE and surrogate-gradient utilities for binary and quantized layers."""

from kerasbnn.binary_layers import (
    binary_sign_ste,
    binary_step_ste,
    binary_threshold_ste,
    binary_sign_threshold_ste,
    quantize_unit_interval_ste,
    quantize_signed_unit_interval_ste,
    quantize_signed_int_ste,
    fake_quantize_signed_output_ste,
    quantize_symmetric_range_ste,
    quantize_positive_range_ste,
    round_ste,
)
from kerasbnn.xnor2_layers import (
    binary_sign_clipped_ste,
    binary_sign_clipped_gain_ste,
    binary_sign_triangular_ste,
    binary_sign_triangular_gain_ste,
    binary_weight_sign_magnitude_aware_ste,
    binary_weight_sign_scaled_ste,
    apply_binary_activation_surrogate,
    apply_binary_weight_surrogate,
)

__all__ = [
    "binary_sign_ste",
    "binary_step_ste",
    "binary_threshold_ste",
    "binary_sign_threshold_ste",
    "quantize_unit_interval_ste",
    "quantize_signed_unit_interval_ste",
    "quantize_signed_int_ste",
    "fake_quantize_signed_output_ste",
    "quantize_symmetric_range_ste",
    "quantize_positive_range_ste",
    "round_ste",
    "binary_sign_clipped_ste",
    "binary_sign_clipped_gain_ste",
    "binary_sign_triangular_ste",
    "binary_sign_triangular_gain_ste",
    "binary_weight_sign_magnitude_aware_ste",
    "binary_weight_sign_scaled_ste",
    "apply_binary_activation_surrogate",
    "apply_binary_weight_surrogate",
]
