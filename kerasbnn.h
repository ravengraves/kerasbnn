#ifndef KERASBNN_H
#define KERASBNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KBNN_FX_SHIFT 12
#define KBNN_FX_ONE (1 << KBNN_FX_SHIFT)
#define KBNN_MAX_RANK 5
#define KBNN_MAX_TENSORS 256
#define KBNN_MAX_PROFILE_OPS 512

typedef enum {
    KBNN_TENSOR_EXTERNAL_F32 = 0,
    KBNN_TENSOR_I32 = 1,
    KBNN_TENSOR_PACKED_BINARY = 2,
} kbnn_tensor_kind;

typedef enum {
    KBNN_OP_THERMO_INPUT = 0,
    KBNN_OP_QUANTIZED_INPUT = 1,
    KBNN_OP_QCONV2D = 2,
    KBNN_OP_XNOR_CONV2D = 3,
    KBNN_OP_XNOR_DENSE = 4,
    KBNN_OP_SPACE_TO_DEPTH = 5,
    KBNN_OP_CONCAT = 6,
    KBNN_OP_UPSAMPLE2D = 7,
    KBNN_OP_BINARY_AFFINE_NORM = 8,
    KBNN_OP_QUANT_ACT = 9,
    KBNN_OP_QDENSE = 10,
    KBNN_OP_QACT_QDENSE = 11,
} kbnn_op_kind;

typedef struct {
    const char *name;
    int32_t kind;
    int32_t rank;
    int32_t shape[KBNN_MAX_RANK];
    int32_t words_per_point;
    size_t offset_bytes;
    size_t bytes;
} kbnn_tensor_desc;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t height;
    int32_t width;
    int32_t channels;
    int32_t levels;
    float min_value;
    float max_value;
} kbnn_thermo_input_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t rank;
    int32_t shape[KBNN_MAX_RANK];
    int32_t levels;
    float min_value;
    float max_value;
    int32_t signed_output;
} kbnn_quantized_input_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t in_h;
    int32_t in_w;
    int32_t in_c;
    int32_t out_h;
    int32_t out_w;
    int32_t out_c;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;
    int32_t input_is_packed_binary;
    int32_t weight_divisor;
    int32_t levels;
    int32_t integer_output;
    int32_t output_scale_q;
    const int16_t *weights;
    const int32_t *bias_q;
} kbnn_qconv2d_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t scratch_id;
    int32_t in_h;
    int32_t in_w;
    int32_t in_c;
    int32_t out_h;
    int32_t out_w;
    int32_t out_c;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;
    int32_t channel_words;
    uint64_t last_word_mask;
    const uint64_t *weights_bits;
    const int32_t *alpha_q;
    const int32_t *bias_q;
    const int8_t *act_scale_i8;
    const int8_t *act_bias_i8;
    int32_t input_cast_fp16;
} kbnn_xnor_conv2d_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t scratch_id;
    int32_t outer_count;
    int32_t in_dim;
    int32_t out_dim;
    int32_t channel_words;
    uint64_t last_word_mask;
    const uint64_t *weights_bits;
    const int32_t *alpha_q;
    const int32_t *bias_q;
    const int8_t *act_scale_i8;
    const int8_t *act_bias_i8;
    int32_t input_cast_fp16;
} kbnn_xnor_dense_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t in_h;
    int32_t in_w;
    int32_t channels;
    int32_t block_size;
} kbnn_space_to_depth_params;

typedef struct {
    int32_t output_id;
    int32_t input_count;
    const int32_t *input_ids;
    int32_t rank;
    int32_t output_shape[KBNN_MAX_RANK];
    int32_t axis;
} kbnn_concat_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t in_h;
    int32_t in_w;
    int32_t channels;
    int32_t scale_h;
    int32_t scale_w;
} kbnn_upsample2d_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t rank;
    int32_t shape[KBNN_MAX_RANK];
    const int32_t *mul_q;
    const int32_t *add_q;
} kbnn_binary_affine_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t rank;
    int32_t shape[KBNN_MAX_RANK];
    int32_t levels;
    int32_t signed_mode;
} kbnn_quant_act_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t outer_count;
    int32_t in_dim;
    int32_t out_dim;
    int32_t weight_divisor;
    int32_t levels;
    int32_t integer_output;
    int32_t output_scale_q;
    const int16_t *weights;
    const int32_t *bias_q;
} kbnn_qdense_params;

typedef struct {
    int32_t input_id;
    int32_t output_id;
    int32_t outer_count;
    int32_t in_dim;
    int32_t out_dim;
    int32_t act_levels;
    int32_t act_signed_mode;
    int32_t weight_divisor;
    int32_t levels;
    int32_t integer_output;
    int32_t output_scale_q;
    const int16_t *weights;
    const int32_t *bias_q;
} kbnn_qact_qdense_params;

typedef struct {
    kbnn_op_kind kind;
    const char *name;
    const void *params;
} kbnn_op;

typedef struct {
    const char *name;
    const kbnn_tensor_desc *tensors;
    int32_t tensor_count;
    const kbnn_op *ops;
    int32_t op_count;
    const int32_t *output_ids;
    int32_t output_count;
    size_t arena_bytes;
} kbnn_model;

typedef struct {
    uint64_t total_ns;
    uint64_t op_ns[KBNN_MAX_PROFILE_OPS];
    uint64_t op_calls[KBNN_MAX_PROFILE_OPS];
    uint64_t thermo_input_ns;
    uint64_t quantized_input_ns;
    uint64_t qconv2d_ns;
    uint64_t xnor_conv_prep_ns;
    uint64_t xnor_conv_dot_ns;
    uint64_t xnor_dense_prep_ns;
    uint64_t xnor_dense_dot_ns;
    uint64_t space_to_depth_ns;
    uint64_t concat_ns;
    uint64_t upsample2d_ns;
    uint64_t binary_affine_ns;
    uint64_t quant_act_ns;
    uint64_t qdense_ns;
    uint64_t qact_qdense_ns;
} kbnn_profile;

typedef struct {
    const kbnn_model *model;
    uint8_t *arena;
    size_t arena_bytes;
    void *tensor_ptrs[KBNN_MAX_TENSORS];
    kbnn_profile profile;
} kbnn_context;

typedef struct {
    float cx;
    float cy;
    float w;
    float h;
    float score;
    int32_t class_id;
} kbnn_detection;

int kbnn_context_init(kbnn_context *ctx, const kbnn_model *model, void *arena, size_t arena_bytes);
void kbnn_context_release(kbnn_context *ctx);
void kbnn_context_set_input_f32(kbnn_context *ctx, int32_t tensor_id, const float *input);
const int32_t *kbnn_context_get_output_i32(const kbnn_context *ctx, int32_t tensor_id);
void *kbnn_context_tensor_ptr(const kbnn_context *ctx, int32_t tensor_id);
void kbnn_context_reset_profile(kbnn_context *ctx);
const kbnn_profile *kbnn_context_get_profile(const kbnn_context *ctx);
int kbnn_model_run(kbnn_context *ctx);

int32_t kbnn_xnor_dot_u64(const uint64_t *a, const uint64_t *b, int32_t words, uint64_t last_mask);
int kbnn_nms(const kbnn_detection *detections, int32_t count, float iou_threshold, int32_t top_k, int32_t *keep_indices);

#ifdef __cplusplus
}
#endif

#endif
