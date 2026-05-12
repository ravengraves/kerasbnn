#include "fruits_example_detector.h"
#include "fruits_example_export.h"

int fruits_example_init(kbnn_context *ctx, void *arena, size_t arena_bytes);
size_t fruits_example_arena_bytes(void);
void fruits_example_set_input_f32(kbnn_context *ctx, int32_t input_index, const float *input);
const int32_t *fruits_example_output_i32(const kbnn_context *ctx, int32_t output_index);
int fruits_example_run(kbnn_context *ctx);
int kbnn_model_run_from(kbnn_context *ctx, int32_t op_start);

size_t fruits_example_detector_arena_bytes(void) {
    return fruits_example_arena_bytes();
}

int32_t fruits_example_detector_input_w(void) {
    return FRUITS_EXAMPLE_YOLO_INPUT_W;
}

int32_t fruits_example_detector_input_h(void) {
    return FRUITS_EXAMPLE_YOLO_INPUT_H;
}

int32_t fruits_example_detector_num_classes(void) {
    return FRUITS_EXAMPLE_YOLO_NUM_CLASSES;
}

const char *fruits_example_detector_class_name(int32_t class_id) {
    if (class_id < 0 || class_id >= FRUITS_EXAMPLE_YOLO_NUM_CLASSES) {
        return "unknown";
    }
    return fruits_example_yolo_class_names[class_id];
}

const kbnn_yolo_postprocess_params *fruits_example_detector_default_params(void) {
    return &fruits_example_yolo_params;
}

int fruits_example_detector_init(fruits_example_detector *detector, void *arena, size_t arena_bytes) {
    if (detector == NULL || arena == NULL) {
        return -1;
    }
    detector->heads[0].pred_q12 = NULL;
    detector->heads[0].grid_h = fruits_example_yolo_head_grid_h[0];
    detector->heads[0].grid_w = fruits_example_yolo_head_grid_w[0];
    detector->heads[0].anchors_per_cell = fruits_example_yolo_head_anchor_count[0];
    detector->heads[0].anchor_indices = fruits_example_yolo_head_0_anchor_indices;
    detector->heads[1].pred_q12 = NULL;
    detector->heads[1].grid_h = fruits_example_yolo_head_grid_h[1];
    detector->heads[1].grid_w = fruits_example_yolo_head_grid_w[1];
    detector->heads[1].anchors_per_cell = fruits_example_yolo_head_anchor_count[1];
    detector->heads[1].anchor_indices = fruits_example_yolo_head_1_anchor_indices;
    return fruits_example_init(&detector->ctx, arena, arena_bytes);
}

void fruits_example_detector_release(fruits_example_detector *detector) {
    if (detector == NULL) {
        return;
    }
    kbnn_context_release(&detector->ctx);
}

int32_t fruits_example_detector_run_f32_with_params(
    fruits_example_detector *detector,
    const float *input_rgb,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *out_detections,
    int32_t out_capacity
) {
    int32_t i;
    if (detector == NULL || input_rgb == NULL || out_detections == NULL || out_capacity <= 0) {
        return -1;
    }
    fruits_example_set_input_f32(&detector->ctx, 0, input_rgb);
    if (fruits_example_run(&detector->ctx) != 0) {
        return -2;
    }
    for (i = 0; i < 2; ++i) {
        detector->heads[i].pred_q12 = fruits_example_output_i32(&detector->ctx, fruits_example_yolo_head_output_indices[i]);
    }
    return kbnn_yolo_postprocess(
        detector->heads,
        2,
        fruits_example_yolo_anchors,
        (int32_t)(sizeof(fruits_example_yolo_anchors) / sizeof(fruits_example_yolo_anchors[0]) / 2u),
        (params != NULL) ? params : &fruits_example_yolo_params,
        detector->scratch,
        FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS,
        detector->keep_indices,
        FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS,
        out_detections,
        out_capacity
    );
}

int32_t fruits_example_detector_run_rgb8_with_params(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *out_detections,
    int32_t out_capacity
) {
    int32_t i;
    if (detector == NULL || input_rgb_u8 == NULL || out_detections == NULL || out_capacity <= 0) {
        return -1;
    }
    kbnn_pack_thermo_input_u8(&fruits_example_binary_input_params, input_rgb_u8, FRUITS_EXAMPLE_YOLO_INPUT_W * 3,
        (uint64_t *)kbnn_context_tensor_ptr(&detector->ctx, 1));
    if (kbnn_model_run_from(&detector->ctx, 1) != 0) {
        return -2;
    }
    for (i = 0; i < 2; ++i) {
        detector->heads[i].pred_q12 = fruits_example_output_i32(&detector->ctx, fruits_example_yolo_head_output_indices[i]);
    }
    return kbnn_yolo_postprocess(
        detector->heads,
        2,
        fruits_example_yolo_anchors,
        (int32_t)(sizeof(fruits_example_yolo_anchors) / sizeof(fruits_example_yolo_anchors[0]) / 2u),
        (params != NULL) ? params : &fruits_example_yolo_params,
        detector->scratch,
        FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS,
        detector->keep_indices,
        FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS,
        out_detections,
        out_capacity
    );
}

int32_t fruits_example_detector_run_rgb8(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    kbnn_detection *out_detections,
    int32_t out_capacity
) {
    return fruits_example_detector_run_rgb8_with_params(
        detector,
        input_rgb_u8,
        &fruits_example_yolo_params,
        out_detections,
        out_capacity
    );
}

int32_t fruits_example_detector_run_rgb8_resize_with_params(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    int32_t src_w,
    int32_t src_h,
    int32_t src_stride_bytes,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *out_detections,
    int32_t out_capacity
) {
    int32_t i;
    if (detector == NULL || input_rgb_u8 == NULL || src_w <= 0 || src_h <= 0 || out_detections == NULL || out_capacity <= 0) {
        return -1;
    }
    kbnn_pack_thermo_input_u8_resize(&fruits_example_binary_input_params, input_rgb_u8, src_w, src_h, src_stride_bytes,
        (uint64_t *)kbnn_context_tensor_ptr(&detector->ctx, 1));
    if (kbnn_model_run_from(&detector->ctx, 1) != 0) {
        return -2;
    }
    for (i = 0; i < 2; ++i) {
        detector->heads[i].pred_q12 = fruits_example_output_i32(&detector->ctx, fruits_example_yolo_head_output_indices[i]);
    }
    return kbnn_yolo_postprocess(
        detector->heads,
        2,
        fruits_example_yolo_anchors,
        (int32_t)(sizeof(fruits_example_yolo_anchors) / sizeof(fruits_example_yolo_anchors[0]) / 2u),
        (params != NULL) ? params : &fruits_example_yolo_params,
        detector->scratch,
        FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS,
        detector->keep_indices,
        FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS,
        out_detections,
        out_capacity
    );
}

int32_t fruits_example_detector_run_rgb8_resize(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    int32_t src_w,
    int32_t src_h,
    int32_t src_stride_bytes,
    kbnn_detection *out_detections,
    int32_t out_capacity
) {
    return fruits_example_detector_run_rgb8_resize_with_params(
        detector,
        input_rgb_u8,
        src_w,
        src_h,
        src_stride_bytes,
        &fruits_example_yolo_params,
        out_detections,
        out_capacity
    );
}

int32_t fruits_example_detector_run_f32(
    fruits_example_detector *detector,
    const float *input_rgb,
    kbnn_detection *out_detections,
    int32_t out_capacity
) {
    return fruits_example_detector_run_f32_with_params(
        detector,
        input_rgb,
        &fruits_example_yolo_params,
        out_detections,
        out_capacity
    );
}
