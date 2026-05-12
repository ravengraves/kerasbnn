#ifndef FRUITS_EXAMPLE_DETECTOR_H
#define FRUITS_EXAMPLE_DETECTOR_H

#include <stddef.h>
#include <stdint.h>

#include "kerasbnn.h"
#include "yolo_postprocess.h"
#include "fruits_example_yolo_config.h"

/*
 * Minimal deployment API for the exported YOLO detector.
 *
 * Input contract:
 * - `run_f32` expects a contiguous float32 HWC RGB image in [0, 1]
 * - `run_rgb8` expects a contiguous uint8 HWC RGB image at model size
 * - `run_rgb8_resize` accepts arbitrary RGB size and resizes internally
 *
 * Runtime contract:
 * - caller owns the arena buffer
 * - caller owns the output detection buffer
 * - returned value is detection count on success, negative on error
 */

typedef struct {
    kbnn_context ctx;
    kbnn_yolo_head heads[FRUITS_EXAMPLE_YOLO_HEAD_COUNT];
    kbnn_detection scratch[FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS];
    int32_t keep_indices[FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS];
    float input_rgb_f32[FRUITS_EXAMPLE_YOLO_INPUT_W * FRUITS_EXAMPLE_YOLO_INPUT_H * 3];
} fruits_example_detector;

size_t fruits_example_detector_arena_bytes(void);
int32_t fruits_example_detector_input_w(void);
int32_t fruits_example_detector_input_h(void);
int32_t fruits_example_detector_num_classes(void);
const char *fruits_example_detector_class_name(int32_t class_id);
const kbnn_yolo_postprocess_params *fruits_example_detector_default_params(void);
int fruits_example_detector_init(fruits_example_detector *detector, void *arena, size_t arena_bytes);
void fruits_example_detector_release(fruits_example_detector *detector);
int32_t fruits_example_detector_run_f32(
    fruits_example_detector *detector,
    const float *input_rgb,
    kbnn_detection *out_detections,
    int32_t out_capacity
);
int32_t fruits_example_detector_run_f32_with_params(
    fruits_example_detector *detector,
    const float *input_rgb,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *out_detections,
    int32_t out_capacity
);
int32_t fruits_example_detector_run_rgb8(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    kbnn_detection *out_detections,
    int32_t out_capacity
);
int32_t fruits_example_detector_run_rgb8_with_params(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *out_detections,
    int32_t out_capacity
);
int32_t fruits_example_detector_run_rgb8_resize(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    int32_t src_w,
    int32_t src_h,
    int32_t src_stride_bytes,
    kbnn_detection *out_detections,
    int32_t out_capacity
);
int32_t fruits_example_detector_run_rgb8_resize_with_params(
    fruits_example_detector *detector,
    const uint8_t *input_rgb_u8,
    int32_t src_w,
    int32_t src_h,
    int32_t src_stride_bytes,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *out_detections,
    int32_t out_capacity
);

#endif /* FRUITS_EXAMPLE_DETECTOR_H */
