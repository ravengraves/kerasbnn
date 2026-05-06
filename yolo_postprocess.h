#ifndef KERASBNN_YOLO_POSTPROCESS_H
#define KERASBNN_YOLO_POSTPROCESS_H

#include <stdint.h>

#include "kerasbnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * YOLO-specific postprocessing helpers for exported `kerasbnn` detector models.
 *
 * These utilities sit on top of the generic runtime:
 * - model execution still goes through `kerasbnn.c` / generated `*_runtime.c`
 * - these helpers only decode raw output tensors into `kbnn_detection` records
 *   and run NMS
 *
 * Expected head tensor layout:
 * - contiguous NHWAxC int32 tensors returned by `kbnn_context_get_output_i32()`
 * - shape: [grid_h, grid_w, anchors_per_cell, 5 + num_classes]
 * - values are the exported raw detector outputs in the same Q12 fixed-point
 *   scale emitted by the generic runtime
 *
 * Supported decode modes:
 * - `KBNN_YOLO_BOX_LINEAR`
 * - `KBNN_YOLO_BOX_LINEAR_FLOAT`
 * - `KBNN_YOLO_BOX_YOLO`
 *
 * Supported class score modes:
 * - `KBNN_YOLO_CLASS_BCE`
 * - `KBNN_YOLO_CLASS_SOFTMAX`
 *
 * The API is allocation-free. Callers provide scratch/output buffers.
 */

typedef enum {
    KBNN_YOLO_CLASS_BCE = 0,
    KBNN_YOLO_CLASS_SOFTMAX = 1,
} kbnn_yolo_class_mode;

typedef enum {
    KBNN_YOLO_BOX_LINEAR = 0,
    KBNN_YOLO_BOX_LINEAR_FLOAT = 1,
    KBNN_YOLO_BOX_YOLO = 2,
} kbnn_yolo_box_mode;

typedef struct {
    const int32_t *pred_q12;
    int32_t grid_h;
    int32_t grid_w;
    int32_t anchors_per_cell;
    const int32_t *anchor_indices;
} kbnn_yolo_head;

typedef struct {
    int32_t input_w;
    int32_t input_h;
    int32_t num_classes;
    float bbox_output_scale;
    float obj_output_scale;
    float cls_output_scale;
    float logit_scale;
    float score_threshold;
    float objectness_threshold;
    float nms_iou_threshold;
    int32_t max_detections;
    int32_t per_class_nms;
    kbnn_yolo_class_mode class_mode;
    kbnn_yolo_box_mode box_mode;
} kbnn_yolo_postprocess_params;

int32_t kbnn_yolo_decode_heads(
    const kbnn_yolo_head *heads,
    int32_t head_count,
    const float *anchors_xy,
    int32_t anchor_count,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *detections,
    int32_t detections_capacity
);

int32_t kbnn_yolo_postprocess(
    const kbnn_yolo_head *heads,
    int32_t head_count,
    const float *anchors_xy,
    int32_t anchor_count,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *scratch_detections,
    int32_t scratch_capacity,
    int32_t *keep_indices,
    int32_t keep_capacity,
    kbnn_detection *out_detections,
    int32_t out_capacity
);

#ifdef __cplusplus
}
#endif

#endif
