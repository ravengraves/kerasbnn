#ifndef FRUITS_EXAMPLE_YOLO_CONFIG_H
#define FRUITS_EXAMPLE_YOLO_CONFIG_H

#include <stdint.h>
#include "yolo_postprocess.h"

static const float fruits_example_yolo_anchors[] = {
#ifndef FRUITS_EXAMPLE_YOLO_INPUT_W
#define FRUITS_EXAMPLE_YOLO_INPUT_W 256
#endif
#ifndef FRUITS_EXAMPLE_YOLO_INPUT_H
#define FRUITS_EXAMPLE_YOLO_INPUT_H 256
#endif
#ifndef FRUITS_EXAMPLE_YOLO_NUM_CLASSES
#define FRUITS_EXAMPLE_YOLO_NUM_CLASSES 4
#endif
#ifndef FRUITS_EXAMPLE_YOLO_HEAD_COUNT
#define FRUITS_EXAMPLE_YOLO_HEAD_COUNT 2
#endif
#ifndef FRUITS_EXAMPLE_YOLO_MAX_DETECTIONS
#define FRUITS_EXAMPLE_YOLO_MAX_DETECTIONS 256
#endif
#ifndef FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS
#define FRUITS_EXAMPLE_YOLO_SCRATCH_DETECTIONS 4096
#endif
#ifndef FRUITS_EXAMPLE_YOLO_BBOX_OUTPUT_SCALE
#define FRUITS_EXAMPLE_YOLO_BBOX_OUTPUT_SCALE 2.0f
#endif
#ifndef FRUITS_EXAMPLE_YOLO_OBJ_OUTPUT_SCALE
#define FRUITS_EXAMPLE_YOLO_OBJ_OUTPUT_SCALE 64.0f
#endif
#ifndef FRUITS_EXAMPLE_YOLO_CLS_OUTPUT_SCALE
#define FRUITS_EXAMPLE_YOLO_CLS_OUTPUT_SCALE 32.0f
#endif
#ifndef FRUITS_EXAMPLE_YOLO_LOGIT_SCALE
#define FRUITS_EXAMPLE_YOLO_LOGIT_SCALE 32.0f
#endif
#ifndef FRUITS_EXAMPLE_YOLO_SCORE_THRESHOLD
#define FRUITS_EXAMPLE_YOLO_SCORE_THRESHOLD 0.001f
#endif
#ifndef FRUITS_EXAMPLE_YOLO_OBJECTNESS_THRESHOLD
#define FRUITS_EXAMPLE_YOLO_OBJECTNESS_THRESHOLD 0.001f
#endif
#ifndef FRUITS_EXAMPLE_YOLO_NMS_IOU_THRESHOLD
#define FRUITS_EXAMPLE_YOLO_NMS_IOU_THRESHOLD 0.35f
#endif
#ifndef FRUITS_EXAMPLE_YOLO_PER_CLASS_NMS
#define FRUITS_EXAMPLE_YOLO_PER_CLASS_NMS 1
#endif
#ifndef FRUITS_EXAMPLE_YOLO_CLASS_FILTER_MASK
#define FRUITS_EXAMPLE_YOLO_CLASS_FILTER_MASK 0ull
#endif
#ifndef FRUITS_EXAMPLE_YOLO_USE_CLASS_FILTER
#define FRUITS_EXAMPLE_YOLO_USE_CLASS_FILTER 0
#endif
#ifndef FRUITS_EXAMPLE_YOLO_CLASS_MODE
#define FRUITS_EXAMPLE_YOLO_CLASS_MODE KBNN_YOLO_CLASS_SOFTMAX
#endif
#ifndef FRUITS_EXAMPLE_YOLO_BOX_MODE
#define FRUITS_EXAMPLE_YOLO_BOX_MODE KBNN_YOLO_BOX_LINEAR
#endif

    14.0799999f, 17.9200001f,
    25.0666656f, 38.5709229f,
    37.1199989f, 46.9333344f,
    49.4933319f, 82.8011856f,
    89.5999985f, 117.760002f,
    129.919998f, 138.240005f,
};

static const int32_t fruits_example_yolo_head_0_anchor_indices[] = {0, 1, 2};
static const int32_t fruits_example_yolo_head_1_anchor_indices[] = {3, 4, 5};

static const int32_t fruits_example_yolo_head_grid_h[] = {16, 8};
static const int32_t fruits_example_yolo_head_grid_w[] = {16, 8};
static const int32_t fruits_example_yolo_head_anchor_count[] = {3, 3};
static const int32_t fruits_example_yolo_head_output_indices[] = {1, 0};
static const char *fruits_example_yolo_output_names[] = {
    "pred_26",
    "pred_13",
};

static const char *fruits_example_yolo_class_names[] = {
    "banana",
    "dragon fruit",
    "pineapple",
    "snake fruit",
};

static const kbnn_yolo_postprocess_params fruits_example_yolo_params = {
    .input_w = FRUITS_EXAMPLE_YOLO_INPUT_W,
    .input_h = FRUITS_EXAMPLE_YOLO_INPUT_H,
    .num_classes = FRUITS_EXAMPLE_YOLO_NUM_CLASSES,
    .bbox_output_scale = FRUITS_EXAMPLE_YOLO_BBOX_OUTPUT_SCALE,
    .obj_output_scale = FRUITS_EXAMPLE_YOLO_OBJ_OUTPUT_SCALE,
    .cls_output_scale = FRUITS_EXAMPLE_YOLO_CLS_OUTPUT_SCALE,
    .logit_scale = FRUITS_EXAMPLE_YOLO_LOGIT_SCALE,
    .score_threshold = FRUITS_EXAMPLE_YOLO_SCORE_THRESHOLD,
    .objectness_threshold = FRUITS_EXAMPLE_YOLO_OBJECTNESS_THRESHOLD,
    .nms_iou_threshold = FRUITS_EXAMPLE_YOLO_NMS_IOU_THRESHOLD,
    .max_detections = FRUITS_EXAMPLE_YOLO_MAX_DETECTIONS,
    .per_class_nms = FRUITS_EXAMPLE_YOLO_PER_CLASS_NMS,
    .class_filter_mask = FRUITS_EXAMPLE_YOLO_CLASS_FILTER_MASK,
    .use_class_filter = FRUITS_EXAMPLE_YOLO_USE_CLASS_FILTER,
    .class_mode = FRUITS_EXAMPLE_YOLO_CLASS_MODE,
    .box_mode = FRUITS_EXAMPLE_YOLO_BOX_MODE,
};

#endif
