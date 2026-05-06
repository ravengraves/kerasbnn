#include "yolo_postprocess.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static float kbnn_yolo_clipf(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static float kbnn_yolo_sigmoid(float x) {
    x = kbnn_yolo_clipf(x, -20.0f, 20.0f);
    return 1.0f / (1.0f + expf(-x));
}

static float kbnn_yolo_piecewise_ratio(float u) {
    return (u <= 0.5f) ? (2.0f * u) : (1.0f + 4.0f * (u - 0.5f));
}

static float kbnn_yolo_uint01_from_linear(float raw, float output_scale) {
    const float sat = kbnn_yolo_clipf(raw / output_scale, -127.0f, 127.0f);
    return (sat + 127.0f) / 254.0f;
}

static float kbnn_yolo_objectness_linear(float raw, float output_scale, float logit_scale) {
    const float sat = kbnn_yolo_clipf(raw / output_scale, -127.0f, 127.0f);
    return kbnn_yolo_sigmoid(sat / logit_scale);
}

static float kbnn_yolo_class_logit_linear(float raw, float output_scale, float logit_scale) {
    const float sat = kbnn_yolo_clipf(raw / output_scale, -127.0f, 127.0f);
    return sat / logit_scale;
}

static float kbnn_yolo_raw_class_logit(
    float raw,
    const kbnn_yolo_postprocess_params *params
) {
    if (params->box_mode == KBNN_YOLO_BOX_LINEAR) {
        return kbnn_yolo_class_logit_linear(raw, params->cls_output_scale, params->logit_scale);
    }
    return raw / params->logit_scale;
}

static int kbnn_yolo_cmp_score_desc(const void *lhs, const void *rhs) {
    const kbnn_detection *a = (const kbnn_detection *) lhs;
    const kbnn_detection *b = (const kbnn_detection *) rhs;
    if (a->score > b->score) {
        return -1;
    }
    if (a->score < b->score) {
        return 1;
    }
    if (a->class_id < b->class_id) {
        return -1;
    }
    if (a->class_id > b->class_id) {
        return 1;
    }
    return 0;
}

static int32_t kbnn_yolo_copy_kept(
    const kbnn_detection *scratch,
    const int32_t *keep_indices,
    int32_t keep_count,
    kbnn_detection *out_detections,
    int32_t out_capacity
) {
    int32_t i;
    const int32_t count = (keep_count < out_capacity) ? keep_count : out_capacity;
    for (i = 0; i < count; ++i) {
        out_detections[i] = scratch[keep_indices[i]];
    }
    return count;
}

int32_t kbnn_yolo_decode_heads(
    const kbnn_yolo_head *heads,
    int32_t head_count,
    const float *anchors_xy,
    int32_t anchor_count,
    const kbnn_yolo_postprocess_params *params,
    kbnn_detection *detections,
    int32_t detections_capacity
) {
    int32_t head_idx;
    int32_t out_count = 0;
    if (heads == NULL || anchors_xy == NULL || params == NULL || detections == NULL || detections_capacity <= 0) {
        return 0;
    }
    for (head_idx = 0; head_idx < head_count; ++head_idx) {
        const kbnn_yolo_head *head = &heads[head_idx];
        const int32_t stride = 5 + params->num_classes;
        int32_t gy;
        if (head->pred_q12 == NULL || head->anchor_indices == NULL) {
            continue;
        }
        for (gy = 0; gy < head->grid_h; ++gy) {
            int32_t gx;
            for (gx = 0; gx < head->grid_w; ++gx) {
                int32_t anchor_slot;
                for (anchor_slot = 0; anchor_slot < head->anchors_per_cell; ++anchor_slot) {
                    const int32_t anchor_id = head->anchor_indices[anchor_slot];
                    const size_t base = (((size_t) gy * (size_t) head->grid_w + (size_t) gx)
                        * (size_t) head->anchors_per_cell + (size_t) anchor_slot) * (size_t) stride;
                    const float anchor_w = anchors_xy[(size_t) anchor_id * 2u + 0u];
                    const float anchor_h = anchors_xy[(size_t) anchor_id * 2u + 1u];
                    const float raw_x = (float) head->pred_q12[base + 0u] / (float) KBNN_FX_ONE;
                    const float raw_y = (float) head->pred_q12[base + 1u] / (float) KBNN_FX_ONE;
                    const float raw_w = (float) head->pred_q12[base + 2u] / (float) KBNN_FX_ONE;
                    const float raw_h = (float) head->pred_q12[base + 3u] / (float) KBNN_FX_ONE;
                    const float raw_obj = (float) head->pred_q12[base + 4u] / (float) KBNN_FX_ONE;
                    float cx;
                    float cy;
                    float bw;
                    float bh;
                    float obj;
                    int32_t cls;
                    float cls_score;
                    float score;
                    int32_t best_id = 0;
                    float best_logit = 0.0f;
                    if (anchor_id < 0 || anchor_id >= anchor_count) {
                        continue;
                    }
                    if (params->box_mode == KBNN_YOLO_BOX_YOLO) {
                        cx = (kbnn_yolo_sigmoid(raw_x) + (float) gx) / (float) head->grid_w;
                        cy = (kbnn_yolo_sigmoid(raw_y) + (float) gy) / (float) head->grid_h;
                        bw = expf(kbnn_yolo_clipf(raw_w, -6.0f, 6.0f)) * anchor_w / (float) params->input_w;
                        bh = expf(kbnn_yolo_clipf(raw_h, -6.0f, 6.0f)) * anchor_h / (float) params->input_h;
                        obj = kbnn_yolo_sigmoid(raw_obj / params->logit_scale);
                    } else if (params->box_mode == KBNN_YOLO_BOX_LINEAR_FLOAT) {
                        cx = (kbnn_yolo_clipf(raw_x, 0.0f, 1.0f) + (float) gx) / (float) head->grid_w;
                        cy = (kbnn_yolo_clipf(raw_y, 0.0f, 1.0f) + (float) gy) / (float) head->grid_h;
                        bw = kbnn_yolo_piecewise_ratio(kbnn_yolo_clipf(raw_w, 0.0f, 1.0f)) * anchor_w / (float) params->input_w;
                        bh = kbnn_yolo_piecewise_ratio(kbnn_yolo_clipf(raw_h, 0.0f, 1.0f)) * anchor_h / (float) params->input_h;
                        obj = kbnn_yolo_sigmoid(raw_obj / params->logit_scale);
                    } else {
                        cx = (kbnn_yolo_uint01_from_linear(raw_x, params->bbox_output_scale) + (float) gx) / (float) head->grid_w;
                        cy = (kbnn_yolo_uint01_from_linear(raw_y, params->bbox_output_scale) + (float) gy) / (float) head->grid_h;
                        bw = kbnn_yolo_piecewise_ratio(kbnn_yolo_uint01_from_linear(raw_w, params->bbox_output_scale)) * anchor_w / (float) params->input_w;
                        bh = kbnn_yolo_piecewise_ratio(kbnn_yolo_uint01_from_linear(raw_h, params->bbox_output_scale)) * anchor_h / (float) params->input_h;
                        obj = kbnn_yolo_objectness_linear(raw_obj, params->obj_output_scale, params->logit_scale);
                    }
                    if (params->num_classes <= 0) {
                        cls = 0;
                        cls_score = 1.0f;
                    } else if (params->class_mode == KBNN_YOLO_CLASS_SOFTMAX) {
                        float denom = 0.0f;
                        for (cls = 0; cls < params->num_classes; ++cls) {
                            const float raw_cls = (float) head->pred_q12[base + 5u + (size_t) cls] / (float) KBNN_FX_ONE;
                            const float logit = kbnn_yolo_raw_class_logit(raw_cls, params);
                            if (cls == 0 || logit > best_logit) {
                                best_logit = logit;
                                best_id = cls;
                            }
                        }
                        for (cls = 0; cls < params->num_classes; ++cls) {
                            const float raw_cls = (float) head->pred_q12[base + 5u + (size_t) cls] / (float) KBNN_FX_ONE;
                            denom += expf(kbnn_yolo_raw_class_logit(raw_cls, params) - best_logit);
                        }
                        cls = best_id;
                        cls_score = 1.0f / ((denom > 1e-6f) ? denom : 1.0f);
                    } else {
                        float best_prob = 0.0f;
                        for (cls = 0; cls < params->num_classes; ++cls) {
                            const float raw_cls = (float) head->pred_q12[base + 5u + (size_t) cls] / (float) KBNN_FX_ONE;
                            const float prob = kbnn_yolo_sigmoid(kbnn_yolo_raw_class_logit(raw_cls, params));
                            if (cls == 0 || prob > best_prob) {
                                best_prob = prob;
                                best_id = cls;
                            }
                        }
                        cls = best_id;
                        cls_score = best_prob;
                    }
                    score = obj * cls_score;
                    if (obj < params->objectness_threshold || score < params->score_threshold) {
                        continue;
                    }
                    if (out_count >= detections_capacity) {
                        goto done;
                    }
                    detections[out_count].cx = cx;
                    detections[out_count].cy = cy;
                    detections[out_count].w = bw;
                    detections[out_count].h = bh;
                    detections[out_count].score = score;
                    detections[out_count].class_id = cls;
                    ++out_count;
                }
            }
        }
    }

done:
    return out_count;
}

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
) {
    int32_t decoded_count;
    int32_t keep_count;
    if (scratch_detections == NULL || keep_indices == NULL || out_detections == NULL || out_capacity <= 0) {
        return 0;
    }
    decoded_count = kbnn_yolo_decode_heads(
        heads,
        head_count,
        anchors_xy,
        anchor_count,
        params,
        scratch_detections,
        scratch_capacity
    );
    if (decoded_count <= 0) {
        return 0;
    }
    qsort(scratch_detections, (size_t) decoded_count, sizeof(kbnn_detection), kbnn_yolo_cmp_score_desc);
    if (params->per_class_nms) {
        const int32_t top_k = (params->max_detections > 0 && params->max_detections < out_capacity)
            ? params->max_detections
            : out_capacity;
        keep_count = kbnn_nms(
            scratch_detections,
            decoded_count,
            params->nms_iou_threshold,
            top_k,
            keep_indices
        );
        if (keep_count > keep_capacity) {
            keep_count = keep_capacity;
        }
        return kbnn_yolo_copy_kept(scratch_detections, keep_indices, keep_count, out_detections, out_capacity);
    }
    {
        int32_t kept = 0;
        int32_t i;
        for (i = 0; i < decoded_count && kept < out_capacity; ++i) {
            int suppress = 0;
            int32_t j;
            for (j = 0; j < kept; ++j) {
                const kbnn_detection *a = &out_detections[j];
                const kbnn_detection *b = &scratch_detections[i];
                const float ax0 = a->cx - 0.5f * a->w;
                const float ay0 = a->cy - 0.5f * a->h;
                const float ax1 = a->cx + 0.5f * a->w;
                const float ay1 = a->cy + 0.5f * a->h;
                const float bx0 = b->cx - 0.5f * b->w;
                const float by0 = b->cy - 0.5f * b->h;
                const float bx1 = b->cx + 0.5f * b->w;
                const float by1 = b->cy + 0.5f * b->h;
                const float ix0 = (ax0 > bx0) ? ax0 : bx0;
                const float iy0 = (ay0 > by0) ? ay0 : by0;
                const float ix1 = (ax1 < bx1) ? ax1 : bx1;
                const float iy1 = (ay1 < by1) ? ay1 : by1;
                const float iw = ix1 - ix0;
                const float ih = iy1 - iy0;
                if (iw > 0.0f && ih > 0.0f) {
                    const float inter = iw * ih;
                    const float union_area = a->w * a->h + b->w * b->h - inter;
                    if (union_area > 1e-6f && inter / union_area > params->nms_iou_threshold) {
                        suppress = 1;
                        break;
                    }
                }
            }
            if (!suppress) {
                out_detections[kept++] = scratch_detections[i];
                if (params->max_detections > 0 && kept >= params->max_detections) {
                    break;
                }
            }
        }
        return kept;
    }
}
