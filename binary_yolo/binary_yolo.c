#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PACK_WORDS(channels) (((channels) + 63) / 64)

#define INPUT_H 320
#define INPUT_W 320
#define INPUT_C 3
#define THERMO_LEVELS 4
#define BINARY_INPUT_C (INPUT_C * THERMO_LEVELS)
#define NUM_CLASSES 20
#define NUM_HEAD_ANCHORS 3
#define HEAD_LOGITS (NUM_CLASSES + 5)
#define NUM_RECORDS 37

#define P3_H 40
#define P3_W 40
#define P4_H 20
#define P4_W 20
#define P5_H 10
#define P5_W 10

#define MAIN_BUF_WORDS ((size_t) INPUT_H * (size_t) INPUT_W * (size_t) PACK_WORDS(32))
#define ROUTE_P3_WORDS ((size_t) P3_H * (size_t) P3_W * (size_t) PACK_WORDS(160))
#define ROUTE_P4_WORDS ((size_t) P4_H * (size_t) P4_W * (size_t) PACK_WORDS(320))
#define X10_WORDS ((size_t) P5_H * (size_t) P5_W * (size_t) PACK_WORDS(1280))

#define P3_HEAD_COUNT ((size_t) P3_H * (size_t) P3_W * (size_t) NUM_HEAD_ANCHORS * (size_t) HEAD_LOGITS)
#define P4_HEAD_COUNT ((size_t) P4_H * (size_t) P4_W * (size_t) NUM_HEAD_ANCHORS * (size_t) HEAD_LOGITS)
#define P5_HEAD_COUNT ((size_t) P5_H * (size_t) P5_W * (size_t) NUM_HEAD_ANCHORS * (size_t) HEAD_LOGITS)

#define MAX_CANDIDATES 6300
#define MAX_FINAL_DETECTIONS 256
#define CONV_OC_BLOCK 16

#define FILE_MAGIC "BYOLOv1"
#define FILE_MAGIC_SIZE 8
#define FILE_VERSION 5u

#define HEAD_DUMP_MAGIC "BYHDMP1"

#define RECORD_BINARY_CONV 1u
#define RECORD_RAW_PROJ 2u
#define RECORD_QDENSE 3u
#define RECORD_QDENSE_OUT 4u

#define CMP_MODE_GE 0u
#define CMP_MODE_LE 1u
#define CMP_MODE_CONST_POS 2u
#define CMP_MODE_CONST_NEG 3u

#define RAW_PROJ_MODE_INC 0u
#define RAW_PROJ_MODE_DEC 1u
#define RAW_PROJ_MODE_CONST_MIN 2u
#define RAW_PROJ_MODE_CONST_MAX 3u
#define RAW_PROJ_STEPS 15

typedef struct {
    const char *name;
    int idx;
    int in_h;
    int in_w;
    int in_c;
    int out_h;
    int out_w;
    int out_c;
    int kernel;
    int stride;
    int in_words;
    int kernel_elems;
    int32_t raw_scale_q;
    int32_t input_limit_q;
    int32_t input_step_q;
    int32_t center_limit_q;
    int32_t center_step_q;
    int32_t norm_limit_q;
    int32_t norm_step_q;
    uint64_t *weights;
    uint8_t *cmp_mode;
    int16_t *cmp_threshold;
    void *reserved0;
    void *reserved1;
    void *reserved2;
} BinaryConvLayer;

typedef struct {
    const char *name;
    int idx;
    int in_h;
    int in_w;
    int in_c;
    int out_h;
    int out_w;
    int out_c;
    int in_words;
    int kernel_elems;
    uint64_t *weights;
    uint8_t *mode;
    int16_t *thresholds;
} RawProjLayer;

typedef struct {
    const char *name;
    int idx;
    int in_dim;
    int out_dim;
    int8_t *weights;
    float *bias;
} QDenseLayer;

typedef struct {
    const char *name;
    int idx;
    int in_dim;
    int out_dim;
    float output_scale;
    float inv_scale_div8;
    int8_t *weights;
    int8_t *bias_q;
} QDenseOutLayer;

typedef struct {
    int h;
    int w;
    int c;
    int words;
    uint64_t *data;
} PackedMap;

typedef struct {
    uint16_t x1;
    uint16_t y1;
    uint16_t x2;
    uint16_t y2;
    uint16_t score_q15;
    float score;
    uint8_t class_id;
} Detection;

static BinaryConvLayer g_binary_layers[16] = {
    {"stage1_conv", 0, 320, 320, 12, 320, 320, 32, 3, 1, PACK_WORDS(12), 3 * 3 * 12, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"stage2_conv", 1, 320, 320, 32, 160, 160, 40, 3, 2, PACK_WORDS(32), 3 * 3 * 32, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"stage3_conv", 2, 160, 160, 40, 80, 80, 80, 3, 2, PACK_WORDS(40), 3 * 3 * 40, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"stage4_conv", 3, 80, 80, 80, 40, 40, 160, 3, 2, PACK_WORDS(80), 3 * 3 * 80, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"stage5_conv", 4, 40, 40, 160, 20, 20, 320, 3, 2, PACK_WORDS(160), 3 * 3 * 160, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"stage6_conv", 5, 20, 20, 320, 10, 10, 640, 3, 2, PACK_WORDS(320), 3 * 3 * 320, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"stage7_conv", 6, 10, 10, 640, 10, 10, 1280, 3, 1, PACK_WORDS(640), 3 * 3 * 640, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"stage8_conv", 7, 10, 10, 1280, 10, 10, 640, 1, 1, PACK_WORDS(1280), 1 * 1 * 1280, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"p5_feat_conv", 8, 10, 10, 640, 10, 10, 320, 3, 1, PACK_WORDS(640), 3 * 3 * 640, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"up4_reduce_conv", 9, 10, 10, 640, 10, 10, 320, 1, 1, PACK_WORDS(640), 1 * 1 * 640, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"p4_feat_conv", 10, 20, 20, 640, 20, 20, 320, 3, 1, PACK_WORDS(640), 3 * 3 * 640, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"up3_reduce_conv", 11, 20, 20, 320, 20, 20, 160, 1, 1, PACK_WORDS(320), 1 * 1 * 320, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"p3_feat_conv", 12, 40, 40, 320, 40, 40, 160, 3, 1, PACK_WORDS(320), 3 * 3 * 320, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"p3_pre_conv", 13, 40, 40, 160, 40, 40, 320, 3, 1, PACK_WORDS(160), 3 * 3 * 160, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"p4_pre_conv", 14, 20, 20, 320, 20, 20, 320, 3, 1, PACK_WORDS(320), 3 * 3 * 320, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
    {"p5_pre_conv", 15, 10, 10, 320, 10, 10, 320, 3, 1, PACK_WORDS(320), 3 * 3 * 320, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL},
};

static RawProjLayer g_raw_proj_layers[3] = {
    {"p3_proj_conv", 16, 40, 40, 320, 40, 40, 144, PACK_WORDS(320), 320, NULL, NULL, NULL},
    {"p4_proj_conv", 17, 20, 20, 320, 20, 20, 144, PACK_WORDS(320), 320, NULL, NULL, NULL},
    {"p5_proj_conv", 18, 10, 10, 320, 10, 10, 144, PACK_WORDS(320), 320, NULL, NULL, NULL},
};

static QDenseLayer g_qdense_layers[9] = {
    {"p3_obj_hidden_dense", 19, 48, 128, NULL, NULL},
    {"p3_obj_remap_dense", 20, 128, 64, NULL, NULL},
    {"p3_cls_hidden_dense", 21, 48, 128, NULL, NULL},
    {"p4_obj_hidden_dense", 25, 48, 128, NULL, NULL},
    {"p4_obj_remap_dense", 26, 128, 64, NULL, NULL},
    {"p4_cls_hidden_dense", 27, 48, 128, NULL, NULL},
    {"p5_obj_hidden_dense", 31, 48, 128, NULL, NULL},
    {"p5_obj_remap_dense", 32, 128, 64, NULL, NULL},
    {"p5_cls_hidden_dense", 33, 48, 128, NULL, NULL},
};

static QDenseOutLayer g_qdense_out_layers[9] = {
    {"p3_bbox", 22, 48, 4, 2.0f, 1.0f / 16.0f, NULL, NULL},
    {"p3_obj", 23, 64, 1, 64.0f, 1.0f / 512.0f, NULL, NULL},
    {"p3_cls", 24, 128, 20, 32.0f, 1.0f / 256.0f, NULL, NULL},
    {"p4_bbox", 28, 48, 4, 2.0f, 1.0f / 16.0f, NULL, NULL},
    {"p4_obj", 29, 64, 1, 64.0f, 1.0f / 512.0f, NULL, NULL},
    {"p4_cls", 30, 128, 20, 32.0f, 1.0f / 256.0f, NULL, NULL},
    {"p5_bbox", 34, 48, 4, 2.0f, 1.0f / 16.0f, NULL, NULL},
    {"p5_obj", 35, 64, 1, 64.0f, 1.0f / 512.0f, NULL, NULL},
    {"p5_cls", 36, 128, 20, 32.0f, 1.0f / 256.0f, NULL, NULL},
};

static const char *g_class_names[NUM_CLASSES] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
};

static const float g_anchors[9][2] = {
    {15.3599987f, 27.9517956f},
    {45.4400024f, 39.5180664f},
    {26.2399960f, 70.8266754f},
    {49.2800064f, 132.2666626f},
    {85.1200027f, 82.7733307f},
    {97.2800064f, 195.0802307f},
    {193.1531372f, 131.5295715f},
    {168.2384949f, 247.4666595f},
    {284.1599731f, 274.6987915f},
};

static uint64_t g_buf_a[MAIN_BUF_WORDS];
static uint64_t g_buf_b[MAIN_BUF_WORDS];
static uint64_t g_route_p3[ROUTE_P3_WORDS];
static uint64_t g_route_p4[ROUTE_P4_WORDS];
static uint64_t g_x10[X10_WORDS];
static uint8_t g_input_u8[INPUT_H * INPUT_W * INPUT_C];
static float g_input_f32[INPUT_H * INPUT_W * INPUT_C];

static int8_t g_p3_head[P3_HEAD_COUNT];
static int8_t g_p4_head[P4_HEAD_COUNT];
static int8_t g_p5_head[P5_HEAD_COUNT];

static float g_sigmoid_lut[255];
static float g_u01_lut[255];
static float g_ratio_lut[255];

static uint8_t *g_weight_arena = NULL;
static size_t g_weight_arena_size = 0u;
static size_t g_weight_arena_used = 0u;

typedef enum {
    PROFILE_STAGE1_CONV = 0,
    PROFILE_STAGE2_CONV,
    PROFILE_STAGE3_CONV,
    PROFILE_STAGE4_CONV,
    PROFILE_STAGE5_CONV,
    PROFILE_STAGE6_CONV,
    PROFILE_STAGE7_CONV,
    PROFILE_STAGE8_CONV,
    PROFILE_P5_FEAT_CONV,
    PROFILE_P5_PRE_CONV,
    PROFILE_P5_HEAD,
    PROFILE_UP4_REDUCE_CONV,
    PROFILE_UP4_UPSAMPLE,
    PROFILE_P4_CONCAT,
    PROFILE_P4_FEAT_CONV,
    PROFILE_P4_PRE_CONV,
    PROFILE_P4_HEAD,
    PROFILE_UP3_REDUCE_CONV,
    PROFILE_UP3_UPSAMPLE,
    PROFILE_P3_CONCAT,
    PROFILE_P3_FEAT_CONV,
    PROFILE_P3_PRE_CONV,
    PROFILE_P3_HEAD,
    PROFILE_POST_P3_DECODE,
    PROFILE_POST_P4_DECODE,
    PROFILE_POST_P5_DECODE,
    PROFILE_POST_NMS,
    PROFILE_COUNT
} ProfileStage;

static const char *g_profile_names[PROFILE_COUNT] = {
    "stage1_conv",
    "stage2_conv",
    "stage3_conv",
    "stage4_conv",
    "stage5_conv",
    "stage6_conv",
    "stage7_conv",
    "stage8_conv",
    "p5_feat_conv",
    "p5_pre_conv",
    "p5_head",
    "up4_reduce_conv",
    "up4_upsample",
    "p4_concat",
    "p4_feat_conv",
    "p4_pre_conv",
    "p4_head",
    "up3_reduce_conv",
    "up3_upsample",
    "p3_concat",
    "p3_feat_conv",
    "p3_pre_conv",
    "p3_head",
    "post_p3_decode",
    "post_p4_decode",
    "post_p5_decode",
    "post_nms",
};

static double g_profile_seconds[PROFILE_COUNT];
static int g_profile_enabled = 0;

static size_t packed_map_words(int h, int w, int c) {
    return (size_t) h * (size_t) w * (size_t) PACK_WORDS(c);
}

static size_t align_up_size(size_t value, size_t alignment) {
    size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

static void *weight_arena_alloc(size_t size, size_t alignment) {
    size_t offset = align_up_size(g_weight_arena_used, alignment);
    void *ptr;
    if (offset + size > g_weight_arena_size) {
        return NULL;
    }
    ptr = g_weight_arena + offset;
    g_weight_arena_used = offset + size;
    return ptr;
}

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static double profile_begin(void) {
    return g_profile_enabled ? wall_seconds() : 0.0;
}

static void profile_end(ProfileStage stage, double t0) {
    if (g_profile_enabled) {
        g_profile_seconds[stage] += wall_seconds() - t0;
    }
}

static void profile_reset(void) {
    memset(g_profile_seconds, 0, sizeof(g_profile_seconds));
}

static void profile_print_summary(double network_total, double post_total, int iterations) {
    int i;
    double grand_total = network_total + post_total;
    if (!g_profile_enabled || iterations <= 0 || grand_total <= 0.0) {
        return;
    }
    fprintf(stderr, "profile breakdown (avg ms / %% of total):\n");
    for (i = 0; i < PROFILE_COUNT; ++i) {
        double avg_ms = g_profile_seconds[i] * 1000.0 / (double) iterations;
        double pct = g_profile_seconds[i] / grand_total * 100.0;
        fprintf(stderr, "  %-16s %8.3f ms  %6.2f%%\n", g_profile_names[i], avg_ms, pct);
    }
}

static uint64_t word_mask_for_channels(int channels, int word_idx) {
    int start = word_idx * 64;
    int remain = channels - start;
    if (remain <= 0) {
        return 0ull;
    }
    if (remain >= 64) {
        return ~0ull;
    }
    return (1ull << remain) - 1ull;
}

static int active_bits_for_word(int channels, int word_idx) {
    int start = word_idx * 64;
    int remain = channels - start;
    if (remain <= 0) {
        return 0;
    }
    if (remain >= 64) {
        return 64;
    }
    return remain;
}

static uint64_t *packed_pixel(uint64_t *data, int width, int words, int y, int x) {
    return data + (((size_t) y * (size_t) width + (size_t) x) * (size_t) words);
}

static const uint64_t *packed_pixel_const(const uint64_t *data, int width, int words, int y, int x) {
    return data + (((size_t) y * (size_t) width + (size_t) x) * (size_t) words);
}

static int same_pad_before(int in_size, int kernel, int stride) {
    int out = (in_size + stride - 1) / stride;
    int total = (out - 1) * stride + kernel - in_size;
    if (total < 0) {
        total = 0;
    }
    return total / 2;
}

static int binary_affine_eval_from_raw(int raw, const BinaryConvLayer *layer, int out_idx) {
    switch (layer->cmp_mode[out_idx]) {
        case CMP_MODE_GE:
            return raw >= (int) layer->cmp_threshold[out_idx];
        case CMP_MODE_LE:
            return raw <= (int) layer->cmp_threshold[out_idx];
        case CMP_MODE_CONST_POS:
            return 1;
        case CMP_MODE_CONST_NEG:
            return 0;
        default:
            return 0;
    }
}

static void init_lookup_tables(void) {
    int q;
    for (q = -127; q <= 127; ++q) {
        float qf = (float) q;
        float u = (qf + 127.0f) / 254.0f;
        g_u01_lut[q + 127] = u;
        if (u <= 0.5f) {
            g_ratio_lut[q + 127] = 2.0f * u;
        } else {
            g_ratio_lut[q + 127] = 1.0f + 4.0f * (u - 0.5f);
        }
        g_sigmoid_lut[q + 127] = 1.0f / (1.0f + expf(-qf / 32.0f));
    }
}

static void trace_packed_map(const char *name, const PackedMap *map) {
    size_t pixel_count = (size_t) map->h * (size_t) map->w;
    size_t words = pixel_count * (size_t) map->words;
    size_t word_idx;
    uint64_t total_ones = 0ull;
    uint64_t total_bits = (uint64_t) pixel_count * (uint64_t) map->c;
    for (word_idx = 0; word_idx < words; ++word_idx) {
        total_ones += (uint64_t) __builtin_popcountll(map->data[word_idx]);
    }
    printf("%s packed: shape=%dx%dx%d ones=%llu bits=%llu\n",
           name,
           map->h,
           map->w,
           map->c,
           (unsigned long long) total_ones,
           (unsigned long long) total_bits);
}

static void trace_qtensor(const char *name, const int8_t *data, size_t count) {
    size_t idx;
    long long sum = 0;
    int min_v = 127;
    int max_v = -127;
    for (idx = 0; idx < count; ++idx) {
        int v = (int) data[idx];
        sum += (long long) v;
        if (v < min_v) {
            min_v = v;
        }
        if (v > max_v) {
            max_v = v;
        }
    }
    printf("%s q8: count=%zu min=%d max=%d sum=%lld\n", name, count, min_v, max_v, sum);
}

static void set_packed_output_bit(uint64_t *dst, int bit_index) {
    dst[bit_index >> 6] |= 1ull << (bit_index & 63);
}

static int conv_packed_out_channels(const BinaryConvLayer *layer) {
    return ((layer->out_c + CONV_OC_BLOCK - 1) / CONV_OC_BLOCK) * CONV_OC_BLOCK;
}

static size_t conv_weight_block_words(const BinaryConvLayer *layer) {
    return (size_t) layer->kernel * (size_t) layer->kernel * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK;
}

static size_t conv_weight_words(const BinaryConvLayer *layer) {
    return ((size_t) conv_packed_out_channels(layer) / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer);
}

static inline void finalize_conv_block(const BinaryConvLayer *layer, int out_base, const int *raw_acc, uint64_t *dst) {
    int lane_count = layer->out_c - out_base;
    int lane;
    if (lane_count > CONV_OC_BLOCK) {
        lane_count = CONV_OC_BLOCK;
    }
    for (lane = 0; lane < lane_count; ++lane) {
        int out_idx = out_base + lane;
        if (binary_affine_eval_from_raw(raw_acc[lane], layer, out_idx)) {
            set_packed_output_bit(dst, out_idx);
        }
    }
}

static inline void accum_lanes_tail(const uint64_t *weights, uint64_t src_word, uint64_t mask, int active, int *raw_acc, int lane_count) {
    int lane;
    for (lane = 0; lane < lane_count; ++lane) {
        uint64_t diff = (src_word ^ weights[lane]) & mask;
        raw_acc[lane] += active - 2 * (int) __builtin_popcountll(diff);
    }
}

#define ACCUM_LANES_16(weights, src_word, mask, active, raw_acc) \
    do { \
        uint64_t diff0 = ((src_word) ^ (weights)[0]) & (mask); \
        uint64_t diff1 = ((src_word) ^ (weights)[1]) & (mask); \
        uint64_t diff2 = ((src_word) ^ (weights)[2]) & (mask); \
        uint64_t diff3 = ((src_word) ^ (weights)[3]) & (mask); \
        uint64_t diff4 = ((src_word) ^ (weights)[4]) & (mask); \
        uint64_t diff5 = ((src_word) ^ (weights)[5]) & (mask); \
        uint64_t diff6 = ((src_word) ^ (weights)[6]) & (mask); \
        uint64_t diff7 = ((src_word) ^ (weights)[7]) & (mask); \
        uint64_t diff8 = ((src_word) ^ (weights)[8]) & (mask); \
        uint64_t diff9 = ((src_word) ^ (weights)[9]) & (mask); \
        uint64_t diff10 = ((src_word) ^ (weights)[10]) & (mask); \
        uint64_t diff11 = ((src_word) ^ (weights)[11]) & (mask); \
        uint64_t diff12 = ((src_word) ^ (weights)[12]) & (mask); \
        uint64_t diff13 = ((src_word) ^ (weights)[13]) & (mask); \
        uint64_t diff14 = ((src_word) ^ (weights)[14]) & (mask); \
        uint64_t diff15 = ((src_word) ^ (weights)[15]) & (mask); \
        (raw_acc)[0] += (active) - 2 * (int) __builtin_popcountll(diff0); \
        (raw_acc)[1] += (active) - 2 * (int) __builtin_popcountll(diff1); \
        (raw_acc)[2] += (active) - 2 * (int) __builtin_popcountll(diff2); \
        (raw_acc)[3] += (active) - 2 * (int) __builtin_popcountll(diff3); \
        (raw_acc)[4] += (active) - 2 * (int) __builtin_popcountll(diff4); \
        (raw_acc)[5] += (active) - 2 * (int) __builtin_popcountll(diff5); \
        (raw_acc)[6] += (active) - 2 * (int) __builtin_popcountll(diff6); \
        (raw_acc)[7] += (active) - 2 * (int) __builtin_popcountll(diff7); \
        (raw_acc)[8] += (active) - 2 * (int) __builtin_popcountll(diff8); \
        (raw_acc)[9] += (active) - 2 * (int) __builtin_popcountll(diff9); \
        (raw_acc)[10] += (active) - 2 * (int) __builtin_popcountll(diff10); \
        (raw_acc)[11] += (active) - 2 * (int) __builtin_popcountll(diff11); \
        (raw_acc)[12] += (active) - 2 * (int) __builtin_popcountll(diff12); \
        (raw_acc)[13] += (active) - 2 * (int) __builtin_popcountll(diff13); \
        (raw_acc)[14] += (active) - 2 * (int) __builtin_popcountll(diff14); \
        (raw_acc)[15] += (active) - 2 * (int) __builtin_popcountll(diff15); \
    } while (0)

#define ACCUM_WORD(weights, src, word_idx, masks, active_bits, raw_acc, lane_count) \
    do { \
        uint64_t src_word_ = (src)[word_idx]; \
        if ((lane_count) == CONV_OC_BLOCK) { \
            ACCUM_LANES_16((weights) + (size_t) (word_idx) * (size_t) CONV_OC_BLOCK, src_word_, (masks)[word_idx], (active_bits)[word_idx], raw_acc); \
        } else { \
            accum_lanes_tail((weights) + (size_t) (word_idx) * (size_t) CONV_OC_BLOCK, src_word_, (masks)[word_idx], (active_bits)[word_idx], raw_acc, lane_count); \
        } \
    } while (0)

#define DEFINE_ACCUM_WORDS(IW) \
    static inline void accum_words_iw##IW(const uint64_t *src, const uint64_t *weights, const uint64_t *masks, const int *active_bits, int *raw_acc, int lane_count) { \
        ACCUM_WORD(weights, src, 0, masks, active_bits, raw_acc, lane_count); \
        if ((IW) > 1) { ACCUM_WORD(weights, src, 1, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 2) { ACCUM_WORD(weights, src, 2, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 3) { ACCUM_WORD(weights, src, 3, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 4) { ACCUM_WORD(weights, src, 4, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 5) { ACCUM_WORD(weights, src, 5, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 6) { ACCUM_WORD(weights, src, 6, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 7) { ACCUM_WORD(weights, src, 7, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 8) { ACCUM_WORD(weights, src, 8, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 9) { ACCUM_WORD(weights, src, 9, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 10) { ACCUM_WORD(weights, src, 10, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 11) { ACCUM_WORD(weights, src, 11, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 12) { ACCUM_WORD(weights, src, 12, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 13) { ACCUM_WORD(weights, src, 13, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 14) { ACCUM_WORD(weights, src, 14, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 15) { ACCUM_WORD(weights, src, 15, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 16) { ACCUM_WORD(weights, src, 16, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 17) { ACCUM_WORD(weights, src, 17, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 18) { ACCUM_WORD(weights, src, 18, masks, active_bits, raw_acc, lane_count); } \
        if ((IW) > 19) { ACCUM_WORD(weights, src, 19, masks, active_bits, raw_acc, lane_count); } \
    }

DEFINE_ACCUM_WORDS(1)
DEFINE_ACCUM_WORDS(2)
DEFINE_ACCUM_WORDS(3)
DEFINE_ACCUM_WORDS(5)
DEFINE_ACCUM_WORDS(10)
DEFINE_ACCUM_WORDS(20)

typedef void (*AccumWordsFn)(const uint64_t *, const uint64_t *, const uint64_t *, const int *, int *, int);

static AccumWordsFn conv_accum_words_fn(int in_words) {
    switch (in_words) {
        case 1:
            return accum_words_iw1;
        case 2:
            return accum_words_iw2;
        case 3:
            return accum_words_iw3;
        case 5:
            return accum_words_iw5;
        case 10:
            return accum_words_iw10;
        case 20:
            return accum_words_iw20;
        default:
            return NULL;
    }
}

static inline void accum_words_generic(const uint64_t *src, const uint64_t *weights, const uint64_t *masks, const int *active_bits, int *raw_acc, int lane_count, int in_words) {
    int word_idx;
    for (word_idx = 0; word_idx < in_words; ++word_idx) {
        uint64_t src_word = src[word_idx];
        if (lane_count == CONV_OC_BLOCK) {
            ACCUM_LANES_16(weights + (size_t) word_idx * (size_t) CONV_OC_BLOCK, src_word, masks[word_idx], active_bits[word_idx], raw_acc);
        } else {
            accum_lanes_tail(weights + (size_t) word_idx * (size_t) CONV_OC_BLOCK, src_word, masks[word_idx], active_bits[word_idx], raw_acc, lane_count);
        }
    }
}

static inline void accum_words_any(const uint64_t *src, const uint64_t *weights, const uint64_t *masks, const int *active_bits, int *raw_acc, int lane_count, int in_words, AccumWordsFn fn) {
    if (fn != NULL) {
        fn(src, weights, masks, active_bits, raw_acc, lane_count);
    } else {
        accum_words_generic(src, weights, masks, active_bits, raw_acc, lane_count, in_words);
    }
}

static inline int conv_inner_begin(int pad_before, int stride) {
    return (pad_before + stride - 1) / stride;
}

static inline int conv_inner_end(int input_size, int kernel, int stride, int pad_before) {
    return (input_size - kernel + pad_before) / stride + 1;
}

static inline void conv_zero_raw_acc(int *raw_acc) {
    int lane;
    for (lane = 0; lane < CONV_OC_BLOCK; ++lane) {
        raw_acc[lane] = 0;
    }
}

static inline void load_column_words(uint64_t dst[3][20], const PackedMap *input, int sy0, int sx, int in_words) {
    int row;
    for (row = 0; row < 3; ++row) {
        const uint64_t *src = packed_pixel_const(input->data, input->w, input->words, sy0 + row, sx);
        memcpy(dst[row], src, (size_t) in_words * sizeof(uint64_t));
    }
}

static inline void load_column_words_iw1(uint64_t dst[3], const PackedMap *input, int sy0, int sx) {
    dst[0] = *packed_pixel_const(input->data, input->w, input->words, sy0 + 0, sx);
    dst[1] = *packed_pixel_const(input->data, input->w, input->words, sy0 + 1, sx);
    dst[2] = *packed_pixel_const(input->data, input->w, input->words, sy0 + 2, sx);
}

static inline void accum_word_iw1(const uint64_t *weights, uint64_t src_word, uint64_t mask, int active, int *raw_acc, int lane_count) {
    if (lane_count == CONV_OC_BLOCK) {
        ACCUM_LANES_16(weights, src_word, mask, active, raw_acc);
    } else {
        accum_lanes_tail(weights, src_word, mask, active, raw_acc, lane_count);
    }
}

static inline void conv_emit_block_3x3_iw1_taps(
    const BinaryConvLayer *layer,
    const uint64_t *weights_block,
    const uint64_t taps[9],
    int out_base,
    uint64_t *dst,
    uint64_t mask,
    int active
) {
    int raw_acc[CONV_OC_BLOCK];
    int lane_count = layer->out_c - out_base;
    if (lane_count > CONV_OC_BLOCK) {
        lane_count = CONV_OC_BLOCK;
    }
    conv_zero_raw_acc(raw_acc);
    accum_word_iw1(weights_block + (size_t) 0 * (size_t) CONV_OC_BLOCK, taps[0], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 1 * (size_t) CONV_OC_BLOCK, taps[1], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 2 * (size_t) CONV_OC_BLOCK, taps[2], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 3 * (size_t) CONV_OC_BLOCK, taps[3], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 4 * (size_t) CONV_OC_BLOCK, taps[4], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 5 * (size_t) CONV_OC_BLOCK, taps[5], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 6 * (size_t) CONV_OC_BLOCK, taps[6], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 7 * (size_t) CONV_OC_BLOCK, taps[7], mask, active, raw_acc, lane_count);
    accum_word_iw1(weights_block + (size_t) 8 * (size_t) CONV_OC_BLOCK, taps[8], mask, active, raw_acc, lane_count);
    finalize_conv_block(layer, out_base, raw_acc, dst);
}

static inline void conv_emit_block_3x3_iw1_border(
    const PackedMap *input,
    const BinaryConvLayer *layer,
    int in_y0,
    int in_x0,
    int out_base,
    uint64_t *dst,
    uint64_t mask,
    int active
) {
    int raw_acc[CONV_OC_BLOCK];
    int lane_count = layer->out_c - out_base;
    const uint64_t *weights_block = layer->weights + ((size_t) out_base / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer);
    int ky;
    int kx;
    if (lane_count > CONV_OC_BLOCK) {
        lane_count = CONV_OC_BLOCK;
    }
    conv_zero_raw_acc(raw_acc);
    for (ky = 0; ky < 3; ++ky) {
        int sy = in_y0 + ky;
        if ((unsigned) sy >= (unsigned) input->h) {
            continue;
        }
        for (kx = 0; kx < 3; ++kx) {
            int sx = in_x0 + kx;
            uint64_t src_word;
            if ((unsigned) sx >= (unsigned) input->w) {
                continue;
            }
            src_word = *packed_pixel_const(input->data, input->w, input->words, sy, sx);
            accum_word_iw1(weights_block + (size_t) (ky * 3 + kx) * (size_t) CONV_OC_BLOCK, src_word, mask, active, raw_acc, lane_count);
        }
    }
    finalize_conv_block(layer, out_base, raw_acc, dst);
}

static inline void conv_emit_block_1x1(
    const PackedMap *input,
    const BinaryConvLayer *layer,
    const uint64_t *src,
    int out_base,
    uint64_t *dst,
    const uint64_t *word_masks,
    const int *active_bits,
    AccumWordsFn fn
) {
    int raw_acc[CONV_OC_BLOCK];
    int lane_count = layer->out_c - out_base;
    const uint64_t *weights_block = layer->weights + ((size_t) out_base / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer);
    if (lane_count > CONV_OC_BLOCK) {
        lane_count = CONV_OC_BLOCK;
    }
    (void) input;
    conv_zero_raw_acc(raw_acc);
    accum_words_any(src, weights_block, word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    finalize_conv_block(layer, out_base, raw_acc, dst);
}

static inline void conv_emit_block_3x3_generic(
    const PackedMap *input,
    const BinaryConvLayer *layer,
    int in_y0,
    int in_x0,
    int out_base,
    uint64_t *dst,
    const uint64_t *word_masks,
    const int *active_bits,
    AccumWordsFn fn
) {
    int raw_acc[CONV_OC_BLOCK];
    int lane_count = layer->out_c - out_base;
    const uint64_t *weights_block = layer->weights + ((size_t) out_base / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer);
    int ky;
    int kx;
    if (lane_count > CONV_OC_BLOCK) {
        lane_count = CONV_OC_BLOCK;
    }
    conv_zero_raw_acc(raw_acc);
    for (ky = 0; ky < 3; ++ky) {
        int sy = in_y0 + ky;
        if ((unsigned) sy >= (unsigned) input->h) {
            continue;
        }
        for (kx = 0; kx < 3; ++kx) {
            int sx = in_x0 + kx;
            const uint64_t *src;
            const uint64_t *weights_tap;
            if ((unsigned) sx >= (unsigned) input->w) {
                continue;
            }
            src = packed_pixel_const(input->data, input->w, input->words, sy, sx);
            weights_tap = weights_block + (((size_t) (ky * 3 + kx) * (size_t) layer->in_words) * (size_t) CONV_OC_BLOCK);
            accum_words_any(src, weights_tap, word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
        }
    }
    finalize_conv_block(layer, out_base, raw_acc, dst);
}

static inline void conv_emit_block_3x3_interior(
    const BinaryConvLayer *layer,
    const uint64_t src00[20],
    const uint64_t src01[20],
    const uint64_t src02[20],
    const uint64_t src10[20],
    const uint64_t src11[20],
    const uint64_t src12[20],
    const uint64_t src20[20],
    const uint64_t src21[20],
    const uint64_t src22[20],
    int out_base,
    uint64_t *dst,
    const uint64_t *word_masks,
    const int *active_bits,
    AccumWordsFn fn
) {
    int raw_acc[CONV_OC_BLOCK];
    int lane_count = layer->out_c - out_base;
    const uint64_t *weights_block = layer->weights + ((size_t) out_base / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer);
    if (lane_count > CONV_OC_BLOCK) {
        lane_count = CONV_OC_BLOCK;
    }
    conv_zero_raw_acc(raw_acc);
    accum_words_any(src00, weights_block + ((size_t) 0 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src01, weights_block + ((size_t) 1 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src02, weights_block + ((size_t) 2 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src10, weights_block + ((size_t) 3 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src11, weights_block + ((size_t) 4 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src12, weights_block + ((size_t) 5 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src20, weights_block + ((size_t) 6 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src21, weights_block + ((size_t) 7 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    accum_words_any(src22, weights_block + ((size_t) 8 * (size_t) layer->in_words * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count, layer->in_words, fn);
    finalize_conv_block(layer, out_base, raw_acc, dst);
}

#define DEFINE_CONV_EMIT_BLOCK_3X3_INTERIOR_FIXED(IW) \
    static inline void conv_emit_block_3x3_interior_iw##IW( \
        const BinaryConvLayer *layer, \
        const uint64_t src00[20], \
        const uint64_t src01[20], \
        const uint64_t src02[20], \
        const uint64_t src10[20], \
        const uint64_t src11[20], \
        const uint64_t src12[20], \
        const uint64_t src20[20], \
        const uint64_t src21[20], \
        const uint64_t src22[20], \
        int out_base, \
        uint64_t *dst, \
        const uint64_t *word_masks, \
        const int *active_bits \
    ) { \
        int raw_acc[CONV_OC_BLOCK]; \
        int lane_count = layer->out_c - out_base; \
        const uint64_t *weights_block = layer->weights + ((size_t) out_base / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer); \
        if (lane_count > CONV_OC_BLOCK) { \
            lane_count = CONV_OC_BLOCK; \
        } \
        conv_zero_raw_acc(raw_acc); \
        accum_words_iw##IW(src00, weights_block + ((size_t) 0 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src01, weights_block + ((size_t) 1 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src02, weights_block + ((size_t) 2 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src10, weights_block + ((size_t) 3 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src11, weights_block + ((size_t) 4 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src12, weights_block + ((size_t) 5 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src20, weights_block + ((size_t) 6 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src21, weights_block + ((size_t) 7 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        accum_words_iw##IW(src22, weights_block + ((size_t) 8 * (size_t) (IW) * (size_t) CONV_OC_BLOCK), word_masks, active_bits, raw_acc, lane_count); \
        finalize_conv_block(layer, out_base, raw_acc, dst); \
    }

DEFINE_CONV_EMIT_BLOCK_3X3_INTERIOR_FIXED(5)
DEFINE_CONV_EMIT_BLOCK_3X3_INTERIOR_FIXED(10)

static void conv_bipolar_run_1x1(const PackedMap *input, const BinaryConvLayer *layer, PackedMap *output, const uint64_t *word_masks, const int *active_bits) {
    AccumWordsFn fn = conv_accum_words_fn(layer->in_words);
    int oy;
    int ox;
    int out_base;
    for (oy = 0; oy < output->h; ++oy) {
        for (ox = 0; ox < output->w; ++ox) {
            const uint64_t *src = packed_pixel_const(input->data, input->w, input->words, oy, ox);
            uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
            for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                conv_emit_block_1x1(input, layer, src, out_base, dst, word_masks, active_bits, fn);
            }
        }
    }
}

static void conv_bipolar_run_3x3_stride2(const PackedMap *input, const BinaryConvLayer *layer, PackedMap *output, const uint64_t *word_masks, const int *active_bits) {
    AccumWordsFn fn = conv_accum_words_fn(layer->in_words);
    int pad_top = same_pad_before(layer->in_h, layer->kernel, layer->stride);
    int pad_left = same_pad_before(layer->in_w, layer->kernel, layer->stride);
    int inner_y_begin = conv_inner_begin(pad_top, layer->stride);
    int inner_y_end = conv_inner_end(input->h, layer->kernel, layer->stride, pad_top);
    int inner_x_begin = conv_inner_begin(pad_left, layer->stride);
    int inner_x_end = conv_inner_end(input->w, layer->kernel, layer->stride, pad_left);
    int oy;
    int ox;
    int out_base;
    for (oy = 0; oy < output->h; ++oy) {
        int in_y0 = oy * layer->stride - pad_top;
        for (ox = 0; ox < output->w; ++ox) {
            int in_x0 = ox * layer->stride - pad_left;
            uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
            if (oy >= inner_y_begin && oy < inner_y_end && ox >= inner_x_begin && ox < inner_x_end) {
                const uint64_t *src00 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 0, in_x0 + 0);
                const uint64_t *src01 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 0, in_x0 + 1);
                const uint64_t *src02 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 0, in_x0 + 2);
                const uint64_t *src10 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 1, in_x0 + 0);
                const uint64_t *src11 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 1, in_x0 + 1);
                const uint64_t *src12 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 1, in_x0 + 2);
                const uint64_t *src20 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 2, in_x0 + 0);
                const uint64_t *src21 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 2, in_x0 + 1);
                const uint64_t *src22 = packed_pixel_const(input->data, input->w, input->words, in_y0 + 2, in_x0 + 2);
                for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                    conv_emit_block_3x3_interior(layer, src00, src01, src02, src10, src11, src12, src20, src21, src22, out_base, dst, word_masks, active_bits, fn);
                }
            } else {
                for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                    conv_emit_block_3x3_generic(input, layer, in_y0, in_x0, out_base, dst, word_masks, active_bits, fn);
                }
            }
        }
    }
}

static void conv_bipolar_run_3x3_stride1(const PackedMap *input, const BinaryConvLayer *layer, PackedMap *output, const uint64_t *word_masks, const int *active_bits) {
    AccumWordsFn fn = conv_accum_words_fn(layer->in_words);
    int pad_top = same_pad_before(layer->in_h, layer->kernel, layer->stride);
    int pad_left = same_pad_before(layer->in_w, layer->kernel, layer->stride);
    int inner_y_begin = conv_inner_begin(pad_top, layer->stride);
    int inner_y_end = conv_inner_end(input->h, layer->kernel, layer->stride, pad_top);
    int inner_x_begin = conv_inner_begin(pad_left, layer->stride);
    int inner_x_end = conv_inner_end(input->w, layer->kernel, layer->stride, pad_left);
    int oy;
    int ox;
    int out_base;
    for (oy = 0; oy < output->h; ++oy) {
        int in_y0 = oy * layer->stride - pad_top;
        if (oy < inner_y_begin || oy >= inner_y_end) {
            for (ox = 0; ox < output->w; ++ox) {
                int in_x0 = ox * layer->stride - pad_left;
                uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
                for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                    conv_emit_block_3x3_generic(input, layer, in_y0, in_x0, out_base, dst, word_masks, active_bits, fn);
                }
            }
            continue;
        }
        for (ox = 0; ox < inner_x_begin; ++ox) {
            int in_x0 = ox * layer->stride - pad_left;
            uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
            for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                conv_emit_block_3x3_generic(input, layer, in_y0, in_x0, out_base, dst, word_masks, active_bits, fn);
            }
        }
        if (inner_x_begin < inner_x_end) {
            uint64_t colbuf0[3][20];
            uint64_t colbuf1[3][20];
            uint64_t colbuf2[3][20];
            uint64_t (*col0)[20] = colbuf0;
            uint64_t (*col1)[20] = colbuf1;
            uint64_t (*col2)[20] = colbuf2;
            int sy0 = in_y0;
            load_column_words(col0, input, sy0, inner_x_begin - 1, layer->in_words);
            load_column_words(col1, input, sy0, inner_x_begin, layer->in_words);
            load_column_words(col2, input, sy0, inner_x_begin + 1, layer->in_words);
                for (ox = inner_x_begin; ox < inner_x_end; ++ox) {
                    uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
                    for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                        conv_emit_block_3x3_interior(
                            layer,
                            col0[0], col1[0], col2[0],
                        col0[1], col1[1], col2[1],
                        col0[2], col1[2], col2[2],
                        out_base,
                        dst,
                            word_masks,
                            active_bits,
                            fn
                        );
                    }
                if (ox + 1 < inner_x_end) {
                    uint64_t (*tmp)[20] = col0;
                    col0 = col1;
                    col1 = col2;
                    col2 = tmp;
                    load_column_words(col2, input, sy0, ox + 2, layer->in_words);
                }
            }
        }
        for (ox = inner_x_end; ox < output->w; ++ox) {
            int in_x0 = ox * layer->stride - pad_left;
            uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
            for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                conv_emit_block_3x3_generic(input, layer, in_y0, in_x0, out_base, dst, word_masks, active_bits, fn);
            }
        }
    }
}

#define DEFINE_CONV_BIPOLAR_RUN_3X3_STRIDE1_FIXED(IW) \
    static void conv_bipolar_run_3x3_stride1_iw##IW(const PackedMap *input, const BinaryConvLayer *layer, PackedMap *output, const uint64_t *word_masks, const int *active_bits) { \
        int pad_top = same_pad_before(layer->in_h, layer->kernel, layer->stride); \
        int pad_left = same_pad_before(layer->in_w, layer->kernel, layer->stride); \
        int inner_y_begin = conv_inner_begin(pad_top, layer->stride); \
        int inner_y_end = conv_inner_end(input->h, layer->kernel, layer->stride, pad_top); \
        int inner_x_begin = conv_inner_begin(pad_left, layer->stride); \
        int inner_x_end = conv_inner_end(input->w, layer->kernel, layer->stride, pad_left); \
        int oy; \
        int ox; \
        int out_base; \
        for (oy = 0; oy < output->h; ++oy) { \
            int in_y0 = oy * layer->stride - pad_top; \
            if (oy < inner_y_begin || oy >= inner_y_end) { \
                for (ox = 0; ox < output->w; ++ox) { \
                    int in_x0 = ox * layer->stride - pad_left; \
                    uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox); \
                    for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) { \
                        conv_emit_block_3x3_generic(input, layer, in_y0, in_x0, out_base, dst, word_masks, active_bits, accum_words_iw##IW); \
                    } \
                } \
                continue; \
            } \
            for (ox = 0; ox < inner_x_begin; ++ox) { \
                int in_x0 = ox * layer->stride - pad_left; \
                uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox); \
                for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) { \
                    conv_emit_block_3x3_generic(input, layer, in_y0, in_x0, out_base, dst, word_masks, active_bits, accum_words_iw##IW); \
                } \
            } \
            if (inner_x_begin < inner_x_end) { \
                uint64_t colbuf0[3][20]; \
                uint64_t colbuf1[3][20]; \
                uint64_t colbuf2[3][20]; \
                uint64_t (*col0)[20] = colbuf0; \
                uint64_t (*col1)[20] = colbuf1; \
                uint64_t (*col2)[20] = colbuf2; \
                load_column_words(col0, input, in_y0, inner_x_begin - 1, layer->in_words); \
                load_column_words(col1, input, in_y0, inner_x_begin, layer->in_words); \
                load_column_words(col2, input, in_y0, inner_x_begin + 1, layer->in_words); \
                for (ox = inner_x_begin; ox < inner_x_end; ++ox) { \
                    uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox); \
                    for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) { \
                        conv_emit_block_3x3_interior_iw##IW( \
                            layer, \
                            col0[0], col1[0], col2[0], \
                            col0[1], col1[1], col2[1], \
                            col0[2], col1[2], col2[2], \
                            out_base, \
                            dst, \
                            word_masks, \
                            active_bits \
                        ); \
                    } \
                    if (ox + 1 < inner_x_end) { \
                        uint64_t (*tmp)[20] = col0; \
                        col0 = col1; \
                        col1 = col2; \
                        col2 = tmp; \
                        load_column_words(col2, input, in_y0, ox + 2, layer->in_words); \
                    } \
                } \
            } \
            for (ox = inner_x_end; ox < output->w; ++ox) { \
                int in_x0 = ox * layer->stride - pad_left; \
                uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox); \
                for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) { \
                    conv_emit_block_3x3_generic(input, layer, in_y0, in_x0, out_base, dst, word_masks, active_bits, accum_words_iw##IW); \
                } \
            } \
        } \
    }

DEFINE_CONV_BIPOLAR_RUN_3X3_STRIDE1_FIXED(5)
DEFINE_CONV_BIPOLAR_RUN_3X3_STRIDE1_FIXED(10)

static void conv_bipolar_run_3x3_iw1(const PackedMap *input, const BinaryConvLayer *layer, PackedMap *output) {
    int pad_top = same_pad_before(layer->in_h, layer->kernel, layer->stride);
    int pad_left = same_pad_before(layer->in_w, layer->kernel, layer->stride);
    int inner_y_begin = conv_inner_begin(pad_top, layer->stride);
    int inner_y_end = conv_inner_end(input->h, layer->kernel, layer->stride, pad_top);
    int inner_x_begin = conv_inner_begin(pad_left, layer->stride);
    int inner_x_end = conv_inner_end(input->w, layer->kernel, layer->stride, pad_left);
    uint64_t mask = word_mask_for_channels(layer->in_c, 0);
    int active = active_bits_for_word(layer->in_c, 0);
    int oy;
    int ox;
    int out_base;
    for (oy = 0; oy < output->h; ++oy) {
        int in_y0 = oy * layer->stride - pad_top;
        if (oy < inner_y_begin || oy >= inner_y_end) {
            for (ox = 0; ox < output->w; ++ox) {
                int in_x0 = ox * layer->stride - pad_left;
                uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
                for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                    conv_emit_block_3x3_iw1_border(input, layer, in_y0, in_x0, out_base, dst, mask, active);
                }
            }
            continue;
        }
        for (ox = 0; ox < inner_x_begin; ++ox) {
            int in_x0 = ox * layer->stride - pad_left;
            uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
            for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                conv_emit_block_3x3_iw1_border(input, layer, in_y0, in_x0, out_base, dst, mask, active);
            }
        }
        if (inner_x_begin < inner_x_end) {
            if (layer->stride == 1) {
                uint64_t colbuf0[3];
                uint64_t colbuf1[3];
                uint64_t colbuf2[3];
                uint64_t *col0 = colbuf0;
                uint64_t *col1 = colbuf1;
                uint64_t *col2 = colbuf2;
                load_column_words_iw1(col0, input, in_y0, inner_x_begin - 1);
                load_column_words_iw1(col1, input, in_y0, inner_x_begin);
                load_column_words_iw1(col2, input, in_y0, inner_x_begin + 1);
                for (ox = inner_x_begin; ox < inner_x_end; ++ox) {
                    uint64_t taps[9];
                    uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
                    taps[0] = col0[0];
                    taps[1] = col1[0];
                    taps[2] = col2[0];
                    taps[3] = col0[1];
                    taps[4] = col1[1];
                    taps[5] = col2[1];
                    taps[6] = col0[2];
                    taps[7] = col1[2];
                    taps[8] = col2[2];
                    for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                        const uint64_t *weights_block = layer->weights + ((size_t) out_base / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer);
                        conv_emit_block_3x3_iw1_taps(layer, weights_block, taps, out_base, dst, mask, active);
                    }
                    if (ox + 1 < inner_x_end) {
                        uint64_t *tmp = col0;
                        col0 = col1;
                        col1 = col2;
                        col2 = tmp;
                        load_column_words_iw1(col2, input, in_y0, ox + 2);
                    }
                }
            } else {
                for (ox = inner_x_begin; ox < inner_x_end; ++ox) {
                    int in_x0 = ox * layer->stride - pad_left;
                    uint64_t taps[9];
                    uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
                    taps[0] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 0, in_x0 + 0);
                    taps[1] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 0, in_x0 + 1);
                    taps[2] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 0, in_x0 + 2);
                    taps[3] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 1, in_x0 + 0);
                    taps[4] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 1, in_x0 + 1);
                    taps[5] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 1, in_x0 + 2);
                    taps[6] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 2, in_x0 + 0);
                    taps[7] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 2, in_x0 + 1);
                    taps[8] = *packed_pixel_const(input->data, input->w, input->words, in_y0 + 2, in_x0 + 2);
                    for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                        const uint64_t *weights_block = layer->weights + ((size_t) out_base / (size_t) CONV_OC_BLOCK) * conv_weight_block_words(layer);
                        conv_emit_block_3x3_iw1_taps(layer, weights_block, taps, out_base, dst, mask, active);
                    }
                }
            }
        }
        for (ox = inner_x_end; ox < output->w; ++ox) {
            int in_x0 = ox * layer->stride - pad_left;
            uint64_t *dst = packed_pixel(output->data, output->w, output->words, oy, ox);
            for (out_base = 0; out_base < layer->out_c; out_base += CONV_OC_BLOCK) {
                conv_emit_block_3x3_iw1_border(input, layer, in_y0, in_x0, out_base, dst, mask, active);
            }
        }
    }
}

static void conv_bipolar_run(const PackedMap *input, const BinaryConvLayer *layer, PackedMap *output) {
    uint64_t word_masks[PACK_WORDS(1280)];
    int active_bits[PACK_WORDS(1280)];
    size_t out_words_total = packed_map_words(output->h, output->w, output->c);
    int in_word;
    memset(output->data, 0, out_words_total * sizeof(output->data[0]));
    for (in_word = 0; in_word < input->words; ++in_word) {
        word_masks[in_word] = word_mask_for_channels(layer->in_c, in_word);
        active_bits[in_word] = active_bits_for_word(layer->in_c, in_word);
    }
    if (layer->kernel == 3 && layer->in_words == 1) {
        conv_bipolar_run_3x3_iw1(input, layer, output);
    } else if (layer->kernel == 3 && layer->stride == 1 && layer->in_words == 5) {
        conv_bipolar_run_3x3_stride1_iw5(input, layer, output, word_masks, active_bits);
    } else if (layer->kernel == 3 && layer->stride == 1 && layer->in_words == 10) {
        conv_bipolar_run_3x3_stride1_iw10(input, layer, output, word_masks, active_bits);
    } else if (layer->kernel == 1 && layer->stride == 1) {
        conv_bipolar_run_1x1(input, layer, output, word_masks, active_bits);
    } else if (layer->kernel == 3 && layer->stride == 1) {
        conv_bipolar_run_3x3_stride1(input, layer, output, word_masks, active_bits);
    } else if (layer->kernel == 3 && layer->stride == 2) {
        conv_bipolar_run_3x3_stride2(input, layer, output, word_masks, active_bits);
    }
}

static void upsample_nearest_packed(const PackedMap *input, PackedMap *output) {
    int y;
    int x;
    int words = input->words;
    size_t out_words_total = packed_map_words(output->h, output->w, output->c);
    memset(output->data, 0, out_words_total * sizeof(output->data[0]));
    for (y = 0; y < input->h; ++y) {
        for (x = 0; x < input->w; ++x) {
            const uint64_t *src = packed_pixel_const(input->data, input->w, words, y, x);
            memcpy(packed_pixel(output->data, output->w, words, y * 2, x * 2), src, (size_t) words * sizeof(uint64_t));
            memcpy(packed_pixel(output->data, output->w, words, y * 2, x * 2 + 1), src, (size_t) words * sizeof(uint64_t));
            memcpy(packed_pixel(output->data, output->w, words, y * 2 + 1, x * 2), src, (size_t) words * sizeof(uint64_t));
            memcpy(packed_pixel(output->data, output->w, words, y * 2 + 1, x * 2 + 1), src, (size_t) words * sizeof(uint64_t));
        }
    }
}

static void copy_bits_with_offset(uint64_t *dst, int dst_channels, int dst_bit_offset, const uint64_t *src, int src_channels) {
    int src_words = PACK_WORDS(src_channels);
    int src_word;
    int dst_total_words = PACK_WORDS(dst_channels);
    int shift = dst_bit_offset & 63;
    int dst_word_base = dst_bit_offset >> 6;
    for (src_word = 0; src_word < src_words; ++src_word) {
        uint64_t word = src[src_word] & word_mask_for_channels(src_channels, src_word);
        if (shift == 0) {
            dst[dst_word_base + src_word] |= word;
        } else {
            dst[dst_word_base + src_word] |= word << shift;
            if (dst_word_base + src_word + 1 < dst_total_words) {
                dst[dst_word_base + src_word + 1] |= word >> (64 - shift);
            }
        }
    }
}

static void concat_packed(const PackedMap *a, const PackedMap *b, PackedMap *output) {
    int y;
    int x;
    size_t out_words_total = packed_map_words(output->h, output->w, output->c);
    memset(output->data, 0, out_words_total * sizeof(output->data[0]));
    for (y = 0; y < output->h; ++y) {
        for (x = 0; x < output->w; ++x) {
            uint64_t *dst = packed_pixel(output->data, output->w, output->words, y, x);
            const uint64_t *src_a = packed_pixel_const(a->data, a->w, a->words, y, x);
            const uint64_t *src_b = packed_pixel_const(b->data, b->w, b->words, y, x);
            copy_bits_with_offset(dst, output->c, 0, src_a, a->c);
            copy_bits_with_offset(dst, output->c, a->c, src_b, b->c);
        }
    }
}

#define DEFINE_QDENSE_FORWARD_BATCH3_FIXED(IN_DIM) \
    static inline void qdense_forward_batch3_in##IN_DIM( \
        const int8_t *restrict input0_q8, \
        const int8_t *restrict input1_q8, \
        const int8_t *restrict input2_q8, \
        const QDenseLayer *layer, \
        int8_t *restrict output0_q8, \
        int8_t *restrict output1_q8, \
        int8_t *restrict output2_q8 \
    ) { \
        int out_idx; \
        for (out_idx = 0; out_idx < layer->out_dim; ++out_idx) { \
            const int8_t *weights = layer->weights + (size_t) out_idx * (size_t) (IN_DIM); \
            int acc0_i = 0; \
            int acc1_i = 0; \
            int acc2_i = 0; \
            int in_idx; \
            float acc0 = layer->bias[out_idx]; \
            float acc1 = acc0; \
            float acc2 = acc0; \
            for (in_idx = 0; in_idx < (IN_DIM); ++in_idx) { \
                int w = (int) weights[in_idx]; \
                acc0_i += (int) input0_q8[in_idx] * w; \
                acc1_i += (int) input1_q8[in_idx] * w; \
                acc2_i += (int) input2_q8[in_idx] * w; \
            } \
            acc0 += (float) acc0_i * (1.0f / 64.0f); \
            acc1 += (float) acc1_i * (1.0f / 64.0f); \
            acc2 += (float) acc2_i * (1.0f / 64.0f); \
            if (acc0 < -1.0f) { \
                acc0 = -1.0f; \
            } \
            if (acc0 > 0.875f) { \
                acc0 = 0.875f; \
            } \
            if (acc1 < -1.0f) { \
                acc1 = -1.0f; \
            } \
            if (acc1 > 0.875f) { \
                acc1 = 0.875f; \
            } \
            if (acc2 < -1.0f) { \
                acc2 = -1.0f; \
            } \
            if (acc2 > 0.875f) { \
                acc2 = 0.875f; \
            } \
            output0_q8[out_idx] = (int8_t) floorf(acc0 * 8.0f + 1e-12f); \
            output1_q8[out_idx] = (int8_t) floorf(acc1 * 8.0f + 1e-12f); \
            output2_q8[out_idx] = (int8_t) floorf(acc2 * 8.0f + 1e-12f); \
        } \
    }

DEFINE_QDENSE_FORWARD_BATCH3_FIXED(48)
DEFINE_QDENSE_FORWARD_BATCH3_FIXED(64)
DEFINE_QDENSE_FORWARD_BATCH3_FIXED(128)

static void qdense_forward_batch3(
    const int8_t *restrict input0_q8,
    const int8_t *restrict input1_q8,
    const int8_t *restrict input2_q8,
    const QDenseLayer *layer,
    int8_t *restrict output0_q8,
    int8_t *restrict output1_q8,
    int8_t *restrict output2_q8
) {
    switch (layer->in_dim) {
        case 48:
            qdense_forward_batch3_in48(input0_q8, input1_q8, input2_q8, layer, output0_q8, output1_q8, output2_q8);
            return;
        case 64:
            qdense_forward_batch3_in64(input0_q8, input1_q8, input2_q8, layer, output0_q8, output1_q8, output2_q8);
            return;
        case 128:
            qdense_forward_batch3_in128(input0_q8, input1_q8, input2_q8, layer, output0_q8, output1_q8, output2_q8);
            return;
        default:
            break;
    }
    int out_idx;
    for (out_idx = 0; out_idx < layer->out_dim; ++out_idx) {
        const int8_t *weights = layer->weights + (size_t) out_idx * (size_t) layer->in_dim;
        int acc0_i = 0;
        int acc1_i = 0;
        int acc2_i = 0;
        int in_idx;
        float acc0 = layer->bias[out_idx];
        float acc1 = acc0;
        float acc2 = acc0;
        for (in_idx = 0; in_idx < layer->in_dim; ++in_idx) {
            int w = (int) weights[in_idx];
            acc0_i += (int) input0_q8[in_idx] * w;
            acc1_i += (int) input1_q8[in_idx] * w;
            acc2_i += (int) input2_q8[in_idx] * w;
        }
        acc0 += (float) acc0_i * (1.0f / 64.0f);
        acc1 += (float) acc1_i * (1.0f / 64.0f);
        acc2 += (float) acc2_i * (1.0f / 64.0f);
        if (acc0 < -1.0f) {
            acc0 = -1.0f;
        }
        if (acc0 > 0.875f) {
            acc0 = 0.875f;
        }
        if (acc1 < -1.0f) {
            acc1 = -1.0f;
        }
        if (acc1 > 0.875f) {
            acc1 = 0.875f;
        }
        if (acc2 < -1.0f) {
            acc2 = -1.0f;
        }
        if (acc2 > 0.875f) {
            acc2 = 0.875f;
        }
        output0_q8[out_idx] = (int8_t) floorf(acc0 * 8.0f + 1e-12f);
        output1_q8[out_idx] = (int8_t) floorf(acc1 * 8.0f + 1e-12f);
        output2_q8[out_idx] = (int8_t) floorf(acc2 * 8.0f + 1e-12f);
    }
}

#define DEFINE_QDENSE_OUT_FORWARD_BATCH3_FIXED(IN_DIM) \
    static inline void qdense_out_forward_batch3_in##IN_DIM( \
        const int8_t *restrict input0_q8, \
        const int8_t *restrict input1_q8, \
        const int8_t *restrict input2_q8, \
        const QDenseOutLayer *layer, \
        int8_t *restrict output0_q, \
        int8_t *restrict output1_q, \
        int8_t *restrict output2_q \
    ) { \
        int out_idx; \
        for (out_idx = 0; out_idx < layer->out_dim; ++out_idx) { \
            const int8_t *weights = layer->weights + (size_t) out_idx * (size_t) (IN_DIM); \
            int acc0_i = 0; \
            int acc1_i = 0; \
            int acc2_i = 0; \
            int in_idx; \
            float acc0; \
            float acc1; \
            float acc2; \
            for (in_idx = 0; in_idx < (IN_DIM); ++in_idx) { \
                int w = (int) weights[in_idx]; \
                acc0_i += (int) input0_q8[in_idx] * w; \
                acc1_i += (int) input1_q8[in_idx] * w; \
                acc2_i += (int) input2_q8[in_idx] * w; \
            } \
            acc0 = (float) acc0_i * layer->inv_scale_div8 + (float) layer->bias_q[out_idx]; \
            acc1 = (float) acc1_i * layer->inv_scale_div8 + (float) layer->bias_q[out_idx]; \
            acc2 = (float) acc2_i * layer->inv_scale_div8 + (float) layer->bias_q[out_idx]; \
            if (acc0 < -127.0f) { \
                acc0 = -127.0f; \
            } \
            if (acc0 > 127.0f) { \
                acc0 = 127.0f; \
            } \
            if (acc1 < -127.0f) { \
                acc1 = -127.0f; \
            } \
            if (acc1 > 127.0f) { \
                acc1 = 127.0f; \
            } \
            if (acc2 < -127.0f) { \
                acc2 = -127.0f; \
            } \
            if (acc2 > 127.0f) { \
                acc2 = 127.0f; \
            } \
            output0_q[out_idx] = (int8_t) lrintf(acc0); \
            output1_q[out_idx] = (int8_t) lrintf(acc1); \
            output2_q[out_idx] = (int8_t) lrintf(acc2); \
        } \
    }

DEFINE_QDENSE_OUT_FORWARD_BATCH3_FIXED(48)
DEFINE_QDENSE_OUT_FORWARD_BATCH3_FIXED(64)
DEFINE_QDENSE_OUT_FORWARD_BATCH3_FIXED(128)

static void qdense_out_forward_batch3(
    const int8_t *restrict input0_q8,
    const int8_t *restrict input1_q8,
    const int8_t *restrict input2_q8,
    const QDenseOutLayer *layer,
    int8_t *restrict output0_q,
    int8_t *restrict output1_q,
    int8_t *restrict output2_q
) {
    switch (layer->in_dim) {
        case 48:
            qdense_out_forward_batch3_in48(input0_q8, input1_q8, input2_q8, layer, output0_q, output1_q, output2_q);
            return;
        case 64:
            qdense_out_forward_batch3_in64(input0_q8, input1_q8, input2_q8, layer, output0_q, output1_q, output2_q);
            return;
        case 128:
            qdense_out_forward_batch3_in128(input0_q8, input1_q8, input2_q8, layer, output0_q, output1_q, output2_q);
            return;
        default:
            break;
    }
    int out_idx;
    for (out_idx = 0; out_idx < layer->out_dim; ++out_idx) {
        const int8_t *weights = layer->weights + (size_t) out_idx * (size_t) layer->in_dim;
        int acc0_i = 0;
        int acc1_i = 0;
        int acc2_i = 0;
        int in_idx;
        float acc0;
        float acc1;
        float acc2;
        for (in_idx = 0; in_idx < layer->in_dim; ++in_idx) {
            int w = (int) weights[in_idx];
            acc0_i += (int) input0_q8[in_idx] * w;
            acc1_i += (int) input1_q8[in_idx] * w;
            acc2_i += (int) input2_q8[in_idx] * w;
        }
        acc0 = (float) acc0_i * layer->inv_scale_div8 + (float) layer->bias_q[out_idx];
        acc1 = (float) acc1_i * layer->inv_scale_div8 + (float) layer->bias_q[out_idx];
        acc2 = (float) acc2_i * layer->inv_scale_div8 + (float) layer->bias_q[out_idx];
        if (acc0 < -127.0f) {
            acc0 = -127.0f;
        }
        if (acc0 > 127.0f) {
            acc0 = 127.0f;
        }
        if (acc1 < -127.0f) {
            acc1 = -127.0f;
        }
        if (acc1 > 127.0f) {
            acc1 = 127.0f;
        }
        if (acc2 < -127.0f) {
            acc2 = -127.0f;
        }
        if (acc2 > 127.0f) {
            acc2 = 127.0f;
        }
        output0_q[out_idx] = (int8_t) lrintf(acc0);
        output1_q[out_idx] = (int8_t) lrintf(acc1);
        output2_q[out_idx] = (int8_t) lrintf(acc2);
    }
}

static inline int8_t raw_proj_eval_channel(int raw, uint8_t mode, const int16_t *thresholds) {
    switch (mode) {
        case RAW_PROJ_MODE_CONST_MIN:
            return -8;
        case RAW_PROJ_MODE_CONST_MAX:
            return 7;
        case RAW_PROJ_MODE_INC: {
            int hits = 0;
            hits += raw >= (int) thresholds[0];
            hits += raw >= (int) thresholds[1];
            hits += raw >= (int) thresholds[2];
            hits += raw >= (int) thresholds[3];
            hits += raw >= (int) thresholds[4];
            hits += raw >= (int) thresholds[5];
            hits += raw >= (int) thresholds[6];
            hits += raw >= (int) thresholds[7];
            hits += raw >= (int) thresholds[8];
            hits += raw >= (int) thresholds[9];
            hits += raw >= (int) thresholds[10];
            hits += raw >= (int) thresholds[11];
            hits += raw >= (int) thresholds[12];
            hits += raw >= (int) thresholds[13];
            hits += raw >= (int) thresholds[14];
            return (int8_t) (-8 + hits);
        }
        case RAW_PROJ_MODE_DEC: {
            int hits = 0;
            hits += raw >= (int) thresholds[0];
            hits += raw >= (int) thresholds[1];
            hits += raw >= (int) thresholds[2];
            hits += raw >= (int) thresholds[3];
            hits += raw >= (int) thresholds[4];
            hits += raw >= (int) thresholds[5];
            hits += raw >= (int) thresholds[6];
            hits += raw >= (int) thresholds[7];
            hits += raw >= (int) thresholds[8];
            hits += raw >= (int) thresholds[9];
            hits += raw >= (int) thresholds[10];
            hits += raw >= (int) thresholds[11];
            hits += raw >= (int) thresholds[12];
            hits += raw >= (int) thresholds[13];
            hits += raw >= (int) thresholds[14];
            return (int8_t) (7 - hits);
        }
        default:
            return 0;
    }
}

static void run_head_from_pre(
    const PackedMap *pre_map,
    const RawProjLayer *proj,
    const QDenseLayer *obj_hidden,
    const QDenseLayer *obj_remap,
    const QDenseLayer *cls_hidden,
    const QDenseOutLayer *bbox,
    const QDenseOutLayer *obj,
    const QDenseOutLayer *cls,
    int8_t *head_output
) {
    int y;
    int x;
    int oc;
    uint64_t word_masks[PACK_WORDS(320)];
    int active_bits[PACK_WORDS(320)];
    for (oc = 0; oc < pre_map->words; ++oc) {
        word_masks[oc] = word_mask_for_channels(pre_map->c, oc);
        active_bits[oc] = active_bits_for_word(pre_map->c, oc);
    }
    for (y = 0; y < pre_map->h; ++y) {
        for (x = 0; x < pre_map->w; ++x) {
            const uint64_t *src = packed_pixel_const(pre_map->data, pre_map->w, pre_map->words, y, x);
            int8_t proj_q8[144];
            {
                const uint64_t *weights = proj->weights;
                const uint8_t *mode = proj->mode;
                const int16_t *thresholds = proj->thresholds;
                for (oc = 0; oc < proj->out_c; ++oc) {
                    int raw = 0;
                    int in_word;
                    for (in_word = 0; in_word < proj->in_words; ++in_word) {
                        uint64_t diff = (src[in_word] ^ weights[in_word]) & word_masks[in_word];
                        raw += active_bits[in_word] - 2 * (int) __builtin_popcountll(diff);
                    }
                    proj_q8[oc] = raw_proj_eval_channel(raw, *mode, thresholds);
                    mode += 1;
                    weights += proj->in_words;
                    thresholds += RAW_PROJ_STEPS;
                }
            }
            {
                const int8_t *anchor0 = proj_q8 + 0;
                const int8_t *anchor1 = proj_q8 + 48;
                const int8_t *anchor2 = proj_q8 + 96;
                int8_t obj_hidden_q8[3][128];
                int8_t obj_remap_q8[3][64];
                int8_t cls_hidden_q8[3][128];
                int8_t *dst0 = head_output + ((((size_t) y * (size_t) pre_map->w + (size_t) x) * (size_t) NUM_HEAD_ANCHORS + 0u) * (size_t) HEAD_LOGITS);
                int8_t *dst1 = dst0 + HEAD_LOGITS;
                int8_t *dst2 = dst1 + HEAD_LOGITS;
                qdense_forward_batch3(anchor0, anchor1, anchor2, obj_hidden, obj_hidden_q8[0], obj_hidden_q8[1], obj_hidden_q8[2]);
                qdense_forward_batch3(obj_hidden_q8[0], obj_hidden_q8[1], obj_hidden_q8[2], obj_remap, obj_remap_q8[0], obj_remap_q8[1], obj_remap_q8[2]);
                qdense_forward_batch3(anchor0, anchor1, anchor2, cls_hidden, cls_hidden_q8[0], cls_hidden_q8[1], cls_hidden_q8[2]);
                qdense_out_forward_batch3(anchor0, anchor1, anchor2, bbox, dst0, dst1, dst2);
                qdense_out_forward_batch3(obj_remap_q8[0], obj_remap_q8[1], obj_remap_q8[2], obj, dst0 + 4, dst1 + 4, dst2 + 4);
                qdense_out_forward_batch3(cls_hidden_q8[0], cls_hidden_q8[1], cls_hidden_q8[2], cls, dst0 + 5, dst1 + 5, dst2 + 5);
            }
        }
    }
}

static int read_exact(FILE *handle, void *buffer, size_t size) {
    return fread(buffer, 1, size, handle) == size ? 0 : -1;
}

static BinaryConvLayer *binary_layer_by_idx(uint32_t idx) {
    if (idx < 16u) {
        return &g_binary_layers[idx];
    }
    return NULL;
}

static RawProjLayer *raw_proj_layer_by_idx(uint32_t idx) {
    if (idx >= 16u && idx <= 18u) {
        return &g_raw_proj_layers[idx - 16u];
    }
    return NULL;
}

static QDenseLayer *qdense_layer_by_idx(uint32_t idx) {
    size_t i;
    for (i = 0; i < sizeof(g_qdense_layers) / sizeof(g_qdense_layers[0]); ++i) {
        if ((uint32_t) g_qdense_layers[i].idx == idx) {
            return &g_qdense_layers[i];
        }
    }
    return NULL;
}

static QDenseOutLayer *qdense_out_layer_by_idx(uint32_t idx) {
    size_t i;
    for (i = 0; i < sizeof(g_qdense_out_layers) / sizeof(g_qdense_out_layers[0]); ++i) {
        if ((uint32_t) g_qdense_out_layers[i].idx == idx) {
            return &g_qdense_out_layers[i];
        }
    }
    return NULL;
}

static size_t expected_binary_blob_bytes(const BinaryConvLayer *layer) {
    size_t packed_out_c = (size_t) conv_packed_out_channels(layer);
    return conv_weight_words(layer) * sizeof(uint64_t)
           + packed_out_c * sizeof(uint8_t)
           + packed_out_c * sizeof(int16_t);
}

static size_t expected_raw_proj_blob_bytes(const RawProjLayer *layer) {
    return (size_t) layer->out_c * (size_t) layer->in_words * sizeof(uint64_t)
           + (size_t) layer->out_c * sizeof(uint8_t)
           + (size_t) layer->out_c * (size_t) RAW_PROJ_STEPS * sizeof(int16_t);
}

static size_t expected_qdense_blob_bytes(const QDenseLayer *layer) {
    return (size_t) layer->out_dim * (size_t) layer->in_dim * sizeof(int8_t)
           + (size_t) layer->out_dim * sizeof(float);
}

static size_t expected_qdense_out_blob_bytes(const QDenseOutLayer *layer) {
    return (size_t) layer->out_dim * (size_t) layer->in_dim * sizeof(int8_t)
           + (size_t) layer->out_dim * sizeof(int8_t);
}

static size_t total_weight_arena_bytes(void) {
    size_t total = 0u;
    uint32_t idx;
    for (idx = 0u; idx < NUM_RECORDS; ++idx) {
        const BinaryConvLayer *binary = binary_layer_by_idx(idx);
        const RawProjLayer *raw = raw_proj_layer_by_idx(idx);
        const QDenseLayer *dense = qdense_layer_by_idx(idx);
        const QDenseOutLayer *dense_out = qdense_out_layer_by_idx(idx);
        if (binary != NULL) {
            size_t weights_words = conv_weight_words(binary);
            size_t packed_out_c = (size_t) conv_packed_out_channels(binary);
            total = align_up_size(total, _Alignof(uint64_t));
            total += weights_words * sizeof(uint64_t);
            total = align_up_size(total, _Alignof(uint8_t));
            total += packed_out_c * sizeof(uint8_t);
            total = align_up_size(total, _Alignof(int16_t));
            total += packed_out_c * sizeof(int16_t);
        } else if (raw != NULL) {
            size_t weights_words = (size_t) raw->out_c * (size_t) raw->in_words;
            total = align_up_size(total, _Alignof(uint64_t));
            total += weights_words * sizeof(uint64_t);
            total = align_up_size(total, _Alignof(uint8_t));
            total += (size_t) raw->out_c * sizeof(uint8_t);
            total = align_up_size(total, _Alignof(int16_t));
            total += (size_t) raw->out_c * (size_t) RAW_PROJ_STEPS * sizeof(int16_t);
        } else if (dense != NULL) {
            size_t weight_count = (size_t) dense->out_dim * (size_t) dense->in_dim;
            total = align_up_size(total, _Alignof(int8_t));
            total += weight_count * sizeof(int8_t);
            total = align_up_size(total, _Alignof(float));
            total += (size_t) dense->out_dim * sizeof(float);
        } else if (dense_out != NULL) {
            size_t weight_count = (size_t) dense_out->out_dim * (size_t) dense_out->in_dim;
            total = align_up_size(total, _Alignof(int8_t));
            total += weight_count * sizeof(int8_t);
            total = align_up_size(total, _Alignof(int8_t));
            total += (size_t) dense_out->out_dim * sizeof(int8_t);
        } else {
            return 0u;
        }
    }
    return total;
}

static int prepare_weight_arena(void) {
    free(g_weight_arena);
    g_weight_arena_size = total_weight_arena_bytes();
    g_weight_arena = (uint8_t *) malloc(g_weight_arena_size);
    g_weight_arena_used = 0u;
    return g_weight_arena != NULL ? 0 : -1;
}

static int load_record_binary_conv(FILE *handle, const BinaryConvLayer *spec) {
    size_t weights_words = conv_weight_words(spec);
    size_t packed_out_c = (size_t) conv_packed_out_channels(spec);
    BinaryConvLayer *layer = binary_layer_by_idx((uint32_t) spec->idx);
    layer->weights = (uint64_t *) weight_arena_alloc(weights_words * sizeof(uint64_t), _Alignof(uint64_t));
    layer->cmp_mode = (uint8_t *) weight_arena_alloc(packed_out_c * sizeof(uint8_t), _Alignof(uint8_t));
    layer->cmp_threshold = (int16_t *) weight_arena_alloc(packed_out_c * sizeof(int16_t), _Alignof(int16_t));
    if (layer->weights == NULL || layer->cmp_mode == NULL || layer->cmp_threshold == NULL) {
        return -1;
    }
    if (read_exact(handle, layer->weights, weights_words * sizeof(uint64_t)) != 0) {
        return -1;
    }
    if (read_exact(handle, layer->cmp_mode, packed_out_c * sizeof(uint8_t)) != 0) {
        return -1;
    }
    if (read_exact(handle, layer->cmp_threshold, packed_out_c * sizeof(int16_t)) != 0) {
        return -1;
    }
    return 0;
}

static int load_record_raw_proj(FILE *handle, const RawProjLayer *spec) {
    size_t weights_words = (size_t) spec->out_c * (size_t) spec->in_words;
    RawProjLayer *layer = raw_proj_layer_by_idx((uint32_t) spec->idx);
    layer->weights = (uint64_t *) weight_arena_alloc(weights_words * sizeof(uint64_t), _Alignof(uint64_t));
    layer->mode = (uint8_t *) weight_arena_alloc((size_t) layer->out_c * sizeof(uint8_t), _Alignof(uint8_t));
    layer->thresholds = (int16_t *) weight_arena_alloc((size_t) layer->out_c * (size_t) RAW_PROJ_STEPS * sizeof(int16_t), _Alignof(int16_t));
    if (layer->weights == NULL || layer->mode == NULL || layer->thresholds == NULL) {
        return -1;
    }
    if (read_exact(handle, layer->weights, weights_words * sizeof(uint64_t)) != 0) {
        return -1;
    }
    if (read_exact(handle, layer->mode, (size_t) layer->out_c * sizeof(uint8_t)) != 0) {
        return -1;
    }
    if (read_exact(handle, layer->thresholds, (size_t) layer->out_c * (size_t) RAW_PROJ_STEPS * sizeof(int16_t)) != 0) {
        return -1;
    }
    return 0;
}

static int load_record_qdense(FILE *handle, const QDenseLayer *spec) {
    size_t weight_count = (size_t) spec->out_dim * (size_t) spec->in_dim;
    QDenseLayer *layer = qdense_layer_by_idx((uint32_t) spec->idx);
    layer->weights = (int8_t *) weight_arena_alloc(weight_count * sizeof(int8_t), _Alignof(int8_t));
    layer->bias = (float *) weight_arena_alloc((size_t) layer->out_dim * sizeof(float), _Alignof(float));
    if (layer->weights == NULL || layer->bias == NULL) {
        return -1;
    }
    if (read_exact(handle, layer->weights, weight_count * sizeof(int8_t)) != 0) {
        return -1;
    }
    if (read_exact(handle, layer->bias, (size_t) layer->out_dim * sizeof(float)) != 0) {
        return -1;
    }
    return 0;
}

static int load_record_qdense_out(FILE *handle, const QDenseOutLayer *spec) {
    size_t weight_count = (size_t) spec->out_dim * (size_t) spec->in_dim;
    QDenseOutLayer *layer = qdense_out_layer_by_idx((uint32_t) spec->idx);
    layer->weights = (int8_t *) weight_arena_alloc(weight_count * sizeof(int8_t), _Alignof(int8_t));
    layer->bias_q = (int8_t *) weight_arena_alloc((size_t) layer->out_dim * sizeof(int8_t), _Alignof(int8_t));
    if (layer->weights == NULL || layer->bias_q == NULL) {
        return -1;
    }
    if (read_exact(handle, layer->weights, weight_count * sizeof(int8_t)) != 0) {
        return -1;
    }
    if (read_exact(handle, layer->bias_q, (size_t) layer->out_dim * sizeof(int8_t)) != 0) {
        return -1;
    }
    return 0;
}

static void free_loaded_weights(void) {
    size_t i;
    for (i = 0; i < sizeof(g_binary_layers) / sizeof(g_binary_layers[0]); ++i) {
        g_binary_layers[i].weights = NULL;
        g_binary_layers[i].cmp_mode = NULL;
        g_binary_layers[i].cmp_threshold = NULL;
    }
    for (i = 0; i < sizeof(g_raw_proj_layers) / sizeof(g_raw_proj_layers[0]); ++i) {
        g_raw_proj_layers[i].weights = NULL;
        g_raw_proj_layers[i].mode = NULL;
        g_raw_proj_layers[i].thresholds = NULL;
    }
    for (i = 0; i < sizeof(g_qdense_layers) / sizeof(g_qdense_layers[0]); ++i) {
        g_qdense_layers[i].weights = NULL;
        g_qdense_layers[i].bias = NULL;
    }
    for (i = 0; i < sizeof(g_qdense_out_layers) / sizeof(g_qdense_out_layers[0]); ++i) {
        g_qdense_out_layers[i].weights = NULL;
        g_qdense_out_layers[i].bias_q = NULL;
    }
    free(g_weight_arena);
    g_weight_arena = NULL;
    g_weight_arena_size = 0u;
    g_weight_arena_used = 0u;
}

static int load_weight_file(const char *path) {
    FILE *handle = fopen(path, "rb");
    unsigned char magic[FILE_MAGIC_SIZE];
    uint32_t version;
    uint32_t num_records;
    uint32_t reserved;
    uint8_t seen[NUM_RECORDS];
    uint32_t rec_idx;
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }
    memset(seen, 0, sizeof(seen));
    if (read_exact(handle, magic, FILE_MAGIC_SIZE) != 0
        || read_exact(handle, &version, sizeof(version)) != 0
        || read_exact(handle, &num_records, sizeof(num_records)) != 0
        || read_exact(handle, &reserved, sizeof(reserved)) != 0) {
        fprintf(stderr, "failed to read header from %s\n", path);
        fclose(handle);
        return -1;
    }
    if (memcmp(magic, FILE_MAGIC, FILE_MAGIC_SIZE) != 0) {
        fprintf(stderr, "unexpected magic in %s\n", path);
        fclose(handle);
        return -1;
    }
    if (version != FILE_VERSION || num_records != NUM_RECORDS) {
        fprintf(stderr, "unsupported file version or record count in %s\n", path);
        fclose(handle);
        return -1;
    }
    if (prepare_weight_arena() != 0) {
        fprintf(stderr, "failed to allocate %zu-byte weight arena\n", total_weight_arena_bytes());
        fclose(handle);
        return -1;
    }
    for (rec_idx = 0; rec_idx < num_records; ++rec_idx) {
        uint32_t idx;
        uint32_t kind;
        uint32_t blob_bytes;
        uint32_t rec_reserved;
        if (read_exact(handle, &idx, sizeof(idx)) != 0
            || read_exact(handle, &kind, sizeof(kind)) != 0
            || read_exact(handle, &blob_bytes, sizeof(blob_bytes)) != 0
            || read_exact(handle, &rec_reserved, sizeof(rec_reserved)) != 0) {
            fprintf(stderr, "failed to read record header from %s\n", path);
            fclose(handle);
            return -1;
        }
        if (idx >= NUM_RECORDS || seen[idx]) {
            fprintf(stderr, "duplicate or invalid record %u in %s\n", idx, path);
            fclose(handle);
            return -1;
        }
        seen[idx] = 1u;
        if (kind == RECORD_BINARY_CONV) {
            BinaryConvLayer *layer = binary_layer_by_idx(idx);
            if (layer == NULL || blob_bytes != expected_binary_blob_bytes(layer) || load_record_binary_conv(handle, layer) != 0) {
                fprintf(stderr, "failed to load binary conv record %u\n", idx);
                fclose(handle);
                return -1;
            }
        } else if (kind == RECORD_RAW_PROJ) {
            RawProjLayer *layer = raw_proj_layer_by_idx(idx);
            if (layer == NULL || blob_bytes != expected_raw_proj_blob_bytes(layer) || load_record_raw_proj(handle, layer) != 0) {
                fprintf(stderr, "failed to load raw proj record %u\n", idx);
                fclose(handle);
                return -1;
            }
        } else if (kind == RECORD_QDENSE) {
            QDenseLayer *layer = qdense_layer_by_idx(idx);
            if (layer == NULL || blob_bytes != expected_qdense_blob_bytes(layer) || load_record_qdense(handle, layer) != 0) {
                fprintf(stderr, "failed to load qdense record %u\n", idx);
                fclose(handle);
                return -1;
            }
        } else if (kind == RECORD_QDENSE_OUT) {
            QDenseOutLayer *layer = qdense_out_layer_by_idx(idx);
            if (layer == NULL || blob_bytes != expected_qdense_out_blob_bytes(layer) || load_record_qdense_out(handle, layer) != 0) {
                fprintf(stderr, "failed to load qdense-out record %u\n", idx);
                fclose(handle);
                return -1;
            }
        } else {
            fprintf(stderr, "unknown record kind %u in %s\n", kind, path);
            fclose(handle);
            return -1;
        }
    }
    fclose(handle);
    return 0;
}

static int load_input_u8(const char *path, uint8_t *buffer, size_t size) {
    FILE *handle = fopen(path, "rb");
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (read_exact(handle, buffer, size) != 0) {
        fprintf(stderr, "failed to read %zu bytes from %s\n", size, path);
        fclose(handle);
        return -1;
    }
    fclose(handle);
    return 0;
}

static int load_input_f32(const char *path, float *buffer, size_t count) {
    FILE *handle = fopen(path, "rb");
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (read_exact(handle, buffer, count * sizeof(float)) != 0) {
        fprintf(stderr, "failed to read %zu floats from %s\n", count, path);
        fclose(handle);
        return -1;
    }
    fclose(handle);
    return 0;
}

static void pack_binary_input_u8(const uint8_t *rgb, PackedMap *output) {
    size_t pix;
    memset(output->data, 0, packed_map_words(output->h, output->w, output->c) * sizeof(output->data[0]));
    for (pix = 0; pix < (size_t) INPUT_H * (size_t) INPUT_W; ++pix) {
        uint64_t word = 0ull;
        int c;
        for (c = 0; c < INPUT_C; ++c) {
            uint8_t value = rgb[pix * INPUT_C + (size_t) c];
            if (value >= 51u) {
                word |= 1ull << (c * THERMO_LEVELS + 0);
            }
            if (value >= 102u) {
                word |= 1ull << (c * THERMO_LEVELS + 1);
            }
            if (value >= 153u) {
                word |= 1ull << (c * THERMO_LEVELS + 2);
            }
            if (value >= 204u) {
                word |= 1ull << (c * THERMO_LEVELS + 3);
            }
        }
        output->data[pix] = word;
    }
}

static void pack_binary_input_f32(const float *rgb, PackedMap *output) {
    size_t pix;
    memset(output->data, 0, packed_map_words(output->h, output->w, output->c) * sizeof(output->data[0]));
    for (pix = 0; pix < (size_t) INPUT_H * (size_t) INPUT_W; ++pix) {
        uint64_t word = 0ull;
        int c;
        for (c = 0; c < INPUT_C; ++c) {
            float value = rgb[pix * INPUT_C + (size_t) c];
            if (value >= 0.2f) {
                word |= 1ull << (c * THERMO_LEVELS + 0);
            }
            if (value >= 0.4f) {
                word |= 1ull << (c * THERMO_LEVELS + 1);
            }
            if (value >= 0.6f) {
                word |= 1ull << (c * THERMO_LEVELS + 2);
            }
            if (value >= 0.8f) {
                word |= 1ull << (c * THERMO_LEVELS + 3);
            }
        }
        output->data[pix] = word;
    }
}

static void run_network(int trace_enabled) {
    PackedMap cur = {INPUT_H, INPUT_W, BINARY_INPUT_C, PACK_WORDS(BINARY_INPUT_C), g_buf_a};
    PackedMap nxt = {320, 320, 32, PACK_WORDS(32), g_buf_b};
    PackedMap route_p3 = {40, 40, 160, PACK_WORDS(160), g_route_p3};
    PackedMap route_p4 = {20, 20, 320, PACK_WORDS(320), g_route_p4};
    PackedMap x10 = {10, 10, 1280, PACK_WORDS(1280), g_x10};
    PackedMap tmp_small;
    double t0;

    if (trace_enabled) {
        trace_packed_map("binary_input", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[0], &nxt);
    profile_end(PROFILE_STAGE1_CONV, t0);
    cur = (PackedMap) {320, 320, 32, PACK_WORDS(32), g_buf_b};
    nxt = (PackedMap) {160, 160, 40, PACK_WORDS(40), g_buf_a};
    if (trace_enabled) {
        trace_packed_map("stage1_conv", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[1], &nxt);
    profile_end(PROFILE_STAGE2_CONV, t0);
    cur = (PackedMap) {160, 160, 40, PACK_WORDS(40), g_buf_a};
    nxt = (PackedMap) {80, 80, 80, PACK_WORDS(80), g_buf_b};
    if (trace_enabled) {
        trace_packed_map("stage2_conv", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[2], &nxt);
    profile_end(PROFILE_STAGE3_CONV, t0);
    cur = (PackedMap) {80, 80, 80, PACK_WORDS(80), g_buf_b};
    nxt = (PackedMap) {40, 40, 160, PACK_WORDS(160), g_buf_a};
    if (trace_enabled) {
        trace_packed_map("stage3_conv", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[3], &nxt);
    profile_end(PROFILE_STAGE4_CONV, t0);
    memcpy(route_p3.data, nxt.data, packed_map_words(nxt.h, nxt.w, nxt.c) * sizeof(uint64_t));
    cur = nxt;
    nxt = (PackedMap) {20, 20, 320, PACK_WORDS(320), g_buf_b};
    if (trace_enabled) {
        trace_packed_map("stage4_conv", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[4], &nxt);
    profile_end(PROFILE_STAGE5_CONV, t0);
    memcpy(route_p4.data, nxt.data, packed_map_words(nxt.h, nxt.w, nxt.c) * sizeof(uint64_t));
    cur = nxt;
    nxt = (PackedMap) {10, 10, 640, PACK_WORDS(640), g_buf_a};
    if (trace_enabled) {
        trace_packed_map("stage5_conv", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[5], &nxt);
    profile_end(PROFILE_STAGE6_CONV, t0);
    cur = nxt;
    nxt = (PackedMap) {10, 10, 1280, PACK_WORDS(1280), g_buf_b};
    if (trace_enabled) {
        trace_packed_map("stage6_conv", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[6], &nxt);
    profile_end(PROFILE_STAGE7_CONV, t0);
    cur = nxt;
    nxt = (PackedMap) {10, 10, 640, PACK_WORDS(640), x10.data};
    if (trace_enabled) {
        trace_packed_map("stage7_conv", &cur);
    }

    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[7], &nxt);
    profile_end(PROFILE_STAGE8_CONV, t0);
    if (trace_enabled) {
        trace_packed_map("stage8_conv", &nxt);
    }

    cur = (PackedMap) {10, 10, 640, PACK_WORDS(640), x10.data};
    tmp_small = (PackedMap) {10, 10, 320, PACK_WORDS(320), g_buf_a};
    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[8], &tmp_small);
    profile_end(PROFILE_P5_FEAT_CONV, t0);
    if (trace_enabled) {
        trace_packed_map("p5_feat_conv", &tmp_small);
    }

    nxt = (PackedMap) {10, 10, 320, PACK_WORDS(320), g_buf_b};
    t0 = profile_begin();
    conv_bipolar_run(&tmp_small, &g_binary_layers[15], &nxt);
    profile_end(PROFILE_P5_PRE_CONV, t0);
    if (trace_enabled) {
        trace_packed_map("p5_pre_conv", &nxt);
    }
    t0 = profile_begin();
    run_head_from_pre(
        &nxt,
        &g_raw_proj_layers[2],
        &g_qdense_layers[6],
        &g_qdense_layers[7],
        &g_qdense_layers[8],
        &g_qdense_out_layers[6],
        &g_qdense_out_layers[7],
        &g_qdense_out_layers[8],
        g_p5_head
    );
    profile_end(PROFILE_P5_HEAD, t0);
    if (trace_enabled) {
        trace_qtensor("p5_head", g_p5_head, P5_HEAD_COUNT);
    }

    tmp_small = (PackedMap) {10, 10, 320, PACK_WORDS(320), g_buf_a};
    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[9], &tmp_small);
    profile_end(PROFILE_UP4_REDUCE_CONV, t0);
    nxt = (PackedMap) {20, 20, 320, PACK_WORDS(320), g_buf_b};
    t0 = profile_begin();
    upsample_nearest_packed(&tmp_small, &nxt);
    profile_end(PROFILE_UP4_UPSAMPLE, t0);
    cur = (PackedMap) {20, 20, 640, PACK_WORDS(640), g_buf_a};
    t0 = profile_begin();
    concat_packed(&nxt, &route_p4, &cur);
    profile_end(PROFILE_P4_CONCAT, t0);
    if (trace_enabled) {
        trace_packed_map("p4_concat", &cur);
    }

    nxt = (PackedMap) {20, 20, 320, PACK_WORDS(320), g_buf_b};
    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[10], &nxt);
    profile_end(PROFILE_P4_FEAT_CONV, t0);
    if (trace_enabled) {
        trace_packed_map("p4_feat_conv", &nxt);
    }

    cur = (PackedMap) {20, 20, 320, PACK_WORDS(320), g_buf_a};
    t0 = profile_begin();
    conv_bipolar_run(&nxt, &g_binary_layers[14], &cur);
    profile_end(PROFILE_P4_PRE_CONV, t0);
    if (trace_enabled) {
        trace_packed_map("p4_pre_conv", &cur);
    }
    t0 = profile_begin();
    run_head_from_pre(
        &cur,
        &g_raw_proj_layers[1],
        &g_qdense_layers[3],
        &g_qdense_layers[4],
        &g_qdense_layers[5],
        &g_qdense_out_layers[3],
        &g_qdense_out_layers[4],
        &g_qdense_out_layers[5],
        g_p4_head
    );
    profile_end(PROFILE_P4_HEAD, t0);
    if (trace_enabled) {
        trace_qtensor("p4_head", g_p4_head, P4_HEAD_COUNT);
    }

    cur = (PackedMap) {20, 20, 320, PACK_WORDS(320), g_buf_b};
    nxt = (PackedMap) {20, 20, 160, PACK_WORDS(160), g_buf_a};
    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[11], &nxt);
    profile_end(PROFILE_UP3_REDUCE_CONV, t0);
    cur = (PackedMap) {40, 40, 160, PACK_WORDS(160), g_buf_b};
    t0 = profile_begin();
    upsample_nearest_packed(&nxt, &cur);
    profile_end(PROFILE_UP3_UPSAMPLE, t0);
    nxt = (PackedMap) {40, 40, 320, PACK_WORDS(320), g_buf_a};
    t0 = profile_begin();
    concat_packed(&cur, &route_p3, &nxt);
    profile_end(PROFILE_P3_CONCAT, t0);
    if (trace_enabled) {
        trace_packed_map("p3_concat", &nxt);
    }

    cur = (PackedMap) {40, 40, 160, PACK_WORDS(160), g_buf_b};
    t0 = profile_begin();
    conv_bipolar_run(&nxt, &g_binary_layers[12], &cur);
    profile_end(PROFILE_P3_FEAT_CONV, t0);
    if (trace_enabled) {
        trace_packed_map("p3_feat_conv", &cur);
    }

    nxt = (PackedMap) {40, 40, 320, PACK_WORDS(320), g_buf_a};
    t0 = profile_begin();
    conv_bipolar_run(&cur, &g_binary_layers[13], &nxt);
    profile_end(PROFILE_P3_PRE_CONV, t0);
    if (trace_enabled) {
        trace_packed_map("p3_pre_conv", &nxt);
    }
    t0 = profile_begin();
    run_head_from_pre(
        &nxt,
        &g_raw_proj_layers[0],
        &g_qdense_layers[0],
        &g_qdense_layers[1],
        &g_qdense_layers[2],
        &g_qdense_out_layers[0],
        &g_qdense_out_layers[1],
        &g_qdense_out_layers[2],
        g_p3_head
    );
    profile_end(PROFILE_P3_HEAD, t0);
    if (trace_enabled) {
        trace_qtensor("p3_head", g_p3_head, P3_HEAD_COUNT);
    }
}

static int detection_desc_score(const void *lhs, const void *rhs) {
    const Detection *a = (const Detection *) lhs;
    const Detection *b = (const Detection *) rhs;
    return (int) b->score_q15 - (int) a->score_q15;
}

static int decode_candidates_from_head(
    const int8_t *head,
    int head_h,
    int head_w,
    int anchor_offset,
    float score_threshold,
    float objectness_threshold,
    Detection *candidates,
    int count,
    int capacity
) {
    int y;
    int x;
    int anchor_idx;
    for (y = 0; y < head_h; ++y) {
        for (x = 0; x < head_w; ++x) {
            for (anchor_idx = 0; anchor_idx < NUM_HEAD_ANCHORS; ++anchor_idx) {
                const int8_t *logits = head + ((((size_t) y * (size_t) head_w + (size_t) x) * (size_t) NUM_HEAD_ANCHORS + (size_t) anchor_idx) * (size_t) HEAD_LOGITS);
                float obj = g_sigmoid_lut[(int) logits[4] + 127];
                int c;
                float max_logit = -1e30f;
                float sum_exp = 0.0f;
                float best_prob;
                int best_class = 0;
                if (obj < objectness_threshold) {
                    continue;
                }
                for (c = 0; c < NUM_CLASSES; ++c) {
                    float logit = (float) logits[5 + c] / 32.0f;
                    if (logit > max_logit) {
                        max_logit = logit;
                    }
                }
                for (c = 0; c < NUM_CLASSES; ++c) {
                    sum_exp += expf((float) logits[5 + c] / 32.0f - max_logit);
                }
                best_prob = 0.0f;
                for (c = 0; c < NUM_CLASSES; ++c) {
                    float prob = expf((float) logits[5 + c] / 32.0f - max_logit) / sum_exp;
                    if (prob > best_prob) {
                        best_prob = prob;
                        best_class = c;
                    }
                }
                if (obj * best_prob < score_threshold) {
                    continue;
                }
                if (count < capacity) {
                    float cx = ((float) x + g_u01_lut[(int) logits[0] + 127]) * ((float) INPUT_W / (float) head_w);
                    float cy = ((float) y + g_u01_lut[(int) logits[1] + 127]) * ((float) INPUT_H / (float) head_h);
                    float bw = g_ratio_lut[(int) logits[2] + 127] * g_anchors[anchor_offset + anchor_idx][0];
                    float bh = g_ratio_lut[(int) logits[3] + 127] * g_anchors[anchor_offset + anchor_idx][1];
                    float x1 = cx - bw * 0.5f;
                    float y1 = cy - bh * 0.5f;
                    float x2 = cx + bw * 0.5f;
                    float y2 = cy + bh * 0.5f;
                    if (x1 < 0.0f) {
                        x1 = 0.0f;
                    }
                    if (y1 < 0.0f) {
                        y1 = 0.0f;
                    }
                    if (x2 > (float) INPUT_W) {
                        x2 = (float) INPUT_W;
                    }
                    if (y2 > (float) INPUT_H) {
                        y2 = (float) INPUT_H;
                    }
                    candidates[count].x1 = (uint16_t) lrintf(x1 * 128.0f);
                    candidates[count].y1 = (uint16_t) lrintf(y1 * 128.0f);
                    candidates[count].x2 = (uint16_t) lrintf(x2 * 128.0f);
                    candidates[count].y2 = (uint16_t) lrintf(y2 * 128.0f);
                    candidates[count].score = obj * best_prob;
                    candidates[count].score_q15 = (uint16_t) lrintf(candidates[count].score * 32767.0f);
                    candidates[count].class_id = (uint8_t) best_class;
                }
                count += 1;
            }
        }
    }
    return count;
}

static int iou_exceeds_threshold(const Detection *a, const Detection *b) {
    uint16_t x1 = a->x1 > b->x1 ? a->x1 : b->x1;
    uint16_t y1 = a->y1 > b->y1 ? a->y1 : b->y1;
    uint16_t x2 = a->x2 < b->x2 ? a->x2 : b->x2;
    uint16_t y2 = a->y2 < b->y2 ? a->y2 : b->y2;
    uint64_t inter_w = x2 > x1 ? (uint64_t) (x2 - x1) : 0ull;
    uint64_t inter_h = y2 > y1 ? (uint64_t) (y2 - y1) : 0ull;
    uint64_t inter = inter_w * inter_h;
    uint64_t area_a = (uint64_t) (a->x2 - a->x1) * (uint64_t) (a->y2 - a->y1);
    uint64_t area_b = (uint64_t) (b->x2 - b->x1) * (uint64_t) (b->y2 - b->y1);
    uint64_t uni = area_a + area_b - inter;
    if (uni == 0ull) {
        return 0;
    }
    return inter * 100ull >= 45ull * uni;
}

static int run_nms(Detection *candidates, int count, Detection *output, int max_output) {
    int kept = 0;
    int cls;
    qsort(candidates, (size_t) count, sizeof(candidates[0]), detection_desc_score);
    for (cls = 0; cls < NUM_CLASSES; ++cls) {
        int i;
        for (i = 0; i < count; ++i) {
            int suppress = 0;
            int j;
            if ((int) candidates[i].class_id != cls) {
                continue;
            }
            for (j = 0; j < kept; ++j) {
                if ((int) output[j].class_id == cls && iou_exceeds_threshold(&candidates[i], &output[j])) {
                    suppress = 1;
                    break;
                }
            }
            if (!suppress && kept < max_output) {
                output[kept++] = candidates[i];
            }
        }
    }
    qsort(output, (size_t) kept, sizeof(output[0]), detection_desc_score);
    return kept;
}

static int run_postprocess(Detection *candidates, int *candidate_count_out, Detection *detections) {
    int candidate_count = 0;
    double t0;
    t0 = profile_begin();
    candidate_count = decode_candidates_from_head(g_p3_head, P3_H, P3_W, 0, 0.20f, 0.05f, candidates, candidate_count, MAX_CANDIDATES);
    profile_end(PROFILE_POST_P3_DECODE, t0);
    t0 = profile_begin();
    candidate_count = decode_candidates_from_head(g_p4_head, P4_H, P4_W, 3, 0.20f, 0.05f, candidates, candidate_count, MAX_CANDIDATES);
    profile_end(PROFILE_POST_P4_DECODE, t0);
    t0 = profile_begin();
    candidate_count = decode_candidates_from_head(g_p5_head, P5_H, P5_W, 6, 0.20f, 0.05f, candidates, candidate_count, MAX_CANDIDATES);
    profile_end(PROFILE_POST_P5_DECODE, t0);
    if (candidate_count > MAX_CANDIDATES) {
        candidate_count = MAX_CANDIDATES;
    }
    if (candidate_count_out != NULL) {
        *candidate_count_out = candidate_count;
    }
    t0 = profile_begin();
    candidate_count = run_nms(candidates, candidate_count, detections, MAX_FINAL_DETECTIONS);
    profile_end(PROFILE_POST_NMS, t0);
    return candidate_count;
}

static int dump_head_file(const char *path) {
    FILE *handle = fopen(path, "wb");
    uint32_t counts[3];
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }
    counts[0] = (uint32_t) P3_HEAD_COUNT;
    counts[1] = (uint32_t) P4_HEAD_COUNT;
    counts[2] = (uint32_t) P5_HEAD_COUNT;
    if (fwrite(HEAD_DUMP_MAGIC, 1, 8, handle) != 8
        || fwrite(counts, sizeof(counts[0]), 3, handle) != 3
        || fwrite(g_p3_head, sizeof(g_p3_head[0]), P3_HEAD_COUNT, handle) != P3_HEAD_COUNT
        || fwrite(g_p4_head, sizeof(g_p4_head[0]), P4_HEAD_COUNT, handle) != P4_HEAD_COUNT
        || fwrite(g_p5_head, sizeof(g_p5_head[0]), P5_HEAD_COUNT, handle) != P5_HEAD_COUNT) {
        fprintf(stderr, "failed to write %s\n", path);
        fclose(handle);
        return -1;
    }
    fclose(handle);
    return 0;
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --weights FILE (--input-u8 FILE | --input-f32 FILE) [--dump-heads FILE] [--dump-detections FILE] [--trace] [--benchmark ITERS] [--profile]\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *weights_path = NULL;
    const char *input_u8_path = NULL;
    const char *input_f32_path = NULL;
    const char *dump_heads_path = NULL;
    const char *dump_dets_path = NULL;
    int trace_enabled = 0;
    int benchmark_iters = 0;
    int profile_enabled = 0;
    int argi;
    uint8_t *input_u8 = g_input_u8;
    float *input_f32 = g_input_f32;
    PackedMap input_map = {INPUT_H, INPUT_W, BINARY_INPUT_C, PACK_WORDS(BINARY_INPUT_C), g_buf_a};
    Detection candidates[MAX_CANDIDATES];
    Detection detections[MAX_FINAL_DETECTIONS];
    int candidate_count;
    int det_count;

    init_lookup_tables();

    for (argi = 1; argi < argc; ++argi) {
        if (strcmp(argv[argi], "--weights") == 0 && argi + 1 < argc) {
            weights_path = argv[++argi];
        } else if (strcmp(argv[argi], "--input-u8") == 0 && argi + 1 < argc) {
            input_u8_path = argv[++argi];
        } else if (strcmp(argv[argi], "--input-f32") == 0 && argi + 1 < argc) {
            input_f32_path = argv[++argi];
        } else if (strcmp(argv[argi], "--dump-heads") == 0 && argi + 1 < argc) {
            dump_heads_path = argv[++argi];
        } else if (strcmp(argv[argi], "--dump-detections") == 0 && argi + 1 < argc) {
            dump_dets_path = argv[++argi];
        } else if (strcmp(argv[argi], "--trace") == 0) {
            trace_enabled = 1;
        } else if (strcmp(argv[argi], "--benchmark") == 0 && argi + 1 < argc) {
            benchmark_iters = atoi(argv[++argi]);
            if (benchmark_iters < 0) {
                benchmark_iters = 0;
            }
        } else if (strcmp(argv[argi], "--profile") == 0) {
            profile_enabled = 1;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (weights_path == NULL || (input_u8_path == NULL && input_f32_path == NULL) || (input_u8_path != NULL && input_f32_path != NULL)) {
        print_usage(argv[0]);
        return 2;
    }

    if (load_weight_file(weights_path) != 0) {
        free_loaded_weights();
        return 1;
    }

    if (input_u8_path != NULL) {
        size_t input_bytes = (size_t) INPUT_H * (size_t) INPUT_W * (size_t) INPUT_C;
        if (load_input_u8(input_u8_path, input_u8, input_bytes) != 0) {
            free_loaded_weights();
            return 1;
        }
        pack_binary_input_u8(input_u8, &input_map);
    } else {
        size_t input_count = (size_t) INPUT_H * (size_t) INPUT_W * (size_t) INPUT_C;
        if (load_input_f32(input_f32_path, input_f32, input_count) != 0) {
            free_loaded_weights();
            return 1;
        }
        pack_binary_input_f32(input_f32, &input_map);
    }

    if (benchmark_iters > 0) {
        double network_total = 0.0;
        double post_total = 0.0;
        int iter;
        g_profile_enabled = profile_enabled;
        run_network(0);
        det_count = run_postprocess(candidates, &candidate_count, detections);
        profile_reset();
        for (iter = 0; iter < benchmark_iters; ++iter) {
            double t0 = wall_seconds();
            run_network(0);
            double t1 = wall_seconds();
            det_count = run_postprocess(candidates, &candidate_count, detections);
            double t2 = wall_seconds();
            network_total += t1 - t0;
            post_total += t2 - t1;
        }
        fprintf(stderr,
                "benchmark iterations=%d network_ms=%.3f post_ms=%.3f total_ms=%.3f fps=%.2f candidates=%d detections=%d\n",
                benchmark_iters,
                network_total * 1000.0 / (double) benchmark_iters,
                post_total * 1000.0 / (double) benchmark_iters,
                (network_total + post_total) * 1000.0 / (double) benchmark_iters,
                (double) benchmark_iters / (network_total + post_total),
                candidate_count,
                det_count);
        profile_print_summary(network_total, post_total, benchmark_iters);
    } else {
        g_profile_enabled = 0;
        run_network(trace_enabled);
        det_count = run_postprocess(candidates, &candidate_count, detections);
    }

    if (dump_heads_path != NULL && dump_head_file(dump_heads_path) != 0) {
        free_loaded_weights();
        return 1;
    }

    if (dump_dets_path != NULL) {
        FILE *handle = fopen(dump_dets_path, "w");
        int i;
        if (handle == NULL) {
            fprintf(stderr, "failed to open %s: %s\n", dump_dets_path, strerror(errno));
            free_loaded_weights();
            return 1;
        }
        for (i = 0; i < det_count; ++i) {
            fprintf(handle,
                    "%s %.6f %.2f %.2f %.2f %.2f\n",
                    g_class_names[detections[i].class_id],
                    detections[i].score,
                    (double) detections[i].x1 / 128.0,
                    (double) detections[i].y1 / 128.0,
                    (double) detections[i].x2 / 128.0,
                    (double) detections[i].y2 / 128.0);
        }
        fclose(handle);
    } else {
        int i;
        for (i = 0; i < det_count; ++i) {
            printf("%s %.6f %.2f %.2f %.2f %.2f\n",
                   g_class_names[detections[i].class_id],
                   detections[i].score,
                   (double) detections[i].x1 / 128.0,
                   (double) detections[i].y1 / 128.0,
                   (double) detections[i].x2 / 128.0,
                   (double) detections[i].y2 / 128.0);
        }
    }

    free_loaded_weights();
    return 0;
}
