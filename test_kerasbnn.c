#define _POSIX_C_SOURCE 200809L

#include "kerasbnn.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void test_xnor_dot(void) {
    const uint64_t a[2] = {0xadull, 0xbull};
    const uint64_t b[2] = {0xf1ull, 0x3ull};
    const uint64_t last_mask = 0xfull;
    const int32_t dot = kbnn_xnor_dot_u64(a, b, 2, last_mask);
    int32_t ref = 0;
    for (int i = 0; i < 64; ++i) {
        ref += (((a[0] >> i) & 1ull) == ((b[0] >> i) & 1ull)) ? 1 : -1;
    }
    for (int i = 0; i < 4; ++i) {
        ref += (((a[1] >> i) & 1ull) == ((b[1] >> i) & 1ull)) ? 1 : -1;
    }
    assert(dot == ref);
}

static void test_thermo_qconv(void) {
    static const kbnn_tensor_desc tensors[] = {
        {.name = "image", .kind = KBNN_TENSOR_EXTERNAL_F32, .rank = 3, .shape = {1, 1, 1, 0, 0}, .words_per_point = 0, .offset_bytes = 0, .bytes = 0},
        {.name = "thermo", .kind = KBNN_TENSOR_PACKED_BINARY, .rank = 3, .shape = {1, 1, 4, 0, 0}, .words_per_point = 1, .offset_bytes = 0, .bytes = 8},
        {.name = "conv", .kind = KBNN_TENSOR_I32, .rank = 3, .shape = {1, 1, 1, 0, 0}, .words_per_point = 0, .offset_bytes = 16, .bytes = 4},
    };
    static const int16_t weights[] = {1, -2, 3, -4};
    static const kbnn_thermo_input_params thermo = {
        .input_id = 0, .output_id = 1, .height = 1, .width = 1, .channels = 1, .levels = 4, .min_value = 0.0f, .max_value = 1.0f,
    };
    static const kbnn_qconv2d_params qconv = {
        .input_id = 1, .output_id = 2,
        .in_h = 1, .in_w = 1, .in_c = 4,
        .out_h = 1, .out_w = 1, .out_c = 1,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
        .input_is_packed_binary = 1,
        .weight_divisor = 1,
        .levels = 16,
        .integer_output = 0,
        .output_scale_q = KBNN_FX_ONE,
        .weights = weights,
        .bias_q = NULL,
    };
    static const kbnn_op ops[] = {
        {.kind = KBNN_OP_THERMO_INPUT, .name = "thermo", .params = &thermo},
        {.kind = KBNN_OP_QCONV2D, .name = "conv", .params = &qconv},
    };
    static const int32_t outputs[] = {2};
    static const kbnn_model model = {
        .name = "mini",
        .tensors = tensors,
        .tensor_count = 3,
        .ops = ops,
        .op_count = 2,
        .output_ids = outputs,
        .output_count = 1,
        .arena_bytes = 32,
    };
    float image[1] = {0.75f};
    uint8_t arena[32];
    kbnn_context ctx;
    int rc = kbnn_context_init(&ctx, &model, arena, sizeof(arena));
    assert(rc == 0);
    kbnn_context_set_input_f32(&ctx, 0, image);
    rc = kbnn_model_run(&ctx);
    assert(rc == 0);
    {
        const int32_t *out = kbnn_context_get_output_i32(&ctx, 2);
        /* 0.75 -> thermo bits [1,1,1,0], as signed [+1,+1,+1,-1]. */
        const int32_t expected = (1 * 1 + 1 * -2 + 1 * 3 + -1 * -4) * KBNN_FX_ONE;
        assert(out[0] == expected);
    }
    kbnn_context_release(&ctx);
}

static void benchmark_xnor_dot(void) {
    uint64_t a[8];
    uint64_t b[8];
    int64_t sink = 0;
    for (int i = 0; i < 8; ++i) {
        a[i] = 0xaaaaaaaa55555555ull ^ (uint64_t)(i * 0x1111111111111111ull);
        b[i] = 0x33333333ccccccccull ^ (uint64_t)(i * 0x0101010101010101ull);
    }
    {
        const int iters = 200000;
        uint64_t t0 = now_ns();
        for (int i = 0; i < iters; ++i) {
            sink += kbnn_xnor_dot_u64(a, b, 8, ~0ull);
        }
        uint64_t dt = now_ns() - t0;
        printf("xnor_dot: %.3f ns/iter (sink=%lld)\n", (double)dt / (double)iters, (long long)sink);
    }
}

int main(void) {
    test_xnor_dot();
    test_thermo_qconv();
    benchmark_xnor_dot();
    printf("kerasbnn C tests passed\n");
    return 0;
}
