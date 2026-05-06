#define _POSIX_C_SOURCE 200809L
#include "kerasbnn.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef KBNN_ENABLE_PROFILE
#include <time.h>
#endif

#ifdef KBNN_ENABLE_PROFILE
static inline uint64_t kbnn_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

static inline int32_t kbnn_clip_i32(int64_t value, int32_t lo, int32_t hi) {
    if (value < (int64_t)lo) {
        return lo;
    }
    if (value > (int64_t)hi) {
        return hi;
    }
    return (int32_t)value;
}

static inline int32_t kbnn_round_div_i64(int64_t num, int32_t den) {
    if (den <= 1) {
        return (int32_t)num;
    }
    if (num >= 0) {
        return (int32_t)((num + den / 2) / den);
    }
    return (int32_t)(-(((-num) + den / 2) / den));
}

static inline int32_t kbnn_floor_to_step_i32(int32_t x, int32_t step) {
    if (step <= 1) {
        return x;
    }
    if (x >= 0) {
        return (x / step) * step;
    }
    return -(((-x + step - 1) / step) * step);
}

static inline int32_t kbnn_round_to_step_i32(int32_t x, int32_t step) {
    if (step <= 1) {
        return x;
    }
    if (x >= 0) {
        return ((x + step / 2) / step) * step;
    }
    return -(((-x + step / 2) / step) * step);
}

static inline int32_t kbnn_quantize_signed_unit_i32(int32_t x, int32_t levels) {
    const int32_t max_q = ((levels - 1) * KBNN_FX_ONE) / levels;
    const int32_t step = KBNN_FX_ONE / levels;
    x = kbnn_clip_i32(x, -KBNN_FX_ONE, max_q);
    return kbnn_floor_to_step_i32(x, step);
}

static inline int32_t kbnn_quantize_unit_i32(int32_t x, int32_t levels) {
    const int32_t step = KBNN_FX_ONE / levels;
    x = kbnn_clip_i32(x, 0, KBNN_FX_ONE);
    return kbnn_floor_to_step_i32(x, step);
}

static inline int32_t kbnn_fake_quant_output_i32(int32_t x, int32_t levels, int32_t scale_q) {
    const int32_t qmax = levels - 1;
    const int32_t max_q = qmax * scale_q;
    x = kbnn_clip_i32(x, -max_q, max_q);
    return kbnn_round_to_step_i32(x, scale_q);
}

static inline size_t kbnn_numel_from_shape(const int32_t *shape, int32_t rank) {
    size_t total = 1;
    for (int32_t i = 0; i < rank; ++i) {
        total *= (size_t)shape[i];
    }
    return total;
}

static inline int32_t kbnn_apply_pre_linear(int32_t x_q, int32_t mul_q, int32_t add_q) {
    return (int32_t)(((int64_t)mul_q * (int64_t)x_q) >> KBNN_FX_SHIFT) + add_q;
}

static inline int32_t kbnn_finalize_linear_output_i32(
    int64_t dot_acc,
    int32_t bias_q,
    int32_t weight_divisor,
    int32_t integer_output,
    int32_t levels,
    int32_t output_scale_q
) {
    if (weight_divisor > 1) {
        dot_acc = kbnn_round_div_i64(dot_acc, weight_divisor);
    }
    {
        const int64_t total = dot_acc + (int64_t)bias_q;
        int32_t out_q = kbnn_clip_i32(total, INT32_MIN / 2, INT32_MAX / 2);
        if (integer_output) {
            out_q = kbnn_fake_quant_output_i32(out_q, levels, output_scale_q);
        }
        return out_q;
    }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#if defined(__FLT16_MAX__) || defined(__GNUC__) || defined(__clang__)
static inline float kbnn_fp16_like(float x) {
    _Float16 h = (_Float16)x;
    return (float)h;
}
#else
static inline float kbnn_fp16_like(float x) {
    return x;
}
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

static inline int32_t kbnn_round_half_away_from_zero_f32(float x) {
    if (x >= 0.0f) {
        return (int32_t)floorf(x + 0.5f);
    }
    return (int32_t)(-floorf(-x + 0.5f));
}

static inline int32_t kbnn_fx_cast_via_fp16(int32_t x_q) {
    const float x = (float)x_q / (float)KBNN_FX_ONE;
    const float x_h = kbnn_fp16_like(x);
    return kbnn_round_half_away_from_zero_f32(x_h * (float)KBNN_FX_ONE);
}

static inline int32_t kbnn_apply_xnor_affine_f32(int32_t acc, int32_t alpha_q, int32_t bias_q) {
    const float scale = (float)alpha_q / (float)KBNN_FX_ONE;
    const float bias = (float)bias_q / (float)KBNN_FX_ONE;
    const float value = fmaf((float)acc, scale, bias);
    const float scaled = value * (float)KBNN_FX_ONE;
    const float fixed = (float)kbnn_round_half_away_from_zero_f32(scaled);
    if (fixed > (float)INT32_MAX) {
        return INT32_MAX;
    }
    if (fixed < (float)INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)fixed;
}

static inline float kbnn_thermo_normalize_f32(float input, float min_value, float inv_range) {
    float x = (input - min_value) * inv_range;
    if (x < 0.0f) {
        x = 0.0f;
    }
    if (x > 1.0f) {
        x = 1.0f;
    }
    return x;
}

#if defined(__clang__) || defined(__GNUC__)
#define KBNN_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#else
#define KBNN_ASSUME_ALIGNED(ptr, align) (ptr)
#endif

static inline int32_t *kbnn_tensor_i32(const kbnn_context *ctx, int32_t tensor_id) {
    return (int32_t *)ctx->tensor_ptrs[tensor_id];
}

static inline uint64_t *kbnn_tensor_bits(const kbnn_context *ctx, int32_t tensor_id) {
    return (uint64_t *)ctx->tensor_ptrs[tensor_id];
}

static inline const float *kbnn_tensor_f32(const kbnn_context *ctx, int32_t tensor_id) {
    return (const float *)ctx->tensor_ptrs[tensor_id];
}

static inline size_t kbnn_packed_words_for_point(int32_t channels) {
    return (size_t)(channels + 63) / 64;
}

int32_t kbnn_xnor_dot_u64(const uint64_t *a, const uint64_t *b, int32_t words, uint64_t last_mask) {
    int32_t dot = 0;
    const int32_t full_words = words - 1;
    for (int32_t w = 0; w < full_words; ++w) {
        dot += 64 - 2 * (int32_t)__builtin_popcountll(a[w] ^ b[w]);
    }
    if (words > 0) {
        const int32_t active = (int32_t)__builtin_popcountll(last_mask);
        const uint64_t mism = (a[words - 1] ^ b[words - 1]) & last_mask;
        dot += active - 2 * (int32_t)__builtin_popcountll(mism);
    }
    return dot;
}

static inline void kbnn_xnor_accum4(
    const uint64_t *restrict in_words,
    const uint64_t *restrict w0,
    const uint64_t *restrict w1,
    const uint64_t *restrict w2,
    const uint64_t *restrict w3,
    int32_t channel_words,
    uint64_t last_mask,
    int32_t last_active,
    int32_t *acc0,
    int32_t *acc1,
    int32_t *acc2,
    int32_t *acc3
) {
    const int32_t full_words = channel_words - 1;
    for (int32_t w = 0; w < full_words; ++w) {
        const uint64_t x = in_words[w];
        *acc0 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w0[w]);
        *acc1 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w1[w]);
        *acc2 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w2[w]);
        *acc3 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w3[w]);
    }
    if (channel_words > 0) {
        const int32_t last = channel_words - 1;
        const uint64_t x = in_words[last];
        *acc0 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w0[last]) & last_mask);
        *acc1 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w1[last]) & last_mask);
        *acc2 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w2[last]) & last_mask);
        *acc3 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w3[last]) & last_mask);
    }
}

static inline void kbnn_xnor_accum8(
    const uint64_t *restrict in_words,
    const uint64_t *restrict w0,
    const uint64_t *restrict w1,
    const uint64_t *restrict w2,
    const uint64_t *restrict w3,
    const uint64_t *restrict w4,
    const uint64_t *restrict w5,
    const uint64_t *restrict w6,
    const uint64_t *restrict w7,
    int32_t channel_words,
    uint64_t last_mask,
    int32_t last_active,
    int32_t *acc0,
    int32_t *acc1,
    int32_t *acc2,
    int32_t *acc3,
    int32_t *acc4,
    int32_t *acc5,
    int32_t *acc6,
    int32_t *acc7
) {
    const int32_t full_words = channel_words - 1;
    for (int32_t w = 0; w < full_words; ++w) {
        const uint64_t x = in_words[w];
        *acc0 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w0[w]);
        *acc1 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w1[w]);
        *acc2 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w2[w]);
        *acc3 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w3[w]);
        *acc4 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w4[w]);
        *acc5 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w5[w]);
        *acc6 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w6[w]);
        *acc7 += 64 - 2 * (int32_t)__builtin_popcountll(x ^ w7[w]);
    }
    if (channel_words > 0) {
        const int32_t last = channel_words - 1;
        const uint64_t x = in_words[last];
        *acc0 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w0[last]) & last_mask);
        *acc1 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w1[last]) & last_mask);
        *acc2 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w2[last]) & last_mask);
        *acc3 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w3[last]) & last_mask);
        *acc4 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w4[last]) & last_mask);
        *acc5 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w5[last]) & last_mask);
        *acc6 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w6[last]) & last_mask);
        *acc7 += last_active - 2 * (int32_t)__builtin_popcountll((x ^ w7[last]) & last_mask);
    }
}

static inline int32_t kbnn_xnor_dot_masked_word(uint64_t x, uint64_t w, uint64_t mask, int32_t active) {
    return active - 2 * (int32_t)__builtin_popcountll((x ^ w) & mask);
}

static void kbnn_exec_thermo_input(const kbnn_context *ctx, const kbnn_thermo_input_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const float *input = kbnn_tensor_f32(ctx, p->input_id);
    uint64_t *output = kbnn_tensor_bits(ctx, p->output_id);
    const size_t spatial = (size_t)p->height * (size_t)p->width;
    const int32_t out_channels = p->channels * p->levels;
    const size_t words_per_point = kbnn_packed_words_for_point(out_channels);
    const float inv_range = 1.0f / (p->max_value - p->min_value);
    #ifndef KBNN_DISABLE_FAST_THERMO
    if (words_per_point == 1 && p->levels > 0 && p->levels < 64) {
        const uint64_t ones_mask = (p->levels == 64) ? UINT64_MAX : ((1ull << p->levels) - 1ull);
        for (int32_t y = 0; y < p->height; ++y) {
            for (int32_t x = 0; x < p->width; ++x) {
                uint64_t packed = 0ull;
                const size_t pixel_base = ((size_t)y * (size_t)p->width + (size_t)x) * (size_t)p->channels;
                for (int32_t c = 0; c < p->channels; ++c) {
                    const float raw = input[pixel_base + (size_t)c];
                    const float v = kbnn_thermo_normalize_f32(raw, p->min_value, inv_range);
                    int32_t active = (int32_t)floorf(v * (float)(p->levels + 1) + 1e-6f);
                    if (active < 0) {
                        active = 0;
                    } else if (active > p->levels) {
                        active = p->levels;
                    }
                    if (active > 0) {
                        const uint64_t mask = (active >= 64) ? UINT64_MAX : ((1ull << active) - 1ull);
                        packed |= (mask & ones_mask) << (c * p->levels);
                    }
                }
                output[(size_t)y * (size_t)p->width + (size_t)x] = packed;
            }
        }
#ifdef KBNN_ENABLE_PROFILE
        ((kbnn_context *)ctx)->profile.thermo_input_ns += kbnn_now_ns() - t0;
#endif
        return;
    }
    #endif
    memset(output, 0, spatial * words_per_point * sizeof(uint64_t));
    for (int32_t y = 0; y < p->height; ++y) {
        for (int32_t x = 0; x < p->width; ++x) {
            const size_t base = ((size_t)y * (size_t)p->width + (size_t)x) * words_per_point;
            for (int32_t c = 0; c < p->channels; ++c) {
                const float raw = input[((size_t)y * (size_t)p->width + (size_t)x) * (size_t)p->channels + (size_t)c];
                const float v = kbnn_thermo_normalize_f32(raw, p->min_value, inv_range);
                for (int32_t level = 0; level < p->levels; ++level) {
                    const float threshold = (float)(level + 1) / (float)(p->levels + 1);
                    const int bit = (v >= threshold);
                    const int32_t out_c = c * p->levels + level;
                    const size_t word_idx = base + (size_t)(out_c / 64);
                    const int32_t bit_idx = out_c & 63;
                    if (bit) {
                        output[word_idx] |= (1ull << bit_idx);
                    }
                }
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.thermo_input_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_quantized_input(const kbnn_context *ctx, const kbnn_quantized_input_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const float *input = kbnn_tensor_f32(ctx, p->input_id);
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    const size_t count = kbnn_numel_from_shape(p->shape, p->rank);
    const float inv_range = 1.0f / (p->max_value - p->min_value);

    for (size_t i = 0; i < count; ++i) {
        float v = (input[i] - p->min_value) * inv_range;
        if (v < 0.0f) {
            v = 0.0f;
        }
        if (v > 1.0f) {
            v = 1.0f;
        }
        int32_t q = (int32_t)lrintf(v * (float)KBNN_FX_ONE);
        if (p->signed_output) {
            q = q * 2 - KBNN_FX_ONE;
            output[i] = kbnn_quantize_signed_unit_i32(q, p->levels);
        } else {
            output[i] = kbnn_quantize_unit_i32(q, p->levels);
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.quantized_input_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_qconv2d(const kbnn_context *ctx, const kbnn_qconv2d_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    const size_t kernel_stride = (size_t)p->kernel_h * (size_t)p->kernel_w * (size_t)p->in_c;
    if (p->input_is_packed_binary) {
        const uint64_t *restrict input_bits = kbnn_tensor_bits(ctx, p->input_id);
        const size_t packed_words = kbnn_packed_words_for_point(p->in_c);
        for (int32_t oh = 0; oh < p->out_h; ++oh) {
            for (int32_t ow = 0; ow < p->out_w; ++ow) {
                int32_t oc = 0;
                for (; oc + 3 < p->out_c; oc += 4) {
                    int64_t acc0 = 0;
                    int64_t acc1 = 0;
                    int64_t acc2 = 0;
                    int64_t acc3 = 0;
                    for (int32_t kh = 0; kh < p->kernel_h; ++kh) {
                        const int32_t iy = oh * p->stride_h + kh - p->pad_top;
                        if (iy < 0 || iy >= p->in_h) {
                            continue;
                        }
                        for (int32_t kw = 0; kw < p->kernel_w; ++kw) {
                            const int32_t ix = ow * p->stride_w + kw - p->pad_left;
                            if (ix < 0 || ix >= p->in_w) {
                                continue;
                            }
                            const uint64_t *restrict point =
                                input_bits + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * packed_words);
                            const size_t kbase = ((size_t)kh * (size_t)p->kernel_w + (size_t)kw) * (size_t)p->in_c;
                            const int16_t *restrict w0 = p->weights + (size_t)(oc + 0) * kernel_stride + kbase;
                            const int16_t *restrict w1 = p->weights + (size_t)(oc + 1) * kernel_stride + kbase;
                            const int16_t *restrict w2 = p->weights + (size_t)(oc + 2) * kernel_stride + kbase;
                            const int16_t *restrict w3 = p->weights + (size_t)(oc + 3) * kernel_stride + kbase;
                            for (size_t w = 0; w < packed_words; ++w) {
                                const uint64_t word = point[w];
                                const int32_t base_c = (int32_t)(w * 64u);
                                int32_t limit = p->in_c - base_c;
                                if (limit > 64) {
                                    limit = 64;
                                }
                                for (int32_t bit = 0; bit < limit; ++bit) {
                                    const int64_t sign = ((word >> bit) & 1ull) ? (int64_t)KBNN_FX_ONE : -(int64_t)KBNN_FX_ONE;
                                    acc0 += sign * (int64_t)w0[base_c + bit];
                                    acc1 += sign * (int64_t)w1[base_c + bit];
                                    acc2 += sign * (int64_t)w2[base_c + bit];
                                    acc3 += sign * (int64_t)w3[base_c + bit];
                                }
                            }
                        }
                    }
                    {
                        const size_t out_base = ((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                        const int32_t bias0 = (p->bias_q != NULL) ? p->bias_q[oc + 0] : 0;
                        const int32_t bias1 = (p->bias_q != NULL) ? p->bias_q[oc + 1] : 0;
                        const int32_t bias2 = (p->bias_q != NULL) ? p->bias_q[oc + 2] : 0;
                        const int32_t bias3 = (p->bias_q != NULL) ? p->bias_q[oc + 3] : 0;
                        int32_t out0 = kbnn_finalize_linear_output_i32(acc0, bias0, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        int32_t out1 = kbnn_finalize_linear_output_i32(acc1, bias1, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        int32_t out2 = kbnn_finalize_linear_output_i32(acc2, bias2, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        int32_t out3 = kbnn_finalize_linear_output_i32(acc3, bias3, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        output[out_base + 0] = out0;
                        output[out_base + 1] = out1;
                        output[out_base + 2] = out2;
                        output[out_base + 3] = out3;
                    }
                }
                for (; oc < p->out_c; ++oc) {
                    int64_t acc = 0;
                    for (int32_t kh = 0; kh < p->kernel_h; ++kh) {
                        const int32_t iy = oh * p->stride_h + kh - p->pad_top;
                        if (iy < 0 || iy >= p->in_h) {
                            continue;
                        }
                        for (int32_t kw = 0; kw < p->kernel_w; ++kw) {
                            const int32_t ix = ow * p->stride_w + kw - p->pad_left;
                            if (ix < 0 || ix >= p->in_w) {
                                continue;
                            }
                            const uint64_t *restrict point =
                                input_bits + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * packed_words);
                            const size_t kbase = ((size_t)kh * (size_t)p->kernel_w + (size_t)kw) * (size_t)p->in_c;
                            const int16_t *restrict wrow = p->weights + (size_t)oc * kernel_stride + kbase;
                            for (size_t w = 0; w < packed_words; ++w) {
                                const uint64_t word = point[w];
                                const int32_t base_c = (int32_t)(w * 64u);
                                int32_t limit = p->in_c - base_c;
                                if (limit > 64) {
                                    limit = 64;
                                }
                                for (int32_t bit = 0; bit < limit; ++bit) {
                                    const int64_t sign = ((word >> bit) & 1ull) ? (int64_t)KBNN_FX_ONE : -(int64_t)KBNN_FX_ONE;
                                    acc += sign * (int64_t)wrow[base_c + bit];
                                }
                            }
                        }
                    }
                    {
                        const int32_t bias_q = (p->bias_q != NULL) ? p->bias_q[oc] : 0;
                        int32_t out_q = kbnn_finalize_linear_output_i32(acc, bias_q, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        output[((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc] = out_q;
                    }
                }
            }
        }
    } else {
        const int32_t *restrict input_fx = kbnn_tensor_i32(ctx, p->input_id);
        for (int32_t oh = 0; oh < p->out_h; ++oh) {
            for (int32_t ow = 0; ow < p->out_w; ++ow) {
                int32_t oc = 0;
                for (; oc + 3 < p->out_c; oc += 4) {
                    int64_t acc0 = 0;
                    int64_t acc1 = 0;
                    int64_t acc2 = 0;
                    int64_t acc3 = 0;
                    for (int32_t kh = 0; kh < p->kernel_h; ++kh) {
                        const int32_t iy = oh * p->stride_h + kh - p->pad_top;
                        if (iy < 0 || iy >= p->in_h) {
                            continue;
                        }
                        for (int32_t kw = 0; kw < p->kernel_w; ++kw) {
                            const int32_t ix = ow * p->stride_w + kw - p->pad_left;
                            if (ix < 0 || ix >= p->in_w) {
                                continue;
                            }
                            const int32_t *restrict point =
                                input_fx + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * (size_t)p->in_c);
                            const size_t kbase = ((size_t)kh * (size_t)p->kernel_w + (size_t)kw) * (size_t)p->in_c;
                            const int16_t *restrict w0 = p->weights + (size_t)(oc + 0) * kernel_stride + kbase;
                            const int16_t *restrict w1 = p->weights + (size_t)(oc + 1) * kernel_stride + kbase;
                            const int16_t *restrict w2 = p->weights + (size_t)(oc + 2) * kernel_stride + kbase;
                            const int16_t *restrict w3 = p->weights + (size_t)(oc + 3) * kernel_stride + kbase;
                            for (int32_t c = 0; c < p->in_c; ++c) {
                                const int64_t xv = point[c];
                                acc0 += xv * (int64_t)w0[c];
                                acc1 += xv * (int64_t)w1[c];
                                acc2 += xv * (int64_t)w2[c];
                                acc3 += xv * (int64_t)w3[c];
                            }
                        }
                    }
                    {
                        const size_t out_base = ((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                        const int32_t bias0 = (p->bias_q != NULL) ? p->bias_q[oc + 0] : 0;
                        const int32_t bias1 = (p->bias_q != NULL) ? p->bias_q[oc + 1] : 0;
                        const int32_t bias2 = (p->bias_q != NULL) ? p->bias_q[oc + 2] : 0;
                        const int32_t bias3 = (p->bias_q != NULL) ? p->bias_q[oc + 3] : 0;
                        int32_t out0 = kbnn_finalize_linear_output_i32(acc0, bias0, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        int32_t out1 = kbnn_finalize_linear_output_i32(acc1, bias1, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        int32_t out2 = kbnn_finalize_linear_output_i32(acc2, bias2, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        int32_t out3 = kbnn_finalize_linear_output_i32(acc3, bias3, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        output[out_base + 0] = out0;
                        output[out_base + 1] = out1;
                        output[out_base + 2] = out2;
                        output[out_base + 3] = out3;
                    }
                }
                for (; oc < p->out_c; ++oc) {
                    int64_t acc = 0;
                    for (int32_t kh = 0; kh < p->kernel_h; ++kh) {
                        const int32_t iy = oh * p->stride_h + kh - p->pad_top;
                        if (iy < 0 || iy >= p->in_h) {
                            continue;
                        }
                        for (int32_t kw = 0; kw < p->kernel_w; ++kw) {
                            const int32_t ix = ow * p->stride_w + kw - p->pad_left;
                            if (ix < 0 || ix >= p->in_w) {
                                continue;
                            }
                            const int32_t *restrict point =
                                input_fx + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * (size_t)p->in_c);
                            const size_t kbase = ((size_t)kh * (size_t)p->kernel_w + (size_t)kw) * (size_t)p->in_c;
                            const int16_t *restrict wrow = p->weights + (size_t)oc * kernel_stride + kbase;
                            for (int32_t c = 0; c < p->in_c; ++c) {
                                acc += (int64_t)point[c] * (int64_t)wrow[c];
                            }
                        }
                    }
                    {
                        const int32_t bias_q = (p->bias_q != NULL) ? p->bias_q[oc] : 0;
                        int32_t out_q = kbnn_finalize_linear_output_i32(acc, bias_q, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                        output[((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc] = out_q;
                    }
                }
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.qconv2d_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_prepare_binary_input(const kbnn_context *ctx, const kbnn_xnor_conv2d_params *p) {
    const kbnn_tensor_desc *input_desc = &ctx->model->tensors[p->input_id];
    const int32_t *input_i32 = NULL;
    const float *input_f32 = NULL;
    const uint64_t *input_bits = NULL;
    uint64_t *restrict scratch = kbnn_tensor_bits(ctx, p->scratch_id);
    if (input_desc->kind == KBNN_TENSOR_EXTERNAL_F32) {
        input_f32 = kbnn_tensor_f32(ctx, p->input_id);
    } else if (input_desc->kind == KBNN_TENSOR_PACKED_BINARY) {
        input_bits = kbnn_tensor_bits(ctx, p->input_id);
    } else {
        input_i32 = kbnn_tensor_i32(ctx, p->input_id);
    }

#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    if (input_bits != NULL) {
        memcpy(
            scratch,
            input_bits,
            (size_t)p->in_h * (size_t)p->in_w * (size_t)p->channel_words * sizeof(uint64_t)
        );
#ifdef KBNN_ENABLE_PROFILE
        ((kbnn_context *)ctx)->profile.xnor_conv_prep_ns += kbnn_now_ns() - t0;
#endif
        return;
    }
    if (input_f32 != NULL) {
        const float *restrict input = (const float *)KBNN_ASSUME_ALIGNED(input_f32, 64);
        const int8_t *restrict act_scale_i8 = p->act_scale_i8;
        const int8_t *restrict act_bias_i8 = p->act_bias_i8;
        for (int32_t y = 0; y < p->in_h; ++y) {
            for (int32_t x = 0; x < p->in_w; ++x) {
                uint64_t *restrict packed = scratch + (((size_t)y * (size_t)p->in_w + (size_t)x) * (size_t)p->channel_words);
                const size_t base = (((size_t)y * (size_t)p->in_w + (size_t)x) * (size_t)p->in_c);
                for (int32_t w = 0; w < p->channel_words; ++w) {
                    uint64_t bits = 0ull;
                    const int32_t base_c = w * 64;
                    int32_t limit = p->in_c - base_c;
                    if (limit > 64) {
                        limit = 64;
                    }
                    for (int32_t bit = 0; bit < limit; ++bit) {
                        const int32_t c = base_c + bit;
                        const float scale = (act_scale_i8 != NULL) ? (float)act_scale_i8[c] : 1.0f;
                        const float bias = (act_bias_i8 != NULL) ? (float)act_bias_i8[c] : 0.0f;
                        bits |= (uint64_t)((scale * input[base + (size_t)c] + bias) >= 0.0f) << bit;
                    }
                    packed[w] = bits;
                }
            }
        }
    } else {
        const int32_t *restrict input = (const int32_t *)KBNN_ASSUME_ALIGNED(input_i32, 64);
        const int8_t *restrict act_scale_i8 = p->act_scale_i8;
        const int8_t *restrict act_bias_i8 = p->act_bias_i8;
        for (int32_t y = 0; y < p->in_h; ++y) {
            for (int32_t x = 0; x < p->in_w; ++x) {
                uint64_t *restrict packed = scratch + (((size_t)y * (size_t)p->in_w + (size_t)x) * (size_t)p->channel_words);
                const size_t base = (((size_t)y * (size_t)p->in_w + (size_t)x) * (size_t)p->in_c);
                for (int32_t w = 0; w < p->channel_words; ++w) {
                    uint64_t bits = 0ull;
                    const int32_t base_c = w * 64;
                    int32_t limit = p->in_c - base_c;
                    if (limit > 64) {
                        limit = 64;
                    }
                    for (int32_t bit = 0; bit < limit; ++bit) {
                        const int32_t c = base_c + bit;
                        int32_t x_q = input[base + (size_t)c];
                        if (p->input_cast_fp16) {
                            x_q = kbnn_fx_cast_via_fp16(x_q);
                        }
                        const int32_t scale_q = (act_scale_i8 != NULL) ? (int32_t)act_scale_i8[c] : 1;
                        const int32_t bias_q = (act_bias_i8 != NULL) ? (int32_t)act_bias_i8[c] * KBNN_FX_ONE : 0;
                        bits |= (uint64_t)((((int64_t)scale_q * (int64_t)x_q) + (int64_t)bias_q) >= 0) << bit;
                    }
                    packed[w] = bits;
                }
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.xnor_conv_prep_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_prepare_binary_dense_input(const kbnn_context *ctx, const kbnn_xnor_dense_params *p) {
    const kbnn_tensor_desc *input_desc = &ctx->model->tensors[p->input_id];
    const int32_t *input_i32 = NULL;
    const float *input_f32 = NULL;
    const uint64_t *input_bits = NULL;
    uint64_t *restrict scratch = kbnn_tensor_bits(ctx, p->scratch_id);

    if (input_desc->kind == KBNN_TENSOR_EXTERNAL_F32) {
        input_f32 = kbnn_tensor_f32(ctx, p->input_id);
    } else if (input_desc->kind == KBNN_TENSOR_PACKED_BINARY) {
        input_bits = kbnn_tensor_bits(ctx, p->input_id);
    } else {
        input_i32 = kbnn_tensor_i32(ctx, p->input_id);
    }

#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    if (input_bits != NULL) {
        memcpy(
            scratch,
            input_bits,
            (size_t)p->outer_count * (size_t)p->channel_words * sizeof(uint64_t)
        );
#ifdef KBNN_ENABLE_PROFILE
        ((kbnn_context *)ctx)->profile.xnor_dense_prep_ns += kbnn_now_ns() - t0;
#endif
        return;
    }
    if (input_f32 != NULL) {
        const float *restrict input = (const float *)KBNN_ASSUME_ALIGNED(input_f32, 64);
        const int8_t *restrict act_scale_i8 = p->act_scale_i8;
        const int8_t *restrict act_bias_i8 = p->act_bias_i8;
        for (int32_t outer = 0; outer < p->outer_count; ++outer) {
            const size_t base = (size_t)outer * (size_t)p->in_dim;
            uint64_t *restrict packed = scratch + (size_t)outer * (size_t)p->channel_words;
            for (int32_t w = 0; w < p->channel_words; ++w) {
                uint64_t bits = 0ull;
                const int32_t base_c = w * 64;
                int32_t limit = p->in_dim - base_c;
                if (limit > 64) {
                    limit = 64;
                }
                for (int32_t bit = 0; bit < limit; ++bit) {
                    const int32_t c = base_c + bit;
                    const float scale = (act_scale_i8 != NULL) ? (float)act_scale_i8[c] : 1.0f;
                    const float bias = (act_bias_i8 != NULL) ? (float)act_bias_i8[c] : 0.0f;
                    bits |= (uint64_t)((scale * input[base + (size_t)c] + bias) >= 0.0f) << bit;
                }
                packed[w] = bits;
            }
        }
    } else {
        const int32_t *restrict input = (const int32_t *)KBNN_ASSUME_ALIGNED(input_i32, 64);
        const int8_t *restrict act_scale_i8 = p->act_scale_i8;
        const int8_t *restrict act_bias_i8 = p->act_bias_i8;
        for (int32_t outer = 0; outer < p->outer_count; ++outer) {
            const size_t base = (size_t)outer * (size_t)p->in_dim;
            uint64_t *restrict packed = scratch + (size_t)outer * (size_t)p->channel_words;
            for (int32_t w = 0; w < p->channel_words; ++w) {
                uint64_t bits = 0ull;
                const int32_t base_c = w * 64;
                int32_t limit = p->in_dim - base_c;
                if (limit > 64) {
                    limit = 64;
                }
                for (int32_t bit = 0; bit < limit; ++bit) {
                    const int32_t c = base_c + bit;
                    int32_t x_q = input[base + (size_t)c];
                    if (p->input_cast_fp16) {
                        x_q = kbnn_fx_cast_via_fp16(x_q);
                    }
                    const int32_t scale_q = (act_scale_i8 != NULL) ? (int32_t)act_scale_i8[c] : 1;
                    const int32_t bias_q = (act_bias_i8 != NULL) ? (int32_t)act_bias_i8[c] * KBNN_FX_ONE : 0;
                    bits |= (uint64_t)((((int64_t)scale_q * (int64_t)x_q) + (int64_t)bias_q) >= 0) << bit;
                }
                packed[w] = bits;
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.xnor_dense_prep_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_xnor_conv2d_3x3_s1_core(
    const kbnn_xnor_conv2d_params *p,
    int32_t *restrict output,
    const uint64_t *restrict packed,
    int32_t last_active
) {
    const int32_t kernel_area = 9;
    for (int32_t oh = 0; oh < p->out_h; ++oh) {
        const int32_t iy0 = oh - p->pad_top;
        const int32_t iy1 = iy0 + 1;
        const int32_t iy2 = iy0 + 2;
        for (int32_t ow = 0; ow < p->out_w; ++ow) {
            const int32_t ix0 = ow - p->pad_left;
            const int32_t ix1 = ix0 + 1;
            const int32_t ix2 = ix0 + 2;
            int32_t oc = 0;
            for (; oc + 7 < p->out_c; oc += 8) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                int32_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
#define KBNN_ACCUM_POINT8(IY, IX, KIDX) \
                do { \
                    if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                        const uint64_t *restrict in_words = packed + (((size_t)(IY) * (size_t)p->in_w + (size_t)(IX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w0 = p->weights_bits + ((((size_t)(oc + 0) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w1 = p->weights_bits + ((((size_t)(oc + 1) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w2 = p->weights_bits + ((((size_t)(oc + 2) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w3 = p->weights_bits + ((((size_t)(oc + 3) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w4 = p->weights_bits + ((((size_t)(oc + 4) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w5 = p->weights_bits + ((((size_t)(oc + 5) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w6 = p->weights_bits + ((((size_t)(oc + 6) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w7 = p->weights_bits + ((((size_t)(oc + 7) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        kbnn_xnor_accum8(in_words, w0, w1, w2, w3, w4, w5, w6, w7, p->channel_words, p->last_word_mask, last_active, &acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7); \
                    } \
                } while (0)
                KBNN_ACCUM_POINT8(iy0, ix0, 0);
                KBNN_ACCUM_POINT8(iy0, ix1, 1);
                KBNN_ACCUM_POINT8(iy0, ix2, 2);
                KBNN_ACCUM_POINT8(iy1, ix0, 3);
                KBNN_ACCUM_POINT8(iy1, ix1, 4);
                KBNN_ACCUM_POINT8(iy1, ix2, 5);
                KBNN_ACCUM_POINT8(iy2, ix0, 6);
                KBNN_ACCUM_POINT8(iy2, ix1, 7);
                KBNN_ACCUM_POINT8(iy2, ix2, 8);
#undef KBNN_ACCUM_POINT8
                {
                    const size_t out_base = ((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                    output[out_base + 0] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                    output[out_base + 1] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                    output[out_base + 2] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                    output[out_base + 3] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
                    output[out_base + 4] = kbnn_apply_xnor_affine_f32(acc4, p->alpha_q[oc + 4], p->bias_q[oc + 4]);
                    output[out_base + 5] = kbnn_apply_xnor_affine_f32(acc5, p->alpha_q[oc + 5], p->bias_q[oc + 5]);
                    output[out_base + 6] = kbnn_apply_xnor_affine_f32(acc6, p->alpha_q[oc + 6], p->bias_q[oc + 6]);
                    output[out_base + 7] = kbnn_apply_xnor_affine_f32(acc7, p->alpha_q[oc + 7], p->bias_q[oc + 7]);
                }
            }
            for (; oc + 3 < p->out_c; oc += 4) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#define KBNN_ACCUM_POINT4(IY, IX, KIDX) \
                do { \
                    if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                        const uint64_t *restrict in_words = packed + (((size_t)(IY) * (size_t)p->in_w + (size_t)(IX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w0 = p->weights_bits + ((((size_t)(oc + 0) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w1 = p->weights_bits + ((((size_t)(oc + 1) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w2 = p->weights_bits + ((((size_t)(oc + 2) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w3 = p->weights_bits + ((((size_t)(oc + 3) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        kbnn_xnor_accum4(in_words, w0, w1, w2, w3, p->channel_words, p->last_word_mask, last_active, &acc0, &acc1, &acc2, &acc3); \
                    } \
                } while (0)
                KBNN_ACCUM_POINT4(iy0, ix0, 0);
                KBNN_ACCUM_POINT4(iy0, ix1, 1);
                KBNN_ACCUM_POINT4(iy0, ix2, 2);
                KBNN_ACCUM_POINT4(iy1, ix0, 3);
                KBNN_ACCUM_POINT4(iy1, ix1, 4);
                KBNN_ACCUM_POINT4(iy1, ix2, 5);
                KBNN_ACCUM_POINT4(iy2, ix0, 6);
                KBNN_ACCUM_POINT4(iy2, ix1, 7);
                KBNN_ACCUM_POINT4(iy2, ix2, 8);
#undef KBNN_ACCUM_POINT4
                {
                    const size_t out_base = ((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                    output[out_base + 0] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                    output[out_base + 1] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                    output[out_base + 2] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                    output[out_base + 3] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
                }
            }
            for (; oc < p->out_c; ++oc) {
                int32_t acc = 0;
#define KBNN_ACCUM_POINT1(IY, IX, KIDX) \
                do { \
                    if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                        const uint64_t *restrict in_words = packed + (((size_t)(IY) * (size_t)p->in_w + (size_t)(IX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict wrow = p->weights_bits + ((((size_t)oc * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        acc += kbnn_xnor_dot_u64(in_words, wrow, p->channel_words, p->last_word_mask); \
                    } \
                } while (0)
                KBNN_ACCUM_POINT1(iy0, ix0, 0);
                KBNN_ACCUM_POINT1(iy0, ix1, 1);
                KBNN_ACCUM_POINT1(iy0, ix2, 2);
                KBNN_ACCUM_POINT1(iy1, ix0, 3);
                KBNN_ACCUM_POINT1(iy1, ix1, 4);
                KBNN_ACCUM_POINT1(iy1, ix2, 5);
                KBNN_ACCUM_POINT1(iy2, ix0, 6);
                KBNN_ACCUM_POINT1(iy2, ix1, 7);
                KBNN_ACCUM_POINT1(iy2, ix2, 8);
#undef KBNN_ACCUM_POINT1
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)oc] =
                    kbnn_apply_xnor_affine_f32(acc, p->alpha_q[oc], p->bias_q[oc]);
            }
        }
    }
}

static void kbnn_exec_xnor_conv2d_3x3_s2_core(
    const kbnn_xnor_conv2d_params *p,
    int32_t *restrict output,
    const uint64_t *restrict packed,
    int32_t last_active
) {
    const int32_t kernel_area = 9;
    if (p->channel_words == 1 && p->pad_top == 1 && p->pad_left == 1) {
        const uint64_t mask = p->last_word_mask;
        const size_t in_w = (size_t)p->in_w;
        const size_t out_w = (size_t)p->out_w;
        for (int32_t oh = 0; oh < p->out_h; ++oh) {
            const int32_t base_y = oh * 2 - 1;
            for (int32_t ow = 0; ow < p->out_w; ++ow) {
                const int32_t base_x = ow * 2 - 1;
                const int boundary = (oh == 0) || (ow == 0);
                int32_t oc = 0;
                for (; oc + 7 < p->out_c; oc += 8) {
                    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                    int32_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
                    const uint64_t *restrict w0 = p->weights_bits + (size_t)(oc + 0) * (size_t)kernel_area;
                    const uint64_t *restrict w1 = p->weights_bits + (size_t)(oc + 1) * (size_t)kernel_area;
                    const uint64_t *restrict w2 = p->weights_bits + (size_t)(oc + 2) * (size_t)kernel_area;
                    const uint64_t *restrict w3 = p->weights_bits + (size_t)(oc + 3) * (size_t)kernel_area;
                    const uint64_t *restrict w4 = p->weights_bits + (size_t)(oc + 4) * (size_t)kernel_area;
                    const uint64_t *restrict w5 = p->weights_bits + (size_t)(oc + 5) * (size_t)kernel_area;
                    const uint64_t *restrict w6 = p->weights_bits + (size_t)(oc + 6) * (size_t)kernel_area;
                    const uint64_t *restrict w7 = p->weights_bits + (size_t)(oc + 7) * (size_t)kernel_area;
                    if (!boundary) {
                        const uint64_t x0 = packed[((size_t)base_y * in_w) + (size_t)base_x];
                        const uint64_t x1 = packed[((size_t)base_y * in_w) + (size_t)(base_x + 1)];
                        const uint64_t x2 = packed[((size_t)base_y * in_w) + (size_t)(base_x + 2)];
                        const uint64_t x3 = packed[(((size_t)base_y + 1) * in_w) + (size_t)base_x];
                        const uint64_t x4 = packed[(((size_t)base_y + 1) * in_w) + (size_t)(base_x + 1)];
                        const uint64_t x5 = packed[(((size_t)base_y + 1) * in_w) + (size_t)(base_x + 2)];
                        const uint64_t x6 = packed[(((size_t)base_y + 2) * in_w) + (size_t)base_x];
                        const uint64_t x7 = packed[(((size_t)base_y + 2) * in_w) + (size_t)(base_x + 1)];
                        const uint64_t x8 = packed[(((size_t)base_y + 2) * in_w) + (size_t)(base_x + 2)];
#define KBNN_ACCUM8_WORD_X(XV, KIDX) \
                        do { \
                            acc0 += kbnn_xnor_dot_masked_word((XV), w0[(KIDX)], mask, last_active); \
                            acc1 += kbnn_xnor_dot_masked_word((XV), w1[(KIDX)], mask, last_active); \
                            acc2 += kbnn_xnor_dot_masked_word((XV), w2[(KIDX)], mask, last_active); \
                            acc3 += kbnn_xnor_dot_masked_word((XV), w3[(KIDX)], mask, last_active); \
                            acc4 += kbnn_xnor_dot_masked_word((XV), w4[(KIDX)], mask, last_active); \
                            acc5 += kbnn_xnor_dot_masked_word((XV), w5[(KIDX)], mask, last_active); \
                            acc6 += kbnn_xnor_dot_masked_word((XV), w6[(KIDX)], mask, last_active); \
                            acc7 += kbnn_xnor_dot_masked_word((XV), w7[(KIDX)], mask, last_active); \
                        } while (0)
                        KBNN_ACCUM8_WORD_X(x0, 0);
                        KBNN_ACCUM8_WORD_X(x1, 1);
                        KBNN_ACCUM8_WORD_X(x2, 2);
                        KBNN_ACCUM8_WORD_X(x3, 3);
                        KBNN_ACCUM8_WORD_X(x4, 4);
                        KBNN_ACCUM8_WORD_X(x5, 5);
                        KBNN_ACCUM8_WORD_X(x6, 6);
                        KBNN_ACCUM8_WORD_X(x7, 7);
                        KBNN_ACCUM8_WORD_X(x8, 8);
#undef KBNN_ACCUM8_WORD_X
                    } else {
#define KBNN_ACCUM8_WORD_BOUND(IY, IX, KIDX) \
                        do { \
                            if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                                const uint64_t x = packed[((size_t)(IY) * in_w) + (size_t)(IX)]; \
                                acc0 += kbnn_xnor_dot_masked_word(x, w0[(KIDX)], mask, last_active); \
                                acc1 += kbnn_xnor_dot_masked_word(x, w1[(KIDX)], mask, last_active); \
                                acc2 += kbnn_xnor_dot_masked_word(x, w2[(KIDX)], mask, last_active); \
                                acc3 += kbnn_xnor_dot_masked_word(x, w3[(KIDX)], mask, last_active); \
                                acc4 += kbnn_xnor_dot_masked_word(x, w4[(KIDX)], mask, last_active); \
                                acc5 += kbnn_xnor_dot_masked_word(x, w5[(KIDX)], mask, last_active); \
                                acc6 += kbnn_xnor_dot_masked_word(x, w6[(KIDX)], mask, last_active); \
                                acc7 += kbnn_xnor_dot_masked_word(x, w7[(KIDX)], mask, last_active); \
                            } \
                        } while (0)
                        KBNN_ACCUM8_WORD_BOUND(base_y + 0, base_x + 0, 0);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 0, base_x + 1, 1);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 0, base_x + 2, 2);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 1, base_x + 0, 3);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 1, base_x + 1, 4);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 1, base_x + 2, 5);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 2, base_x + 0, 6);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 2, base_x + 1, 7);
                        KBNN_ACCUM8_WORD_BOUND(base_y + 2, base_x + 2, 8);
#undef KBNN_ACCUM8_WORD_BOUND
                    }
                    {
                        const size_t out_base = ((size_t)oh * out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                        output[out_base + 0] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                        output[out_base + 1] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                        output[out_base + 2] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                        output[out_base + 3] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
                        output[out_base + 4] = kbnn_apply_xnor_affine_f32(acc4, p->alpha_q[oc + 4], p->bias_q[oc + 4]);
                        output[out_base + 5] = kbnn_apply_xnor_affine_f32(acc5, p->alpha_q[oc + 5], p->bias_q[oc + 5]);
                        output[out_base + 6] = kbnn_apply_xnor_affine_f32(acc6, p->alpha_q[oc + 6], p->bias_q[oc + 6]);
                        output[out_base + 7] = kbnn_apply_xnor_affine_f32(acc7, p->alpha_q[oc + 7], p->bias_q[oc + 7]);
                    }
                }
                for (; oc + 3 < p->out_c; oc += 4) {
                    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                    const uint64_t *restrict w0 = p->weights_bits + (size_t)(oc + 0) * (size_t)kernel_area;
                    const uint64_t *restrict w1 = p->weights_bits + (size_t)(oc + 1) * (size_t)kernel_area;
                    const uint64_t *restrict w2 = p->weights_bits + (size_t)(oc + 2) * (size_t)kernel_area;
                    const uint64_t *restrict w3 = p->weights_bits + (size_t)(oc + 3) * (size_t)kernel_area;
                    if (!boundary) {
                        const uint64_t x0 = packed[((size_t)base_y * in_w) + (size_t)base_x];
                        const uint64_t x1 = packed[((size_t)base_y * in_w) + (size_t)(base_x + 1)];
                        const uint64_t x2 = packed[((size_t)base_y * in_w) + (size_t)(base_x + 2)];
                        const uint64_t x3 = packed[(((size_t)base_y + 1) * in_w) + (size_t)base_x];
                        const uint64_t x4 = packed[(((size_t)base_y + 1) * in_w) + (size_t)(base_x + 1)];
                        const uint64_t x5 = packed[(((size_t)base_y + 1) * in_w) + (size_t)(base_x + 2)];
                        const uint64_t x6 = packed[(((size_t)base_y + 2) * in_w) + (size_t)base_x];
                        const uint64_t x7 = packed[(((size_t)base_y + 2) * in_w) + (size_t)(base_x + 1)];
                        const uint64_t x8 = packed[(((size_t)base_y + 2) * in_w) + (size_t)(base_x + 2)];
#define KBNN_ACCUM4_WORD_X(XV, KIDX) \
                        do { \
                            acc0 += kbnn_xnor_dot_masked_word((XV), w0[(KIDX)], mask, last_active); \
                            acc1 += kbnn_xnor_dot_masked_word((XV), w1[(KIDX)], mask, last_active); \
                            acc2 += kbnn_xnor_dot_masked_word((XV), w2[(KIDX)], mask, last_active); \
                            acc3 += kbnn_xnor_dot_masked_word((XV), w3[(KIDX)], mask, last_active); \
                        } while (0)
                        KBNN_ACCUM4_WORD_X(x0, 0);
                        KBNN_ACCUM4_WORD_X(x1, 1);
                        KBNN_ACCUM4_WORD_X(x2, 2);
                        KBNN_ACCUM4_WORD_X(x3, 3);
                        KBNN_ACCUM4_WORD_X(x4, 4);
                        KBNN_ACCUM4_WORD_X(x5, 5);
                        KBNN_ACCUM4_WORD_X(x6, 6);
                        KBNN_ACCUM4_WORD_X(x7, 7);
                        KBNN_ACCUM4_WORD_X(x8, 8);
#undef KBNN_ACCUM4_WORD_X
                    } else {
#define KBNN_ACCUM4_WORD_BOUND(IY, IX, KIDX) \
                        do { \
                            if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                                const uint64_t x = packed[((size_t)(IY) * in_w) + (size_t)(IX)]; \
                                acc0 += kbnn_xnor_dot_masked_word(x, w0[(KIDX)], mask, last_active); \
                                acc1 += kbnn_xnor_dot_masked_word(x, w1[(KIDX)], mask, last_active); \
                                acc2 += kbnn_xnor_dot_masked_word(x, w2[(KIDX)], mask, last_active); \
                                acc3 += kbnn_xnor_dot_masked_word(x, w3[(KIDX)], mask, last_active); \
                            } \
                        } while (0)
                        KBNN_ACCUM4_WORD_BOUND(base_y + 0, base_x + 0, 0);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 0, base_x + 1, 1);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 0, base_x + 2, 2);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 1, base_x + 0, 3);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 1, base_x + 1, 4);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 1, base_x + 2, 5);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 2, base_x + 0, 6);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 2, base_x + 1, 7);
                        KBNN_ACCUM4_WORD_BOUND(base_y + 2, base_x + 2, 8);
#undef KBNN_ACCUM4_WORD_BOUND
                    }
                    {
                        const size_t out_base = ((size_t)oh * out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                        output[out_base + 0] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                        output[out_base + 1] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                        output[out_base + 2] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                        output[out_base + 3] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
                    }
                }
                for (; oc < p->out_c; ++oc) {
                    int32_t acc = 0;
                    const uint64_t *restrict wrow = p->weights_bits + (size_t)oc * (size_t)kernel_area;
                    if (!boundary) {
                        acc += kbnn_xnor_dot_masked_word(packed[((size_t)base_y * in_w) + (size_t)base_x], wrow[0], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[((size_t)base_y * in_w) + (size_t)(base_x + 1)], wrow[1], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[((size_t)base_y * in_w) + (size_t)(base_x + 2)], wrow[2], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[(((size_t)base_y + 1) * in_w) + (size_t)base_x], wrow[3], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[(((size_t)base_y + 1) * in_w) + (size_t)(base_x + 1)], wrow[4], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[(((size_t)base_y + 1) * in_w) + (size_t)(base_x + 2)], wrow[5], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[(((size_t)base_y + 2) * in_w) + (size_t)base_x], wrow[6], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[(((size_t)base_y + 2) * in_w) + (size_t)(base_x + 1)], wrow[7], mask, last_active);
                        acc += kbnn_xnor_dot_masked_word(packed[(((size_t)base_y + 2) * in_w) + (size_t)(base_x + 2)], wrow[8], mask, last_active);
                    } else {
#define KBNN_ACCUM1_WORD_BOUND(IY, IX, KIDX) \
                        do { \
                            if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                                const uint64_t x = packed[((size_t)(IY) * in_w) + (size_t)(IX)]; \
                                acc += kbnn_xnor_dot_masked_word(x, wrow[(KIDX)], mask, last_active); \
                            } \
                        } while (0)
                        KBNN_ACCUM1_WORD_BOUND(base_y + 0, base_x + 0, 0);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 0, base_x + 1, 1);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 0, base_x + 2, 2);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 1, base_x + 0, 3);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 1, base_x + 1, 4);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 1, base_x + 2, 5);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 2, base_x + 0, 6);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 2, base_x + 1, 7);
                        KBNN_ACCUM1_WORD_BOUND(base_y + 2, base_x + 2, 8);
#undef KBNN_ACCUM1_WORD_BOUND
                    }
                    output[(((size_t)oh * out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)oc] =
                        kbnn_apply_xnor_affine_f32(acc, p->alpha_q[oc], p->bias_q[oc]);
                }
            }
        }
        return;
    }
    for (int32_t oh = 0; oh < p->out_h; ++oh) {
        const int32_t base_y = oh * 2 - p->pad_top;
        const int32_t iy0 = base_y;
        const int32_t iy1 = base_y + 1;
        const int32_t iy2 = base_y + 2;
        for (int32_t ow = 0; ow < p->out_w; ++ow) {
            const int32_t base_x = ow * 2 - p->pad_left;
            const int32_t ix0 = base_x;
            const int32_t ix1 = base_x + 1;
            const int32_t ix2 = base_x + 2;
            int32_t oc = 0;
            for (; oc + 7 < p->out_c; oc += 8) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                int32_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
#define KBNN_ACCUM_POINT8S2(IY, IX, KIDX) \
                do { \
                    if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                        const uint64_t *restrict in_words = packed + (((size_t)(IY) * (size_t)p->in_w + (size_t)(IX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w0 = p->weights_bits + ((((size_t)(oc + 0) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w1 = p->weights_bits + ((((size_t)(oc + 1) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w2 = p->weights_bits + ((((size_t)(oc + 2) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w3 = p->weights_bits + ((((size_t)(oc + 3) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w4 = p->weights_bits + ((((size_t)(oc + 4) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w5 = p->weights_bits + ((((size_t)(oc + 5) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w6 = p->weights_bits + ((((size_t)(oc + 6) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w7 = p->weights_bits + ((((size_t)(oc + 7) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        kbnn_xnor_accum8(in_words, w0, w1, w2, w3, w4, w5, w6, w7, p->channel_words, p->last_word_mask, last_active, &acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7); \
                    } \
                } while (0)
                KBNN_ACCUM_POINT8S2(iy0, ix0, 0);
                KBNN_ACCUM_POINT8S2(iy0, ix1, 1);
                KBNN_ACCUM_POINT8S2(iy0, ix2, 2);
                KBNN_ACCUM_POINT8S2(iy1, ix0, 3);
                KBNN_ACCUM_POINT8S2(iy1, ix1, 4);
                KBNN_ACCUM_POINT8S2(iy1, ix2, 5);
                KBNN_ACCUM_POINT8S2(iy2, ix0, 6);
                KBNN_ACCUM_POINT8S2(iy2, ix1, 7);
                KBNN_ACCUM_POINT8S2(iy2, ix2, 8);
#undef KBNN_ACCUM_POINT8S2
                {
                    const size_t out_base = ((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                    output[out_base + 0] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                    output[out_base + 1] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                    output[out_base + 2] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                    output[out_base + 3] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
                    output[out_base + 4] = kbnn_apply_xnor_affine_f32(acc4, p->alpha_q[oc + 4], p->bias_q[oc + 4]);
                    output[out_base + 5] = kbnn_apply_xnor_affine_f32(acc5, p->alpha_q[oc + 5], p->bias_q[oc + 5]);
                    output[out_base + 6] = kbnn_apply_xnor_affine_f32(acc6, p->alpha_q[oc + 6], p->bias_q[oc + 6]);
                    output[out_base + 7] = kbnn_apply_xnor_affine_f32(acc7, p->alpha_q[oc + 7], p->bias_q[oc + 7]);
                }
            }
            for (; oc + 3 < p->out_c; oc += 4) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#define KBNN_ACCUM_POINT4S2(IY, IX, KIDX) \
                do { \
                    if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                        const uint64_t *restrict in_words = packed + (((size_t)(IY) * (size_t)p->in_w + (size_t)(IX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w0 = p->weights_bits + ((((size_t)(oc + 0) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w1 = p->weights_bits + ((((size_t)(oc + 1) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w2 = p->weights_bits + ((((size_t)(oc + 2) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict w3 = p->weights_bits + ((((size_t)(oc + 3) * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        kbnn_xnor_accum4(in_words, w0, w1, w2, w3, p->channel_words, p->last_word_mask, last_active, &acc0, &acc1, &acc2, &acc3); \
                    } \
                } while (0)
                KBNN_ACCUM_POINT4S2(iy0, ix0, 0);
                KBNN_ACCUM_POINT4S2(iy0, ix1, 1);
                KBNN_ACCUM_POINT4S2(iy0, ix2, 2);
                KBNN_ACCUM_POINT4S2(iy1, ix0, 3);
                KBNN_ACCUM_POINT4S2(iy1, ix1, 4);
                KBNN_ACCUM_POINT4S2(iy1, ix2, 5);
                KBNN_ACCUM_POINT4S2(iy2, ix0, 6);
                KBNN_ACCUM_POINT4S2(iy2, ix1, 7);
                KBNN_ACCUM_POINT4S2(iy2, ix2, 8);
#undef KBNN_ACCUM_POINT4S2
                {
                    const size_t out_base = ((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c + (size_t)oc;
                    output[out_base + 0] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                    output[out_base + 1] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                    output[out_base + 2] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                    output[out_base + 3] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
                }
            }
            for (; oc < p->out_c; ++oc) {
                int32_t acc = 0;
#define KBNN_ACCUM_POINT1S2(IY, IX, KIDX) \
                do { \
                    if ((unsigned)(IY) < (unsigned)p->in_h && (unsigned)(IX) < (unsigned)p->in_w) { \
                        const uint64_t *restrict in_words = packed + (((size_t)(IY) * (size_t)p->in_w + (size_t)(IX)) * (size_t)p->channel_words); \
                        const uint64_t *restrict wrow = p->weights_bits + ((((size_t)oc * (size_t)kernel_area) + (size_t)(KIDX)) * (size_t)p->channel_words); \
                        acc += kbnn_xnor_dot_u64(in_words, wrow, p->channel_words, p->last_word_mask); \
                    } \
                } while (0)
                KBNN_ACCUM_POINT1S2(iy0, ix0, 0);
                KBNN_ACCUM_POINT1S2(iy0, ix1, 1);
                KBNN_ACCUM_POINT1S2(iy0, ix2, 2);
                KBNN_ACCUM_POINT1S2(iy1, ix0, 3);
                KBNN_ACCUM_POINT1S2(iy1, ix1, 4);
                KBNN_ACCUM_POINT1S2(iy1, ix2, 5);
                KBNN_ACCUM_POINT1S2(iy2, ix0, 6);
                KBNN_ACCUM_POINT1S2(iy2, ix1, 7);
                KBNN_ACCUM_POINT1S2(iy2, ix2, 8);
#undef KBNN_ACCUM_POINT1S2
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)oc] =
                    kbnn_apply_xnor_affine_f32(acc, p->alpha_q[oc], p->bias_q[oc]);
            }
        }
    }
}

static void kbnn_exec_xnor_conv2d(const kbnn_context *ctx, const kbnn_xnor_conv2d_params *p) {
    int32_t *restrict output = kbnn_tensor_i32(ctx, p->output_id);
    uint64_t *restrict packed = kbnn_tensor_bits(ctx, p->scratch_id);
    const int32_t kernel_area = p->kernel_h * p->kernel_w;
    const int32_t last_active = (int32_t)__builtin_popcountll(p->last_word_mask);
    kbnn_prepare_binary_input(ctx, p);
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
#ifndef KBNN_DISABLE_SPECIALIZED_3X3
    if (p->kernel_h == 3 && p->kernel_w == 3 && p->stride_h == 1 && p->stride_w == 1) {
        kbnn_exec_xnor_conv2d_3x3_s1_core(p, output, packed, last_active);
#ifdef KBNN_ENABLE_PROFILE
        ((kbnn_context *)ctx)->profile.xnor_conv_dot_ns += kbnn_now_ns() - t0;
#endif
        return;
    }
    if (p->kernel_h == 3 && p->kernel_w == 3 && p->stride_h == 2 && p->stride_w == 2) {
        kbnn_exec_xnor_conv2d_3x3_s2_core(p, output, packed, last_active);
#ifdef KBNN_ENABLE_PROFILE
        ((kbnn_context *)ctx)->profile.xnor_conv_dot_ns += kbnn_now_ns() - t0;
#endif
        return;
    }
#endif
    for (int32_t oh = 0; oh < p->out_h; ++oh) {
        for (int32_t ow = 0; ow < p->out_w; ++ow) {
            int32_t oc = 0;
            for (; oc + 7 < p->out_c; oc += 8) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                int32_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
                for (int32_t kh = 0; kh < p->kernel_h; ++kh) {
                    const int32_t iy = oh * p->stride_h + kh - p->pad_top;
                    if (iy < 0 || iy >= p->in_h) {
                        continue;
                    }
                    for (int32_t kw = 0; kw < p->kernel_w; ++kw) {
                        const int32_t ix = ow * p->stride_w + kw - p->pad_left;
                        if (ix < 0 || ix >= p->in_w) {
                            continue;
                        }
                        const uint64_t *restrict in_words = packed + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * (size_t)p->channel_words);
                        const int32_t kidx = kh * p->kernel_w + kw;
                        const uint64_t *restrict w0 = p->weights_bits + ((((size_t)(oc + 0) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w1 = p->weights_bits + ((((size_t)(oc + 1) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w2 = p->weights_bits + ((((size_t)(oc + 2) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w3 = p->weights_bits + ((((size_t)(oc + 3) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w4 = p->weights_bits + ((((size_t)(oc + 4) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w5 = p->weights_bits + ((((size_t)(oc + 5) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w6 = p->weights_bits + ((((size_t)(oc + 6) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w7 = p->weights_bits + ((((size_t)(oc + 7) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        kbnn_xnor_accum8(
                            in_words, w0, w1, w2, w3, w4, w5, w6, w7,
                            p->channel_words, p->last_word_mask, last_active,
                            &acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7
                        );
                    }
                }
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 0)] =
                    kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 1)] =
                    kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 2)] =
                    kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 3)] =
                    kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 4)] =
                    kbnn_apply_xnor_affine_f32(acc4, p->alpha_q[oc + 4], p->bias_q[oc + 4]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 5)] =
                    kbnn_apply_xnor_affine_f32(acc5, p->alpha_q[oc + 5], p->bias_q[oc + 5]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 6)] =
                    kbnn_apply_xnor_affine_f32(acc6, p->alpha_q[oc + 6], p->bias_q[oc + 6]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 7)] =
                    kbnn_apply_xnor_affine_f32(acc7, p->alpha_q[oc + 7], p->bias_q[oc + 7]);
            }
            for (; oc + 3 < p->out_c; oc += 4) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                for (int32_t kh = 0; kh < p->kernel_h; ++kh) {
                    const int32_t iy = oh * p->stride_h + kh - p->pad_top;
                    if (iy < 0 || iy >= p->in_h) {
                        continue;
                    }
                    for (int32_t kw = 0; kw < p->kernel_w; ++kw) {
                        const int32_t ix = ow * p->stride_w + kw - p->pad_left;
                        if (ix < 0 || ix >= p->in_w) {
                            continue;
                        }
                        const uint64_t *restrict in_words = packed + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * (size_t)p->channel_words);
                        const int32_t kidx = kh * p->kernel_w + kw;
                        const uint64_t *restrict w0 = p->weights_bits + ((((size_t)(oc + 0) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w1 = p->weights_bits + ((((size_t)(oc + 1) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w2 = p->weights_bits + ((((size_t)(oc + 2) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        const uint64_t *restrict w3 = p->weights_bits + ((((size_t)(oc + 3) * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        kbnn_xnor_accum4(
                            in_words, w0, w1, w2, w3,
                            p->channel_words, p->last_word_mask, last_active,
                            &acc0, &acc1, &acc2, &acc3
                        );
                    }
                }
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 0)] =
                    kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 1)] =
                    kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 2)] =
                    kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)(oc + 3)] =
                    kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
            }
            for (; oc < p->out_c; ++oc) {
                int32_t acc = 0;
                for (int32_t kh = 0; kh < p->kernel_h; ++kh) {
                    const int32_t iy = oh * p->stride_h + kh - p->pad_top;
                    if (iy < 0 || iy >= p->in_h) {
                        continue;
                    }
                    for (int32_t kw = 0; kw < p->kernel_w; ++kw) {
                        const int32_t ix = ow * p->stride_w + kw - p->pad_left;
                        if (ix < 0 || ix >= p->in_w) {
                            continue;
                        }
                        const uint64_t *in_words = packed + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * (size_t)p->channel_words);
                        const int32_t kidx = kh * p->kernel_w + kw;
                        const uint64_t *wrow = p->weights_bits + ((((size_t)oc * (size_t)kernel_area) + (size_t)kidx) * (size_t)p->channel_words);
                        acc += kbnn_xnor_dot_u64(in_words, wrow, p->channel_words, p->last_word_mask);
                    }
                }
                output[(((size_t)oh * (size_t)p->out_w + (size_t)ow) * (size_t)p->out_c) + (size_t)oc] =
                    kbnn_apply_xnor_affine_f32(acc, p->alpha_q[oc], p->bias_q[oc]);
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.xnor_conv_dot_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_xnor_dense(const kbnn_context *ctx, const kbnn_xnor_dense_params *p) {
    int32_t *restrict output = kbnn_tensor_i32(ctx, p->output_id);
    uint64_t *restrict packed = kbnn_tensor_bits(ctx, p->scratch_id);
    const int32_t last_active = (int32_t)__builtin_popcountll(p->last_word_mask);

    kbnn_prepare_binary_dense_input(ctx, p);
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    for (int32_t outer = 0; outer < p->outer_count; ++outer) {
        const uint64_t *restrict in_words = packed + (size_t)outer * (size_t)p->channel_words;
        int32_t oc = 0;
        for (; oc + 7 < p->out_dim; oc += 8) {
            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            int32_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
            kbnn_xnor_accum8(
                in_words,
                p->weights_bits + (size_t)(oc + 0) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 1) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 2) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 3) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 4) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 5) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 6) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 7) * (size_t)p->channel_words,
                p->channel_words,
                p->last_word_mask,
                last_active,
                &acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7
            );
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 0)] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 1)] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 2)] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 3)] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 4)] = kbnn_apply_xnor_affine_f32(acc4, p->alpha_q[oc + 4], p->bias_q[oc + 4]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 5)] = kbnn_apply_xnor_affine_f32(acc5, p->alpha_q[oc + 5], p->bias_q[oc + 5]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 6)] = kbnn_apply_xnor_affine_f32(acc6, p->alpha_q[oc + 6], p->bias_q[oc + 6]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 7)] = kbnn_apply_xnor_affine_f32(acc7, p->alpha_q[oc + 7], p->bias_q[oc + 7]);
        }
        for (; oc + 3 < p->out_dim; oc += 4) {
            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            kbnn_xnor_accum4(
                in_words,
                p->weights_bits + (size_t)(oc + 0) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 1) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 2) * (size_t)p->channel_words,
                p->weights_bits + (size_t)(oc + 3) * (size_t)p->channel_words,
                p->channel_words,
                p->last_word_mask,
                last_active,
                &acc0, &acc1, &acc2, &acc3
            );
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 0)] = kbnn_apply_xnor_affine_f32(acc0, p->alpha_q[oc + 0], p->bias_q[oc + 0]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 1)] = kbnn_apply_xnor_affine_f32(acc1, p->alpha_q[oc + 1], p->bias_q[oc + 1]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 2)] = kbnn_apply_xnor_affine_f32(acc2, p->alpha_q[oc + 2], p->bias_q[oc + 2]);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)(oc + 3)] = kbnn_apply_xnor_affine_f32(acc3, p->alpha_q[oc + 3], p->bias_q[oc + 3]);
        }
        for (; oc < p->out_dim; ++oc) {
            const uint64_t *wrow = p->weights_bits + (size_t)oc * (size_t)p->channel_words;
            const int32_t acc = kbnn_xnor_dot_u64(in_words, wrow, p->channel_words, p->last_word_mask);
            output[(size_t)outer * (size_t)p->out_dim + (size_t)oc] = kbnn_apply_xnor_affine_f32(acc, p->alpha_q[oc], p->bias_q[oc]);
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.xnor_dense_dot_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_space_to_depth(const kbnn_context *ctx, const kbnn_space_to_depth_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const int32_t *input = kbnn_tensor_i32(ctx, p->input_id);
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    const int32_t out_h = p->in_h / p->block_size;
    const int32_t out_w = p->in_w / p->block_size;
    const int32_t out_c = p->channels * p->block_size * p->block_size;

    for (int32_t oy = 0; oy < out_h; ++oy) {
        for (int32_t ox = 0; ox < out_w; ++ox) {
            for (int32_t by = 0; by < p->block_size; ++by) {
                for (int32_t bx = 0; bx < p->block_size; ++bx) {
                    const int32_t iy = oy * p->block_size + by;
                    const int32_t ix = ox * p->block_size + bx;
                    const int32_t out_block = (by * p->block_size + bx) * p->channels;
                    const int32_t *src = input + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * (size_t)p->channels);
                    int32_t *dst = output + (((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c) + (size_t)out_block;
                    memcpy(dst, src, (size_t)p->channels * sizeof(int32_t));
                }
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.space_to_depth_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_concat(const kbnn_context *ctx, const kbnn_concat_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    size_t outer = 1;
    size_t inner = 1;
    int32_t axis = p->axis;
    if (axis < 0) {
        axis += p->rank;
    }
    for (int32_t i = 0; i < axis; ++i) {
        outer *= (size_t)p->output_shape[i];
    }
    for (int32_t i = axis + 1; i < p->rank; ++i) {
        inner *= (size_t)p->output_shape[i];
    }
    {
        size_t out_axis = (size_t)p->output_shape[axis];
        for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
            size_t offset = outer_idx * out_axis * inner;
            size_t cursor = 0;
            for (int32_t input_idx = 0; input_idx < p->input_count; ++input_idx) {
                const kbnn_tensor_desc *src_desc = &ctx->model->tensors[p->input_ids[input_idx]];
                const int32_t *src = kbnn_tensor_i32(ctx, p->input_ids[input_idx]);
                size_t src_axis = (size_t)src_desc->shape[axis];
                size_t copy_count = src_axis * inner;
                memcpy(output + offset + cursor, src + outer_idx * copy_count, copy_count * sizeof(int32_t));
                cursor += copy_count;
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.concat_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_add(const kbnn_context *ctx, const kbnn_add_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    const size_t count = kbnn_numel_from_shape(p->shape, p->rank);
    memset(output, 0, count * sizeof(int32_t));
    for (int32_t input_idx = 0; input_idx < p->input_count; ++input_idx) {
        const int32_t *src = kbnn_tensor_i32(ctx, p->input_ids[input_idx]);
        for (size_t i = 0; i < count; ++i) {
            output[i] += src[i];
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.add_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_upsample2d(const kbnn_context *ctx, const kbnn_upsample2d_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const int32_t *input = kbnn_tensor_i32(ctx, p->input_id);
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    const int32_t out_h = p->in_h * p->scale_h;
    const int32_t out_w = p->in_w * p->scale_w;

    for (int32_t oy = 0; oy < out_h; ++oy) {
        const int32_t iy = oy / p->scale_h;
        for (int32_t ox = 0; ox < out_w; ++ox) {
            const int32_t ix = ox / p->scale_w;
            const int32_t *src = input + (((size_t)iy * (size_t)p->in_w + (size_t)ix) * (size_t)p->channels);
            int32_t *dst = output + (((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)p->channels);
            memcpy(dst, src, (size_t)p->channels * sizeof(int32_t));
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.upsample2d_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_binary_affine(const kbnn_context *ctx, const kbnn_binary_affine_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const int32_t *input = kbnn_tensor_i32(ctx, p->input_id);
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    const size_t outer = kbnn_numel_from_shape(p->shape, p->rank) / (size_t)p->shape[p->rank - 1];
    const int32_t channels = p->shape[p->rank - 1];

    for (size_t i = 0; i < outer; ++i) {
        const int32_t *src = input + i * (size_t)channels;
        int32_t *dst = output + i * (size_t)channels;
        for (int32_t c = 0; c < channels; ++c) {
            dst[c] = kbnn_apply_pre_linear(src[c], p->mul_q[c], p->add_q[c]);
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.binary_affine_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_quant_act(const kbnn_context *ctx, const kbnn_quant_act_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const int32_t *input = kbnn_tensor_i32(ctx, p->input_id);
    int32_t *output = kbnn_tensor_i32(ctx, p->output_id);
    const size_t count = kbnn_numel_from_shape(p->shape, p->rank);

    for (size_t i = 0; i < count; ++i) {
        if (p->signed_mode) {
            output[i] = kbnn_quantize_signed_unit_i32(input[i], p->levels);
        } else {
            output[i] = kbnn_quantize_unit_i32(input[i], p->levels);
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.quant_act_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_qdense(const kbnn_context *ctx, const kbnn_qdense_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const int32_t *restrict input =
        (const int32_t *)KBNN_ASSUME_ALIGNED(kbnn_tensor_i32(ctx, p->input_id), 64);
    int32_t *restrict output =
        (int32_t *)KBNN_ASSUME_ALIGNED(kbnn_tensor_i32(ctx, p->output_id), 64);
    const int16_t *restrict weights =
        (const int16_t *)KBNN_ASSUME_ALIGNED(p->weights, 64);
    const int32_t *restrict bias_q = p->bias_q;

    for (int32_t outer = 0; outer < p->outer_count; ++outer) {
        const int32_t *restrict src = input + (size_t)outer * (size_t)p->in_dim;
        int32_t *dst = output + (size_t)outer * (size_t)p->out_dim;
        int32_t o = 0;
        for (; o + 7 < p->out_dim; o += 8) {
            int64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            int64_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
            const int16_t *restrict w0 = weights + (size_t)(o + 0) * (size_t)p->in_dim;
            const int16_t *restrict w1 = weights + (size_t)(o + 1) * (size_t)p->in_dim;
            const int16_t *restrict w2 = weights + (size_t)(o + 2) * (size_t)p->in_dim;
            const int16_t *restrict w3 = weights + (size_t)(o + 3) * (size_t)p->in_dim;
            const int16_t *restrict w4 = weights + (size_t)(o + 4) * (size_t)p->in_dim;
            const int16_t *restrict w5 = weights + (size_t)(o + 5) * (size_t)p->in_dim;
            const int16_t *restrict w6 = weights + (size_t)(o + 6) * (size_t)p->in_dim;
            const int16_t *restrict w7 = weights + (size_t)(o + 7) * (size_t)p->in_dim;
            for (int32_t i = 0; i < p->in_dim; ++i) {
                const int64_t xv = (int64_t)src[i];
                acc0 += xv * (int64_t)w0[i];
                acc1 += xv * (int64_t)w1[i];
                acc2 += xv * (int64_t)w2[i];
                acc3 += xv * (int64_t)w3[i];
                acc4 += xv * (int64_t)w4[i];
                acc5 += xv * (int64_t)w5[i];
                acc6 += xv * (int64_t)w6[i];
                acc7 += xv * (int64_t)w7[i];
            }
            {
                const int32_t bias0 = (bias_q != NULL) ? bias_q[o + 0] : 0;
                const int32_t bias1 = (bias_q != NULL) ? bias_q[o + 1] : 0;
                const int32_t bias2 = (bias_q != NULL) ? bias_q[o + 2] : 0;
                const int32_t bias3 = (bias_q != NULL) ? bias_q[o + 3] : 0;
                const int32_t bias4 = (bias_q != NULL) ? bias_q[o + 4] : 0;
                const int32_t bias5 = (bias_q != NULL) ? bias_q[o + 5] : 0;
                const int32_t bias6 = (bias_q != NULL) ? bias_q[o + 6] : 0;
                const int32_t bias7 = (bias_q != NULL) ? bias_q[o + 7] : 0;
                dst[o + 0] = kbnn_finalize_linear_output_i32(acc0, bias0, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 1] = kbnn_finalize_linear_output_i32(acc1, bias1, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 2] = kbnn_finalize_linear_output_i32(acc2, bias2, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 3] = kbnn_finalize_linear_output_i32(acc3, bias3, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 4] = kbnn_finalize_linear_output_i32(acc4, bias4, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 5] = kbnn_finalize_linear_output_i32(acc5, bias5, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 6] = kbnn_finalize_linear_output_i32(acc6, bias6, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 7] = kbnn_finalize_linear_output_i32(acc7, bias7, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
            }
        }
        for (; o + 3 < p->out_dim; o += 4) {
            int64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            const int16_t *restrict w0 = weights + (size_t)(o + 0) * (size_t)p->in_dim;
            const int16_t *restrict w1 = weights + (size_t)(o + 1) * (size_t)p->in_dim;
            const int16_t *restrict w2 = weights + (size_t)(o + 2) * (size_t)p->in_dim;
            const int16_t *restrict w3 = weights + (size_t)(o + 3) * (size_t)p->in_dim;
            for (int32_t i = 0; i < p->in_dim; ++i) {
                const int64_t xv = (int64_t)src[i];
                acc0 += xv * (int64_t)w0[i];
                acc1 += xv * (int64_t)w1[i];
                acc2 += xv * (int64_t)w2[i];
                acc3 += xv * (int64_t)w3[i];
            }
            {
                const int32_t bias0 = (bias_q != NULL) ? bias_q[o + 0] : 0;
                const int32_t bias1 = (bias_q != NULL) ? bias_q[o + 1] : 0;
                const int32_t bias2 = (bias_q != NULL) ? bias_q[o + 2] : 0;
                const int32_t bias3 = (bias_q != NULL) ? bias_q[o + 3] : 0;
                dst[o + 0] = kbnn_finalize_linear_output_i32(acc0, bias0, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 1] = kbnn_finalize_linear_output_i32(acc1, bias1, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 2] = kbnn_finalize_linear_output_i32(acc2, bias2, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                dst[o + 3] = kbnn_finalize_linear_output_i32(acc3, bias3, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
            }
        }
        for (; o < p->out_dim; ++o) {
            int64_t acc = 0;
            const int16_t *restrict wcol = weights + (size_t)o * (size_t)p->in_dim;
            for (int32_t i = 0; i < p->in_dim; ++i) {
                acc += (int64_t)src[i] * (int64_t)wcol[i];
            }
            {
                const int32_t bias_val = (bias_q != NULL) ? bias_q[o] : 0;
                dst[o] = kbnn_finalize_linear_output_i32(acc, bias_val, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.qdense_ns += kbnn_now_ns() - t0;
#endif
}

static void kbnn_exec_qact_qdense(const kbnn_context *ctx, const kbnn_qact_qdense_params *p) {
#ifdef KBNN_ENABLE_PROFILE
    uint64_t t0 = kbnn_now_ns();
#endif
    const kbnn_tensor_desc *input_desc = &ctx->model->tensors[p->input_id];
    const int32_t *input_i32 = NULL;
    const float *input_f32 = NULL;
    if (input_desc->kind == KBNN_TENSOR_EXTERNAL_F32) {
        input_f32 = kbnn_tensor_f32(ctx, p->input_id);
    } else {
        input_i32 = kbnn_tensor_i32(ctx, p->input_id);
    }
    int32_t *restrict output =
        (int32_t *)KBNN_ASSUME_ALIGNED(kbnn_tensor_i32(ctx, p->output_id), 64);
    const int16_t *restrict weights =
        (const int16_t *)KBNN_ASSUME_ALIGNED(p->weights, 64);
    const int32_t *restrict bias_q = p->bias_q;
    for (int32_t outer = 0; outer < p->outer_count; ++outer) {
        int32_t *dst = output + (size_t)outer * (size_t)p->out_dim;
        int32_t qbuf_local[4096] __attribute__((aligned(64)));
        int32_t *restrict qbuf = qbuf_local;
        if (p->in_dim <= 4096) {
            if (input_f32 != NULL) {
                const float *restrict src_f = input_f32 + (size_t)outer * (size_t)p->in_dim;
                for (int32_t i = 0; i < p->in_dim; ++i) {
                    const int32_t xv_q = kbnn_clip_i32((int64_t)lrintf(src_f[i] * (float)KBNN_FX_ONE), -KBNN_FX_ONE, KBNN_FX_ONE);
                    qbuf[i] = p->act_signed_mode
                        ? kbnn_quantize_signed_unit_i32(xv_q, p->act_levels)
                        : kbnn_quantize_unit_i32(xv_q, p->act_levels);
                }
            } else {
                const int32_t *restrict src_i32 = input_i32 + (size_t)outer * (size_t)p->in_dim;
                for (int32_t i = 0; i < p->in_dim; ++i) {
                    qbuf[i] = p->act_signed_mode
                        ? kbnn_quantize_signed_unit_i32(src_i32[i], p->act_levels)
                        : kbnn_quantize_unit_i32(src_i32[i], p->act_levels);
                }
            }
        }
        int32_t o = 0;
        if (p->in_dim <= 4096) {
            for (; o + 7 < p->out_dim; o += 8) {
                int64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                int64_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
                const int16_t *restrict w0 = weights + (size_t)(o + 0) * (size_t)p->in_dim;
                const int16_t *restrict w1 = weights + (size_t)(o + 1) * (size_t)p->in_dim;
                const int16_t *restrict w2 = weights + (size_t)(o + 2) * (size_t)p->in_dim;
                const int16_t *restrict w3 = weights + (size_t)(o + 3) * (size_t)p->in_dim;
                const int16_t *restrict w4 = weights + (size_t)(o + 4) * (size_t)p->in_dim;
                const int16_t *restrict w5 = weights + (size_t)(o + 5) * (size_t)p->in_dim;
                const int16_t *restrict w6 = weights + (size_t)(o + 6) * (size_t)p->in_dim;
                const int16_t *restrict w7 = weights + (size_t)(o + 7) * (size_t)p->in_dim;
                for (int32_t i = 0; i < p->in_dim; ++i) {
                    const int64_t xv = (int64_t)qbuf[i];
                    acc0 += xv * (int64_t)w0[i];
                    acc1 += xv * (int64_t)w1[i];
                    acc2 += xv * (int64_t)w2[i];
                    acc3 += xv * (int64_t)w3[i];
                    acc4 += xv * (int64_t)w4[i];
                    acc5 += xv * (int64_t)w5[i];
                    acc6 += xv * (int64_t)w6[i];
                    acc7 += xv * (int64_t)w7[i];
                }
                {
                    const int32_t bias0 = (bias_q != NULL) ? bias_q[o + 0] : 0;
                    const int32_t bias1 = (bias_q != NULL) ? bias_q[o + 1] : 0;
                    const int32_t bias2 = (bias_q != NULL) ? bias_q[o + 2] : 0;
                    const int32_t bias3 = (bias_q != NULL) ? bias_q[o + 3] : 0;
                    const int32_t bias4 = (bias_q != NULL) ? bias_q[o + 4] : 0;
                    const int32_t bias5 = (bias_q != NULL) ? bias_q[o + 5] : 0;
                    const int32_t bias6 = (bias_q != NULL) ? bias_q[o + 6] : 0;
                    const int32_t bias7 = (bias_q != NULL) ? bias_q[o + 7] : 0;
                    dst[o + 0] = kbnn_finalize_linear_output_i32(acc0, bias0, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 1] = kbnn_finalize_linear_output_i32(acc1, bias1, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 2] = kbnn_finalize_linear_output_i32(acc2, bias2, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 3] = kbnn_finalize_linear_output_i32(acc3, bias3, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 4] = kbnn_finalize_linear_output_i32(acc4, bias4, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 5] = kbnn_finalize_linear_output_i32(acc5, bias5, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 6] = kbnn_finalize_linear_output_i32(acc6, bias6, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 7] = kbnn_finalize_linear_output_i32(acc7, bias7, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                }
            }
            for (; o + 3 < p->out_dim; o += 4) {
                int64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                const int16_t *restrict w0 = weights + (size_t)(o + 0) * (size_t)p->in_dim;
                const int16_t *restrict w1 = weights + (size_t)(o + 1) * (size_t)p->in_dim;
                const int16_t *restrict w2 = weights + (size_t)(o + 2) * (size_t)p->in_dim;
                const int16_t *restrict w3 = weights + (size_t)(o + 3) * (size_t)p->in_dim;
                for (int32_t i = 0; i < p->in_dim; ++i) {
                    const int64_t xv = (int64_t)qbuf[i];
                    acc0 += xv * (int64_t)w0[i];
                    acc1 += xv * (int64_t)w1[i];
                    acc2 += xv * (int64_t)w2[i];
                    acc3 += xv * (int64_t)w3[i];
                }
                {
                    const int32_t bias0 = (bias_q != NULL) ? bias_q[o + 0] : 0;
                    const int32_t bias1 = (bias_q != NULL) ? bias_q[o + 1] : 0;
                    const int32_t bias2 = (bias_q != NULL) ? bias_q[o + 2] : 0;
                    const int32_t bias3 = (bias_q != NULL) ? bias_q[o + 3] : 0;
                    dst[o + 0] = kbnn_finalize_linear_output_i32(acc0, bias0, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 1] = kbnn_finalize_linear_output_i32(acc1, bias1, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 2] = kbnn_finalize_linear_output_i32(acc2, bias2, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                    dst[o + 3] = kbnn_finalize_linear_output_i32(acc3, bias3, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                }
            }
            for (; o < p->out_dim; ++o) {
                int64_t acc = 0;
                const int16_t *restrict wcol = weights + (size_t)o * (size_t)p->in_dim;
                for (int32_t i = 0; i < p->in_dim; ++i) {
                    acc += (int64_t)qbuf[i] * (int64_t)wcol[i];
                }
                {
                    const int32_t bias_val = (bias_q != NULL) ? bias_q[o] : 0;
                    dst[o] = kbnn_finalize_linear_output_i32(acc, bias_val, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                }
            }
        } else {
            for (; o < p->out_dim; ++o) {
                int64_t acc = 0;
                const int16_t *restrict wcol = weights + (size_t)o * (size_t)p->in_dim;
                for (int32_t i = 0; i < p->in_dim; ++i) {
                    int32_t xv_q;
                    if (input_f32 != NULL) {
                        xv_q = kbnn_clip_i32((int64_t)lrintf(input_f32[(size_t)outer * (size_t)p->in_dim + (size_t)i] * (float)KBNN_FX_ONE), -KBNN_FX_ONE, KBNN_FX_ONE);
                    } else {
                        xv_q = input_i32[(size_t)outer * (size_t)p->in_dim + (size_t)i];
                    }
                    const int32_t qv = p->act_signed_mode
                        ? kbnn_quantize_signed_unit_i32(xv_q, p->act_levels)
                        : kbnn_quantize_unit_i32(xv_q, p->act_levels);
                    acc += (int64_t)qv * (int64_t)wcol[i];
                }
                {
                    const int32_t bias_val = (bias_q != NULL) ? bias_q[o] : 0;
                    dst[o] = kbnn_finalize_linear_output_i32(acc, bias_val, p->weight_divisor, p->integer_output, p->levels, p->output_scale_q);
                }
            }
        }
    }
#ifdef KBNN_ENABLE_PROFILE
    ((kbnn_context *)ctx)->profile.qact_qdense_ns += kbnn_now_ns() - t0;
#endif
}

int kbnn_context_init(kbnn_context *ctx, const kbnn_model *model, void *arena, size_t arena_bytes) {
    if (ctx == NULL || model == NULL || arena == NULL) {
        return -1;
    }
    if (arena_bytes < model->arena_bytes) {
        return -2;
    }
    memset(ctx, 0, sizeof(*ctx));
    ctx->model = model;
    ctx->arena = (uint8_t *)arena;
    ctx->arena_bytes = arena_bytes;
    if (model->tensor_count > KBNN_MAX_TENSORS) {
        return -3;
    }
    for (int32_t i = 0; i < model->tensor_count; ++i) {
        const kbnn_tensor_desc *desc = &model->tensors[i];
        if (desc->kind == KBNN_TENSOR_EXTERNAL_F32) {
            ctx->tensor_ptrs[i] = NULL;
        } else {
            ctx->tensor_ptrs[i] = (void *)(ctx->arena + desc->offset_bytes);
        }
    }
    return 0;
}

void kbnn_context_release(kbnn_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    memset(ctx, 0, sizeof(*ctx));
}

void kbnn_context_reset_profile(kbnn_context *ctx) {
    if (ctx == NULL) {
        return;
    }
    memset(&ctx->profile, 0, sizeof(ctx->profile));
}

const kbnn_profile *kbnn_context_get_profile(const kbnn_context *ctx) {
    if (ctx == NULL) {
        return NULL;
    }
    return &ctx->profile;
}

void kbnn_context_set_input_f32(kbnn_context *ctx, int32_t tensor_id, const float *input) {
    if (ctx == NULL || tensor_id < 0 || tensor_id >= ctx->model->tensor_count) {
        return;
    }
    ctx->tensor_ptrs[tensor_id] = (void *)input;
}

const int32_t *kbnn_context_get_output_i32(const kbnn_context *ctx, int32_t tensor_id) {
    if (ctx == NULL || tensor_id < 0 || tensor_id >= ctx->model->tensor_count) {
        return NULL;
    }
    return (const int32_t *)ctx->tensor_ptrs[tensor_id];
}

void *kbnn_context_tensor_ptr(const kbnn_context *ctx, int32_t tensor_id) {
    if (ctx == NULL || tensor_id < 0 || tensor_id >= ctx->model->tensor_count) {
        return NULL;
    }
    return ctx->tensor_ptrs[tensor_id];
}

int kbnn_model_run(kbnn_context *ctx) {
    if (ctx == NULL || ctx->model == NULL) {
        return -1;
    }
    for (int32_t i = 0; i < ctx->model->op_count; ++i) {
        const kbnn_op *op = &ctx->model->ops[i];
#ifdef KBNN_ENABLE_PROFILE
        uint64_t t0 = kbnn_now_ns();
#endif
        switch (op->kind) {
            case KBNN_OP_THERMO_INPUT:
                kbnn_exec_thermo_input(ctx, (const kbnn_thermo_input_params *)op->params);
                break;
            case KBNN_OP_QUANTIZED_INPUT:
                kbnn_exec_quantized_input(ctx, (const kbnn_quantized_input_params *)op->params);
                break;
            case KBNN_OP_QCONV2D:
                kbnn_exec_qconv2d(ctx, (const kbnn_qconv2d_params *)op->params);
                break;
            case KBNN_OP_XNOR_CONV2D:
                kbnn_exec_xnor_conv2d(ctx, (const kbnn_xnor_conv2d_params *)op->params);
                break;
            case KBNN_OP_XNOR_DENSE:
                kbnn_exec_xnor_dense(ctx, (const kbnn_xnor_dense_params *)op->params);
                break;
            case KBNN_OP_SPACE_TO_DEPTH:
                kbnn_exec_space_to_depth(ctx, (const kbnn_space_to_depth_params *)op->params);
                break;
            case KBNN_OP_CONCAT:
                kbnn_exec_concat(ctx, (const kbnn_concat_params *)op->params);
                break;
            case KBNN_OP_UPSAMPLE2D:
                kbnn_exec_upsample2d(ctx, (const kbnn_upsample2d_params *)op->params);
                break;
            case KBNN_OP_BINARY_AFFINE_NORM:
                kbnn_exec_binary_affine(ctx, (const kbnn_binary_affine_params *)op->params);
                break;
            case KBNN_OP_QUANT_ACT:
                kbnn_exec_quant_act(ctx, (const kbnn_quant_act_params *)op->params);
                break;
            case KBNN_OP_QDENSE:
                kbnn_exec_qdense(ctx, (const kbnn_qdense_params *)op->params);
                break;
            case KBNN_OP_QACT_QDENSE:
                kbnn_exec_qact_qdense(ctx, (const kbnn_qact_qdense_params *)op->params);
                break;
            case KBNN_OP_ADD:
                kbnn_exec_add(ctx, (const kbnn_add_params *)op->params);
                break;
            default:
                return -2;
        }
#ifdef KBNN_ENABLE_PROFILE
        {
            uint64_t dt = kbnn_now_ns() - t0;
            ctx->profile.total_ns += dt;
            if (i < KBNN_MAX_PROFILE_OPS) {
                ctx->profile.op_ns[i] += dt;
                ctx->profile.op_calls[i] += 1;
            }
        }
#endif
    }
    return 0;
}

static float kbnn_iou(const kbnn_detection *a, const kbnn_detection *b) {
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
    if (iw <= 0.0f || ih <= 0.0f) {
        return 0.0f;
    }
    {
        const float inter = iw * ih;
        const float area_a = a->w * a->h;
        const float area_b = b->w * b->h;
        const float denom = area_a + area_b - inter;
        if (denom <= 0.0f) {
            return 0.0f;
        }
        return inter / denom;
    }
}

int kbnn_nms(const kbnn_detection *detections, int32_t count, float iou_threshold, int32_t top_k, int32_t *keep_indices) {
    int32_t kept = 0;
    if (detections == NULL || keep_indices == NULL || count <= 0) {
        return 0;
    }
    for (int32_t i = 0; i < count; ++i) {
        int suppress = 0;
        for (int32_t j = 0; j < kept; ++j) {
            const int32_t kept_idx = keep_indices[j];
            if (detections[kept_idx].class_id != detections[i].class_id) {
                continue;
            }
            if (kbnn_iou(&detections[kept_idx], &detections[i]) > iou_threshold) {
                suppress = 1;
                break;
            }
        }
        if (!suppress) {
            keep_indices[kept++] = i;
            if (top_k > 0 && kept >= top_k) {
                break;
            }
        }
    }
    return kept;
}
