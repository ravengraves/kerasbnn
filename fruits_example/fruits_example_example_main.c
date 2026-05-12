#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "kerasbnn.h"
#include "fruits_example_export.h"

int fruits_example_init(kbnn_context *ctx, void *arena, size_t arena_bytes);
int32_t fruits_example_output_count(void);
size_t fruits_example_arena_bytes(void);
void fruits_example_set_input_f32(kbnn_context *ctx, int32_t input_index, const float *input);
const int32_t *fruits_example_output_i32(const kbnn_context *ctx, int32_t output_index);
int32_t fruits_example_output_tensor_id(int32_t index);
int fruits_example_run(kbnn_context *ctx);

int main(void) {
    uint8_t *arena = (uint8_t *)calloc(1, fruits_example_arena_bytes());
    if (!arena) return 1;
    kbnn_context ctx;
    if (fruits_example_init(&ctx, arena, fruits_example_arena_bytes()) != 0) {
        free(arena);
        return 1;
    }
    float *input_0 = (float *)calloc(196608, sizeof(float));
    if (!input_0) { kbnn_context_release(&ctx); free(arena); return 1; }
    fruits_example_set_input_f32(&ctx, 0, input_0);
    if (fruits_example_run(&ctx) != 0) {
        kbnn_context_release(&ctx);
        free(input_0);
        free(arena);
        return 1;
    }
    for (int32_t i = 0; i < fruits_example_output_count(); ++i) {
        const int32_t *out = fruits_example_output_i32(&ctx, i);
        printf("output[%d] tensor_id=%d first_q12=%d\n", (int)i, (int)fruits_example_output_tensor_id(i), out ? (int)out[0] : 0);
    }
    kbnn_context_release(&ctx);
    free(input_0);
    free(arena);
    return 0;
}
