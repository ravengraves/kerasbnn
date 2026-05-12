#include "kerasbnn.h"
#include "fruits_example_export.h"

static const int32_t fruits_example_input_ids[] = {0};

int32_t fruits_example_input_count(void) { return 1; }
int32_t fruits_example_output_count(void) { return 2; }
size_t fruits_example_arena_bytes(void) { return 20139520; }

int fruits_example_init(kbnn_context *ctx, void *arena, size_t arena_bytes) {
    return kbnn_context_init(ctx, &fruits_example, arena, arena_bytes);
}

int32_t fruits_example_input_tensor_id(int32_t index) {
    if (index < 0 || index >= 1) return -1;
    return fruits_example_input_ids[index];
}

int32_t fruits_example_output_tensor_id(int32_t index) {
    if (index < 0 || index >= 2) return -1;
    return fruits_example.output_ids[index];
}

void fruits_example_set_input_f32(kbnn_context *ctx, int32_t input_index, const float *input) {
    if (input_index < 0 || input_index >= 1) return;
    kbnn_context_set_input_f32(ctx, fruits_example_input_ids[input_index], input);
}

const int32_t *fruits_example_output_i32(const kbnn_context *ctx, int32_t output_index) {
    if (output_index < 0 || output_index >= 2) return (const int32_t *)0;
    return kbnn_context_get_output_i32(ctx, fruits_example.output_ids[output_index]);
}

int fruits_example_run(kbnn_context *ctx) {
    return kbnn_model_run(ctx);
}
