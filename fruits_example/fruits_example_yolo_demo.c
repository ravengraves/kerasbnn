/*
 * Example detector demo for a `kerasbnn` YOLO-style export.
 *
 * Build example:
 *   clang -Ofast -march=native -std=c11 -Wall -Wextra -pedantic \
 *     -I../../kerasbnn -I../../kerasbnn/third_party/stb \
 *     fruits_example_runtime.c fruits_example_detector.c fruits_example_yolo_demo.c \
 *     ../../kerasbnn/kerasbnn.c ../../kerasbnn/yolo_postprocess.c -lm -o yolo_demo
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "fruits_example_detector.h"

int main(int argc, char **argv) {
    int src_w = 0;
    int src_h = 0;
    int src_c = 0;
    uint8_t *arena = NULL;
    uint8_t *image = NULL;
    fruits_example_detector detector;
    kbnn_detection detections[FRUITS_EXAMPLE_YOLO_MAX_DETECTIONS];
    int32_t det_count;
    int i;
    if (argc != 2) {
        fprintf(stderr, "usage: %s image_path\n", argv[0]);
        return 1;
    }
    image = (uint8_t *) stbi_load(argv[1], &src_w, &src_h, &src_c, 3);
    if (image == NULL) {
        fprintf(stderr, "failed to load image: %s\n", argv[1]);
        return 1;
    }
    (void)src_c;
    arena = (uint8_t *) calloc(1u, fruits_example_detector_arena_bytes());
    if (arena == NULL) {
        stbi_image_free(image);
        free(arena);
        return 1;
    }
    if (fruits_example_detector_init(&detector, arena, fruits_example_detector_arena_bytes()) != 0) {
        stbi_image_free(image);
        free(arena);
        return 1;
    }
    det_count = fruits_example_detector_run_rgb8_resize(
        &detector,
        image,
        (int32_t)src_w,
        (int32_t)src_h,
        0,
        detections,
        FRUITS_EXAMPLE_YOLO_MAX_DETECTIONS
    );
    stbi_image_free(image);
    if (det_count < 0) {
        fruits_example_detector_release(&detector);
        free(arena);
        return 1;
    }
    printf("detections=%d\n", (int)det_count);
    for (i = 0; i < det_count; ++i) {
        const kbnn_detection *det = &detections[i];
        const float x1 = (det->cx - 0.5f * det->w) * (float)src_w;
        const float y1 = (det->cy - 0.5f * det->h) * (float)src_h;
        const float x2 = (det->cx + 0.5f * det->w) * (float)src_w;
        const float y2 = (det->cy + 0.5f * det->h) * (float)src_h;
        const char *name = fruits_example_detector_class_name(det->class_id);
        printf("%s score=%.4f box=[%.1f %.1f %.1f %.1f]\n", name, det->score, x1, y1, x2, y2);
    }
    fruits_example_detector_release(&detector);
    free(arena);
    return 0;
}
