# binary_yolo

## Dataset layout

The current loaders accept three dataset families:

- Fruits XML dataset
- Pascal VOC
- COCO-style `instances_*.json`

### Fruits

Expected layout:

```text
fruits_detection/
  images/
    fruit0.png
    ...
  annotations/
    fruit0.xml
    ...
```

Notes:

- the XML files are Pascal-VOC-style annotations
- image filenames are resolved from the `<filename>` field inside each XML
- Windows `Zone.Identifier` sidecar files are ignored

### Pascal VOC

Expected layout:

```text
VOCdevkit/
  VOC2012/
    Annotations/
    JPEGImages/
    ImageSets/Main/
      train.txt
      val.txt
```

With `--dataset-format voc --dataset-split official`, the loader uses:

- `ImageSets/Main/train.txt`
- `ImageSets/Main/val.txt`

If those split files are missing, the loader falls back to a random split over
all annotations.

### COCO

Expected layout:

```text
mscoco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/    or images/train2017/
  val2017/      or images/val2017/
```

The loader searches for `instances_*.json` recursively and then resolves the
matching image directory automatically.

### Cache behavior

For VOC and COCO, the loader can build a shared resized cache under:

```text
<dataset>/cache/360x360/
```

That cache is reused across training runs even when the train input size
changes.

example starting a training run from a previous model:

CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=jax python3 -m binary_yolo.train_fruits
--dataset mscoco --dataset-format coco --dataset-split official --input-size 320 --batch-size 16 --skip-overfit --optimizer
adam --schedule cosine --epochs 256 --warm-start-weights runs/voc2012_320_warm111_strongaug_dropcut10_matcharch_gpu1/
model.voc.best.weights.h5 --no-loader-multiprocessing --loader-workers 4 --loader-queue-size 32 --run-dir runs/
coco2017_from_voc320_warmheadreset_bs16_gpu1

## Tinier Teacher / Student

Best full VOC float teacher so far:

- run: `runs/tinier_float_teacher_voc_full_80_10_10_curriculum_gpu0_r6_bs16_restart1000x256`
- best `val_map50 = 0.32602`
- best point at epoch `252` in `restart_cycle = 1`
- latest observed `val_map50 = 0.30110`
- `256x256`
- official VOC split, then `80/10/10`
- `batch_size=16`
- curriculum train augmentation
- no loader multiprocessing
- `2` loader workers
- no JAX preallocation
- warm start from the previous best full teacher checkpoint
- restart loop enabled:
  - `restart_cycles=1000`
  - `epochs=256`

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
KERAS_BACKEND=jax \
python3 -m binary_yolo.train_tinier \
  --float-model \
  --dataset VOCdevkit/VOC2012 \
  --dataset-format voc \
  --dataset-split official \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --input-size 256 \
  --dtype-policy float32 \
  --train-augment \
  --augment-start-strength 0.2 \
  --augment-ramp-epochs 100 \
  --epochs 256 \
  --restart-cycles 1000 \
  --batch-size 16 \
  --optimizer adam \
  --lr 5e-4 \
  --schedule cosine \
  --class-mode softmax \
  --box-mode yolo \
  --logit-scale 32 \
  --loss-logit-scale 32 \
  --obj-gain 4.0 \
  --obj-pos-gain 8.0 \
  --noobj-weight 0.04 \
  --cls-gain 8.0 \
  --box-gain 5.0 \
  --hard-negative-min 1024 \
  --conf-neg-normalizer pos \
  --positive-top-k 3 \
  --score-threshold 0.25 \
  --objectness-threshold 0.25 \
  --clipnorm 1.0 \
  --val-map-every 1 \
  --no-use-bn-in-fire \
  --head-pre-filters 0 \
  --head-channels 0 \
  --obj-cls-hidden 0 \
  --obj-remap-hidden 0 \
  --loader-workers 2 \
  --loader-queue-size 64 \
  --no-loader-multiprocessing \
  --train-cache-refresh-divisor 33 \
  --no-backup-and-restore \
  --teacher-distill 0 \
  --teacher-bbox-distill 0 \
  --teacher-obj-distill 0 \
  --teacher-cls-distill 0 \
  --warm-start-weights runs/tinier_float_teacher_voc_full_80_10_10_curriculum_gpu0_r3_bs16_mem10g/model.voc.best.weights.h5 \
  --warm-start-source-model float \
  --run-dir runs/tinier_float_teacher_voc_full_80_10_10_curriculum_gpu0_r6_bs16_restart1000x256
```

When launching a standalone float teacher, keep all teacher distillation weights at `0`. The training entry points now default to nonzero student distillation settings, so omitting these flags will incorrectly try to apply teacher-target losses inside the teacher run.

Best completed full VOC binary student so far:

- run: `runs/tinier_binary_lagging_student_fullvoc_gpu0_r10_clsobjbbox_bs64_e1024_curriculum`
- best `val_map50 = 0.36196`
- final `val_map50 = 0.27471`

Current best in-progress full VOC binary student so far:

- run: `runs/tinier_binary_lagging_student_fullvoc_gpu0_r25_warm_nodrop_bs16_restart1000x512`
- best `val_map50 = 0.49747`
- best point at epoch `3427` in `restart_cycle = 7`
- latest observed `val_map50 = 0.43571`

Current recommended full VOC binary student recipe:

- binary Tinier student
- warm start from the best current student checkpoint
- float teacher loaded from the teacher run directory
- bbox + objectness + class distillation
- `batch_size=16`
- curriculum train augmentation from `10%` to full over first `256` epochs
- restart loop enabled:
  - `restart_cycles=1000`
  - `outer_epochs=64`
  - `student_inner_epochs=8`
- this gives `512` student epochs per restart cycle, with a fresh optimizer/model rebuild at each restart while reloading the current best checkpoint
- the current leading run is the no-dropcut warm-start variant; dropcut `5%` is still useful to test as an alternate branch

Recommended launch command:

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
KERAS_BACKEND=jax \
python3 -m binary_yolo.train_tinier_lagging_student \
  --dataset VOCdevkit/VOC2012 \
  --dataset-format voc \
  --dataset-split official \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --no-subset-train-as-val \
  --train-sample-fraction 1.0 \
  --val-sample-fraction 1.0 \
  --input-size 256 \
  --restart-cycles 1000 \
  --outer-epochs 64 \
  --student-inner-epochs 8 \
  --batch-size 16 \
  --teacher-batch-size 16 \
  --repeat 1 \
  --optimizer adam \
  --lr 5e-5 \
  --schedule constant \
  --class-mode softmax \
  --box-mode linear \
  --obj-gain 4.0 \
  --obj-pos-gain 8.0 \
  --noobj-weight 0.02 \
  --cls-gain 8.0 \
  --box-gain 5.0 \
  --hard-negative-min 256 \
  --conf-neg-normalizer pos \
  --positive-top-k 3 \
  --score-threshold 0.05 \
  --objectness-threshold 0.05 \
  --val-map-every 4 \
  --thermo-levels 4 \
  --binary-batchnorm \
  --use-bn-in-fire \
  --bn-quantize-input \
  --no-bn-quantize-centered \
  --no-bn-quantize-normalized \
  --bn-quantize-output \
  --bn-quantize-eval-stats \
  --train-augment \
  --augment-start-strength 0.1 \
  --augment-ramp-epochs 256 \
  --head-pre-filters 256 \
  --head-channels 64 \
  --head-proj-levels 128 \
  --obj-cls-hidden 128 \
  --obj-remap-hidden 64 \
  --late-quantized-from 0 \
  --late-conv-levels 128 \
  --late-act-levels 16 \
  --dtype-policy float32 \
  --jit-compile \
  --clipnorm 1.0 \
  --teacher-run-dir runs/tinier_float_teacher_voc_full_80_10_10_curriculum_gpu0_r6_bs16_restart1000x256 \
  --teacher-distill 0.005 \
  --teacher-bbox-distill 0.05 \
  --teacher-obj-distill 0.1 \
  --teacher-cls-distill 1.0 \
  --distill-score-threshold 0.0 \
  --warm-start-weights runs/tinier_binary_lagging_student_fullvoc_gpu0_r24_warm_rot20_dropcut005_bs16_restart1000_tdistill0005/model.voc.best.weights.h5 \
  --warm-start-source-model binary \
  --run-dir runs/tinier_binary_lagging_student_fullvoc_gpu0_r25_warm_nodrop_bs16_restart1000x512
```

## C-native XNOR path

`binary_yolo/export_binary_yolo.py` exports the hardcoded default VOC-320 binary YOLO model into a simple packed record file.

`binary_yolo/binary_yolo.c` loads that file through a single contiguous weight arena, runs the hardcoded network from fixed buffers with no live allocations during inference, emits quantized head outputs, decodes detections, and applies fixed-point NMS.

For the binary conv stack, the export now derives the exact eval-time `BinaryAffineNorm(training=False)` transition in raw XNOR-popcount space and stores it as a per-channel comparator mode plus raw threshold. Binary conv weights are packed in 16-output-channel blocks so the C path can stream blocked `1x1` and `3x3` kernels directly.

The current default export target is the documented match-architecture VOC checkpoint:

`runs/voc2012_320_warm111_strongaug_dropcut10_matcharch_gpu1/model.voc.best.weights.h5`

## kerasbnn export path

The newer generic export path lives in `kerasbnn` and is the recommended route
for current `tinier_yolo_xnor2` models.

Generic bundle export:

```bash
python3 -m binary_yolo.export_tinier_xnor2_c \
  --checkpoint runs/.../model.voc.best_map.weights.h5 \
  --output-dir out/export \
  --model-name tinier_xnor2_export \
  --num-classes 4 \
  --dataset fruits_detection
```

Core emitted artifacts:

- `{model_name}_export.h`
- `{model_name}_export.json`
- `{model_name}_runtime.c`
- `{model_name}_example_main.c`

If anchors are available, the script also emits detector-side example files:

- `{model_name}_yolo_config.h`
- `{model_name}_yolo_demo.c`

The detector demo uses:

- `kerasbnn/yolo_postprocess.c`
- `kerasbnn/third_party/stb/stb_image.h`

and expects a direct resized RGB preprocessing path matching the current Python
loader behavior.

Example compile command for the emitted detector demo:

```bash
clang -Ofast -march=native -std=c11 -Wall -Wextra -pedantic \
  -Ikerasbnn -Ikerasbnn/third_party/stb \
  out/export/<model_name>_runtime.c \
  out/export/<model_name>_yolo_demo.c \
  kerasbnn/kerasbnn.c \
  kerasbnn/yolo_postprocess.c \
  -lm -o yolo_demo
```

## Stages

1. Weights export
   Export the default VOC-320 binary model into the packed `(layer_idx, blob)` runtime format with `binary_yolo/export_binary_yolo.py`.
2. Weights loading
   Load the packed records in `binary_yolo/binary_yolo.c` using hardcoded record order and hardcoded model dimensions.
3. Model execution
   Run the binary input path, binary backbone, binary neck, binary pre-head convs, raw projection path, quantized dense tail, and emit `p3/p4/p5` head outputs.
4. NMS
   Decode quantized box/objectness/class outputs and run fixed-point NMS to remove duplicates.
5. Verification
   Compare dumped quantized `p3/p4/p5` heads from the C runtime against the Python model on the same input.

## Space-to-depth / passthrough

Tinier-YOLO-style passthrough uses a `space-to-depth` reorganization with `block_size = 2` before concatenating a higher-resolution feature map into a lower-resolution head.

This operation does not compute a convolution, threshold, XNOR, popcount, or activation. It is only a data permutation:

- input spatial size shrinks by `2` in each dimension
- channel count grows by `2 * 2 = 4`
- every `2x2` neighborhood is packed into the channel axis

For an NHWC tensor `x[n, h, w, c]`, the output `y` has shape:

`y.shape = [N, H / 2, W / 2, C * 4]`

and the exact mapping is:

`y[n, ho, wo, ((dy * 2) + dx) * C + c] = x[n, ho * 2 + dy, wo * 2 + dx, c]`

where:

- `dy in {0, 1}`
- `dx in {0, 1}`
- `ho in [0, H / 2)`
- `wo in [0, W / 2)`
- `c in [0, C)`

This is the exact layout produced by the Keras implementation in `tiniermodel.py`, which does:

1. reshape from `[N, H, W, C]` to `[N, H / 2, 2, W / 2, 2, C]`
2. transpose to `[N, H / 2, W / 2, 2, 2, C]`
3. flatten the last three axes into `[N, H / 2, W / 2, 4 * C]`

The order of the output channels matters. Because the flattened order is `(dy, dx, c)`, the output channel blocks are:

- channels `0 .. C - 1`: top-left pixel of each `2x2` block
- channels `C .. 2C - 1`: top-right pixel
- channels `2C .. 3C - 1`: bottom-left pixel
- channels `3C .. 4C - 1`: bottom-right pixel

Example:

- input: `26 x 26 x 64`
- output: `13 x 13 x 256`

No information is discarded. The same `26 x 26 x 64 = 43264` feature values are still present after the transform, just arranged as `13 x 13 x 256 = 43264`.

### Why this is cheap in a binary runtime

For binary activations, space-to-depth is especially attractive because it is a pure routing layer:

- if activations are stored as `{-1, +1}` bytes, it is only copies
- if activations are stored as bits, it is only bit moves
- there are no multiplies, accumulators, thresholds, or normalization parameters

So the cost is almost entirely memory traffic and layout conversion. That makes it much cheaper than even a small binary convolution.

### C sketch: unpacked binary activations

If a prototype stores one binary activation per byte, the implementation is straightforward and easy to verify:

```c
static inline size_t idx_nhwc(
    uint32_t h,
    uint32_t w,
    uint32_t c,
    uint32_t H,
    uint32_t W,
    uint32_t C)
{
    return ((size_t)h * W + w) * C + c;
}

void binary_space_to_depth_u8(
    const uint8_t *src,
    uint8_t *dst,
    uint32_t H,
    uint32_t W,
    uint32_t C)
{
    const uint32_t block = 2;
    const uint32_t out_h = H / block;
    const uint32_t out_w = W / block;
    const uint32_t out_c = C * block * block;

    for (uint32_t oh = 0; oh < out_h; ++oh) {
        for (uint32_t ow = 0; ow < out_w; ++ow) {
            for (uint32_t dy = 0; dy < block; ++dy) {
                for (uint32_t dx = 0; dx < block; ++dx) {
                    const uint32_t ih = oh * block + dy;
                    const uint32_t iw = ow * block + dx;
                    const uint32_t out_base = ((dy * block) + dx) * C;
                    for (uint32_t ic = 0; ic < C; ++ic) {
                        const uint32_t oc = out_base + ic;
                        dst[idx_nhwc(oh, ow, oc, out_h, out_w, out_c)] =
                            src[idx_nhwc(ih, iw, ic, H, W, C)];
                    }
                }
            }
        }
    }
}
```

If `src` uses `0/1` bits, the values are copied unchanged. If it uses `-1/+1` bytes, those bytes are copied unchanged too.

### C sketch: packed binary activations

In a real binary runtime, channels are usually packed so each pixel stores `C` channels in `(C + 63) / 64` machine words. With that layout, space-to-depth is still just a permutation, but now it becomes a bit repack from one pixel's channel words into a different pixel's expanded channel words.

Assume:

- NHWC logical layout
- channels packed contiguously per pixel
- bit `1` means `+1`, bit `0` means `-1` or `false`

Then a simple reference implementation is:

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t get_bit(const uint64_t *words, uint32_t bit_idx)
{
    return (words[bit_idx >> 6] >> (bit_idx & 63u)) & 1ull;
}

static inline void set_bit(uint64_t *words, uint32_t bit_idx, uint64_t bit)
{
    words[bit_idx >> 6] |= bit << (bit_idx & 63u);
}

void binary_space_to_depth_bits(
    const uint64_t *src,
    uint64_t *dst,
    uint32_t H,
    uint32_t W,
    uint32_t C)
{
    const uint32_t block = 2;
    const uint32_t out_h = H / block;
    const uint32_t out_w = W / block;
    const uint32_t out_c = C * block * block;
    const uint32_t in_words = (C + 63u) >> 6;
    const uint32_t out_words = (out_c + 63u) >> 6;

    memset(dst, 0, (size_t)out_h * out_w * out_words * sizeof(uint64_t));

    for (uint32_t oh = 0; oh < out_h; ++oh) {
        for (uint32_t ow = 0; ow < out_w; ++ow) {
            uint64_t *dst_pixel = dst + ((size_t)oh * out_w + ow) * out_words;

            for (uint32_t dy = 0; dy < block; ++dy) {
                for (uint32_t dx = 0; dx < block; ++dx) {
                    const uint32_t ih = oh * block + dy;
                    const uint32_t iw = ow * block + dx;
                    const uint64_t *src_pixel =
                        src + ((size_t)ih * W + iw) * in_words;

                    const uint32_t out_base = ((dy * block) + dx) * C;
                    for (uint32_t ic = 0; ic < C; ++ic) {
                        const uint64_t bit = get_bit(src_pixel, ic);
                        set_bit(dst_pixel, out_base + ic, bit);
                    }
                }
            }
        }
    }
}
```

This is the easiest form to validate against Python because the mapping is explicit and follows the same channel order as the Keras `reshape -> transpose -> reshape`.

### Optimization notes for the C path

The reference packed implementation above is correct but not yet fast. A production version would usually:

- process one output pixel at a time from four source pixels
- move entire `uint64_t` words when `C` is large and aligned
- specialize the `block = 2` case rather than keeping it generic
- precompute source and destination word offsets
- use word-level shuffles instead of per-bit `get_bit` / `set_bit` where possible

The main point is that even the optimized version is still a re-packing kernel, not a math kernel. That is why space-to-depth is a good fit for a binary YOLO passthrough branch.

## Checklist

- [x] Weight export for the default VOC-320 binary model
- [x] Binary weight loading from a simple `(layer_idx, blob)` file format
- [x] Hardcoded model metadata and buffer sizes for the default architecture
- [x] XNOR-popcount execution for the binary backbone, neck, and pre-head conv stack
- [x] Mixed quantized head execution for the raw projection and dense output tail
- [x] Int8/fixed-point candidate decode and NMS
- [x] Head-dump verification path for comparing C output against the Python model
- [x] Single-arena weight storage and fixed inference buffers with no live allocation during the run

## Weight file format

The exported file is little-endian and starts with:

- magic: `BYOLOv1`
- version: `5`
- record count: `37`

Each record has:

- `uint32 layer_idx`
- `uint32 kind`
- `uint32 blob_bytes`
- `uint32 reserved`
- `blob_bytes` bytes of payload

Kinds:

- `1`: binary conv weights packed in `16`-channel output blocks plus exact raw-domain comparator thresholds derived from eval-time `BinaryAffineNorm`
- `2`: raw 1x1 projection conv with per-channel compact monotonic thresholds in raw-popcount space
- `3`: quantized hidden dense
- `4`: quantized output dense

The matching JSON manifest is written next to the binary file.

## Export

```bash
python3 binary_yolo/export_binary_yolo.py \
  --weights runs/voc2012_320_warm111_strongaug_dropcut10_matcharch_gpu1/model.voc.best.weights.h5 \
  --output binary_yolo/default_voc320_xnor.weights.bin
```

## Build

```bash
clang -O3 -march=native -mtune=native -std=c11 -Wall -Wextra -pedantic \
  binary_yolo/binary_yolo.c -lm -o binary_yolo/binary_yolo
```

If `clang` is unavailable, the same flags work with `gcc`.

## Run

The C binary accepts either:

- `--input-u8`: raw `320x320x3` RGB bytes
- `--input-f32`: raw `320x320x3` float32 values in `[0, 1]`

Example:

```bash
binary_yolo/binary_yolo \
  --weights binary_yolo/default_voc320_xnor.weights.bin \
  --input-u8 /tmp/input_320x320_rgb.bin \
  --dump-heads /tmp/binary_yolo_heads.bin \
  --dump-detections /tmp/binary_yolo_detections.txt
```

Add `--trace` to print per-layer packed activation and head summaries.

Add `--benchmark N` to run `N` timed inference iterations after weights and inputs are already loaded and report average network time, postprocess time, and FPS.

Add `--profile` together with `--benchmark` to print a per-stage timing breakdown.

## Verification

One simple smoke path is:

1. Export the weights with `export_binary_yolo.py`.
2. Run `binary_yolo.c` with `--dump-heads`.
3. Compare the dumped `p3/p4/p5` quantized heads against the Python model run on the same input.

Commands used for the current implementation:

1. `python3 -m py_compile binary_yolo/export_binary_yolo.py`
2. `python3 binary_yolo/export_binary_yolo.py --output /tmp/binary_yolo_test.weights.bin`
3. `clang -O3 -march=native -mtune=native -std=c11 -Wall -Wextra -pedantic binary_yolo/binary_yolo.c -lm -o /tmp/binary_yolo`
4. Run `/tmp/binary_yolo --weights /tmp/binary_yolo_test.weights.bin --input-u8 /tmp/binary_yolo_input_u8.bin --dump-heads /tmp/binary_yolo_heads.bin`
5. Compare `/tmp/binary_yolo_heads.bin` with the Python model outputs for the same input tensor

With the current export/runtime, the binary backbone, neck, and pre-head maps match the Python model exactly on the verification input. The remaining head drift is in the raw projection and quantized dense tail.
