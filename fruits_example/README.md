# fruits_example

Generated C export for the current fruit detector checkpoint:

`../../runs/tinier_xnor2_fruits_gpu1_r11_256_thermo4_binarys2_advaugramp_bs8_cycle512_warmr10_lr5e5/model.voc.best.weights.h5`

Build the YOLO demo from this directory with:

```bash
clang -Ofast -march=native -std=c11 -Wall -Wextra -pedantic \
  -I.. -I../third_party/stb -I. \
  fruits_example_runtime.c \
  fruits_example_detector.c \
  fruits_example_yolo_demo.c \
  ../kerasbnn.c \
  ../yolo_postprocess.c \
  -lm -o fruits_example_demo
```
