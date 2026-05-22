#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export KERAS_BACKEND="${KERAS_BACKEND:-jax}"

run_dir_specified=0
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--run-dir" ]]; then
    run_dir_specified=1
  fi
  prev="$arg"
done

if [[ "$run_dir_specified" -ne 1 ]]; then
  echo "train_student_2.sh requires --run-dir." >&2
  exit 2
fi

exec python3 -m binary_yolo.train_tinier_xnor2 \
  --dataset-format voc \
  --dataset-split official \
  --val-fraction 0.1 \
  --test-fraction 0.0 \
  --no-subset-train-as-val \
  --train-sample-fraction 1.0 \
  --val-sample-fraction 1.0 \
  --input-size 320 \
  --epochs 8 \
  --restart-cycles 1000 \
  --batch-size 8 \
  --loader-workers 1 \
  --repeat 1 \
  --optimizer adam \
  --lr 5e-4 \
  --schedule constant \
  --class-mode softmax \
  --box-mode linear \
  --obj-gain 4.0 \
  --obj-pos-gain 8.0 \
  --noobj-weight 0.02 \
  --cls-gain 8.0 \
  --box-gain 5.0 \
  --box-iou-aux-gain 0.5 \
  --box-iou-aux-mode iou \
  --hard-negative-min 64 \
  --conf-neg-normalizer pos \
  --positive-top-k 3 \
  --score-threshold 0.001 \
  --objectness-threshold 0.001 \
  --val-map-every 8 \
  --validation-freq 8 \
  --quantized-input \
  --input-levels 16 \
  --logit-scale 32 \
  --bn-recalibrate-before-val \
  --bn-recalibrate-batches 0 \
  --train-augment \
  --augment-start-strength 0.1 \
  --augment-ramp-epochs 256 \
  --train-cache-refresh-epochs 8 \
  --dtype-policy mixed_float16 \
  --jit-compile \
  --clipnorm 1.0 \
  --teacher-distill 0.0 \
  --restart-reload-monitor val_map50 \
  "$@"
