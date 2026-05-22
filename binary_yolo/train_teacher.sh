#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export KERAS_BACKEND="${KERAS_BACKEND:-jax}"

exec python3 -m binary_yolo.train_tinier \
  --float-model \
  --dtype-policy float32 \
  --dataset-format voc \
  --dataset-split official \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --no-subset-train-as-val \
  --train-sample-fraction 1.0 \
  --val-sample-fraction 1.0 \
  --input-size 416 \
  --batch-size 16 \
  --epochs 256 \
  --restart-cycles 1000 \
  --optimizer adam \
  --lr 5e-4 \
  --schedule constant \
  --class-mode softmax \
  --box-mode yolo \
  --logit-scale 1 \
  --loss-logit-scale 1 \
  --obj-gain 4.0 \
  --obj-pos-gain 8.0 \
  --noobj-weight 0.30 \
  --cls-gain 8.0 \
  --box-gain 5.0 \
  --hard-negative-min 1024 \
  --conf-neg-normalizer pos \
  --positive-top-k 1 \
  --rescore-objectness \
  --score-threshold 0.05 \
  --objectness-threshold 0.05 \
  --clipnorm 1.0 \
  --val-map-every 16 \
  --validation-freq 16 \
  --no-use-bn-in-fire \
  --head-pre-filters 256 \
  --head-channels 64 \
  --obj-cls-hidden 128 \
  --obj-remap-hidden 64 \
  --loader-workers 1 \
  --loader-queue-size 64 \
  --no-loader-multiprocessing \
  --repeat 1 \
  --train-cache-refresh-divisor 33 \
  --train-augment \
  --augment-start-strength 0.1 \
  --augment-ramp-epochs 256 \
  --no-backup-and-restore \
  --teacher-distill 0 \
  --teacher-bbox-distill 0 \
  --teacher-obj-distill 0 \
  --teacher-cls-distill 0 \
  --restart-reload-monitor val_map50 \
  "$@"
