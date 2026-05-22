#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export KERAS_BACKEND="${KERAS_BACKEND:-jax}"

have_teacher_path=0
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--teacher-run-dir" || "$prev" == "--teacher-weights" ]]; then
    have_teacher_path=1
  fi
  prev="$arg"
done

if [[ "$have_teacher_path" -ne 1 ]]; then
  echo "train_student.sh requires --teacher-run-dir or --teacher-weights." >&2
  exit 2
fi

exec python3 -m binary_yolo.train_tinier_lagging_student \
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
  --student-inner-epochs 4 \
  --batch-size 8 \
  --teacher-batch-size 8 \
  --loader-workers 2 \
  --repeat 1 \
  --optimizer adam \
  --lr 1e-4 \
  --schedule cosine \
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
  --val-map-every 1 \
  --thermo-levels 4 \
  --input-dropcut 0.05 \
  --input-dropcut-size-min 0.05 \
  --input-dropcut-size-max 0.2 \
  --input-dropcut-method both \
  --input-dropcut-boxes 1 \
  --binary-batchnorm \
  --use-bn-in-fire \
  --bn-quantize-input \
  --no-bn-quantize-centered \
  --no-bn-quantize-normalized \
  --bn-quantize-output \
  --bn-quantize-eval-stats \
  --train-augment \
  --augment-start-strength 0.1 \
  --augment-ramp-epochs 100 \
  --cycle-early-stop-patience-epochs 64 \
  --cycle-early-stop-monitor val_map50 \
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
  --teacher-distill 0.005 \
  --teacher-bbox-distill 0.05 \
  --teacher-obj-distill 0.1 \
  --teacher-cls-distill 1.0 \
  --distill-score-threshold 0.25 \
  --teacher-checkpoint-monitor val_map50 \
  --restart-reload-monitor val_map50 \
  "$@"
