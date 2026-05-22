from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np

import binary_yolo.yad2k_core as yad2k_core
from binary_yolo import train_tinier_lagging_student
from binary_yolo.train_tinier import (
    MeanAPCallback,
    TINIER_ANCHOR_MASKS,
    _build_model,
    _prepare_split_samples,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-format", default="voc")
    ap.add_argument("--dataset-split", default="official")
    ap.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--train-sample-fraction", type=float, default=1.0)
    ap.add_argument("--val-sample-fraction", type=float, default=1.0)
    ap.add_argument("--test-sample-fraction", type=float, default=1.0)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--subset-train-as-val", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--score-threshold", type=float, default=0.05)
    ap.add_argument("--objectness-threshold", type=float, default=0.05)
    ap.add_argument("--thermo-levels", type=int, default=4)
    ap.add_argument("--input-levels", type=int, default=16)
    ap.add_argument("--input-signed", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-input", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-stem-first-layer", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--quantized-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--class-mode", default="softmax", choices=("softmax", "bce"))
    ap.add_argument("--box-mode", default="linear", choices=("linear", "linear_float", "yolo"))
    ap.add_argument("--bbox-output-scale", type=float, default=2.0)
    ap.add_argument("--obj-output-scale", type=float, default=64.0)
    ap.add_argument("--cls-output-scale", type=float, default=32.0)
    ap.add_argument("--logit-scale", type=float, default=32.0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--trainer", default="lagging", choices=("lagging", "teacher"))
    ap.add_argument("--anchors-csv", default="")
    ap.add_argument("--weights-path", default="")
    ap.add_argument("--use-last-keras", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use-bn-in-fire", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--head-pre-filters", type=int, default=256)
    ap.add_argument("--head-channels", type=int, default=64)
    ap.add_argument("--obj-cls-hidden", type=int, default=128)
    ap.add_argument("--obj-remap-hidden", type=int, default=64)
    ap.add_argument("--obj-bias-init", type=float, default=0.0)
    ap.add_argument("--input-dropcut", type=float, default=0.0)
    ap.add_argument("--input-dropcut-size-min", type=float, default=0.05)
    ap.add_argument("--input-dropcut-size-max", type=float, default=0.20)
    ap.add_argument("--input-dropcut-method", default="both", choices=("black", "white", "both", "noise"))
    ap.add_argument("--input-dropcut-boxes", type=int, default=1)
    ap.add_argument("--early-quantized-layers", type=int, default=1)
    ap.add_argument("--activation-surrogate", default="triangular", choices=("clipped", "triangular"))
    ap.add_argument("--activation-surrogate-gain", type=float, default=1.0)
    ap.add_argument("--weight-surrogate", default="magnitude_aware", choices=("clipped", "magnitude_aware"))
    ap.add_argument("--weight-surrogate-floor", type=float, default=0.0)
    ap.add_argument("--weight-surrogate-gain", type=float, default=1.0)
    ap.add_argument("--norm-gamma-init", type=float, default=0.25)
    ap.add_argument("--use-rprelu", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--rprelu-init-beta", type=float, default=0.25)
    ap.add_argument("--rprelu-init-gamma", type=float, default=0.0)
    ap.add_argument("--rprelu-init-zeta", type=float, default=0.0)
    ap.add_argument("--binary-kernel-init-scale", type=float, default=0.0)
    ap.add_argument("--activation-levels", type=int, default=128)
    ap.add_argument("--output-levels", type=int, default=128)
    ap.add_argument("--binary-batchnorm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-momentum", type=float, default=0.95)
    ap.add_argument("--bn-quantize-input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-centered", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-normalized", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-quantize-eval-stats", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bn-eval-use-batch-stats", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-force-full-quant-eval", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bn-quantize-affine-params", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stem-output-levels", type=int, default=0)
    ap.add_argument("--head-proj-levels", type=int, default=128)
    ap.add_argument("--late-quantized-from", type=int, default=0)
    ap.add_argument("--late-conv-levels", type=int, default=128)
    ap.add_argument("--late-act-levels", type=int, default=16)
    ap.add_argument("--backbone-width-mult", type=float, default=1.0)
    ap.add_argument("--tinyv3-backbone", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-prediction-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--float-neck-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--direct-quantized-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dtype-policy", default="float32", choices=("float32", "mixed_float16"))
    ap.add_argument("--eval-train", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def _teacher_spec_path(run_dir: Path) -> Path:
    return run_dir / "teacher_spec.json"


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def _resolve_model_args(args):
    run_dir = Path(args.run_dir)
    summary_path = _summary_path(run_dir)
    spec_path = _teacher_spec_path(run_dir)
    if any(
        bool(getattr(args, key, False))
        for key in (
            "tinyv3_backbone",
            "float_stem_first_layer",
            "quantized_input",
            "float_neck_head",
            "float_prediction_head",
            "direct_quantized_head",
        )
    ):
        import binary_yolo.train_tinier_xnor2  # noqa: F401
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        cfg = summary.get("tinier_cfg", {})
        if any(
            key in cfg
            for key in (
                "backbone_width_mult",
                "direct_quantized_head",
                "float_stem_first_layer",
                "float_neck_head",
                "bn_eval_use_batch_stats",
            )
        ):
            import binary_yolo.train_tinier_xnor2  # noqa: F401
        for key, value in cfg.items():
            setattr(args, key, value)
    if args.float_model and spec_path.exists():
        spec = json.loads(spec_path.read_text())
        args.use_bn_in_fire = bool(spec["use_bn_in_fire"])
        args.head_pre_filters = int(spec["head_pre_filters"])
        args.head_channels = int(spec["head_channels"])
        args.obj_cls_hidden = int(spec["obj_cls_hidden"])
        args.obj_remap_hidden = int(spec["obj_remap_hidden"])
        args.obj_bias_init = float(spec.get("obj_bias_init", 0.0))
        args.box_mode = str(spec["box_mode"])
        args.class_mode = str(spec["class_mode"])
        args.bbox_output_scale = float(spec["bbox_output_scale"])
        args.obj_output_scale = float(spec["obj_output_scale"])
        args.cls_output_scale = float(spec["cls_output_scale"])
        args.logit_scale = float(spec["logit_scale"])
    elif spec_path.exists():
        spec = json.loads(spec_path.read_text())
        args.use_bn_in_fire = bool(spec["use_bn_in_fire"])
        args.head_pre_filters = int(spec["head_pre_filters"])
        args.head_channels = int(spec["head_channels"])
        args.obj_cls_hidden = int(spec["obj_cls_hidden"])
        args.obj_remap_hidden = int(spec["obj_remap_hidden"])
        args.obj_bias_init = float(spec.get("obj_bias_init", 0.0))
        args.box_mode = str(spec["box_mode"])
        args.class_mode = str(spec["class_mode"])
        args.bbox_output_scale = float(spec["bbox_output_scale"])
        args.obj_output_scale = float(spec["obj_output_scale"])
        args.cls_output_scale = float(spec["cls_output_scale"])
        args.logit_scale = float(spec["logit_scale"])
    return args


def _weights_path(args):
    run_dir = Path(args.run_dir)
    if args.weights_path:
        return Path(args.weights_path)
    if args.use_last_keras:
        return run_dir / "model.voc.last.keras"
    return run_dir / "model.voc.best.weights.h5"


def _anchors_path(args) -> Path | None:
    if args.anchors_csv:
        return Path(args.anchors_csv)
    path = Path(args.run_dir) / "bbox_priors.csv"
    return path if path.exists() else None


def _load_anchors_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open() as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append((float(parts[1]), float(parts[2])))
    if not rows:
        raise ValueError(f"No anchors found in {path}")
    return np.asarray(rows, np.float32)


def _load_samples(args):
    split_args = SimpleNamespace(
        dataset=args.dataset,
        dataset_format=args.dataset_format,
        dataset_split=args.dataset_split,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        test_sample_fraction=args.test_sample_fraction,
        seed=args.seed,
        include_difficult=args.include_difficult,
        train_sample_fraction=args.train_sample_fraction,
        val_sample_fraction=args.val_sample_fraction,
        train_sample_count=0,
        subset_train_as_val=args.subset_train_as_val,
    )
    if args.trainer == "lagging":
        train_samples, val_samples, classes = train_tinier_lagging_student._prepare_split_samples(split_args, args.input_size)
        return train_samples, val_samples, [], classes
    return _prepare_split_samples(split_args, args.input_size)


def _build_images(samples, input_size: int, thermo_levels: int):
    if not samples:
        return np.zeros((0, input_size, input_size, 3), np.float32)
    return np.stack(
        [yad2k_core._prepare_sample(sample, input_size, thermo_levels, train=False)[0] for sample in samples]
    ).astype(np.float32)


def _eval_split(model, samples, images, anchors, args, num_classes: int):
    callback = MeanAPCallback(
        data=images,
        samples=samples,
        anchors=anchors,
        input_size=args.input_size,
        num_classes=num_classes,
        batch_size=args.batch_size,
        bbox_output_scale=args.bbox_output_scale,
        obj_output_scale=args.obj_output_scale,
        cls_output_scale=args.cls_output_scale,
        logit_scale=args.logit_scale,
        score_threshold=args.score_threshold,
        objectness_threshold=args.objectness_threshold,
        class_mode=args.class_mode,
        box_mode=args.box_mode,
        every=1,
    )
    callback.set_model(model)
    logs = {"_": 0}
    callback.on_epoch_end(0, logs)
    return float(logs["val_map50"])


def main():
    args = _resolve_model_args(parse_args())
    keras.config.set_dtype_policy(args.dtype_policy)

    train_samples, val_samples, test_samples, classes = _load_samples(args)
    anchors_path = _anchors_path(args)
    if anchors_path is not None and anchors_path.exists():
        anchors = _load_anchors_csv(anchors_path)
    else:
        anchors = yad2k_core.fit_anchors(train_samples, args.input_size, n=6, seed=args.seed)
    model_path = _weights_path(args)

    if args.use_last_keras or model_path.suffix == ".keras":
        model = keras.models.load_model(model_path, compile=False)
    else:
        model = _build_model(len(classes), args.input_size, args)
        model.load_weights(model_path, skip_mismatch=False)

    out = {
        "run_dir": str(Path(args.run_dir)),
        "artifact": str(model_path),
        "seed": int(args.seed),
        "subset_train_as_val": bool(args.subset_train_as_val),
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "anchors": anchors.tolist(),
    }
    if anchors_path is not None and anchors_path.exists():
        out["anchors_csv"] = str(anchors_path)

    if args.eval_train:
        train_images = _build_images(train_samples, args.input_size, args.thermo_levels)
        out["train_map50"] = _eval_split(model, train_samples, train_images, anchors, args, len(classes))
    val_images = _build_images(val_samples, args.input_size, args.thermo_levels)
    out["val_map50"] = _eval_split(model, val_samples, val_images, anchors, args, len(classes))
    if test_samples:
        test_images = _build_images(test_samples, args.input_size, args.thermo_levels)
        out["test_map50"] = _eval_split(model, test_samples, test_images, anchors, args, len(classes))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
