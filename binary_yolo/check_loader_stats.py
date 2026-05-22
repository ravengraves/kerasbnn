from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from binary_yolo import yad2k_core


def _summarize_images(images: list[np.ndarray]) -> dict[str, float]:
    batch = np.stack(images).astype(np.float32)
    return {
        "min": float(batch.min()),
        "max": float(batch.max()),
        "mean": float(batch.mean()),
        "std": float(batch.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-format", default="auto")
    ap.add_argument("--dataset-split", default="auto")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-count", type=int, default=16)
    ap.add_argument("--positive-top-k", type=int, default=3)
    ap.add_argument("--thermo-levels", type=int, default=16)
    ap.add_argument("--train-augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--augment-progress", type=float, default=1.0)
    ap.add_argument("--advanced-augmentations", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--advanced-augment-prob", type=float, default=0.05)
    args = ap.parse_args()

    train_samples, val_samples, classes = yad2k_core.load_dataset_splits(
        Path(args.dataset),
        dataset_format=args.dataset_format,
        split_mode=args.dataset_split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        input_size=args.input_size,
    )
    rng = random.Random(args.seed)
    chosen = list(train_samples)
    rng.shuffle(chosen)
    chosen = chosen[: max(1, min(len(chosen), args.sample_count))]

    base_images = []
    aug_images = []
    base_boxes = 0
    aug_boxes = 0
    per_sample = []
    for sample in chosen:
        base_img, base_b, _ = yad2k_core._prepare_sample(
            sample,
            args.input_size,
            args.thermo_levels,
            train=False,
        )
        aug_img, aug_b, _ = yad2k_core._prepare_sample(
            sample,
            args.input_size,
            args.thermo_levels,
            train=args.train_augment,
            augment_progress=args.augment_progress,
            advanced_augmentations=args.advanced_augmentations,
            advanced_augment_prob=args.advanced_augment_prob,
        )
        base_images.append(base_img.astype(np.float32))
        aug_images.append(aug_img.astype(np.float32))
        base_boxes += int(len(base_b))
        aug_boxes += int(len(aug_b))
        per_sample.append({"image": sample.image_path.name, "base_boxes": int(len(base_b)), "aug_boxes": int(len(aug_b))})

    anchors = yad2k_core.fit_anchors(train_samples, args.input_size, n=6, seed=args.seed)
    batch_images, batch_targets = yad2k_core.build_targets(
        chosen,
        len(classes),
        anchors,
        args.input_size,
        thermo_levels=args.thermo_levels,
        augment=args.train_augment,
        positive_top_k=args.positive_top_k,
        image_dtype=np.float32,
        augment_progress=args.augment_progress,
        advanced_augmentations=args.advanced_augmentations,
        advanced_augment_prob=args.advanced_augment_prob,
    )
    thermo_preview = np.stack([yad2k_core._thermo_encode(img.astype(np.float32), args.thermo_levels) for img in aug_images], axis=0)

    report = {
        "dataset": str(args.dataset),
        "dataset_format": args.dataset_format,
        "dataset_split": args.dataset_split,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "classes": classes,
        "sample_count": len(chosen),
        "augment_progress": float(args.augment_progress),
        "advanced_augmentations": bool(args.advanced_augmentations),
        "advanced_augment_prob": float(args.advanced_augment_prob),
        "base_stats": _summarize_images(base_images),
        "aug_stats": _summarize_images(aug_images),
        "thermo_shape": list(thermo_preview.shape),
        "thermo_mean": float(thermo_preview.mean()),
        "thermo_std": float(thermo_preview.std()),
        "bbox_retention_ratio": float(aug_boxes / max(base_boxes, 1)),
        "base_box_count": int(base_boxes),
        "aug_box_count": int(aug_boxes),
        "batch_image_shape": list(batch_images.shape),
        "batch_target_shapes": {name: list(values.shape) for name, values in batch_targets.items()},
        "per_sample_boxes": per_sample,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
