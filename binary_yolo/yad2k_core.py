from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
import json
from pathlib import Path
import random
import threading
import xml.etree.ElementTree as ET

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from keras import ops


HEADS = ("p3", "p4", "p5")
STRIDES = (8, 16, 32)
ANCHOR_MASKS = ([0, 1, 2], [3, 4, 5], [6, 7, 8])
MAX_TRUE_BOXES = 20
_TRANSFORMS = threading.local()
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_COCO_CACHE_MANIFEST = "_cached_images.txt"
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    size: tuple[int, int]
    boxes_xyxy: np.ndarray
    class_ids: np.ndarray


def _as_hw(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError(f"`size` must be an int or a (height, width) pair. Got: {size!r}")
        return int(size[0]), int(size[1])
    side = int(size)
    return side, side


def _as_wh(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    h, w = _as_hw(size)
    return w, h


def _grid_hw(size: int | tuple[int, int] | list[int], stride: int) -> tuple[int, int]:
    h, w = _as_hw(size)
    return (h + stride - 1) // stride, (w + stride - 1) // stride


def _resolve_head_spec(heads=None, strides=None, anchor_masks=None):
    resolved_heads = tuple(HEADS if heads is None else heads)
    resolved_masks = tuple(tuple(mask) for mask in (ANCHOR_MASKS if anchor_masks is None else anchor_masks))
    if strides is None:
        if heads is not None or anchor_masks is not None:
            resolved_strides = tuple([None] * len(resolved_heads))
        else:
            resolved_strides = tuple(STRIDES)
    else:
        resolved_strides = tuple(strides)
    if not (len(resolved_heads) == len(resolved_strides) == len(resolved_masks)):
        raise ValueError("`heads`, `strides`, and `anchor_masks` must have matching lengths.")
    return resolved_heads, resolved_strides, resolved_masks


def _xyxy_to_cxcywh(boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.column_stack(
        (
            (boxes[:, 0] + boxes[:, 2]) * 0.5 / w,
            (boxes[:, 1] + boxes[:, 3]) * 0.5 / h,
            (boxes[:, 2] - boxes[:, 0]) / w,
            (boxes[:, 3] - boxes[:, 1]) / h,
        )
    ).astype(np.float32)


def _iou_wh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a[:, None, :], b[None, :, :]
    inter = np.minimum(a[..., 0], b[..., 0]) * np.minimum(a[..., 1], b[..., 1])
    union = a[..., 0] * a[..., 1] + b[..., 0] * b[..., 1] - inter
    return inter / np.maximum(union, 1e-6)


def _pairwise_iou_xyxy_ops(a, b, valid=None):
    a = ops.cast(a, "float32")
    b = ops.cast(b, "float32")
    a = ops.expand_dims(a, axis=2)
    b = ops.expand_dims(b, axis=1)
    tl = ops.maximum(a[..., :2], b[..., :2])
    br = ops.minimum(a[..., 2:], b[..., 2:])
    inter_wh = ops.maximum(br - tl, 0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = ops.maximum(a[..., 2] - a[..., 0], 0.0) * ops.maximum(a[..., 3] - a[..., 1], 0.0)
    area_b = ops.maximum(b[..., 2] - b[..., 0], 0.0) * ops.maximum(b[..., 3] - b[..., 1], 0.0)
    iou = inter / ops.maximum(area_a + area_b - inter, 1e-6)
    if valid is not None:
        iou = iou * ops.expand_dims(ops.cast(valid, "float32"), axis=1)
    return iou


def _bce_logits(target, logits):
    target, logits = ops.cast(target, "float32"), ops.cast(logits, "float32")
    return ops.maximum(logits, 0.0) - logits * target + ops.log1p(ops.exp(-ops.abs(logits)))


def _softmax_ce(target, logits):
    target, logits = ops.cast(target, "float32"), ops.cast(logits, "float32")
    logits = logits - ops.max(logits, axis=-1, keepdims=True)
    log_probs = logits - ops.log(ops.sum(ops.exp(logits), axis=-1, keepdims=True) + 1e-6)
    return -ops.sum(target * log_probs, axis=-1)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-6)


def _hard_negative_mask(neg_mask, neg_loss, pos_count, ratio: float, min_neg: int):
    if min_neg <= 0 and ratio <= 0.0:
        return neg_mask
    flat_mask = ops.reshape(ops.cast(neg_mask > 0.0, "float32"), (-1,))
    flat_loss = ops.reshape(ops.cast(neg_loss, "float32"), (-1,))
    masked_loss = ops.where(flat_mask > 0.0, flat_loss, ops.full_like(flat_loss, -1e9))
    total = int(masked_loss.shape[0])
    k_min = max(0, min(int(min_neg), total))
    if ratio > 0.0:
        k_ratio = ops.cast(ops.ceil(ops.cast(pos_count, "float32") * ratio), "int32")
    else:
        k_ratio = ops.cast(0, "int32")
    k = ops.maximum(ops.cast(k_min, "int32"), k_ratio)
    k = ops.minimum(k, ops.cast(total, "int32"))
    k = ops.maximum(k, 1)
    thresh = ops.take(ops.sort(masked_loss, axis=0), ops.maximum(total - k, 0), axis=0)
    flat_keep = ops.cast(masked_loss >= thresh, "float32") * flat_mask
    return ops.reshape(flat_keep, ops.shape(neg_mask))


def _detect_dataset_format(root: Path) -> str:
    if root.joinpath("Annotations").is_dir() and root.joinpath("JPEGImages").is_dir():
        return "voc"
    if _discover_coco_splits(root):
        return "coco"
    if root.joinpath("annotations").is_dir() and root.joinpath("images").is_dir():
        return "fruits"
    raise FileNotFoundError(f"Unsupported dataset layout under {root}")


def _load_fruit_dataset(root: Path) -> tuple[list[DetectionSample], list[str]]:
    root = Path(root)
    names, rows = set(), []
    for xml_path in sorted(root.joinpath("annotations").glob("*.xml")):
        if xml_path.name.endswith("Zone.Identifier"):
            continue
        xml = ET.parse(xml_path).getroot()
        image_path = root / "images" / xml.findtext("filename", "")
        if not image_path.exists():
            continue
        size = (int(xml.findtext("./size/width", "0")), int(xml.findtext("./size/height", "0")))
        labels, boxes = [], []
        for obj in xml.findall("object"):
            name, box = obj.findtext("name"), obj.find("bndbox")
            if not name or box is None:
                continue
            xyxy = [float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            names.add(name)
            labels.append(name)
            boxes.append(xyxy)
        if boxes:
            rows.append((image_path, size, boxes, labels))
    classes = sorted(names)
    class_to_id = {name: i for i, name in enumerate(classes)}
    samples = [
        DetectionSample(path, size, np.asarray(boxes, np.float32), np.asarray([class_to_id[n] for n in labels], np.int32))
        for path, size, boxes, labels in rows
    ]
    return samples, classes


def _read_split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _voc_image_path(image_dir: Path, stem: str, filename: str | None) -> Path:
    candidates = []
    if filename:
        candidates.append(image_dir / filename)
    candidates.extend(image_dir / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_voc_annotation(xml_path: Path, image_dir: Path, class_to_id: dict[str, int], include_difficult: bool) -> DetectionSample | None:
    xml = ET.parse(xml_path).getroot()
    stem = xml_path.stem
    image_path = _voc_image_path(image_dir, stem, xml.findtext("filename"))
    if not image_path.exists():
        return None
    size = (int(xml.findtext("./size/width", "0")), int(xml.findtext("./size/height", "0")))
    labels, boxes = [], []
    for obj in xml.findall("object"):
        name = obj.findtext("name")
        if not name or name not in class_to_id:
            continue
        difficult = int(obj.findtext("difficult", "0") or "0")
        if difficult and not include_difficult:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xyxy = [float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        labels.append(class_to_id[name])
        boxes.append(xyxy)
    if not boxes:
        return None
    return DetectionSample(image_path, size, np.asarray(boxes, np.float32), np.asarray(labels, np.int32))


def _load_voc_split(
    root: Path,
    split_ids: list[str],
    include_difficult: bool = False,
    cache_size: int | None = None,
) -> tuple[list[DetectionSample], list[str]]:
    class_to_id = {name: i for i, name in enumerate(VOC_CLASSES)}
    image_dir = root / "JPEGImages"
    samples = []
    for stem in split_ids:
        sample = _parse_voc_annotation(root / "Annotations" / f"{stem}.xml", image_dir, class_to_id, include_difficult)
        if sample is not None:
            samples.append(sample)
    if cache_size is not None and samples:
        cache_dir = root / "cache" / f"{cache_size}x{cache_size}" / "JPEGImages"
        cached_names = _ensure_image_cache(cache_dir, [sample.image_path for sample in samples], cache_size)
        resized_samples = []
        for sample in samples:
            if sample.image_path.name not in cached_names:
                resized_samples.append(sample)
                continue
            resized_samples.append(
                DetectionSample(
                    cache_dir / sample.image_path.name,
                    (cache_size, cache_size),
                    _scale_boxes_xyxy(sample.boxes_xyxy, sample.size, (cache_size, cache_size)).astype(np.float32),
                    sample.class_ids,
                )
            )
        samples = resized_samples
    return samples, list(VOC_CLASSES)


def _has_images(path: Path) -> bool:
    try:
        return any(child.is_file() and child.suffix.lower() in _IMAGE_SUFFIXES for child in path.iterdir())
    except FileNotFoundError:
        return False


def _resolve_coco_image_candidate(path: Path, split: str) -> Path | None:
    if not path.is_dir():
        return None
    if _has_images(path):
        return path
    nested = path / split
    if nested.is_dir() and _has_images(nested):
        return nested
    children = [child for child in sorted(path.iterdir()) if child.is_dir()]
    if len(children) == 1 and _has_images(children[0]):
        return children[0]
    return None


def _resolve_coco_image_dir(root: Path, split: str) -> Path | None:
    seen = set()
    candidates = [root / split, root / "images" / split]
    candidates.extend(sorted(root.glob(f"**/{split}")))
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        resolved = _resolve_coco_image_candidate(candidate, split)
        if resolved is not None:
            return resolved
    return None


def _discover_coco_splits(root: Path) -> dict[str, dict[str, Path | None]]:
    splits: dict[str, dict[str, Path | None]] = {}
    for ann_path in sorted(root.glob("**/instances_*.json"), key=lambda path: (len(path.parts), str(path))):
        split = ann_path.stem.removeprefix("instances_")
        if split in splits:
            continue
        splits[split] = {
            "annotation": ann_path,
            "image_dir": _resolve_coco_image_dir(root, split),
        }
    return splits


def _sort_coco_splits(names: list[str]) -> list[str]:
    def key(name: str):
        if name.startswith("train"):
            return (0, name)
        if name.startswith("val"):
            return (1, name)
        return (2, name)

    return sorted(names, key=key)


def _preferred_coco_split(splits: dict[str, dict[str, Path | None]], prefix: str) -> str | None:
    matches = [name for name, row in splits.items() if name.startswith(prefix) and row["image_dir"] is not None]
    if not matches:
        return None
    preferred = f"{prefix}2017"
    if preferred in matches:
        return preferred
    return sorted(matches)[0]


def _scale_boxes_xyxy(boxes: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    if not len(boxes):
        return boxes.astype(np.float32)
    scaled = boxes.astype(np.float32).copy()
    sx = float(dst_size[0]) / max(float(src_size[0]), 1.0)
    sy = float(dst_size[1]) / max(float(src_size[1]), 1.0)
    scaled[:, [0, 2]] *= sx
    scaled[:, [1, 3]] *= sy
    return scaled


def _dataset_cache_size(input_size: int | None) -> int:
    del input_size
    # Keep one shared cache resolution so warm-starting at a new train size
    # reuses the same cached JPEGs and scaled annotations.
    return 360


def _coco_cache_dir(root: Path, split: str, cache_size: int) -> Path:
    return root / "cache" / f"{cache_size}x{cache_size}" / split


def _read_cache_names(cache_dir: Path) -> set[str]:
    manifest = cache_dir / _COCO_CACHE_MANIFEST
    if manifest.exists():
        return {line.strip() for line in manifest.read_text().splitlines() if line.strip()}
    if not cache_dir.exists():
        return set()
    return {
        path.name
        for path in cache_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }


def _write_cache_names(cache_dir: Path, names: set[str]):
    manifest = cache_dir / _COCO_CACHE_MANIFEST
    manifest.write_text("".join(f"{name}\n" for name in sorted(names)))


def _resize_cached_image(src_path: Path, dst_path: Path, cache_size: int) -> str | None:
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            return dst_path.name
        tmp_path = dst_path.with_name(f"{dst_path.stem}.tmp{dst_path.suffix}")
        with Image.open(src_path) as image:
            resized = image.convert("RGB").resize((cache_size, cache_size), Image.BILINEAR)
            save_kwargs = {"quality": 90} if dst_path.suffix.lower() in {".jpg", ".jpeg"} else {}
            resized.save(tmp_path, **save_kwargs)
        tmp_path.replace(dst_path)
        return dst_path.name
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        return None


def _ensure_image_cache(cache_dir: Path, image_paths: list[Path], cache_size: int) -> set[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_names = {name for name in _read_cache_names(cache_dir) if cache_dir.joinpath(name).exists()}
    jobs = []
    for image_path in image_paths:
        if image_path.name in cached_names or not image_path.exists():
            continue
        jobs.append((image_path, cache_dir / image_path.name))
    if jobs:
        worker_count = max(1, min(len(jobs), os.cpu_count() or 1, 8))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for name in pool.map(lambda job: _resize_cached_image(job[0], job[1], cache_size), jobs):
                if name:
                    cached_names.add(name)
    if cached_names:
        _write_cache_names(cache_dir, cached_names)
    return cached_names


def _ensure_coco_split_cache(root: Path, split: str, image_dir: Path, image_meta: dict[int, dict], cache_size: int) -> tuple[Path, set[str]]:
    cache_dir = _coco_cache_dir(root, split, cache_size)
    cached_names = _ensure_image_cache(
        cache_dir,
        [image_dir / str(meta.get("file_name", "")) for meta in image_meta.values() if meta.get("file_name")],
        cache_size,
    )
    return cache_dir, cached_names


def _load_coco_split(
    root: Path,
    ann_path: Path,
    image_dir: Path,
    class_to_id: dict[int, int] | None = None,
    classes: list[str] | None = None,
    cache_size: int | None = None,
) -> tuple[list[DetectionSample], list[str], dict[int, int]]:
    with ann_path.open() as fp:
        payload = json.load(fp)
    if class_to_id is None or classes is None:
        categories = sorted(payload.get("categories", []), key=lambda row: int(row.get("id", 0)))
        classes = [str(row.get("name", row.get("id", ""))) for row in categories]
        class_to_id = {int(row["id"]): i for i, row in enumerate(categories)}
    image_meta = {int(row["id"]): row for row in payload.get("images", []) if "id" in row}
    boxes_by_image: dict[int, list[list[float]]] = {}
    labels_by_image: dict[int, list[int]] = {}
    for ann in payload.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image_id = int(ann.get("image_id", -1))
        meta = image_meta.get(image_id)
        if meta is None:
            continue
        category_id = int(ann.get("category_id", -1))
        if category_id not in class_to_id:
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0.0 or h <= 0.0:
            continue
        width = float(meta.get("width", 0) or 0)
        height = float(meta.get("height", 0) or 0)
        if width <= 0.0 or height <= 0.0:
            continue
        x1 = min(max(x, 0.0), width)
        y1 = min(max(y, 0.0), height)
        x2 = min(max(x + w, 0.0), width)
        y2 = min(max(y + h, 0.0), height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_by_image.setdefault(image_id, []).append([x1, y1, x2, y2])
        labels_by_image.setdefault(image_id, []).append(class_to_id[category_id])
    split = ann_path.stem.removeprefix("instances_") if ann_path.stem.startswith("instances_") else image_dir.name
    used_image_meta = {image_id: image_meta[image_id] for image_id in boxes_by_image if image_id in image_meta}
    cache_dir = None
    cached_names: set[str] = set()
    if cache_size is not None and used_image_meta:
        cache_dir, cached_names = _ensure_coco_split_cache(root, split, image_dir, used_image_meta, cache_size)
    samples = []
    for image_id in sorted(boxes_by_image):
        meta = image_meta.get(image_id)
        if meta is None:
            continue
        file_name = str(meta.get("file_name", ""))
        if not file_name:
            continue
        image_path = image_dir / file_name
        if not image_path.exists():
            continue
        size = (int(meta.get("width", 0) or 0), int(meta.get("height", 0) or 0))
        if size[0] <= 0 or size[1] <= 0:
            continue
        sample_path = image_path
        sample_size = size
        sample_boxes = np.asarray(boxes_by_image[image_id], np.float32)
        if cache_dir is not None and file_name in cached_names:
            sample_path = cache_dir / file_name
            sample_size = (cache_size, cache_size)
            sample_boxes = _scale_boxes_xyxy(sample_boxes, size, sample_size)
        samples.append(
            DetectionSample(
                sample_path,
                sample_size,
                sample_boxes.astype(np.float32),
                np.asarray(labels_by_image[image_id], np.int32),
            )
        )
    return samples, list(classes), dict(class_to_id)


def load_dataset(
    root: str | Path,
    dataset_format: str = "auto",
    include_difficult: bool = False,
    input_size: int | None = None,
) -> tuple[list[DetectionSample], list[str]]:
    root = Path(root)
    dataset_format = _detect_dataset_format(root) if dataset_format == "auto" else dataset_format
    if dataset_format == "voc":
        cache_size = _dataset_cache_size(input_size)
        split_ids = sorted(path.stem for path in root.joinpath("Annotations").glob("*.xml"))
        return _load_voc_split(root, split_ids, include_difficult=include_difficult, cache_size=cache_size)
    if dataset_format == "coco":
        del include_difficult
        cache_size = _dataset_cache_size(input_size)
        splits = _discover_coco_splits(root)
        available = [(name, row) for name, row in splits.items() if row["image_dir"] is not None]
        if not available:
            raise FileNotFoundError(f"No usable COCO image directories found under {root}")
        samples = []
        classes: list[str] | None = None
        class_to_id: dict[int, int] | None = None
        split_rows = {name: row for name, row in available}
        for name in _sort_coco_splits([name for name, _ in available]):
            split_samples, classes, class_to_id = _load_coco_split(
                root,
                split_rows[name]["annotation"],
                split_rows[name]["image_dir"],
                class_to_id=class_to_id,
                classes=classes,
                cache_size=cache_size,
            )
            samples.extend(split_samples)
        return samples, classes or []
    return _load_fruit_dataset(root)


def load_dataset_splits(
    root: str | Path,
    dataset_format: str = "auto",
    split_mode: str = "auto",
    val_fraction: float = 0.1,
    seed: int = 7,
    include_difficult: bool = False,
    input_size: int | None = None,
):
    root = Path(root)
    dataset_format = _detect_dataset_format(root) if dataset_format == "auto" else dataset_format
    if split_mode == "auto":
        if dataset_format == "voc" and root.joinpath("ImageSets/Main/train.txt").exists() and root.joinpath("ImageSets/Main/val.txt").exists():
            split_mode = "official"
        elif dataset_format == "coco":
            coco_splits = _discover_coco_splits(root)
            split_mode = "official" if _preferred_coco_split(coco_splits, "train") and _preferred_coco_split(coco_splits, "val") else "random"
        else:
            split_mode = "random"
    if dataset_format == "voc" and split_mode == "official":
        cache_size = _dataset_cache_size(input_size)
        train_ids = _read_split_ids(root / "ImageSets/Main/train.txt")
        val_ids = _read_split_ids(root / "ImageSets/Main/val.txt")
        train_samples, classes = _load_voc_split(root, train_ids, include_difficult=include_difficult, cache_size=cache_size)
        val_samples, _ = _load_voc_split(root, val_ids, include_difficult=include_difficult, cache_size=cache_size)
        return train_samples, val_samples, classes
    if dataset_format == "coco" and split_mode == "official":
        del include_difficult
        cache_size = _dataset_cache_size(input_size)
        coco_splits = _discover_coco_splits(root)
        train_split = _preferred_coco_split(coco_splits, "train")
        val_split = _preferred_coco_split(coco_splits, "val")
        if train_split is None or val_split is None:
            available = ", ".join(_sort_coco_splits(list(coco_splits))) or "none"
            raise FileNotFoundError(
                f"COCO official splits require train and val images under {root}; found {available}. "
                "Use split_mode='random' until the dataset copy finishes."
            )
        train_info = coco_splits[train_split]
        val_info = coco_splits[val_split]
        train_samples, classes, class_to_id = _load_coco_split(root, train_info["annotation"], train_info["image_dir"], cache_size=cache_size)
        val_samples, _, _ = _load_coco_split(root, val_info["annotation"], val_info["image_dir"], class_to_id=class_to_id, classes=classes, cache_size=cache_size)
        return train_samples, val_samples, classes
    samples, classes = load_dataset(root, dataset_format=dataset_format, include_difficult=include_difficult, input_size=input_size)
    train_samples, val_samples = split_samples(samples, val_fraction=val_fraction, seed=seed)
    return train_samples, val_samples, classes


def split_samples(samples: list[DetectionSample], val_fraction: float = 0.1, seed: int = 7):
    ids = list(range(len(samples)))
    random.Random(seed).shuffle(ids)
    cut = max(1, round(len(ids) * val_fraction))
    val = set(ids[:cut])
    return [samples[i] for i in ids if i not in val], [samples[i] for i in ids if i in val]


def split_samples_three_way(
    samples: list[DetectionSample],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 7,
):
    ids = list(range(len(samples)))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    if n < 3:
        train, val = split_samples(samples, val_fraction=max(val_fraction, test_fraction), seed=seed)
        return train, val, []
    val_count = max(1, round(n * val_fraction)) if val_fraction > 0.0 else 0
    test_count = max(1, round(n * test_fraction)) if test_fraction > 0.0 else 0
    if val_count + test_count >= n:
        overflow = val_count + test_count - (n - 1)
        if test_count >= overflow:
            test_count -= overflow
        else:
            overflow -= test_count
            test_count = 0
            val_count = max(0, val_count - overflow)
    test_ids = set(ids[:test_count])
    val_ids = set(ids[test_count:test_count + val_count])
    train = [samples[i] for i in ids if i not in test_ids and i not in val_ids]
    val = [samples[i] for i in ids if i in val_ids]
    test = [samples[i] for i in ids if i in test_ids]
    return train, val, test


def find_sample(samples: list[DetectionSample], stem: str) -> DetectionSample:
    return next(sample for sample in samples if sample.image_path.stem == stem)


def fit_anchors(samples: list[DetectionSample], input_size: int | tuple[int, int], n: int = 9, seed: int = 7, iters: int = 25) -> np.ndarray:
    target_wh = np.asarray(_as_wh(input_size), np.float32)
    wh = np.concatenate([_xyxy_to_cxcywh(s.boxes_xyxy, s.size)[:, 2:4] * target_wh for s in samples], axis=0)
    rng = np.random.default_rng(seed)
    anchors = wh[rng.choice(len(wh), size=n, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmin(1.0 - _iou_wh(wh, anchors), axis=1)
        new = anchors.copy()
        for i in range(n):
            cluster = wh[assign == i]
            if len(cluster):
                new[i] = np.median(cluster, axis=0)
        if np.allclose(new, anchors):
            break
        anchors = new
    return anchors[np.argsort(np.prod(anchors, axis=1))].astype(np.float32)


def _thermo_encode(image: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 1:
        return image
    thresholds = ((np.arange(levels, dtype=np.float32) + 1.0) / (levels + 1.0)).reshape(1, 1, 1, levels)
    return (image[..., None] >= thresholds).astype(np.float32).reshape(image.shape[0], image.shape[1], image.shape[2] * levels)


def _load_image(path: Path, size: int | tuple[int, int], thermo_levels: int = 1) -> np.ndarray:
    del thermo_levels
    h, w = _as_hw(size)
    image = np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255.0
    return image


@lru_cache(maxsize=12288)
def _cached_rgb_image(path_str: str) -> np.ndarray:
    # Cache decoded uint8 RGB images in-process so repeated epochs do not keep
    # reopening and decoding the same cached VOC/COCO JPEGs on the host.
    return np.asarray(Image.open(path_str).convert("RGB"), np.uint8)


def _augment_strength(progress: float) -> float:
    return float(np.clip(progress, 0.0, 1.0))


class ApplyVignette(A.ImageOnlyTransform):
    def __init__(self, strength_range=(0.15, 0.40), center_jitter=0.12, p=0.5):
        super().__init__(p=p)
        self.strength_range = tuple(float(v) for v in strength_range)
        self.center_jitter = float(center_jitter)

    def apply(self, img, strength=0.25, cx_shift=0.0, cy_shift=0.0, **params):
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (0.5 + float(cx_shift)) * max(w - 1, 1)
        cy = (0.5 + float(cy_shift)) * max(h - 1, 1)
        dx = (xx - cx) / max(float(w), 1.0)
        dy = (yy - cy) / max(float(h), 1.0)
        dist = np.sqrt(dx * dx + dy * dy)
        dist = dist / max(float(dist.max()), 1e-6)
        mask = 1.0 - float(strength) * (dist ** 2)
        mask = np.clip(mask, 0.35, 1.0)[..., None]
        out = img.astype(np.float32) * mask
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def get_params(self):
        return {
            "strength": float(np.random.uniform(*self.strength_range)),
            "cx_shift": float(np.random.uniform(-self.center_jitter, self.center_jitter)),
            "cy_shift": float(np.random.uniform(-self.center_jitter, self.center_jitter)),
        }

    def get_transform_init_args_names(self):
        return ("strength_range", "center_jitter")


class ApplyFilmGrain(A.ImageOnlyTransform):
    def __init__(self, sigma_range=(4.0, 18.0), monochrome_prob=0.75, p=0.5):
        super().__init__(p=p)
        self.sigma_range = tuple(float(v) for v in sigma_range)
        self.monochrome_prob = float(monochrome_prob)

    def apply(self, img, sigma=8.0, monochrome=True, **params):
        img_f = img.astype(np.float32)
        if monochrome:
            noise = np.random.normal(0.0, float(sigma), img.shape[:2]).astype(np.float32)[..., None]
            noise = np.repeat(noise, 3, axis=2)
        else:
            noise = np.random.normal(0.0, float(sigma), img.shape).astype(np.float32)
        out = img_f + noise
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def get_params(self):
        return {
            "sigma": float(np.random.uniform(*self.sigma_range)),
            "monochrome": bool(np.random.random() < self.monochrome_prob),
        }

    def get_transform_init_args_names(self):
        return ("sigma_range", "monochrome_prob")


class ApplyAtmosphericFog(A.ImageOnlyTransform):
    def __init__(self, fog_strength_range=(0.08, 0.30), sigma_frac_range=(0.08, 0.18), p=0.5):
        super().__init__(p=p)
        self.fog_strength_range = tuple(float(v) for v in fog_strength_range)
        self.sigma_frac_range = tuple(float(v) for v in sigma_frac_range)

    def apply(self, img, fog_strength=0.18, sigma_frac=0.12, **params):
        h, w = img.shape[:2]
        sigma = max(1.0, float(sigma_frac) * max(h, w))
        fog_field = np.random.rand(h, w).astype(np.float32)
        fog_field = cv2.GaussianBlur(fog_field, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        fog_field = fog_field - float(fog_field.min())
        fog_field = fog_field / max(float(fog_field.max()), 1e-6)
        fog = (float(fog_strength) * fog_field)[..., None]
        img_f = img.astype(np.float32)
        out = img_f * (1.0 - fog) + 255.0 * fog
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def get_params(self):
        return {
            "fog_strength": float(np.random.uniform(*self.fog_strength_range)),
            "sigma_frac": float(np.random.uniform(*self.sigma_frac_range)),
        }

    def get_transform_init_args_names(self):
        return ("fog_strength_range", "sigma_frac_range")


def _advanced_train_transforms():
    return [
        A.Posterize(num_bits=(2, 4), p=1.0),
        A.Solarize(threshold_range=(0.35, 0.65), p=1.0),
        A.CLAHE(clip_limit=(1.5, 4.0), tile_grid_size=(8, 8), p=1.0),
        A.RandomSnow(method="texture", p=1.0),
        A.RandomSnow(method="bleach", p=1.0),
        A.RandomRain(rain_type="default", p=1.0),
        A.RandomFog(p=1.0),
        A.RandomGravel(p=1.0),
        A.RandomSunFlare(method="physics_based", p=1.0),
        A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.5), p=1.0),
        A.ToGray(p=1.0),
        A.GaussNoise(std_range=(0.04, 0.12), mean_range=(0.0, 0.0), per_channel=True, p=1.0),
        ApplyVignette(p=1.0),
        ApplyFilmGrain(p=1.0),
        ApplyAtmosphericFog(p=1.0),
        A.OneOf(
            [
                A.ImageCompression(compression_type="jpeg", quality_range=(35, 75), p=1.0),
                A.ImageCompression(compression_type="webp", quality_range=(35, 75), p=1.0),
            ],
            p=1.0,
        ),
    ]


def _make_transform(
    size: int | tuple[int, int],
    train: bool,
    progress: float = 1.0,
    advanced_augmentations: bool = False,
    advanced_augment_prob: float = 0.05,
):
    h, w = _as_hw(size)
    if train:
        s = _augment_strength(progress)
        # Stronger curriculum augmentation for full VOC training. Flips, crops,
        # and full-angle rotations stay in the mix throughout the ramp.
        scale_lo = 0.97 - 0.07 * s
        scale_hi = 1.03 + 0.07 * s
        shift = 0.02 + 0.28 * s
        rotate = 2.0 + 358.0 * s
        shear = 0.75 + 14.25 * s
        crop_p = 0.10 + 0.20 * s
        color_jitter = 0.08 + 0.18 * s
        hue = 0.015 + 0.04 * s
        rgb_shift = int(round(4 + 12 * s))
        blur_p = 0.01 + 0.04 * s
        affine_p = 0.15 + 0.20 * s
        color_p = 0.20 + 0.25 * s
        rgb_p = 0.05 + 0.15 * s
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomSizedBBoxSafeCrop(
                height=h,
                width=w,
                erosion_rate=0.0,
                p=crop_p,
            ),
            A.Affine(
                scale=(scale_lo, scale_hi),
                translate_percent={"x": (-shift, shift), "y": (-shift, shift)},
                rotate=(-rotate, rotate),
                shear=(-shear, shear),
                border_mode=cv2.BORDER_REPLICATE,
                p=affine_p,
            ),
            A.ColorJitter(
                brightness=(1.0 - color_jitter, 1.0 + color_jitter),
                contrast=(1.0 - color_jitter, 1.0 + color_jitter),
                saturation=(1.0 - color_jitter, 1.0 + color_jitter),
                hue=(-hue, hue),
                p=color_p,
            ),
            A.RGBShift(r_shift_limit=rgb_shift, g_shift_limit=rgb_shift, b_shift_limit=rgb_shift, p=rgb_p),
        ]
        if advanced_augmentations:
            transforms.append(A.OneOf(_advanced_train_transforms(), p=float(np.clip(advanced_augment_prob, 0.0, 1.0))))
        transforms.extend(
            [
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5, p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MedianBlur(blur_limit=3, p=1.0),
                    ],
                    p=blur_p,
                ),
                A.Resize(height=h, width=w),
            ]
        )
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3, clip=True, filter_invalid_bboxes=True),
        )
    return A.Compose(
        [A.Resize(height=h, width=w)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.0, clip=True, filter_invalid_bboxes=True),
    )


def _transform(
    size: int | tuple[int, int],
    train: bool,
    progress: float = 1.0,
    advanced_augmentations: bool = False,
    advanced_augment_prob: float = 0.05,
):
    cache = getattr(_TRANSFORMS, "cache", None)
    if cache is None:
        cache = {}
        _TRANSFORMS.cache = cache
    strength_key = int(round(_augment_strength(progress) * 1000.0))
    prob_key = int(round(float(np.clip(advanced_augment_prob, 0.0, 1.0)) * 1000.0))
    key = (_as_hw(size), train, strength_key, bool(advanced_augmentations), prob_key)
    if key not in cache:
        cache[key] = _make_transform(
            size,
            train,
            progress=progress,
            advanced_augmentations=advanced_augmentations,
            advanced_augment_prob=advanced_augment_prob,
        )
    return cache[key]


def _prepare_sample(
    sample: DetectionSample,
    size: int | tuple[int, int],
    thermo_levels: int,
    train: bool,
    augment_progress: float = 1.0,
    advanced_augmentations: bool = False,
    advanced_augment_prob: float = 0.05,
):
    del thermo_levels
    image = _cached_rgb_image(str(sample.image_path)).copy()
    boxes = sample.boxes_xyxy.astype(np.float32).tolist()
    class_ids = sample.class_ids.astype(np.int32).tolist()
    out = _transform(
        size,
        train=train,
        progress=augment_progress,
        advanced_augmentations=advanced_augmentations,
        advanced_augment_prob=advanced_augment_prob,
    )(image=image, bboxes=boxes, class_labels=class_ids)
    if train and not out["bboxes"]:
        out = _transform(
            size,
            train=False,
            progress=augment_progress,
            advanced_augmentations=False,
            advanced_augment_prob=advanced_augment_prob,
        )(image=image, bboxes=boxes, class_labels=class_ids)
    image = out["image"].astype(np.float32) / 255.0
    boxes = np.asarray(out["bboxes"], np.float32).reshape(-1, 4)
    class_ids = np.asarray(out["class_labels"], np.int32)
    return image.astype(np.float16), boxes, class_ids


def _resolve_workers(num_workers: int, count: int) -> int:
    if count <= 1:
        return 1
    if num_workers > 0:
        return min(num_workers, count)
    auto = os.cpu_count() or 1
    return max(1, min(count, 4, auto))


def pack_targets(
    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    anchors: np.ndarray,
    input_size: int | tuple[int, int],
    positive_top_k: int = 1,
    max_true_boxes: int = MAX_TRUE_BOXES,
    image_dtype: str | np.dtype = np.float16,
    heads=None,
    strides=None,
    anchor_masks=None,
):
    heads, strides, anchor_masks = _resolve_head_spec(heads, strides, anchor_masks)
    input_h, input_w = _as_hw(input_size)
    input_wh = np.asarray([input_w, input_h], np.float32)
    grids = [_grid_hw((input_h, input_w), s) for s in strides]
    anchors_n = anchors / input_wh
    target_channels = 5 + num_classes
    gt_channels = 5 * max_true_boxes
    targets = {
        head: np.zeros((len(packed), gh, gw, len(mask), target_channels + gt_channels), np.float32)
        for head, (gh, gw), mask in zip(heads, grids, anchor_masks)
    }
    images = np.stack([row[0] for row in packed]).astype(image_dtype)
    for b, (_, boxes_xyxy, class_ids) in enumerate(packed):
        boxes_cxcywh = _xyxy_to_cxcywh(boxes_xyxy, (input_w, input_h))
        gt_pack = np.zeros((max_true_boxes, 5), np.float32)
        count = min(len(boxes_cxcywh), max_true_boxes)
        if count:
            gt_pack[:count, :4] = boxes_cxcywh[:count]
            gt_pack[:count, 4] = 1.0
        gt_flat = gt_pack.reshape(1, 1, 1, -1)
        for head in heads:
            targets[head][b, ..., target_channels:] = gt_flat
        for box, class_id in zip(boxes_cxcywh, class_ids):
            anchor_order = np.argsort(-_iou_wh(box[None, 2:4], anchors_n)[0])[: max(1, int(positive_top_k))]
            for anchor_id in anchor_order.tolist():
                for head, (grid_h, grid_w), mask in zip(heads, grids, anchor_masks):
                    if anchor_id not in mask:
                        continue
                    a = mask.index(anchor_id)
                    gx = min(grid_w - 1, int(box[0] * grid_w))
                    gy = min(grid_h - 1, int(box[1] * grid_h))
                    cell = targets[head][b, gy, gx, a, :target_channels]
                    if cell[4] and cell[2] * cell[3] >= box[2] * box[3]:
                        break
                    cell[:] = 0.0
                    cell[:4], cell[4], cell[5 + int(class_id)] = box, 1.0, 1.0
                    break
    return images, targets


def build_targets(
    samples: list[DetectionSample],
    num_classes: int,
    anchors: np.ndarray,
    input_size: int | tuple[int, int],
    thermo_levels: int = 1,
    augment: bool = False,
    positive_top_k: int = 1,
    max_true_boxes: int = MAX_TRUE_BOXES,
    num_workers: int = 0,
    image_dtype: str | np.dtype = np.float16,
    augment_progress: float = 1.0,
    advanced_augmentations: bool = False,
    advanced_augment_prob: float = 0.05,
    heads=None,
    strides=None,
    anchor_masks=None,
):
    worker_count = _resolve_workers(num_workers, len(samples))
    if worker_count == 1:
        packed = [
            _prepare_sample(
                sample,
                input_size,
                thermo_levels,
                train=augment,
                augment_progress=augment_progress,
                advanced_augmentations=advanced_augmentations if augment else False,
                advanced_augment_prob=advanced_augment_prob,
            )
            for sample in samples
        ]
    else:
        prepare = partial(
            _prepare_sample,
            size=input_size,
            thermo_levels=thermo_levels,
            train=augment,
            augment_progress=augment_progress,
            advanced_augmentations=advanced_augmentations if augment else False,
            advanced_augment_prob=advanced_augment_prob,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            packed = list(pool.map(prepare, samples))
    return pack_targets(
        packed,
        num_classes,
        anchors,
        input_size,
        positive_top_k=positive_top_k,
        max_true_boxes=max_true_boxes,
        image_dtype=image_dtype,
        heads=heads,
        strides=strides,
        anchor_masks=anchor_masks,
    )


def _int8_sat_ops(x, output_scale: float):
    x = ops.cast(x, "float32") / ops.cast(float(output_scale), "float32")
    return ops.clip(x, -127.0, 127.0)


def _scaled_logits_ops(x, output_scale: float, logit_scale: float):
    x = ops.cast(x, "float32") / ops.cast(float(output_scale * logit_scale), "float32")
    return x


def _uint01_ops(x, output_scale: float):
    x = _int8_sat_ops(x, output_scale)
    return (x + ops.cast(127.0, "float32")) / ops.cast(254.0, "float32")


def _piecewise_ratio_from_u_ops(u):
    u = ops.cast(u, "float16")
    return ops.where(u <= 0.5, 2.0 * u, 1.0 + 4.0 * (u - 0.5))


def _piecewise_u_from_ratio_ops(ratio):
    ratio = ops.cast(ratio, "float16")
    return ops.where(ratio <= 1.0, 0.5 * ratio, 0.5 + 0.25 * (ratio - 1.0))


def make_loss(
    anchors: np.ndarray,
    input_size: int | tuple[int, int],
    grid: tuple[int, int],
    num_classes: int,
    obj_gain: float = 1.0,
    obj_pos_gain: float = 1.0,
    noobj_weight: float = 0.2,
    cls_neg_weight: float = 0.0,
    cls_gain: float = 1.0,
    class_mode: str = "bce",
    box_gain: float = 5.0,
    box_iou_aux_gain: float = 0.0,
    box_iou_aux_mode: str = "iou",
    bbox_output_scale: float = 2.0,
    obj_output_scale: float = 32.0,
    cls_output_scale: float = 32.0,
    logit_scale: float = 32.0,
    ignore_iou: float = 0.5,
    rescore_objectness: bool = True,
    hard_negative_ratio: float = 0.0,
    hard_negative_min: int = 0,
    conf_neg_normalizer: str = "neg",
    box_mode: str = "linear",
    max_true_boxes: int = MAX_TRUE_BOXES,
):
    input_h, input_w = _as_hw(input_size)
    grid_h, grid_w = grid
    gy, gx = np.meshgrid(np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32), indexing="ij")
    grid_xy = np.stack([gx, gy], axis=-1).reshape(1, grid_h, grid_w, 1, 2)
    grid_s = np.asarray([grid_w, grid_h], np.float32).reshape(1, 1, 1, 1, 2)
    anchor_wh = (anchors / np.asarray([input_w, input_h], np.float32)).reshape(1, 1, 1, len(anchors), 2)
    target_channels = 5 + num_classes

    def loss(y_true, y_pred):
        y_true, y_pred = ops.cast(y_true, "float32"), ops.cast(y_pred, "float32")
        target = y_true[..., :target_channels]
        gt_pack = ops.reshape(y_true[:, 0, 0, 0, target_channels:], (-1, max_true_boxes, 5))
        gt_boxes, gt_valid = gt_pack[..., :4], gt_pack[..., 4]
        obj, box, cls = target[..., 4:5], target[..., :4], target[..., 5:5 + num_classes]
        xy_t = ops.clip(box[..., :2] * grid_s - grid_xy, 0.0, 1.0)
        if box_mode == "yolo":
            ratio = ops.maximum(box[..., 2:4] / ops.maximum(anchor_wh, 1e-6), 1e-6)
            wh_t = ops.clip(ops.log(ratio), -6.0, 6.0)
            xy_p = ops.cast(ops.sigmoid(y_pred[..., :2]), "float32")
            wh_p = ops.cast(ops.clip(y_pred[..., 2:4], -6.0, 6.0), "float32")
            obj_logits = ops.cast(y_pred[..., 4:5], "float32") / ops.cast(float(logit_scale), "float32")
            cls_logits = ops.cast(y_pred[..., 5:5 + num_classes], "float32") / ops.cast(float(logit_scale), "float32")
            pred_xy = (xy_p + ops.cast(grid_xy, "float32")) / ops.cast(grid_s, "float32")
            pred_wh = ops.cast(ops.exp(wh_p), "float32") * ops.cast(anchor_wh, "float32")
        elif box_mode == "linear_float":
            ratio = ops.clip(box[..., 2:4] / ops.maximum(anchor_wh, 1e-6), 0.0, 3.0)
            wh_t = ops.clip(_piecewise_u_from_ratio_ops(ratio), 0.0, 1.0)
            xy_p = ops.cast(ops.clip(y_pred[..., :2], 0.0, 1.0), "float32")
            wh_p = ops.cast(ops.clip(y_pred[..., 2:4], 0.0, 1.0), "float32")
            obj_logits = ops.cast(y_pred[..., 4:5], "float32") / ops.cast(float(logit_scale), "float32")
            cls_logits = ops.cast(y_pred[..., 5:5 + num_classes], "float32") / ops.cast(float(logit_scale), "float32")
            pred_xy = (xy_p + ops.cast(grid_xy, "float32")) / ops.cast(grid_s, "float32")
            pred_wh = ops.cast(_piecewise_ratio_from_u_ops(wh_p), "float32") * ops.cast(anchor_wh, "float32")
        else:
            ratio = ops.clip(box[..., 2:4] / ops.maximum(anchor_wh, 1e-6), 0.0, 3.0)
            wh_t = ops.clip(_piecewise_u_from_ratio_ops(ratio), 0.0, 1.0)
            xy_p = ops.cast(_uint01_ops(y_pred[..., :2], bbox_output_scale), "float32")
            wh_p = ops.cast(_uint01_ops(y_pred[..., 2:4], bbox_output_scale), "float32")
            obj_logits = _scaled_logits_ops(y_pred[..., 4:5], obj_output_scale, logit_scale)
            cls_logits = _scaled_logits_ops(y_pred[..., 5:5 + num_classes], cls_output_scale, logit_scale)
            pred_xy = (xy_p + ops.cast(grid_xy, "float32")) / ops.cast(grid_s, "float32")
            pred_wh = ops.cast(_piecewise_ratio_from_u_ops(_uint01_ops(y_pred[..., 2:4], bbox_output_scale)), "float32") * ops.cast(anchor_wh, "float32")
        pred_xyxy = ops.concatenate([pred_xy - pred_wh * 0.5, pred_xy + pred_wh * 0.5], axis=-1)
        target_xyxy = ops.concatenate([box[..., :2] - box[..., 2:4] * 0.5, box[..., :2] + box[..., 2:4] * 0.5], axis=-1)
        gt_xyxy = ops.concatenate([gt_boxes[..., :2] - gt_boxes[..., 2:4] * 0.5, gt_boxes[..., :2] + gt_boxes[..., 2:4] * 0.5], axis=-1)
        pair_iou = _pairwise_iou_xyxy_ops(
            ops.reshape(pred_xyxy, (ops.shape(y_pred)[0], -1, 4)),
            gt_xyxy,
            gt_valid,
        )
        best_iou = ops.max(pair_iou, axis=-1, keepdims=True)
        ignore_mask = ops.reshape(
            ops.cast(best_iou < ignore_iou, "float32"),
            (ops.shape(y_pred)[0], grid_h, grid_w, len(anchors), 1),
        )
        box_w = 2.0 - box[..., 2:3] * box[..., 3:4]
        xy = ops.where(
            ops.abs(xy_p - xy_t) < 1.0,
            0.5 * ops.square(xy_p - xy_t),
            ops.abs(xy_p - xy_t) - 0.5,
        ) * obj * box_w
        wh = ops.where(
            ops.abs(wh_p - wh_t) < 1.0,
            0.5 * ops.square(wh_p - wh_t),
            ops.abs(wh_p - wh_t) - 0.5,
        ) * obj * box_w
        tl = ops.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
        br = ops.minimum(pred_xyxy[..., 2:4], target_xyxy[..., 2:4])
        inter = ops.prod(ops.maximum(br - tl, 0.0), axis=-1, keepdims=True)
        area_p = ops.prod(ops.maximum(pred_xyxy[..., 2:4] - pred_xyxy[..., :2], 0.0), axis=-1, keepdims=True)
        area_t = ops.prod(ops.maximum(target_xyxy[..., 2:4] - target_xyxy[..., :2], 0.0), axis=-1, keepdims=True)
        assigned_iou = inter / ops.maximum(area_p + area_t - inter, 1e-6)
        if box_iou_aux_gain > 0.0:
            if box_iou_aux_mode == "diou":
                center_dist = ops.sum(ops.square(pred_xy - box[..., :2]), axis=-1, keepdims=True)
                enc_tl = ops.minimum(pred_xyxy[..., :2], target_xyxy[..., :2])
                enc_br = ops.maximum(pred_xyxy[..., 2:4], target_xyxy[..., 2:4])
                enc_wh = ops.maximum(enc_br - enc_tl, 0.0)
                enc_diag = ops.maximum(ops.sum(ops.square(enc_wh), axis=-1, keepdims=True), 1e-6)
                box_iou_aux = (1.0 - assigned_iou) + center_dist / enc_diag
            else:
                box_iou_aux = 1.0 - assigned_iou
        else:
            box_iou_aux = None
        obj_target = obj * (ops.stop_gradient(assigned_iou) if rescore_objectness else 1.0)
        pos_count = ops.maximum(ops.sum(obj), 1.0)
        neg_mask = (1.0 - obj) * ignore_mask
        neg_loss = _bce_logits(ops.zeros_like(obj_logits), obj_logits)
        neg_mask = _hard_negative_mask(neg_mask, neg_loss, pos_count, hard_negative_ratio, hard_negative_min)
        if conf_neg_normalizer == "pos":
            neg_norm = pos_count
        elif conf_neg_normalizer == "none":
            neg_norm = 1.0
        else:
            neg_norm = ops.maximum(ops.sum(neg_mask), 1.0)
        conf_pos = obj_gain * obj_pos_gain * ops.sum(obj * _bce_logits(obj_target, obj_logits)) / pos_count
        conf_neg = obj_gain * noobj_weight * ops.sum(neg_mask * neg_loss) / neg_norm
        if class_mode == "softmax":
            cls_pos = ops.expand_dims(_softmax_ce(cls, cls_logits), axis=-1)
            cls_pos_l = ops.sum(cls_pos * obj) / pos_count
            cls_neg_l = 0.0
        else:
            cls_pos_l = ops.sum(ops.sum(_bce_logits(cls, cls_logits), axis=-1, keepdims=True) * obj) / pos_count
            if cls_neg_weight > 0.0:
                cls_neg_l = ops.sum(ops.sum(_bce_logits(ops.zeros_like(cls_logits), cls_logits), axis=-1, keepdims=True) * neg_mask) / neg_norm
            else:
                cls_neg_l = 0.0
        cls_l = cls_gain * (cls_pos_l + cls_neg_weight * cls_neg_l)
        box_l = box_gain * (ops.sum(xy) + ops.sum(wh)) / pos_count
        if box_iou_aux is not None:
            box_l = box_l + box_iou_aux_gain * ops.sum(obj * box_iou_aux) / pos_count
        return box_l + conf_pos + conf_neg + cls_l

    return loss


def make_losses(
    anchors: np.ndarray,
    input_size: int | tuple[int, int],
    num_classes: int,
    obj_gain: float = 1.0,
    obj_pos_gain: float = 1.0,
    noobj_weight: float = 0.2,
    cls_neg_weight: float = 0.0,
    cls_gain: float = 1.0,
    class_mode: str = "bce",
    box_gain: float = 5.0,
    box_iou_aux_gain: float = 0.0,
    box_iou_aux_mode: str = "iou",
    bbox_output_scale: float = 2.0,
    obj_output_scale: float = 32.0,
    cls_output_scale: float = 32.0,
    logit_scale: float = 32.0,
    ignore_iou: float = 0.5,
    rescore_objectness: bool = True,
    hard_negative_ratio: float = 0.0,
    hard_negative_min: int = 0,
    hard_negative_mins: tuple[int, int, int] | None = None,
    conf_neg_normalizer: str = "neg",
    box_mode: str = "linear",
    max_true_boxes: int = MAX_TRUE_BOXES,
    heads=None,
    strides=None,
    anchor_masks=None,
):
    heads, strides, anchor_masks = _resolve_head_spec(heads, strides, anchor_masks)
    if hard_negative_mins is None:
        hard_negative_mins = (hard_negative_min,) * len(heads)
    input_hw = _as_hw(input_size)
    return {
        head: make_loss(
            anchors[np.asarray(mask, np.int32)],
            input_hw,
            _grid_hw(input_hw, stride),
            num_classes,
            obj_gain=obj_gain,
            obj_pos_gain=obj_pos_gain,
            noobj_weight=noobj_weight,
            cls_neg_weight=cls_neg_weight,
            cls_gain=cls_gain,
            class_mode=class_mode,
            box_gain=box_gain,
            box_iou_aux_gain=box_iou_aux_gain,
            box_iou_aux_mode=box_iou_aux_mode,
            bbox_output_scale=bbox_output_scale,
            obj_output_scale=obj_output_scale,
            cls_output_scale=cls_output_scale,
            logit_scale=logit_scale,
            ignore_iou=ignore_iou,
            rescore_objectness=rescore_objectness,
            hard_negative_ratio=hard_negative_ratio,
            hard_negative_min=hard_negative_mins[idx],
            conf_neg_normalizer=conf_neg_normalizer,
            box_mode=box_mode,
            max_true_boxes=max_true_boxes,
        )
        for idx, (head, mask, stride) in enumerate(zip(heads, anchor_masks, strides))
    }


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x, dtype=np.float32)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-6)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _int8_sat_np(x: np.ndarray, output_scale: float) -> np.ndarray:
    x = np.asarray(x, np.float32) / np.float32(output_scale)
    return np.clip(x, -127.0, 127.0).astype(np.float32)


def _uint01_np(x: np.ndarray, output_scale: float) -> np.ndarray:
    x = _int8_sat_np(x, output_scale)
    return ((x + 127.0) / 254.0).astype(np.float32)


def _piecewise_ratio_from_u_np(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, np.float32)
    return np.where(u <= 0.5, 2.0 * u, 1.0 + 4.0 * (u - 0.5)).astype(np.float32)


def _nms_single(boxes: np.ndarray, scores: np.ndarray, iou: float, max_boxes: int | None = None) -> np.ndarray:
    order = np.argsort(-scores)
    kept = []
    while len(order):
        kept.append(order[0])
        if (max_boxes and len(kept) >= max_boxes) or len(order) == 1:
            break
        a, b = boxes[order[0]], boxes[order[1:]]
        inter = np.maximum(0.0, np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0])) * np.maximum(
            0.0, np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1])
        )
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
        order = order[1:][inter / np.maximum(union, 1e-6) < iou]
    return np.asarray(kept, np.int32)


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float, max_boxes: int | None = None, per_class: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(boxes):
        empty = np.zeros((0,), np.float32)
        return empty.reshape(0, 4), empty, empty.astype(np.int32)
    if not per_class:
        kept = _nms_single(boxes, scores, iou, max_boxes=max_boxes)
        return boxes[kept], scores[kept], classes[kept]
    kept = []
    for cls in np.unique(classes):
        idx = np.flatnonzero(classes == cls)
        sub_kept = _nms_single(boxes[idx], scores[idx], iou, max_boxes=max_boxes)
        if len(sub_kept):
            kept.append(idx[sub_kept])
    if not kept:
        empty = np.zeros((0,), np.float32)
        return empty.reshape(0, 4), empty, empty.astype(np.int32)
    kept = np.concatenate(kept)
    kept = kept[np.argsort(-scores[kept])]
    if max_boxes:
        kept = kept[:max_boxes]
    return boxes[kept], scores[kept], classes[kept]


def predict_boxes(
    preds: dict[str, np.ndarray],
    anchors: np.ndarray,
    input_size: int | tuple[int, int],
    score: float = 0.25,
    objectness: float = 0.0,
    iou: float = 0.45,
    index: int = 0,
    bbox_output_scale: float = 2.0,
    obj_output_scale: float = 32.0,
    cls_output_scale: float = 32.0,
    logit_scale: float = 32.0,
    max_boxes: int | None = None,
    per_class_nms: bool = True,
    class_mode: str = "bce",
    box_mode: str = "linear",
    heads=None,
    anchor_masks=None,
):
    heads, _, anchor_masks = _resolve_head_spec(heads, None, anchor_masks)
    input_h, input_w = _as_hw(input_size)
    input_wh = np.asarray([input_w, input_h], np.float32)
    boxes_all, scores_all, cls_all = [], [], []
    for head, mask in zip(heads, anchor_masks):
        pred = preds[head][index].astype(np.float32)
        h, w = pred.shape[:2]
        gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        grid = np.stack([gx, gy], axis=-1).reshape(h, w, 1, 2)
        if box_mode == "yolo":
            xy = (_sigmoid_np(pred[..., :2]) + grid) / np.asarray([w, h], np.float32)
            wh = np.exp(np.clip(pred[..., 2:4], -6.0, 6.0).astype(np.float32))
            wh = wh * anchors[np.asarray(mask, np.int32)].astype(np.float32).reshape(1, 1, -1, 2) / input_wh
            obj = _sigmoid_np(pred[..., 4:5] / logit_scale)[..., 0]
            cls_logits = pred[..., 5:] / logit_scale
        elif box_mode == "linear_float":
            xy_u = np.clip(pred[..., :2], 0.0, 1.0).astype(np.float32)
            wh_u = np.clip(pred[..., 2:4], 0.0, 1.0).astype(np.float32)
            xy = (xy_u + grid) / np.asarray([w, h], np.float32)
            ratio = _piecewise_ratio_from_u_np(wh_u)
            wh = ratio * anchors[np.asarray(mask, np.int32)].astype(np.float32).reshape(1, 1, -1, 2) / input_wh
            obj = _sigmoid_np(pred[..., 4:5] / logit_scale)[..., 0]
            cls_logits = pred[..., 5:] / logit_scale
        else:
            xy = (_uint01_np(pred[..., :2], bbox_output_scale) + grid) / np.asarray([w, h], np.float32)
            ratio = _piecewise_ratio_from_u_np(_uint01_np(pred[..., 2:4], bbox_output_scale))
            wh = ratio * anchors[np.asarray(mask, np.int32)].astype(np.float32).reshape(1, 1, -1, 2) / input_wh
            obj = _sigmoid_np(_int8_sat_np(pred[..., 4:5], obj_output_scale) / logit_scale)[..., 0]
            cls_logits = _int8_sat_np(pred[..., 5:], cls_output_scale) / logit_scale
        cls_prob = _softmax_np(cls_logits) if class_mode == "softmax" else _sigmoid_np(cls_logits)
        cls_id = np.argmax(cls_prob, axis=-1)
        cls_score = np.take_along_axis(cls_prob, cls_id[..., None], axis=-1)[..., 0]
        keep = (obj >= objectness) & (obj * cls_score >= score)
        if not np.any(keep):
            continue
        xy, wh = xy[keep].astype(np.float32), wh[keep].astype(np.float32)
        boxes_all.append(np.concatenate([xy - wh * 0.5, xy + wh * 0.5], axis=-1))
        scores_all.append((obj * cls_score)[keep])
        cls_all.append(cls_id[keep].astype(np.int32))
    if not boxes_all:
        empty = np.zeros((0,), np.float32)
        return empty.reshape(0, 4), empty, empty.astype(np.int32)
    return _nms(np.concatenate(boxes_all), np.concatenate(scores_all), np.concatenate(cls_all), iou, max_boxes=max_boxes, per_class=per_class_nms)


def _norm_xyxy(boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    scaled = boxes.astype(np.float32).copy()
    scaled[:, [0, 2]] /= max(w, 1)
    scaled[:, [1, 3]] /= max(h, 1)
    return scaled


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(br - tl, 0.0), axis=-1)
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0.0), axis=-1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0.0), axis=-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def map50(
    samples: list[DetectionSample],
    preds: dict[str, np.ndarray],
    anchors: np.ndarray,
    input_size: int | tuple[int, int],
    num_classes: int,
    score: float = 0.05,
    objectness: float = 0.0,
    iou: float = 0.5,
    bbox_output_scale: float = 2.0,
    obj_output_scale: float = 32.0,
    cls_output_scale: float = 32.0,
    logit_scale: float = 32.0,
    max_boxes: int | None = None,
    per_class_nms: bool = True,
    class_mode: str = "bce",
    box_mode: str = "linear",
    heads=None,
    anchor_masks=None,
) -> float:
    heads, _, anchor_masks = _resolve_head_spec(heads, None, anchor_masks)
    detections = {cls: [] for cls in range(num_classes)}
    gt_by_class = {cls: {} for cls in range(num_classes)}
    for image_id, sample in enumerate(samples):
        gt_boxes = _norm_xyxy(sample.boxes_xyxy, sample.size)
        for cls in range(num_classes):
            mask = sample.class_ids == cls
            gt_by_class[cls][image_id] = {"boxes": gt_boxes[mask], "used": np.zeros(np.sum(mask), bool)}
        boxes, scores, class_ids = predict_boxes(
            preds,
            anchors,
            input_size,
            score=score,
            objectness=objectness,
            iou=0.45,
            index=image_id,
            bbox_output_scale=bbox_output_scale,
            obj_output_scale=obj_output_scale,
            cls_output_scale=cls_output_scale,
            logit_scale=logit_scale,
            max_boxes=max_boxes,
            per_class_nms=per_class_nms,
            class_mode=class_mode,
            box_mode=box_mode,
            heads=heads,
            anchor_masks=anchor_masks,
        )
        for box, conf, cls in zip(boxes, scores, class_ids):
            detections[int(cls)].append((image_id, float(conf), box.astype(np.float32)))
    aps = []
    for cls in range(num_classes):
        total_gt = sum(len(gt_by_class[cls][image_id]["boxes"]) for image_id in gt_by_class[cls])
        if not total_gt:
            continue
        dets = sorted(detections[cls], key=lambda row: -row[1])
        tp, fp = np.zeros(len(dets), np.float32), np.zeros(len(dets), np.float32)
        for i, (image_id, _, box) in enumerate(dets):
            gt = gt_by_class[cls][image_id]
            if not len(gt["boxes"]):
                fp[i] = 1.0
                continue
            overlaps = _box_iou(box[None, :], gt["boxes"])[0]
            j = int(np.argmax(overlaps))
            if overlaps[j] >= iou and not gt["used"][j]:
                gt["used"][j] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        aps.append(_average_precision(tp / total_gt, tp / np.maximum(tp + fp, 1e-6)))
    return float(np.mean(aps)) if aps else 0.0


def draw_boxes(image: np.ndarray, path: Path, boxes: np.ndarray, class_ids: np.ndarray, class_names: list[str], scores=None, color="red"):
    if image.shape[-1] > 3 and image.shape[-1] % 3 == 0:
        image = image.reshape(image.shape[0], image.shape[1], 3, image.shape[-1] // 3).mean(axis=-1)
    canvas = Image.fromarray(np.asarray(np.clip(image[..., :3] * 255.0, 0, 255), np.uint8))
    draw, w, h = ImageDraw.Draw(canvas), *canvas.size
    if boxes.size and np.max(boxes[:, 2:4]) <= 1.5:
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = class_names[int(class_id)]
        if scores is not None:
            text = f"{text} {float(scores[i]):.2f}"
        draw.text((x1 + 2, y1 + 2), text, fill=color)
    canvas.save(path)
