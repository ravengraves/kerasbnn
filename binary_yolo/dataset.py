from __future__ import annotations

import math
import time

import keras
import numpy as np

from binary_yolo.yad2k_core import DetectionSample, MAX_TRUE_BOXES, _prepare_sample, build_targets, pack_targets


class DetectionPyDataset(keras.utils.PyDataset):
    def __init__(
        self,
        samples: list[DetectionSample],
        num_classes: int,
        anchors: np.ndarray,
        input_size: int | tuple[int, int],
        thermo_levels: int = 1,
        augment: bool = False,
        batch_size: int = 16,
        shuffle: bool = False,
        seed: int = 7,
        max_true_boxes: int = MAX_TRUE_BOXES,
        positive_top_k: int = 1,
        workers: int = 8,
        use_multiprocessing: bool = False,
        max_queue_size: int = 32,
        image_dtype: str | np.dtype = np.float16,
        augment_start_strength: float = 0.2,
        augment_ramp_epochs: int = 100,
        augment_constant_strength: bool = False,
        advanced_augmentations: bool = False,
        advanced_augment_prob: float = 0.05,
        advanced_augment_ramp_epochs: int = 0,
        heads=None,
        strides=None,
        anchor_masks=None,
    ):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.samples = list(samples)
        self.num_classes = num_classes
        self.anchors = np.asarray(anchors, np.float32)
        self.input_size = input_size
        self.thermo_levels = int(thermo_levels)
        self.augment = bool(augment)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_true_boxes = int(max_true_boxes)
        self.positive_top_k = int(positive_top_k)
        self.image_dtype = image_dtype
        self.augment_start_strength = float(augment_start_strength)
        self.augment_ramp_epochs = max(1, int(augment_ramp_epochs))
        self.augment_constant_strength = bool(augment_constant_strength)
        self.advanced_augmentations = bool(advanced_augmentations)
        self.advanced_augment_prob = float(advanced_augment_prob)
        self.advanced_augment_ramp_epochs = int(advanced_augment_ramp_epochs)
        self.heads = heads
        self.strides = strides
        self.anchor_masks = anchor_masks
        self._rng = np.random.default_rng(self.seed)
        self._order = np.arange(len(self.samples), dtype=np.int32)
        self._epoch_index = 0
        if self.shuffle:
            self._rng.shuffle(self._order)

    def _augment_progress(self) -> float:
        if not self.augment:
            return 0.0
        if self.augment_constant_strength:
            return 1.0
        progress = min(1.0, float(self._epoch_index) / float(self.augment_ramp_epochs))
        return self.augment_start_strength + (1.0 - self.augment_start_strength) * progress

    def _current_advanced_augment_prob(self) -> float:
        if not self.augment or not self.advanced_augmentations or self.advanced_augment_prob <= 0.0:
            return 0.0
        ramp_epochs = self.advanced_augment_ramp_epochs if self.advanced_augment_ramp_epochs > 0 else self.augment_ramp_epochs
        progress = min(1.0, float(self._epoch_index) / float(max(1, ramp_epochs)))
        return self.advanced_augment_prob * progress

    def __len__(self):
        return max(1, math.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = min(len(self.samples), start + self.batch_size)
        batch_ids = self._order[start:stop]
        augment_progress = self._augment_progress()
        packed = [
            _prepare_sample(
                self.samples[int(i)],
                self.input_size,
                self.thermo_levels,
                train=self.augment,
                augment_progress=augment_progress,
                advanced_augmentations=self.advanced_augmentations,
                advanced_augment_prob=self._current_advanced_augment_prob(),
            )
            for i in batch_ids
        ]
        return pack_targets(
            packed,
            self.num_classes,
            self.anchors,
            self.input_size,
            positive_top_k=self.positive_top_k,
            max_true_boxes=self.max_true_boxes,
            image_dtype=self.image_dtype,
            heads=self.heads,
            strides=self.strides,
            anchor_masks=self.anchor_masks,
        )

    def on_epoch_end(self):
        self._epoch_index += 1
        if self.shuffle:
            self._rng.shuffle(self._order)

    def sample_at(self, index: int):
        sample = self.samples[index]
        image, boxes, class_ids = _prepare_sample(sample, self.input_size, self.thermo_levels, train=False)
        return image, boxes, class_ids, sample


FruitDetectionPyDataset = DetectionPyDataset


class CachedDetectionDataset(keras.utils.PyDataset):
    def __init__(
        self,
        samples: list[DetectionSample],
        num_classes: int,
        anchors: np.ndarray,
        input_size: int | tuple[int, int],
        thermo_levels: int = 1,
        augment: bool = False,
        batch_size: int = 16,
        shuffle: bool = False,
        seed: int = 7,
        max_true_boxes: int = MAX_TRUE_BOXES,
        positive_top_k: int = 1,
        rebuild_every: int = 4,
        refresh_divisor: int = 0,
        build_workers: int = 8,
        image_dtype: str | np.dtype = np.float16,
        augment_start_strength: float = 0.2,
        augment_ramp_epochs: int = 100,
        augment_constant_strength: bool = False,
        advanced_augmentations: bool = False,
        advanced_augment_prob: float = 0.05,
        advanced_augment_ramp_epochs: int = 0,
        heads=None,
        strides=None,
        anchor_masks=None,
    ):
        super().__init__()
        self.samples = list(samples)
        self.num_classes = int(num_classes)
        self.anchors = np.asarray(anchors, np.float32)
        self.input_size = input_size
        self.thermo_levels = int(thermo_levels)
        self.augment = bool(augment)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_true_boxes = int(max_true_boxes)
        self.positive_top_k = int(positive_top_k)
        self.rebuild_every = max(1, int(rebuild_every))
        self.refresh_divisor = int(refresh_divisor)
        self.build_workers = int(build_workers)
        self.image_dtype = image_dtype
        self.augment_start_strength = float(augment_start_strength)
        self.augment_ramp_epochs = max(1, int(augment_ramp_epochs))
        self.augment_constant_strength = bool(augment_constant_strength)
        self.advanced_augmentations = bool(advanced_augmentations)
        self.advanced_augment_prob = float(advanced_augment_prob)
        self.advanced_augment_ramp_epochs = int(advanced_augment_ramp_epochs)
        self.heads = heads
        self.strides = strides
        self.anchor_masks = anchor_masks
        self._rng = np.random.default_rng(self.seed)
        self._epoch_index = 0
        self._refresh_cursor = 0
        self.base_images = None
        self.base_targets = None
        self.images = None
        self.targets = None
        self._order = np.arange(len(self.samples), dtype=np.int32)
        self._profile = {
            "getitem_calls": 0,
            "getitem_seconds": 0.0,
            "refresh_full_calls": 0,
            "refresh_full_seconds": 0.0,
            "refresh_partial_calls": 0,
            "refresh_partial_seconds": 0.0,
            "refresh_partial_items": 0,
        }
        self._build_base()
        self.refresh_full()

    def _augment_progress(self) -> float:
        if not self.augment:
            return 0.0
        if self.augment_constant_strength:
            return 1.0
        progress = min(1.0, float(self._epoch_index) / float(self.augment_ramp_epochs))
        return self.augment_start_strength + (1.0 - self.augment_start_strength) * progress

    def _current_advanced_augment_prob(self) -> float:
        if not self.augment or not self.advanced_augmentations or self.advanced_augment_prob <= 0.0:
            return 0.0
        ramp_epochs = self.advanced_augment_ramp_epochs if self.advanced_augment_ramp_epochs > 0 else self.augment_ramp_epochs
        progress = min(1.0, float(self._epoch_index) / float(max(1, ramp_epochs)))
        return self.advanced_augment_prob * progress

    def _build_base(self):
        self.base_images, self.base_targets = build_targets(
            self.samples,
            self.num_classes,
            self.anchors,
            self.input_size,
            thermo_levels=self.thermo_levels,
            augment=False,
            positive_top_k=self.positive_top_k,
            max_true_boxes=self.max_true_boxes,
            num_workers=self.build_workers,
            image_dtype=self.image_dtype,
            heads=self.heads,
            strides=self.strides,
            anchor_masks=self.anchor_masks,
            advanced_augmentations=False,
            advanced_augment_prob=self.advanced_augment_prob,
        )

    def refresh_full(self):
        started = time.perf_counter()
        if self.augment:
            augment_progress = self._augment_progress()
            self.images, self.targets = build_targets(
                self.samples,
                self.num_classes,
                self.anchors,
                self.input_size,
                thermo_levels=self.thermo_levels,
                augment=True,
                positive_top_k=self.positive_top_k,
                max_true_boxes=self.max_true_boxes,
                num_workers=self.build_workers,
                image_dtype=self.image_dtype,
                augment_progress=augment_progress,
                advanced_augmentations=self.advanced_augmentations,
                advanced_augment_prob=self._current_advanced_augment_prob(),
                heads=self.heads,
                strides=self.strides,
                anchor_masks=self.anchor_masks,
            )
        else:
            self.images = self.base_images.copy()
            self.targets = {name: values.copy() for name, values in self.base_targets.items()}
        self._order = np.arange(int(self.images.shape[0]), dtype=np.int32)
        if self.shuffle:
            self._rng.shuffle(self._order)
        self._profile["refresh_full_calls"] += 1
        self._profile["refresh_full_seconds"] += time.perf_counter() - started

    def refresh_partial(self, index_ids):
        if not self.augment:
            return
        started = time.perf_counter()
        index_ids = np.asarray(index_ids, dtype=np.int32)
        if index_ids.size == 0:
            return
        augment_progress = self._augment_progress()
        sub_samples = [self.samples[int(i)] for i in index_ids.tolist()]
        sub_images, sub_targets = build_targets(
            sub_samples,
            self.num_classes,
            self.anchors,
            self.input_size,
            thermo_levels=self.thermo_levels,
            augment=True,
            positive_top_k=self.positive_top_k,
            max_true_boxes=self.max_true_boxes,
            num_workers=self.build_workers,
            image_dtype=self.image_dtype,
            augment_progress=augment_progress,
            advanced_augmentations=self.advanced_augmentations,
            advanced_augment_prob=self._current_advanced_augment_prob(),
            heads=self.heads,
            strides=self.strides,
            anchor_masks=self.anchor_masks,
        )
        self.images[index_ids] = sub_images
        for name, values in sub_targets.items():
            self.targets[name][index_ids] = values
        self._profile["refresh_partial_calls"] += 1
        self._profile["refresh_partial_seconds"] += time.perf_counter() - started
        self._profile["refresh_partial_items"] += int(index_ids.size)

    def __len__(self):
        return max(1, math.ceil(int(self.images.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        started = time.perf_counter()
        start = index * self.batch_size
        stop = min(start + self.batch_size, int(self.images.shape[0]))
        batch_ids = self._order[start:stop]
        batch = self.images[batch_ids], {name: values[batch_ids] for name, values in self.targets.items()}
        self._profile["getitem_calls"] += 1
        self._profile["getitem_seconds"] += time.perf_counter() - started
        return batch

    def on_epoch_end(self):
        self._epoch_index += 1
        if self.augment:
            if self.refresh_divisor > 0:
                count = max(1, int(math.ceil(len(self.samples) / self.refresh_divisor)))
                start = self._refresh_cursor
                stop = start + count
                ids = np.arange(start, stop, dtype=np.int32) % len(self.samples)
                self.refresh_partial(ids)
                self._refresh_cursor = int(stop % len(self.samples))
            elif self.rebuild_every > 0 and (self._epoch_index % self.rebuild_every) == 0:
                self.refresh_full()
                return
        if self.shuffle:
            self._rng.shuffle(self._order)

    def sample_at(self, index: int):
        sample = self.samples[index]
        image, boxes, class_ids = _prepare_sample(sample, self.input_size, self.thermo_levels, train=False)
        return image, boxes, class_ids, sample

    def consume_profile(self):
        profile = dict(self._profile)
        for key in self._profile:
            self._profile[key] = 0 if key.endswith("_calls") or key.endswith("_items") else 0.0
        return profile
