import logging
import math
import sqlite3

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)


class _DetectionDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        targets: list[dict],
        processor,
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = self.targets[idx]

        # DETR processor expects COCO-style annotations:
        # annotations: list of {"bbox": [x, y, w, h] in pixels, "category_id": int, "area": float}
        w_px, h_px = image.size
        coco_annotations = []
        for box in target["boxes"]:
            x_tl, y_tl, bw, bh = box  # normalized top-left coords
            x_px = x_tl * w_px
            y_px = y_tl * h_px
            bw_px = bw * w_px
            bh_px = bh * h_px
            coco_annotations.append({
                "bbox": [x_px, y_px, bw_px, bh_px],
                "category_id": target["labels"][len(coco_annotations)],
                "area": bw_px * bh_px,
                "iscrowd": 0,
            })

        coco_target = {
            "image_id": idx,
            "annotations": coco_annotations,
        }

        encoding = self.processor(
            images=image,
            annotations=[coco_target],
            return_tensors="pt",
        )
        # Remove batch dimension from pixel_values (processor adds [1, C, H, W])
        pixel_values = encoding["pixel_values"][0]
        # Labels are a list of dicts (one per image); tensors are already
        # correctly shaped (class_labels: [N], boxes: [N,4]) with no batch dim.
        labels = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": labels}


def _collate_fn(batch):
    """Custom collate for variable-length detection targets."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


class KumoDetectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        db_path: str,
        class_names: list[str],
        model_id: str,
        batch_size: int,
        split_ratio: float,
        augmentation_cfg,
        validation_mode: str = "auto",
    ):
        super().__init__()
        self.db_path = db_path
        self.class_names = class_names
        self.label_to_idx = {name: i for i, name in enumerate(class_names)}
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.validation_mode = validation_mode
        self.augmentation_cfg = augmentation_cfg
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("ALTER TABLE images ADD COLUMN split_override TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        # Get all images that have annotations
        rows = conn.execute("""
            SELECT DISTINCT i.id, i.path, COALESCE(i.split_override, i.split) AS effective_split
            FROM images i
            INNER JOIN annotations a ON a.image_id = i.id
            WHERE i.ignored = 0
        """).fetchall()

        # Load annotations for each image
        image_paths = []
        targets = []
        effective_splits: list[str | None] = []
        for row in rows:
            img_id = row["id"]
            ann_rows = conn.execute(
                "SELECT class_name, x, y, width, height FROM annotations WHERE image_id = ?",
                (img_id,),
            ).fetchall()

            boxes = []
            labels = []
            for ann in ann_rows:
                cls = ann["class_name"]
                if cls not in self.label_to_idx:
                    continue
                x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    logger.warning("Skipping annotation with non-finite coords for image %s", img_id)
                    continue
                if w <= 0 or h <= 0:
                    logger.warning("Skipping zero-area box for image %s", img_id)
                    continue
                # Clamp to [0, 1]
                x, y = max(0, x), max(0, y)
                w, h = min(w, 1 - x), min(h, 1 - y)
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, w, h])
                labels.append(self.label_to_idx[cls])

            if not boxes:
                continue

            image_paths.append(row["path"])
            targets.append({"boxes": boxes, "labels": labels})
            effective_splits.append(row["effective_split"])

        # Include "no objects" images as negative training examples
        try:
            neg_rows = conn.execute("""
                SELECT i.id, i.path, COALESCE(i.split_override, i.split) AS effective_split
                FROM images i
                WHERE i.ignored = 0 AND i.no_objects = 1
            """).fetchall()
            for row in neg_rows:
                image_paths.append(row["path"])
                targets.append({"boxes": [], "labels": []})
                effective_splits.append(row["effective_split"])
        except sqlite3.OperationalError:
            pass  # no_objects column may not exist in older databases

        conn.close()

        if len(image_paths) < 2:
            # Need at least 2 samples for train/val split
            self.train_dataset = _DetectionDataset(image_paths, targets, self.processor)
            self.val_dataset = _DetectionDataset(image_paths, targets, self.processor)
            return

        has_val = self.validation_mode != "random" and any(s == "valid" for s in effective_splits)
        if has_val:
            train_paths: list[str] = []
            train_targets: list[dict] = []
            val_paths: list[str] = []
            val_targets: list[dict] = []
            for path, target, eff_split in zip(image_paths, targets, effective_splits):
                if eff_split == "test":
                    continue
                if eff_split == "valid":
                    val_paths.append(path)
                    val_targets.append(target)
                else:
                    train_paths.append(path)
                    train_targets.append(target)
        else:
            train_paths, val_paths, train_targets, val_targets = train_test_split(
                image_paths, targets,
                train_size=self.split_ratio,
                random_state=42,
            )

        self.train_dataset = _DetectionDataset(train_paths, train_targets, self.processor)
        self.val_dataset = _DetectionDataset(val_paths, val_targets, self.processor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=_collate_fn,
        )
