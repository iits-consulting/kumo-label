import sqlite3

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class _MultilabelImageDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list[list[float]], transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"pixel_values": pixel_values, "labels": labels}


class KumoMultilabelDataModule(L.LightningDataModule):
    def __init__(
        self,
        db_path: str,
        class_names: list[str],
        image_size: int,
        batch_size: int,
        split_ratio: float,
        augmentation_cfg,
        validation_mode: str = "auto",
    ):
        super().__init__()
        self.db_path = db_path
        self.class_names = class_names
        self.label_to_idx = {name: i for i, name in enumerate(class_names)}
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.validation_mode = validation_mode

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if augmentation_cfg.enabled:
            aug_transforms = [transforms.Resize((image_size, image_size))]
            if augmentation_cfg.horizontal_flip:
                aug_transforms.append(transforms.RandomHorizontalFlip())
            if augmentation_cfg.rotation > 0:
                aug_transforms.append(transforms.RandomRotation(augmentation_cfg.rotation))
            cj = augmentation_cfg.color_jitter
            if cj.brightness > 0 or cj.contrast > 0 or cj.saturation > 0:
                aug_transforms.append(transforms.ColorJitter(
                    brightness=cj.brightness,
                    contrast=cj.contrast,
                    saturation=cj.saturation,
                ))
            aug_transforms.extend([transforms.ToTensor(), normalize])
            self.train_transform = transforms.Compose(aug_transforms)
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ])

        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("ALTER TABLE images ADD COLUMN split_override TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        rows = conn.execute(
            """
            SELECT i.path, GROUP_CONCAT(l.class_name) AS tags,
                   COALESCE(i.split_override, i.split) AS effective_split
            FROM images i JOIN image_labels l ON l.image_id = i.id
            WHERE i.ignored = 0 GROUP BY i.id
            """
        ).fetchall()
        conn.close()

        records = []  # (path, multi_hot, effective_split)
        for path, tags, effective_split in rows:
            tag_list = (tags or "").split(",")
            known = [t for t in tag_list if t in self.label_to_idx]
            if not known:
                continue
            multi_hot = [0.0] * len(self.class_names)
            for t in known:
                multi_hot[self.label_to_idx[t]] = 1.0
            records.append((path, multi_hot, effective_split))

        # Same assigned-split-vs-random logic as KumoDataModule (see that module
        # for rationale). Label sets can't be stratified, so the random
        # fallback here is always non-stratified.
        has_val = self.validation_mode != "random" and any(r[2] == "valid" for r in records)
        if has_val:
            train_records = [r for r in records if r[2] not in ("valid", "test")]
            val_records = [r for r in records if r[2] == "valid"]
            train_paths = [r[0] for r in train_records]
            train_labels = [r[1] for r in train_records]
            val_paths = [r[0] for r in val_records]
            val_labels = [r[1] for r in val_records]
        else:
            paths = [r[0] for r in records]
            labels = [r[1] for r in records]
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                paths, labels,
                train_size=self.split_ratio,
                random_state=42,
            )

        self.train_dataset = _MultilabelImageDataset(train_paths, train_labels, self.train_transform)
        self.val_dataset = _MultilabelImageDataset(val_paths, val_labels, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
