import sqlite3

import lightning as L
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class _ImageDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list[int], transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values, "labels": self.labels[idx]}


class KumoDataModule(L.LightningDataModule):
    def __init__(
        self,
        db_path: str,
        class_names: list[str],
        image_size: int,
        batch_size: int,
        split_ratio: float,
        augmentation_cfg,
        validation_mode: str = "auto",
        class_balance_strategy: str = "none",
    ):
        super().__init__()
        self.db_path = db_path
        self.class_names = class_names
        self.label_to_idx = {name: i for i, name in enumerate(class_names)}
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.validation_mode = validation_mode
        self.class_balance_strategy = class_balance_strategy

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
            "SELECT path, COALESCE(annotation, class) AS effective_label, "
            "COALESCE(split_override, split) AS effective_split "
            "FROM images WHERE ignored = 0"
        ).fetchall()
        conn.close()

        records = []  # (path, label_idx, effective_split)
        for path, effective_label, effective_split in rows:
            if effective_label in self.label_to_idx:
                records.append((path, self.label_to_idx[effective_label], effective_split))

        # If any image is assigned to the validation set (manually or via folder
        # structure), use the assigned splits and hold out 'test'. Otherwise
        # fall back to a stratified random split for backwards compatibility.
        # validation_mode="random" forces the random split even when a val set exists.
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
            # Stratified split requires at least 2 samples per class.
            # Fall back to non-stratified if any class has fewer.
            from collections import Counter
            class_counts = Counter(labels)
            can_stratify = all(c >= 2 for c in class_counts.values())

            train_paths, val_paths, train_labels, val_labels = train_test_split(
                paths, labels,
                train_size=self.split_ratio,
                stratify=labels if can_stratify else None,
                random_state=42,
            )

        if self.class_balance_strategy == "undersample":
            from collections import Counter as _Counter
            counts = _Counter(train_labels)
            min_count = min(counts.values())
            indices_by_class = {}
            for i, label in enumerate(train_labels):
                indices_by_class.setdefault(label, []).append(i)
            kept = []
            for indices in indices_by_class.values():
                kept.extend(indices[:min_count])
            train_paths = [train_paths[i] for i in kept]
            train_labels = [train_labels[i] for i in kept]

        self.train_dataset = _ImageDataset(train_paths, train_labels, self.train_transform)
        self.val_dataset = _ImageDataset(val_paths, val_labels, self.val_transform)

    def train_dataloader(self):
        if self.class_balance_strategy == "oversample":
            from collections import Counter as _Counter
            from torch.utils.data import WeightedRandomSampler
            counts = _Counter(self.train_dataset.labels)
            weights = [1.0 / counts[label] for label in self.train_dataset.labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4
            )
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
