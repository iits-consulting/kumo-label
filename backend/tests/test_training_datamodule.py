import sqlite3
from types import SimpleNamespace
from PIL import Image
from kumo_label.training.datamodule import KumoDataModule

_AUG_OFF = SimpleNamespace(enabled=False, horizontal_flip=False, rotation=0,
                           color_jitter=SimpleNamespace(brightness=0, contrast=0, saturation=0))


def _create_test_db(tmp_path, num_images=20):
    """Create a kumo.db with labeled images for testing."""
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL,
            split TEXT NOT NULL,
            label TEXT,
            annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0
        )
    """)

    classes = ["cat", "dog"]
    for i in range(num_images):
        cls = classes[i % len(classes)]
        img_path = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color="red").save(str(img_path))
        conn.execute(
            "INSERT INTO images (filename, path, class, split, annotation) VALUES (?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), cls, "train", cls),
        )
    conn.commit()
    conn.close()
    return str(db_path)


def test_datamodule_setup_splits_data(tmp_path):
    db_path = _create_test_db(tmp_path, num_images=20)
    dm = KumoDataModule(
        db_path=db_path,
        class_names=["cat", "dog"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    assert len(dm.train_dataset) == 16
    assert len(dm.val_dataset) == 4


def test_datamodule_batch_shape(tmp_path):
    db_path = _create_test_db(tmp_path, num_images=10)
    dm = KumoDataModule(
        db_path=db_path,
        class_names=["cat", "dog"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["pixel_values"].shape == (4, 3, 32, 32)
    assert batch["labels"].shape == (4,)


def test_datamodule_uses_ground_truth_as_fallback(tmp_path):
    """Images with no annotation but a valid class are included via COALESCE."""
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL, split TEXT NOT NULL, label TEXT, annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0
        )
    """)
    # 5 annotated + 5 with only ground truth class — all should be used
    for i in range(10):
        img_path = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color="red").save(str(img_path))
        annotation = "cat" if i < 5 else None
        conn.execute(
            "INSERT INTO images (filename, path, class, split, annotation) VALUES (?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), "cat", "train", annotation),
        )
    conn.commit()
    conn.close()

    dm = KumoDataModule(
        db_path=str(db_path),
        class_names=["cat"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 10


def test_datamodule_annotation_overrides_class(tmp_path):
    """When annotation differs from class, annotation wins."""
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL, split TEXT NOT NULL, label TEXT, annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0
        )
    """)
    for i in range(4):
        img_path = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color="red").save(str(img_path))
        # class is "cat" but annotation overrides to "dog"
        conn.execute(
            "INSERT INTO images (filename, path, class, split, annotation) VALUES (?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), "cat", "train", "dog"),
        )
    conn.commit()
    conn.close()

    dm = KumoDataModule(
        db_path=str(db_path),
        class_names=["cat", "dog"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 4
    # All labels should be "dog" (index 1), not "cat" (index 0)
    all_labels = dm.train_dataset.labels + dm.val_dataset.labels
    assert all(label == 1 for label in all_labels)


def test_datamodule_uses_assigned_splits_when_val_present(tmp_path):
    """When at least one image has effective_split='valid', use folder/override
    splits instead of the random ratio-based split, and exclude 'test' images."""
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL, split TEXT NOT NULL, label TEXT, annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0,
            split_override TEXT
        )
    """)
    # 6 train, 2 valid, 2 test (folder splits), no overrides
    layout = ["train"] * 6 + ["valid"] * 2 + ["test"] * 2
    for i, split in enumerate(layout):
        img_path = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color="red").save(str(img_path))
        conn.execute(
            "INSERT INTO images (filename, path, class, split, annotation) VALUES (?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), "cat", split, "cat"),
        )
    conn.commit()
    conn.close()

    dm = KumoDataModule(
        db_path=str(db_path),
        class_names=["cat"],
        image_size=32,
        batch_size=2,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    assert len(dm.train_dataset) == 6
    assert len(dm.val_dataset) == 2


def test_datamodule_split_override_wins_over_folder(tmp_path):
    """Manual split_override takes precedence over folder split."""
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL, split TEXT NOT NULL, label TEXT, annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0,
            split_override TEXT
        )
    """)
    # All folder-labeled 'train'; override 2 as 'valid'
    for i in range(10):
        img_path = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color="red").save(str(img_path))
        override = "valid" if i < 2 else None
        conn.execute(
            "INSERT INTO images (filename, path, class, split, annotation, split_override) VALUES (?, ?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), "cat", "train", "cat", override),
        )
    conn.commit()
    conn.close()

    dm = KumoDataModule(
        db_path=str(db_path),
        class_names=["cat"],
        image_size=32,
        batch_size=2,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    assert len(dm.train_dataset) == 8
    assert len(dm.val_dataset) == 2


def test_datamodule_falls_back_to_random_when_no_val(tmp_path):
    """When no image has effective_split='valid', use the random split_ratio."""
    db_path = _create_test_db(tmp_path, num_images=20)
    dm = KumoDataModule(
        db_path=db_path,
        class_names=["cat", "dog"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    # All images are folder-split='train' with no override, no 'valid' → random
    assert len(dm.train_dataset) == 16
    assert len(dm.val_dataset) == 4


def test_datamodule_random_mode_ignores_assigned_val(tmp_path):
    """validation_mode='random' forces the ratio-based split even when a 'valid'
    set is defined, folding the val/test images back into the random pool."""
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL, split TEXT NOT NULL, label TEXT, annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0,
            split_override TEXT
        )
    """)
    # 16 train, 2 valid, 2 test — same layout the auto/user_defined path would honor.
    layout = ["train"] * 16 + ["valid"] * 2 + ["test"] * 2
    for i, split in enumerate(layout):
        img_path = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color="red").save(str(img_path))
        conn.execute(
            "INSERT INTO images (filename, path, class, split, annotation) VALUES (?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), "cat", split, "cat"),
        )
    conn.commit()
    conn.close()

    dm = KumoDataModule(
        db_path=str(db_path),
        class_names=["cat"],
        image_size=32,
        batch_size=2,
        split_ratio=0.8,
        validation_mode="random",
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    # All 20 images go through the random 80/20 split, ignoring the assigned val/test.
    assert len(dm.train_dataset) == 16
    assert len(dm.val_dataset) == 4


def test_datamodule_skips_unknown_class(tmp_path):
    """Images whose effective label is not in class_names are skipped."""
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL, split TEXT NOT NULL, label TEXT, annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0
        )
    """)
    for i in range(6):
        img_path = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color="red").save(str(img_path))
        # 3 with known class, 3 with unknown class (no annotation)
        cls = "cat" if i < 3 else "unknown_folder"
        conn.execute(
            "INSERT INTO images (filename, path, class, split, annotation) VALUES (?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), cls, "train", None),
        )
    conn.commit()
    conn.close()

    dm = KumoDataModule(
        db_path=str(db_path),
        class_names=["cat"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 3
