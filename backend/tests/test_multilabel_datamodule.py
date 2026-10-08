import sqlite3
from types import SimpleNamespace
from PIL import Image
from kumo_label.training.multilabel_datamodule import KumoMultilabelDataModule

_AUG_OFF = SimpleNamespace(enabled=False, horizontal_flip=False, rotation=0,
                           color_jitter=SimpleNamespace(brightness=0, contrast=0, saturation=0))


def _make_images_table(conn, with_split_override=False):
    extra = ", split_override TEXT" if with_split_override else ""
    conn.execute(f"""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            class TEXT NOT NULL,
            split TEXT NOT NULL,
            label TEXT,
            annotation TEXT,
            ignored INTEGER NOT NULL DEFAULT 0{extra}
        )
    """)
    conn.execute("""
        CREATE TABLE image_labels (
            image_id INTEGER NOT NULL REFERENCES images(id),
            class_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (image_id, class_name)
        )
    """)


def _insert_image(conn, img_dir, i, split="train", ignored=0, split_override=None):
    img_path = img_dir / f"img_{i}.jpg"
    Image.new("RGB", (64, 64), color="red").save(str(img_path))
    if split_override is not None:
        conn.execute(
            "INSERT INTO images (filename, path, class, split, ignored, split_override) VALUES (?, ?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), "unused", split, ignored, split_override),
        )
    else:
        conn.execute(
            "INSERT INTO images (filename, path, class, split, ignored) VALUES (?, ?, ?, ?, ?)",
            (img_path.name, str(img_path), "unused", split, ignored),
        )
    return conn.execute("SELECT id FROM images WHERE path = ?", (str(img_path),)).fetchone()[0]


def _tag(conn, image_id, class_name):
    conn.execute(
        "INSERT INTO image_labels (image_id, class_name, created_at) VALUES (?, ?, datetime('now'))",
        (image_id, class_name),
    )


def _create_test_db(tmp_path, num_images=20, tags_per_image=None):
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    _make_images_table(conn)

    if tags_per_image is None:
        def tags_per_image(i):
            tags = ["cat"] if i % 2 == 0 else ["dog"]
            if i % 3 == 0:
                tags.append("outdoor")
            return tags

    for i in range(num_images):
        img_id = _insert_image(conn, img_dir, i)
        for t in tags_per_image(i):
            _tag(conn, img_id, t)
    conn.commit()
    conn.close()
    return str(db_path)


def test_multilabel_batch_shape_and_dtype(tmp_path):
    db_path = _create_test_db(tmp_path, num_images=12)
    dm = KumoMultilabelDataModule(
        db_path=db_path,
        class_names=["cat", "dog", "outdoor"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["pixel_values"].shape[0] == batch["labels"].shape[0]
    assert batch["pixel_values"].shape[1:] == (3, 32, 32)
    assert batch["labels"].shape[1] == 3
    assert batch["labels"].dtype.is_floating_point
    # multi-hot values are 0.0 or 1.0
    unique_vals = set(batch["labels"].flatten().tolist())
    assert unique_vals <= {0.0, 1.0}


def test_multilabel_multi_hot_values_correct(tmp_path):
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    _make_images_table(conn)
    img_id = _insert_image(conn, img_dir, 0)
    _tag(conn, img_id, "cat")
    _tag(conn, img_id, "outdoor")
    # Add a second image so the val split isn't the entire (tiny) dataset in weird ways.
    img_id2 = _insert_image(conn, img_dir, 1)
    _tag(conn, img_id2, "dog")
    conn.commit()
    conn.close()

    dm = KumoMultilabelDataModule(
        db_path=str(db_path),
        class_names=["cat", "dog", "outdoor"],
        image_size=32,
        batch_size=1,
        split_ratio=0.5,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    # Regardless of which split each sample lands in, both should be found
    # somewhere across train+val with the correct multi-hot encoding.
    all_paths = dm.train_dataset.image_paths + dm.val_dataset.image_paths
    all_labels = dm.train_dataset.labels + dm.val_dataset.labels
    idx = all_paths.index(str(img_dir / "img_0.jpg"))
    assert all_labels[idx] == [1.0, 0.0, 1.0]

    idx2 = all_paths.index(str(img_dir / "img_1.jpg"))
    assert all_labels[idx2] == [0.0, 1.0, 0.0]


def test_multilabel_untagged_images_excluded(tmp_path):
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    _make_images_table(conn)
    # 5 tagged, 3 untagged
    for i in range(5):
        img_id = _insert_image(conn, img_dir, i)
        _tag(conn, img_id, "cat")
    for i in range(5, 8):
        _insert_image(conn, img_dir, i)  # no tags
    conn.commit()
    conn.close()

    dm = KumoMultilabelDataModule(
        db_path=str(db_path),
        class_names=["cat"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 5


def test_multilabel_unknown_tags_filtered_and_only_unknown_dropped(tmp_path):
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    _make_images_table(conn)
    # image 0: known "cat" + unknown "bird" -> kept, multi-hot [1, 0]
    img0 = _insert_image(conn, img_dir, 0)
    _tag(conn, img0, "cat")
    _tag(conn, img0, "bird")
    # image 1: only unknown "bird" -> dropped entirely
    img1 = _insert_image(conn, img_dir, 1)
    _tag(conn, img1, "bird")
    # image 2: known "dog" -> kept
    img2 = _insert_image(conn, img_dir, 2)
    _tag(conn, img2, "dog")
    conn.commit()
    conn.close()

    dm = KumoMultilabelDataModule(
        db_path=str(db_path),
        class_names=["cat", "dog"],
        image_size=32,
        batch_size=4,
        split_ratio=0.5,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 2

    all_paths = dm.train_dataset.image_paths + dm.val_dataset.image_paths
    all_labels = dm.train_dataset.labels + dm.val_dataset.labels
    idx0 = all_paths.index(str(img_dir / "img_0.jpg"))
    assert all_labels[idx0] == [1.0, 0.0]
    assert str(img_dir / "img_1.jpg") not in all_paths


def test_multilabel_ignored_images_excluded(tmp_path):
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    _make_images_table(conn)
    for i in range(4):
        img_id = _insert_image(conn, img_dir, i, ignored=0)
        _tag(conn, img_id, "cat")
    for i in range(4, 6):
        img_id = _insert_image(conn, img_dir, i, ignored=1)
        _tag(conn, img_id, "cat")
    conn.commit()
    conn.close()

    dm = KumoMultilabelDataModule(
        db_path=str(db_path),
        class_names=["cat"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    total = len(dm.train_dataset) + len(dm.val_dataset)
    assert total == 4


def test_multilabel_uses_assigned_splits_when_val_present(tmp_path):
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    _make_images_table(conn)
    layout = ["train"] * 6 + ["valid"] * 2 + ["test"] * 2
    for i, split in enumerate(layout):
        img_id = _insert_image(conn, img_dir, i, split=split)
        _tag(conn, img_id, "cat")
    conn.commit()
    conn.close()

    dm = KumoMultilabelDataModule(
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


def test_multilabel_split_override_honored(tmp_path):
    db_path = tmp_path / "kumo.db"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    _make_images_table(conn, with_split_override=True)
    for i in range(10):
        override = "valid" if i < 2 else None
        img_id = _insert_image(conn, img_dir, i, split="train", split_override=override)
        _tag(conn, img_id, "cat")
    conn.commit()
    conn.close()

    dm = KumoMultilabelDataModule(
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


def test_multilabel_falls_back_to_random_split_when_no_val(tmp_path):
    db_path = _create_test_db(tmp_path, num_images=20, tags_per_image=lambda i: ["cat"])
    dm = KumoMultilabelDataModule(
        db_path=db_path,
        class_names=["cat"],
        image_size=32,
        batch_size=4,
        split_ratio=0.8,
        augmentation_cfg=_AUG_OFF,
    )
    dm.setup()
    assert len(dm.train_dataset) == 16
    assert len(dm.val_dataset) == 4
