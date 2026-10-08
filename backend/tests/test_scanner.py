import sqlite3
import pytest
from kumo_label.scanner import scan_dataset, get_canonical_split


# --- get_canonical_split ---

def test_split_train_variants():
    assert get_canonical_split("train") == "train"
    assert get_canonical_split("Train") == "train"
    assert get_canonical_split("TRAINING") == "train"

def test_split_test():
    assert get_canonical_split("test") == "test"
    assert get_canonical_split("TEST") == "test"

def test_split_valid_variants():
    assert get_canonical_split("val") == "valid"
    assert get_canonical_split("valid") == "valid"
    assert get_canonical_split("validation") == "valid"
    assert get_canonical_split("VALID") == "valid"

def test_not_a_split():
    assert get_canonical_split("cats") is None
    assert get_canonical_split("0") is None
    assert get_canonical_split("") is None


# --- scan_dataset: with splits ---

def test_scan_with_splits(tmp_path):
    (tmp_path / "train" / "0").mkdir(parents=True)
    (tmp_path / "train" / "1").mkdir(parents=True)
    (tmp_path / "test" / "0").mkdir(parents=True)
    (tmp_path / "train" / "0" / "a.jpg").write_bytes(b"x")
    (tmp_path / "train" / "1" / "b.jpg").write_bytes(b"x")
    (tmp_path / "test" / "0" / "c.jpg").write_bytes(b"x")

    result = scan_dataset(str(tmp_path))

    assert set(result["classes"]) == {"0", "1"}
    assert set(result["splits"]) == {"train", "test"}
    assert result["counts"]["total"] == 3
    assert result["counts"]["by_split"]["train"] == 2
    assert result["counts"]["by_split"]["test"] == 1
    assert result["db_path"] == str(tmp_path / "kumo.db")


def test_scan_without_splits(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "dogs").mkdir()
    (tmp_path / "cats" / "img1.jpg").write_bytes(b"x")
    (tmp_path / "dogs" / "img2.png").write_bytes(b"x")

    result = scan_dataset(str(tmp_path))

    assert set(result["classes"]) == {"cats", "dogs"}
    assert result["splits"] == ["train"]
    assert result["counts"]["total"] == 2


def test_scan_creates_kumo_db(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    assert (tmp_path / "kumo.db").exists()


def test_scan_db_schema(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "img.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))

    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM images LIMIT 1").fetchone()
    assert set(row.keys()) == {"id", "filename", "path", "class", "split", "split_override", "label", "annotation", "ignored", "no_objects"}
    conn.close()


def test_scan_insert_or_ignore_preserves_label(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "img.jpg").write_bytes(b"x")

    scan_dataset(str(tmp_path))

    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    conn.execute("UPDATE images SET label = 'cat' WHERE filename = 'img.jpg'")
    conn.commit()
    conn.close()

    scan_dataset(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    label = conn.execute("SELECT label FROM images WHERE filename = 'img.jpg'").fetchone()[0]
    conn.close()
    assert label == "cat"


def test_scan_no_images_raises(tmp_path):
    (tmp_path / "empty_folder").mkdir()
    with pytest.raises(ValueError, match="No images found"):
        scan_dataset(str(tmp_path))


def test_scan_split_folder_not_a_class(tmp_path):
    (tmp_path / "train" / "cats").mkdir(parents=True)
    (tmp_path / "train" / "cats" / "img.jpg").write_bytes(b"x")
    result = scan_dataset(str(tmp_path))
    assert "train" not in result["classes"]
    assert "cats" in result["classes"]


def test_scan_canonical_split_stored(tmp_path):
    (tmp_path / "training" / "cats").mkdir(parents=True)
    (tmp_path / "training" / "cats" / "img.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    split_val = conn.execute("SELECT split FROM images LIMIT 1").fetchone()[0]
    conn.close()
    assert split_val == "train"


def test_scan_counts_reflect_current_scan(tmp_path):
    # First scan: 2 images
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.jpg").write_bytes(b"x")
    (tmp_path / "cats" / "b.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))

    # Remove one image, re-scan
    (tmp_path / "cats" / "b.jpg").unlink()
    result = scan_dataset(str(tmp_path))

    # counts should reflect current scan (1 image), not DB cumulative (2)
    assert result["counts"]["total"] == 1
    assert result["counts"]["by_split"]["train"] == 1


def test_scan_creates_jobs_table(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    conn.close()
    assert "jobs" in tables
    assert {"id", "type", "status", "message", "progress", "total", "completed",
            "error", "params", "dedup_key", "created_at", "updated_at"} == cols


def test_scan_creates_projections_table(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(projections)").fetchall()}
    conn.close()
    assert "projections" in tables
    assert {"param_hash", "embedding_hash", "model", "method", "params", "coords", "created_at"} == cols


def test_scan_creates_training_runs_table(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(training_runs)").fetchall()}
    conn.close()
    assert "training_runs" in tables
    assert {"id", "status", "task_type", "model_name", "display_name", "config", "num_classes",
            "class_names", "label_count", "epochs", "current_epoch", "current_step", "total_steps",
            "train_loss", "val_loss", "val_accuracy", "val_f1", "best_accuracy", "best_f1",
            "val_auroc", "val_precision", "val_recall", "best_auroc",
            "error", "output_dir", "trackio_run_name", "trackio_dir", "dataset_snapshot",
            "primary_metric_name", "secondary_metric_name", "val_per_class",
            "created_at", "updated_at", "finished_at"} == cols


def test_scan_creates_training_metrics_table(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(training_metrics)").fetchall()}
    conn.close()
    assert "training_metrics" in tables
    assert {"run_id", "epoch", "step", "train_loss", "val_loss", "val_accuracy",
            "val_f1", "learning_rate", "val_auroc", "val_precision", "val_recall"} == cols


# --- dataset moved to a new location ---

def test_scan_repairs_paths_after_dataset_move(tmp_path):
    old_root = tmp_path / "old"
    (old_root / "cats").mkdir(parents=True)
    (old_root / "cats" / "img.jpg").write_bytes(b"x")
    scan_dataset(str(old_root))

    conn = sqlite3.connect(str(old_root / "kumo.db"))
    old_id = conn.execute("SELECT id FROM images").fetchone()[0]
    conn.execute("UPDATE images SET label = 'cat'")
    conn.commit()
    conn.close()

    new_root = tmp_path / "new"
    old_root.rename(new_root)
    scan_dataset(str(new_root))

    conn = sqlite3.connect(str(new_root / "kumo.db"))
    rows = conn.execute("SELECT id, path, label FROM images").fetchall()
    conn.close()
    # Same single row: id and label preserved, path re-pointed, no duplicate.
    assert rows == [(old_id, str(new_root / "cats" / "img.jpg"), "cat")]


def test_scan_repair_skips_already_duplicated_paths(tmp_path):
    new_root = tmp_path / "new"
    (new_root / "cats").mkdir(parents=True)
    (new_root / "cats" / "img.jpg").write_bytes(b"x")
    scan_dataset(str(new_root))

    # Simulate a db that already re-registered the moved file as a new row
    # (the pre-fix behavior): a stale old-path row alongside the new one.
    conn = sqlite3.connect(str(new_root / "kumo.db"))
    conn.execute(
        "INSERT INTO images (filename, path, class, split) VALUES (?, ?, ?, ?)",
        ("img.jpg", str(tmp_path / "old" / "cats" / "img.jpg"), "cats", "train"),
    )
    conn.commit()
    conn.close()

    scan_dataset(str(new_root))  # must not violate UNIQUE(path)

    conn = sqlite3.connect(str(new_root / "kumo.db"))
    count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    conn.close()
    assert count == 2  # stale row left as-is, no crash


# --- multilabel tagging tables ---

def test_scan_creates_image_labels_table(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(image_labels)").fetchall()}
    conn.close()
    assert "image_labels" in tables
    assert {"image_id", "class_name", "created_at"} == cols


def test_scan_creates_image_label_seeds_table(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(image_label_seeds)").fetchall()}
    conn.close()
    assert "image_label_seeds" in tables
    assert {"image_id"} == cols


def test_scan_labeled_count_includes_tagged_images(tmp_path):
    # Root-level images have no folder class, so they start out unlabeled.
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))

    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    image_id = conn.execute("SELECT id FROM images WHERE filename = 'a.jpg'").fetchone()[0]
    conn.execute(
        "INSERT INTO image_labels (image_id, class_name, created_at) VALUES (?, ?, datetime('now'))",
        (image_id, "spot"),
    )
    conn.commit()
    conn.close()

    result = scan_dataset(str(tmp_path))
    assert result["counts"]["labeled"] == 1


def test_scan_classes_include_tagged_classes(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    scan_dataset(str(tmp_path))

    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    image_id = conn.execute("SELECT id FROM images WHERE filename = 'a.jpg'").fetchone()[0]
    conn.execute(
        "INSERT INTO image_labels (image_id, class_name, created_at) VALUES (?, ?, datetime('now'))",
        (image_id, "spot"),
    )
    conn.commit()
    conn.close()

    result = scan_dataset(str(tmp_path))
    assert "spot" in result["classes"]


def test_scan_skips_zero_byte_images(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "good.jpg").write_bytes(b"x")
    (tmp_path / "cats" / "empty.jpg").touch()

    result = scan_dataset(str(tmp_path))
    assert result["counts"]["total"] == 1

    conn = sqlite3.connect(str(tmp_path / "kumo.db"))
    filenames = [row[0] for row in conn.execute("SELECT filename FROM images")]
    conn.close()
    assert filenames == ["good.jpg"]


def test_scan_only_zero_byte_images_raises(tmp_path):
    (tmp_path / "empty.jpg").touch()
    with pytest.raises(ValueError):
        scan_dataset(str(tmp_path))
