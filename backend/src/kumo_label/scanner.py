import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "val": "valid",
    "valid": "valid",
    "validation": "valid",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".avif"}


def _is_image_file(p: Path) -> bool:
    """Non-empty file with an image extension. Zero-byte files (e.g. the two
    corrupt images shipped in the Kaggle cats-vs-dogs dataset) crash PIL in
    DataLoader workers, so they are never registered."""
    return (
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.stat().st_size > 0
    )


def _ensure_ml_tables(conn: sqlite3.Connection) -> None:
    """Create jobs and projections tables and enable WAL journal mode."""
    # WAL mode allows concurrent reads + one writer without "database is locked".
    # It is persisted to the DB file, so all future connections automatically use it.
    result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if result[0] != "wal":
        raise RuntimeError(f"Failed to enable WAL mode: got {result[0]!r}")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            status      TEXT NOT NULL,
            message     TEXT,
            progress    REAL DEFAULT 0,
            total       INTEGER DEFAULT 0,
            completed   INTEGER DEFAULT 0,
            error       TEXT,
            params      TEXT NOT NULL,
            dedup_key   TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projections (
            param_hash      TEXT PRIMARY KEY,
            embedding_hash  TEXT NOT NULL,
            model           TEXT NOT NULL,
            method          TEXT NOT NULL,
            params          TEXT NOT NULL,
            coords          TEXT NOT NULL,
            created_at      TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id              TEXT PRIMARY KEY,
            status          TEXT NOT NULL,
            task_type       TEXT NOT NULL DEFAULT 'classification',
            model_name      TEXT NOT NULL,
            display_name    TEXT NOT NULL,
            config          TEXT,
            num_classes     INTEGER NOT NULL,
            class_names     TEXT,
            label_count     INTEGER DEFAULT 0,
            epochs          INTEGER NOT NULL,
            current_epoch   INTEGER DEFAULT 0,
            current_step    INTEGER DEFAULT 0,
            total_steps     INTEGER DEFAULT 0,
            train_loss      REAL,
            val_loss        REAL,
            val_accuracy    REAL,
            val_f1          REAL,
            best_accuracy   REAL,
            best_f1         REAL,
            val_auroc       REAL,
            val_precision   REAL,
            val_recall      REAL,
            best_auroc      REAL,
            error           TEXT,
            output_dir      TEXT,
            trackio_run_name TEXT,
            trackio_dir     TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            finished_at     TEXT,
            dataset_snapshot TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE training_runs ADD COLUMN task_type TEXT NOT NULL DEFAULT 'classification'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE training_runs ADD COLUMN trackio_run_name TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE training_runs ADD COLUMN trackio_dir TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE training_runs ADD COLUMN dataset_snapshot TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE training_runs ADD COLUMN primary_metric_name TEXT DEFAULT 'accuracy'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE training_runs ADD COLUMN secondary_metric_name TEXT DEFAULT 'f1'")
    except sqlite3.OperationalError:
        pass
    for col in ["val_auroc", "val_precision", "val_recall", "best_auroc"]:
        try:
            conn.execute(f"ALTER TABLE training_runs ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass
    for col in ["current_step", "total_steps"]:
        try:
            conn.execute(f"ALTER TABLE training_runs ADD COLUMN {col} INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute("ALTER TABLE training_runs ADD COLUMN val_per_class TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            run_id          TEXT NOT NULL REFERENCES training_runs(id),
            epoch           INTEGER NOT NULL,
            step            INTEGER NOT NULL DEFAULT 0,
            train_loss      REAL,
            val_loss        REAL,
            val_accuracy    REAL,
            val_f1          REAL,
            learning_rate   REAL,
            val_auroc       REAL,
            val_precision   REAL,
            val_recall      REAL,
            PRIMARY KEY (run_id, step)
        )
    """)
    for col in ["val_auroc", "val_precision", "val_recall"]:
        try:
            conn.execute(f"ALTER TABLE training_metrics ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            run_id                TEXT NOT NULL REFERENCES training_runs(id),
            image_id              INTEGER NOT NULL REFERENCES images(id),
            predicted_class       TEXT NOT NULL,
            confidence            REAL NOT NULL,
            uncertainty           REAL NOT NULL DEFAULT 0,
            al_score              REAL NOT NULL DEFAULT 0,
            top_predictions       TEXT NOT NULL,
            is_unlabeled          INTEGER NOT NULL DEFAULT 0,
            is_wrong              INTEGER NOT NULL DEFAULT 0,
            class_imbalance_score REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (run_id, image_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id    INTEGER NOT NULL REFERENCES images(id),
            class_name  TEXT NOT NULL,
            x           REAL NOT NULL,
            y           REAL NOT NULL,
            width       REAL NOT NULL,
            height      REAL NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_image_id ON annotations(image_id)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL REFERENCES training_runs(id),
            image_id    INTEGER NOT NULL REFERENCES images(id),
            class_name  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            x           REAL NOT NULL,
            y           REAL NOT NULL,
            width       REAL NOT NULL,
            height      REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_det_preds_run_image ON detection_predictions(run_id, image_id)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_labels (
            image_id INTEGER NOT NULL REFERENCES images(id),
            class_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (image_id, class_name)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_labels_image_id ON image_labels(image_id)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_label_seeds (
            image_id INTEGER PRIMARY KEY
        )
    """)


def get_canonical_split(name: str):
    """Return canonical split name for a folder name, or None if not a split folder."""
    return SPLIT_NAMES.get(name.lower())


def _parse_yolo_class_names(root: Path) -> dict[int, str] | None:
    """Try to load class-id-to-name mapping from data.yaml or classes.txt."""
    # data.yaml (standard YOLO format)
    for name in ("data.yaml", "data.yml"):
        yaml_path = root / name
        if yaml_path.exists():
            try:
                import yaml

                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                names = data.get("names")
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
                if isinstance(names, list):
                    return {i: str(v) for i, v in enumerate(names)}
            except Exception:
                pass
    # classes.txt (one name per line)
    classes_path = root / "classes.txt"
    if classes_path.exists():
        lines = classes_path.read_text().strip().splitlines()
        return {i: line.strip() for i, line in enumerate(lines) if line.strip()}
    return None


def _find_yolo_label_file(img_path: Path) -> Path | None:
    """Find the YOLO .txt label file for an image, checking common layouts."""
    stem = img_path.stem
    # Same directory
    txt = img_path.with_suffix(".txt")
    if txt.exists():
        return txt
    # Parallel labels/ directory: images/split/img.jpg -> labels/split/img.txt
    parts = img_path.parts
    for i, part in enumerate(parts):
        if part.lower() == "images":
            label_path = Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(
                ".txt"
            )
            if label_path.exists():
                return label_path
    return None


def _parse_yolo_label_file(
    txt_path: Path, class_map: dict[int, str] | None
) -> list[tuple[str, float, float, float, float]]:
    """Parse a YOLO label file. Returns list of (class_name, x, y, w, h) in top-left normalized coords."""
    annotations = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            continue
        # Convert center-based to top-left
        x = cx - w / 2
        y = cy - h / 2
        # Clamp to [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))
        class_name = class_map.get(class_id, str(class_id)) if class_map else str(class_id)
        annotations.append((class_name, x, y, w, h))
    return annotations


def _import_yolo_annotations(
    conn: sqlite3.Connection,
    root: Path,
    records: list[tuple[str, str, str, str]],
) -> tuple[int, set[str]]:
    """Import YOLO label files into the annotations table. Returns (annotated_image_count, class_names)."""
    class_map = _parse_yolo_class_names(root)
    now = datetime.now(timezone.utc).isoformat()
    annotated_count = 0
    yolo_classes: set[str] = set()

    # Build path -> image_id map
    path_to_id: dict[str, int] = {}
    for row in conn.execute("SELECT id, path FROM images").fetchall():
        path_to_id[row[0] if isinstance(row[0], str) else str(row[1])] = row[0]
    # Re-query properly
    path_to_id = {}
    for row in conn.execute("SELECT id, path FROM images"):
        path_to_id[row[1]] = row[0]

    # Collect image IDs that have YOLO label files so we can clear old imports
    image_ids_with_labels: list[int] = []
    import_data: list[tuple[int, list[tuple[str, float, float, float, float]]]] = []

    for _filename, abs_path, _class_name, _split in records:
        img_path = Path(abs_path)
        txt_path = _find_yolo_label_file(img_path)
        if txt_path is None:
            continue
        annotations = _parse_yolo_label_file(txt_path, class_map)
        if not annotations:
            continue
        image_id = path_to_id.get(abs_path)
        if image_id is None:
            continue
        image_ids_with_labels.append(image_id)
        import_data.append((image_id, annotations))

    # Delete existing annotations for these images to avoid duplicates on re-scan
    if image_ids_with_labels:
        placeholders = ",".join("?" * len(image_ids_with_labels))
        conn.execute(
            f"DELETE FROM annotations WHERE image_id IN ({placeholders})",
            image_ids_with_labels,
        )

    for image_id, annotations in import_data:
        annotated_count += 1
        for class_name, x, y, w, h in annotations:
            yolo_classes.add(class_name)
            conn.execute(
                "INSERT INTO annotations (image_id, class_name, x, y, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (image_id, class_name, x, y, w, h, now),
            )
    if annotated_count > 0:
        conn.commit()
    return annotated_count, yolo_classes


def _scan_yolo_images_dir(images_dir: Path) -> list[tuple[str, str, str, str]]:
    """Scan a YOLO-style images/ directory with optional split subdirs."""
    records: list[tuple[str, str, str, str]] = []
    for item in sorted(images_dir.iterdir()):
        if _is_image_file(item):
            records.append((item.name, str(item), "", "train"))
        elif item.is_dir():
            split = get_canonical_split(item.name)
            split_name = split if split else "train"
            for img_file in sorted(item.iterdir()):
                if _is_image_file(img_file):
                    records.append((img_file.name, str(img_file), "", split_name))
    return records


def _repair_moved_paths(
    conn: sqlite3.Connection, root: Path, records: list[tuple[str, str, str, str]]
) -> None:
    """
    Re-point image rows whose stored absolute path lies outside the current
    dataset root — i.e. the dataset directory was moved since it was indexed.

    Rows are matched to freshly scanned files by their path relative to the
    dataset root, so image ids (and everything keyed to them: embeddings,
    predictions, annotations, training runs) survive the move instead of the
    old rows going dead and the same files being re-registered as new images.
    """
    prefix = str(root) + os.sep
    stale = conn.execute(
        "SELECT id, path FROM images WHERE substr(path, 1, ?) != ?",
        (len(prefix), prefix),
    ).fetchall()
    if not stale:
        return

    new_by_rel = {Path(p).relative_to(root).parts: p for _, p, _, _ in records}
    max_depth = max(len(rel) for rel in new_by_rel)
    taken = {row[0] for row in conn.execute("SELECT path FROM images")}
    updates = []
    for img_id, old_path in stale:
        parts = Path(old_path).parts
        # Longest suffix first, so a root-level filename can't shadow a
        # class/filename match.
        for k in range(min(max_depth, len(parts) - 1), 0, -1):
            new_path = new_by_rel.get(parts[-k:])
            if new_path is None:
                continue
            # Skip if another row already claims the new path (e.g. a rescan
            # before this fix existed already duplicated it) — UNIQUE(path).
            if new_path not in taken:
                updates.append((new_path, img_id))
                taken.add(new_path)
            break
    if updates:
        conn.executemany("UPDATE images SET path = ? WHERE id = ?", updates)


def scan_dataset(dataset_path: str) -> dict:
    """
    Scan a dataset directory, index images in kumo.db, return metadata.

    Supports three structures:
      1. dataset/[split]/[class]/image.ext     (classification)
      2. dataset/[class]/image.ext             (classification, all train)
      3. dataset/images/[split]/image.ext      (YOLO detection, with labels/ parallel dir)

    Returns dict with: db_path, classes, splits, counts, annotated_images.
    Raises ValueError if no images are found.
    """
    root = Path(dataset_path).resolve()
    db_path = root / "kumo.db"

    # (filename, abs_path, class_name, split)
    records: list[tuple[str, str, str, str]] = []
    is_yolo_layout = False

    # Check for YOLO-style layout: root/images/ directory
    images_dir = root / "images"
    if images_dir.is_dir():
        yolo_records = _scan_yolo_images_dir(images_dir)
        if yolo_records:
            records = yolo_records
            is_yolo_layout = True

    if not records:
        for item in sorted(root.iterdir()):
            if _is_image_file(item):
                # Root-level images are unlabeled (no class)
                records.append((item.name, str(item), "", "train"))
                continue
            if not item.is_dir():
                continue
            split = get_canonical_split(item.name)
            if split is not None:
                # Check if this is a YOLO split with images directly in it
                has_class_dirs = any(
                    d.is_dir() for d in item.iterdir() if not d.name.startswith(".")
                )
                has_images = any(
                    f.suffix.lower() in IMAGE_EXTENSIONS
                    for f in item.iterdir()
                    if f.is_file()
                )
                if has_class_dirs and not has_images:
                    # Classification layout: split/class/image
                    for class_dir in sorted(item.iterdir()):
                        if not class_dir.is_dir():
                            continue
                        for img_file in sorted(class_dir.iterdir()):
                            if _is_image_file(img_file):
                                records.append((img_file.name, str(img_file), class_dir.name, split))
                else:
                    # YOLO layout: split/image (no class dirs)
                    for img_file in sorted(item.iterdir()):
                        if _is_image_file(img_file):
                            records.append((img_file.name, str(img_file), "", split))
                    if records:
                        is_yolo_layout = True
            else:
                for img_file in sorted(item.iterdir()):
                    if _is_image_file(img_file):
                        records.append((img_file.name, str(img_file), item.name, "train"))

    if not records:
        raise ValueError("No images found in dataset directory")

    conn = sqlite3.connect(str(db_path))
    annotated_images = 0
    yolo_classes: set[str] = set()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT NOT NULL,
                path       TEXT NOT NULL UNIQUE,
                class      TEXT NOT NULL,
                split      TEXT NOT NULL,
                label      TEXT,
                annotation TEXT,
                ignored    INTEGER NOT NULL DEFAULT 0
            )
        """)
        try:
            conn.execute("ALTER TABLE images ADD COLUMN annotation TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists in older databases
        try:
            conn.execute("ALTER TABLE images ADD COLUMN ignored INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE images ADD COLUMN no_objects INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE images ADD COLUMN split_override TEXT")
        except sqlite3.OperationalError:
            pass
        _ensure_ml_tables(conn)
        _repair_moved_paths(conn, root, records)
        conn.executemany(
            "INSERT OR IGNORE INTO images (filename, path, class, split) VALUES (?, ?, ?, ?)",
            records,
        )
        conn.commit()

        # Import YOLO annotations if label files exist
        annotated_images, yolo_classes = _import_yolo_annotations(conn, root, records)

        # Count labeled images from the DB:
        # - has a ground-truth class from directory structure, OR
        # - has a manual classification label (annotation column), OR
        # - has at least one bounding-box annotation, OR
        # - has at least one multilabel tag
        labeled = conn.execute(
            """
            SELECT COUNT(*) FROM images
            WHERE class != ''
               OR annotation IS NOT NULL
               OR id IN (SELECT DISTINCT image_id FROM annotations)
               OR id IN (SELECT DISTINCT image_id FROM image_labels)
            """
        ).fetchone()[0]

        # Collect classes from manual annotations in the DB:
        # classification labels (images.annotation), object-detection
        # bounding-box classes (annotations.class_name), and multilabel
        # tags (image_labels.class_name).
        db_annotation_classes = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT annotation FROM images WHERE annotation IS NOT NULL"
            )
        }
        db_annotation_classes |= {
            row[0]
            for row in conn.execute("SELECT DISTINCT class_name FROM annotations")
        }
        db_annotation_classes |= {
            row[0]
            for row in conn.execute("SELECT DISTINCT class_name FROM image_labels")
        }
    finally:
        conn.close()

    classes = sorted({r[2] for r in records if r[2]} | yolo_classes | db_annotation_classes)
    splits = sorted({r[3] for r in records})
    total = len(records)
    by_split = {s: sum(1 for r in records if r[3] == s) for s in splits}

    return {
        "db_path": str(db_path),
        "classes": classes,
        "splits": splits,
        "counts": {"total": total, "labeled": labeled, "by_split": by_split},
        "annotated_images": annotated_images,
    }
