import mimetypes
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kumo_label.db_utils import validate_db_path
from kumo_label.scanner import scan_dataset

router = APIRouter(prefix="/api/datasets")


class LoadRequest(BaseModel):
    path: str


class LabelRequest(BaseModel):
    db_path: str
    ids: List[int] = Field(min_length=1)
    label: Optional[str] = Field(default=None, min_length=1)


class IgnoreRequest(BaseModel):
    db_path: str
    ids: List[int] = Field(min_length=1)
    ignored: bool = True


class NoObjectsRequest(BaseModel):
    db_path: str
    ids: List[int] = Field(min_length=1)
    no_objects: bool = True


class SplitOverrideRequest(BaseModel):
    db_path: str
    ids: List[int] = Field(min_length=1)
    # null/None clears the override (falls back to folder split)
    split: Optional[str] = None


class TagRequest(BaseModel):
    db_path: str
    ids: List[int] = Field(min_length=1)
    class_name: str = Field(min_length=1)
    present: bool = True


class SeedRequest(BaseModel):
    db_path: str


def _ensure_image_labels_tables(conn: sqlite3.Connection) -> None:
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


@router.get("/browse")
def browse_directories(path: Optional[str] = Query(None)):
    """List subdirectories under `path` for the dataset directory picker.

    Defaults to the user's home directory. Hidden directories (`.foo`) are
    skipped. Only directories are listed; symlinks are followed.
    """
    target = Path(path).expanduser() if path else Path.home()
    try:
        target = target.resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Path does not exist")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    entries: List[dict] = []
    try:
        with os.scandir(target) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=True):
                        entries.append({"name": entry.name, "path": str(target / entry.name)})
                except OSError:
                    continue
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    entries.sort(key=lambda e: e["name"].lower())
    parent = str(target.parent) if target.parent != target else None
    return {"path": str(target), "parent": parent, "entries": entries}


@router.post("/load")
def load_dataset(req: LoadRequest):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=400, detail="Path does not exist")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    try:
        result = scan_dataset(req.path)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rescan")
def rescan_dataset(req: LoadRequest):
    """Re-scan an already-loaded dataset directory and return count of newly-added images."""
    p = Path(req.path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Path does not exist or is not a directory")
    db_file = p / "kumo.db"
    if not db_file.exists():
        raise HTTPException(status_code=400, detail="Dataset has not been loaded yet")

    conn = sqlite3.connect(str(db_file))
    try:
        before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    finally:
        conn.close()

    try:
        result = scan_dataset(req.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    after = result["counts"]["total"]
    result["new_images"] = max(0, after - before)
    return result


@router.get("/images")
def list_images(
    db_path: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=250000),
    split: str = Query(default=None),
    class_: str = Query(default=None, alias="class"),
    ignored: str = Query(default=None),
    sort_by: str = Query(default=None),
    sort_order: str = Query(default="desc"),
    run_id: str = Query(default=None),
):
    db_file = Path(db_path)
    if not db_file.is_absolute():
        raise HTTPException(status_code=400, detail="db_path must be an absolute path")
    if not db_file.exists():
        raise HTTPException(status_code=400, detail="db_path does not exist")
    if db_file.name != "kumo.db":
        raise HTTPException(status_code=400, detail="db_path must point to a kumo.db file")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("SELECT 1 FROM images LIMIT 1")
        except sqlite3.DatabaseError:
            raise HTTPException(status_code=400, detail="Not a valid kumo.db file")

        try:
            conn.execute("ALTER TABLE images ADD COLUMN annotation TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute("ALTER TABLE images ADD COLUMN ignored INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE images ADD COLUMN no_objects INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE images ADD COLUMN split_override TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        # Ensure annotations table exists for annotation_count join
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
        # Ensure image_labels table exists for label_classes join (multilabel tags)
        _ensure_image_labels_tables(conn)
        # Ensure prediction columns exist for older databases
        try:
            conn.execute("SELECT 1 FROM predictions LIMIT 0")
            for col, default in [("uncertainty", "0"), ("al_score", "0")]:
                try:
                    conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} REAL NOT NULL DEFAULT {default}")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass
        except sqlite3.OperationalError:
            pass  # predictions table doesn't exist yet

        conditions = []
        params: list = []
        if split:
            conditions.append("split = ?")
            params.append(split)
        if class_:
            conditions.append("class = ?")
            params.append(class_)
        if ignored is not None:
            conditions.append("ignored = ?")
            params.append(1 if ignored == "true" else 0)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Sorting by confidence/uncertainty requires joining the predictions table
        order_clause = ""
        select_cols = "images.id, images.filename, images.path, images.class, images.split, images.split_override, images.label, images.annotation, images.ignored, images.no_objects, COALESCE(ac.cnt, 0) AS annotation_count, acl.classes AS annotation_classes, il.classes AS label_classes"
        annotation_joins = (
            "LEFT JOIN (SELECT image_id, COUNT(*) AS cnt FROM annotations GROUP BY image_id) ac ON ac.image_id = images.id "
            "LEFT JOIN (SELECT image_id, GROUP_CONCAT(class_name) AS classes FROM annotations GROUP BY image_id) acl ON acl.image_id = images.id "
            "LEFT JOIN (SELECT image_id, GROUP_CONCAT(class_name) AS classes FROM image_labels GROUP BY image_id) il ON il.image_id = images.id"
        )
        table_ref = "images " + annotation_joins
        join_params: list = []

        if sort_by in ("confidence", "uncertainty", "al_score") and run_id:
            prediction_join = "LEFT JOIN predictions ON images.id = predictions.image_id AND predictions.run_id = ?"
            join_params = [run_id]
            table_ref = "images " + prediction_join + " " + annotation_joins
            direction = "DESC" if sort_order == "desc" else "ASC"
            sort_col = {"uncertainty": "predictions.uncertainty", "confidence": "predictions.confidence", "al_score": "predictions.al_score"}[sort_by]
            # NULLS LAST: images without predictions go to the end
            order_clause = f"ORDER BY CASE WHEN {sort_col} IS NULL THEN 1 ELSE 0 END, {sort_col} {direction}"

        total = conn.execute(
            f"SELECT COUNT(*) FROM {table_ref} {where}", join_params + params
        ).fetchone()[0]
        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT {select_cols} FROM {table_ref} {where} {order_clause} LIMIT ? OFFSET ?",
            join_params + params + [limit, offset],
        ).fetchall()
        return {"images": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}
    finally:
        if conn:
            conn.close()


def _resolve_safe_image_path(file_path: str):
    """Return resolved Path if file_path is within a dataset directory, else None."""
    p = Path(file_path).resolve()
    current = p.parent
    while True:
        if (current / "kumo.db").exists():
            try:
                p.relative_to(current)
                return p
            except ValueError:
                pass
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


@router.patch("/images/labels")
def update_labels(req: LabelRequest):
    db_file = Path(req.db_path)
    if not db_file.is_absolute():
        raise HTTPException(status_code=400, detail="db_path must be an absolute path")
    if not db_file.exists():
        raise HTTPException(status_code=400, detail="db_path does not exist")
    if db_file.name != "kumo.db":
        raise HTTPException(status_code=400, detail="db_path must point to a kumo.db file")

    conn = None
    try:
        conn = sqlite3.connect(req.db_path)
        try:
            conn.execute("SELECT 1 FROM images LIMIT 1")
        except sqlite3.DatabaseError:
            raise HTTPException(status_code=400, detail="Not a valid kumo.db file")

        try:
            conn.execute("ALTER TABLE images ADD COLUMN annotation TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        placeholders = ",".join("?" * len(req.ids))
        cursor = conn.execute(
            f"UPDATE images SET annotation = ? WHERE id IN ({placeholders})",
            [req.label, *req.ids],
        )
        if req.label is None:
            _ensure_image_labels_tables(conn)
            conn.execute(
                f"DELETE FROM image_labels WHERE image_id IN ({placeholders})",
                req.ids,
            )
            conn.execute(
                f"INSERT OR IGNORE INTO image_label_seeds (image_id) SELECT id FROM images WHERE id IN ({placeholders})",
                req.ids,
            )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="No images found with the given IDs")

        return {"updated": cursor.rowcount}
    finally:
        if conn:
            conn.close()


@router.patch("/images/ignore")
def update_ignored(req: IgnoreRequest):
    db_file = Path(req.db_path)
    if not db_file.is_absolute():
        raise HTTPException(status_code=400, detail="db_path must be an absolute path")
    if not db_file.exists():
        raise HTTPException(status_code=400, detail="db_path does not exist")
    if db_file.name != "kumo.db":
        raise HTTPException(status_code=400, detail="db_path must point to a kumo.db file")

    conn = None
    try:
        conn = sqlite3.connect(req.db_path)
        try:
            conn.execute("ALTER TABLE images ADD COLUMN ignored INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        placeholders = ",".join("?" * len(req.ids))
        cursor = conn.execute(
            f"UPDATE images SET ignored = ? WHERE id IN ({placeholders})",
            [int(req.ignored), *req.ids],
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="No images found with the given IDs")

        return {"updated": cursor.rowcount}
    finally:
        if conn:
            conn.close()


class RemoveClassRequest(BaseModel):
    db_path: str
    class_name: str = Field(min_length=1)


@router.post("/classes/remove")
def remove_class(req: RemoveClassRequest):
    """Clear every annotation referencing this class, so removal sticks.

    Without this, the explorer's recovery effect re-adds the class on the
    next refetch because images still carry it as a label.
    """
    validate_db_path(req.db_path)

    conn = sqlite3.connect(req.db_path)
    try:
        try:
            conn.execute("ALTER TABLE images ADD COLUMN annotation TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists in older databases

        cleared = conn.execute(
            "UPDATE images SET annotation = NULL WHERE annotation = ?",
            (req.class_name,),
        ).rowcount

        annotations_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotations'"
        ).fetchone() is not None
        deleted_boxes = (
            conn.execute(
                "DELETE FROM annotations WHERE class_name = ?",
                (req.class_name,),
            ).rowcount
            if annotations_exists
            else 0
        )

        image_labels_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='image_labels'"
        ).fetchone() is not None
        tags_deleted = (
            conn.execute(
                "DELETE FROM image_labels WHERE class_name = ?",
                (req.class_name,),
            ).rowcount
            if image_labels_exists
            else 0
        )
        conn.commit()

        return {"labels_cleared": cleared, "boxes_deleted": deleted_boxes, "tags_deleted": tags_deleted}
    finally:
        conn.close()


@router.patch("/images/no-objects")
def update_no_objects(req: NoObjectsRequest):
    db_file = Path(req.db_path)
    if not db_file.is_absolute():
        raise HTTPException(status_code=400, detail="db_path must be an absolute path")
    if not db_file.exists():
        raise HTTPException(status_code=400, detail="db_path does not exist")
    if db_file.name != "kumo.db":
        raise HTTPException(status_code=400, detail="db_path must point to a kumo.db file")

    conn = None
    try:
        conn = sqlite3.connect(req.db_path)
        try:
            conn.execute("ALTER TABLE images ADD COLUMN no_objects INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        placeholders = ",".join("?" * len(req.ids))
        cursor = conn.execute(
            f"UPDATE images SET no_objects = ? WHERE id IN ({placeholders})",
            [int(req.no_objects), *req.ids],
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="No images found with the given IDs")

        return {"updated": cursor.rowcount}
    finally:
        if conn:
            conn.close()


@router.patch("/images/split")
def update_split_override(req: SplitOverrideRequest):
    db_file = Path(req.db_path)
    if not db_file.is_absolute():
        raise HTTPException(status_code=400, detail="db_path must be an absolute path")
    if not db_file.exists():
        raise HTTPException(status_code=400, detail="db_path does not exist")
    if db_file.name != "kumo.db":
        raise HTTPException(status_code=400, detail="db_path must point to a kumo.db file")

    if req.split is not None and req.split not in {"train", "valid", "test"}:
        raise HTTPException(status_code=400, detail="split must be one of: train, valid, test, or null")

    conn = None
    try:
        conn = sqlite3.connect(req.db_path)
        try:
            conn.execute("ALTER TABLE images ADD COLUMN split_override TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        placeholders = ",".join("?" * len(req.ids))
        cursor = conn.execute(
            f"UPDATE images SET split_override = ? WHERE id IN ({placeholders})",
            [req.split, *req.ids],
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="No images found with the given IDs")

        return {"updated": cursor.rowcount}
    finally:
        if conn:
            conn.close()


@router.patch("/images/tags")
def update_tags(req: TagRequest):
    validate_db_path(req.db_path)

    conn = sqlite3.connect(req.db_path)
    try:
        _ensure_image_labels_tables(conn)

        placeholders = ",".join("?" * len(req.ids))
        valid_ids = [
            row[0]
            for row in conn.execute(
                f"SELECT id FROM images WHERE id IN ({placeholders})", req.ids
            )
        ]
        if not valid_ids:
            raise HTTPException(status_code=404, detail="No images found with the given IDs")

        # Only mutate rows for ids that actually exist in images — an unfiltered
        # id list would otherwise write orphan image_labels/image_label_seeds
        # rows for nonexistent images (no FK enforcement in sqlite by default).
        valid_placeholders = ",".join("?" * len(valid_ids))
        if req.present:
            conn.executemany(
                "INSERT OR IGNORE INTO image_labels (image_id, class_name, created_at) VALUES (?, ?, datetime('now'))",
                [(image_id, req.class_name) for image_id in valid_ids],
            )
        else:
            conn.execute(
                f"DELETE FROM image_labels WHERE image_id IN ({valid_placeholders}) AND class_name = ?",
                [*valid_ids, req.class_name],
            )

        conn.executemany(
            "INSERT OR IGNORE INTO image_label_seeds (image_id) VALUES (?)",
            [(image_id,) for image_id in valid_ids],
        )
        conn.commit()

        return {"updated": len(valid_ids)}
    finally:
        conn.close()


@router.post("/multilabel/seed")
def seed_multilabel_tags(req: SeedRequest):
    validate_db_path(req.db_path)

    conn = sqlite3.connect(req.db_path)
    try:
        _ensure_image_labels_tables(conn)

        cursor = conn.execute("""
            INSERT OR IGNORE INTO image_labels (image_id, class_name, created_at)
                SELECT id, class, datetime('now') FROM images
                WHERE class != '' AND id NOT IN (SELECT image_id FROM image_label_seeds)
        """)
        seeded = cursor.rowcount

        conn.execute(
            "INSERT OR IGNORE INTO image_label_seeds (image_id) SELECT id FROM images"
        )
        conn.commit()

        return {"seeded": seeded}
    finally:
        conn.close()


@router.get("/images/high-value")
def get_high_value_images(
    db_path: str = Query(...),
    run_id: str = Query(...),
    limit: int = Query(default=50, ge=1, le=1000),
):
    db_file = Path(db_path)
    if not db_file.is_absolute():
        raise HTTPException(status_code=400, detail="db_path must be an absolute path")
    if not db_file.exists():
        raise HTTPException(status_code=400, detail="db_path does not exist")
    if db_file.name != "kumo.db":
        raise HTTPException(status_code=400, detail="db_path must point to a kumo.db file")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """SELECT i.id, i.filename, i.path, p.al_score, p.uncertainty, p.confidence, p.predicted_class
               FROM images i
               JOIN predictions p ON i.id = p.image_id AND p.run_id = ?
               WHERE i.ignored = 0
               ORDER BY p.al_score DESC
               LIMIT ?""",
            (run_id, limit),
        ).fetchall()

        total_high_value = conn.execute(
            """SELECT COUNT(*) FROM images i
               JOIN predictions p ON i.id = p.image_id AND p.run_id = ?
               WHERE i.ignored = 0 AND p.al_score > 0.3""",
            (run_id,),
        ).fetchone()[0]

        return {
            "images": [dict(r) for r in rows],
            "total_high_value": total_high_value,
        }
    finally:
        if conn:
            conn.close()


@router.get("/image")
def get_image(file_path: str):
    resolved = _resolve_safe_image_path(file_path)
    if resolved is None:
        raise HTTPException(status_code=403, detail="Access denied")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    media_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    return FileResponse(str(resolved), media_type=media_type)


# ---------------------------------------------------------------------------
# Bounding-box annotation endpoints (object detection)
# ---------------------------------------------------------------------------


class AnnotationItem(BaseModel):
    class_name: str
    x: float
    y: float
    width: float
    height: float


class SaveAnnotationsRequest(BaseModel):
    db_path: str
    annotations: List[AnnotationItem]


class BulkAnnotationsRequest(BaseModel):
    db_path: str
    image_ids: List[int]


@router.get("/images/{image_id}/annotations")
def get_annotations(image_id: int, db_path: str = Query(...)):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
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
        rows = conn.execute(
            "SELECT id, class_name, x, y, width, height FROM annotations WHERE image_id = ?",
            (image_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@router.put("/images/{image_id}/annotations")
def save_annotations(image_id: int, req: SaveAnnotationsRequest):
    validate_db_path(req.db_path)
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(req.db_path)
    try:
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
        conn.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
        conn.executemany(
            "INSERT INTO annotations (image_id, class_name, x, y, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (image_id, a.class_name, a.x, a.y, a.width, a.height, now)
                for a in req.annotations
            ],
        )
        # Clear no_objects flag if saving non-empty annotations
        if req.annotations:
            try:
                conn.execute("UPDATE images SET no_objects = 0 WHERE id = ?", (image_id,))
            except sqlite3.OperationalError:
                pass  # Column may not exist yet in older databases
        conn.commit()
        return {"count": len(req.annotations)}
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


@router.post("/annotations/bulk")
def bulk_load_annotations(req: BulkAnnotationsRequest):
    validate_db_path(req.db_path)
    conn = sqlite3.connect(req.db_path)
    conn.row_factory = sqlite3.Row
    try:
        if not req.image_ids:
            return {"annotations": {}}
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
        placeholders = ",".join("?" * len(req.image_ids))
        rows = conn.execute(
            f"SELECT id, image_id, class_name, x, y, width, height FROM annotations WHERE image_id IN ({placeholders})",
            req.image_ids,
        ).fetchall()
        result: dict[int, list] = {img_id: [] for img_id in req.image_ids}
        for r in rows:
            row = dict(r)
            img_id = row.pop("image_id")
            result[img_id].append(row)
        return {"annotations": result}
    finally:
        conn.close()
