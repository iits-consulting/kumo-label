import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from kumo_label.db_utils import validate_db_path
from kumo_label.workers import executor, update_job

router = APIRouter(prefix="/api/projections")


class ComputeRequest(BaseModel):
    db_path: str
    model: Literal["dinov2", "clip"]
    method: Literal["umap", "tsne", "pca"]
    params: dict = {}


def compute_embedding_hash(npy_path: Path) -> str:
    """Fast fingerprint: sha256(file_size + file_mtime + first 4 KB)."""
    stat = npy_path.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime).encode())
    with open(npy_path, "rb") as f:
        h.update(f.read(4096))
    return h.hexdigest()


def compute_param_hash(embedding_hash: str, model: str, method: str, params: dict) -> str:
    """sha256(embedding_hash + model + method + sorted params JSON)."""
    h = hashlib.sha256()
    h.update(embedding_hash.encode())
    h.update(model.encode())
    h.update(method.encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()


@router.get("/check")
def check_projection(
    db_path: str,
    model: Literal["dinov2", "clip"],
    method: Literal["umap", "tsne", "pca"],
    params: str = "{}",
):
    """Check if a projection is cached without fetching coords."""
    db_file = validate_db_path(db_path)
    npy_path = db_file.parent / f"kumo_embeddings_{model}.npy"

    if not npy_path.exists():
        raise HTTPException(status_code=400, detail=f"No embeddings found for model {model}")

    parsed_params = json.loads(params)
    embedding_hash = compute_embedding_hash(npy_path)
    param_hash = compute_param_hash(embedding_hash, model, method, parsed_params)

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        row = conn.execute(
            "SELECT 1 FROM projections WHERE param_hash=?", (param_hash,)
        ).fetchone()
        return {"cached": row is not None, "param_hash": param_hash}
    finally:
        conn.close()


@router.post("/compute")
def compute_projection(req: ComputeRequest):
    db_file = validate_db_path(req.db_path)
    npy_path = db_file.parent / f"kumo_embeddings_{req.model}.npy"

    if not npy_path.exists():
        raise HTTPException(status_code=400, detail=f"No embeddings found for model {req.model}")

    embedding_hash = compute_embedding_hash(npy_path)
    param_hash = compute_param_hash(embedding_hash, req.model, req.method, req.params)

    conn = sqlite3.connect(req.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Cache hit — respond without inlining coords. The client fetches them
        # via GET /projections/{hash}/blob (binary, ~3-5x smaller than JSON).
        cached = conn.execute(
            "SELECT 1 FROM projections WHERE param_hash=?", (param_hash,)
        ).fetchone()
        if cached:
            return {"cached": True, "param_hash": param_hash}

        # Deduplicate in-flight jobs — dedup_key is param_hash (format-independent, no LIKE)
        existing = conn.execute(
            "SELECT id FROM jobs WHERE type='projection' AND status IN ('running','queued') AND dedup_key=?",
            (param_hash,),
        ).fetchone()
        if existing:
            return {"cached": False, "param_hash": param_hash, "job_id": existing["id"]}

        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        job_id = str(uuid.uuid4())
        job_params = json.dumps({
            "db_path": req.db_path,
            "model": req.model,
            "method": req.method,
            "params": req.params,
            "param_hash": param_hash,
            "embedding_hash": embedding_hash,
        })
        conn.execute(
            "INSERT INTO jobs (id,type,status,message,progress,total,completed,error,params,dedup_key,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "projection", "queued", None, 0.0, 0, 0, None, job_params, param_hash, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    executor.submit(
        _run_projection_job,
        job_id, req.db_path, req.model, req.method, req.params, param_hash, embedding_hash,
    )
    return {"cached": False, "param_hash": param_hash, "job_id": job_id}


@router.get("/{param_hash}")
def get_projection(param_hash: str, db_path: str):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT coords FROM projections WHERE param_hash=?", (param_hash,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Projection not found")
        return {"coords": json.loads(row["coords"])}
    finally:
        conn.close()


# Packed binary projection payload — much smaller than JSON for large datasets.
# Layout (little-endian):
#   bytes  0..3   magic 'KUMP'
#   bytes  4..7   uint32 version (1)
#   bytes  8..11  uint32 count
#   bytes 12..15  uint32 reserved
#   bytes 16..    uint32[count] ids
#                 float32[count] x
#                 float32[count] y
PROJECTION_BLOB_MAGIC = b"KUMP"
PROJECTION_BLOB_VERSION = 1


@router.get("/{param_hash}/blob")
def get_projection_blob(param_hash: str, db_path: str):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT coords FROM projections WHERE param_hash=?", (param_hash,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Projection not found")
        coords = json.loads(row["coords"])
    finally:
        conn.close()

    n = len(coords)
    ids = np.fromiter((c["id"] for c in coords), dtype=np.uint32, count=n)
    xs = np.fromiter((c["x"] for c in coords), dtype=np.float32, count=n)
    ys = np.fromiter((c["y"] for c in coords), dtype=np.float32, count=n)

    header = (
        PROJECTION_BLOB_MAGIC
        + np.uint32(PROJECTION_BLOB_VERSION).tobytes()
        + np.uint32(n).tobytes()
        + np.uint32(0).tobytes()
    )
    body = ids.tobytes() + xs.tobytes() + ys.tobytes()
    return Response(content=header + body, media_type="application/octet-stream")


def _job_status(db_path: str, job_id: str) -> str | None:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _run_projection_job(
    job_id: str,
    db_path: str,
    model: str,
    method: str,
    params: dict,
    param_hash: str,
    embedding_hash: str,
) -> None:
    from kumo_label.embeddings.reducer import reduce_embeddings

    # Job may have been cancelled while queued.
    if _job_status(db_path, job_id) == "cancelled":
        return

    update_job(db_path, job_id, status="running", message="Loading embeddings...")
    try:
        db_file = Path(db_path)
        dataset_dir = db_file.parent
        embeddings = np.load(str(dataset_dir / f"kumo_embeddings_{model}.npy"))
        image_ids = json.loads((dataset_dir / f"kumo_embeddings_{model}.json").read_text())

        update_job(db_path, job_id, status="running", message=f"Running {method.upper()}...", total=len(image_ids))

        coords_array = reduce_embeddings(embeddings, method, params)

        # The compute itself can't be interrupted; check after it finishes
        # and skip persisting if the user cancelled in the meantime.
        if _job_status(db_path, job_id) == "cancelled":
            return

        coords = [
            {"id": image_ids[i], "x": float(coords_array[i, 0]), "y": float(coords_array[i, 1])}
            for i in range(len(image_ids))
        ]

        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        store_conn = sqlite3.connect(db_path)
        try:
            store_conn.execute(
                "INSERT OR REPLACE INTO projections"
                " (param_hash,embedding_hash,model,method,params,coords,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (param_hash, embedding_hash, model, method, json.dumps(params), json.dumps(coords), now),
            )
            store_conn.commit()
        finally:
            store_conn.close()

        update_job(db_path, job_id, status="done", progress=1.0, completed=len(image_ids), message="Done")
    except Exception as e:
        update_job(db_path, job_id, status="failed", error=str(e))
