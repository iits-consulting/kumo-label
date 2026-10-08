import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from kumo_label.db_utils import validate_db_path
from kumo_label.workers import executor, update_job

router = APIRouter(prefix="/api/embeddings")

SUPPORTED_MODELS = ("dinov2", "clip")


class ComputeRequest(BaseModel):
    db_path: str
    model: Literal["dinov2", "clip"]
    batch_size: int = 32


@router.get("/status")
def embedding_status(db_path: str):
    db_file = validate_db_path(db_path)
    dataset_dir = db_file.parent
    result = {}
    for model_name in SUPPORTED_MODELS:
        npy_path = dataset_dir / f"kumo_embeddings_{model_name}.npy"
        json_path = dataset_dir / f"kumo_embeddings_{model_name}.json"
        if npy_path.exists() and json_path.exists():
            image_ids = json.loads(json_path.read_text())
            created_at = datetime.utcfromtimestamp(npy_path.stat().st_mtime).isoformat()
            result[model_name] = {
                "exists": True,
                "count": len(image_ids),
                "created_at": created_at,
            }
        else:
            result[model_name] = {"exists": False}
    return result


@router.post("/compute")
def compute_embeddings(req: ComputeRequest):
    validate_db_path(req.db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    # dedup_key is stable and format-independent — no fragile LIKE matching
    dedup_key = f"{req.db_path}:{req.model}"

    conn = sqlite3.connect(req.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Deduplicate: return existing job if running or queued for same dataset + model
        existing = conn.execute(
            "SELECT id FROM jobs WHERE type='embedding' AND status IN ('running','queued') AND dedup_key=?",
            (dedup_key,),
        ).fetchone()
        if existing:
            return {"job_id": existing["id"]}

        job_id = str(uuid.uuid4())
        params = json.dumps({"model": req.model, "batch_size": req.batch_size, "db_path": req.db_path})
        conn.execute(
            "INSERT INTO jobs (id,type,status,message,progress,total,completed,error,params,dedup_key,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "embedding", "queued", None, 0.0, 0, 0, None, params, dedup_key, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    executor.submit(_run_embedding_job, job_id, req.db_path, req.model, req.batch_size)
    return {"job_id": job_id}


def _run_embedding_job(job_id: str, db_path: str, model_name: str, batch_size: int) -> None:
    from kumo_label.embeddings.extractor import extract_embeddings

    update_job(db_path, job_id, status="running", message="Starting...")

    def progress_cb(completed: int, total: int, message: str) -> None:
        progress = completed / total if total > 0 else 0.0
        update_job(
            db_path, job_id,
            status="running", progress=progress,
            completed=completed, total=total, message=message,
        )

    try:
        extract_embeddings(db_path, model_name, batch_size, progress_cb)
        update_job(db_path, job_id, status="done", progress=1.0, message="Done")
    except MemoryError:
        update_job(
            db_path, job_id, status="failed",
            error=f"Out of memory. Try a smaller batch size (current: {batch_size})",
        )
    except Exception as e:
        update_job(db_path, job_id, status="failed", error=str(e))
