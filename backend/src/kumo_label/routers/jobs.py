import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kumo_label.db_utils import validate_db_path

router = APIRouter(prefix="/api/jobs")

STALE_SECONDS = 60


@router.get("/{job_id}")
def get_job(job_id: str, db_path: str):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")

        job = dict(row)

        if job["status"] in ("running", "queued"):
            updated = datetime.fromisoformat(job["updated_at"])
            elapsed = (datetime.now(timezone.utc).replace(tzinfo=None) - updated).total_seconds()
            if elapsed > STALE_SECONDS:
                now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                conn.execute(
                    "UPDATE jobs SET status='failed', error='Server restarted during job', updated_at=? WHERE id=?",
                    (now, job_id),
                )
                conn.commit()
                job["status"] = "failed"
                job["error"] = "Server restarted during job"

        return job
    finally:
        conn.close()


class CancelRequest(BaseModel):
    db_path: str


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, req: CancelRequest):
    """Mark a queued/running job as cancelled.

    The worker thread cannot be hard-killed mid-compute, but its result will
    be discarded (the job runner checks the status before storing output).
    """
    validate_db_path(req.db_path)
    conn = sqlite3.connect(req.db_path, timeout=30)
    try:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if row[0] not in ("queued", "running"):
            return {"status": row[0]}
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        conn.execute(
            "UPDATE jobs SET status='cancelled', updated_at=? WHERE id=?",
            (now, job_id),
        )
        conn.commit()
        return {"status": "cancelled"}
    finally:
        conn.close()
