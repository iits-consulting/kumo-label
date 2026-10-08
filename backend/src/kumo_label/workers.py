import atexit
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

executor = ThreadPoolExecutor(max_workers=1)
atexit.register(executor.shutdown, wait=True)

_ALLOWED_JOB_FIELDS = frozenset({"status", "message", "progress", "total", "completed", "error"})


def update_job(db_path: str, job_id: str, **kwargs) -> None:
    """Update arbitrary job fields plus updated_at timestamp.

    Uses timeout=30 so concurrent access from the background thread and the
    FastAPI thread does not immediately raise OperationalError. WAL mode
    (set during DB creation in scanner.py) allows readers and the writer
    to proceed concurrently.
    """
    unknown = set(kwargs) - _ALLOWED_JOB_FIELDS
    if unknown:
        raise ValueError(f"update_job: unknown fields {unknown}")

    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    fields = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [now, job_id]
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        # Once cancelled, ignore subsequent worker updates so the status sticks.
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is not None and row[0] == "cancelled":
            return
        conn.execute(f"UPDATE jobs SET {fields}, updated_at=? WHERE id=?", values)
        conn.commit()
    finally:
        conn.close()
