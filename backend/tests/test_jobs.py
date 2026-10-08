import sqlite3
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from kumo_label.main import app

client = TestClient(app)


def make_db(tmp_path, job_id="job1", status="running", updated_offset_secs=0):
    """Create a minimal kumo.db with one job row."""
    db = tmp_path / "kumo.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, status TEXT NOT NULL,
            message TEXT, progress REAL DEFAULT 0, total INTEGER DEFAULT 0,
            completed INTEGER DEFAULT 0, error TEXT, params TEXT NOT NULL,
            dedup_key TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
    """)
    updated = (datetime.utcnow() + timedelta(seconds=updated_offset_secs)).isoformat()
    conn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id, "embedding", status, None, 0.5, 100, 50, None, "{}", None, updated, updated)
    )
    conn.commit()
    conn.close()
    return str(db)


def test_get_job_returns_status(tmp_path):
    db_path = make_db(tmp_path, status="running")
    resp = client.get(f"/api/jobs/job1?db_path={db_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "job1"
    assert data["status"] == "running"
    assert data["progress"] == 0.5


def test_get_job_not_found(tmp_path):
    db_path = make_db(tmp_path)
    resp = client.get(f"/api/jobs/nonexistent?db_path={db_path}")
    assert resp.status_code == 404


def test_get_job_invalid_db_path():
    resp = client.get("/api/jobs/job1?db_path=/nonexistent/kumo.db")
    assert resp.status_code == 400


def test_get_job_marks_stale_running_job_as_failed(tmp_path):
    # updated_at is 120 seconds in the past — should be marked failed
    db_path = make_db(tmp_path, status="running", updated_offset_secs=-120)
    resp = client.get(f"/api/jobs/job1?db_path={db_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "restarted" in data["error"].lower()


def test_get_job_marks_stale_queued_job_as_failed(tmp_path):
    # queued job with updated_at 120 seconds in the past — should also be marked failed
    db_path = make_db(tmp_path, status="queued", updated_offset_secs=-120)
    resp = client.get(f"/api/jobs/job1?db_path={db_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "restarted" in data["error"].lower()


def test_get_job_done_is_not_stale(tmp_path):
    db_path = make_db(tmp_path, status="done", updated_offset_secs=-120)
    resp = client.get(f"/api/jobs/job1?db_path={db_path}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"
