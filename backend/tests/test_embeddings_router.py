import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from kumo_label.main import app

client = TestClient(app)


def make_dataset(tmp_path: Path) -> str:
    """Minimal dataset with kumo.db, images table, and jobs/projections tables."""
    db_path = tmp_path / "kumo.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE images (id INTEGER PRIMARY KEY, filename TEXT, path TEXT, class TEXT, split TEXT);
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, status TEXT NOT NULL,
            message TEXT, progress REAL DEFAULT 0, total INTEGER DEFAULT 0,
            completed INTEGER DEFAULT 0, error TEXT, params TEXT NOT NULL,
            dedup_key TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE projections (
            param_hash TEXT PRIMARY KEY, embedding_hash TEXT NOT NULL,
            model TEXT NOT NULL, method TEXT NOT NULL, params TEXT NOT NULL,
            coords TEXT NOT NULL, created_at TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO images VALUES (1,'a.jpg','/tmp/a.jpg','cat','train')")
    conn.commit()
    conn.close()
    return str(db_path)


def write_fake_embeddings(tmp_path: Path, model: str = "dinov2"):
    arr = np.zeros((1, 384), dtype=np.float32)
    np.save(str(tmp_path / f"kumo_embeddings_{model}.npy"), arr)
    (tmp_path / f"kumo_embeddings_{model}.json").write_text(json.dumps([1]))


# --- Status endpoint ---

def test_status_no_embeddings(tmp_path):
    db_path = make_dataset(tmp_path)
    resp = client.get(f"/api/embeddings/status?db_path={db_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dinov2"]["exists"] is False
    assert data["clip"]["exists"] is False


def test_status_with_embeddings(tmp_path):
    db_path = make_dataset(tmp_path)
    write_fake_embeddings(tmp_path, "dinov2")
    resp = client.get(f"/api/embeddings/status?db_path={db_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dinov2"]["exists"] is True
    assert data["dinov2"]["count"] == 1
    assert "created_at" in data["dinov2"]


def test_status_invalid_db_path():
    resp = client.get("/api/embeddings/status?db_path=/nonexistent/kumo.db")
    assert resp.status_code == 400


# --- Compute endpoint ---

def test_compute_starts_job(tmp_path):
    db_path = make_dataset(tmp_path)
    with patch("kumo_label.routers.embeddings._run_embedding_job"):
        resp = client.post(
            "/api/embeddings/compute",
            json={"db_path": db_path, "model": "dinov2", "batch_size": 32},
        )
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_compute_deduplicates_running_job(tmp_path):
    db_path = make_dataset(tmp_path)
    # Pre-insert a running job for same model
    conn = sqlite3.connect(db_path)
    now = "2026-01-01T00:00:00"
    conn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("existing-id", "embedding", "running", None, 0, 0, 0, None,
         json.dumps({"model": "dinov2", "db_path": db_path}),
         f"{db_path}:dinov2",  # dedup_key
         now, now),
    )
    conn.commit()
    conn.close()

    with patch("kumo_label.routers.embeddings._run_embedding_job"):
        resp = client.post(
            "/api/embeddings/compute",
            json={"db_path": db_path, "model": "dinov2", "batch_size": 32},
        )
    assert resp.json()["job_id"] == "existing-id"


def test_compute_invalid_model(tmp_path):
    db_path = make_dataset(tmp_path)
    resp = client.post(
        "/api/embeddings/compute",
        json={"db_path": db_path, "model": "ijepa", "batch_size": 32},
    )
    assert resp.status_code == 422
