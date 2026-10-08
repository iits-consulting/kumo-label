import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from kumo_label.main import app

client = TestClient(app)


def make_db(tmp_path: Path) -> str:
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


def write_embeddings(tmp_path: Path, model: str = "dinov2") -> Path:
    arr = np.zeros((1, 384), dtype=np.float32)
    npy = tmp_path / f"kumo_embeddings_{model}.npy"
    np.save(str(npy), arr)
    (tmp_path / f"kumo_embeddings_{model}.json").write_text(json.dumps([1]))
    return npy


def test_compute_returns_cache_miss_and_job_id(tmp_path):
    db_path = make_db(tmp_path)
    write_embeddings(tmp_path)
    with patch("kumo_label.routers.projections._run_projection_job"):
        resp = client.post(
            "/api/projections/compute",
            json={"db_path": db_path, "model": "dinov2", "method": "pca", "params": {}},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cached"] is False
    assert "job_id" in data
    assert "param_hash" in data


def test_compute_returns_cache_hit(tmp_path):
    db_path = make_db(tmp_path)
    npy = write_embeddings(tmp_path)

    # Pre-compute the hash the same way the router does
    from kumo_label.routers.projections import compute_embedding_hash, compute_param_hash
    emb_hash = compute_embedding_hash(npy)
    p_hash = compute_param_hash(emb_hash, "dinov2", "pca", {})
    coords = [{"id": 1, "x": 0.1, "y": 0.2}]

    conn = sqlite3.connect(db_path)
    now = "2026-01-01T00:00:00"
    conn.execute(
        "INSERT INTO projections VALUES (?,?,?,?,?,?,?)",
        (p_hash, emb_hash, "dinov2", "pca", "{}", json.dumps(coords), now),
    )
    conn.commit()
    conn.close()

    resp = client.post(
        "/api/projections/compute",
        json={"db_path": db_path, "model": "dinov2", "method": "pca", "params": {}},
    )
    data = resp.json()
    assert data["cached"] is True
    assert data["param_hash"] == p_hash

    # Cache hit no longer inlines coords — clients fetch the binary blob.
    blob_resp = client.get(
        f"/api/projections/{p_hash}/blob", params={"db_path": db_path}
    )
    assert blob_resp.status_code == 200
    assert blob_resp.headers["content-type"] == "application/octet-stream"
    body = blob_resp.content
    assert body[:4] == b"KUMP"
    count = int.from_bytes(body[8:12], "little")
    assert count == len(coords)
    first_id = int.from_bytes(body[16:20], "little")
    assert first_id == coords[0]["id"]


def test_compute_no_embeddings_returns_400(tmp_path):
    db_path = make_db(tmp_path)
    resp = client.post(
        "/api/projections/compute",
        json={"db_path": db_path, "model": "dinov2", "method": "pca", "params": {}},
    )
    assert resp.status_code == 400


def test_get_projection_not_found(tmp_path):
    db_path = make_db(tmp_path)
    resp = client.get(f"/api/projections/badhash?db_path={db_path}")
    assert resp.status_code == 404


def test_get_projection_returns_coords(tmp_path):
    db_path = make_db(tmp_path)
    coords = [{"id": 1, "x": 0.5, "y": -0.3}]
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO projections VALUES (?,?,?,?,?,?,?)",
        ("myhash", "emb123", "dinov2", "pca", "{}", json.dumps(coords), "2026-01-01"),
    )
    conn.commit()
    conn.close()
    resp = client.get(f"/api/projections/myhash?db_path={db_path}")
    assert resp.status_code == 200
    assert resp.json()["coords"] == coords


def test_compute_deduplicates_in_flight_job(tmp_path):
    db_path = make_db(tmp_path)
    npy = write_embeddings(tmp_path)

    from kumo_label.routers.projections import compute_embedding_hash, compute_param_hash
    emb_hash = compute_embedding_hash(npy)
    p_hash = compute_param_hash(emb_hash, "dinov2", "pca", {})

    conn = sqlite3.connect(db_path)
    now = "2026-01-01T00:00:00"
    conn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("existing-proj-job", "projection", "running", None, 0, 0, 0, None,
         json.dumps({"param_hash": p_hash}),
         p_hash,  # dedup_key
         now, now),
    )
    conn.commit()
    conn.close()

    with patch("kumo_label.routers.projections._run_projection_job"):
        resp = client.post(
            "/api/projections/compute",
            json={"db_path": db_path, "model": "dinov2", "method": "pca", "params": {}},
        )
    data = resp.json()
    assert data["cached"] is False
    assert data["job_id"] == "existing-proj-job"


def test_check_cached_true(tmp_path):
    db_path = make_db(tmp_path)
    npy = write_embeddings(tmp_path)

    from kumo_label.routers.projections import compute_embedding_hash, compute_param_hash
    emb_hash = compute_embedding_hash(npy)
    p_hash = compute_param_hash(emb_hash, "dinov2", "pca", {})
    coords = [{"id": 1, "x": 0.1, "y": 0.2}]

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO projections VALUES (?,?,?,?,?,?,?)",
        (p_hash, emb_hash, "dinov2", "pca", "{}", json.dumps(coords), "2026-01-01"),
    )
    conn.commit()
    conn.close()

    resp = client.get(
        f"/api/projections/check?db_path={db_path}&model=dinov2&method=pca&params={{}}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cached"] is True
    assert data["param_hash"] == p_hash


def test_check_cached_false(tmp_path):
    db_path = make_db(tmp_path)
    write_embeddings(tmp_path)

    resp = client.get(
        f"/api/projections/check?db_path={db_path}&model=dinov2&method=pca&params={{}}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cached"] is False
    assert "param_hash" in data


def test_check_no_embeddings_returns_400(tmp_path):
    db_path = make_db(tmp_path)
    resp = client.get(
        f"/api/projections/check?db_path={db_path}&model=dinov2&method=pca&params={{}}"
    )
    assert resp.status_code == 400
