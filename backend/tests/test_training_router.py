import json
import os
import sqlite3
from PIL import Image
from fastapi.testclient import TestClient
from kumo_label.main import app
from kumo_label.scanner import scan_dataset

client = TestClient(app)


def _setup_dataset(tmp_path, num_images=10):
    """Create a minimal labeled dataset."""
    for cls in ["cat", "dog"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(num_images // 2):
            img_path = cls_dir / f"{cls}_{i}.jpg"
            Image.new("RGB", (64, 64), color="red").save(str(img_path))

    result = scan_dataset(str(tmp_path))
    db_path = result["db_path"]

    # Label all images
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE images SET annotation = class")
    conn.commit()
    conn.close()

    return db_path


def test_list_runs_empty(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.get(f"/api/training/runs?db_path={db_path}")
    assert resp.status_code == 200
    assert resp.json() == []


def test_start_training_creates_run(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.post("/api/training/start", json={
        "db_path": db_path,
        "model": "vit_base",
        "lr": 0.001,
        "epochs": 1,
        "batch_size": 4,
        "augmentation": False,
        "split_ratio": 0.8,
        "class_names": ["cat", "dog"],
        "num_classes": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data

    # Verify DB row was created
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = ?", (data["run_id"],)).fetchone()
    conn.close()
    assert run is not None
    assert run["status"] == "queued"
    assert run["model_name"] == "google/vit-base-patch16-224"
    assert run["display_name"] == "ViT-Base"
    assert run["task_type"] == "classification"


def test_list_runs_filtered_by_task_type(tmp_path):
    db_path = _setup_dataset(tmp_path)
    client.post("/api/training/start", json={
        "db_path": db_path,
        "model": "vit_base",
        "task_type": "classification",
        "epochs": 1, "batch_size": 4,
        "class_names": ["cat", "dog"], "num_classes": 2,
    })
    client.post("/api/training/start", json={
        "db_path": db_path,
        "model": "vit_base",
        "task_type": "object-detection",
        "epochs": 1, "batch_size": 4,
        "class_names": ["cat", "dog"], "num_classes": 2,
    })

    # Unfiltered returns both
    all_runs = client.get(f"/api/training/runs?db_path={db_path}").json()
    assert len(all_runs) == 2

    # Filtered returns only matching
    cls_runs = client.get(f"/api/training/runs?db_path={db_path}&task_type=classification").json()
    assert len(cls_runs) == 1
    assert cls_runs[0]["task_type"] == "classification"

    det_runs = client.get(f"/api/training/runs?db_path={db_path}&task_type=object-detection").json()
    assert len(det_runs) == 1
    assert det_runs[0]["task_type"] == "object-detection"


def test_start_training_multilabel_creates_run_with_f1_exact_match_metric_names(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.post("/api/training/start", json={
        "db_path": db_path,
        "model": "vit_base",
        "task_type": "multilabel-classification",
        "lr": 0.001,
        "epochs": 1,
        "batch_size": 4,
        "augmentation": False,
        "split_ratio": 0.8,
        "class_names": ["cat", "dog"],
        "num_classes": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = ?", (data["run_id"],)).fetchone()
    conn.close()
    assert run is not None
    assert run["task_type"] == "multilabel-classification"
    assert run["primary_metric_name"] == "f1"
    assert run["secondary_metric_name"] == "exact_match"


def test_start_training_unknown_task_type_returns_400(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.post("/api/training/start", json={
        "db_path": db_path,
        "model": "vit_base",
        "task_type": "not-a-real-task",
        "epochs": 1, "batch_size": 4,
        "class_names": ["cat", "dog"], "num_classes": 2,
    })
    assert resp.status_code == 400


def test_start_training_invalid_model(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.post("/api/training/start", json={
        "db_path": db_path,
        "model": "nonexistent",
        "class_names": ["cat", "dog"],
        "num_classes": 2,
    })
    assert resp.status_code == 400


def test_get_run_not_found(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.get(f"/api/training/runs/nonexistent?db_path={db_path}")
    assert resp.status_code == 404


def _seed_detection_predictions(db_path: str, run_id: str, image_ids_with_preds: list[int]) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL
        )
        """
    )
    for img_id in image_ids_with_preds:
        conn.execute(
            "INSERT INTO detection_predictions (run_id, image_id, class_name, confidence, x, y, width, height)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, img_id, "cat", 0.9, 0.1, 0.2, 0.3, 0.4),
        )
    conn.commit()
    conn.close()


def test_bulk_detection_predictions_returns_grouped_results(tmp_path):
    db_path = _setup_dataset(tmp_path)
    run_id = "test-run-bulk"
    _seed_detection_predictions(db_path, run_id, [1, 2, 3])

    resp = client.post(
        f"/api/training/runs/{run_id}/detection-predictions/bulk",
        json={"db_path": db_path, "image_ids": [1, 2, 3, 99, 100]},
    )
    assert resp.status_code == 200
    data = resp.json()
    preds = data["predictions"]
    # Pydantic returns int keys as strings over JSON
    assert set(preds.keys()) == {"1", "2", "3", "99", "100"}
    assert len(preds["1"]) == 1
    assert preds["1"][0]["class_name"] == "cat"
    assert preds["99"] == []
    assert preds["100"] == []


def test_bulk_detection_predictions_empty_input(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.post(
        "/api/training/runs/any-run/detection-predictions/bulk",
        json={"db_path": db_path, "image_ids": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {"predictions": {}}


def test_bulk_detection_predictions_chunks_large_input(tmp_path):
    db_path = _setup_dataset(tmp_path)
    run_id = "test-run-chunked"
    seeded = list(range(1, 1201))  # 1200 ids, exercises chunking (CHUNK=500)
    _seed_detection_predictions(db_path, run_id, seeded)

    resp = client.post(
        f"/api/training/runs/{run_id}/detection-predictions/bulk",
        json={"db_path": db_path, "image_ids": seeded},
    )
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 1200
    assert all(len(preds[str(i)]) == 1 for i in seeded)


def test_stop_run(tmp_path):
    db_path = _setup_dataset(tmp_path)
    resp = client.post("/api/training/start", json={
        "db_path": db_path,
        "model": "resnet50",
        "epochs": 1,
        "batch_size": 4,
        "class_names": ["cat", "dog"],
        "num_classes": 2,
    })
    run_id = resp.json()["run_id"]

    resp = client.post(f"/api/training/runs/{run_id}/stop?db_path={db_path}")
    assert resp.status_code == 200

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    assert run["status"] == "stopped"
    assert run["error"] is None


def _seed_run_with_checkpoint(
    db_path, tmp_path, run_id, task_type, model_name, image_size, class_names, status="stopped",
):
    output_dir = tmp_path / f"run_output_{run_id}"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best.ckpt").write_bytes(b"fake-checkpoint")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """INSERT INTO training_runs
           (id, status, task_type, model_name, display_name, config, num_classes, class_names,
            label_count, epochs, current_epoch, output_dir, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'Test', ?, ?, ?, 0, 1, 0, ?, datetime('now'), datetime('now'))""",
        (run_id, status, task_type, model_name, json.dumps({"image_size": image_size}),
         len(class_names), json.dumps(class_names), str(output_dir)),
    )
    conn.commit()
    conn.close()
    return str(output_dir)


def test_predict_endpoint_dispatches_classification_kwargs_via_registry(tmp_path, monkeypatch):
    db_path = _setup_dataset(tmp_path)
    run_id = "predict-cls"
    output_dir = _seed_run_with_checkpoint(
        db_path, tmp_path, run_id, "classification",
        "google/vit-base-patch16-224", 224, ["cat", "dog"],
    )

    calls = []
    import kumo_label.training.registry as registry_module
    monkeypatch.setattr(
        registry_module.TASK_REGISTRY["classification"], "predict_fn",
        lambda **kwargs: calls.append(kwargs),
    )

    resp = client.post(f"/api/training/runs/{run_id}/predict?db_path={db_path}")
    assert resp.status_code == 200
    assert resp.json() == {"status": "predicting"}

    from kumo_label.workers import executor as _executor
    _executor.submit(lambda: None).result(timeout=10)

    assert calls == [{
        "db_path": db_path,
        "run_id": run_id,
        "checkpoint_path": os.path.join(output_dir, "checkpoints", "best.ckpt"),
        "class_names": ["cat", "dog"],
        "image_size": 224,
        "image_ids_filter": None,
    }]


def test_predict_endpoint_dispatches_detection_kwargs_via_registry(tmp_path, monkeypatch):
    db_path = _setup_dataset(tmp_path)
    run_id = "predict-det"
    output_dir = _seed_run_with_checkpoint(
        db_path, tmp_path, run_id, "object-detection",
        "facebook/detr-resnet-50", 224, ["cat", "dog"],
    )

    calls = []
    import kumo_label.training.registry as registry_module
    monkeypatch.setattr(
        registry_module.TASK_REGISTRY["object-detection"], "predict_fn",
        lambda **kwargs: calls.append(kwargs),
    )

    resp = client.post(f"/api/training/runs/{run_id}/predict?db_path={db_path}")
    assert resp.status_code == 200
    assert resp.json() == {"status": "predicting"}

    from kumo_label.workers import executor as _executor
    _executor.submit(lambda: None).result(timeout=10)

    assert calls == [{
        "db_path": db_path,
        "run_id": run_id,
        "checkpoint_path": os.path.join(output_dir, "checkpoints", "best.ckpt"),
        "class_names": ["cat", "dog"],
        "model_id": "facebook/detr-resnet-50",
        "image_ids_filter": None,
    }]


def test_get_training_run_parses_val_per_class(tmp_path):
    db_path = _setup_dataset(tmp_path)
    run_id = "run-with-per-class"
    _seed_run_with_checkpoint(
        db_path, tmp_path, run_id, "multilabel-classification",
        "google/vit-base-patch16-224", 224, ["cat", "dog"],
    )
    per_class = {"classes": ["cat", "dog"], "precision": [0.9, 0.8],
                 "recall": [0.7, 0.6], "f1": [0.75, 0.65], "support": [10, 5]}
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE training_runs SET val_per_class = ? WHERE id = ?",
        (json.dumps(per_class), run_id),
    )
    conn.commit()
    conn.close()

    resp = client.get(f"/api/training/runs/{run_id}?db_path={db_path}")
    assert resp.status_code == 200
    assert resp.json()["val_per_class"] == per_class

    # list endpoint applies the same parsing
    list_resp = client.get(f"/api/training/runs?db_path={db_path}")
    matching = [r for r in list_resp.json() if r["id"] == run_id]
    assert matching[0]["val_per_class"] == per_class
