import json
import sqlite3
from unittest.mock import MagicMock
from kumo_label.training.callbacks import DBMetricsCallback


def _create_training_db(tmp_path, run_id="test-run"):
    db_path = str(tmp_path / "kumo.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE training_runs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, model_name TEXT NOT NULL,
            display_name TEXT NOT NULL, config TEXT, num_classes INTEGER NOT NULL,
            class_names TEXT, label_count INTEGER DEFAULT 0, epochs INTEGER NOT NULL,
            current_epoch INTEGER DEFAULT 0, current_step INTEGER DEFAULT 0,
            total_steps INTEGER DEFAULT 0, train_loss REAL, val_loss REAL,
            val_accuracy REAL, val_f1 REAL, best_accuracy REAL, best_f1 REAL,
            val_auroc REAL, val_precision REAL, val_recall REAL, best_auroc REAL,
            error TEXT, output_dir TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, finished_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE training_metrics (
            run_id TEXT NOT NULL, epoch INTEGER NOT NULL,
            step INTEGER NOT NULL DEFAULT 0,
            train_loss REAL, val_loss REAL, val_accuracy REAL, val_f1 REAL,
            learning_rate REAL, val_auroc REAL, val_precision REAL, val_recall REAL,
            PRIMARY KEY (run_id, step)
        )
    """)
    conn.execute(
        """INSERT INTO training_runs (id, status, model_name, display_name, num_classes, epochs, created_at, updated_at)
           VALUES (?, 'running', 'test-model', 'Test', 2, 5, datetime('now'), datetime('now'))""",
        (run_id,),
    )
    conn.commit()
    conn.close()
    return db_path


def _mock_trainer(epoch, metrics):
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.callback_metrics = metrics
    trainer.optimizers = [MagicMock()]
    trainer.optimizers[0].param_groups = [{"lr": 0.001}]
    trainer.global_step = epoch * 10
    trainer.estimated_stepping_batches = 50
    trainer.max_epochs = 5
    return trainer


def test_db_callback_writes_epoch_metrics(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")

    trainer = _mock_trainer(0, {"train_loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.8, "val_f1": 0.75})
    cb.on_validation_epoch_end(trainer, object())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM training_metrics WHERE run_id = 'test-run' AND epoch = 0").fetchone()
    conn.close()
    assert row is not None
    assert row["run_id"] == "test-run"
    assert row["epoch"] == 0
    assert row["step"] == 0
    assert abs(row["train_loss"] - 0.5) < 0.01


def test_db_callback_updates_run_status(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")

    trainer = _mock_trainer(2, {"train_loss": 0.3, "val_loss": 0.2, "val_accuracy": 0.9, "val_f1": 0.88})
    cb.on_validation_epoch_end(trainer, object())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert run["current_epoch"] == 3
    assert abs(run["val_accuracy"] - 0.9) < 0.01
    assert abs(run["best_accuracy"] - 0.9) < 0.01


def test_db_callback_tracks_best_metrics(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")

    trainer = _mock_trainer(0, {"train_loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.8, "val_f1": 0.7})
    cb.on_validation_epoch_end(trainer, object())

    trainer = _mock_trainer(1, {"train_loss": 0.3, "val_loss": 0.2, "val_accuracy": 0.9, "val_f1": 0.85})
    cb.on_validation_epoch_end(trainer, object())

    trainer = _mock_trainer(2, {"train_loss": 0.25, "val_loss": 0.22, "val_accuracy": 0.85, "val_f1": 0.8})
    cb.on_validation_epoch_end(trainer, object())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert abs(run["best_accuracy"] - 0.9) < 0.01
    assert abs(run["best_f1"] - 0.85) < 0.01


def test_db_callback_writes_step_progress(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")

    trainer = _mock_trainer(2, {"train_loss": 0.3, "val_loss": 0.2, "val_accuracy": 0.9, "val_f1": 0.88})
    trainer.global_step = 25
    trainer.estimated_stepping_batches = 50
    cb.on_validation_epoch_end(trainer, object())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert run["current_step"] == 25
    assert run["total_steps"] == 50


def test_db_callback_on_fit_end(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")
    cb.on_fit_end(MagicMock(), MagicMock())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert run["status"] == "predicting"


def test_db_callback_on_exception(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")
    cb.on_exception(MagicMock(), MagicMock(), RuntimeError("CUDA OOM"))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert run["status"] == "failed"
    assert "CUDA OOM" in run["error"]


def test_db_callback_persists_per_class_metrics(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")

    per_class = {
        "classes": ["cat", "dog"],
        "precision": [0.9, 0.8],
        "recall": [0.7, 0.6],
        "f1": [0.75, 0.65],
        "support": [10, 5],
    }

    class _ModuleStub:
        _per_class_metrics = per_class

    trainer = _mock_trainer(0, {"train_loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.8, "val_f1": 0.75})
    cb.on_validation_epoch_end(trainer, _ModuleStub())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert json.loads(run["val_per_class"]) == per_class


def test_db_callback_no_per_class_metrics_leaves_column_null(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(db_path=db_path, run_id="test-run")

    trainer = _mock_trainer(0, {"train_loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.8, "val_f1": 0.75})
    cb.on_validation_epoch_end(trainer, object())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert run["val_per_class"] is None


def test_db_callback_primary_secondary_metric_name_mapping_for_multilabel(tmp_path):
    db_path = _create_training_db(tmp_path)
    cb = DBMetricsCallback(
        db_path=db_path, run_id="test-run",
        metric_names={"primary": "f1", "secondary": "exact_match"},
    )

    trainer = _mock_trainer(0, {
        "train_loss": 0.5, "val_loss": 0.4,
        "val_f1": 0.77, "val_exact_match": 0.55,
    })
    cb.on_validation_epoch_end(trainer, object())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM training_runs WHERE id = 'test-run'").fetchone()
    conn.close()
    assert abs(run["val_accuracy"] - 0.77) < 0.01
    assert abs(run["val_f1"] - 0.55) < 0.01
