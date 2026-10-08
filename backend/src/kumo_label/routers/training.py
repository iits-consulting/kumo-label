import json
import os
import shutil
import subprocess
import sys
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from kumo_label.db_utils import validate_db_path
from kumo_label.training.registry import get_pipeline

router = APIRouter(prefix="/api/training", tags=["training"])

_active_processes: dict[str, subprocess.Popen] = {}


def _check_orphaned(run: dict, db_path: str) -> dict:
    """If a run claims to be queued/running but has no active process, mark it failed."""
    if run["status"] not in ("queued", "running"):
        return run

    run_id = run["id"]
    if run_id in _active_processes:
        proc = _active_processes[run_id]
        if proc.poll() is not None:
            if proc.returncode != 0:
                stderr_path = os.path.join(run.get("output_dir") or "", "stderr.log")
                stderr = ""
                if os.path.exists(stderr_path):
                    with open(stderr_path) as f:
                        stderr = f.read()
                error_msg = f"Process exited with code {proc.returncode}: {stderr}"
                conn = sqlite3.connect(db_path, timeout=30)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "UPDATE training_runs SET status = 'failed', error = ?, updated_at = datetime('now') WHERE id = ?",
                    (error_msg, run_id),
                )
                conn.commit()
                conn.close()
                run["status"] = "failed"
                run["error"] = error_msg
            del _active_processes[run_id]
    else:
        error_msg = "Orphaned run (server restarted)"
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "UPDATE training_runs SET status = 'failed', error = ?, updated_at = datetime('now') WHERE id = ?",
            (error_msg, run_id),
        )
        conn.commit()
        conn.close()
        run["status"] = "failed"
        run["error"] = error_msg

    return run


MODEL_CONFIGS = {
    "vit_base": ("google/vit-base-patch16-224", "ViT-Base"),
    "resnet50": ("microsoft/resnet-50", "ResNet-50"),
    "efficientnet_b0": ("google/efficientnet-b0", "EfficientNet-B0"),
    "detr": ("facebook/detr-resnet-50", "DETR"),
}


class TrainingStartRequest(BaseModel):
    db_path: str
    model: str
    task_type: str = "classification"
    model_id: Optional[str] = None
    local_checkpoint_path: Optional[str] = None
    checkpoint_run_id: Optional[str] = None
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    augmentation: bool = True
    split_ratio: float = 0.8
    validation_mode: str = "auto"
    class_names: list[str] = []
    num_classes: int = 0
    # Advanced settings
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    horizontal_flip: bool = True
    rotation: float = 15.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    image_size: int = 224
    weight_decay: float = 0.01
    freeze_backbone: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 5
    precision: str = "32-true"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.0
    val_check_interval: float = 1.0
    limit_train_batches: int = 0
    class_balance_strategy: str = "none"


@router.post("/start")
def start_training(req: TrainingStartRequest):
    validate_db_path(req.db_path)

    checkpoint_path = None
    if req.model == "checkpoint":
        if not req.checkpoint_run_id:
            raise HTTPException(400, "checkpoint_run_id required for checkpoint model")
        conn_tmp = sqlite3.connect(req.db_path, timeout=30)
        conn_tmp.row_factory = sqlite3.Row
        parent_run = conn_tmp.execute(
            "SELECT * FROM training_runs WHERE id = ?", (req.checkpoint_run_id,)
        ).fetchone()
        conn_tmp.close()
        if not parent_run:
            raise HTTPException(404, "Referenced training run not found")
        parent_output_dir = parent_run["output_dir"]
        checkpoint_path = os.path.join(parent_output_dir, "checkpoints", "best.ckpt")
        if not os.path.exists(checkpoint_path):
            raise HTTPException(400, "Checkpoint file not found for the referenced training run")
        model_name = parent_run["model_name"]
        display_name = f"{parent_run['display_name']} (fine-tuned)"
    elif req.model == "local_checkpoint":
        if not req.local_checkpoint_path:
            raise HTTPException(400, "local_checkpoint_path is required")
        if not os.path.isfile(req.local_checkpoint_path):
            raise HTTPException(400, f"Checkpoint file not found: {req.local_checkpoint_path}")
        backbone = req.model_id
        if not backbone:
            # Try to auto-detect from embedded hyperparameters
            import torch as _torch
            try:
                ckpt_data = _torch.load(req.local_checkpoint_path, map_location="cpu", weights_only=False)
                backbone = ckpt_data.get("hyper_parameters", {}).get("model_id")
            except Exception:
                pass
        if not backbone:
            raise HTTPException(400, "Could not determine backbone architecture. Please select it manually.")
        if backbone in MODEL_CONFIGS:
            model_name, _ = MODEL_CONFIGS[backbone]
        else:
            model_name = backbone
        checkpoint_path = req.local_checkpoint_path
        display_name = f"{Path(req.local_checkpoint_path).stem} (ckpt)"
    elif req.model == "custom":
        if not req.model_id:
            raise HTTPException(400, "model_id required for custom model")
        model_name = req.model_id
        display_name = "Custom"
    elif req.model in MODEL_CONFIGS:
        model_name, display_name = MODEL_CONFIGS[req.model]
    else:
        raise HTTPException(400, f"Unknown model: {req.model}")

    dataset_dir = str(Path(req.db_path).parent)
    run_id = str(uuid.uuid4())
    output_dir = os.path.join(dataset_dir, "kumo_training_runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    conn = sqlite3.connect(req.db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    label_count = conn.execute(
        "SELECT COUNT(*) FROM images WHERE ignored = 0 AND COALESCE(annotation, class) IS NOT NULL"
    ).fetchone()[0]

    # Snapshot dataset state for this training run
    snapshot_rows = conn.execute(
        "SELECT id, COALESCE(annotation, class) AS effective_label, ignored FROM images"
    ).fetchall()
    label_counts: dict[str, int] = {}
    for sr in snapshot_rows:
        lbl = sr[1]
        if lbl and not sr[2]:  # exclude ignored images
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    dataset_snapshot = json.dumps({
        "image_count": sum(1 for sr in snapshot_rows if not sr[2]),
        "ignored_count": sum(1 for sr in snapshot_rows if sr[2]),
        "label_counts": label_counts,
        "images": [{"id": sr[0], "label": sr[1], "ignored": bool(sr[2])} for sr in snapshot_rows],
    })

    config = {
        "model": req.model,
        "model_name": model_name,
        "image_size": req.image_size,
        "lr": req.lr,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "augmentation": {
            "enabled": req.augmentation,
            "horizontal_flip": req.horizontal_flip,
            "rotation": req.rotation,
            "color_jitter_brightness": req.color_jitter_brightness,
            "color_jitter_contrast": req.color_jitter_contrast,
            "color_jitter_saturation": req.color_jitter_saturation,
        },
        "split_ratio": req.split_ratio,
        "validation_mode": req.validation_mode,
        "optimizer": req.optimizer,
        "scheduler": req.scheduler,
        "warmup_epochs": req.warmup_epochs,
        "weight_decay": req.weight_decay,
        "freeze_backbone": req.freeze_backbone,
        "early_stopping": req.early_stopping,
        "early_stopping_patience": req.early_stopping_patience,
        "precision": req.precision,
        "accumulate_grad_batches": req.accumulate_grad_batches,
        "gradient_clip_val": req.gradient_clip_val,
        "val_check_interval": req.val_check_interval,
        "limit_train_batches": req.limit_train_batches,
        "class_balance_strategy": req.class_balance_strategy,
    }

    try:
        pipeline = get_pipeline(req.task_type)
    except ValueError as e:
        conn.close()
        raise HTTPException(400, str(e))
    primary_metric_name = pipeline.metric_names.get("primary", "accuracy")
    secondary_metric_name = pipeline.metric_names.get("secondary", "f1")

    conn.execute(
        """INSERT INTO training_runs
           (id, status, task_type, model_name, display_name, config, num_classes, class_names,
            label_count, epochs, current_epoch, output_dir,
            dataset_snapshot, primary_metric_name, secondary_metric_name, created_at, updated_at)
           VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)""",
        (run_id, req.task_type, model_name, display_name, json.dumps(config), req.num_classes,
         json.dumps(req.class_names), label_count, req.epochs, output_dir,
         dataset_snapshot, primary_metric_name, secondary_metric_name, now, now),
    )
    conn.commit()
    conn.close()

    overrides = [
        f"db_path={req.db_path}",
        f"run_id={run_id}",
        f"output_dir={output_dir}",
        f"task_type={req.task_type}",
        f"num_classes={req.num_classes}",
        f"class_names={json.dumps(req.class_names)}",
        f"split_ratio={req.split_ratio}",
        f"validation_mode={req.validation_mode}",
        # Training
        f"training.lr={req.lr}",
        f"training.epochs={req.epochs}",
        f"training.batch_size={req.batch_size}",
        f"training.optimizer={req.optimizer}",
        f"training.scheduler={req.scheduler}",
        f"training.warmup_epochs={req.warmup_epochs}",
        f"training.weight_decay={req.weight_decay}",
        f"training.freeze_backbone={str(req.freeze_backbone).lower()}",
        f"training.early_stopping={str(req.early_stopping).lower()}",
        f"training.early_stopping_patience={req.early_stopping_patience}",
        f"training.precision={req.precision}",
        f"training.accumulate_grad_batches={req.accumulate_grad_batches}",
        f"training.gradient_clip_val={req.gradient_clip_val}",
        f"training.val_check_interval={req.val_check_interval}",
        f"training.limit_train_batches={req.limit_train_batches}",
        f"training.class_balance_strategy={req.class_balance_strategy}",
        # Augmentation — always use 'default' config, override individual fields
        "augmentation=default",
        f"augmentation.enabled={str(req.augmentation).lower()}",
        f"augmentation.horizontal_flip={str(req.horizontal_flip).lower()}",
        f"augmentation.rotation={req.rotation}",
        f"augmentation.color_jitter.brightness={req.color_jitter_brightness}",
        f"augmentation.color_jitter.contrast={req.color_jitter_contrast}",
        f"augmentation.color_jitter.saturation={req.color_jitter_saturation}",
    ]

    if req.model in ("checkpoint", "local_checkpoint"):
        overrides.extend([
            f"model.hf_model_id={model_name}",
            f"model.display_name='{display_name}'",
            f"model.image_size={req.image_size}",
            f"checkpoint_path={checkpoint_path}",
        ])
    elif req.model == "custom" and req.model_id:
        overrides.extend([
            f"model.hf_model_id={req.model_id}",
            "model.display_name=Custom",
            f"model.image_size={req.image_size}",
        ])
    else:
        overrides.append(f"model={req.model}")
        overrides.append(f"model.image_size={req.image_size}")

    stdout_log = open(os.path.join(output_dir, "stdout.log"), "w")
    stderr_log = open(os.path.join(output_dir, "stderr.log"), "w")
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "kumo_label.training.train"] + overrides,
        stdout=stdout_log,
        stderr=stderr_log,
        env=env,
    )
    _active_processes[run_id] = proc

    return {"run_id": run_id}


@router.get("/runs/{run_id}")
def get_training_run(run_id: str, db_path: str = Query(...)):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Training run not found")

    result = dict(row)
    result = _check_orphaned(result, db_path)

    result["class_names"] = json.loads(result["class_names"]) if result["class_names"] else []
    result["config"] = json.loads(result["config"]) if result["config"] else {}
    val_per_class = result.get("val_per_class")
    result["val_per_class"] = json.loads(val_per_class) if val_per_class else None
    if result["output_dir"]:
        result["has_checkpoint"] = os.path.exists(
            os.path.join(result["output_dir"], "checkpoints", "best.ckpt")
        )
    else:
        result["has_checkpoint"] = False
    return result


@router.get("/runs/{run_id}/metrics")
def get_training_metrics(run_id: str, db_path: str = Query(...)):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM training_metrics WHERE run_id = ? ORDER BY step", (run_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/runs/{run_id}/predictions")
def get_predictions(run_id: str, db_path: str = Query(...)):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    # Ensure columns exist for older databases
    for col, col_type, default in [
        ("uncertainty", "REAL", "0"),
        ("al_score", "REAL", "0"),
        ("is_unlabeled", "INTEGER", "0"),
        ("is_wrong", "INTEGER", "0"),
        ("class_imbalance_score", "REAL", "0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type} NOT NULL DEFAULT {default}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    rows = conn.execute(
        "SELECT image_id, predicted_class, confidence, uncertainty, al_score, top_predictions, is_unlabeled, is_wrong, class_imbalance_score FROM predictions WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "image_id": r["image_id"],
            "predicted_class": r["predicted_class"],
            "confidence": r["confidence"],
            "uncertainty": r["uncertainty"],
            "al_score": r["al_score"],
            "top_predictions": json.loads(r["top_predictions"]),
            "is_unlabeled": bool(r["is_unlabeled"]),
            "is_wrong": bool(r["is_wrong"]),
            "class_imbalance_score": r["class_imbalance_score"],
        }
        for r in rows
    ]


@router.get("/runs/{run_id}/detection-predictions")
def get_detection_predictions(
    run_id: str,
    db_path: str = Query(...),
    image_id: Optional[int] = Query(default=None),
):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        if image_id is not None:
            rows = conn.execute(
                "SELECT class_name, confidence, x, y, width, height FROM detection_predictions WHERE run_id = ? AND image_id = ?",
                (run_id, image_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT image_id, class_name, confidence, x, y, width, height FROM detection_predictions WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


class BulkDetectionPredictionsRequest(BaseModel):
    db_path: str
    image_ids: list[int]


@router.post("/runs/{run_id}/detection-predictions/bulk")
def bulk_detection_predictions(run_id: str, req: BulkDetectionPredictionsRequest):
    validate_db_path(req.db_path)
    conn = sqlite3.connect(req.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        if not req.image_ids:
            return {"predictions": {}}
        result: dict[int, list[dict]] = {img_id: [] for img_id in req.image_ids}
        CHUNK = 500
        for i in range(0, len(req.image_ids), CHUNK):
            batch = req.image_ids[i : i + CHUNK]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT image_id, class_name, confidence, x, y, width, height "
                f"FROM detection_predictions WHERE run_id = ? AND image_id IN ({placeholders})",
                (run_id, *batch),
            ).fetchall()
            for r in rows:
                row = dict(r)
                result[row.pop("image_id")].append(row)
        return {"predictions": result}
    except sqlite3.OperationalError:
        return {"predictions": {}}
    finally:
        conn.close()


@router.get("/runs/{run_id}/snapshot")
def get_dataset_snapshot(run_id: str, db_path: str = Query(...)):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT dataset_snapshot FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, "Training run not found")

    if not row["dataset_snapshot"]:
        conn.close()
        return {"snapshot": None, "diff": None, "has_changes": False}

    snapshot = json.loads(row["dataset_snapshot"])

    # Build current state for diff
    current_rows = conn.execute(
        "SELECT id, COALESCE(annotation, class) AS effective_label, ignored FROM images"
    ).fetchall()
    conn.close()

    current_map = {r["id"]: {"label": r["effective_label"], "ignored": bool(r["ignored"])} for r in current_rows}
    snap_map = {img["id"]: img for img in snapshot.get("images", [])}

    new_labels = 0
    changed_labels = 0
    new_ignored = 0
    removed_ignored = 0
    new_images = 0

    for img_id, curr in current_map.items():
        snap = snap_map.get(img_id)
        if snap is None:
            new_images += 1
            continue
        if curr["label"] and not snap.get("label"):
            new_labels += 1
        elif curr["label"] != snap.get("label") and snap.get("label"):
            changed_labels += 1
        if curr["ignored"] and not snap.get("ignored"):
            new_ignored += 1
        elif not curr["ignored"] and snap.get("ignored"):
            removed_ignored += 1

    has_changes = any([new_labels, changed_labels, new_ignored, removed_ignored, new_images])

    return {
        "snapshot": {
            "image_count": snapshot.get("image_count", 0),
            "ignored_count": snapshot.get("ignored_count", 0),
            "label_counts": snapshot.get("label_counts", {}),
        },
        "diff": {
            "new_labels": new_labels,
            "changed_labels": changed_labels,
            "new_ignored": new_ignored,
            "removed_ignored": removed_ignored,
            "new_images": new_images,
        },
        "has_changes": has_changes,
    }


@router.get("/runs")
def list_training_runs(db_path: str = Query(...), task_type: str | None = Query(None)):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    if task_type:
        rows = conn.execute(
            "SELECT * FROM training_runs WHERE task_type = ? ORDER BY created_at DESC",
            (task_type,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY created_at DESC"
        ).fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        r["class_names"] = json.loads(r["class_names"]) if r["class_names"] else []
        r["config"] = json.loads(r["config"]) if r["config"] else {}
        val_per_class = r.get("val_per_class")
        r["val_per_class"] = json.loads(val_per_class) if val_per_class else None
        if r["output_dir"]:
            r["has_checkpoint"] = os.path.exists(
                os.path.join(r["output_dir"], "checkpoints", "best.ckpt")
            )
        else:
            r["has_checkpoint"] = False
        r = _check_orphaned(r, db_path)
        results.append(r)
    return results


@router.get("/runs/{run_id}/config.yaml")
def download_run_config(run_id: str, db_path: str = Query(...)):
    """Return the run's stored hyperparameters as a YAML download."""
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT config, display_name, model_name, task_type, num_classes, class_names, epochs, created_at "
        "FROM training_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "Training run not found")

    cfg = json.loads(row["config"]) if row["config"] else {}
    document = {
        "run_id": run_id,
        "display_name": row["display_name"],
        "model_name": row["model_name"],
        "task_type": row["task_type"],
        "num_classes": row["num_classes"],
        "class_names": json.loads(row["class_names"]) if row["class_names"] else [],
        "epochs": row["epochs"],
        "created_at": row["created_at"],
        "config": cfg,
    }

    import yaml
    body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    from fastapi.responses import Response
    return Response(
        content=body,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id[:8]}.yaml"'},
    )


@router.get("/runs/{run_id}/download")
def download_training_model(run_id: str, db_path: str = Query(...)):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    row = conn.execute(
        "SELECT output_dir FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        raise HTTPException(404, "Training run not found")
    ckpt_path = Path(row[0]) / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        raise HTTPException(404, "Checkpoint file not found")
    return FileResponse(
        str(ckpt_path),
        media_type="application/octet-stream",
        filename=f"model_{run_id[:8]}.ckpt",
    )


@router.get("/runs/{run_id}/export")
def export_training_model(
    run_id: str,
    background_tasks: BackgroundTasks,
    db_path: str = Query(...),
):
    """Export a trained model as a portable HuggingFace bundle.

    Produces a zip containing `save_pretrained()` layout (config.json +
    model.safetensors), the matching image processor, and `kumo_meta.json`
    with task/class metadata. Consumers load it with
    `AutoModelForImageClassification.from_pretrained(path)`.
    """
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT output_dir, task_type, model_name, num_classes, class_names, config "
        "FROM training_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    if not row or not row["output_dir"]:
        raise HTTPException(404, "Training run not found")
    ckpt_path = Path(row["output_dir"]) / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        raise HTTPException(404, "Checkpoint file not found")

    from transformers import AutoImageProcessor

    task_type = row["task_type"]
    model_id = row["model_name"]
    class_names = json.loads(row["class_names"]) if row["class_names"] else []
    cfg = json.loads(row["config"]) if row["config"] else {}

    pipeline = get_pipeline(task_type)
    module = pipeline.module_cls.load_from_checkpoint(str(ckpt_path), map_location="cpu")

    tmp_root = Path(tempfile.mkdtemp(prefix=f"kumo_export_{run_id[:8]}_"))
    bundle_dir = tmp_root / "bundle"
    bundle_dir.mkdir()

    module.model.save_pretrained(bundle_dir)
    processor = getattr(module, "processor", None) or AutoImageProcessor.from_pretrained(model_id)
    processor.save_pretrained(bundle_dir)

    meta = {
        "task_type": task_type,
        "model_id": model_id,
        "num_classes": row["num_classes"],
        "class_names": class_names,
        "image_size": cfg.get("image_size"),
        "run_id": run_id,
    }
    (bundle_dir / "kumo_meta.json").write_text(json.dumps(meta, indent=2))

    zip_path = shutil.make_archive(
        str(tmp_root / f"kumo_model_{run_id[:8]}"),
        "zip",
        root_dir=str(bundle_dir),
    )

    background_tasks.add_task(shutil.rmtree, tmp_root, ignore_errors=True)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"kumo_model_{run_id[:8]}.zip",
    )


class ProbeCheckpointRequest(BaseModel):
    path: str


@router.post("/probe-checkpoint")
def probe_checkpoint(req: ProbeCheckpointRequest):
    """Read hyper_parameters from a Lightning checkpoint and return the embedded model_id."""
    if not os.path.isfile(req.path):
        raise HTTPException(400, f"File not found: {req.path}")
    import torch as _torch
    try:
        ckpt = _torch.load(req.path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise HTTPException(400, f"Failed to read checkpoint: {exc}")
    hparams = ckpt.get("hyper_parameters", {})
    model_id = hparams.get("model_id")
    if not model_id:
        return {"model_id": None, "display_name": None}
    display_name = model_id
    for cfg_model_id, cfg_display_name in MODEL_CONFIGS.values():
        if cfg_model_id == model_id:
            display_name = cfg_display_name
            break
    return {"model_id": model_id, "display_name": display_name}


class RenameRunRequest(BaseModel):
    display_name: str


@router.patch("/runs/{run_id}")
def rename_training_run(run_id: str, req: RenameRunRequest, db_path: str = Query(...)):
    validate_db_path(db_path)
    name = req.display_name.strip()
    if not name:
        raise HTTPException(400, "display_name must not be empty")
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    result = conn.execute(
        "UPDATE training_runs SET display_name = ?, updated_at = datetime('now') WHERE id = ?",
        (name, run_id),
    )
    conn.commit()
    conn.close()
    if result.rowcount == 0:
        raise HTTPException(404, "Training run not found")
    return {"status": "ok"}


@router.delete("/runs/{run_id}")
def delete_training_run(run_id: str, db_path: str = Query(...)):
    validate_db_path(db_path)

    # Fetch output_dir before deleting the row
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT output_dir FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()
    conn.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))
    try:
        conn.execute("DELETE FROM detection_predictions WHERE run_id = ?", (run_id,))
    except sqlite3.OperationalError:
        pass  # Table may not exist yet
    conn.execute("DELETE FROM training_metrics WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()

    # Delete local output folder
    if row and row["output_dir"]:
        output_dir = row["output_dir"]
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

    return {"status": "deleted"}



@router.post("/runs/{run_id}/stop")
def stop_training(run_id: str, db_path: str = Query(...)):
    if run_id in _active_processes:
        proc = _active_processes[run_id]
        proc.terminate()
        del _active_processes[run_id]

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """UPDATE training_runs
           SET status = 'stopped', error = NULL, updated_at = datetime('now')
           WHERE id = ? AND status IN ('queued', 'running', 'predicting')""",
        (run_id,),
    )
    conn.commit()
    conn.close()

    return {"status": "stopped"}


@router.post("/runs/{run_id}/predict")
def generate_predictions(
    run_id: str,
    db_path: str = Query(...),
    only_new: bool = Query(default=False),
):
    validate_db_path(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, "Training run not found")

    allowed_statuses = ("stopped", "failed", "done") if only_new else ("stopped", "failed")
    if row["status"] not in allowed_statuses:
        conn.close()
        raise HTTPException(400, f"Cannot generate predictions for run with status '{row['status']}'")

    output_dir = row["output_dir"]
    ckpt_path = os.path.join(output_dir, "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        conn.close()
        raise HTTPException(400, "No checkpoint found. Training must complete at least one epoch.")

    class_names = json.loads(row["class_names"]) if row["class_names"] else []
    task_type = row["task_type"] or "classification"
    original_status = row["status"]
    # sqlite3.Row objects must not cross into the background-thread closure
    # after conn.close() below — capture a plain dict while conn is open.
    row_dict = dict(row)

    image_ids_filter: list[int] | None = None
    if only_new:
        rows = conn.execute(
            "SELECT id FROM images WHERE id NOT IN (SELECT image_id FROM predictions WHERE run_id = ?)",
            (run_id,),
        ).fetchall()
        image_ids_filter = [r[0] for r in rows]
        if not image_ids_filter:
            conn.close()
            return {"status": "noop", "predicted": 0}

    conn.execute(
        "UPDATE training_runs SET status = 'predicting', error = NULL, current_step = 0, total_steps = 0, updated_at = datetime('now') WHERE id = ?",
        (run_id,),
    )
    conn.commit()
    conn.close()

    from kumo_label.workers import executor

    def _run():
        try:
            pipeline = get_pipeline(task_type)
            predict_kwargs = pipeline.router_predict_kwargs_fn(
                db_path, run_id, ckpt_path, class_names, row_dict
            )
            pipeline.predict_fn(**predict_kwargs, image_ids_filter=image_ids_filter)
            c = sqlite3.connect(db_path, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(
                "UPDATE training_runs SET status = 'done', finished_at = datetime('now'), updated_at = datetime('now') WHERE id = ?",
                (run_id,),
            )
            c.commit()
            c.close()
        except Exception as exc:
            c = sqlite3.connect(db_path, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(
                "UPDATE training_runs SET status = ?, error = ?, updated_at = datetime('now') WHERE id = ?",
                (original_status, f"Prediction failed: {exc}", run_id),
            )
            c.commit()
            c.close()

    executor.submit(_run)
    return {"status": "predicting"}
