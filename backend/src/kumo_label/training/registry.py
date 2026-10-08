from dataclasses import dataclass, field
from typing import Callable

import lightning as L


@dataclass
class TaskPipeline:
    module_cls: type[L.LightningModule]
    datamodule_cls: type[L.LightningDataModule]
    predict_fn: Callable
    checkpoint_monitor: str                 # e.g. "val_accuracy" or "val_mAP"
    checkpoint_mode: str = "max"            # "max" or "min"
    metric_names: dict[str, str] = field(default_factory=dict)  # {"primary": "accuracy", "secondary": "f1"}
    # Callables that extract the right kwargs from the Hydra config for each component.
    # This avoids passing mismatched kwargs to constructors that don't expect them.
    module_kwargs_fn: Callable | None = None
    datamodule_kwargs_fn: Callable | None = None
    predict_kwargs_fn: Callable | None = None
    # (db_path, run_id, ckpt_path, class_names, run_row: dict) -> dict.
    # Produces the kwargs for predict_fn (everything except image_ids_filter)
    # from a training_runs row, for the on-demand re-prediction router endpoint.
    router_predict_kwargs_fn: Callable | None = None


TASK_REGISTRY: dict[str, TaskPipeline] = {}


def register_task(task_type: str, pipeline: TaskPipeline):
    TASK_REGISTRY[task_type] = pipeline


def get_pipeline(task_type: str) -> TaskPipeline:
    if task_type not in TASK_REGISTRY:
        raise ValueError(f"Unknown task type: {task_type}. Registered: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_type]


def _parse_run_config(run_row: dict) -> dict:
    import json as _json
    raw = run_row.get("config")
    return _json.loads(raw) if raw else {}


# --- Kwargs extraction functions for classification ---

def _classification_module_kwargs(cfg) -> dict:
    import json as _json
    class_names = _json.loads(cfg.class_names) if isinstance(cfg.class_names, str) else list(cfg.class_names)
    return dict(
        model_id=cfg.model.hf_model_id,
        num_classes=cfg.num_classes,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler,
        weight_decay=cfg.training.weight_decay,
        freeze_backbone=cfg.training.freeze_backbone,
        class_names=class_names,
        class_balance_strategy=cfg.training.class_balance_strategy,
    )


def _classification_datamodule_kwargs(cfg, class_names: list[str]) -> dict:
    return dict(
        db_path=cfg.db_path,
        class_names=class_names,
        image_size=cfg.model.image_size,
        batch_size=cfg.training.batch_size,
        split_ratio=cfg.split_ratio,
        validation_mode=cfg.validation_mode,
        augmentation_cfg=cfg.augmentation,
        class_balance_strategy=cfg.training.class_balance_strategy,
    )


def _classification_predict_kwargs(cfg, class_names: list[str], checkpoint_path: str) -> dict:
    return dict(
        db_path=cfg.db_path,
        run_id=cfg.run_id,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        image_size=cfg.model.image_size,
    )


def _classification_router_predict_kwargs(db_path, run_id, ckpt_path, class_names, run_row: dict) -> dict:
    return dict(
        db_path=db_path,
        run_id=run_id,
        checkpoint_path=ckpt_path,
        class_names=class_names,
        image_size=_parse_run_config(run_row).get("image_size", 224),
    )


# --- Kwargs extraction functions for object detection ---

def _detection_module_kwargs(cfg) -> dict:
    return dict(
        model_id=cfg.model.hf_model_id,
        num_classes=cfg.num_classes,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler,
        weight_decay=cfg.training.weight_decay,
        freeze_backbone=cfg.training.freeze_backbone,
    )


def _detection_datamodule_kwargs(cfg, class_names: list[str]) -> dict:
    return dict(
        db_path=cfg.db_path,
        class_names=class_names,
        model_id=cfg.model.hf_model_id,
        batch_size=cfg.training.batch_size,
        split_ratio=cfg.split_ratio,
        validation_mode=cfg.validation_mode,
        augmentation_cfg=cfg.augmentation,
    )


def _detection_predict_kwargs(cfg, class_names: list[str], checkpoint_path: str) -> dict:
    return dict(
        db_path=cfg.db_path,
        run_id=cfg.run_id,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        model_id=cfg.model.hf_model_id,
    )


def _detection_router_predict_kwargs(db_path, run_id, ckpt_path, class_names, run_row: dict) -> dict:
    return dict(
        db_path=db_path,
        run_id=run_id,
        checkpoint_path=ckpt_path,
        class_names=class_names,
        model_id=run_row["model_name"],
    )


# --- Kwargs extraction functions for multilabel classification ---

def _multilabel_module_kwargs(cfg) -> dict:
    import json as _json
    class_names = _json.loads(cfg.class_names) if isinstance(cfg.class_names, str) else list(cfg.class_names)
    return dict(
        model_id=cfg.model.hf_model_id,
        num_classes=cfg.num_classes,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler,
        weight_decay=cfg.training.weight_decay,
        freeze_backbone=cfg.training.freeze_backbone,
        class_names=class_names,
    )


def _multilabel_datamodule_kwargs(cfg, class_names: list[str]) -> dict:
    return dict(
        db_path=cfg.db_path,
        class_names=class_names,
        image_size=cfg.model.image_size,
        batch_size=cfg.training.batch_size,
        split_ratio=cfg.split_ratio,
        validation_mode=cfg.validation_mode,
        augmentation_cfg=cfg.augmentation,
    )


def _multilabel_predict_kwargs(cfg, class_names: list[str], checkpoint_path: str) -> dict:
    return dict(
        db_path=cfg.db_path,
        run_id=cfg.run_id,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        image_size=cfg.model.image_size,
    )


def _multilabel_router_predict_kwargs(db_path, run_id, ckpt_path, class_names, run_row: dict) -> dict:
    return dict(
        db_path=db_path,
        run_id=run_id,
        checkpoint_path=ckpt_path,
        class_names=class_names,
        image_size=_parse_run_config(run_row).get("image_size", 224),
    )


# --- Register both tasks ---

def _register_defaults():
    from kumo_label.training.module import KumoClassifier
    from kumo_label.training.datamodule import KumoDataModule
    from kumo_label.training.predict import run_predictions
    from kumo_label.training.detection_module import KumoDetector
    from kumo_label.training.detection_datamodule import KumoDetectionDataModule
    from kumo_label.training.detection_predict import run_detection_predictions
    from kumo_label.training.multilabel_module import KumoMultilabelClassifier
    from kumo_label.training.multilabel_datamodule import KumoMultilabelDataModule
    from kumo_label.training.multilabel_predict import run_multilabel_predictions

    register_task("classification", TaskPipeline(
        module_cls=KumoClassifier,
        datamodule_cls=KumoDataModule,
        predict_fn=run_predictions,
        checkpoint_monitor="val_accuracy",
        metric_names={"primary": "accuracy", "secondary": "f1"},
        module_kwargs_fn=_classification_module_kwargs,
        datamodule_kwargs_fn=_classification_datamodule_kwargs,
        predict_kwargs_fn=_classification_predict_kwargs,
        router_predict_kwargs_fn=_classification_router_predict_kwargs,
    ))

    register_task("object-detection", TaskPipeline(
        module_cls=KumoDetector,
        datamodule_cls=KumoDetectionDataModule,
        predict_fn=run_detection_predictions,
        checkpoint_monitor="val_mAP",
        metric_names={"primary": "mAP", "secondary": "mAP@50"},
        module_kwargs_fn=_detection_module_kwargs,
        datamodule_kwargs_fn=_detection_datamodule_kwargs,
        predict_kwargs_fn=_detection_predict_kwargs,
        router_predict_kwargs_fn=_detection_router_predict_kwargs,
    ))

    register_task("multilabel-classification", TaskPipeline(
        module_cls=KumoMultilabelClassifier,
        datamodule_cls=KumoMultilabelDataModule,
        predict_fn=run_multilabel_predictions,
        checkpoint_monitor="val_f1",
        checkpoint_mode="max",
        metric_names={"primary": "f1", "secondary": "exact_match"},
        module_kwargs_fn=_multilabel_module_kwargs,
        datamodule_kwargs_fn=_multilabel_datamodule_kwargs,
        predict_kwargs_fn=_multilabel_predict_kwargs,
        router_predict_kwargs_fn=_multilabel_router_predict_kwargs,
    ))


_register_defaults()
