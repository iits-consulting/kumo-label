import json

import pytest

from kumo_label.training.registry import get_pipeline


def test_get_pipeline_unknown_task_raises():
    with pytest.raises(ValueError):
        get_pipeline("not-a-real-task")


def test_multilabel_pipeline_registered_with_right_monitor_and_metric_names():
    pipeline = get_pipeline("multilabel-classification")
    assert pipeline.checkpoint_monitor == "val_f1"
    assert pipeline.checkpoint_mode == "max"
    assert pipeline.metric_names == {"primary": "f1", "secondary": "exact_match"}


def test_multilabel_pipeline_has_all_kwargs_fns():
    pipeline = get_pipeline("multilabel-classification")
    assert pipeline.module_kwargs_fn is not None
    assert pipeline.datamodule_kwargs_fn is not None
    assert pipeline.predict_kwargs_fn is not None
    assert pipeline.router_predict_kwargs_fn is not None


def test_classification_router_predict_kwargs_matches_old_hardcoded_dispatch():
    pipeline = get_pipeline("classification")
    run_row = {
        "id": "run-1",
        "task_type": "classification",
        "model_name": "google/vit-base-patch16-224",
        "config": json.dumps({"image_size": 224, "lr": 0.001}),
    }
    kwargs = pipeline.router_predict_kwargs_fn(
        "/tmp/kumo.db", "run-1", "/tmp/ckpt/best.ckpt", ["cat", "dog"], run_row
    )
    assert kwargs == {
        "db_path": "/tmp/kumo.db",
        "run_id": "run-1",
        "checkpoint_path": "/tmp/ckpt/best.ckpt",
        "class_names": ["cat", "dog"],
        "image_size": 224,
    }


def test_classification_router_predict_kwargs_defaults_image_size(tmp_path):
    pipeline = get_pipeline("classification")
    run_row = {"config": json.dumps({})}
    kwargs = pipeline.router_predict_kwargs_fn(
        "/tmp/kumo.db", "run-1", "/tmp/ckpt/best.ckpt", ["cat"], run_row
    )
    assert kwargs["image_size"] == 224


def test_classification_router_predict_kwargs_null_config_defaults_image_size():
    pipeline = get_pipeline("classification")
    run_row = {"config": None}
    kwargs = pipeline.router_predict_kwargs_fn(
        "/tmp/kumo.db", "run-1", "/tmp/ckpt/best.ckpt", ["cat"], run_row
    )
    assert kwargs["image_size"] == 224


def test_detection_router_predict_kwargs_matches_old_hardcoded_dispatch():
    pipeline = get_pipeline("object-detection")
    run_row = {
        "id": "run-2",
        "task_type": "object-detection",
        "model_name": "facebook/detr-resnet-50",
        "config": json.dumps({"image_size": 224}),
    }
    kwargs = pipeline.router_predict_kwargs_fn(
        "/tmp/kumo.db", "run-2", "/tmp/ckpt/best.ckpt", ["cat", "dog"], run_row
    )
    assert kwargs == {
        "db_path": "/tmp/kumo.db",
        "run_id": "run-2",
        "checkpoint_path": "/tmp/ckpt/best.ckpt",
        "class_names": ["cat", "dog"],
        "model_id": "facebook/detr-resnet-50",
    }


def test_multilabel_router_predict_kwargs_matches_expected_shape():
    pipeline = get_pipeline("multilabel-classification")
    run_row = {
        "id": "run-3",
        "task_type": "multilabel-classification",
        "model_name": "google/vit-base-patch16-224",
        "config": json.dumps({"image_size": 224, "lr": 0.001}),
    }
    kwargs = pipeline.router_predict_kwargs_fn(
        "/tmp/kumo.db", "run-3", "/tmp/ckpt/best.ckpt", ["cat", "dog"], run_row
    )
    assert kwargs == {
        "db_path": "/tmp/kumo.db",
        "run_id": "run-3",
        "checkpoint_path": "/tmp/ckpt/best.ckpt",
        "class_names": ["cat", "dog"],
        "image_size": 224,
    }


def test_multilabel_router_predict_kwargs_null_config_defaults_image_size():
    pipeline = get_pipeline("multilabel-classification")
    run_row = {"config": None}
    kwargs = pipeline.router_predict_kwargs_fn(
        "/tmp/kumo.db", "run-3", "/tmp/ckpt/best.ckpt", ["cat"], run_row
    )
    assert kwargs["image_size"] == 224
