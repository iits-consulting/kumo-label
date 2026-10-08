import json
import os
import sqlite3
import sys
import traceback

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig

from kumo_label.training.callbacks import DBMetricsCallback
from kumo_label.training.registry import get_pipeline


def _mark_failed(db_path: str, run_id: str, error: str):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute(
            """UPDATE training_runs
               SET status = 'failed', error = ?, updated_at = datetime('now')
               WHERE id = ?""",
            (error, run_id),
        )
        conn.commit()
    finally:
        conn.close()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Parse class_names — may arrive as a JSON string or OmegaConf list
    if isinstance(cfg.class_names, str):
        class_names = json.loads(cfg.class_names)
    else:
        class_names = list(cfg.class_names)

    task_type = getattr(cfg, "task_type", "classification")
    pipeline = get_pipeline(task_type)

    try:
        datamodule = pipeline.datamodule_cls(**pipeline.datamodule_kwargs_fn(cfg, class_names))
        module_kwargs = pipeline.module_kwargs_fn(cfg)

        # For weighted loss, compute class counts from the training set.
        # Only applies to classification: multi-hot / multi-label datasets
        # can't be Counter()-ed the same way (labels are tensors, not ints).
        if task_type == "classification" and getattr(cfg.training, "class_balance_strategy", "none") == "weighted_loss":
            from collections import Counter
            datamodule.setup()
            label_counts = Counter(datamodule.train_dataset.labels)
            module_kwargs["class_counts"] = [
                label_counts.get(i, 0) for i in range(len(class_names))
            ]

        module = pipeline.module_cls(**module_kwargs)

        callbacks = [
            DBMetricsCallback(
                db_path=cfg.db_path,
                run_id=cfg.run_id,
                metric_names=pipeline.metric_names,
            ),
            ModelCheckpoint(
                dirpath=f"{cfg.output_dir}/checkpoints",
                filename="best",
                monitor=pipeline.checkpoint_monitor,
                mode=pipeline.checkpoint_mode,
                save_top_k=1,
            ),
        ]

        if cfg.training.early_stopping:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                patience=cfg.training.early_stopping_patience,
                mode="min",
                verbose=True,
            ))

        ckpt_path = getattr(cfg, "checkpoint_path", None)
        if ckpt_path and os.path.exists(ckpt_path):
            module.load_backbone_from_checkpoint(ckpt_path)

        trainer = L.Trainer(
            max_epochs=cfg.training.epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices=1,
            default_root_dir=cfg.output_dir,
            log_every_n_steps=1,
            precision=cfg.training.precision,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            gradient_clip_val=cfg.training.gradient_clip_val if cfg.training.gradient_clip_val > 0 else None,
            val_check_interval=cfg.training.val_check_interval,
            limit_train_batches=cfg.training.limit_train_batches if cfg.training.limit_train_batches > 0 else 1.0,
        )

        trainer.fit(module, datamodule)

        # Run predictions on all images using the best checkpoint
        ckpt_callback = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
        best_ckpt_path = ckpt_callback.best_model_path or f"{cfg.output_dir}/checkpoints/best.ckpt"

        predict_kwargs = pipeline.predict_kwargs_fn(cfg, class_names, best_ckpt_path)
        pipeline.predict_fn(**predict_kwargs)

        # Mark run as done after predictions complete
        conn = sqlite3.connect(cfg.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """UPDATE training_runs
               SET status = 'done', finished_at = datetime('now'), updated_at = datetime('now')
               WHERE id = ?""",
            (cfg.run_id,),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        _mark_failed(cfg.db_path, cfg.run_id, traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
