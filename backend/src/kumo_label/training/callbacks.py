import json
import sqlite3
import traceback

import lightning as L


class DBMetricsCallback(L.Callback):
    """Writes per-epoch metrics to kumo.db for frontend polling."""

    def __init__(self, db_path: str, run_id: str, metric_names: dict[str, str] | None = None):
        self.db_path = db_path
        self.run_id = run_id
        # metric_names maps {"primary": "accuracy", "secondary": "f1"} etc.
        # The Lightning metric keys are derived as "val_<metric_name>" with
        # special handling for names containing uppercase (e.g. "mAP" -> "val_mAP").
        if metric_names is None:
            metric_names = {"primary": "accuracy", "secondary": "f1"}
        self.primary_metric_key = self._metric_key(metric_names.get("primary", "accuracy"))
        self.secondary_metric_key = self._metric_key(metric_names.get("secondary", "f1"))
        self._migrated = False

    @staticmethod
    def _metric_key(name: str) -> str:
        """Convert a metric display name to the Lightning logged key.

        Convention: classification logs val_accuracy, val_f1;
        detection logs val_mAP, val_mAP_50.
        """
        key_map = {
            "accuracy": "val_accuracy",
            "f1": "val_f1",
            "mAP": "val_mAP",
            "mAP@50": "val_mAP_50",
        }
        return key_map.get(name, f"val_{name}")

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        if not self._migrated:
            for table, cols in [
                ("training_metrics", ["val_auroc", "val_precision", "val_recall"]),
                ("training_runs", ["val_auroc", "val_precision", "val_recall", "best_auroc", "current_step", "total_steps"]),
            ]:
                for col in cols:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
                    except sqlite3.OperationalError:
                        pass
            try:
                conn.execute("ALTER TABLE training_runs ADD COLUMN val_per_class TEXT")
            except sqlite3.OperationalError:
                pass
            # Migrate training_metrics to use step-based primary key so that
            # sub-epoch validations (val_check_interval < 1.0) are not collapsed.
            cursor = conn.execute("PRAGMA table_info(training_metrics)")
            col_names = [row[1] for row in cursor.fetchall()]
            if "step" not in col_names:
                conn.execute("""CREATE TABLE training_metrics_new (
                    run_id        TEXT NOT NULL REFERENCES training_runs(id),
                    epoch         INTEGER NOT NULL,
                    step          INTEGER NOT NULL DEFAULT 0,
                    train_loss    REAL,
                    val_loss      REAL,
                    val_accuracy  REAL,
                    val_f1        REAL,
                    learning_rate REAL,
                    val_auroc     REAL,
                    val_precision REAL,
                    val_recall    REAL,
                    PRIMARY KEY (run_id, step)
                )""")
                conn.execute("""INSERT OR IGNORE INTO training_metrics_new
                    (run_id, epoch, step, train_loss, val_loss, val_accuracy,
                     val_f1, learning_rate, val_auroc, val_precision, val_recall)
                    SELECT run_id, epoch, epoch, train_loss, val_loss, val_accuracy,
                           val_f1, learning_rate, val_auroc, val_precision, val_recall
                    FROM training_metrics""")
                conn.execute("DROP TABLE training_metrics")
                conn.execute("ALTER TABLE training_metrics_new RENAME TO training_metrics")
            conn.commit()
            self._migrated = True
        return conn

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % 5 != 0:
            return
        try:
            total_steps = int(trainer.estimated_stepping_batches)
        except (TypeError, AttributeError):
            total_steps = 0
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE training_runs
                   SET current_step = ?, total_steps = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                (step, total_steps, self.run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        train_loss = float(metrics.get("train_loss", 0))
        val_loss = float(metrics.get("val_loss", 0))
        val_accuracy = float(metrics.get(self.primary_metric_key, 0))
        val_f1 = float(metrics.get(self.secondary_metric_key, 0))
        val_auroc = float(metrics.get("val_auroc", 0))
        val_precision = float(metrics.get("val_precision", 0))
        val_recall = float(metrics.get("val_recall", 0))

        # Compute smooth progress for sub-epoch validation (val_check_interval < 1.0).
        # trainer.current_epoch stays the same within an epoch, so use global_step.
        current_step = trainer.global_step
        try:
            estimated_total = trainer.estimated_stepping_batches
            max_epochs = trainer.max_epochs
            if isinstance(estimated_total, (int, float)) and estimated_total > 0:
                total_steps = int(estimated_total)
                progress_epoch = int(current_step / total_steps * max_epochs) + 1
                progress_epoch = min(progress_epoch, max_epochs)
            else:
                total_steps = 0
                progress_epoch = epoch + 1
        except (TypeError, AttributeError):
            total_steps = 0
            progress_epoch = epoch + 1

        per_class = getattr(pl_module, "_per_class_metrics", None)

        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO training_metrics
                   (run_id, epoch, step, train_loss, val_loss, val_accuracy, val_f1, learning_rate,
                    val_auroc, val_precision, val_recall)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.run_id, epoch, current_step, train_loss, val_loss, val_accuracy, val_f1, lr,
                 val_auroc, val_precision, val_recall),
            )
            conn.execute(
                """UPDATE training_runs SET
                       current_epoch = ?,
                       current_step = ?, total_steps = ?,
                       train_loss = ?, val_loss = ?,
                       val_accuracy = ?, val_f1 = ?,
                       best_accuracy = MAX(COALESCE(best_accuracy, 0), ?),
                       best_f1 = MAX(COALESCE(best_f1, 0), ?),
                       val_auroc = ?, val_precision = ?, val_recall = ?,
                       best_auroc = MAX(COALESCE(best_auroc, 0), ?),
                       status = 'running',
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (progress_epoch, current_step, total_steps,
                 train_loss, val_loss, val_accuracy, val_f1,
                 val_accuracy, val_f1,
                 val_auroc, val_precision, val_recall, val_auroc,
                 self.run_id),
            )
            if per_class:
                # Latest epoch wins — no history kept, matches other metric columns.
                conn.execute(
                    "UPDATE training_runs SET val_per_class = ? WHERE id = ?",
                    (json.dumps(per_class), self.run_id),
                )
            conn.commit()
        finally:
            conn.close()

    def on_fit_end(self, trainer, pl_module):
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE training_runs
                   SET status = 'predicting',
                       current_step = 0, total_steps = 0,
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (self.run_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def on_exception(self, trainer, pl_module, exception):
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE training_runs
                   SET status = 'failed', error = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                ("".join(traceback.format_exception(exception)), self.run_id),
            )
            conn.commit()
        finally:
            conn.close()


