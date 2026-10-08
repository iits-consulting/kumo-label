import json
import sqlite3

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from kumo_label.training.detection_module import KumoDetector
from kumo_label.training.predict import compute_al_score


class _PredictionDataset(Dataset):
    def __init__(self, image_ids: list[int], image_paths: list[str], processor):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = image.size
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        return {
            "image_id": self.image_ids[idx],
            "pixel_values": pixel_values,
            "orig_size": (h, w),
        }


def run_detection_predictions(
    db_path: str,
    run_id: str,
    checkpoint_path: str,
    class_names: list[str],
    model_id: str,
    confidence_threshold: float = 0.0,
    nms_iou_threshold: float = 0.5,
    image_ids_filter: list[int] | None = None,
):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    if image_ids_filter is not None:
        if not image_ids_filter:
            conn.close()
            return
        placeholders = ",".join("?" * len(image_ids_filter))
        rows = conn.execute(
            f"SELECT id, path FROM images WHERE id IN ({placeholders})", image_ids_filter
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, path FROM images").fetchall()
    image_ids = [r[0] for r in rows]
    image_paths = [r[1] for r in rows]

    processor = AutoImageProcessor.from_pretrained(model_id)
    dataset = _PredictionDataset(image_ids, image_paths, processor)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    model = KumoDetector.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    det_rows = []  # for detection_predictions table
    summary_rows = []  # for predictions table (image-level summary)

    total_images = len(image_ids)
    conn.execute(
        "UPDATE training_runs SET current_step = 0, total_steps = ?, updated_at = datetime('now') WHERE id = ?",
        (total_images, run_id),
    )
    conn.commit()
    processed = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            ids = batch["image_id"]
            orig_sizes = batch["orig_size"]  # (h, w) tuples as tensors

            outputs = model(pixel_values)

            # Compute per-query objectness from raw logits
            # (before post-processing discards the background class)
            logits = outputs.logits  # (batch, num_queries, num_classes + 1)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            bg_probs = probs[:, :, -1]  # (batch, num_queries)
            max_objectness = (1.0 - bg_probs).max(dim=-1).values  # (batch,)

            # Post-process: convert to pixel-space boxes
            target_sizes = torch.stack(
                [torch.tensor([orig_sizes[0][i], orig_sizes[1][i]]) for i in range(len(ids))]
            ).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )

            for i, result in enumerate(results):
                img_id = int(ids[i])
                orig_h = int(orig_sizes[0][i])
                orig_w = int(orig_sizes[1][i])

                boxes = result["boxes"]
                scores = result["scores"]
                labels = result["labels"]

                # Apply NMS
                if len(boxes) > 0:
                    keep = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                # Convert to normalized top-left (x, y, w, h) format
                max_conf = 0.0
                max_conf_class = class_names[0] if class_names else ""
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j].tolist()
                    conf = float(scores[j])
                    cls_idx = int(labels[j])
                    cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)

                    # Normalize to 0-1
                    nx = max(0.0, x1 / orig_w)
                    ny = max(0.0, y1 / orig_h)
                    nw = max(0.0, (x2 - x1) / orig_w)
                    nh = max(0.0, (y2 - y1) / orig_h)

                    det_rows.append((run_id, img_id, cls_name, conf, nx, ny, nw, nh))

                    if conf > max_conf:
                        max_conf = conf
                        max_conf_class = cls_name

                # Image-level summary for scatter plot coloring / AL scoring
                num_dets = len(boxes)
                if num_dets > 0:
                    uncertainty = 1.0 - max_conf
                    confidence_val = max_conf
                else:
                    # Use raw objectness from DETR query slots: if even the
                    # best query has low objectness, the model is confident
                    # nothing is here → low uncertainty.
                    img_max_objectness = float(max_objectness[i])
                    uncertainty = img_max_objectness
                    confidence_val = 1.0 - img_max_objectness

                is_unlabeled = conn.execute(
                    "SELECT COUNT(*) FROM annotations WHERE image_id = ?", (img_id,)
                ).fetchone()[0] == 0

                al_score = compute_al_score(
                    uncertainty=uncertainty,
                    is_unlabeled=is_unlabeled,
                    is_wrong=False,
                    confidence=confidence_val,
                )

                top_preds = []
                if num_dets > 0:
                    for j in range(min(3, len(boxes))):
                        cls_idx = int(labels[j])
                        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                        top_preds.append({"label": cls_name, "confidence": round(float(scores[j]), 4)})

                summary_rows.append((
                    run_id, img_id,
                    max_conf_class if num_dets > 0 else "",
                    confidence_val,
                    uncertainty,
                    al_score,
                    json.dumps(top_preds),
                    int(is_unlabeled),
                    0,  # is_wrong — not computed for detection
                    0.0,  # class_imbalance_score — simplified for detection
                ))

            processed = min(processed + len(ids), total_images)
            conn.execute(
                "UPDATE training_runs SET current_step = ?, updated_at = datetime('now') WHERE id = ?",
                (processed, run_id),
            )
            conn.commit()

    # Ensure tables exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL REFERENCES training_runs(id),
            image_id    INTEGER NOT NULL REFERENCES images(id),
            class_name  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            x           REAL NOT NULL,
            y           REAL NOT NULL,
            width       REAL NOT NULL,
            height      REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_det_preds_run_image ON detection_predictions(run_id, image_id)"
    )

    # Clear previous predictions for this run (scope to the images we just predicted)
    if image_ids_filter is not None:
        placeholders = ",".join("?" * len(image_ids))
        conn.execute(
            f"DELETE FROM detection_predictions WHERE run_id = ? AND image_id IN ({placeholders})",
            (run_id, *image_ids),
        )
    else:
        conn.execute("DELETE FROM detection_predictions WHERE run_id = ?", (run_id,))

    conn.executemany(
        "INSERT INTO detection_predictions (run_id, image_id, class_name, confidence, x, y, width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        det_rows,
    )

    # Also write image-level summaries to predictions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            run_id                TEXT NOT NULL REFERENCES training_runs(id),
            image_id              INTEGER NOT NULL REFERENCES images(id),
            predicted_class       TEXT NOT NULL,
            confidence            REAL NOT NULL,
            uncertainty           REAL NOT NULL DEFAULT 0,
            al_score              REAL NOT NULL DEFAULT 0,
            top_predictions       TEXT NOT NULL,
            is_unlabeled          INTEGER NOT NULL DEFAULT 0,
            is_wrong              INTEGER NOT NULL DEFAULT 0,
            class_imbalance_score REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (run_id, image_id)
        )
    """)
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (run_id, image_id, predicted_class, confidence, uncertainty, al_score, top_predictions, is_unlabeled, is_wrong, class_imbalance_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        summary_rows,
    )
    conn.commit()
    conn.close()
