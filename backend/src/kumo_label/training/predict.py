import json
import sqlite3

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from kumo_label.training.module import KumoClassifier


def compute_al_score(
    uncertainty: float,
    is_unlabeled: bool,
    is_wrong: bool,
    confidence: float,
    class_imbalance: float = 0.0,
) -> float:
    """Compute composite active learning score in [0, 1]."""
    score = 0.0
    score += 0.35 * uncertainty
    score += 0.25 * float(is_unlabeled)
    score += 0.15 * float(is_wrong)
    score += 0.10 * (1.0 - confidence)
    score += 0.15 * class_imbalance
    return min(score, 1.0)


class _PredictionDataset(Dataset):
    def __init__(self, image_ids: list[int], image_paths: list[str], transform):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        return {"image_id": self.image_ids[idx], "pixel_values": pixel_values}


def run_predictions(
    db_path: str,
    run_id: str,
    checkpoint_path: str,
    class_names: list[str],
    image_size: int,
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

    # Fetch effective labels for al_score computation
    label_rows = conn.execute(
        "SELECT id, COALESCE(annotation, class) AS effective_label FROM images"
    ).fetchall()
    label_map = {r[0]: r[1] for r in label_rows}

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = _PredictionDataset(image_ids, image_paths, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = KumoClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    top_k = min(3, len(class_names))
    num_classes = len(class_names)
    predictions = []

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
            outputs = model(pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

            # Entropy-based uncertainty normalized to [0, 1]
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(float(num_classes)))
            normalized_uncertainty = (entropy / max_entropy).clamp(0, 1)

            for i in range(len(ids)):
                img_id = int(ids[i])
                pred_idx = int(top_indices[i][0])
                confidence = float(top_probs[i][0])
                uncertainty = float(normalized_uncertainty[i])
                top_preds = [
                    {
                        "label": class_names[int(top_indices[i][j])],
                        "confidence": round(float(top_probs[i][j]), 4),
                    }
                    for j in range(top_k)
                ]

                effective_label = label_map.get(img_id)
                is_unlabeled = not effective_label
                is_wrong = (not is_unlabeled) and (class_names[pred_idx] != effective_label)

                predictions.append({
                    "run_id": run_id,
                    "image_id": img_id,
                    "predicted_class": class_names[pred_idx],
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "top_predictions": json.dumps(top_preds),
                    "is_unlabeled": is_unlabeled,
                    "is_wrong": is_wrong,
                })

            processed = min(processed + len(ids), total_images)
            conn.execute(
                "UPDATE training_runs SET current_step = ?, updated_at = datetime('now') WHERE id = ?",
                (processed, run_id),
            )
            conn.commit()

    # Compute class imbalance scores from combined distribution.
    # Count labels across the whole dataset, plus predicted classes for the
    # images we just predicted that are unlabeled. Equivalent to the previous
    # behavior for full-dataset runs (where every image is in `predictions`).
    class_counts: dict[str, int] = {}
    for lbl in label_map.values():
        if lbl:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
    for p in predictions:
        if not label_map.get(p["image_id"]):
            class_counts[p["predicted_class"]] = class_counts.get(p["predicted_class"], 0) + 1
    max_count = max(class_counts.values()) if class_counts else 1

    rows = []
    for p in predictions:
        effective_label = label_map.get(p["image_id"])
        cls = effective_label if effective_label else p["predicted_class"]
        imbalance = 1.0 - (class_counts.get(cls, 0) / max_count)
        al_score = compute_al_score(
            p["uncertainty"], p["is_unlabeled"], p["is_wrong"], p["confidence"], imbalance
        )
        rows.append((
            p["run_id"], p["image_id"], p["predicted_class"], p["confidence"],
            p["uncertainty"], al_score, p["top_predictions"],
            int(p["is_unlabeled"]), int(p["is_wrong"]), imbalance,
        ))

    # Ensure table exists (safety net for older DBs)
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
    # Add columns to older DBs that don't have them
    for col, col_type, default in [
        ("uncertainty", "REAL", "0"),
        ("al_score", "REAL", "0"),
        ("is_unlabeled", "INTEGER", "0"),
        ("is_wrong", "INTEGER", "0"),
        ("class_imbalance_score", "REAL", "0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type} NOT NULL DEFAULT {default}")
        except sqlite3.OperationalError:
            pass
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (run_id, image_id, predicted_class, confidence, uncertainty, al_score, top_predictions, is_unlabeled, is_wrong, class_imbalance_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
