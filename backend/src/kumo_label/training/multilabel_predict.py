import json
import sqlite3

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from kumo_label.training.multilabel_module import KumoMultilabelClassifier
from kumo_label.training.predict import compute_al_score


# --- Pure helpers (unit-tested without a checkpoint/model) ---

def multilabel_uncertainty(probs: list[float]) -> float:
    if not probs:
        return 0.0
    return sum(1.0 - 2.0 * abs(p - 0.5) for p in probs) / len(probs)


def multilabel_confidence(probs: list[float], predicted_idx: list[int]) -> float:
    if predicted_idx:
        return sum(probs[i] for i in predicted_idx) / len(predicted_idx)
    if not probs:
        return 0.0
    return 1.0 - max(probs)


def multilabel_is_wrong(predicted: set[str], actual: set[str], is_labeled: bool) -> bool:
    if not is_labeled:
        return False
    return predicted != actual


def multilabel_imbalance(tags: set[str], counts: dict[str, int]) -> float:
    if not tags:
        return 0.0
    max_count = max(counts.values()) if counts else 0
    if max_count == 0:
        return 0.0
    rarest_count = min(counts.get(t, 0) for t in tags)
    return 1.0 - (rarest_count / max_count)


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


def run_multilabel_predictions(
    db_path: str,
    run_id: str,
    checkpoint_path: str,
    class_names: list[str],
    image_size: int,
    image_ids_filter: list[int] | None = None,
    threshold: float = 0.5,
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

    # Ground-truth tag sets for al_score computation, across the whole
    # dataset (not just the filtered subset) — mirrors run_predictions'
    # full-dataset label_map, needed for dataset-wide class imbalance counts.
    tag_rows = conn.execute(
        "SELECT image_id, GROUP_CONCAT(class_name) FROM image_labels GROUP BY image_id"
    ).fetchall()
    tag_map: dict[int, set[str]] = {
        r[0]: set(r[1].split(",")) for r in tag_rows if r[1]
    }

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = _PredictionDataset(image_ids, image_paths, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = KumoMultilabelClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
            probs = torch.sigmoid(outputs.logits)

            for i in range(len(ids)):
                img_id = int(ids[i])
                probs_i = [float(p) for p in probs[i]]

                predicted_idx = sorted(
                    (idx for idx, p in enumerate(probs_i) if p > threshold),
                    key=lambda idx: probs_i[idx],
                    reverse=True,
                )
                predicted_names = [class_names[idx] for idx in predicted_idx]
                predicted_set = set(predicted_names)
                predicted_class = ",".join(predicted_names)

                top_preds = sorted(
                    (
                        {"label": class_names[idx], "confidence": round(p, 4)}
                        for idx, p in enumerate(probs_i)
                    ),
                    key=lambda d: d["confidence"],
                    reverse=True,
                )

                uncertainty = multilabel_uncertainty(probs_i)
                confidence = multilabel_confidence(probs_i, predicted_idx)

                actual_tags = tag_map.get(img_id, set())
                is_unlabeled = not actual_tags
                is_wrong = multilabel_is_wrong(predicted_set, actual_tags, not is_unlabeled)

                predictions.append({
                    "run_id": run_id,
                    "image_id": img_id,
                    "predicted_class": predicted_class,
                    "predicted_set": predicted_set,
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

    # Build tag counts across all labeled images in the dataset, plus the
    # predicted tags of unlabeled images we just predicted — mirrors
    # predict.py's class_counts construction for the single-label case.
    class_counts: dict[str, int] = {}
    for tags in tag_map.values():
        for t in tags:
            class_counts[t] = class_counts.get(t, 0) + 1
    for p in predictions:
        if not tag_map.get(p["image_id"]):
            for t in p["predicted_set"]:
                class_counts[t] = class_counts.get(t, 0) + 1

    rows = []
    for p in predictions:
        actual_tags = tag_map.get(p["image_id"])
        effective_tags = actual_tags if actual_tags else p["predicted_set"]
        imbalance = multilabel_imbalance(effective_tags, class_counts)
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
