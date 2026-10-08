import json
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Model registry: short name → HuggingFace model ID and class
_MODELS = {
    "dinov2": "facebook/dinov2-small",
    "clip": "openai/clip-vit-base-patch32",
}

# Lazy cache: model_name -> (processor, model, device)
_model_cache: dict = {}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_name: str):
    """Load and cache processor + model. Downloads from HuggingFace on first call."""
    if model_name not in _model_cache:
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel

        model_id = _MODELS[model_name]
        processor = AutoImageProcessor.from_pretrained(model_id)
        if model_name == "clip":
            model = CLIPVisionModel.from_pretrained(model_id)
        else:
            model = AutoModel.from_pretrained(model_id)
        model.eval()
        model.to(_device)
        _model_cache[model_name] = (processor, model)
    return _model_cache[model_name]


class _ImageDataset(Dataset):
    """Simple dataset that loads images as PIL RGB."""

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return Image.open(self.paths[idx]).convert("RGB")


def _collate_pil(batch: list):
    """Pass-through collate that keeps PIL images as a list."""
    return batch


def _get_batch_embeddings(processor, model, images: list, model_name: str) -> np.ndarray:
    """Run model on a batch of PIL images. Returns float32 (batch, D).

    DINOv2 (AutoModel): pooler_output is None; use last_hidden_state[:, 0] (CLS token).
    CLIP (CLIPVisionModel): pooler_output is the projected CLS embedding — use directly.
    """
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(_device, non_blocking=True) for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast(
        device_type=_device.type, enabled=_device.type == "cuda"
    ):
        outputs = model(**inputs)
    if model_name == "clip":
        embedding = outputs.pooler_output
    else:
        embedding = outputs.last_hidden_state[:, 0]
    return embedding.cpu().numpy().astype(np.float32)


def extract_embeddings(
    db_path: str,
    model_name: str,
    batch_size: int,
    progress_cb: Callable[[int, int, str], None],
) -> None:
    """
    Extract embeddings for every image in the dataset and save atomically.

    Writes:
      {dataset_dir}/kumo_embeddings_{model_name}.npy  — float32 (N, D)
      {dataset_dir}/kumo_embeddings_{model_name}.json — JSON list of image IDs

    If a cache for this model already exists, only computes embeddings for
    images not present in the cached JSON ID list, then merges with the
    existing embeddings (final order is image-id ascending).

    progress_cb(completed, total, message) is called after each batch.
    """
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {list(_MODELS)}")

    db_file = Path(db_path)
    dataset_dir = db_file.parent

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, path FROM images ORDER BY id").fetchall()
    finally:
        conn.close()

    all_ids = [r[0] for r in rows]
    id_to_path = {r[0]: r[1] for r in rows}

    npy_path = dataset_dir / f"kumo_embeddings_{model_name}.npy"
    json_path = dataset_dir / f"kumo_embeddings_{model_name}.json"

    existing_ids: list[int] = []
    existing_embeddings: np.ndarray | None = None
    if npy_path.exists() and json_path.exists():
        try:
            existing_ids = json.loads(json_path.read_text())
            existing_embeddings = np.load(str(npy_path))
            if len(existing_ids) != existing_embeddings.shape[0]:
                # Cache mismatch — fall back to full recompute
                existing_ids = []
                existing_embeddings = None
        except Exception:
            existing_ids = []
            existing_embeddings = None

    existing_id_set = set(existing_ids)
    new_ids = [i for i in all_ids if i not in existing_id_set]
    new_paths = [id_to_path[i] for i in new_ids]
    total_new = len(new_ids)

    if total_new == 0 and existing_embeddings is not None:
        # Nothing new to embed and cache is valid — nothing to do.
        progress_cb(0, 0, "Up to date")
        return

    progress_cb(0, total_new, "Downloading model...")
    processor, model = _load_model(model_name)

    new_embeddings_list: list[np.ndarray] = []
    progress_cb(0, total_new, "Extracting features...")

    use_cuda = _device.type == "cuda"
    loader = DataLoader(
        _ImageDataset(new_paths),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=use_cuda,
        collate_fn=_collate_pil,
    )

    completed = 0
    for images in loader:
        batch_embeddings = _get_batch_embeddings(processor, model, images, model_name)
        new_embeddings_list.append(batch_embeddings)
        completed += len(images)
        progress_cb(completed, total_new, "Extracting features...")

    new_embeddings = (
        np.concatenate(new_embeddings_list, axis=0).astype(np.float32)
        if new_embeddings_list
        else np.zeros((0, existing_embeddings.shape[1] if existing_embeddings is not None else 0), dtype=np.float32)
    )

    # Merge old + new, then reorder by image_id ascending to match DB order.
    if existing_embeddings is not None and existing_embeddings.shape[0] > 0:
        merged_ids = existing_ids + new_ids
        merged = np.concatenate([existing_embeddings, new_embeddings], axis=0)
    else:
        merged_ids = new_ids
        merged = new_embeddings

    order = sorted(range(len(merged_ids)), key=lambda i: merged_ids[i])
    image_ids = [merged_ids[i] for i in order]
    embeddings_array = merged[order].astype(np.float32)

    # Atomic write: write to .tmp then rename
    tmp_npy = dataset_dir / f"kumo_embeddings_{model_name}.tmp.npy"
    tmp_json = Path(str(json_path) + ".tmp")

    try:
        np.save(str(tmp_npy), embeddings_array)
        tmp_json.write_text(json.dumps(image_ids))
        tmp_npy.rename(npy_path)
        tmp_json.rename(json_path)
    except Exception:
        tmp_npy.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
        raise
