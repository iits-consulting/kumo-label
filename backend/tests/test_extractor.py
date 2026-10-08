import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from kumo_label.embeddings.extractor import extract_embeddings


def make_dataset(tmp_path: Path) -> str:
    """Minimal dataset: 2 tiny images indexed in kumo.db."""
    db_path = tmp_path / "kumo.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE images (id INTEGER PRIMARY KEY, filename TEXT, path TEXT, class TEXT, split TEXT)"
    )
    for i in range(2):
        img_file = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (4, 4), color=(i * 100, 0, 0)).save(str(img_file))
        conn.execute(
            "INSERT INTO images VALUES (?,?,?,?,?)",
            (i + 1, img_file.name, str(img_file), "a", "train"),
        )
    conn.commit()
    conn.close()
    return str(db_path)


FAKE_EMBEDDINGS = np.zeros((2, 384), dtype=np.float32)


def test_extract_creates_npy_and_json(tmp_path):
    db_path = make_dataset(tmp_path)
    with patch("kumo_label.embeddings.extractor._load_model", return_value=(None, None)), \
         patch("kumo_label.embeddings.extractor._get_batch_embeddings", return_value=FAKE_EMBEDDINGS):
        extract_embeddings(db_path, "dinov2", batch_size=2, progress_cb=lambda *_: None)

    assert (tmp_path / "kumo_embeddings_dinov2.npy").exists()
    assert (tmp_path / "kumo_embeddings_dinov2.json").exists()


def test_extract_npy_shape(tmp_path):
    db_path = make_dataset(tmp_path)
    with patch("kumo_label.embeddings.extractor._load_model", return_value=(None, None)), \
         patch("kumo_label.embeddings.extractor._get_batch_embeddings", return_value=FAKE_EMBEDDINGS):
        extract_embeddings(db_path, "dinov2", batch_size=2, progress_cb=lambda *_: None)

    arr = np.load(str(tmp_path / "kumo_embeddings_dinov2.npy"))
    assert arr.shape == (2, 384)
    assert arr.dtype == np.float32


def test_extract_json_contains_image_ids(tmp_path):
    db_path = make_dataset(tmp_path)
    with patch("kumo_label.embeddings.extractor._load_model", return_value=(None, None)), \
         patch("kumo_label.embeddings.extractor._get_batch_embeddings", return_value=FAKE_EMBEDDINGS):
        extract_embeddings(db_path, "dinov2", batch_size=2, progress_cb=lambda *_: None)

    ids = json.loads((tmp_path / "kumo_embeddings_dinov2.json").read_text())
    assert ids == [1, 2]


def test_extract_progress_cb_called(tmp_path):
    db_path = make_dataset(tmp_path)
    calls = []
    with patch("kumo_label.embeddings.extractor._load_model", return_value=(None, None)), \
         patch("kumo_label.embeddings.extractor._get_batch_embeddings", return_value=FAKE_EMBEDDINGS):
        extract_embeddings(db_path, "dinov2", batch_size=2, progress_cb=lambda c, t, m: calls.append((c, t, m)))

    assert len(calls) >= 1
    # Last call should report all images processed
    assert calls[-1][0] == 2
