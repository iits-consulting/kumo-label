from fastapi.testclient import TestClient
from kumo_label.main import app

client = TestClient(app)


# ---- POST /api/datasets/load ----

def test_load_returns_metadata(tmp_path):
    (tmp_path / "train" / "0").mkdir(parents=True)
    (tmp_path / "train" / "1").mkdir(parents=True)
    (tmp_path / "train" / "0" / "a.jpg").write_bytes(b"x")
    (tmp_path / "train" / "1" / "b.jpg").write_bytes(b"x")

    response = client.post("/api/datasets/load", json={"path": str(tmp_path)})

    assert response.status_code == 200
    data = response.json()
    assert set(data["classes"]) == {"0", "1"}
    assert data["splits"] == ["train"]
    assert data["counts"]["total"] == 2
    assert data["db_path"].endswith("kumo.db")


def test_load_path_not_exist():
    response = client.post("/api/datasets/load", json={"path": "/nonexistent/path"})
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_load_path_not_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello")
    response = client.post("/api/datasets/load", json={"path": str(f)})
    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"]


def test_load_no_images(tmp_path):
    (tmp_path / "empty").mkdir()
    response = client.post("/api/datasets/load", json={"path": str(tmp_path)})
    assert response.status_code == 400
    assert "No images found" in response.json()["detail"]


# ---- GET /api/datasets/images ----

def _make_db(tmp_path) -> str:
    """Helper: create a kumo.db with 3 images."""
    (tmp_path / "train" / "cats").mkdir(parents=True)
    (tmp_path / "train" / "dogs").mkdir(parents=True)
    (tmp_path / "train" / "cats" / "a.jpg").write_bytes(b"x")
    (tmp_path / "train" / "cats" / "b.jpg").write_bytes(b"x")
    (tmp_path / "train" / "dogs" / "c.jpg").write_bytes(b"x")
    from kumo_label.scanner import scan_dataset
    scan_dataset(str(tmp_path))
    return str(tmp_path / "kumo.db")


def test_list_images_returns_paginated(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.get(f"/api/datasets/images?db_path={db_path}&limit=2&page=1")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["images"]) == 2
    assert data["page"] == 1
    assert data["limit"] == 2


def test_list_images_page_2(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.get(f"/api/datasets/images?db_path={db_path}&limit=2&page=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) == 1


def test_list_images_filter_by_class(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.get(f"/api/datasets/images?db_path={db_path}&class=cats")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert all(img["class"] == "cats" for img in data["images"])


def test_list_images_db_not_exist():
    response = client.get("/api/datasets/images?db_path=/nonexistent/kumo.db")
    assert response.status_code == 400


def test_list_images_db_wrong_name(tmp_path):
    f = tmp_path / "notkumo.db"
    f.touch()
    response = client.get(f"/api/datasets/images?db_path={f}")
    assert response.status_code == 400


def test_list_images_image_record_fields(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.get(f"/api/datasets/images?db_path={db_path}&limit=1")
    img = response.json()["images"][0]
    assert {"id", "filename", "path", "class", "split", "label", "annotation"}.issubset(img.keys())


# ---- GET /api/datasets/image ----

def test_serve_image(tmp_path):
    (tmp_path / "cats").mkdir()
    img_file = tmp_path / "cats" / "img.jpg"
    img_file.write_bytes(b"fakejpeg")
    from kumo_label.scanner import scan_dataset
    scan_dataset(str(tmp_path))

    response = client.get(f"/api/datasets/image?file_path={img_file}")
    assert response.status_code == 200
    assert response.content == b"fakejpeg"


def test_serve_image_not_found(tmp_path):
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "real.jpg").write_bytes(b"x")
    from kumo_label.scanner import scan_dataset
    scan_dataset(str(tmp_path))

    response = client.get(f"/api/datasets/image?file_path={tmp_path / 'cats' / 'missing.jpg'}")
    assert response.status_code == 404


def test_serve_image_path_traversal(tmp_path):
    # Request a file outside any dataset (no kumo.db in ancestry)
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive")
    response = client.get(f"/api/datasets/image?file_path={secret}")
    assert response.status_code == 403


def test_serve_image_real_traversal(tmp_path):
    # Set up a valid dataset
    (tmp_path / "cats").mkdir()
    img_file = tmp_path / "cats" / "img.jpg"
    img_file.write_bytes(b"data")
    from kumo_label.scanner import scan_dataset
    scan_dataset(str(tmp_path))

    # Create a file OUTSIDE the dataset directory
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("sensitive")

    # Attempt traversal: cats/../../../outside_secret.txt — resolves outside the dataset root
    traversal_path = str(tmp_path / "cats" / ".." / ".." / outside.name)
    response = client.get(f"/api/datasets/image?file_path={traversal_path}")
    assert response.status_code == 403

    outside.unlink()


# ---- PATCH /api/datasets/images/labels ----

def test_patch_labels_returns_updated_count(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    ids = [img["id"] for img in images[:2]]

    response = client.patch("/api/datasets/images/labels", json={
        "db_path": db_path, "ids": ids, "label": "cats"
    })

    assert response.status_code == 200
    assert response.json()["updated"] == 2


def test_patch_labels_round_trip(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target_id = images[0]["id"]

    client.patch("/api/datasets/images/labels", json={
        "db_path": db_path, "ids": [target_id], "label": "my-label"
    })

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target = next(img for img in refreshed if img["id"] == target_id)
    assert target["annotation"] == "my-label"
    assert target["label"] is None  # label column is unchanged


def test_patch_labels_empty_ids_returns_422(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.patch("/api/datasets/images/labels", json={
        "db_path": db_path, "ids": [], "label": "cats"
    })
    assert response.status_code == 422


def test_patch_labels_invalid_db_path_returns_400():
    response = client.patch("/api/datasets/images/labels", json={
        "db_path": "/nonexistent/kumo.db", "ids": [1], "label": "cats"
    })
    assert response.status_code == 400


def test_patch_labels_relative_db_path_returns_400(tmp_path):
    _make_db(tmp_path)
    response = client.patch("/api/datasets/images/labels", json={
        "db_path": "relative/kumo.db", "ids": [1], "label": "cats"
    })
    assert response.status_code == 400


def test_patch_labels_wrong_db_filename_returns_400(tmp_path):
    wrong = tmp_path / "notakumo.db"
    wrong.touch()
    response = client.patch("/api/datasets/images/labels", json={
        "db_path": str(wrong), "ids": [1], "label": "cats"
    })
    assert response.status_code == 400


def test_patch_labels_nonexistent_ids_returns_404(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.patch("/api/datasets/images/labels", json={
        "db_path": db_path, "ids": [99999, 99998], "label": "cats"
    })
    assert response.status_code == 404


def test_patch_labels_partial_match_returns_200(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    real_id = images[0]["id"]

    response = client.patch("/api/datasets/images/labels", json={
        "db_path": db_path, "ids": [real_id, 99999], "label": "cats"
    })

    assert response.status_code == 200
    assert response.json()["updated"] == 1


def test_patch_labels_empty_label_returns_422(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    response = client.patch("/api/datasets/images/labels", json={
        "db_path": db_path, "ids": [images[0]["id"]], "label": ""
    })
    assert response.status_code == 422


# ---- PATCH /api/datasets/images/split ----

def test_patch_split_round_trip(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target_id = images[0]["id"]

    response = client.patch("/api/datasets/images/split", json={
        "db_path": db_path, "ids": [target_id], "split": "valid"
    })
    assert response.status_code == 200
    assert response.json()["updated"] == 1

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target = next(img for img in refreshed if img["id"] == target_id)
    assert target["split_override"] == "valid"


def test_patch_split_clear_override(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target_id = images[0]["id"]

    client.patch("/api/datasets/images/split", json={
        "db_path": db_path, "ids": [target_id], "split": "test"
    })
    # Now clear it with null
    response = client.patch("/api/datasets/images/split", json={
        "db_path": db_path, "ids": [target_id], "split": None
    })
    assert response.status_code == 200

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target = next(img for img in refreshed if img["id"] == target_id)
    assert target["split_override"] is None


def test_patch_split_invalid_value_returns_400(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    response = client.patch("/api/datasets/images/split", json={
        "db_path": db_path, "ids": [images[0]["id"]], "split": "bogus"
    })
    assert response.status_code == 400


# ---- PATCH /api/datasets/images/tags ----

def test_patch_tags_add_appears_in_list(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    ids = [img["id"] for img in images[:2]]

    response = client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": ids, "class_name": "spot"
    })
    assert response.status_code == 200
    assert response.json()["updated"] == 2

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    for img_id in ids:
        target = next(img for img in refreshed if img["id"] == img_id)
        assert "spot" in target["label_classes"].split(",")


def test_patch_tags_add_is_additive_multilabel(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target_id = images[0]["id"]

    client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [target_id], "class_name": "spot"
    })
    client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [target_id], "class_name": "striped"
    })

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target = next(img for img in refreshed if img["id"] == target_id)
    assert set(target["label_classes"].split(",")) == {"spot", "striped"}


def test_patch_tags_remove_disappears(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target_id = images[0]["id"]

    client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [target_id], "class_name": "spot"
    })
    response = client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [target_id], "class_name": "spot", "present": False
    })
    assert response.status_code == 200
    assert response.json()["updated"] == 1

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target = next(img for img in refreshed if img["id"] == target_id)
    assert target["label_classes"] is None


def test_patch_tags_empty_ids_returns_422(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [], "class_name": "spot"
    })
    assert response.status_code == 422


def test_patch_tags_missing_class_name_returns_422(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    response = client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [images[0]["id"]], "class_name": ""
    })
    assert response.status_code == 422


def test_patch_tags_invalid_db_path_returns_400():
    response = client.patch("/api/datasets/images/tags", json={
        "db_path": "/nonexistent/kumo.db", "ids": [1], "class_name": "spot"
    })
    assert response.status_code == 400


def test_patch_tags_nonexistent_ids_returns_404(tmp_path):
    db_path = _make_db(tmp_path)
    response = client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [99999, 99998], "class_name": "spot"
    })
    assert response.status_code == 404


def test_patch_tags_mixed_valid_and_invalid_ids_writes_no_orphans(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    valid_id = images[0]["id"]
    invalid_id = 99999

    response = client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [valid_id, invalid_id], "class_name": "spot"
    })
    assert response.status_code == 200
    # Only the valid id was actually touched, even though 2 ids were sent.
    assert response.json()["updated"] == 1

    import sqlite3
    conn = sqlite3.connect(db_path)
    label_image_ids = {
        row[0] for row in conn.execute("SELECT image_id FROM image_labels WHERE class_name = 'spot'")
    }
    seed_image_ids = {row[0] for row in conn.execute("SELECT image_id FROM image_label_seeds")}
    conn.close()

    assert label_image_ids == {valid_id}
    assert invalid_id not in label_image_ids
    assert invalid_id not in seed_image_ids
    assert valid_id in seed_image_ids


# ---- POST /api/datasets/multilabel/seed ----

def test_seed_class_folder_dataset_creates_tags(tmp_path):
    db_path = _make_db(tmp_path)

    response = client.post("/api/datasets/multilabel/seed", json={"db_path": db_path})
    assert response.status_code == 200
    assert response.json()["seeded"] == 3  # a.jpg, b.jpg (cats) + c.jpg (dogs)

    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    for img in images:
        assert img["label_classes"] == img["class"]


def test_seed_is_idempotent(tmp_path):
    db_path = _make_db(tmp_path)
    client.post("/api/datasets/multilabel/seed", json={"db_path": db_path})

    response = client.post("/api/datasets/multilabel/seed", json={"db_path": db_path})
    assert response.status_code == 200
    assert response.json()["seeded"] == 0


def test_seed_does_not_reintroduce_untagged_seed(tmp_path):
    db_path = _make_db(tmp_path)
    client.post("/api/datasets/multilabel/seed", json={"db_path": db_path})

    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target_id = images[0]["id"]
    target_class = images[0]["class"]

    # User removes the seeded tag.
    client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": [target_id], "class_name": target_class, "present": False
    })

    # Re-seeding must not bring the tag back.
    response = client.post("/api/datasets/multilabel/seed", json={"db_path": db_path})
    assert response.status_code == 200
    assert response.json()["seeded"] == 0

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    target = next(img for img in refreshed if img["id"] == target_id)
    assert target["label_classes"] is None


def test_seed_after_rescan_only_seeds_new_images(tmp_path):
    db_path = _make_db(tmp_path)
    client.post("/api/datasets/multilabel/seed", json={"db_path": db_path})

    # Add a new image to the dataset and rescan.
    import shutil
    src = tmp_path / "train" / "cats" / "a.jpg"
    dst = tmp_path / "train" / "cats" / "new.jpg"
    shutil.copy(src, dst)
    rescan_resp = client.post("/api/datasets/rescan", json={"path": str(tmp_path)})
    assert rescan_resp.status_code == 200
    assert rescan_resp.json()["new_images"] == 1

    response = client.post("/api/datasets/multilabel/seed", json={"db_path": db_path})
    assert response.status_code == 200
    assert response.json()["seeded"] == 1


# ---- POST /api/datasets/classes/remove (tags) ----

def test_remove_class_deletes_tags(tmp_path):
    db_path = _make_db(tmp_path)
    images = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    ids = [img["id"] for img in images[:2]]
    client.patch("/api/datasets/images/tags", json={
        "db_path": db_path, "ids": ids, "class_name": "spot"
    })

    response = client.post("/api/datasets/classes/remove", json={
        "db_path": db_path, "class_name": "spot"
    })
    assert response.status_code == 200
    assert response.json()["tags_deleted"] == 2

    refreshed = client.get(f"/api/datasets/images?db_path={db_path}").json()["images"]
    for img_id in ids:
        target = next(img for img in refreshed if img["id"] == img_id)
        assert target["label_classes"] is None
