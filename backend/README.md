# Backend

FastAPI server handling dataset management, embedding extraction, dimensionality reduction, and model training.

## Running

```bash
uv sync                                      # CPU-only install
uv sync --group gpu                          # Add GPU-accelerated UMAP/t-SNE/PCA via RAPIDS cuML

uv run fastapi dev src/kumo_label/main.py
uv run pytest                                # Run tests
uv run pytest tests/test_scanner.py          # Single test file
```

## GPU Acceleration

The `gpu` dependency group installs [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) (`cuml-cu13`),
which provides GPU implementations of UMAP, t-SNE, and PCA. When cuML is importable,
[`embeddings/reducer.py`](src/kumo_label/embeddings/reducer.py) dispatches reductions to the GPU
(typically 30–60× faster on large embedding sets). When cuML is missing or fails at runtime, the reducer
silently falls back to scikit-learn / umap-learn on CPU, so the codebase still works on CPU-only machines.

Requirements: NVIDIA GPU with compute capability 7.0+, Linux or WSL2, CUDA 12.x or 13.x driver.

## Entry Point

[src/kumo_label/main.py](src/kumo_label/main.py) -- Mounts all routers under `/api` and sets up CORS for `:8080`.

## API Routers (`routers/`)

| Router | Prefix | Purpose |
|---|---|---|
| [`datasets.py`](src/kumo_label/routers/datasets.py) | `/api/datasets` | Scan directories, create `kumo.db`, list/label images |
| [`embeddings.py`](src/kumo_label/routers/embeddings.py) | `/api/embeddings` | Compute image embeddings (DINOv2, CLIP) via background jobs |
| [`projections.py`](src/kumo_label/routers/projections.py) | `/api/projections` | Run UMAP/t-SNE/PCA on embeddings, cache results |
| [`jobs.py`](src/kumo_label/routers/jobs.py) | `/api/jobs` | Poll status of embedding/projection background jobs |
| [`training.py`](src/kumo_label/routers/training.py) | `/api/training` | Start/stop/monitor classification training runs |

## Key Modules

- [**`scanner.py`**](src/kumo_label/scanner.py) -- Scans a dataset directory, detects splits and classes from folder structure, initializes SQLite tables.
- [**`workers.py`**](src/kumo_label/workers.py) -- Single-threaded `ThreadPoolExecutor` for embedding/projection jobs. Updates job status in SQLite.
- [**`db_utils.py`**](src/kumo_label/db_utils.py) -- Validates `db_path` query parameters.
- [**`embeddings/extractor.py`**](src/kumo_label/embeddings/extractor.py) -- Extracts features using HuggingFace `transformers` models.
- [**`embeddings/reducer.py`**](src/kumo_label/embeddings/reducer.py) -- Dimensionality reduction (UMAP, t-SNE, PCA). Uses RAPIDS cuML on GPU when available, falls back to scikit-learn / umap-learn on CPU.

## Training Pipeline (`training/`)

Training runs as a **subprocess** spawned by the training router, configured via Hydra ([`conf/`](src/kumo_label/conf/)).

- [**`train.py`**](src/kumo_label/training/train.py) -- CLI entry point (`python -m kumo_label.training.train`), receives Hydra overrides from the router.
- [**`module.py`**](src/kumo_label/training/module.py) -- PyTorch Lightning `LightningModule` for image classification.
- [**`datamodule.py`**](src/kumo_label/training/datamodule.py) -- Loads labeled images from SQLite, applies augmentations, handles train/val splits.
- [**`callbacks.py`**](src/kumo_label/training/callbacks.py) -- `DBMetricsCallback` writes metrics to SQLite for frontend polling and inline charts.

Available models: `vit_base`, `resnet50`, `efficientnet_b0`, or any HuggingFace model ID.
