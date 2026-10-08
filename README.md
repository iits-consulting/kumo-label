# Kumo Label

A data labeling and active learning platform for image classification and object detection. Load a dataset, extract embeddings, visualize them as a 2D scatter plot, annotate images, and train models.

## Architecture

- **Frontend** ([frontend/](frontend/)) -- Svelte 5 + TypeScript SPA. Scatter plot visualization, annotation UI, class management, training controls. Runs on `:8080`, proxies `/api` to the backend.
- **Backend** ([backend/](backend/)) -- FastAPI + PyTorch Lightning. Dataset scanning, embedding extraction, dimensionality reduction, model training. Runs on `:8000`.
- **Storage** -- Each dataset gets a `kumo.db` SQLite file at its root containing images, labels, jobs, projections, and training history.

## Quick Start

### Docker

```bash
mkdir -p data           # put your datasets in here
docker compose up --build
```

Open `http://localhost:8000`. The container serves both the UI and the API; datasets are read from the `data/` directory, which appears as `/data` inside the container (use `/data/<name>` as the dataset path in the app). All state — `kumo.db`, embeddings, training runs — is written next to the dataset images, so it persists on the host. Downloaded HuggingFace models are cached in a named volume.

The image runs PyTorch on CPU to stay small. For GPU training, run the backend natively (see below) with `uv sync --group gpu`.

### Local development

```bash
# Terminal 1: Backend
cd backend
uv run fastapi dev src/kumo_label/main.py

# Terminal 2: Frontend
cd frontend
npm install # First run only
npm run dev
```

Open `http://localhost:8080`. Point the app at a dataset directory structured as:

```
dataset/
  train/
    class_a/
      img1.jpg
    class_b/
      img2.jpg
  test/
    ...
```

Or simply `class_a/img1.jpg` (defaults to train split).

## Workflow

1. **Project Setup** -- Select dataset path and task type
2. **Compute Embeddings** -- Choose a model (DINOv2, CLIP) and extract features
3. **Explore** -- View embeddings as a scatter plot (UMAP/t-SNE/PCA), filter by class or label status
4. **Annotate** -- Select points and assign labels via the annotation overlay
5. **Train** -- Start a classification model (ViT, ResNet50, EfficientNet) and watch metrics live
