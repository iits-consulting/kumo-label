# ---- Frontend build ----
FROM node:22-alpine AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Backend runtime ----
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

WORKDIR /app

# Resolve deps from pyproject instead of uv.lock: the lock pins the default
# PyPI torch build, which drags in ~6 GB of CUDA libraries. --torch-backend=cpu
# swaps torch/torchvision to the CPU wheels and keeps the image small.
COPY backend/pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --torch-backend=cpu -r pyproject.toml

COPY backend/src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps .

# Built SPA, served by FastAPI (see KUMO_STATIC_DIR handling in main.py)
COPY --from=frontend-build /build/dist /app/static

ENV KUMO_STATIC_DIR=/app/static \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Single worker only: job tracking (_active_processes, workers.py executor)
# is in-process state and breaks with multiple workers.
CMD ["uvicorn", "kumo_label.main:app", "--host", "0.0.0.0", "--port", "8000"]
