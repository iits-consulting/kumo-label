import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cuml  # noqa: F401

    _HAS_CUML = True
except ImportError:
    _HAS_CUML = False


def reduce_embeddings(
    embeddings: np.ndarray,
    method: Literal["umap", "tsne", "pca"],
    params: dict,
) -> np.ndarray:
    """
    Reduce (N, D) embeddings to (N, 2).
    Returns float32 array of shape (N, 2).
    Uses GPU via cuML when available, falls back to CPU otherwise.
    """
    if method == "umap":
        n_neighbors = int(params.get("n_neighbors", 15))
        if _HAS_CUML:
            try:
                from cuml.manifold import UMAP as CuUMAP

                reducer = CuUMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
                return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
            except Exception as e:
                logger.warning("cuML UMAP failed, falling back to CPU: %s", e)
        from umap import UMAP

        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        return reducer.fit_transform(embeddings).astype(np.float32)

    if method == "tsne":
        perplexity = float(params.get("perplexity", 30))
        # t-SNE requires perplexity < n_samples
        perplexity = min(perplexity, embeddings.shape[0] - 1)
        if _HAS_CUML:
            try:
                from cuml.manifold import TSNE as CuTSNE

                reducer = CuTSNE(n_components=2, perplexity=perplexity, random_state=42)
                return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
            except Exception as e:
                logger.warning("cuML TSNE failed, falling back to CPU: %s", e)
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        return reducer.fit_transform(embeddings).astype(np.float32)

    if method == "pca":
        if _HAS_CUML:
            try:
                from cuml.decomposition import PCA as CuPCA

                return np.asarray(CuPCA(n_components=2).fit_transform(embeddings), dtype=np.float32)
            except Exception as e:
                logger.warning("cuML PCA failed, falling back to CPU: %s", e)
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        return reducer.fit_transform(embeddings).astype(np.float32)

    raise ValueError(f"Unknown reduction method: {method}")
