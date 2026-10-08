import numpy as np
import pytest
from kumo_label.embeddings.reducer import reduce_embeddings

# Use 20 samples so UMAP/t-SNE have valid n_neighbors/perplexity values
RNG = np.random.default_rng(42)
EMBEDDINGS = RNG.random((20, 50)).astype(np.float32)


def test_pca_output_shape():
    result = reduce_embeddings(EMBEDDINGS, "pca", {})
    assert result.shape == (20, 2)
    assert result.dtype == np.float32


def test_tsne_output_shape():
    result = reduce_embeddings(EMBEDDINGS, "tsne", {"perplexity": 5})
    assert result.shape == (20, 2)
    assert result.dtype == np.float32


def test_umap_output_shape():
    result = reduce_embeddings(EMBEDDINGS, "umap", {"n_neighbors": 5})
    assert result.shape == (20, 2)
    assert result.dtype == np.float32


def test_tsne_clamps_perplexity_to_valid_range():
    # perplexity must be < n_samples; passing 100 with 20 samples should not raise
    result = reduce_embeddings(EMBEDDINGS, "tsne", {"perplexity": 100})
    assert result.shape == (20, 2)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown reduction method"):
        reduce_embeddings(EMBEDDINGS, "mds", {})


def test_cuml_path_when_available():
    from kumo_label.embeddings import reducer

    if not reducer._HAS_CUML:
        pytest.skip("cuML not installed")
    for method, params in [("umap", {"n_neighbors": 5}), ("tsne", {"perplexity": 5}), ("pca", {})]:
        result = reducer.reduce_embeddings(EMBEDDINGS, method, params)
        assert result.shape == (20, 2)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
