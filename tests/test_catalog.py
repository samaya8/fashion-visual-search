"""Unit tests for the CatalogManager and micro-catalog fallback engine.

Tests coverage:
- Real dataset loading when files exist
- Synthetic fallback generation when assets are absent
- Embedding normalization (unit L2 norm)
- Procedural image synthesis and PIL compatibility
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from catalog import CatalogManager


def test_catalog_manager_default():
    mgr = CatalogManager()
    assert mgr.catalog_size > 0
    assert len(mgr.metadata_df) == len(mgr.embeddings_matrix)
    assert mgr.embeddings_matrix.shape[1] == 576

    # Verify unit L2 norm
    norms = np.linalg.norm(mgr.embeddings_matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)


def test_catalog_manager_fallback(tmp_path: Path):
    non_existent_meta = tmp_path / "fake_meta.csv"
    non_existent_embed = tmp_path / "fake_embed.npy"

    mgr = CatalogManager(metadata_path=non_existent_meta, embeddings_path=non_existent_embed)
    assert mgr.is_synthetic is True
    assert mgr.catalog_size == len(CatalogManager.MICRO_CATALOG_DATA)
    assert mgr.embeddings_matrix.shape == (mgr.catalog_size, 576)

    # Check vector unit norm
    norms = np.linalg.norm(mgr.embeddings_matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)


def test_catalog_manager_get_image():
    mgr = CatalogManager()
    # Test with index 0
    img = mgr.get_image(0)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0

    # Test with row dictionary
    sample_row = {
        "category": "Shoes",
        "color": "Red",
        "product_name": "Test Synthetic Shoes",
    }
    synth_img = mgr.get_image(sample_row)
    assert isinstance(synth_img, Image.Image)
    assert synth_img.size == (224, 224)
