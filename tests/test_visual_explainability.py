"""Unit tests for Visual Search Explainability (Spatial Patch Heatmaps & Color Palettes).

Tests coverage:
- DominantColorExtractor (palette extraction, percentage calculation, hex formatting)
- SpatialPatchComparator (grid feature extraction, positional patch similarity, region rankings)
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visual_explainability import DominantColorExtractor, SpatialPatchComparator


@pytest.fixture
def test_images():
    """Create synthetic test images with known color regions."""
    solid_red = Image.new("RGB", (64, 64), color=(255, 0, 0))
    solid_blue = Image.new("RGB", (64, 64), color=(0, 0, 255))

    # Two-tone image: left half red, right half blue
    two_tone = Image.new("RGB", (64, 64))
    pixels = two_tone.load()
    for x in range(64):
        for y in range(64):
            pixels[x, y] = (255, 0, 0) if x < 32 else (0, 0, 255)

    return {
        "solid_red": solid_red,
        "solid_blue": solid_blue,
        "two_tone": two_tone,
    }


def test_dominant_color_extractor(test_images):
    # Solid red
    palette = DominantColorExtractor.extract_palette(test_images["solid_red"], num_colors=3)
    assert len(palette) > 0
    top = palette[0]
    assert top["hex"].lower() in ["#ff0000", "#fe0000", "#ff0101"]
    assert top["percentage"] > 90.0

    # Two-tone
    palette_tt = DominantColorExtractor.extract_palette(test_images["two_tone"], num_colors=2)
    assert len(palette_tt) == 2
    assert 40.0 <= palette_tt[0]["percentage"] <= 60.0
    assert 40.0 <= palette_tt[1]["percentage"] <= 60.0


def test_dominant_color_extractor_errors():
    with pytest.raises(TypeError, match="image must be a PIL Image"):
        DominantColorExtractor.extract_palette("not_an_image")


def test_spatial_patch_comparator_identical(test_images):
    comp = SpatialPatchComparator(grid_size=4)
    res = comp.compute_patch_similarity_heatmap(
        test_images["two_tone"],
        test_images["two_tone"],
    )

    assert res["grid_size"] == 4
    matrix = np.array(res["heatmap_matrix"])
    assert matrix.shape == (4, 4)
    # Self-comparison must be near 1.0 everywhere
    assert pytest.approx(res["mean_patch_similarity"], rel=0.05) == 1.0
    assert len(res["top_aligned_regions"]) == 3
    assert len(res["least_aligned_regions"]) == 2


def test_spatial_patch_comparator_contrast(test_images):
    comp = SpatialPatchComparator(grid_size=4)
    res = comp.compute_patch_similarity_heatmap(
        test_images["solid_red"],
        test_images["solid_blue"],
    )

    # Red vs Blue should have low patch similarity
    assert res["mean_patch_similarity"] < 0.7


def test_spatial_patch_comparator_invalid_input(test_images):
    comp = SpatialPatchComparator(grid_size=4)
    with pytest.raises(TypeError, match="Both images must be PIL Image instances"):
        comp.compute_patch_similarity_heatmap(test_images["solid_red"], "invalid_image")
