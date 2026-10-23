"""Unit tests for Multi-Modal Visual Search and Color-Aware Re-Ranking.

Tests coverage:
- ColorHistogramExtractor (3D HSV histogram, intersection, Bhattacharyya distance)
- MultiModalQueryComposer (text query parsing, attribute scoring)
- MultiModalReRanker (multi-factor weighting, facet filtering, candidate reordering)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from multimodal_search import (
    ColorHistogramExtractor,
    MultiModalQueryComposer,
    MultiModalReRanker,
)


@pytest.fixture
def color_images():
    """Create distinct uniform color test images."""
    red_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    blue_img = Image.new("RGB", (64, 64), color=(0, 0, 255))
    green_img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    return {"red": red_img, "blue": blue_img, "green": green_img}


@pytest.fixture
def sample_metadata():
    """Create sample metadata DataFrame."""
    return pd.DataFrame([
        {
            "image_id": 101,
            "category": "Shoes",
            "article_type": "Casual Shoes",
            "color": "Red",
            "gender": "Men",
            "product_name": "Nike Air Red Running Shoes",
        },
        {
            "image_id": 102,
            "category": "Shoes",
            "article_type": "Sneakers",
            "color": "Blue",
            "gender": "Women",
            "product_name": "Adidas Blue Ultraboost Sneakers",
        },
        {
            "image_id": 103,
            "category": "Topwear",
            "article_type": "Tshirts",
            "color": "Black",
            "gender": "Men",
            "product_name": "Puma Black Sports T-shirt",
        },
    ])


def test_color_histogram_extraction(color_images):
    extractor = ColorHistogramExtractor(h_bins=8, s_bins=4, v_bins=4)
    hist_red = extractor.extract_histogram(color_images["red"])

    assert hist_red.shape == (128,)
    assert pytest.approx(np.sum(hist_red), rel=1e-5) == 1.0

    # Self-comparison must be 1.0
    self_sim = extractor.compute_histogram_intersection(hist_red, hist_red)
    assert pytest.approx(self_sim, rel=1e-5) == 1.0

    # Red vs Blue should have low intersection
    hist_blue = extractor.extract_histogram(color_images["blue"])
    cross_sim = extractor.compute_histogram_intersection(hist_red, hist_blue)
    assert cross_sim < 0.1

    # Bhattacharyya similarity
    bc_self = extractor.compute_bhattacharyya_similarity(hist_red, hist_red)
    assert pytest.approx(bc_self, rel=1e-5) == 1.0

    bc_cross = extractor.compute_bhattacharyya_similarity(hist_red, hist_blue)
    assert bc_cross < 0.2


def test_color_histogram_invalid_input():
    extractor = ColorHistogramExtractor()
    with pytest.raises(TypeError, match="image must be a PIL Image"):
        extractor.extract_histogram("not_an_image")

    with pytest.raises(ValueError, match="Histogram shapes do not match"):
        extractor.compute_histogram_intersection(np.zeros(10), np.zeros(12))


def test_query_composer_parsing():
    parsed = MultiModalQueryComposer.parse_text_query("Looking for red running shoes for sports")
    assert "red" in parsed["target_colors"]
    assert "shoes" in parsed["target_categories"]
    assert len(parsed["tokens"]) > 0


def test_query_composer_attribute_scoring(sample_metadata):
    parsed = MultiModalQueryComposer.parse_text_query("red running shoes")

    # Item 0 is "Red", "Shoes" -> high match
    score_0 = MultiModalQueryComposer.compute_attribute_score(parsed, sample_metadata.iloc[0].to_dict())
    # Item 1 is "Blue", "Shoes" -> partial match (category matches, color does not)
    score_1 = MultiModalQueryComposer.compute_attribute_score(parsed, sample_metadata.iloc[1].to_dict())
    # Item 2 is "Black", "Topwear" -> low match
    score_2 = MultiModalQueryComposer.compute_attribute_score(parsed, sample_metadata.iloc[2].to_dict())

    assert score_0 > score_1
    assert score_1 > score_2


def test_reranker_fusion_and_ordering(sample_metadata, color_images):
    reranker = MultiModalReRanker(weight_visual=0.5, weight_color=0.3, weight_attr=0.2)

    candidate_indices = [0, 1, 2]
    # Suppose initial visual similarity favored item 1 slightly
    initial_scores = [0.85, 0.90, 0.70]

    candidate_images = {
        0: color_images["red"],
        1: color_images["blue"],
        2: color_images["green"],
    }

    # Query with a red image and query text "red shoes"
    top_indices, final_scores, breakdowns = reranker.rerank(
        candidate_indices=candidate_indices,
        initial_scores=initial_scores,
        metadata_df=sample_metadata,
        query_image=color_images["red"],
        query_text="red shoes",
        candidate_images=candidate_images,
        top_k=2,
    )

    assert len(top_indices) == 2
    # Item 0 (Red Shoes) should now be boosted to rank 1 over Item 1 (Blue Shoes)
    assert top_indices[0] == 0
    assert final_scores[0] > final_scores[1]
    assert len(breakdowns) == 2
    assert "visual_score" in breakdowns[0]
    assert "color_score" in breakdowns[0]
    assert "attr_score" in breakdowns[0]


def test_reranker_facet_filtering(sample_metadata):
    reranker = MultiModalReRanker()

    candidate_indices = [0, 1, 2]
    initial_scores = [0.90, 0.85, 0.80]

    # Filter by gender = "Women"
    top_indices, final_scores, _ = reranker.rerank(
        candidate_indices=candidate_indices,
        initial_scores=initial_scores,
        metadata_df=sample_metadata,
        filters={"gender": "Women"},
        top_k=5,
    )

    assert len(top_indices) == 1
    assert top_indices[0] == 1  # Item 1 is the only Women product


def test_reranker_empty_candidates(sample_metadata):
    reranker = MultiModalReRanker()
    top_indices, final_scores, breakdowns = reranker.rerank(
        candidate_indices=[],
        initial_scores=[],
        metadata_df=sample_metadata,
    )
    assert len(top_indices) == 0
    assert len(final_scores) == 0
    assert len(breakdowns) == 0
