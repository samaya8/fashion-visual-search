import numpy as np
import pytest

from search import find_similar_images


def test_find_similar_images_exact_match():
    # 3 vectors in 4 dimensions, normalized
    dataset = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    indices, scores = find_similar_images(query, dataset, top_k=2)

    assert len(indices) == 2
    assert indices[0] == 1  # exact match
    assert pytest.approx(scores[0], rel=1e-5) == 1.0
    assert pytest.approx(scores[1], rel=1e-5) == 0.0


def test_find_similar_images_with_candidates():
    dataset = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # Restrict to index 0 and 2
    indices, scores = find_similar_images(query, dataset, top_k=2, candidate_indices=[0, 2])

    assert len(indices) == 2
    assert set(indices).issubset({0, 2})


def test_find_similar_images_empty_candidate():
    dataset = np.eye(4, dtype=np.float32)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    indices, scores = find_similar_images(query, dataset, top_k=2, candidate_indices=[])

    assert len(indices) == 0
    assert len(scores) == 0


def test_find_similar_images_dimension_mismatch():
    dataset = np.eye(4, dtype=np.float32)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Query and dataset embedding sizes do not match"):
        find_similar_images(query, dataset)
