"""Enterprise Multi-Modal Visual Search & Color-Aware Re-Ranking Engine.

Provides:
1. ColorHistogramExtractor: 3D HSV color histogram (128 bins) extraction and histogram intersection.
2. MultiModalQueryComposer: Combines visual query embeddings with textual attribute modifiers.
3. MultiModalReRanker: Two-stage re-ranking combining deep visual embeddings, color fidelity,
   and metadata attribute relevance with facet filtering.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


class ColorHistogramExtractor:
    """Extracts 3D HSV color distribution signatures for visual similarity matching."""

    def __init__(self, h_bins: int = 8, s_bins: int = 4, v_bins: int = 4):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.total_bins = h_bins * s_bins * v_bins

    def extract_histogram(self, image: Image.Image) -> np.ndarray:
        """Extract normalized 3D HSV histogram vector (shape: (total_bins,))."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        img_hsv = image.convert("HSV")
        # Resize for fast, robust histogram extraction
        img_small = img_hsv.resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.array(img_small, dtype=np.float32)

        # H: [0, 255] in PIL, S: [0, 255], V: [0, 255]
        h = arr[:, :, 0] / 256.0 * self.h_bins
        s = arr[:, :, 1] / 256.0 * self.s_bins
        v = arr[:, :, 2] / 256.0 * self.v_bins

        h_idx = np.clip(h.astype(int), 0, self.h_bins - 1)
        s_idx = np.clip(s.astype(int), 0, self.s_bins - 1)
        v_idx = np.clip(v.astype(int), 0, self.v_bins - 1)

        flat_indices = h_idx * (self.s_bins * self.v_bins) + s_idx * self.v_bins + v_idx

        hist, _ = np.histogram(flat_indices, bins=self.total_bins, range=(0, self.total_bins))
        hist = hist.astype(np.float32)

        # L1 normalize
        total = np.sum(hist)
        if total > 0:
            hist /= total

        return hist

    @staticmethod
    def compute_histogram_intersection(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """Compute histogram intersection similarity in range [0.0, 1.0]."""
        if hist_a.shape != hist_b.shape:
            raise ValueError(f"Histogram shapes do not match: {hist_a.shape} vs {hist_b.shape}")
        return float(np.sum(np.minimum(hist_a, hist_b)))

    @staticmethod
    def compute_bhattacharyya_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """Compute Bhattacharyya similarity coefficient in range [0.0, 1.0]."""
        if hist_a.shape != hist_b.shape:
            raise ValueError(f"Histogram shapes do not match: {hist_a.shape} vs {hist_b.shape}")
        bc = np.sum(np.sqrt(np.clip(hist_a * hist_b, 0.0, None)))
        return float(np.clip(bc, 0.0, 1.0))


class MultiModalQueryComposer:
    """Composes multimodal search intent combining visual signals and textual attributes."""

    COLOR_SYNONYMS: Dict[str, List[str]] = {
        "black": ["black", "dark", "charcoal", "ebony"],
        "white": ["white", "ivory", "cream", "snow"],
        "red": ["red", "crimson", "scarlet", "ruby", "maroon"],
        "blue": ["blue", "navy", "cyan", "azure", "teal", "indigo"],
        "green": ["green", "olive", "emerald", "lime", "mint"],
        "yellow": ["yellow", "gold", "amber", "mustard"],
        "brown": ["brown", "tan", "beige", "khaki", "camel"],
        "pink": ["pink", "rose", "magenta", "blush"],
        "grey": ["grey", "gray", "silver", "ash"],
    }

    CATEGORY_SYNONYMS: Dict[str, List[str]] = {
        "shoes": ["shoes", "sandals", "sneakers", "boots", "flats", "footwear", "loafers", "heels"],
        "topwear": ["tshirt", "t-shirt", "shirt", "top", "blouse", "sweater", "hoodie", "jacket"],
        "bottomwear": ["jeans", "pants", "trousers", "shorts", "skirt", "leggings"],
        "eyewear": ["sunglasses", "glasses", "shades", "eyewear"],
        "watches": ["watch", "timepiece", "chronograph"],
        "bags": ["bag", "backpack", "handbag", "purse", "wallet", "clutch", "tote"],
        "jewellery": ["earrings", "necklace", "ring", "bracelet", "jewellery", "jewelry"],
    }

    @classmethod
    def parse_text_query(cls, query_text: str) -> Dict[str, Any]:
        """Extract explicit and latent attributes from query text string."""
        tokens = re.findall(r"\b\w+\b", query_text.lower())
        detected_colors = []
        detected_categories = []

        for token in tokens:
            for color, synonyms in cls.COLOR_SYNONYMS.items():
                if token in synonyms and color not in detected_colors:
                    detected_colors.append(color)
            for category, synonyms in cls.CATEGORY_SYNONYMS.items():
                if token in synonyms and category not in detected_categories:
                    detected_categories.append(category)

        return {
            "raw_text": query_text,
            "tokens": tokens,
            "target_colors": detected_colors,
            "target_categories": detected_categories,
        }

    @classmethod
    def compute_attribute_score(
        cls,
        parsed_query: Dict[str, Any],
        item_metadata: Dict[str, Any],
    ) -> float:
        """Score metadata alignment against text query tokens and attributes."""
        tokens = set(parsed_query.get("tokens", []))
        if not tokens:
            return 1.0

        target_colors = set(parsed_query.get("target_colors", []))
        target_categories = set(parsed_query.get("target_categories", []))

        item_color = str(item_metadata.get("color", "")).lower()
        item_cat = str(item_metadata.get("category", "")).lower()
        item_name = str(item_metadata.get("product_name", "")).lower()
        item_article = str(item_metadata.get("article_type", "")).lower()

        score = 0.0
        weights_sum = 0.0

        # Color match
        if target_colors:
            weights_sum += 2.0
            if any(tc in item_color or item_color in cls.COLOR_SYNONYMS.get(tc, []) for tc in target_colors):
                score += 2.0

        # Category match
        if target_categories:
            weights_sum += 2.5
            if any(tc in item_cat or any(syn in item_name or syn in item_article for syn in cls.CATEGORY_SYNONYMS.get(tc, [])) for tc in target_categories):
                score += 2.5

        # Name token overlap
        name_tokens = set(re.findall(r"\b\w+\b", item_name))
        overlap = len(tokens.intersection(name_tokens))
        weights_sum += 1.5
        score += 1.5 * (overlap / max(1, len(tokens)))

        return float(round(score / max(1e-6, weights_sum), 4))


class MultiModalReRanker:
    """Re-ranks visual search candidates using multi-factor similarity weighting."""

    def __init__(
        self,
        weight_visual: float = 0.60,
        weight_color: float = 0.25,
        weight_attr: float = 0.15,
    ):
        tot = weight_visual + weight_color + weight_attr
        self.w_visual = weight_visual / tot
        self.w_color = weight_color / tot
        self.w_attr = weight_attr / tot
        self.color_extractor = ColorHistogramExtractor()

    def rerank(
        self,
        candidate_indices: Sequence[int],
        initial_scores: Sequence[float],
        metadata_df: Any,
        query_image: Optional[Image.Image] = None,
        query_text: Optional[str] = None,
        candidate_images: Optional[Dict[int, Image.Image]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Perform multi-factor re-ranking over candidate matches.

        Returns:
            top_indices: Array of catalog indices
            final_scores: Array of composite similarity scores
            breakdown_list: List of factor contribution dicts
        """
        if len(candidate_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=float), []

        parsed_query = MultiModalQueryComposer.parse_text_query(query_text) if query_text else {}
        query_hist = self.color_extractor.extract_histogram(query_image) if query_image is not None else None

        scored_candidates = []

        for idx, base_score in zip(candidate_indices, initial_scores):
            # 1. Apply hard facet filters if specified
            if filters and hasattr(metadata_df, "iloc"):
                row = metadata_df.iloc[idx]
                skip = False
                for f_col, f_val in filters.items():
                    if f_val and f_col in row:
                        if str(row[f_col]).lower() != str(f_val).lower():
                            skip = True
                            break
                if skip:
                    continue

            # 2. Compute Color Similarity
            color_sim = 1.0
            if query_hist is not None and candidate_images is not None and idx in candidate_images:
                try:
                    cand_hist = self.color_extractor.extract_histogram(candidate_images[idx])
                    color_sim = self.color_extractor.compute_histogram_intersection(query_hist, cand_hist)
                except Exception:
                    color_sim = 0.5
            elif query_hist is not None:
                # Approximate color agreement based on metadata if image tensor unavailable
                if hasattr(metadata_df, "iloc"):
                    row = metadata_df.iloc[idx]
                    color_sim = 0.8 if any(tc in str(row.get("color", "")).lower() for tc in parsed_query.get("target_colors", [])) else 0.5

            # 3. Compute Attribute Relevance
            attr_sim = 1.0
            if parsed_query and hasattr(metadata_df, "iloc"):
                row = metadata_df.iloc[idx].to_dict()
                attr_sim = MultiModalQueryComposer.compute_attribute_score(parsed_query, row)

            # 4. Composite weighted fusion
            final_score = (
                self.w_visual * float(base_score)
                + self.w_color * float(color_sim)
                + self.w_attr * float(attr_sim)
            )

            scored_candidates.append({
                "index": int(idx),
                "final_score": round(float(final_score), 4),
                "visual_score": round(float(base_score), 4),
                "color_score": round(float(color_sim), 4),
                "attr_score": round(float(attr_sim), 4),
            })

        # Sort descending by composite score
        scored_candidates.sort(key=lambda item: item["final_score"], reverse=True)
        top_candidates = scored_candidates[:top_k]

        if not top_candidates:
            return np.array([], dtype=int), np.array([], dtype=float), []

        top_indices = np.array([c["index"] for c in top_candidates], dtype=int)
        top_scores = np.array([c["final_score"] for c in top_candidates], dtype=float)

        return top_indices, top_scores, top_candidates
