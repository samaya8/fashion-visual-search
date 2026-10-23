"""Explainable Visual Search Diagnostics: Spatial Patch Heatmaps & Color Palette Analytics.

Provides:
1. SpatialPatchComparator: Divides query and matched images into an NxN grid of spatial patches
   and calculates localized visual similarity to explain which image regions drove retrieval.
2. DominantColorExtractor: Quantizes RGB pixel distributions into dominant color palettes with
   hex codes and percentage coverage.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class DominantColorExtractor:
    """Extracts top dominant color palettes with percentage coverage."""

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """Convert RGB triplet to hex string format."""
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    @classmethod
    def extract_palette(cls, image: Image.Image, num_colors: int = 5) -> List[Dict[str, Any]]:
        """Extract dominant color palette via fast median-cut color quantization."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        img_rgb = image.convert("RGB")
        # Scale down for instant CPU quantization
        img_thumb = img_rgb.resize((64, 64), Image.Resampling.BILINEAR)

        # Quantize to 16 colors first using adaptive palette
        quantized = img_thumb.quantize(colors=max(8, num_colors * 2), method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()[:num_colors * 2 * 3]

        # Count pixel frequencies for each palette index
        counts = collections.Counter(np.array(quantized).flatten())
        total_pixels = sum(counts.values()) or 1

        color_entries = []
        for color_idx, count in counts.most_common(num_colors):
            base_idx = color_idx * 3
            if base_idx + 2 < len(palette):
                r, g, b = palette[base_idx], palette[base_idx + 1], palette[base_idx + 2]
                pct = round((count / total_pixels) * 100.0, 1)
                color_entries.append({
                    "hex": cls.rgb_to_hex(r, g, b),
                    "rgb": (int(r), int(g), int(b)),
                    "percentage": pct,
                })

        return color_entries


class SpatialPatchComparator:
    """Computes regional patch similarity between query and retrieved product images."""

    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        self.total_patches = grid_size * grid_size

    def _extract_patch_features(self, img: Image.Image) -> np.ndarray:
        """Split image into grid_size x grid_size tiles and compute normalized RGB feature vectors."""
        img_rgb = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr_rgb = np.array(img_rgb, dtype=np.float32) / 255.0

        step = 128 // self.grid_size
        patch_features = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                y0, y1 = r * step, (r + 1) * step
                x0, x1 = c * step, (c + 1) * step

                sub_rgb = arr_rgb[y0:y1, x0:x1]

                mean_rgb = np.mean(sub_rgb, axis=(0, 1))
                std_rgb = np.std(sub_rgb, axis=(0, 1))

                feat = np.concatenate([mean_rgb, std_rgb])
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat /= norm
                patch_features.append(feat)

        return np.array(patch_features, dtype=np.float32)

    def compute_patch_similarity_heatmap(
        self,
        query_image: Image.Image,
        target_image: Image.Image,
    ) -> Dict[str, Any]:
        """Calculate spatial patch-to-patch similarity grid and identify highest-alignment visual regions."""
        if not isinstance(query_image, Image.Image) or not isinstance(target_image, Image.Image):
            raise TypeError("Both images must be PIL Image instances")

        q_patches = self._extract_patch_features(query_image)
        t_patches = self._extract_patch_features(target_image)

        # Positional alignment: dot product between corresponding spatial tiles
        patch_sims = np.clip(np.sum(q_patches * t_patches, axis=1), 0.0, 1.0)
        heatmap_matrix = patch_sims.reshape(self.grid_size, self.grid_size)

        # Region labeling for explainability
        row_labels = ["Top", "Upper-Middle", "Lower-Middle", "Bottom"] if self.grid_size == 4 else [f"Row {i}" for i in range(self.grid_size)]
        col_labels = ["Left", "Center-Left", "Center-Right", "Right"] if self.grid_size == 4 else [f"Col {j}" for j in range(self.grid_size)]

        region_scores = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                region_name = f"{row_labels[r]} {col_labels[c]}"
                score = round(float(heatmap_matrix[r, c]), 3)
                region_scores.append({"region": region_name, "row": r, "col": c, "similarity": score})

        region_scores.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "grid_size": self.grid_size,
            "heatmap_matrix": np.round(heatmap_matrix, 3).tolist(),
            "mean_patch_similarity": round(float(np.mean(patch_sims)), 3),
            "max_patch_similarity": round(float(np.max(patch_sims)), 3),
            "top_aligned_regions": region_scores[:3],
            "least_aligned_regions": region_scores[-2:],
        }
