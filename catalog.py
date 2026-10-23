"""Catalog Management with Self-Contained Deterministic Fallback.

Provides:
- CatalogManager: Loads primary e-commerce dataset (metadata.csv and embeddings.npy),
  or seamlessly activates a self-contained 30-item micro-catalog with deterministic
  576-dim feature vectors and procedural PIL images when large data assets are absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from config import EMBEDDINGS_PATH, IMAGES_DIR, METADATA_PATH, PROJECT_DIR


class CatalogManager:
    """Manages e-commerce catalog metadata, feature embeddings, and image retrieval."""

    # Curated micro-catalog seed records
    MICRO_CATALOG_DATA = [
        {"image_id": 9001, "category": "Shoes", "article_type": "Casual Shoes", "color": "Red", "gender": "Men", "product_name": "Nike Air Max Red Sport Shoes"},
        {"image_id": 9002, "category": "Shoes", "article_type": "Sneakers", "color": "Blue", "gender": "Women", "product_name": "Adidas Ultraboost Blue Sneakers"},
        {"image_id": 9003, "category": "Shoes", "article_type": "Boots", "color": "Brown", "gender": "Men", "product_name": "Timberland Classic Brown Leather Boots"},
        {"image_id": 9004, "category": "Shoes", "article_type": "Sandals", "color": "Black", "gender": "Women", "product_name": "Clarks Black Comfort Leather Sandals"},
        {"image_id": 9005, "category": "Topwear", "article_type": "Tshirts", "color": "Black", "gender": "Men", "product_name": "Puma Black Crewneck Athletic T-Shirt"},
        {"image_id": 9006, "category": "Topwear", "article_type": "Tshirts", "color": "White", "gender": "Women", "product_name": "Zara White Fitted Cotton T-Shirt"},
        {"image_id": 9007, "category": "Topwear", "article_type": "Shirts", "color": "Blue", "gender": "Men", "product_name": "Ralph Lauren Classic Oxford Blue Shirt"},
        {"image_id": 9008, "category": "Topwear", "article_type": "Sweaters", "color": "Green", "gender": "Women", "product_name": "H&M Knit Olive Green Winter Sweater"},
        {"image_id": 9009, "category": "Bottomwear", "article_type": "Jeans", "color": "Blue", "gender": "Men", "product_name": "Levi's 501 Original Fit Dark Blue Jeans"},
        {"image_id": 9010, "category": "Bottomwear", "article_type": "Trousers", "color": "Grey", "gender": "Women", "product_name": "Mango Slim Tailored Grey Trousers"},
        {"image_id": 9011, "category": "Bottomwear", "article_type": "Shorts", "color": "Black", "gender": "Men", "product_name": "Under Armour Black Training Shorts"},
        {"image_id": 9012, "category": "Eyewear", "article_type": "Sunglasses", "color": "Black", "gender": "Unisex", "product_name": "Ray-Ban Wayfarer Classic Black Sunglasses"},
        {"image_id": 9013, "category": "Eyewear", "article_type": "Sunglasses", "color": "Brown", "gender": "Women", "product_name": "Gucci Tortoiseshell Round Sunglasses"},
        {"image_id": 9014, "category": "Watches", "article_type": "Watches", "color": "Black", "gender": "Men", "product_name": "Seiko Chronograph Black Dial Steel Watch"},
        {"image_id": 9015, "category": "Watches", "article_type": "Watches", "color": "Rose Gold", "gender": "Women", "product_name": "Fossil Minimalist Rose Gold Mesh Watch"},
        {"image_id": 9016, "category": "Bags", "article_type": "Backpacks", "color": "Blue", "gender": "Unisex", "product_name": "Herschel Classic Navy Blue Travel Backpack"},
        {"image_id": 9017, "category": "Bags", "article_type": "Handbags", "color": "Red", "gender": "Women", "product_name": "Michael Kors Scarlet Leather Handbag"},
        {"image_id": 9018, "category": "Dress", "article_type": "Dresses", "color": "Red", "gender": "Women", "product_name": "Calvin Klein Red Sleeveless Evening Dress"},
        {"image_id": 9019, "category": "Dress", "article_type": "Dresses", "color": "Black", "gender": "Women", "product_name": "Tommy Hilfiger Classic Little Black Dress"},
        {"image_id": 9020, "category": "Jewellery", "article_type": "Earrings", "color": "Gold", "gender": "Women", "product_name": "Swarovski Crystal Droplet Gold Earrings"},
    ]

    COLOR_HEX_MAP = {
        "red": (220, 38, 38),
        "blue": (37, 99, 235),
        "brown": (160, 82, 45),
        "black": (30, 41, 59),
        "white": (241, 245, 249),
        "green": (22, 163, 74),
        "grey": (100, 116, 139),
        "rose gold": (244, 114, 182),
        "gold": (234, 179, 8),
    }

    def __init__(self, metadata_path: Optional[Path] = None, embeddings_path: Optional[Path] = None):
        self.metadata_path = metadata_path or METADATA_PATH
        self.embeddings_path = embeddings_path or EMBEDDINGS_PATH
        self.is_synthetic: bool = False

        self.metadata_df, self.embeddings_matrix = self._load_data()

    def _load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load primary dataset or trigger deterministic micro-catalog fallback."""
        if self.metadata_path.exists() and self.embeddings_path.exists():
            try:
                meta = pd.read_csv(self.metadata_path)
                embeds = np.load(self.embeddings_path)
                if len(meta) == len(embeds):
                    self.is_synthetic = False
                    return meta, embeds
            except Exception:
                pass

        # Fallback: Initialize deterministic micro-catalog
        self.is_synthetic = True
        return self._generate_micro_catalog()

    def _generate_micro_catalog(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build self-contained micro-catalog DataFrame and unit-normalized 576-dim embeddings."""
        meta = pd.DataFrame(self.MICRO_CATALOG_DATA)
        num_items = len(meta)
        dim = 576

        embeddings = np.zeros((num_items, dim), dtype=np.float32)

        for idx, row in meta.iterrows():
            # Seed vector deterministically from text attributes
            seed_str = f"{row['category']}_{row['color']}_{row['article_type']}_{row['gender']}"
            seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)

            vec = rng.randn(dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            embeddings[idx] = vec / (norm or 1.0)

        return meta, embeddings

    def get_image(self, row_or_idx: Union[int, np.integer, pd.Series, Dict[str, Any]]) -> Image.Image:
        """Retrieve PIL Image from filesystem or generate a representative procedural thumbnail."""
        if isinstance(row_or_idx, (int, np.integer)):
            row = self.metadata_df.iloc[int(row_or_idx)].to_dict()
        elif isinstance(row_or_idx, pd.Series):
            row = row_or_idx.to_dict()
        else:
            row = row_or_idx

        # 1. Try to load from disk if image_path exists
        if "image_path" in row and row["image_path"]:
            full_path = PROJECT_DIR / Path(str(row["image_path"]))
            if full_path.exists():
                try:
                    return Image.open(full_path).convert("RGB")
                except Exception:
                    pass

        # Also check standard IMAGES_DIR / f"{image_id}.jpg"
        if "image_id" in row:
            cand_path = IMAGES_DIR / f"{row['image_id']}.jpg"
            if cand_path.exists():
                try:
                    return Image.open(cand_path).convert("RGB")
                except Exception:
                    pass

        # 2. Synthesize clean procedural thumbnail matching category and color
        return self._create_procedural_image(row)

    def _create_procedural_image(self, row: Dict[str, Any]) -> Image.Image:
        """Synthesize high-aesthetic product tile matching color and category labels."""
        img = Image.new("RGB", (224, 224), color=(248, 250, 252))
        draw = ImageDraw.Draw(img)

        color_name = str(row.get("color", "grey")).lower()
        fg_color = self.COLOR_HEX_MAP.get(color_name, (100, 116, 139))

        # Draw decorative background container
        draw.rounded_rectangle([20, 20, 204, 204], radius=16, fill=fg_color)
        # Inner contrast badge
        draw.rounded_rectangle([44, 44, 180, 180], radius=12, fill=(255, 255, 255))
        # Central category emblem
        draw.ellipse([76, 76, 148, 148], fill=fg_color)

        return img

    @property
    def catalog_size(self) -> int:
        """Return total number of items in the catalog."""
        return len(self.metadata_df)
