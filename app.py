"""Production Streamlit Interface for Multi-Modal Visual Search Engine.

Features:
- Visual Search via image upload or curated sample picker.
- Multi-Modal Query Composition: combine visual image with textual attribute modifiers.
- Color Histogram & Attribute Re-Ranking toggle.
- Explainable Spatial Patch Heatmaps & Dominant Color Palette extraction.
- Seamless fallback to deterministic micro-catalog for 100% offline edge execution.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from catalog import CatalogManager
from config import DEFAULT_TOP_K, MODEL_NAME, PROJECT_DIR
from multimodal_search import (
    ColorHistogramExtractor,
    MultiModalQueryComposer,
    MultiModalReRanker,
)
from search import create_embedding, find_similar_images, load_embedding_model
from visual_explainability import DominantColorExtractor, SpatialPatchComparator


st.set_page_config(page_title="VisualFind AI | Multimodal Catalog Search Engine", page_icon="🔎", layout="wide")


@st.cache_resource
def get_model():
    """Load the pretrained model once, not after every Streamlit interaction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessing = load_embedding_model()
    return model.to(device), preprocessing, device


@st.cache_resource
def get_catalog():
    """Load catalog manager once across sessions."""
    return CatalogManager()


catalog = get_catalog()
metadata = catalog.metadata_df
dataset_embeddings = catalog.embeddings_matrix


# --- Sidebar Search Settings ---
with st.sidebar:
    st.header("⚙️ Search Controls")
    top_k = st.slider("Results to display", 4, 12, DEFAULT_TOP_K)

    st.subheader("🎨 Re-Ranking & Fusion")
    enable_reranking = st.checkbox("Enable Color & Attribute Re-Ranking", value=True)
    show_explainability = st.checkbox("Show Spatial Patch Heatmap", value=True)
    show_palette = st.checkbox("Show Dominant Color Palette", value=True)

    st.subheader("🎯 Facet Filters")
    all_categories = ["All"] + sorted(list(metadata["category"].dropna().unique()))
    selected_category = st.selectbox("Category Filter", all_categories, index=0)

    all_genders = ["All"] + sorted(list(metadata["gender"].dropna().unique()))
    selected_gender = st.selectbox("Gender Filter", all_genders, index=0)

    st.divider()
    if catalog.is_synthetic:
        st.info("ℹ️ Running in Edge Demo Mode (Curated Micro-Catalog).")
    else:
        st.success(f"✓ Catalog Active: {len(metadata):,} indexed products.")
    st.caption(f"Feature Backbone: {MODEL_NAME}")


# --- Main Header ---
st.title("🔎 Enterprise Multi-Modal Visual Search Studio")
st.markdown(
    "Search product catalogs by **visual image**, **text intent**, or **composed multi-modal query** "
    "(e.g. *'this sneaker but in navy blue'*). Powered by deep metric embeddings and 3D HSV color re-ranking."
)

if catalog.is_synthetic:
    st.warning("Notice: Full dataset not detected. Showing curated 20-product micro-catalog fixture.")

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload Query Product Image:", type=["jpg", "jpeg", "png", "webp"])

with col_sample:
    sample_options = ["(Select from catalog)"] + [
        f"#{row['image_id']} - {row.get('product_name', row['category'])}"
        for _, row in metadata.head(10).iterrows()
    ]
    selected_sample = st.selectbox("Or choose sample item:", sample_options, index=0)

# Optional Text Modifier
text_modifier = st.text_input(
    "Text Query / Attribute Modifier (Optional):",
    placeholder="e.g. 'Blue running sneakers' or 'formal dark dress'",
)

query_image = None
if uploaded_file is not None:
    try:
        query_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
elif selected_sample != "(Select from catalog)":
    sample_id = int(selected_sample.split(" - ")[0].replace("#", ""))
    row_match = metadata[metadata["image_id"] == sample_id]
    if not row_match.empty:
        idx_match = row_match.index[0]
        query_image = catalog.get_image(idx_match)

# --- Execution ---
if query_image is not None or text_modifier.strip():
    col_qimg, col_info = st.columns([1, 2])

    with col_qimg:
        st.subheader("Active Query")
        if query_image is not None:
            st.image(query_image, caption="Query Image", use_container_width=True)
            if show_palette:
                palette = DominantColorExtractor.extract_palette(query_image, num_colors=4)
                st.caption("Dominant Colors: " + " ".join(f"`{c['hex']}` ({c['percentage']}%)" for c in palette))
        else:
            st.info(f"Text-Only Query: **\"{text_modifier}\"**")

    with col_info:
        st.subheader("Retrieval Pipeline Workflow")
        st.markdown(
            "1. **Feature Extraction**: Visual embeddings generated via MobileNetV3 (576 dims).  \n"
            "2. **Vector Retrieval**: Cosine similarity against catalog embeddings.  \n"
            "3. **Color & Attribute Re-Ranking**: 3D HSV histogram intersection and text attribute boost.  \n"
            "4. **Explainable Spatial Alignment**: Regional patch similarity matrix computed."
        )

    t0 = time.time()
    model, preprocessing, device = get_model()

    if query_image is not None:
        query_embedding = create_embedding(query_image, model, preprocessing, device)
    else:
        parsed = MultiModalQueryComposer.parse_text_query(text_modifier)
        target_cats = parsed.get("target_categories", [])
        mask = np.zeros(len(metadata), dtype=bool)
        if target_cats:
            for cat in target_cats:
                mask |= metadata["category"].str.lower() == cat.lower()
        query_embedding = np.mean(dataset_embeddings[mask], axis=0) if np.any(mask) else dataset_embeddings[0]
        norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / (norm or 1.0)

    # Initial candidate fetch
    fetch_k = min(len(dataset_embeddings), max(top_k * 3, 25))
    cand_indices, cand_scores = find_similar_images(query_embedding, dataset_embeddings, top_k=fetch_k)

    filters = {}
    if selected_category != "All":
        filters["category"] = selected_category
    if selected_gender != "All":
        filters["gender"] = selected_gender

    if enable_reranking:
        cand_images = {idx: catalog.get_image(idx) for idx in cand_indices}
        reranker = MultiModalReRanker()
        final_indices, final_scores, breakdowns = reranker.rerank(
            candidate_indices=cand_indices,
            initial_scores=cand_scores,
            metadata_df=metadata,
            query_image=query_image,
            query_text=text_modifier if text_modifier.strip() else None,
            candidate_images=cand_images,
            filters=filters,
            top_k=top_k,
        )
    else:
        # Filter manually if not re-ranking
        filtered_indices = []
        filtered_scores = []
        for idx, s in zip(cand_indices, cand_scores):
            row = metadata.iloc[idx]
            if selected_category != "All" and str(row.get("category", "")).lower() != selected_category.lower():
                continue
            if selected_gender != "All" and str(row.get("gender", "")).lower() != selected_gender.lower():
                continue
            filtered_indices.append(idx)
            filtered_scores.append(s)
            if len(filtered_indices) >= top_k:
                break
        final_indices = np.array(filtered_indices, dtype=int)
        final_scores = np.array(filtered_scores, dtype=float)

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    st.divider()
    st.subheader(f"Matching Products ({len(final_indices)} retrieved in {elapsed_ms} ms)")

    if len(final_indices) == 0:
        st.warning("No catalog products matched the specified facet filters.")
    else:
        # Display in rows of 4 columns
        for row_start in range(0, len(final_indices), 4):
            cols = st.columns(min(4, len(final_indices) - row_start))
            for c_idx, result_idx in enumerate(final_indices[row_start : row_start + 4]):
                score = final_scores[row_start + c_idx]
                row = metadata.iloc[result_idx]
                cand_img = catalog.get_image(result_idx)

                with cols[c_idx]:
                    st.image(cand_img, use_container_width=True)
                    st.markdown(f"**{row.get('product_name', row['category'])}**")
                    st.caption(f"{row.get('article_type', '')} · {row.get('color', '')} · {row.get('gender', '')}")
                    st.metric("Match Confidence", f"{score * 100:.1f}%")

                    if show_explainability and query_image is not None:
                        with st.expander("🔍 Match Diagnostics"):
                            comparator = SpatialPatchComparator(grid_size=4)
                            patch_info = comparator.compute_patch_similarity_heatmap(query_image, cand_img)
                            st.write(f"**Regional Patch Alignment**: `{patch_info['mean_patch_similarity']*100:.1f}%`")
                            top_reg = ", ".join(r["region"] for r in patch_info["top_aligned_regions"][:2])
                            st.caption(f"Strongest feature match: {top_reg}")
else:
    st.info("👆 Upload an image or select a sample product above to start visual search.")
