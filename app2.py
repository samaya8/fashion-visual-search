"""Premium Streamlit interface for the Visual Search Engine LinkedIn demo."""

from html import escape
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image, ImageOps

from config import EMBEDDINGS_PATH, METADATA_PATH, MODEL_NAME, PROJECT_DIR
from search import create_embedding, find_similar_images, load_embedding_model


st.set_page_config(
    page_title="VisualFind AI | Visual Search Analytics Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #0b1220;
        --muted: #64748b;
        --line: rgba(15, 23, 42, 0.10);
        --blue: #2563eb;
        --cyan: #06b6d4;
        --violet: #7c3aed;
        --surface: rgba(255, 255, 255, 0.86);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 5%, rgba(6, 182, 212, 0.12), transparent 25%),
            radial-gradient(circle at 88% 4%, rgba(124, 58, 237, 0.12), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, #f5f7fb 45%, #ffffff 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(248, 250, 252, 0.94);
        border-right: 1px solid rgba(15, 23, 42, 0.08);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 3.1rem 3.25rem;
        border-radius: 30px;
        background:
            radial-gradient(circle at 88% 10%, rgba(34, 211, 238, 0.28), transparent 28%),
            radial-gradient(circle at 72% 90%, rgba(139, 92, 246, 0.24), transparent 31%),
            linear-gradient(125deg, #071426 0%, #102b55 57%, #163e68 100%);
        box-shadow: 0 30px 80px rgba(15, 42, 78, 0.20);
        margin-bottom: 1.15rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -65px;
        top: -85px;
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 50%;
        box-shadow: 0 0 0 42px rgba(255,255,255,0.025), 0 0 0 84px rgba(255,255,255,0.018);
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.85rem;
        border: 1px solid rgba(125, 211, 252, 0.25);
        border-radius: 999px;
        color: #b9efff;
        background: rgba(14, 116, 144, 0.20);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        color: white;
        font-size: clamp(2.6rem, 5vw, 5.1rem);
        line-height: 0.98;
        letter-spacing: -0.055em;
        max-width: 900px;
        margin: 1.25rem 0 1rem;
    }

    .gradient-word {
        background: linear-gradient(90deg, #67e8f9, #93c5fd 48%, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #c9d9ec;
        max-width: 760px;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
    }

    .hero-footer {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.6rem;
    }

    .hero-chip {
        padding: 0.48rem 0.78rem;
        border-radius: 10px;
        color: #e5f3ff;
        background: rgba(255,255,255,0.075);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 0.78rem;
        font-weight: 650;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 1rem 0 1.65rem;
    }

    .stat-card {
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 17px;
        background: var(--surface);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.055);
        backdrop-filter: blur(12px);
    }

    .stat-value {
        color: var(--ink);
        font-size: 1.3rem;
        font-weight: 850;
        letter-spacing: -0.03em;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.15rem;
    }

    .section-kicker {
        color: #0891b2;
        font-size: 0.72rem;
        font-weight: 850;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 850;
        letter-spacing: -0.035em;
        margin-bottom: 0.25rem;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .query-label {
        display: inline-flex;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.73rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem;
        margin-top: 0.9rem;
    }

    .pipeline-step {
        min-height: 115px;
        padding: 0.95rem;
        border-radius: 16px;
        background: linear-gradient(145deg, #ffffff, #f7fbff);
        border: 1px solid rgba(37, 99, 235, 0.12);
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.06);
    }

    .pipeline-number {
        display: grid;
        place-items: center;
        width: 27px;
        height: 27px;
        border-radius: 9px;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        color: white;
        font-size: 0.72rem;
        font-weight: 900;
        margin-bottom: 0.65rem;
    }

    .pipeline-step b {
        display: block;
        color: #172554;
        font-size: 0.83rem;
        margin-bottom: 0.25rem;
    }

    .pipeline-step span {
        color: var(--muted);
        font-size: 0.72rem;
        line-height: 1.4;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 155px;
        border: 1.5px dashed rgba(37, 99, 235, 0.38);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(239,246,255,.9), rgba(236,254,255,.78));
    }

    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 999px;
        border: 0;
        background: #0f2f57;
        color: white;
        font-weight: 750;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(15, 23, 42, 0.09) !important;
        border-radius: 21px !important;
        background: rgba(255, 255, 255, 0.84);
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.065);
        overflow: hidden;
    }

    .result-rank {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.32rem 0.62rem;
        border-radius: 999px;
        color: #1d4ed8;
        background: #eff6ff;
        border: 1px solid #dbeafe;
        font-size: 0.7rem;
        font-weight: 850;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .product-name {
        color: #0f172a;
        font-size: 0.96rem;
        font-weight: 800;
        line-height: 1.35;
        min-height: 2.65rem;
        margin: 0.6rem 0 0.45rem;
    }

    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: var(--muted);
        font-size: 0.72rem;
        margin-bottom: 0.35rem;
    }

    .score-value {
        color: #0f766e;
        font-size: 0.96rem;
        font-weight: 900;
    }

    .score-track {
        height: 6px;
        border-radius: 999px;
        background: #e8eef5;
        overflow: hidden;
        margin-bottom: 0.7rem;
    }

    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #06b6d4, #2563eb, #7c3aed);
    }

    .meta-chip {
        display: inline-block;
        padding: 0.27rem 0.48rem;
        margin: 0.12rem 0.08rem 0.08rem 0;
        border-radius: 7px;
        background: #f1f5f9;
        color: #475569;
        font-size: 0.67rem;
        font-weight: 700;
    }

    .empty-state {
        padding: 2rem;
        text-align: center;
        border-radius: 24px;
        border: 1px solid rgba(37, 99, 235, 0.10);
        background: rgba(255,255,255,0.68);
    }

    .empty-icon {
        width: 62px;
        height: 62px;
        display: grid;
        place-items: center;
        margin: 0 auto 0.8rem;
        border-radius: 19px;
        color: white;
        font-size: 1.6rem;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        box-shadow: 0 15px 35px rgba(37,99,235,.25);
    }

    .footer-card {
        margin-top: 2rem;
        padding: 1rem 1.2rem;
        border-top: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        color: var(--muted);
        font-size: 0.75rem;
    }

    .sidebar-brand {
        font-size: 1.45rem;
        font-weight: 900;
        color: #0f2f57;
        letter-spacing: -0.04em;
        margin-bottom: 0.1rem;
    }

    .sidebar-brand span { color: #0891b2; }
    .sidebar-copy { color: #64748b; font-size: 0.78rem; line-height: 1.5; }

    .sidebar-card {
        padding: 0.9rem;
        border-radius: 15px;
        background: linear-gradient(145deg, #eef8ff, #eefcfb);
        border: 1px solid #d9eef5;
        color: #334155;
        font-size: 0.75rem;
        line-height: 1.55;
        margin-top: 1rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 750;
        border: 1px solid rgba(37,99,235,.18);
    }

    .quick-start-box {
        margin: 1.25rem 0 0.8rem;
        padding: 1.1rem 1.25rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid rgba(37, 99, 235, 0.12);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    }

    .quick-start-title {
        color: #0f2f57;
        font-size: 1.05rem;
        font-weight: 850;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .quick-start-subtitle {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.5;
        margin: 0;
    }

    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .product-price {
        color: #0f766e;
        font-size: 1.05rem;
        font-weight: 900;
        margin: 0.25rem 0 0.55rem;
    }

    .cart-summary-box {
        padding: 0.85rem;
        border-radius: 14px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        margin-bottom: 0.8rem;
    }

    .filter-count-badge {
        font-size: 0.72rem;
        color: #0369a1;
        font-weight: 700;
        background: #e0f2fe;
        padding: 0.25rem 0.55rem;
        border-radius: 7px;
        display: inline-block;
        margin: 0.3rem 0 0.5rem;
    }

    @media (max-width: 900px) {
        .hero { padding: 2rem 1.5rem; }
        .stat-grid { grid-template-columns: repeat(2, 1fr); }
        .pipeline { grid-template-columns: repeat(2, 1fr); }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the pretrained model once for the entire Streamlit session."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocessing = load_embedding_model()
    return model.to(device), preprocessing, device


@st.cache_data(show_spinner=False)
def load_search_data():
    """Load the saved catalog once instead of re-reading it after every click."""
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    return embeddings, metadata


@st.cache_data(show_spinner=False)
def make_square_thumbnail(image_path, size=620):
    """Place a product photo on a consistent square canvas for clean cards."""
    with Image.open(image_path) as image:
        product_image = image.convert("RGB")

    # The teaching dataset stores compact 60x80 catalog JPEGs. Upscale them for
    # presentation while preserving their original aspect ratio.
    available_size = size - 72
    scale = min(
        available_size / product_image.width,
        available_size / product_image.height,
    )
    resized_size = (
        max(1, round(product_image.width * scale)),
        max(1, round(product_image.height * scale)),
    )
    product_image = product_image.resize(resized_size, Image.Resampling.LANCZOS)

    background = Image.new("RGB", (size, size), "#f8fafc")
    left = (size - product_image.width) // 2
    top = (size - product_image.height) // 2
    background.paste(product_image, (left, top))
    return background


def download_image_from_url(url: str, timeout: int = 7) -> Image.Image:
    """Download an image from a public URL with validation and safety headers."""
    clean_url = url.strip()
    if not clean_url or not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        raise ValueError("Please provide a valid HTTP or HTTPS image URL.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(clean_url, headers=headers, timeout=timeout, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and not any(
        valid_type in content_type for valid_type in ["image/", "octet-stream"]
    ):
        raise ValueError(f"URL does not point to an image (Content-Type: {content_type}).")

    content = response.raw.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Image file exceeds the 10 MB size limit.")

    image = Image.open(BytesIO(content)).convert("RGB")
    return image


SAMPLE_QUERIES = [
    {
        "id": "10294",
        "title": "Formal Shoes",
        "name": "Clarks Brown Leather Formal Shoes",
        "category": "Footwear",
        "path": "data/images/10294.jpg",
    },
    {
        "id": "6388",
        "title": "Sports Shoe",
        "name": "Puma PowerCat Black Sports Shoe",
        "category": "Footwear",
        "path": "data/images/6388.jpg",
    },
    {
        "id": "43672",
        "title": "Summer Dress",
        "name": "ONLY Women Peach Dress",
        "category": "Apparel",
        "path": "data/images/43672.jpg",
    },
    {
        "id": "6946",
        "title": "Graphic Tee",
        "name": "Tantra Men's Graphic Black T-shirt",
        "category": "Apparel",
        "path": "data/images/6946.jpg",
    },
    {
        "id": "45268",
        "title": "G-Shock Watch",
        "name": "CASIO G-Shock Digital Watch",
        "category": "Accessories",
        "path": "data/images/45268.jpg",
    },
    {
        "id": "51913",
        "title": "Sling Bag",
        "name": "Nyk Women White Sling Bag",
        "category": "Accessories",
        "path": "data/images/51913.jpg",
    },
    {
        "id": "49664",
        "title": "Sunglasses",
        "name": "Benetton Men Sunglasses",
        "category": "Accessories",
        "path": "data/images/49664.jpg",
    },
    {
        "id": "28268",
        "title": "Deo Fragrance",
        "name": "Gatsby Men Intense Deo Spray",
        "category": "Personal Care",
        "path": "data/images/28268.jpg",
    },
]


def get_simulated_price(image_id: str) -> float:
    """Generate a stable, realistic simulated price based on the product ID."""
    try:
        numeric_id = int(image_id)
    except ValueError:
        numeric_id = sum(ord(c) for c in str(image_id))
    base = 29.99
    offset = (numeric_id % 70) * 1.75
    return round(base + offset, 2)


def add_to_cart(product_id: str, name: str, price: float, image_path: str):
    """Add a product to the persistent session shopping cart."""
    if "cart" not in st.session_state:
        st.session_state.cart = []

    for item in st.session_state.cart:
        if item["id"] == str(product_id):
            item["quantity"] += 1
            return
    st.session_state.cart.append(
        {
            "id": str(product_id),
            "name": name,
            "price": price,
            "image_path": str(image_path),
            "quantity": 1,
        }
    )


@st.dialog("Product Specifications & Details")
def show_product_detail_modal(row, score, price):
    """Render a luxury modal dialog with full product specifications and actions."""
    img_path = PROJECT_DIR / Path(row["image_path"])
    score_percent = max(0.0, min(100.0, float(score) * 100))

    modal_left, modal_right = st.columns([1, 1.25], gap="medium")
    with modal_left:
        if img_path.exists():
            st.image(make_square_thumbnail(str(img_path), size=480), width="stretch")
        st.metric("Visual Match Confidence", f"{score_percent:.1f}%")
        st.caption("Based on 576-D MobileNetV3 visual embedding")

    with modal_right:
        st.subheader(row.get("product_name", "Fashion Item"))
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.75rem;margin:0.5rem 0 1rem;">
                <span style="font-size:1.6rem;font-weight:900;color:#0f766e;">${price:.2f}</span>
                <span style="color:#16a34a;background:#dcfce7;border:1px solid #bbf7d0;padding:0.2rem 0.6rem;border-radius:999px;font-size:0.75rem;font-weight:800;">● IN STOCK</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        spec_data = {
            "Attribute": [
                "Product ID",
                "Master Category",
                "Subcategory",
                "Article Type",
                "Color",
                "Gender",
                "Season",
                "Usage Occasion",
            ],
            "Details": [
                str(row.get("image_id", "N/A")),
                str(row.get("master_category", "N/A")),
                str(row.get("subcategory", "N/A")),
                str(row.get("article_type", "N/A")),
                str(row.get("color", "N/A")),
                str(row.get("gender", "N/A")),
                str(row.get("season", "N/A")),
                str(row.get("usage", "N/A")),
            ],
        }
        st.dataframe(pd.DataFrame(spec_data), hide_index=True, use_container_width=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🛒 Add to Bag", key=f"modal_add_btn_{row['image_id']}", use_container_width=True):
                add_to_cart(str(row["image_id"]), str(row.get("product_name", "Item")), price, str(row["image_path"]))
                st.toast(f"Added '{row.get('product_name', 'Item')}' to your shopping cart!")
        with btn_col2:
            if st.button("🔄 Pivot to this", key=f"modal_pivot_btn_{row['image_id']}", use_container_width=True):
                if img_path.exists():
                    st.session_state.query_image = Image.open(img_path).convert("RGB")
                    st.session_state.query_name = row.get("product_name", "Catalog Item")
                    st.session_state.query_source = f"Pivot: #{row['image_id']}"
                    st.rerun()


def text_value(row, field, fallback="Unknown"):
    """Read clean display text from a metadata row."""
    value = row.get(field, fallback)
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value)


def show_result_card(column, rank, row, score):
    image_path = PROJECT_DIR / Path(row["image_path"])
    score_percent = max(0.0, min(100.0, float(score) * 100))

    product_name = text_value(row, "product_name", text_value(row, "category"))
    escaped_name = escape(product_name)
    article_type = escape(text_value(row, "article_type", text_value(row, "category")))
    color = escape(text_value(row, "color"))
    gender = escape(text_value(row, "gender"))
    usage = escape(text_value(row, "usage"))
    category = escape(text_value(row, "category"))
    image_id = str(text_value(row, "image_id"))
    price = get_simulated_price(image_id)

    with column:
        with st.container(border=True):
            st.markdown(
                f'<div class="result-rank">✦ Match {rank:02d}</div>',
                unsafe_allow_html=True,
            )
            st.image(
                make_square_thumbnail(str(image_path)),
                width="stretch",
            )
            st.markdown(
                f"""
                <div class="product-name">{escaped_name}</div>
                <div class="product-price">${price:.2f}</div>
                <div class="score-row">
                    <span>Cosine match</span>
                    <span class="score-value">{score_percent:.1f}%</span>
                </div>
                <div class="score-track">
                    <div class="score-fill" style="width:{score_percent:.1f}%"></div>
                </div>
                <span class="meta-chip">{article_type}</span>
                <span class="meta-chip">{color}</span>
                <span class="meta-chip">{gender}</span>
                <span class="meta-chip">{usage}</span>
                <div style="color:#94a3b8;font-size:.66rem;margin:0.5rem 0 0.65rem;">
                    {category} &nbsp;•&nbsp; Product #{image_id}
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_row1, btn_row2 = st.columns(2)
            with btn_row1:
                if st.button("🔍 View", key=f"qv_{image_id}_{rank}", use_container_width=True, help="Open full product specifications"):
                    show_product_detail_modal(row, score, price)
            with btn_row2:
                if st.button("🛒 Add", key=f"cart_{image_id}_{rank}", use_container_width=True, help="Add to shopping cart"):
                    add_to_cart(image_id, product_name, price, str(image_path))
                    st.toast(f"Added to cart: ${price:.2f}")

            if st.button(
                "🔄 More Like This",
                key=f"pivot_{image_id}_{rank}",
                use_container_width=True,
                help="Re-run search using this item as query",
            ):
                if image_path.exists():
                    st.session_state.query_image = Image.open(image_path).convert("RGB")
                    st.session_state.query_name = product_name
                    st.session_state.query_source = f"Pivot: #{image_id}"
                    st.rerun()


def build_results_table(indices, scores, metadata):
    rows = []
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        item = metadata.iloc[index]
        rows.append(
            {
                "rank": rank,
                "similarity": round(float(score), 5),
                "product_id": text_value(item, "image_id"),
                "product_name": text_value(item, "product_name"),
                "category": text_value(item, "category"),
                "article_type": text_value(item, "article_type"),
                "color": text_value(item, "color"),
                "gender": text_value(item, "gender"),
                "usage": text_value(item, "usage"),
            }
        )
    return pd.DataFrame(rows)


# Validate and load the prepared search catalog before rendering dynamic stats.
if not METADATA_PATH.exists() or not EMBEDDINGS_PATH.exists():
    st.error("The prepared catalog is missing. Generate metadata and embeddings first.")
    st.code("python prepare_dataset.py\npython generate_embeddings.py")
    st.stop()

try:
    dataset_embeddings, metadata = load_search_data()
except Exception as error:
    st.error(f"The search catalog could not be loaded: {error}")
    st.stop()

if len(dataset_embeddings) != len(metadata):
    st.error("Metadata and embedding row counts do not match. Regenerate embeddings.")
    st.stop()


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">LUMINA<span>AI</span></div>
        <div class="sidebar-copy">Visual intelligence for effortless product discovery.</div>
        """,
        unsafe_allow_html=True,
    )

    # Task 5: Shopping Bag / Cart Drawer
    st.divider()
    cart_items = st.session_state.get("cart", [])
    cart_count = sum(item["quantity"] for item in cart_items)
    cart_total = sum(item["price"] * item["quantity"] for item in cart_items)

    st.markdown(f"#### 🛍️ Shopping Bag `({cart_count})`")
    if not cart_items:
        st.caption("Your bag is empty. Click **🛒 Add** on any product card.")
    else:
        with st.expander(f"Bag Total: ${cart_total:.2f}", expanded=True):
            for c_item in cart_items:
                st.markdown(f"**{c_item['name']}**")
                st.caption(f"${c_item['price']:.2f} × {c_item['quantity']} = ${c_item['price'] * c_item['quantity']:.2f}")
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Checkout", key="btn_checkout_sidebar", use_container_width=True):
                    st.balloons()
                    st.toast("🎉 Order placed successfully in simulated checkout!")
                    st.session_state.cart = []
                    st.rerun()
            with c_btn2:
                if st.button("Empty", key="btn_empty_sidebar", use_container_width=True):
                    st.session_state.cart = []
                    st.rerun()

    # Task 3: Hybrid Search Metadata Filters
    st.divider()
    st.markdown("#### 🎯 Hybrid Filters")
    st.caption("Filter catalog candidates before visual ranking:")

    master_cats = ["All"] + sorted(metadata["master_category"].dropna().unique().tolist())
    filter_master_cat = st.selectbox("Category", master_cats, index=0)

    genders = ["All", "Men", "Women", "Unisex", "Boys", "Girls"]
    filter_gender = st.selectbox("Target Gender", genders, index=0)

    top_colors = [
        "All",
        "Black",
        "White",
        "Brown",
        "Blue",
        "Grey",
        "Red",
        "Green",
        "Pink",
        "Purple",
        "Navy Blue",
    ]
    filter_color = st.selectbox("Dominant Color", top_colors, index=0)

    usages = ["All", "Casual", "Sports", "Formal", "Ethnic"]
    filter_usage = st.selectbox("Occasion / Usage", usages, index=0)

    filter_mask = pd.Series(True, index=metadata.index)
    if filter_master_cat != "All":
        filter_mask &= (metadata["master_category"] == filter_master_cat)
    if filter_gender != "All":
        filter_mask &= (metadata["gender"] == filter_gender)
    if filter_color != "All":
        filter_mask &= (metadata["color"] == filter_color)
    if filter_usage != "All":
        filter_mask &= (metadata["usage"] == filter_usage)

    candidate_indices = np.where(filter_mask)[0]
    is_filtered = (
        filter_master_cat != "All"
        or filter_gender != "All"
        or filter_color != "All"
        or filter_usage != "All"
    )

    st.markdown(
        f'<div class="filter-count-badge">Matching: {len(candidate_indices):,} / {len(metadata):,} items</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### Search experience")
    top_k = st.select_slider(
        "Number of visual matches",
        options=[5, 6, 8, 10],
        value=5,
    )
    show_details = st.toggle("Show technical details", value=False)
    st.divider()
    st.markdown("#### Engine")
    st.caption(MODEL_NAME)
    st.caption(f"Device selected automatically: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.markdown(
        """
        <div class="sidebar-card">
            <b>Educational AI, real retrieval</b><br>
            No model training. No vector database. One pretrained vision model,
            normalized embeddings, and transparent cosine similarity.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">✦ Enterprise Visual Intelligence Platform</div>
        <h1>See it. Search it.<br><span class="gradient-word">Find your visual match.</span></h1>
        <p>
            Upload any product image and explore the closest matches from a curated
            catalog of 15,000 real fashion products - powered by pretrained visual
            embeddings and cosine similarity.
        </p>
        <div class="hero-footer">
            <span class="hero-chip">MobileNetV3-Small</span>
            <span class="hero-chip">576-D embeddings</span>
            <span class="hero-chip">Cosine similarity</span>
            <span class="hero-chip">Prepared by Mohamed Hassan</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-value">{len(metadata):,}</div><div class="stat-label">Catalog products</div></div>
        <div class="stat-card"><div class="stat-value">{dataset_embeddings.shape[1]}</div><div class="stat-label">Embedding dimensions</div></div>
        <div class="stat-card"><div class="stat-value">{metadata['category'].nunique()}</div><div class="stat-label">Search categories</div></div>
        <div class="stat-card"><div class="stat-value">Top-{top_k}</div><div class="stat-label">Live ranked results</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for query image and metadata
if "query_image" not in st.session_state:
    st.session_state.query_image = None
if "query_name" not in st.session_state:
    st.session_state.query_name = None
if "query_source" not in st.session_state:
    st.session_state.query_source = None

# ----------------- Task 1: Sample Query Gallery -----------------
st.markdown(
    """
    <div class="quick-start-box">
        <div class="quick-start-title">⚡ Instant Quick Start — Try a Curated Sample</div>
        <div class="quick-start-subtitle">
            Click any item below to immediately test visual search across 15,000 catalog products with zero upload required:
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

sample_cols = st.columns(8)
for s_idx, sample_item in enumerate(SAMPLE_QUERIES):
    with sample_cols[s_idx]:
        sample_img_file = PROJECT_DIR / sample_item["path"]
        if sample_img_file.exists():
            st.image(make_square_thumbnail(str(sample_img_file), size=300), width="stretch")
            if st.button(
                sample_item["title"],
                key=f"sample_btn_{sample_item['id']}",
                use_container_width=True,
                help=sample_item["name"],
            ):
                st.session_state.query_image = Image.open(sample_img_file).convert("RGB")
                st.session_state.query_name = sample_item["name"]
                st.session_state.query_source = f"Sample: {sample_item['category']}"
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Custom Query Search</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Or provide your own product image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Upload a file, paste an online image link, or capture a photo directly with your camera:</div>',
    unsafe_allow_html=True,
)

# ----------------- Task 2: Multi-Input Search -----------------
tab_upload, tab_url, tab_camera = st.tabs(
    ["📁 Upload Image", "🔗 Image URL", "📷 Live Camera"]
)

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drop or browse a product image",
        type=["jpg", "jpeg", "png", "webp"],
        key="file_uploader_widget",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        try:
            uploaded_img = Image.open(uploaded_file).convert("RGB")
            if st.session_state.get("query_name") != uploaded_file.name:
                st.session_state.query_image = uploaded_img
                st.session_state.query_name = uploaded_file.name
                st.session_state.query_source = "Uploaded File"
                st.rerun()
        except Exception as err:
            st.error(f"Could not open image file: {err}")

with tab_url:
    st.caption("Paste any public direct image link (JPG, PNG, or WEBP) from an online store or photo catalog:")
    url_input_col, url_action_col = st.columns([3.3, 1])
    with url_input_col:
        image_url_input = st.text_input(
            "Image URL",
            placeholder="https://example.com/shoe-photo.jpg",
            label_visibility="collapsed",
            key="url_input_field",
        )
    with url_action_col:
        fetch_clicked = st.button("Fetch & Search", key="btn_fetch_url", use_container_width=True)

    if fetch_clicked:
        if not image_url_input or not image_url_input.strip():
            st.warning("Please enter an image URL first.")
        else:
            with st.spinner("Downloading image from web..."):
                try:
                    fetched_img = download_image_from_url(image_url_input.strip())
                    clean_name = image_url_input.strip().split("?")[0].rstrip("/").split("/")[-1] or "Web Image"
                    st.session_state.query_image = fetched_img
                    st.session_state.query_name = clean_name[:40]
                    st.session_state.query_source = "Web Image URL"
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed to load image from URL: {err}")

with tab_camera:
    st.caption("Hold up an item, shoe, or accessory and snap a photo:")
    camera_photo = st.camera_input("Take photo", key="camera_input_widget", label_visibility="collapsed")
    if camera_photo is not None:
        try:
            cam_img = Image.open(camera_photo).convert("RGB")
            if st.session_state.get("query_name") != "Live Camera Snapshot":
                st.session_state.query_image = cam_img
                st.session_state.query_name = "Live Camera Snapshot"
                st.session_state.query_source = "Live Camera"
                st.rerun()
        except Exception as err:
            st.error(f"Failed to process camera snapshot: {err}")

query_image = st.session_state.get("query_image")
query_name = st.session_state.get("query_name", "Product Query")
query_source = st.session_state.get("query_source", "Custom Input")

if query_image is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">⌁</div>
            <div style="font-size:1.15rem;font-weight:850;color:#0f172a;">Your visual search starts here</div>
            <div style="color:#64748b;font-size:.86rem;margin-top:.35rem;">
                Select any sample product above, upload an image, paste an online link, or use your webcam to reveal the closest catalog matches.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    query_column, pipeline_column = st.columns([0.9, 1.6], gap="large")

    with query_column:
        with st.container(border=True):
            st.markdown(f'<div class="source-badge">⚡ {escape(query_source)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="query-label">ACTIVE QUERY</div>', unsafe_allow_html=True)
            st.image(ImageOps.contain(query_image, (720, 720)), width="stretch")
            st.caption(f"{query_name}  •  {query_image.width} × {query_image.height} px")
            if st.button("✕ Clear Search", key="btn_clear_query", use_container_width=True):
                st.session_state.query_image = None
                st.session_state.query_name = None
                st.session_state.query_source = None
                st.rerun()

    with pipeline_column:
        st.markdown('<div class="section-kicker">Live AI pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">From pixels to product discovery</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="pipeline">
                <div class="pipeline-step"><div class="pipeline-number">01</div><b>Prepare</b><span>Resize, crop, and normalize the uploaded pixels.</span></div>
                <div class="pipeline-step"><div class="pipeline-number">02</div><b>Embed</b><span>Create one normalized 576-D feature vector.</span></div>
                <div class="pipeline-step"><div class="pipeline-number">03</div><b>Compare</b><span>Measure cosine similarity against 15,000 vectors.</span></div>
                <div class="pipeline-step"><div class="pipeline-number">04</div><b>Rank</b><span>Return the strongest visual matches first.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    model, preprocessing, device = get_model()
    search_started = perf_counter()

    if len(candidate_indices) == 0:
        st.warning(
            "⚠️ No catalog products match your active hybrid filters. "
            "Please broaden your filter selections in the sidebar."
        )
        st.stop()

    with st.spinner("Reading visual features and ranking the catalog..."):
        query_embedding = create_embedding(query_image, model, preprocessing, device)
        active_candidates = candidate_indices if is_filtered else None
        top_indices, top_scores = find_similar_images(
            query_embedding,
            dataset_embeddings,
            top_k,
            candidate_indices=active_candidates,
        )
    search_seconds = perf_counter() - search_started

    effective_top_k = len(top_indices)
    if effective_top_k == 0:
        st.info("No visual matches found for this candidate subset.")
        st.stop()

    result_rows = metadata.iloc[top_indices]
    dominant_category = result_rows["category"].mode().iloc[0]
    best_score = float(top_scores[0])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Curated for your query</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Your closest visual matches</div>', unsafe_allow_html=True)
    searched_scope = f"{len(candidate_indices):,} filtered candidates" if is_filtered else f"{len(metadata):,} saved catalog items"
    st.markdown(
        f'<div class="section-copy">Searched {searched_scope} in {search_seconds:.2f} seconds. Scores show vector similarity, not probability.</div>',
        unsafe_allow_html=True,
    )

    summary_columns = st.columns(4)
    summary_columns[0].metric("Best cosine match", f"{best_score:.1%}")
    summary_columns[1].metric("Dominant category", dominant_category)
    summary_columns[2].metric("Results returned", effective_top_k)
    summary_columns[3].metric("Search time", f"{search_seconds:.2f}s")

    for row_start in range(0, effective_top_k, 5):
        row_indices = top_indices[row_start : row_start + 5]
        row_scores = top_scores[row_start : row_start + 5]
        result_columns = st.columns(len(row_indices), gap="small")

        for position, (column, result_index, score) in enumerate(
            zip(result_columns, row_indices, row_scores),
            start=row_start + 1,
        ):
            show_result_card(column, position, metadata.iloc[result_index], score)

    # ----------------- Task 6: Side-by-Side Visual Comparison -----------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 Side-by-Side Visual & Attribute Comparison Tool", expanded=True):
        st.markdown('<div class="section-title" style="font-size:1.25rem;">Compare Query vs Catalog Match</div>', unsafe_allow_html=True)
        st.caption("Inspect textures, cuts, and attributes of any retrieved item side-by-side with your active query image.")

        match_labels = [
            f"Match #{r:02d} — {text_value(metadata.iloc[idx], 'product_name')} ({sc:.1%})"
            for r, (idx, sc) in enumerate(zip(top_indices, top_scores), start=1)
        ]
        chosen_label = st.selectbox("Select match to compare:", match_labels, index=0)
        chosen_rank = match_labels.index(chosen_label)
        chosen_idx = top_indices[chosen_rank]
        chosen_score = top_scores[chosen_rank]
        chosen_item = metadata.iloc[chosen_idx]
        chosen_price = get_simulated_price(str(chosen_item["image_id"]))

        cmp_left, cmp_mid, cmp_right = st.columns([1, 0.75, 1], gap="medium")
        with cmp_left:
            st.markdown("**Your Active Query**")
            st.image(ImageOps.contain(query_image, (420, 420)), width="stretch")
            st.caption(f"{query_name} ({query_source})")
            st.write(f"Resolution: `{query_image.width} × {query_image.height}` px")

        with cmp_mid:
            st.markdown("<div style='text-align:center;padding-top:1.2rem;'>", unsafe_allow_html=True)
            st.metric("Cosine Match", f"{chosen_score:.1%}")
            st.progress(max(0.0, min(1.0, float(chosen_score))))
            st.markdown("<b>Attributes:</b>", unsafe_allow_html=True)
            st.markdown(f"Category: <code>{text_value(chosen_item, 'category')}</code>")
            st.markdown(f"Article: <code>{text_value(chosen_item, 'article_type')}</code>")
            st.markdown(f"Color: <code>{text_value(chosen_item, 'color')}</code>")
            st.markdown(f"Gender: <code>{text_value(chosen_item, 'gender')}</code>")
            st.markdown(f"Usage: <code>{text_value(chosen_item, 'usage')}</code>")
            st.markdown("</div>", unsafe_allow_html=True)

        with cmp_right:
            st.markdown(f"**Match #{chosen_rank + 1:02d}: {text_value(chosen_item, 'product_name')}**")
            cmp_img_path = PROJECT_DIR / Path(chosen_item["image_path"])
            if cmp_img_path.exists():
                st.image(make_square_thumbnail(str(cmp_img_path), size=420), width="stretch")
            st.caption(f"Product #{chosen_item['image_id']} · ${chosen_price:.2f}")
            if st.button("🛒 Add to Bag from Comparison", key=f"cmp_add_{chosen_item['image_id']}", use_container_width=True):
                add_to_cart(str(chosen_item["image_id"]), text_value(chosen_item, "product_name"), chosen_price, str(chosen_item["image_path"]))
                st.toast(f"Added '{text_value(chosen_item, 'product_name')}' to bag!")

    results_table = build_results_table(top_indices, top_scores, metadata)
    download_column, note_column = st.columns([1, 2.2], vertical_alignment="center")
    with download_column:
        st.download_button(
            "Download ranked results",
            data=results_table.to_csv(index=False).encode("utf-8"),
            file_name="visual_search_results.csv",
            mime="text/csv",
            width="stretch",
        )
    with note_column:
        st.caption(
            "Similarity is learned from visual features. A high score can reflect shared "
            "shape, color, texture, or product structure, but it does not guarantee identical meaning."
        )

    if show_details:
        with st.expander("Explore the architecture and technical details", expanded=True):
            architecture_path = PROJECT_DIR / "docs" / "assets" / "visual_search_architecture.png"
            if architecture_path.exists():
                st.image(str(architecture_path), width="stretch")
            details_left, details_right = st.columns(2)
            with details_left:
                st.markdown("**Query embedding**")
                st.write(f"Shape: `{query_embedding.shape}`")
                st.write(f"L2 norm: `{np.linalg.norm(query_embedding):.6f}`")
                st.write(f"Device: `{device}`")
            with details_right:
                st.markdown("**Catalog matrix**")
                st.write(f"Shape: `{dataset_embeddings.shape}`")
                st.write(f"Data type: `{dataset_embeddings.dtype}`")
                st.write("Search: normalized matrix-vector multiplication")

st.markdown(
    """
    <div class="footer-card">
        <span>LUMINA AI · Visual Product Search</span>
        <span>Educational showcase · Prepared by Mohamed Hassan</span>
    </div>
    """,
    unsafe_allow_html=True,
)
