# Visual Search Engine — Advanced E-Commerce Edition
> **Domain**: Computer Vision, Deep Metric Learning, Multi-Modal Retrieval & Vector Search  
> **Target System**: Enterprise Multi-Modal Visual Search Engine & E-Commerce Discovery Platform  
> **Repository**: `fashion-visual-search`

---

## 1. Executive Summary & Problem Statement

Keyword-based search in e-commerce encounters severe vocabulary mismatches when shoppers attempt to describe nuanced aesthetic silhouettes, cuts, textures, and color tones.

**Visual Search Engine** implements an enterprise-grade multi-modal retrieval platform. It enables reverse image search, text-to-image semantic retrieval, and composed multi-modal queries (image + text modifiers) across an indexed catalog of **15,000 real e-commerce products**. The platform couples deep metric embeddings (**MobileNetV3**) with 3D HSV color distribution re-ranking, explainable spatial patch heatmaps, and a self-contained micro-catalog fallback for zero-GPU edge execution.

---

## 2. Mathematical Formulations & Retrieval Foundations

### 2.1 Deep Metric Vector Normalization & Cosine Retrieval
Given query image $I_q$ and feature extraction model $f_\theta$:
$$\mathbf{v}_q = f_\theta(I_q) \in \mathbb{R}^{576}, \quad \mathbf{q} = \frac{\mathbf{v}_q}{\|\mathbf{v}_q\|_2}$$
With all catalog embeddings unit-normalized ($\|\mathbf{x}_i\|_2 = 1$), cosine similarity simplifies to matrix-vector dot product:
$$S_{\text{cosine}}(i) = \mathbf{x}_i \cdot \mathbf{q} = \cos(\theta_{\mathbf{x}_i, \mathbf{q}})$$
allowing sub-millisecond evaluation over tens of thousands of items:
$$\mathbf{S} = \mathbf{X}_{\text{catalog}} \mathbf{q} \in \mathbb{R}^N$$

### 2.2 3D HSV Color Histogram Intersection
To prevent deep CNN embeddings from confusing similar garment shapes that differ in color, images are mapped to the HSV color space and binned into $8 \times 4 \times 4 = 128$ quantized bins:
$$H \in [0, 8), \quad S \in [0, 4), \quad V \in [0, 4)$$
Given normalized query histogram $\mathbf{h}_q$ and candidate histogram $\mathbf{h}_i$:
$$S_{\text{color}}(i) = \sum_{b=1}^{128} \min(h_{q}[b], h_{i}[b])$$

### 2.3 Two-Stage Multi-Modal Score Fusion
In the second stage, candidate items are re-scored using weighted linear fusion:
$$S_{\text{final}}(i) = w_{\text{visual}} \cdot S_{\text{cosine}}(i) + w_{\text{color}} \cdot S_{\text{color}}(i) + w_{\text{attr}} \cdot S_{\text{attr}}(i)$$
where default weights are $w_{\text{visual}} = 0.60, w_{\text{color}} = 0.25, w_{\text{attr}} = 0.15$.

### 2.4 Spatial Patch Cross-Attention Heatmaps
Query and target images are partitioned into $G \times G$ spatial tiles (e.g. $4 \times 4 = 16$ patches). For each tile $(r, c)$, normalized mean and variance color features $\mathbf{f}_{r,c}$ are extracted. Positional regional alignment is evaluated as:
$$M(r, c) = \mathbf{f}_{q}(r, c) \cdot \mathbf{f}_{t}(r, c)$$
providing visual explainability regarding which garment regions (collar, sleeve, sole, texture) drove retrieval.

---

## 3. Production Deliverables & Repository Structure

```text
Visual Search Engine - Advanced Data/
├── app.py                          # Streamlit Visual Search Studio
├── app2.py                         # Extended LUMINA Visual Product Studio
├── catalog.py                      # CatalogManager with self-contained fallback
├── cli.py                          # Enterprise CLI (search, index, benchmark)
├── config.py                       # Central system paths and model constants
├── evals/
│   ├── runner.py                   # Automated benchmark evaluation runner
│   └── BENCHMARK_RESULTS.md        # Golden benchmark audit report
├── multimodal_search.py            # 3D HSV color extractor, query composer & re-ranker
├── notebooks/
│   └── 01_visual_search_engine.ipynb # Exploration and embedding notebook
├── prepare_dataset.py              # Kaggle dataset preparation script
├── generate_embeddings.py          # Batch PyTorch vector generation script
├── requirements.txt                # Pinned dependencies
├── search.py                       # Core metric extraction and cosine similarity
├── visual_explainability.py        # Spatial patch heatmaps & dominant color palettes
├── tests/
│   ├── test_catalog.py             # Catalog loading & fallback tests (3 passed)
│   ├── test_cli.py                 # CLI command & exit code tests (11 passed)
│   ├── test_evals.py               # Benchmark runner tests (3 passed)
│   ├── test_multimodal_search.py   # Color extraction & re-ranking tests (7 passed)
│   ├── test_search.py              # Metric ranking tests (4 passed)
│   └── test_visual_explainability.py # Patch heatmap & palette tests (5 passed)
├── LICENSE                         # MIT License
├── PROJECT_DETAILS.md              # Technical specification & mathematical formulations
└── README.md                       # Comprehensive guide and documentation
```

---

## 4. Verification & Quality Assurance

- **Unit Tests**: 33 passed in 5.38s.
- **Core Code Coverage**: 91% across all modules.
- **Offline Edge Resilience**: 100% executable with self-contained micro-catalog when external datasets are absent.
- **Latency Profile**: Sub-25ms end-to-end CPU query response across 15,000 products.
