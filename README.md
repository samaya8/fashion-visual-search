# Multi-Modal Visual Search & Color-Aware Re-Ranking Engine

[![CI](https://github.com/samaya8/fashion-visual-search/actions/workflows/ci.yml/badge.svg)](https://github.com/samaya8/fashion-visual-search/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-33%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `fashion-visual-search`  
> **Repository Description**: Enterprise multi-modal visual product search engine featuring deep metric embeddings (MobileNetV3), 3D HSV color histogram re-ranking, composed hybrid image+text queries, explainable spatial patch heatmaps, and automated retrieval benchmarks.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/pytest-33%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](https://coverage.readthedocs.io/)
[![Zero-GPU](https://img.shields.io/badge/zero--gpu-100%25%20CPU%20compatible-success.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

---

## 🌟 Executive Overview

**Visual Search Engine** is a production-grade multi-modal retrieval and discovery platform for e-commerce. It addresses the fundamental limitation of keyword search—describing complex aesthetics, cuts, textures, and palettes—by mapping consumer photos and text attributes into a unified metric embedding space.

### Key Capabilities:
1. **Deep Metric Feature Embeddings**:
   - ImageNet-pretrained **MobileNetV3-Small** with classification head stripped, yielding compact 576-dimensional feature vectors.
   - Unit $L_2$ sphere normalization ($\|\mathbf{v}\|_2 = 1$), reducing cosine distance to ultra-fast matrix-vector dot products ($S = \mathbf{X} \cdot \mathbf{q}$).
2. **Multi-Modal Composed Queries (Image + Text Modifiers)**:
   - **Image-only search**: Reverse image discovery via uploaded photograph or camera capture.
   - **Text-only search**: Semantic attribute projection across catalog metadata tokens.
   - **Composed hybrid search**: Modulates visual features with textual attribute modifiers (e.g., *"this sneaker but in navy blue"* or *"similar floral pattern in a summer dress"*).
3. **Color-Aware Re-Ranking Pipeline (`multimodal_search.py`)**:
   - 3D HSV color distribution extraction ($8 \times 4 \times 4 = 128$ bins) with histogram intersection.
   - Weighted multi-factor score fusion:
     $$S_{\text{final}} = w_{\text{visual}} \cdot S_{\text{cosine}} + w_{\text{color}} \cdot S_{\text{color}} + w_{\text{attr}} \cdot S_{\text{attr}}$$
   - Elevates color consistency by **+8.0%** over pure CNN embeddings.
4. **Explainable Spatial Patch Heatmaps (`visual_explainability.py`)**:
   - $4 \times 4$ spatial cross-patch similarity matrix explaining regional visual matches (e.g. collar vs sole vs silhouette).
   - Dominant color palette extraction (top-5 hex colors with percentage coverage).
5. **Self-Contained Micro-Catalog Fallback (`catalog.py`)**:
   - Built-in curated 20-product micro-catalog with deterministic embeddings and procedural visual patterns, enabling 100% offline edge execution without requiring the 15,000 Kaggle dataset.
6. **Enterprise CLI Tooling (`cli.py`)**:
   - Full command-line interface supporting `search`, `index`, and `benchmark` subcommands with JSON and table outputs.
7. **Dual Streamlit Applications**:
   - `app.py`: Clean Visual Search Studio with interactive facet filters, color re-ranking, and patch explainability.
   - `app2.py`: Extended LUMINA Product Discovery Studio.
8. **Automated Evaluation Harness (`evals/runner.py`)**:
   - Curated benchmarks generating [`evals/BENCHMARK_RESULTS.md`](evals/BENCHMARK_RESULTS.md) with MRR@5, Precision@5, and latency metrics.

---

## 📐 System Architecture

```text
[Query Photograph / File]           [Text Modifier: "blue running shoes"]
           │                                            │
           ▼                                            ▼
[MobileNetV3 Feature Extractor]           [MultiModalQueryComposer]
 (576-dim L2 Unit Vector)                  (Token & Attribute Parser)
           │                                            │
           └─────────────────────┬──────────────────────┘
                                 │
                                 ▼
                 [Stage 1: Vector Space Retrieval]
                  Cosine Dot Product: S = X_catalog · q
                  (Top-50 Candidates Pre-Selected)
                                 │
                                 ▼
                 [Stage 2: Multi-Factor Re-Ranking]
                  ├── 3D HSV Color Histogram Intersection
                  ├── Metadata Attribute Alignment Boost
                  └── Hard Facet Filtering (Category, Gender)
                                 │
                                 ▼
                     [Final Top-K Ranked SKUs]
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         [Product Display Studio]   [Explainable Diagnostics]
         - Confidence score %       - 4x4 Spatial Patch Heatmap
         - Palette hex codes        - Top Matching Regional Tiles
```

---

## 🚀 Quickstart & CLI Usage

The suite provides a command-line interface (`cli.py`) for automated search and catalog inspection:

### 1. Multi-Modal Visual Search
```bash
# Search by text query with re-ranking and JSON output
python cli.py search --text "Blue sports shoes" --top-k 3 --json

# Search by image with color re-ranking and facet filters
python cli.py search --image photo.jpg --category Shoes --gender Men --rerank

# Search by image with spatial patch explainability
python cli.py search --image photo.jpg --explain --top-k 5
```

### 2. Catalog & Index Health Audit
```bash
# Check index dimensions, category distributions, and missing images
python cli.py index --check-images --json
```

### 3. Retrieval Benchmark
```bash
# Run automated retrieval accuracy and latency diagnostics
python cli.py benchmark --num-queries 10 --json
```

---

## 🖥️ Streamlit Applications

### 1. Visual Search Studio (`app.py`)
Consumer-focused discovery application with drag-and-drop image upload, sample picker, facet filters, and explainable patch similarity:
```bash
streamlit run app.py
```

### 2. LUMINA Visual Product Studio (`app2.py`)
Enterprise catalog intelligence dashboard featuring advanced metric distributions, category affinity breakdowns, and side-by-side visual comparisons:
```bash
streamlit run app2.py
```

---

## 📊 Benchmark Verification

Automated evaluation comparing pure CNN baseline against the multi-modal re-ranker is run via `evals/runner.py`:
```bash
python evals/runner.py
```

### Summary of Benchmark Results:
| Retrieval Pipeline | MRR@5 | Precision@5 | Color Consistency@5 | Avg Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (MobileNetV3 Cosine Similarity)** | `0.80` | `80.0%` | `20.0%` | `2.3 ms` |
| **Multi-Modal Re-Ranker (Visual + Color + Attr)** | `0.80` | `80.0%` | `28.0%` | `18.8 ms` |
| **Net Optimization Delta** | `+0.00` | `+0.0%` | **`+8.0%`** | `+16.4 ms` |

*Full audit report available in [`evals/BENCHMARK_RESULTS.md`](evals/BENCHMARK_RESULTS.md).*

---

## 🧪 Testing & Code Coverage

Run the complete test suite with line-by-line coverage analysis:
```bash
pytest --cov=multimodal_search --cov=visual_explainability --cov=catalog --cov=cli --cov=evals.runner --cov=search --cov-report=term-missing tests/
```

```text
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
catalog.py                    74      7    91%   77-78, 109, 119-120, 128-129
cli.py                       196     29    85%   87-89, 100, 112, 114, 133-135...
evals\runner.py               89      7    92%   23, 154, 163, 226-229
multimodal_search.py         132      8    94%   71, 132, 226-227, 230-232, 260
search.py                     42      3    93%   24, 52, 54
visual_explainability.py      67      0   100%
--------------------------------------------------------
TOTAL                        600     54    91%
============================= 33 passed in 5.38s ==============================
```

---

## 📁 Repository Structure

```text
.
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
└── README.md                       # Master documentation
```

---

## 📜 License & Citation

Released under the **MIT License**.
If you use this system in research or production deployments, please cite:
```bibtex
@software{visual_search_engine_2026,
  author = {AI & Machine Learning Engineering Team},
  title = {Multi-Modal Visual Search Engine: Deep Metric Embeddings, 3D Color Re-Ranking, and Explainability},
  year = {2026}
}
```
