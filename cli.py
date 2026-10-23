"""Enterprise Command-Line Interface (CLI) for Multi-Modal Visual Search Engine.

Subcommands:
- search: Query catalog by image, text keywords, or composed multi-modal attributes with optional re-ranking.
- index: Audit catalog integrity, embedding dimensions, category distributions, and missing image files.
- benchmark: Run quick retrieval accuracy and latency diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from catalog import CatalogManager
from multimodal_search import (
    ColorHistogramExtractor,
    MultiModalQueryComposer,
    MultiModalReRanker,
)
from search import create_embedding, find_similar_images, load_embedding_model
from visual_explainability import DominantColorExtractor, SpatialPatchComparator


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the visual search CLI."""
    parser = argparse.ArgumentParser(
        prog="visual-search-cli",
        description="Enterprise Multi-Modal Visual Search & Color-Aware Re-Ranking Engine",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. search
    p_search = subparsers.add_parser("search", help="Execute visual, text, or multi-modal hybrid search")
    p_search.add_argument("--image", type=str, help="Path to query image file")
    p_search.add_argument("--text", type=str, help="Text query or attribute modifier (e.g. 'red sneakers')")
    p_search.add_argument("--category", type=str, help="Filter by category (e.g. 'Shoes', 'Topwear')")
    p_search.add_argument("--color", type=str, help="Filter by color (e.g. 'Red', 'Blue')")
    p_search.add_argument("--gender", type=str, help="Filter by gender (e.g. 'Men', 'Women')")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    p_search.add_argument("--rerank", action="store_true", help="Apply color histogram and attribute re-ranking")
    p_search.add_argument("--explain", action="store_true", help="Compute spatial patch similarity heatmap")
    p_search.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 2. index
    p_index = subparsers.add_parser("index", help="Audit catalog metadata and embedding index health")
    p_index.add_argument("--check-images", action="store_true", help="Check on-disk image file presence")
    p_index.add_argument("--json", action="store_true", help="Output health audit in JSON format")

    # 3. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run automated retrieval latency and accuracy benchmark")
    p_bench.add_argument("--num-queries", type=int, default=5, help="Number of benchmark queries to run")
    p_bench.add_argument("--json", action="store_true", help="Output benchmark report in JSON format")

    return parser


def handle_search(args: argparse.Namespace) -> int:
    """Execute search subcommand."""
    if not args.image and not args.text:
        print("Error: At least one of --image or --text must be specified.", file=sys.stderr)
        return 1

    catalog = CatalogManager()
    metadata_df = catalog.metadata_df
    dataset_embeddings = catalog.embeddings_matrix

    t0 = time.time()
    query_image = None
    query_embedding = None

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            return 1
        try:
            query_image = Image.open(args.image).convert("RGB")
            model, preprocessing = load_embedding_model()
            query_embedding = create_embedding(query_image, model, preprocessing)
        except Exception as e:
            print(f"Error extracting image embedding: {e}", file=sys.stderr)
            return 1
    else:
        # Text-only query: synthesize average embedding of category or attribute match
        parsed = MultiModalQueryComposer.parse_text_query(args.text)
        target_cats = parsed.get("target_categories", [])
        mask = np.zeros(len(metadata_df), dtype=bool)
        if target_cats:
            for cat in target_cats:
                mask |= metadata_df["category"].str.lower() == cat.lower()
        if not np.any(mask):
            # General fallback: use first embedding or unit vector
            query_embedding = dataset_embeddings[0]
        else:
            query_embedding = np.mean(dataset_embeddings[mask], axis=0)
            norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / (norm or 1.0)

    # Initial candidate retrieval (retrieve top 3 * top_k for re-ranking)
    fetch_k = min(len(dataset_embeddings), max(args.top_k * 3, 20))
    candidate_indices, candidate_scores = find_similar_images(query_embedding, dataset_embeddings, top_k=fetch_k)

    filters = {}
    if args.category:
        filters["category"] = args.category
    if args.color:
        filters["color"] = args.color
    if args.gender:
        filters["gender"] = args.gender

    # Re-ranking
    if args.rerank or query_image is not None or args.text:
        reranker = MultiModalReRanker()
        cand_images = {idx: catalog.get_image(idx) for idx in candidate_indices[:fetch_k]}
        final_indices, final_scores, breakdowns = reranker.rerank(
            candidate_indices=candidate_indices,
            initial_scores=candidate_scores,
            metadata_df=metadata_df,
            query_image=query_image,
            query_text=args.text,
            candidate_images=cand_images,
            filters=filters,
            top_k=args.top_k,
        )
    else:
        final_indices = candidate_indices[:args.top_k]
        final_scores = candidate_scores[:args.top_k]
        breakdowns = [{"index": int(i), "final_score": float(s)} for i, s in zip(final_indices, final_scores)]

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    results_list = []
    for rank, (idx, score) in enumerate(zip(final_indices, final_scores), 1):
        row = metadata_df.iloc[idx].to_dict()
        item_data = {
            "rank": rank,
            "index": int(idx),
            "score": round(float(score), 4),
            "product_name": row.get("product_name", f"Product #{idx}"),
            "category": row.get("category", "Unknown"),
            "color": row.get("color", "Unknown"),
            "gender": row.get("gender", "Unknown"),
            "article_type": row.get("article_type", "Unknown"),
        }

        if args.explain and query_image is not None:
            cand_img = catalog.get_image(idx)
            comparator = SpatialPatchComparator(grid_size=4)
            patch_info = comparator.compute_patch_similarity_heatmap(query_image, cand_img)
            item_data["explainability"] = {
                "mean_patch_similarity": patch_info["mean_patch_similarity"],
                "top_regions": [r["region"] for r in patch_info["top_aligned_regions"]],
            }

        results_list.append(item_data)

    output_payload = {
        "query": {
            "image": args.image,
            "text": args.text,
            "filters": filters,
            "top_k": args.top_k,
            "rerank_applied": args.rerank,
        },
        "latency_ms": elapsed_ms,
        "total_results": len(results_list),
        "results": results_list,
    }

    if args.json:
        print(json.dumps(output_payload, indent=2))
    else:
        print("\n" + "=" * 65)
        print("VISUAL SEARCH RESULTS")
        print("=" * 65)
        print(f"Query Image: {args.image or 'None'} | Text: {args.text or 'None'}")
        print(f"Latency:     {elapsed_ms} ms | Results: {len(results_list)}")
        print("-" * 65)
        for item in results_list:
            print(f"#{item['rank']:<2} [{item['score']*100:.1f}%] {item['product_name']}")
            print(f"    Category: {item['category']} | Color: {item['color']} | Gender: {item['gender']}")
            if "explainability" in item:
                exp = item["explainability"]
                print(f"    Patch Alignment: {exp['mean_patch_similarity']*100:.1f}% (Top: {', '.join(exp['top_regions'][:2])})")
        print("=" * 65)

    return 0


def handle_index(args: argparse.Namespace) -> int:
    """Execute index subcommand."""
    t0 = time.time()
    catalog = CatalogManager()
    metadata = catalog.metadata_df
    embeds = catalog.embeddings_matrix
    elapsed_ms = round((time.time() - t0) * 1000, 2)

    cat_counts = metadata["category"].value_counts().to_dict() if "category" in metadata else {}
    color_counts = metadata["color"].value_counts().head(5).to_dict() if "color" in metadata else {}

    missing_images = 0
    if args.check_images:
        for idx in range(min(500, len(metadata))):
            row = metadata.iloc[idx].to_dict()
            if "image_path" in row:
                p = PROJECT_DIR / Path(str(row["image_path"]))
                if not p.exists():
                    missing_images += 1

    payload = {
        "is_synthetic_fallback": catalog.is_synthetic,
        "total_catalog_items": len(metadata),
        "embedding_dimensions": int(embeds.shape[1]),
        "top_categories": cat_counts,
        "top_colors": color_counts,
        "missing_images_sampled": missing_images,
        "audit_latency_ms": elapsed_ms,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 65)
        print("CATALOG & EMBEDDING INDEX AUDIT REPORT")
        print("=" * 65)
        print(f"Catalog Source:      {'Micro-Catalog Fallback' if catalog.is_synthetic else 'Full Dataset'}")
        print(f"Total Products:      {len(metadata)}")
        print(f"Embedding Dims:      {embeds.shape[1]}")
        print(f"Distinct Categories: {len(cat_counts)}")
        print("\nTop Categories:")
        for cat, cnt in list(cat_counts.items())[:5]:
            print(f"  - {cat:<15}: {cnt}")
        print("\nTop Colors:")
        for col, cnt in color_counts.items():
            print(f"  - {col:<15}: {cnt}")
        if args.check_images:
            print(f"\nMissing Images:      {missing_images} / {min(500, len(metadata))} sampled")
        print(f"Audit Latency:       {elapsed_ms} ms")
        print("=" * 65)

    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark subcommand."""
    catalog = CatalogManager()
    metadata = catalog.metadata_df
    embeds = catalog.embeddings_matrix

    num_q = min(args.num_queries, len(metadata))
    latencies = []
    top1_correct = 0
    topk_correct = 0
    k = 5

    # Run self-retrieval benchmark (a product vector should retrieve itself as rank 1)
    for i in range(num_q):
        q_vec = embeds[i]
        t0 = time.time()
        indices, scores = find_similar_images(q_vec, embeds, top_k=k)
        latencies.append((time.time() - t0) * 1000)

        if len(indices) > 0 and indices[0] == i:
            top1_correct += 1
        if i in indices[:k]:
            topk_correct += 1

    avg_latency = round(float(np.mean(latencies)), 2)
    top1_acc = round((top1_correct / num_q) * 100.0, 1)
    topk_acc = round((topk_correct / num_q) * 100.0, 1)

    payload = {
        "num_queries": num_q,
        "top_k": k,
        "top_1_accuracy_pct": top1_acc,
        "top_k_accuracy_pct": topk_acc,
        "mean_latency_ms": avg_latency,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 65)
        print("RETRIEVAL ENGINE BENCHMARK")
        print("=" * 65)
        print(f"Evaluated Queries:   {num_q}")
        print(f"Top-1 Accuracy:      {top1_acc}%")
        print(f"Top-{k} Accuracy:      {topk_acc}%")
        print(f"Mean Latency:        {avg_latency} ms")
        print("=" * 65)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "search":
        return handle_search(args)
    elif args.subcommand == "index":
        return handle_index(args)
    elif args.subcommand == "benchmark":
        return handle_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
