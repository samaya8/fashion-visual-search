"""Unit tests for the Visual Search automated benchmark runner.

Tests coverage:
- Benchmark execution across test queries
- Metric validation (MRR, Precision@K, Color Consistency)
- Markdown report file generation and content formatting
- Execution with synthetic catalog manager fixture
"""

from __future__ import annotations

from pathlib import Path
import pytest

from catalog import CatalogManager
from evals.runner import VisualSearchBenchmarkRunner


def test_benchmark_runner_execution():
    runner = VisualSearchBenchmarkRunner()
    results = runner.run_benchmark(k=3)

    assert "individual_metrics" in results
    assert "summary" in results

    summary = results["summary"]
    assert "baseline" in summary
    assert "reranked" in summary

    for mode in ["baseline", "reranked"]:
        m = summary[mode]
        assert 0.0 <= m["mrr_at_k"] <= 1.0
        assert 0.0 <= m["precision_at_k_pct"] <= 100.0
        assert 0.0 <= m["color_consistency_pct"] <= 100.0
        assert m["mean_latency_ms"] >= 0.0


def test_benchmark_runner_synthetic(tmp_path: Path):
    non_existent_meta = tmp_path / "fake_meta.csv"
    non_existent_embed = tmp_path / "fake_embed.npy"
    synth_catalog = CatalogManager(metadata_path=non_existent_meta, embeddings_path=non_existent_embed)

    runner = VisualSearchBenchmarkRunner(catalog=synth_catalog)
    results = runner.run_benchmark(k=2)
    assert results["summary"]["reranked"]["mrr_at_k"] >= 0.0


def test_benchmark_report_generation(tmp_path: Path):
    runner = VisualSearchBenchmarkRunner()
    out_file = tmp_path / "TEST_BENCHMARK_RESULTS.md"
    report = runner.generate_report(output_markdown_path=str(out_file))

    assert "Multi-Modal Visual Search Engine — Enterprise Benchmark Report" in report
    assert "MRR@5" in report
    assert "Color Consistency@5" in report
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "Baseline (MobileNetV3 Cosine Similarity)" in content
    assert "Multi-Modal Re-Ranker" in content
