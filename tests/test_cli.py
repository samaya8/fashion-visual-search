"""Unit tests for the Visual Search Engine CLI.

Tests coverage:
- cli index subcommand (JSON and stdout modes)
- cli search subcommand (text queries, image queries, facet filtering, explainability)
- cli benchmark subcommand
- Error handling and exit codes
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from PIL import Image

import cli


@pytest.fixture
def temp_query_image(tmp_path: Path):
    """Generate a temporary query image."""
    img_path = tmp_path / "test_query.jpg"
    img = Image.new("RGB", (64, 64), color=(220, 38, 38))
    img.save(img_path)
    return img_path


def test_cli_help():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0


def test_cli_no_subcommand():
    code = cli.main([])
    assert code == 0


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    return json.loads(text)


def test_cli_index_json(capsys):
    code = cli.main(["index", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = _extract_json(captured.out)
    assert "total_catalog_items" in payload
    assert "embedding_dimensions" in payload
    assert payload["embedding_dimensions"] == 576


def test_cli_index_stdout(capsys):
    code = cli.main(["index"])
    assert code == 0
    captured = capsys.readouterr()
    assert "CATALOG & EMBEDDING INDEX AUDIT REPORT" in captured.out
    assert "Total Products:" in captured.out


def test_cli_search_text_json(capsys):
    code = cli.main(["search", "--text", "Blue sports shoes", "--top-k", "3", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = _extract_json(captured.out)
    assert payload["total_results"] == 3
    assert len(payload["results"]) == 3
    assert payload["results"][0]["score"] > 0.0


def test_cli_search_text_stdout(capsys):
    code = cli.main(["search", "--text", "Blue running shoes", "--top-k", "2"])
    assert code == 0
    captured = capsys.readouterr()
    assert "VISUAL SEARCH RESULTS" in captured.out
    assert "#1" in captured.out


def test_cli_search_image(temp_query_image, capsys):
    code = cli.main(["search", "--image", str(temp_query_image), "--top-k", "2", "--rerank", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = _extract_json(captured.out)
    assert payload["total_results"] == 2
    assert payload["results"][0]["score"] > 0.0


def test_cli_search_with_filters(capsys):
    code = cli.main(["search", "--text", "shoes", "--gender", "Men", "--top-k", "3", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = _extract_json(captured.out)
    for res in payload["results"]:
        assert res["gender"] == "Men"


def test_cli_search_with_explain(temp_query_image, capsys):
    code = cli.main(["search", "--image", str(temp_query_image), "--explain", "--top-k", "2", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = _extract_json(captured.out)
    assert "explainability" in payload["results"][0]
    assert "mean_patch_similarity" in payload["results"][0]["explainability"]


def test_cli_benchmark(capsys):
    code = cli.main(["benchmark", "--num-queries", "3", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = _extract_json(captured.out)
    assert payload["num_queries"] == 3
    assert payload["top_1_accuracy_pct"] >= 90.0
    assert payload["mean_latency_ms"] >= 0.0


def test_cli_errors():
    # Neither text nor image provided
    code_no_args = cli.main(["search"])
    assert code_no_args == 1

    # Missing image file
    code_missing_img = cli.main(["search", "--image", "non_existent_img_12345.jpg"])
    assert code_missing_img == 1
