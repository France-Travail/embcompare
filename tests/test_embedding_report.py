import json
from pathlib import Path

import pytest
from embsuivi import Embedding, EmbeddingReport


@pytest.fixture
def emb_test(test_emb1: dict, test_emb1_freqs: dict):
    return Embedding.load_from_dict(test_emb1, frequencies=test_emb1_freqs)


def test_str(emb_test: Embedding):
    report = EmbeddingReport(emb_test, n_neighbors=2)
    assert "n_neighbors=2" in str(report)


def test_get_item(emb_test: Embedding):
    report = EmbeddingReport(emb_test, n_neighbors=2)

    for feature in report.FEATURES:
        assert hasattr(report, feature)

    with pytest.raises(ValueError):
        report[1]

    with pytest.raises(KeyError):
        report["not_an_existing_feature"]


def test_to_dict(emb_test: Embedding):
    report = EmbeddingReport(emb_test, n_neighbors=2)

    assert report.include == list(report.FEATURES)

    assert report.to_dict() == {
        "vector_size": 2,
        "n_elements": 6,
        "default_frequency": pytest.approx(0),
        "mean_frequency": pytest.approx(0.6),
        "n_neighbors": 2,
        "mean_distance_neighbors": pytest.approx(0.0704, abs=1e-3),
        "mean_distance_first_neigbor": pytest.approx(0.0352, abs=1e-3),
    }

    report.include = ["vector_size", "n_neighbors"]

    assert report.to_dict() == {
        "vector_size": 2,
        "n_neighbors": 2,
    }


def test_to_json(emb_test: Embedding, tmp_path: Path):
    report = EmbeddingReport(
        emb_test,
        n_neighbors=2,
        include=["vector_size", "n_neighbors", "mean_frequency"],
    )

    path = tmp_path / "report.json"
    report_dict = report.to_json(path)

    with path.open("r") as f:
        report_dict = json.load(f)

    assert report_dict == {
        "vector_size": 2,
        "n_neighbors": 2,
        "mean_frequency": pytest.approx(0.6),
    }
