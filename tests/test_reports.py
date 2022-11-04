import json
from pathlib import Path

import numpy as np
import pytest
from embsuivi import (
    Embedding,
    EmbeddingComparison,
    EmbeddingComparisonReport,
    EmbeddingReport,
)


@pytest.fixture
def embedding_A() -> Embedding:
    return Embedding.load_from_dict(
        embedding_dict={
            "a": [4, 1],
            "b": [5, 2],
            "c": [1, 4],
            "d": [2, 3],
            "e": [5, 3],
            "f": [3, 0],
            "g": [1, 2],
            "h": [-1, 2],
        },
        frequencies={
            "a": 0.9,
            "b": 0.8,
            "c": 0.7,
            "d": 0.6,
            "e": 0.5,
            "f": 0.4,
            "g": 0.3,
            "h": 0.2,
        },
    )


@pytest.fixture
def embedding_B() -> Embedding:
    return Embedding.load_from_dict(
        embedding_dict={
            "a": [4, 1],
            "b": [0, 3],
            "c": [3, 3],
            "d": [2, 3],
            "e": [5, 3],
            "f": [3, 0],
            "g": [1, 2],
            "h": [5, 1],
        },
        frequencies={
            "a": 0.9,
            "b": 0.2,
            "c": 0.8,
            "d": 0.6,
            "e": 0.5,
            "f": 0.4,
            "g": 0.3,
            "h": 0.7,
        },
    )


@pytest.fixture
def comparison_AB(
    embedding_A: Embedding, embedding_B: Embedding
) -> EmbeddingComparison:
    return EmbeddingComparison({"A": embedding_A, "B": embedding_B}, n_neighbors=2)


def test_report_str(comparison_AB: EmbeddingComparisonReport, embedding_A: Embedding):
    report = EmbeddingReport(embedding_A, n_neighbors=2)
    assert str(report).startswith("EmbeddingReport")

    report_comparison = EmbeddingComparisonReport(comparison_AB)
    assert str(report_comparison).startswith("EmbeddingComparisonReport")


def test_report_get_item(embedding_A: Embedding):
    report = EmbeddingReport(embedding_A, n_neighbors=2)

    assert report["n_neighbors"] == 2

    for feature in report.FEATURES:
        assert hasattr(report, feature)

    with pytest.raises(ValueError):
        report[1]

    with pytest.raises(KeyError):
        report["not_an_existing_feature"]


def test_report_to_dict(embedding_A: Embedding):
    report = EmbeddingReport(embedding_A, n_neighbors=2)

    assert report.include_features == list(report.FEATURES)

    assert report.to_dict() == {
        "vector_size": 2,
        "n_elements": 8,
        "default_frequency": pytest.approx(0),
        "mean_frequency": pytest.approx(0.55),
        "n_neighbors": 2,
        "mean_distance_neighbors": pytest.approx(0.065, abs=1e-3),
        "mean_distance_first_neigbor": pytest.approx(0.043, abs=1e-3),
    }

    report.include_features = ["vector_size", "n_neighbors"]

    assert report.to_dict() == {
        "vector_size": 2,
        "n_neighbors": 2,
    }


def test_report_to_json(embedding_A: Embedding, tmp_path: Path):
    report = EmbeddingReport(
        embedding_A,
        n_neighbors=2,
        include_features=["vector_size", "n_neighbors", "mean_frequency"],
    )

    path = tmp_path / "report.json"
    report_dict = report.to_json(path)

    with path.open("r") as f:
        report_dict = json.load(f)

    assert report_dict == {
        "vector_size": 2,
        "n_neighbors": 2,
        "mean_frequency": pytest.approx(0.55),
    }


def test_comparison_report_to_dict(comparison_AB: EmbeddingComparisonReport):
    report = EmbeddingComparisonReport(comparison_AB)

    report_dict = report.to_dict()

    report_embedding_A = report_dict["embeddings"][0]
    assert report_embedding_A["name"] == "A"
    assert report_embedding_A["mean_frequency"] == pytest.approx(0.55)

    report_dict["neighborhoods_similarities_median"] == pytest.approx(1 / 3)
    report_dict["neighborhoods_ordered_similarities_median"] == pytest.approx(0.25)

    # Add a custom feature to compute
    report.include_features += [
        (
            "neighborhoods_similarities_mean",
            lambda report: np.mean(report.neighborhoods_similarities_values),
        )
    ]

    report_dict = report.to_dict()
    assert report_dict["neighborhoods_similarities_mean"] == pytest.approx(
        (2 + 4 / 3) / 8
    )

    # Custom features should be like (feature_name, feature_function)
    report.include_features += [None]
    with pytest.raises(TypeError):
        report.to_dict()
