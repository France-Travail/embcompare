from pathlib import Path

import pytest
from embsuivi import embeddings_comparator
from gensim.models.keyedvectors import KeyedVectors

TEST_FOLDER = Path(__file__).parent


@pytest.fixture
def test_emb1():
    """Embedding de test local"""
    # g f
    # c   b
    # . a e
    embeddings_test = {
        "a": [1, 0],
        "b": [2, 1],
        "c": [0, 1],
        "d": [0, 0],  # supprimé car nul
        "e": [2, 0],
        "f": [1, 2],
        "g": [0, 2],
    }
    return embeddings_test


@pytest.fixture
def test_emb2():
    """Embedding de test local"""
    # g b
    # c f
    # . a e
    embeddings_test = {
        "a": [1, 0],
        "b": [1, 2],
        "c": [0, 1],
        "d": [0, 0],  # supprimé car nul
        "e": [2, 0],
        "f": [1, 1],
        "g": [0, 2],
    }
    return embeddings_test


def test_levenshtein():
    """Test de la fonction de distance de levensthein"""
    assert embeddings_comparator.levenshtein("", "aaaa") == 1.0
    assert embeddings_comparator.levenshtein("abcd", "azcd") == 0.25
    assert embeddings_comparator.levenshtein("abcd", "acd") == 0.25
    assert embeddings_comparator.levenshtein("dbca", "abcd") == 0.5


def test_properties_embedding_comparator():
    comparator = embeddings_comparator.EmbeddingsComparator(
        ("emb", {"a": [0, 1], "b": [1, 0]}), n_neighbors=1
    )

    assert comparator.embeddings_names == ["emb"]
    assert comparator.n_neighbors == 1
    assert isinstance(comparator["emb"], KeyedVectors)

    comparator.n_neighbors = 2

    assert comparator.n_neighbors == 2
    assert list(comparator._neighborhood.keys()) == [1, 2]
    assert list(comparator._neighborhood_similarities.keys()) == [1, 2]

    assert str(comparator) == "EmbeddingsComparator[emb]"


def test_items_comparator():
    comparator = embeddings_comparator.EmbeddingsComparator()
    comparator["emb"] = {"a": [0, 1], "b": [1, 0]}

    assert comparator.embeddings_names == ["emb"]
    assert isinstance(comparator["emb"], KeyedVectors)

    comparator["emb"] = {"a": [1, 1, 1], "b": [0, 0, 1], "null": [0, 0, 0]}

    # seulement deux vecteurs car il y a un embedding nul qui est supprimé
    # automatiquement
    assert comparator["emb"].vectors.shape == (2, 3)


def test_get_common_keys():
    comparator = embeddings_comparator.EmbeddingsComparator(
        {
            "emb1": {"a": [0, 1], "b": [0, 1], "c": [0, 1]},
            "emb2": {"b": [0, 1], "c": [0, 1], "d": [0, 1]},
            "emb3": {"c": [0, 1], "d": [0, 1], "e": [0, 1]},
        }
    )

    assert comparator.get_common_keys() == {"c"}
    assert comparator.get_common_keys(["emb1", "emb2"]) == {"b", "c"}


def test_add_embedding():
    comparator = embeddings_comparator.EmbeddingsComparator()

    # Chargement d'un embedding à partir d'un fichier
    emb_path = TEST_FOLDER / "embeddings" / "embeddings_competences.json"
    comparator.add_embedding("emb", emb_path)
    list(comparator["emb"].key_to_index) == ["654321", "654322", "654323"]


def test_compute_neighborhood(test_emb1: dict):
    comparator = embeddings_comparator.EmbeddingsComparator(
        ("emb1", test_emb1),
        n_neighbors=2,
    )
    dist, ids = comparator.compute_neighborhood("emb1")

    # 3 colonnes car il y en a une de plus que le nombre de voisins
    assert dist.shape == (6, 3)
    assert ids.shape == (6, 3)


def test_get_neighbors_by_key(test_emb1: dict):
    comparator = embeddings_comparator.EmbeddingsComparator(
        ("emb1", test_emb1),
        n_neighbors=2,
    )
    neighbors = comparator.get_neighbors_by_key("emb1", ["a", "c"])

    ab = cf = pytest.approx(1 - (2 / (5**0.5)))

    assert neighbors == {"a": [("e", 0), ("b", ab)], "c": [("g", 0), ("f", cf)]}


def test_compute_neighborhood_similarity(test_emb1: dict, test_emb2: dict):
    comparator = embeddings_comparator.EmbeddingsComparator(
        ("emb1", test_emb1),
        ("emb2", test_emb2),
        n_neighbors=2,
    )
    sims = comparator.compute_neighborhood_similarity("emb1", "emb2")
    assert sims == {
        "a": pytest.approx(1 / 3),
        "b": 0.0,
        "c": pytest.approx(1 / 3),
        "e": pytest.approx(1 / 3),
        "f": pytest.approx(1 / 3),
        "g": pytest.approx(1 / 3),
    }


def test_compute_neighborhood_similiraties(test_emb1: dict, test_emb2: dict):
    comparator = embeddings_comparator.EmbeddingsComparator(
        ("emb1", test_emb1),
        ("emb2", test_emb2),
        n_neighbors=2,
    )
    sims = comparator.compute_neighborhood_similiraties()
    assert sims[("emb1", "emb2")] == {
        "a": pytest.approx(1 / 3),
        "b": 0.0,
        "c": pytest.approx(1 / 3),
        "e": pytest.approx(1 / 3),
        "f": pytest.approx(1 / 3),
        "g": pytest.approx(1 / 3),
    }


def test_clear_cache_embedding(test_emb1: dict, test_emb2: dict):
    comparator = embeddings_comparator.EmbeddingsComparator(
        ("emb1", test_emb1),
        ("emb2", test_emb2),
        n_neighbors=2,
    )
    sims = comparator.compute_neighborhood_similiraties()
    assert sims[("emb1", "emb2")]

    comparator.add_embedding("emb2", comparator["emb1"])

    assert ("emb1", "emb2") not in sims
    assert ("emb2", "emb1") not in sims
