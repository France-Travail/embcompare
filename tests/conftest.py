import pytest


@pytest.fixture
def embeddings_datadir(shared_datadir):
    return shared_datadir / "embeddings"


@pytest.fixture
def test_emb1():
    """Embedding de test local"""
    # g f
    # c   b
    # . a e
    return {
        "a": [1, 0],
        "b": [2, 1],
        "c": [0, 1],
        "d": [0, 0],  # null embedding
        "e": [2, 0],
        "f": [1, 2],
        "g": [0, 2],
    }


@pytest.fixture
def test_emb1_freqs():
    """Embedding de test local"""
    # g f
    # c   b
    # . a e
    return {
        "a": 0.9,
        "b": 0.8,
        "c": 0.7,
        "d": 0.0,  # null embedding
        "e": 0.5,
        "f": 0.4,
        "g": 0.3,
    }


@pytest.fixture
def test_emb2():
    """Embedding de test local"""
    # g b
    # c f
    # . a e
    return {
        "a": [1, 0],
        "b": [1, 2],
        "c": [0, 1],
        "d": [0, 0],  # null embedding
        "e": [2, 0],
        "f": [1, 1],
        "g": [0, 2],
    }
