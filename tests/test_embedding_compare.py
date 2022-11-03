import numpy as np
import pytest
from embsuivi import Embedding, EmbeddingComparison
from gensim.models.keyedvectors import KeyedVectors


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


def test_load_embedding(embedding_A: Embedding):
    emb = EmbeddingComparison._load_embedding(embedding_A)

    # Exact same object since is embedding_A is a Embedding instance
    assert id(emb) == id(embedding_A)

    keyedvectors = KeyedVectors(vector_size=2, count=8)
    keyedvectors.add_vector("a", [0, 1])
    keyedvectors.add_vector("b", [1, 2])

    emb = EmbeddingComparison._load_embedding(keyedvectors)
    assert isinstance(emb, Embedding)
    assert list(emb.key_to_index) == ["a", "b"]

    with pytest.raises(TypeError):
        emb = EmbeddingComparison._load_embedding({"a": [0, 1]})


def test_embedding_comparison(
    comparison_AB: EmbeddingComparison, embedding_A: Embedding, embedding_B: Embedding
):
    emb_1, emb_2 = comparison_AB.embeddings

    # Exact same object since both embedding are Embedding instances
    assert id(emb_1) == id(embedding_A)
    assert id(emb_2) == id(embedding_B)

    emb_1_id, emb_2_id = comparison_AB.embeddings_ids
    assert emb_1_id == "A"
    assert emb_2_id == "B"


def test_get_item(
    comparison_AB: EmbeddingComparison, embedding_A: Embedding, embedding_B: Embedding
):
    assert id(comparison_AB[0]) == id(comparison_AB["A"])
    assert id(comparison_AB["A"]) == id(embedding_A)

    assert id(comparison_AB[1]) == id(comparison_AB["B"])
    assert id(comparison_AB["B"]) == id(embedding_B)

    with pytest.raises(KeyError):
        comparison_AB["does not exists"]


def test_is_frequencies_set(comparison_AB: EmbeddingComparison):
    assert comparison_AB.is_frequencies_set()

    emb1, _ = comparison_AB.embeddings
    emb1.frequencies = (
        np.zeros(emb1.frequencies.shape, dtype=emb1.frequencies.dtype)
        + emb1._default_freq
    )

    assert not comparison_AB.is_frequencies_set()


def test_common_keys(comparison_AB: EmbeddingComparison, embedding_A: Embedding):
    # frequencies are taken in account and most frequent keys come firsts
    assert comparison_AB.common_keys == ["a", "c", "d", "b", "e", "h", "f", "g"]

    embedding_C = Embedding.load_from_dict({"e": [0, 1], "b": [2, 3], "z": [3, 4]})

    comparison_AC = EmbeddingComparison({"A": embedding_A, "C": embedding_C})

    assert comparison_AC.common_keys == ["e", "b"]


def test_neighborhoods_smiliarities(comparison_AB: EmbeddingComparison):
    assert comparison_AB.neighborhoods_smiliarities == {
        "d": 1.0,
        "g": 1.0,
        "a": pytest.approx(1 / 3),
        "c": pytest.approx(1 / 3),
        "e": pytest.approx(1 / 3),
        "f": pytest.approx(1 / 3),
        "b": 0.0,
        "h": 0.0,
    }


def test_mean_neighborhoods_smiliarity(comparison_AB: EmbeddingComparison):
    assert comparison_AB.mean_neighborhoods_smiliarity == pytest.approx((2 + 4 / 3) / 8)
