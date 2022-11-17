import numpy as np
import pytest
from embcompare import Embedding, EmbeddingComparison
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


# Frequencies
#     A   | B   | mean
# a : 0.9 | 0.9 | 0.9
# b : 0.8 | 0.2 | 0.5
# c : 0.7 | 0.8 | 0.75
# d : 0.6 | 0.6 | 0.6
# e : 0.5 | 0.5 | 0.5
# f : 0.4 | 0.4 | 0.4
# g : 0.3 | 0.3 | 0.3
# h : 0.2 | 0.7 | 0.45


@pytest.fixture
def comparison_AB(
    embedding_A: Embedding, embedding_B: Embedding
) -> EmbeddingComparison:
    return EmbeddingComparison({"A": embedding_A, "B": embedding_B}, n_neighbors=2)


def test_load_embedding(embedding_A: Embedding):
    emb = EmbeddingComparison._load_embedding(embedding_A)

    # Exact same object since is embedding_A is a Embedding instance
    assert emb == embedding_A

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
    assert emb_1 == embedding_A
    assert emb_2 == embedding_B

    emb_1_id, emb_2_id = comparison_AB.embeddings_ids
    assert emb_1_id == "A"
    assert emb_2_id == "B"


def test_get_item(
    comparison_AB: EmbeddingComparison, embedding_A: Embedding, embedding_B: Embedding
):
    assert comparison_AB[0] == comparison_AB["A"]
    assert comparison_AB["A"] == embedding_A

    assert comparison_AB[1] == comparison_AB["B"]
    assert comparison_AB["B"] == embedding_B

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

    assert comparison_AC.common_keys == ["b", "e"]


def test_neighborhoods_similarities(comparison_AB: EmbeddingComparison):
    assert comparison_AB.neighborhoods_similarities == {
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


def test_neighborhoods_ordered_similarities(comparison_AB: EmbeddingComparison):
    assert comparison_AB.neighborhoods_ordered_similarities == {
        "d": 1.0,
        "g": 1.0,
        "a": 0.25,
        "c": 0.25,
        "e": 0.25,
        "f": 0.25,
        "b": 0.0,
        "h": 0.0,
    }


def test_mean_neighborhoods_ordered_smiliarity(comparison_AB: EmbeddingComparison):
    assert comparison_AB.mean_neighborhoods_ordered_smiliarity == pytest.approx(3 / 8)


def test_get_most_similar(comparison_AB: EmbeddingComparison):
    assert comparison_AB.get_most_similar(3) == [
        ("d", 1.0),
        ("g", 1.0),
        ("a", pytest.approx(1 / 3)),
    ]

    assert comparison_AB.get_most_similar(3, min_frequency=0.6) == [
        ("d", 1.0),
        ("a", pytest.approx(1 / 3)),
        ("c", pytest.approx(1 / 3)),
    ]


def test_get_least_similar(comparison_AB: EmbeddingComparison):
    assert comparison_AB.get_least_similar(3) == [
        ("h", 0),
        ("b", 0),
        ("f", pytest.approx(1 / 3)),
    ]

    assert comparison_AB.get_least_similar(3, min_frequency=0.6) == [
        ("h", 0),
        ("b", 0),
        ("c", pytest.approx(1 / 3)),
    ]


def test_neighborhoods_similarities_iterator(comparison_AB: EmbeddingComparison):
    similarities = {
        "d": 1.0,
        "g": 1.0,
        "a": 1 / 3,
        "c": 1 / 3,
        "e": 1 / 3,
        "f": 1 / 3,
        "b": 0.0,
        "h": 0.0,
    }

    # most_similar
    for expected_sim_item, sim_item in zip(
        similarities.items(),
        comparison_AB.neighborhoods_similarities_iterator(strategy="most_similar"),
    ):
        assert expected_sim_item == sim_item

    # least similar
    for expected_sim_item, sim_item in zip(
        reversed(similarities.items()),
        comparison_AB.neighborhoods_similarities_iterator(strategy="least_similar"),
    ):
        assert expected_sim_item == sim_item

    # random
    set(similarities.items()) == set(
        list(comparison_AB.neighborhoods_similarities_iterator(strategy="random"))
    )

    # wrong strategy
    with pytest.raises(ValueError):
        next(comparison_AB.neighborhoods_similarities_iterator(strategy="noway"))

    assert ["d", "g", "a"] == [
        k for k, _ in comparison_AB.neighborhoods_similarities_iterator(n_elements=3)
    ]


def test_sampled_comparison(comparison_AB: EmbeddingComparison):
    common_keys = comparison_AB.common_keys

    # When the number of samples wanted is higher or equal to the common keys
    # the comparison object is returned as it is
    sampled_comparison = comparison_AB.sampled_comparison(
        n_samples=len(common_keys), keep_common_only=False
    )
    assert sampled_comparison == comparison_AB

    # Sample 3 first elements
    sampled_comparison = comparison_AB.sampled_comparison(n_samples=3)
    assert sampled_comparison != comparison_AB

    emb1, emb2 = sampled_comparison.embeddings
    assert sampled_comparison.common_keys == common_keys[:3]

    # Sample 3 random elements
    sampled_comparison = comparison_AB.sampled_comparison(
        n_samples=3, strategy="random"
    )

    emb1, emb2 = sampled_comparison.embeddings
    assert len(emb1.key_to_index) == 3
    assert len(emb2.key_to_index) == 3

    with pytest.raises(ValueError):
        comparison_AB.sampled_comparison(strategy="noway")
