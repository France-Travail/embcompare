import numpy as np
import pytest
from embsuivi import Embedding
from gensim.models.keyedvectors import KeyedVectors


def test_embedding():
    embedding = Embedding(vector_size=2, count=3, dtype=np.float64)

    assert np.sum(embedding.vectors) == 0
    assert np.sum(embedding.frequencies) == 0

    keys = ["a", "b", "c"]
    vectors = [[1, 2], [2, 4], [3, 6]]
    freqs = [0.1, 0.3, 0.6]

    for k, v, f in zip(keys, vectors, freqs):
        embedding.add_vector(k, v, f)

    assert np.all(embedding.index_to_key == keys)
    assert np.all(embedding.vectors == vectors)
    assert np.all(embedding.frequencies == freqs)

    embedding = Embedding(vector_size=2, count=3, dtype=np.float64, default_freq=0.123)
    assert (
        np.sum(embedding.frequencies) == 0.369
    )  # count * default_freq = 3 * 0.123 = 0.369


def test_load_from_dict(test_emb1):

    with pytest.raises(ValueError):
        embedding = Embedding.load_from_dict({})

    # Test default beahavior
    embedding = Embedding.load_from_dict(test_emb1)
    test_emb1_vectors = np.array(
        [
            [1, 0],  # a
            [2, 1],  # b
            [0, 1],  # c
            # d is missing because its embedding is null
            [2, 0],  # e
            [1, 2],  # f
            [0, 2],  # g
        ],
        dtype=np.float32,
    )
    assert embedding.index_to_key == ["a", "b", "c", "e", "f", "g"]
    assert np.all(embedding.vectors == test_emb1_vectors)

    # Test with remove_null_vector=False
    embedding = Embedding.load_from_dict(test_emb1, remove_null_vectors=False)
    test_emb1_vectors = np.array(
        [
            [1, 0],  # a
            [2, 1],  # b
            [0, 1],  # c
            [0, 0],  # d is missing here because remove_null_vectors is False
            [2, 0],  # e
            [1, 2],  # f
            [0, 2],  # g
        ]
    )

    assert embedding.index_to_key == ["a", "b", "c", "d", "e", "f", "g"]
    assert np.all(embedding.vectors == test_emb1_vectors)

    # Test with frequencies
    frequencies = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6, "g": 0.7}
    embedding = Embedding.load_from_dict(
        test_emb1, frequencies=frequencies, remove_null_vectors=False
    )

    assert embedding.index_to_key == ["g", "f", "e", "d", "c", "b", "a"]
    assert np.all(
        embedding.frequencies
        == np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )
    assert np.all(
        embedding.vectors
        == [[1, 0], [2, 1], [0, 1], [0, 0], [2, 0], [1, 2], [0, 2]][::-1]
    )


def test_load_from_json(embeddings_datadir):
    test_emb1_path = embeddings_datadir / "embedding_test_1.json"
    embedding = Embedding.load_from_json(test_emb1_path)
    test_emb1_vectors = np.array(
        [
            [1, 0],  # a
            [2, 1],  # b
            [0, 1],  # c
            # d is missing because its embedding is null
            [2, 0],  # e
            [1, 2],  # f
            [0, 2],  # g
        ]
    )
    assert embedding.index_to_key == ["a", "b", "c", "e", "f", "g"]
    assert np.all(embedding.vectors == test_emb1_vectors)


def test_load_from_keyedvectors():

    keys = ["a", "b", "c", "d", "e", "f", "g", "h"]
    keyedvectors = KeyedVectors(vector_size=2, count=10)

    for i, k in enumerate(keys):
        keyedvectors.add_vector(k, [i + 1, i + 1])

    embedding = Embedding.load_from_keyedvectors(keyedvectors)

    assert isinstance(embedding, Embedding)
    assert np.all(embedding.index_to_key == keyedvectors.index_to_key)
    assert np.all(embedding.key_to_index == keyedvectors.key_to_index)
    assert np.all(embedding.vectors == keyedvectors.vectors)

    # Test with frequencies as a list
    freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    keyedvectors = KeyedVectors(vector_size=2, count=10)
    for i, k in enumerate(keys):
        keyedvectors.add_vector(k, [i + 1, i + 1])

    embedding = Embedding.load_from_keyedvectors(keyedvectors, frequencies=freqs)
    assert embedding.index_to_key == [
        "h",
        "g",
        "f",
        "e",
        "d",
        "c",
        "b",
        "a",
        None,
        None,
    ]

    # Test with frequencies as a dict
    freqs = {"g": 0.2, "h": 0.1}
    keyedvectors = KeyedVectors(vector_size=2, count=10)
    for i, k in enumerate(keys):
        keyedvectors.add_vector(k, [i + 1, i + 1])

    embedding = Embedding.load_from_keyedvectors(keyedvectors, frequencies=freqs)
    embedding.add_vector("i", [9, 9], 0.9)

    # embedding should be sorted by frequencies from "a" to "h" and "i" should have been
    # added to the end
    assert embedding.index_to_key == [
        "g",
        "h",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "i",
        None,
    ]
    assert np.all(embedding.vectors[0, :] == [7, 7])
    assert np.all(embedding.vectors[8, :] == [9, 9])


def test_add_vector():
    embedding = Embedding(vector_size=2, count=2)
    embedding.add_vector("a", [0, 1], 0.1)
    embedding.add_vector("b", [0, 2])

    assert np.all(embedding.vectors == [[0, 1], [0, 2]])
    assert np.all(embedding.frequencies == np.array([0.1, 0.0], dtype=np.float32))

    embedding.add_vector("c", [0, 0])
    embedding.add_vector("d", [0, 3], 0.99)

    assert np.all(embedding.vectors == [[0, 1], [0, 2], [0, 0], [0, 3]])
    assert np.all(
        embedding.frequencies == np.array([0.1, 0.0, 0.0, 0.99], dtype=np.float32)
    )


def test_add_vectors():
    embedding = Embedding(vector_size=2)

    embedding.add_vectors(["a", "b"], [[0, 1], [0, 2]], [0.1, 0.2])
    assert np.all(embedding.vectors == [[0, 1], [0, 2]])
    assert np.all(embedding.frequencies == np.array([0.1, 0.2], dtype=np.float32))

    # Add the same vectors, without replace the vectors are note updated
    embedding.add_vectors(["a", "b", "c"], [[0, 0], [0, 0], [3, 3]], [0.0, None, 0.3])
    assert np.all(embedding.vectors == [[0, 1], [0, 2], [3, 3]])
    assert np.all(embedding.frequencies == np.array([0.0, 0.2, 0.3], dtype=np.float32))

    # Add the same vectors with replace=True
    # see. https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.add_vectors
    embedding.add_vectors(["a", "b"], [[0, 0], [0, 0]], replace=True)
    assert np.all(embedding.vectors == [[0, 0], [0, 0], [3, 3]])
    assert np.all(embedding.frequencies == np.array([0.0, 0.2, 0.3], dtype=np.float32))


def test_get_frequency():
    embedding = Embedding.load_from_dict(
        {"a": [0, 1], "b": [0, 2]}, frequencies={"a": 0.2, "b": 0.1}
    )

    assert embedding.get_frequency("a") == pytest.approx(0.2)
    assert embedding.get_frequency(1) == pytest.approx(0.1)


def test_set_frequencies():
    embedding = Embedding.load_from_dict({"a": [0, 1], "b": [0, 2], "c": [0, 3]})
    frequencies = [0.1, 0.2, 0.3]

    # Set frequencies with a list
    embedding.set_frequencies(frequencies)
    assert np.all(np.isclose(embedding.frequencies, frequencies))

    # Set frequencies an array
    embedding.set_frequencies(np.array(frequencies))
    assert np.all(np.isclose(embedding.frequencies, frequencies))

    # Set frequencies with a dict
    dict_frequencies = {"c": 0.1, "b": 0.2, "a": 0.3, "z": "not used"}

    embedding.set_frequencies(dict_frequencies)
    assert np.all(np.isclose(embedding.frequencies, frequencies[::-1]))


def test_ordered():
    embedding = Embedding(vector_size=2, count=4)

    keys = ["a", "b", "c"]
    vectors = [[0, 1], [0, 2], [0, 3]]
    freqs = [0.1, 0.2, 0.3]

    for key, vec, freq in zip(keys, vectors, freqs):
        embedding.add_vector(key, vec, freq)

    # We initialized the Embedding with 4 elements
    assert embedding.index_to_key == (keys + [None])
    assert np.all(embedding.vectors == (vectors + [[0.0, 0.0]]))

    # Since ordered method use key_to_index non-used elements are stripped
    ordered_embedding = embedding.ordered()

    assert ordered_embedding.index_to_key == keys[::-1]
    assert np.all(ordered_embedding.vectors == vectors[::-1])


def test_compute_neighborhoods(test_emb1):
    embedding = Embedding.load_from_dict(test_emb1)
    nn_dist, nn_ids = embedding.compute_neighborhoods(n_neighbors=2)

    # cosine_dist = 1 - u . v / ||u|| * ||v||
    # cosine_dist([1, 0], [2, 1]) = 1 - (2 / (1 * sqrt(5)))
    x = pytest.approx(1 - 2 / 5**0.5)

    expected_ids = np.array([[3, 1], [0, 3], [5, 4], [3, 1], [2, 5], [5, 4]])
    expected_dist = np.array([[0.0, x], [x, x], [0.0, x], [0.0, x], [x, x], [0.0, x]])

    assert np.all(nn_ids == expected_ids)
    assert np.all(nn_dist == expected_dist)

    # we modify embedding.__neighborhoods to see if it is reused as it should
    tricked_nn_ids = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 1]])
    embedding._Embedding__neighborhoods = (nn_dist, tricked_nn_ids)

    nn_dist, nn_ids = embedding.compute_neighborhoods(n_neighbors=2)
    assert np.all(nn_ids == tricked_nn_ids)

    nn_dist, nn_ids = embedding.compute_neighborhoods(n_neighbors=1)
    assert np.all(nn_ids == tricked_nn_ids[:, 0:1])

    # For n_neighbors superior to what have already been computed neigborhoods
    # are computed again
    nn_dist, nn_ids = embedding.compute_neighborhoods(n_neighbors=3)
    assert np.all(nn_ids[:, 0:2] == expected_ids)


def test_get_neighbors(test_emb1):
    embedding = Embedding.load_from_dict(test_emb1)
    neighbors = embedding.get_neighbors(n_neighbors=2)

    # consine_similarity = u . v / ||u|| * ||v||
    # consine_similarity([1, 0], [2, 1]) = 2 / (1 * sqrt(5))
    x = pytest.approx(2 / 5**0.5)

    assert neighbors == {
        "a": [("e", 1.0), ("b", x)],
        "b": [("a", x), ("e", x)],
        "c": [("g", 1.0), ("f", x)],
        "e": [("e", 1.0), ("b", x)],
        "f": [("c", x), ("g", x)],
        "g": [("g", 1), ("f", x)],
    }
