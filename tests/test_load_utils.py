from pathlib import Path

from embcompare import load_utils
from embcompare.embedding import Embedding


def test_load_frequencies_from_json(frequencies_datadir: Path):
    """Load frequencies from a json file"""
    frequencies = load_utils.load_frequencies_from_json(
        frequencies_datadir / "test_frequencies.json"
    )

    assert len(frequencies) == 2166
    assert min(frequencies.values()) > 0
    assert max(frequencies.values()) < 1


def test_load_frequencies_from_pickle(frequencies_datadir: Path):
    """Load frequencies from a pickle file"""
    frequencies = load_utils.load_frequencies_from_pickle(
        frequencies_datadir / "test_frequencies_altered.pkl"
    )

    assert len(frequencies) == 2166
    assert min(frequencies.values()) > 0
    assert max(frequencies.values()) < 1


def test_load_embedding_from_json(frequencies_datadir: Path, embeddings_datadir: Path):
    """Load embedding from a json file"""
    embedding = load_utils.load_embedding_from_json(
        embeddings_datadir / "embedding_test_1.json"
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (6, 2)

    embedding = load_utils.load_embedding_from_json(
        embeddings_datadir / "embedding_test_1.json",
        frequencies_path=frequencies_datadir / "test_frequencies.json",
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (6, 2)
    assert embedding.is_frequency_set()


def test_load_embedding_from_pickle(
    frequencies_datadir: Path, embeddings_datadir: Path
):
    """Load embedding from a pickle file"""
    embedding = load_utils.load_embedding_from_pickle(
        embeddings_datadir / "embedding_test_1.pkl"
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (6, 2)

    embedding = load_utils.load_embedding_from_pickle(
        embeddings_datadir / "embedding_test_1.pkl",
        frequencies_path=frequencies_datadir / "test_frequencies.json",
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (6, 2)
    assert embedding.is_frequency_set()


def test_load_embedding_from_keyedvectors(
    frequencies_datadir: Path, embeddings_datadir: Path
):
    """Load embedding from a gensim keyedvector file"""
    embedding = load_utils.load_embedding_from_keyedvectors(
        embeddings_datadir / "embedding_test_1.kv"
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (6, 2)

    embedding = load_utils.load_embedding_from_keyedvectors(
        embeddings_datadir / "embedding_test_1.kv",
        frequencies_path=frequencies_datadir / "test_frequencies.json",
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (6, 2)
    assert embedding.is_frequency_set()


def test_load_embedding_from_fasttext(
    frequencies_datadir: Path, embeddings_datadir: Path
):
    """Load embedding from a fasttext binary format file"""
    embedding = load_utils.load_embedding_from_fasttext(
        embeddings_datadir / "fasttext_ex.bin"
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (2166, 4)
    assert not embedding.is_frequency_set()

    embedding = load_utils.load_embedding_from_fasttext(
        embeddings_datadir / "fasttext_ex.bin",
        frequencies_path=frequencies_datadir / "test_frequencies.json",
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (2166, 4)
    assert embedding.is_frequency_set()


def test_load_embedding_from_word2vec(
    frequencies_datadir: Path, embeddings_datadir: Path
):
    """Load embedding from a word2vec binary format file"""
    embedding = load_utils.load_embedding_from_word2vec(
        embeddings_datadir / "word2vec_ex.bin"
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (2166, 4)
    assert not embedding.is_frequency_set()

    embedding = load_utils.load_embedding_from_word2vec(
        embeddings_datadir / "word2vec_ex.bin",
        frequencies_path=frequencies_datadir / "test_frequencies.json",
    )

    assert isinstance(embedding, Embedding)
    assert embedding.vectors.shape == (2166, 4)
    assert embedding.is_frequency_set()
