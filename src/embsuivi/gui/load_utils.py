import json
import pickle
from functools import wraps

from embsuivi import Embedding
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors

EMBEDDING_FORMATS = {}
FREQUENCIES_FORMATS = {}


def embedding_loader(*formats: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for format in formats:
            EMBEDDING_FORMATS[format] = func

        return wrapper

    return decorator


def frequencies_loader(*formats: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for format in formats:
            FREQUENCIES_FORMATS[format] = func

        return wrapper

    return decorator


@frequencies_loader("json")
def load_frequencies_from_json(frequencies_path: str):
    with open(frequencies_path, "r") as f:
        return json.load(f)


@frequencies_loader("pickle", "pkl")
def load_frequencies_from_pickle(frequencies_path: str, **kwargs):
    with open(frequencies_path, "rb") as f:
        return pickle.load(f, **kwargs)


@embedding_loader("json")
def load_embedding_from_json(
    embedding_path: str,
    frequencies_path: str = None,
    frequencies_format: str = "json",
):
    with open(embedding_path, "r") as f:
        embedding_dict = json.load(f)

    if frequencies_path is not None:
        frequencies = FREQUENCIES_FORMATS[frequencies_format](frequencies_path)
    else:
        frequencies = None

    return Embedding.load_from_dict(embedding_dict, frequencies=frequencies)


@embedding_loader("pickle", "pkl")
def load_embedding_from_pickle(
    embedding_path: str,
    frequencies_path: str = None,
    frequencies_format: str = "json",
    **kwargs
):
    with open(embedding_path, "rb") as f:
        embedding: Embedding = pickle.load(f, **kwargs)

    if frequencies_path is not None:
        frequencies = FREQUENCIES_FORMATS[frequencies_format](frequencies_path)
        embedding.set_frequencies(frequencies)

    return embedding


@embedding_loader("keyedvectors", "kv")
def load_embedding_from_keyedvectors(
    embedding_path: str,
    frequencies_path: str = None,
    frequencies_format: str = "json",
    **kwargs
):
    keyedvectors = KeyedVectors.load(embedding_path, **kwargs)

    if frequencies_path is not None:
        frequencies = FREQUENCIES_FORMATS[frequencies_format](frequencies_path)
    else:
        frequencies = None

    return Embedding.load_from_keyedvectors(keyedvectors, frequencies=frequencies)


@embedding_loader("fasttext")
def load_embedding_from_fasttext(
    embedding_path: str,
    frequencies_path: str = None,
    frequencies_format: str = "json",
    **kwargs
):
    keyedvectors = load_facebook_vectors(embedding_path, **kwargs)

    if frequencies_path is not None:
        frequencies = FREQUENCIES_FORMATS[frequencies_format](frequencies_path)
    else:
        frequencies = None

    return Embedding.load_from_keyedvectors(keyedvectors, frequencies=frequencies)


@embedding_loader("word2vec")
def load_embedding_from_word2vec(
    embedding_path: str,
    frequencies_path: str = None,
    frequencies_format: str = "json",
    binary=True,
    **kwargs
):
    keyedvectors = KeyedVectors.load_word2vec_format(
        embedding_path, binary=binary, **kwargs
    )

    if frequencies_path is not None:
        frequencies = FREQUENCIES_FORMATS[frequencies_format](frequencies_path)
    else:
        frequencies = None

    return Embedding.load_from_keyedvectors(keyedvectors, frequencies=frequencies)
