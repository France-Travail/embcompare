import json
from functools import wraps

from embsuivi import Embedding
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors

EMBEDDING_FORMATS = {}
FREQUENCIES_FORMATS = {}


def embedding_loader(format: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        EMBEDDING_FORMATS[format] = func

        return wrapper

    return decorator


def frequencies_loader(format: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        FREQUENCIES_FORMATS[format] = func

        return wrapper

    return decorator


@frequencies_loader("json")
def load_frequencies_from_json(frequencies_path: str):
    with open(frequencies_path, "r") as f:
        return json.load(f)


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


@embedding_loader("keyedvectors")
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
    **kwargs
):
    keyedvectors = KeyedVectors.load_word2vec_format(embedding_path, **kwargs)

    if frequencies_path is not None:
        frequencies = FREQUENCIES_FORMATS[frequencies_format](frequencies_path)
    else:
        frequencies = None

    return Embedding.load_from_keyedvectors(keyedvectors, frequencies=frequencies)
