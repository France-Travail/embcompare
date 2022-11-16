import json
import pickle
from functools import wraps
from pathlib import Path
from typing import Union

from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors

from embcompare import Embedding

# EMBEDDING_FORMATS is used to associate to a file format like "json"
# or "pickle" a loading function that can load an embedding stored in
# a file in this format
EMBEDDING_FORMATS = {}

# FREQUENCIES_FORMATS is used to associate to a file format like "json"
# or "pickle" a loading function that can load frequencies stored in
# a file in this format
FREQUENCIES_FORMATS = {}


def embedding_loader(*formats: str):
    """Parameterized decorator to register a function as an embedding
    loader for one or several file format

    Args:
        *formats (str): file formats
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add the wrapped function in the EMBEDDING_FORMATS dictionnary
        for format in formats:
            EMBEDDING_FORMATS[format] = func

        return wrapper

    return decorator


def frequencies_loader(*formats: str):
    """Parameterized decorator to register a function as an frequencies
    loader for one or several file format

    Args:
        *formats (str): file formats
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add the wrapped function in the FREQUENCIES_FORMATS dictionnary
        for format in formats:
            FREQUENCIES_FORMATS[format] = func

        return wrapper

    return decorator


@frequencies_loader("json")
def load_frequencies_from_json(frequencies_path: Union[str, Path], **kwargs) -> dict:
    """Loader of frequencies stored in a json file

    Args:
        frequencies_path (Union[str, Path]): file path
        **kwargs: arguments that will be passed to json.load

    Returns:
        dict: A frequencies dict
    """
    with open(frequencies_path, "r") as f:
        return json.load(f, **kwargs)


@frequencies_loader("pickle", "pkl")
def load_frequencies_from_pickle(frequencies_path: Union[str, Path], **kwargs) -> dict:
    """Loader of frequencies stored in a pickle file

    Args:
        frequencies_path (Union[str, Path]): file path
        **kwargs: arguments that will be passed to pickle.load

    Returns:
        dict: A frequencies dict
    """
    with open(frequencies_path, "rb") as f:
        return pickle.load(f, **kwargs)


def load_frequencies(
    frequencies_path: Union[str, Path], format: str = None, **kwargs
) -> dict:
    """Main loading function for frequenciesDefaults to "json"

    Args:
        frequencies_path (Union[str, Path]): file path
        format (str, optional): file format. Defaults to None.
        **kwargs: arguments that will be passed to the relevant loading function

    Returns:
        dict: A frequencies dict
    """
    if format is None:
        format = Path(frequencies_path).suffix[1:]

    return FREQUENCIES_FORMATS[format](frequencies_path, **kwargs)


@embedding_loader("json")
def load_embedding_from_json(
    embedding_path: Union[str, Path],
    frequencies_path: Union[str, Path] = None,
    frequencies_format: str = None,
    **kwargs
) -> Embedding:
    """Loader of embeddings stored in a json file

    Args:
        embedding_path (str): embedding file path
        frequencies_path (str, optional): frequencies file path. Defaults to None.
        frequencies_format (str, optional): frequencies file format. Defaults to None.
        **kwargs: arguments that will be passed to Embedding.load_from_dict

    Returns:
        Embedding: An Embedding object
    """
    with open(embedding_path, "r") as f:
        embedding_dict = json.load(f)

    if frequencies_path is not None:
        frequencies = load_frequencies(frequencies_path, format=frequencies_format)
    else:
        frequencies = None

    return Embedding.load_from_dict(embedding_dict, frequencies=frequencies, **kwargs)


@embedding_loader("pickle", "pkl")
def load_embedding_from_pickle(
    embedding_path: Union[str, Path],
    frequencies_path: Union[str, Path] = None,
    frequencies_format: str = None,
    **kwargs
):
    """Loader of embeddings stored in a pickle file

    Args:
        embedding_path (str): embedding file path
        frequencies_path (str, optional): frequencies file path. Defaults to None.
        frequencies_format (str, optional): frequencies file format. Defaults to None.
        **kwargs: arguments that will be passed to pickle.load

    Returns:
        Embedding: An Embedding object
    """
    with open(embedding_path, "rb") as f:
        embedding: Embedding = pickle.load(f, **kwargs)

    if frequencies_path is not None:
        frequencies = load_frequencies(frequencies_path, format=frequencies_format)
        embedding.set_frequencies(frequencies)

    return embedding


@embedding_loader("keyedvectors", "kv")
def load_embedding_from_keyedvectors(
    embedding_path: Union[str, Path],
    frequencies_path: Union[str, Path] = None,
    frequencies_format: str = None,
    **kwargs
):
    """Loader of embeddings stored in a gensim keyedvector file

    Args:
        embedding_path (str): embedding file path
        frequencies_path (str, optional): frequencies file path. Defaults to None.
        frequencies_format (str, optional): frequencies file format. Defaults to None.
        **kwargs: arguments that will be passed to KeyedVectors.load

    Returns:
        Embedding: An Embedding object
    """
    keyedvectors = KeyedVectors.load(embedding_path.as_posix(), **kwargs)

    if frequencies_path is not None:
        frequencies = load_frequencies(frequencies_path, format=frequencies_format)
    else:
        frequencies = None

    return Embedding.load_from_keyedvectors(keyedvectors, frequencies=frequencies)


@embedding_loader("fasttext", "bin")
def load_embedding_from_fasttext(
    embedding_path: Union[str, Path],
    frequencies_path: Union[str, Path] = None,
    frequencies_format: str = None,
    **kwargs
):
    """Loader of embeddings stored in a fasttext binary file

    Args:
        embedding_path (str): embedding file path
        frequencies_path (str, optional): frequencies file path. Defaults to None.
        frequencies_format (str, optional): frequencies file format. Defaults to None.
        **kwargs: arguments that will be passed to gensim.load_facebook_vectors

    Returns:
        Embedding: An Embedding object
    """
    keyedvectors = load_facebook_vectors(embedding_path, **kwargs)

    if frequencies_path is not None:
        frequencies = load_frequencies(frequencies_path, format=frequencies_format)
    else:
        frequencies = None

    return Embedding.load_from_keyedvectors(keyedvectors, frequencies=frequencies)


@embedding_loader("word2vec")
def load_embedding_from_word2vec(
    embedding_path: Union[str, Path],
    frequencies_path: Union[str, Path] = None,
    frequencies_format: str = None,
    binary=True,
    **kwargs
):
    """Loader of embeddings stored in a word2vec binary file

    Args:
        embedding_path (str): embedding file path
        frequencies_path (str, optional): frequencies file path. Defaults to None.
        frequencies_format (str, optional): frequencies file format. Defaults to None.
        **kwargs: arguments that will be passed to KeyedVectors.load_word2vec_format

    Returns:
        Embedding: An Embedding object
    """
    keyedvectors = KeyedVectors.load_word2vec_format(
        embedding_path, binary=binary, **kwargs
    )

    if frequencies_path is not None:
        frequencies = load_frequencies(frequencies_path, format=frequencies_format)
    else:
        frequencies = None

    return Embedding.load_from_keyedvectors(keyedvectors, frequencies=frequencies)


def load_embedding(
    embedding_path: Union[str, Path],
    embedding_format: str = None,
    frequencies_path: Union[str, Path] = None,
    frequencies_format: str = None,
    **kwargs
) -> Embedding:
    """Main loading function for embeddings

    Args:
        embedding_path (str): embedding file path
        embedding_format (str, optional): embedding file format. Defaults to None.
        frequencies_path (str, optional): frequencies file path. Defaults to None.
        frequencies_format (str, optional): frequencies file format. Defaults to None.
        **kwargs: arguments that will be passed to relevant loading function

    Returns:
        Embedding: An Embedding object
    """
    if embedding_format is None:
        embedding_format = Path(embedding_path).suffix[1:]

    return EMBEDDING_FORMATS[embedding_format](
        embedding_path=embedding_path,
        frequencies_path=frequencies_path,
        frequencies_format=frequencies_format,
        **kwargs,
    )
