from functools import wraps

from embsuivi import Embedding
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors


def load_as_embedding(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        keyedvectors = func(*args, **kwargs)
        return Embedding.load_from_keyedvectors(keyedvectors)

    return wrapper


LOAD_FUNCTIONS = {
    "keyedvectors": load_as_embedding(KeyedVectors.load),
    "fasttext": load_as_embedding(load_facebook_vectors),
    "word2vec": load_as_embedding(KeyedVectors.load_word2vec_format),
    "json": Embedding.load_from_json,
}
