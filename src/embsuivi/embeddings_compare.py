import collections
from functools import cached_property
from itertools import islice
from typing import Any, Dict, Hashable, Tuple, Union

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from .embedding import Embedding
from .sequences_similarity import damerau_levenshtein_similarity as ordered_sim


class EmbeddingComparison:
    def __init__(self, embeddings: Dict[Any, KeyedVectors], n_neighbors: int = 25):
        """Comparison of two embeddings

        Embeddings must be provided in a dict like :
        {"first_emb_name": first_emb, "second_emb_name": second_emb}

        Args:
            embeddings (Dict[Any, KeyedVectors]): python dict containing both embeddings.
            n_neighbors (int, optional): Number of neighbors for comparison. Defaults to 25.
        """
        self.n_neighbors = n_neighbors

        assert (
            isinstance(embeddings, dict) and len(embeddings) == 2
        ), "embeddings should be a python dict containing two embeddings"

        self.__embeddings_ids = tuple(emb_id for emb_id in embeddings.keys())
        self.__embeddings = tuple(
            self._load_embedding(emb) for emb in embeddings.values()
        )

    @property
    def embeddings(self) -> Tuple[Embedding, Embedding]:
        """Tuple containing both embeddings"""
        return self.__embeddings

    @property
    def embeddings_ids(self) -> Tuple[Hashable, Hashable]:
        """Tuple containing both embedding names"""
        return self.__embeddings_ids

    @staticmethod
    def _load_embedding(embedding: KeyedVectors) -> Embedding:
        """Convert a KeyedVectors to an Embedding object if necessary

        Args:
            embedding (KeyedVectors): embedding as a KeyedVectors object

        Raises:
            TypeError: raise an error if the provided embedding is not a KeyedVectors object

        Returns:
            Embedding: embedding as a Embedding object
        """
        if isinstance(embedding, Embedding):
            return embedding
        elif isinstance(embedding, KeyedVectors):
            return Embedding.load_from_keyedvectors(embedding)
        else:
            raise TypeError("Embeddings should be of type Embedding or KeyedVectors")

    def __getitem__(self, identifier: Union[str, int]) -> Embedding:
        """Allows referencing both embedding by its name or index

        >>> comp = EmbeddingComparison({"A": emb_A, "B": emb_B})
        >>> assert comp["A"] == emb_A
        True

        Args:
            identifier (Union[str, int]): embedding identifier

        Raises:
            KeyError: raise an error if the identifier is neither an embedding
                name nor 0 or 1.

        Returns:
            Embedding: the corresponding embedding
        """
        if identifier == self.embeddings_ids[0]:
            return self.embeddings[0]

        elif identifier == self.embeddings_ids[1]:
            return self.embeddings[1]

        elif identifier in (0, 1):
            return self.embeddings[identifier]

        else:
            ids = {
                str(i) for i in [0, 1, self.embeddings_ids[0], self.embeddings_ids[1]]
            }
            raise KeyError(f"identifier should be one of : {', '.join(ids)}")

    def is_frequencies_set(self) -> bool:
        """Verify if element frequencies are set for both embedding

        The verification process consist in verifing if for both embedding
        there is at least one frequency that is not equal to the default
        frequency

        Returns:
            bool: True if both embeddings contain frequencies, False otherwise
        """
        first_emb, second_emb = self.embeddings

        return np.any(first_emb.frequencies != first_emb._default_freq) and np.any(
            second_emb.frequencies != second_emb._default_freq
        )

    @cached_property
    def common_keys(self) -> list:
        """Returns a list of common elements between the two embeddings

        The elements are sorted according to their mean frequencies in both
        embeddings. If frequencies are not set for both embedding, we take
        the mean position in the embeddings of the elements as their frequency

        Returns:
            list: A list of common elements between the two compared embeddings
        """
        common_keys = {}

        first_emb, second_emb = self.embeddings
        frequencies_are_set = self.is_frequencies_set()

        for key in first_emb.key_to_index:
            if key in second_emb.key_to_index:

                if frequencies_are_set:
                    f1 = first_emb.get_frequency(key)
                    f2 = second_emb.get_frequency(key)
                else:
                    f1 = first_emb.key_to_index[key] / len(first_emb.key_to_index)
                    f2 = second_emb.key_to_index[key] / len(first_emb.key_to_index)

                common_keys[key] = f1 + f2

        return sorted(common_keys, key=common_keys.get, reverse=True)

    @cached_property
    def neighborhoods(self) -> Tuple[dict, dict]:
        """Returns both embedding neighborhoods for common keys in a tuple

        Returns:
            Tuple[dict, dict]: a tuple containing both embedding neighborhoods
        """
        first_emb, second_emb = self.embeddings

        first_neighborhoods = first_emb.get_neighbors(
            n_neighbors=self.n_neighbors,
            keys=self.common_keys,
        )
        second_neighborhoods = second_emb.get_neighbors(
            n_neighbors=self.n_neighbors,
            keys=self.common_keys,
        )

        return (first_neighborhoods, second_neighborhoods)

    @cached_property
    def neighborhoods_smiliarities(self) -> Dict[str, float]:
        """Return similarities between common elements

        Returns:
            Dict[str, float]: a {element: similarity} python dict
        """
        emb1_neighborhoods, emb2_neighborhoods = self.neighborhoods

        similarities = {}

        for key in self.common_keys:
            key_neighbors_1 = {k for k, _ in emb1_neighborhoods[key]}
            key_neighbors_2 = {k for k, _ in emb2_neighborhoods[key]}

            similarities[key] = len(
                key_neighbors_1.intersection(key_neighbors_2)
            ) / len(key_neighbors_1.union(key_neighbors_2))

        # Sort by similarity
        return {
            key: sim
            for key, sim in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True
            )
        }

    @cached_property
    def mean_neighborhoods_smiliarity(self) -> float:
        """Mean neighborhoods similarity"""
        return np.mean(list(self.neighborhoods_smiliarities.values()))

    @cached_property
    def neighborhoods_ordered_smiliarities(self) -> Dict[str, float]:
        emb1_neighborhoods, emb2_neighborhoods = self.neighborhoods

        similarities = {}

        for key in self.common_keys:
            key_neighbors_1 = [k for k, _ in emb1_neighborhoods[key]]
            key_neighbors_2 = [k for k, _ in emb2_neighborhoods[key]]

            similarities[key] = ordered_sim(key_neighbors_1, key_neighbors_2)

        # Sort by similarity
        return {
            key: sim
            for key, sim in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True
            )
        }

    @cached_property
    def mean_neighborhoods_ordered_smiliarity(self) -> float:
        """Mean ordered neighborhoods similarity"""
        return np.mean(list(self.neighborhoods_ordered_smiliarities.values()))

    def get_most_similar(self, n_elements: int) -> list:
        """Get most similar elements from self.neighborhoods_smiliarities

        see itertools recipes for implementation :
        https://docs.python.org/3/library/itertools.html#recipes

        Args:
            n_elements (int): number of elements to return

        Returns:
            list: list of tuples (element, element_neighbors)
        """
        return list(islice(self.neighborhoods_smiliarities.items(), n_elements))

    def get_least_similar(self, n_elements: int):
        """Get least similar elements from self.neighborhoods_smiliarities

        see itertools recipes for implementation :
        https://docs.python.org/3/library/itertools.html#recipes

        Args:
            n_elements (int): number of elements to return

        Returns:
            list: list of tuples (element, element_neighbors)
        """
        return list(
            collections.deque(
                self.neighborhoods_smiliarities.items(), maxlen=n_elements
            )
        )
