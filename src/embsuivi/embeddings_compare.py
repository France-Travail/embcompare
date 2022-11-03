from functools import cached_property
from typing import Any, Dict, Tuple, Union

import editdistance
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from .embedding import Embedding


class EmbeddingComparison:
    def __init__(self, embeddings: Dict[Any, KeyedVectors], n_neighbors: int = 25):
        self.n_neighbors = n_neighbors

        assert (
            isinstance(embeddings, dict) and len(embeddings) == 2
        ), "embeddings should be a python dict containing two embeddings"

        self.__embeddings_ids = tuple(emb_id for emb_id in embeddings.keys())
        self.__embeddings = tuple(
            self._load_embedding(emb) for emb in embeddings.values()
        )

    @property
    def embeddings(self):
        return self.__embeddings

    @property
    def embeddings_ids(self):
        return self.__embeddings_ids

    @staticmethod
    def _load_embedding(embedding):
        if isinstance(embedding, Embedding):
            return embedding
        elif isinstance(embedding, KeyedVectors):
            return Embedding.load_from_keyedvectors(embedding)
        else:
            raise TypeError("Embeddings should be of type Embedding or KeyedVectors")

    def __getitem__(self, identifier):
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

    def is_frequencies_set(self):
        first_emb, second_emb = self.embeddings

        return np.any(first_emb.frequencies != first_emb._default_freq) and np.any(
            second_emb.frequencies != second_emb._default_freq
        )

    @cached_property
    def common_keys(self):
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
    def neighborhoods(self):
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
    def neighborhoods_smiliarities(self):
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
    def mean_neighborhoods_smiliarity(self):
        return np.mean(list(self.neighborhoods_smiliarities.values()))

    @cached_property
    def neighborhoods_ordered_smiliarities(self):
        emb1_neighborhoods, emb2_neighborhoods = self.neighborhoods

        similarities = {}

        for key in self.common_keys:
            key_neighbors_1 = [k for k, _ in emb1_neighborhoods[key]]
            key_neighbors_2 = [k for k, _ in emb2_neighborhoods[key]]

            similarities[key] = (
                1
                - editdistance.eval(key_neighbors_1, key_neighbors_2) / self.n_neighbors
            )

        # Sort by similarity
        return {
            key: sim
            for key, sim in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True
            )
        }
