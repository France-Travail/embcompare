import json
import pickle
from itertools import combinations
from pathlib import Path
from random import sample
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from loguru import logger
from sklearn.neighbors import NearestNeighbors


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "dtype") and hasattr(obj, "astype") and hasattr(obj, "tolist"):

            if np.issubdtype(obj.dtype, np.integer):
                return obj.astype(int).tolist()
            elif np.issubdtype(obj.dtype, np.number):
                return obj.astype(float).tolist()

        elif isinstance(obj, set):
            return list(obj)

        return json.JSONEncoder.default(self, obj)


def levenshtein(s1: Iterable, s2: Iterable) -> float:
    """Levenshtein distance between two iterable

    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

    Args:
        s1 (_type_): first iterable
        s2 (_type_): second iterable

    Returns:
        float: levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return 1.0

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1] / len(s1)


class EmbeddingsComparator:
    def __init__(self, *embeddings, n_neighbors: int = 25):
        self.embeddings: Dict[KeyedVectors] = {}

        self._n_neighbors = n_neighbors

        # cache for computed neigborhood similarities
        self._neighborhood_similarities = {n_neighbors: {}}
        self._neighborhood_order_similarities = {n_neighbors: {}}
        self._neighborhood = {n_neighbors: {}}

        # Ajout des embeddings donnés en entrée
        for embedding in embeddings:
            if isinstance(embedding, dict):
                for emb_name, emb in embedding.items():
                    self.add_embedding(emb_name, emb)
            else:
                emb_name, emb = embedding
                self.add_embedding(emb_name, emb)

    @property
    def n_neighbors(self) -> int:
        """Number of neighbors for comparison"""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n: int) -> None:
        """Set number of neighbors"""
        self._n_neighbors = n

        if n not in self._neighborhood_similarities:
            self._neighborhood_similarities[n] = {}
            self._neighborhood_order_similarities[n] = {}
            self._neighborhood[n] = {}

    @property
    def embeddings_names(self) -> list:
        """Added embeddings names"""
        return list(self.embeddings)

    def __getitem__(self, embedding_name: str) -> KeyedVectors:
        """Return an embedding by its name"""
        return self.embeddings[embedding_name]

    def __setitem__(
        self, embedding_name: str, embedding: Union[dict, KeyedVectors]
    ) -> None:
        """Add an embedding dict-like"""
        self.add_embedding(embedding_name, embedding)

    def __repr__(self) -> str:
        """Object representation"""
        return f"EmbeddingsComparator[{', '.join(self.embeddings)}]"

    def get_common_keys(self, embedding_names: List[str] = None) -> set:
        """Get common keys between all embeddings"""
        common_keys = None

        for embedding_name, embedding in self.embeddings.items():
            if embedding_names and embedding_name not in embedding_names:
                continue

            if common_keys is None:
                common_keys = set(embedding.key_to_index)
            else:
                common_keys = common_keys.intersection(embedding.key_to_index)

        return set() if common_keys is None else common_keys

    def clear_cache_embedding(self, embedding_name: str) -> None:
        """Clear all cached informations about an embedding

        Args:
            embedding_name (_type_): embedding name
        """
        # Remove neigborhood and similarities that implies the embedding
        if embedding_name in self.embeddings:
            for cache_dict in (
                self._neighborhood_similarities,
                self._neighborhood_order_similarities,
            ):
                for similarities in cache_dict.values():
                    for emb1_name, emb2_name in list(similarities.keys()):
                        if embedding_name in (emb1_name, emb2_name):
                            del similarities[(emb1_name, emb2_name)]

            for neighborhood in self._neighborhood.values():
                neighborhood.pop(embedding_name, None)

    def add_embedding(
        self,
        embedding_name: str,
        embedding: Union[str, dict, KeyedVectors],
        clean: bool = True,
    ) -> None:
        """Add an embedding to comparator

        Args:
            embedding_name (str): embedding name
            embedding (Union[str, dict, KeyedVectors]): embedding as a file path, a dict or a KeyedVector
            clean (bool, optional): If true, get ride of keys with a null embedding vector.
                Defaults to True.

        Raises:
            TypeError: raise TypeError if embedding is none of dict or KeyedVectors
        """
        self.clear_cache_embedding(embedding_name)

        if isinstance(embedding, str) or isinstance(embedding, Path):
            emb_path = Path(embedding)

            if emb_path.suffix == ".json":
                with emb_path.open("r") as f:
                    embedding = json.load(f)

            else:
                with emb_path.open("rb") as f:
                    embedding = pickle.load(f)

        if isinstance(embedding, KeyedVectors):
            self.embeddings[embedding_name] = embedding

        elif isinstance(embedding, dict):
            if len(embedding) == 0:
                logger.warning(
                    f"L'embedding {embedding_name} ne contient aucune clé. Skipped."
                )
                return None

            if clean:
                embedding = {
                    key: np.array(vec)
                    for key, vec in embedding.items()
                    if not all(v == 0 for v in vec)
                }

            if len(embedding) == 0:
                logger.warning(
                    f"L'embedding {embedding_name} ne contient aucun embedding non null. Skipped."
                )

            self.embeddings[embedding_name] = self.dict_to_keyedvectors(embedding)

        else:
            raise TypeError("L'embedding doit être de type dict")

    def compute_neighborhood_similiraties(
        self, embedding_names: List[str] = None, overwrite: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Compute neighborhood similarities between all pair of embeddings

        For a given key, the neighborhood similarity between two embeddings is the number of
        common neighbors divided by the number of neighbors in the two embeddings.

        The number of neighbors is fixed by the attribute n_neighbors

        Args:
            embedding_names (List[str], optional): List of embeddings for which to compute similarities.
                Defaults to None.
            overwrite (bool, optional): If true, overwrite neighborhood similarities that have been already
                computed. Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: Similarity dict for each pair of embeddings
        """
        if embedding_names is None:
            embedding_names = self.embeddings.keys()

        for emb1_name, emb2_name in combinations(embedding_names, 2):
            if (
                not overwrite
                and (emb1_name, emb2_name)
                in self._neighborhood_similarities[self.n_neighbors]
            ):
                continue

            similarities = self.compute_neighborhood_similarity(emb1_name, emb2_name)

            # Ajout des similarités au dictionnaire de cache
            self._neighborhood_similarities[self.n_neighbors][
                (emb1_name, emb2_name)
            ] = similarities

            # Ajout du symétrique
            self._neighborhood_similarities[self.n_neighbors][
                (emb2_name, emb1_name)
            ] = similarities

        return self._neighborhood_similarities[self.n_neighbors]

    def compute_neighborhood_order_similiraties(
        self, embedding_names: List[str] = None, overwrite: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Compute neighborhood order similarities between all pair of embeddings

        For a given key, the neighborhood order similarity is the levensthein similarity
        between neihbors orders

        The number of neighbors is fixed by the attribute n_neighbors

        Args:
            embedding_names (List[str], optional): List of embeddings for which to compute similarities.
                Defaults to None.
            overwrite (bool, optional): If true, overwrite neighborhood similarities that have been already
                computed. Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: Similarity dict for each pair of embeddings
        """
        if embedding_names is None:
            embedding_names = self.embeddings.keys()

        for emb1_name, emb2_name in combinations(embedding_names, 2):
            if (
                not overwrite
                and (emb1_name, emb2_name)
                in self._neighborhood_order_similarities[self.n_neighbors]
            ):
                continue

            similarities = self.compute_neighborhood_order_similarity(
                emb1_name, emb2_name
            )

            # Ajout des similarités au dictionnaire de cache
            self._neighborhood_order_similarities[self.n_neighbors][
                (emb1_name, emb2_name)
            ] = similarities

            # Ajout du symétrique
            self._neighborhood_order_similarities[self.n_neighbors][
                (emb2_name, emb1_name)
            ] = similarities

        return self._neighborhood_order_similarities[self.n_neighbors]

    def compute_neighborhood(self, emb_name: str) -> np.ndarray:
        """Compute neighborhood for an embeddings

        Args:
            emb_name (emb_name): embedding

        Returns:
            np.ndarray: neighborhood matrix
        """
        if emb_name not in self._neighborhood[self.n_neighbors]:
            emb = self.embeddings[emb_name]

            nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric="cosine")
            nn.fit(emb.vectors)

            self._neighborhood[self.n_neighbors][emb_name] = nn.kneighbors(emb.vectors)

        return self._neighborhood[self.n_neighbors][emb_name]

    def get_neighbors_by_key(
        self, emb_name: str, keys: Union[str, List[str]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Get neighbors of given keys in an embedding

        Args:
            emb_name (str): Embedding name
            keys (Union[str, List[str]]): keys

        Returns:
            Dict[str, List[str]]: dict containing neigbors and distances
        """
        emb: KeyedVectors = self.embeddings[emb_name]
        nn_dist, nn_ids = self.compute_neighborhood(emb_name)

        if isinstance(keys, str):
            keys = [keys]

        neighbors = {}

        for key in keys:
            key_id = emb.key_to_index[key]
            neighbors[key] = [
                (emb.index_to_key[ind], dist)
                for ind, dist in zip(nn_ids[key_id], nn_dist[key_id])
                if ind != key_id
            ]

        return neighbors

    def compute_neighborhood_similarity(
        self, emb1_name: str, emb2_name: str
    ) -> Dict[str, float]:
        """Compute neighbor similarity between two KeyedVectors embeddings

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name

        Returns:
            Dict[str, float]: Similarity dict
        """
        emb1: KeyedVectors = self.embeddings[emb1_name]
        emb2: KeyedVectors = self.embeddings[emb2_name]

        # Clés communes
        common_keys = set(emb1.key_to_index)
        common_keys = common_keys.intersection(emb2.key_to_index)

        # Récupération des voisinages
        emb1_neighborhoods = self.get_neighbors_by_key(emb1_name, common_keys)
        emb2_neighborhoods = self.get_neighbors_by_key(emb2_name, common_keys)

        similarities = {}
        for key in common_keys:
            key_neighbors_1 = {k for k, _ in emb1_neighborhoods[key]}
            key_neighbors_2 = {k for k, _ in emb2_neighborhoods[key]}

            similarities[key] = len(
                key_neighbors_1.intersection(key_neighbors_2)
            ) / len(key_neighbors_1.union(key_neighbors_2))

        # On trie le dictionnaire par similarité
        return {
            key: sim
            for key, sim in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True
            )
        }

    def compute_neighborhood_order_similarity(
        self, emb1_name: str, emb2_name: str, n_neighbors: int = None
    ) -> Dict[str, float]:
        """Compute neighbor order similarity between two KeyedVectors embeddings

        The order of neighbors are compared thanks to levensthein distance

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name
            n_neighbors (int): number of neighbors to take in account

        Returns:
            Dict[str, float]: Similarity dict
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Si le nombre de voisins souhaité est inferieur au nombre de voisins
        # actuel, il n'est pas nécessaire de redéfinir le nombre de voisins puisque
        # qu'il suffit de sélectionner les premiers voisins parmis ceux déjà calculés
        elif n_neighbors > self.n_neighbors:
            self.n_neighbors = n_neighbors

        emb1: KeyedVectors = self.embeddings[emb1_name]
        emb2: KeyedVectors = self.embeddings[emb2_name]

        # Clés communes
        common_keys = set(emb1.key_to_index)
        common_keys = common_keys.intersection(emb2.key_to_index)

        # Récupération des voisinages
        emb1_neighborhoods = self.get_neighbors_by_key(emb1_name, common_keys)
        emb2_neighborhoods = self.get_neighbors_by_key(emb2_name, common_keys)

        similarities = {}
        for key in common_keys:
            key_neighbors_1 = [k for k, _ in emb1_neighborhoods[key]][:n_neighbors]
            key_neighbors_2 = [k for k, _ in emb2_neighborhoods[key]][:n_neighbors]

            similarities[key] = 1 - levenshtein(key_neighbors_1, key_neighbors_2)

        # On trie le dictionnaire par similarité
        return {
            key: sim
            for key, sim in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True
            )
        }

    def get_neighborhood_similarity(self, emb1_name: str, emb2_name: str) -> dict:
        """Get similarities between

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name

        Returns:
            dict: Key similarities
        """
        if (emb1_name, emb2_name) not in self._neighborhood_similarities[
            self.n_neighbors
        ]:
            self.compute_neighborhood_similiraties((emb1_name, emb2_name))

        return self._neighborhood_similarities[self.n_neighbors][(emb1_name, emb2_name)]

    def get_neighborhood_order_similarity(self, emb1_name: str, emb2_name: str) -> dict:
        """Get order similarities between

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name

        Returns:
            dict: Key similarities
        """
        if (emb1_name, emb2_name) not in self._neighborhood_order_similarities[
            self.n_neighbors
        ]:
            self.compute_neighborhood_order_similiraties((emb1_name, emb2_name))

        return self._neighborhood_order_similarities[self.n_neighbors][
            (emb1_name, emb2_name)
        ]

    def get_most_similar(
        self, emb1_name: str, emb2_name: str, topn: int = 10
    ) -> List[Tuple[str, float]]:
        """Get most similar keys

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name
            topn (int, optional): number of keys to return. Defaults to 10.

        Returns:
            List[Tuple[str, float]]: list of tuple (key, similarity)
        """
        similarity = self.get_neighborhood_similarity(emb1_name, emb2_name)
        similarity_keys = list(similarity)
        return [(key, similarity[key]) for key in similarity_keys[:topn]]

    def get_least_similar(
        self, emb1_name: str, emb2_name: str, topn: int = 10
    ) -> List[Tuple[str, float]]:
        """Get least similar keys

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name
            topn (int, optional): number of keys to return. Defaults to 10.

        Returns:
            List[Tuple[str, float]]: list of tuple (key, similarity)
        """
        similarity = self.get_neighborhood_similarity(emb1_name, emb2_name)
        similarity_keys = list(similarity)
        return [(key, similarity[key]) for key in similarity_keys[-topn:][::-1]]

    def get_neighborhood_comparison(
        self, emb1_name: str, emb2_name: str, keys: Union[str, List[str]]
    ) -> Dict[str, Dict[str, List[str]]]:
        """Get neighborhood comparison in two different embeddings

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name
            keys (Union[str, List[str]]): keys to compare.

        Returns:
            Dict[str, Dict[str, List[str]]]: a dict containing for each key : common
                and distinct neighbors
        """
        if isinstance(keys, str):
            keys = [keys]

        neigbors_similar_1 = self.get_neighbors_by_key(emb1_name, keys)
        neigbors_similar_2 = self.get_neighbors_by_key(emb2_name, keys)

        most_similar_comparison = {}

        for key in keys:
            key_neighbors_1 = [k for k, _ in neigbors_similar_1[key]]
            key_neighbors_2 = [k for k, _ in neigbors_similar_2[key]]

            key_neighbors_1_set = set(key_neighbors_1)
            key_neighbors_2_set = set(key_neighbors_2)

            most_similar_comparison[key] = {
                "common": [
                    k
                    for k in key_neighbors_1
                    if k in key_neighbors_1_set and k in key_neighbors_2_set
                ],
                emb1_name: [k for k in key_neighbors_1 if k not in key_neighbors_2_set],
                emb2_name: [k for k in key_neighbors_2 if k not in key_neighbors_1_set],
            }

        return most_similar_comparison

    def get_most_similar_comparison(
        self, emb1_name: str, emb2_name: str, topn: int = 20
    ) -> Dict[str, Dict[str, List[str]]]:
        """Get a neighborhood comparison of most similar keys

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name
            topn (int, optional): Number of keys to compare. Defaults to 20.

        Returns:
            Dict[str, Dict[str, List[str]]]: a dict containing for each key : common
                and distinct neighbors
        """

        most_similar_keys = [
            k for k, _ in self.get_most_similar(emb1_name, emb2_name, topn=topn)
        ]

        return self.get_neighborhood_comparison(emb1_name, emb2_name, most_similar_keys)

    def get_least_similar_comparison(
        self, emb1_name: str, emb2_name: str, topn: int = 20
    ) -> Dict[str, Dict[str, List[str]]]:
        """Get a neighborhood comparison of least similar keys

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name
            topn (int, optional): Number of keys to compare. Defaults to 20.

        Returns:
            Dict[str, Dict[str, List[str]]]: a dict containing for each key : common
                and distinct neighbors
        """

        least_similar_keys = [
            k for k, _ in self.get_least_similar(emb1_name, emb2_name, topn=topn)
        ]

        return self.get_neighborhood_comparison(
            emb1_name, emb2_name, least_similar_keys
        )

    def compute_mean_distance_to_neighbors(
        self, emb_name: str, distribution: bool = False
    ) -> Union[float, np.ndarray]:
        """Compute the mean distance between a key and its nearest neighbors

        Args:
            emb_name (str): name of the embedding
            distribution (bool, optional): if True, return the distribution instead of the
                mean value. Defaults to False.

        Returns:
            Union[float, np.ndarray]:: mean distance between a key and its nearest neighbors
                or the whole distribution if distribution is True
        """
        nn_dist, _ = self.compute_neighborhood(emb_name)

        # the first dist of each row is 0 because it is the distance between the key and itself
        if distribution:
            return np.mean(nn_dist[:, 1:], axis=1)
        else:
            return np.mean(nn_dist[:, 1:])

    def compute_mean_min_distance_to_neighbors(
        self, emb_name: str, distribution: bool = False
    ) -> Union[float, np.ndarray]:
        """Compute the mean of distance between a key and its nearest neighbor

        Args:
            emb_name (str): name of the embedding
            distribution (bool, optional): if True, return the distribution instead of the
                mean value. Defaults to False.

        Returns:
            float: mean distance between a key and its nearest neighbor or the whole distribution
                if distribution is True
        """
        nn_dist, _ = self.compute_neighborhood(emb_name)

        # the first dist of each row is 0 because it is the distance between the key and itself
        nn_dist_min = np.min(nn_dist[:, 1:], axis=1)

        if distribution:
            return nn_dist_min
        else:
            return np.mean(nn_dist_min)

    def statistics_to_dict(
        self, emb_name: str, decimals: int = 4, bins: int = 10
    ) -> dict:
        """Return statistics about an embedding in a dict

        Computed statistics :
        - mean distance (distribution and mean)
        - mean min distance (distribution and mean)

        Args:
            emb_name (str): embedding name
            decimals (int, optional): Number of decimals. Defaults to 4.
            bins (int, optional): Number of bins for histograms. Defaults to 10.

        Returns:
            dict: dict of statistics
        """
        mean_dist_distrib = self.compute_mean_distance_to_neighbors(
            emb_name, distribution=True
        )
        mean_min_dist_distrib = self.compute_mean_min_distance_to_neighbors(
            emb_name, distribution=True
        )

        # Histogrames
        mean_dist_hist = [list(a) for a in np.histogram(mean_dist_distrib, bins=bins)]
        mean_min_dist_hist = [
            list(a) for a in np.histogram(mean_min_dist_distrib, bins=bins)
        ]

        return {
            "name": emb_name,
            "mean_dist": round(np.mean(mean_dist_distrib), decimals),
            "mean_min_dist": round(np.mean(mean_min_dist_distrib), decimals),
            "mean_dist_hist": mean_dist_hist,
            "mean_min_dist_hist": mean_min_dist_hist,
        }

    def comparison_to_dict(
        self, emb1_name: str, emb2_name: str, decimals: int = 4, bins: int = 10
    ) -> dict:
        """Return a comparison between two embeddings as a dict

        Args:
            emb1_name (str): first embedding name
            emb1_name (str): second embedding name
            decimals (int, optional): Number of decimals. Defaults to 4.
            bins (int, optional): Number of bins for histograms. Defaults to 10.

        Returns:
            dict: dict with comparison statistics
        """
        # Embedding statistics
        emb1_stats = self.statistics_to_dict(emb1_name, decimals=decimals, bins=bins)
        emb2_stats = self.statistics_to_dict(emb2_name, decimals=decimals, bins=bins)

        # Neighborhood similarities
        neighborhood_sim = self.compute_neighborhood_similarity(emb1_name, emb2_name)
        neighborhood_order_sim = self.compute_neighborhood_order_similarity(
            emb1_name, emb2_name
        )

        # ATTENTION : on utilise des arrays pour permettre au NumpyArrayEncoder de decoder
        # les objets en json
        neighborhood_sim_array = np.array(list(neighborhood_sim.values()))
        neighborhood_order_sim_array = np.array(list(neighborhood_order_sim.values()))

        # mediane
        neighborhood_sim_median = np.median(neighborhood_sim_array)
        neighborhood_order_sim_median = np.median(neighborhood_order_sim_array)

        # histogrammes
        neighborhood_sim_hist = [
            list(a) for a in np.histogram(neighborhood_sim_array, bins=bins)
        ]
        neighborhood_order_sim_hist = [
            list(a) for a in np.histogram(neighborhood_order_sim_array, bins=bins)
        ]

        # Plus ou moins similaires
        most_similar_comp = self.get_most_similar_comparison(emb1_name, emb2_name)
        least_similar_comp = self.get_least_similar_comparison(emb1_name, emb2_name)

        return {
            "embeddings": [emb1_stats, emb2_stats],
            "neighborhood_sim_median": round(neighborhood_sim_median, decimals),
            "neighborhood_order_sim_median": round(
                neighborhood_order_sim_median, decimals
            ),
            "neighborhood_sim_hist": neighborhood_sim_hist,
            "neighborhood_order_sim_hist": neighborhood_order_sim_hist,
            "most_similar": most_similar_comp,
            "least_similar": least_similar_comp,
        }

    def statistics_to_json(
        self, emb_name: str, filepath: str, decimals: int = 4, bins: int = 10, **kwargs
    ) -> dict:
        """Compute statistics about an embedding, save it as a json file and return the dict

        Args:
            emb_name (str): embedding name
            filepath (str): file path
            decimals (int, optional): Number of decimals. Defaults to 4.
            bins (int, optional): Number of bins for histograms. Defaults to 10.

        Returns:
            dict: dict of statistics
        """
        statistics = self.statistics_to_dict(emb_name, decimals=decimals, bins=bins)

        kwargs["cls"] = kwargs.get("cls", NumpyArrayEncoder)

        with open(filepath, "w") as f:
            json.dump(statistics, f, **kwargs)

        return statistics

    def comparison_to_json(
        self,
        emb1_name: str,
        emb2_name: str,
        filepath: str,
        decimals: int = 4,
        bins: int = 10,
        **kwargs,
    ) -> dict:
        """Compare two embedding, save comparison statistics as a json file and return the dict

        Args:
            emb1_name (str): first embedding name
            emb2_name (str): second embedding name
            filepath (str): file path
            decimals (int, optional): Number of decimals. Defaults to 4.
            bins (int, optional): Number of bins for histograms. Defaults to 10.

        Returns:
            dict: dict of statistics
        """
        statistics = self.comparison_to_dict(
            emb1_name, emb2_name, decimals=decimals, bins=bins
        )

        kwargs["cls"] = kwargs.get("cls", NumpyArrayEncoder)

        with open(filepath, "w") as f:
            json.dump(statistics, f, **kwargs)

        return statistics

    @staticmethod
    def dict_to_keyedvectors(embedding_dict: dict) -> KeyedVectors:
        """Transform a dict embedding to a gensim KeyedVectors object

        Args:
            embedding_dict (dict): embedding dict

        Returns:
            KeyedVectors: gensim KeyedVectors
        """
        embedding: KeyedVectors = KeyedVectors(
            vector_size=len(embedding_dict[next(iter(embedding_dict))]),
            count=len(embedding_dict),
        )
        for key, key_emb in embedding_dict.items():
            embedding.add_vector(key, key_emb)

        return embedding

    @staticmethod
    def sample_embeddings(
        *embeddings: KeyedVectors, n_samples: int = 30000
    ) -> List[KeyedVectors]:
        """Create sampled embeddings containing the same keys

        Args:
            n_samples (int, optional): number of sample. Defaults to 30000.

        Returns:
            List[KeyedVectors]: sampled embeddings
        """
        # Détermination des clés en commun
        common_words = None
        for emb in embeddings:
            if common_words is None:
                common_words = set(emb.index_to_key)
            else:
                common_words = common_words.intersection(emb.index_to_key)

        # S'il n'y a pas assez de clés en commun, on renvoie les embeddings tels quels
        if n_samples >= len(common_words):
            return embeddings

        # Sinon on crée de nouveaux embeddings contenant des clés en commun tirées au sort
        random_words = sample(common_words, int(n_samples))
        sampled_embeddings = []

        for emb in embeddings:
            sampled_emb: KeyedVectors = KeyedVectors(
                vector_size=emb.vectors.shape[1],
                count=n_samples,
            )
            for key in random_words:
                sampled_emb.add_vector(key, emb.get_vector(key))

            sampled_embeddings.append(sampled_emb)

        return sampled_embeddings
