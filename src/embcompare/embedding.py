import json
from pathlib import Path
from typing import Dict, List, Tuple, Type, TypeVar, Union

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.neighbors import NearestNeighbors

# Types and aliases
TEmbedding = TypeVar("TEmbedding", bound="Embedding")
EmbeddingDict = Dict[str, Union[List[float], np.ndarray]]
EmbeddingNeighborhoodsMatrices = Tuple[np.ndarray, np.ndarray]
NeighborhoodsDict = Dict[str, List[Tuple[str, float]]]


DEFAULT_N_NEIGHBORS = 25


class Embedding(KeyedVectors):
    def __init__(
        self,
        vector_size: int,
        count: int = 0,
        dtype=np.float32,
        default_freq: float = 0,
        **kwargs,
    ):
        """Initialize an Embedding object

        Args:
            vector_size (int): Intended number of dimensions for all contained vectors.
            count (int, optional): If provided, vectors wil be pre-allocated for at least this
                many vectors (Otherwise they can be added later). Defaults to 0.
            dtype (type, optional): Vector dimensions will default to np.float32
                (AKA REAL in some Gensim code) unless another type is provided here.
                Defaults to np.float32.
            default_freq (float, optional): default frequency for elements. Defaults to 0.0.
        """
        super(Embedding, self).__init__(
            vector_size=vector_size, count=count, dtype=dtype, **kwargs
        )

        self.__neighborhoods: NeighborhoodsDict = None
        self._default_freq = default_freq
        self._dtype = dtype
        self.frequencies = np.zeros(shape=count, dtype=dtype)

        if default_freq:
            self.frequencies = self.frequencies + default_freq

    def add_vector(self, key: str, vector: np.ndarray, frequency: float = None) -> int:
        """Add one new vector at the given key, into existing slot if available.

        Warning: using this repeatedly is inefficient, requiring a full reallocation & copy,
            if this instance hasn’t been preallocated to be ready for such incremental additions.

        Args:
            key (str): Key identifier of the added vector.
            vector (np.ndarray): 1D numpy array with the vector values.
            frequency (float, optional): frequency othe the element. Defaults to None.

        Returns:
            int: index of the element
        """
        # If key is new, self.vectors and self.frequencies arrays has to be augmented.
        # Since KeyedVectors.add_vector is using add_vectors under the hood,
        # Embedding.add_vectors is called here and self.frequencies is augmented by it.
        ind = super(Embedding, self).add_vector(key, vector)

        if frequency is not None:
            self.frequencies[ind] = frequency

        return ind

    def add_vectors(
        self,
        keys,
        weights,
        frequencies: Union[List[float], np.ndarray] = None,
        **kwargs,
    ):
        """Append keys and their vectors in a manual way.

        If some key is already in the vocabulary, the old vector is kept unless replace flag is True.

        Args:
            keys (_type_): Keys specified by string or int ids.
            weights (_type_): List of 1D np.array vectors or a 2D np.array of vectors.
            frequencies (Union[List[float], np.ndarray], optional): List of elements frequencies. Defaults to None.
        """
        super(Embedding, self).add_vectors(keys, weights, **kwargs)

        n_new_elements = len(self.index_to_key) - self.frequencies.shape[0]

        if n_new_elements > 0:
            self.frequencies = np.concatenate(
                [self.frequencies, [0] * n_new_elements], dtype=self._dtype
            )

        if frequencies:
            for key, freq in zip(keys, frequencies):
                if freq is not None:
                    ind = self.key_to_index[key]
                    self.frequencies[ind] = freq

    def get_frequency(self, key: Union[str, int]) -> float:
        """Return element frequency

        Args:
            key (Union[str, int]): key or indice of an element.

        Returns:
            float: frequency
        """
        if isinstance(key, str):
            key = self.key_to_index[key]

        return self.frequencies[key]

    def set_frequencies(self, frequencies: Union[list, np.ndarray, dict]):
        """Set frequencies from a dict or a array-like

        Args:
            frequencies (Union[list, np.ndarray, dict]): frequencies in a dict or
                a array-like
        """
        if isinstance(frequencies, dict):
            for key, freq in frequencies.items():
                if key in self.key_to_index:
                    ind = self.key_to_index[key]
                    self.frequencies[ind] = freq

        else:
            for ind, freq in enumerate(frequencies):
                self.frequencies[ind] = freq

    def ordered(self) -> TEmbedding:
        """Return an Embedding object ordered by element frequencies

        Returns:
            TEmbedding: A new Embedding object ordered descendingly by element frequencies
        """
        # key_to_index only contain keys that have been added so the resulting
        # ordered_embedding is a "clean" embedding that does not contain null
        # vectors that may have been initialized earlier
        ordered_embedding = Embedding(
            vector_size=self.vector_size, count=len(self.key_to_index)
        )
        for key, emb, freq in sorted(
            [
                (key, self.vectors[ind], self.frequencies[ind])
                for key, ind in self.key_to_index.items()
            ],
            key=lambda x: x[2],
            reverse=True,
        ):
            ordered_embedding.add_vector(key, emb, freq)

        return ordered_embedding

    def compute_neighborhoods(
        self, n_neighbors: int = DEFAULT_N_NEIGHBORS
    ) -> EmbeddingNeighborhoodsMatrices:
        """Compute neighbors of all elements

        Args:
            n_neighbors (int, optional): Number of neighbors to consider. Defaults to DEFAULT_N_NEIGHBORS.

        Returns:
            EmbeddingNeighborhoodsMatrices: A tuple containing two matrices : a matrix
                of distances with nearest neighbors and a matrix of nearest neighbors
                indices
        """
        # If neighborhoods have already been computed with enough neighbors we can reuse
        # _neighborhoods
        if self.__neighborhoods is not None:
            nn_dist, nn_ids = self.__neighborhoods

            # _neighborhoods can be reused only if enough neighbors have been computed
            if n_neighbors <= nn_dist.shape[1]:
                return (
                    nn_dist[:, 0:n_neighbors],
                    nn_ids[:, 0:n_neighbors],
                )

        # We compute nearest neighbors distance and indices matrices thanks to
        # sklearn NearestNeighbors. We have to use n_neighbors = n_neighbors + 1
        # because the nereast neighbor of each element is the element itself
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
        nn.fit(self.vectors)
        nn_dist, nn_ids = nn.kneighbors(self.vectors)

        assert nn_dist.shape == nn_ids.shape
        assert nn_dist.shape[0] == len(self.index_to_key)
        assert nn_dist.shape[1] == n_neighbors + 1

        # We skip the first element of each row because it is the element itself
        self.__neighborhoods = (
            nn_dist[:, 1 : n_neighbors + 1],
            nn_ids[:, 1 : n_neighbors + 1],
        )

        return self.__neighborhoods

    def get_neighbors(
        self, n_neighbors: int = DEFAULT_N_NEIGHBORS, keys: List[str] = None
    ) -> NeighborhoodsDict:
        """Compute a dict containing most similar elements for all given keys

        Args:
            n_neighbors (int, optional): Number of similar elements to return. Defaults to
                DEFAULT_N_NEIGHBORS.
            keys (List[str], optional): Keys to consider. If None, return neighbors
                for all keys. Defaults to None.

        Returns:
            NeighborhoodsDict: A dict whose keys are elements and values are a list of tuple
                (neighbor, similarity) order by similarity
        """
        if keys is None:
            keys = self.index_to_key

        neighbors = {}
        nn_dist, nn_ids = self.compute_neighborhoods(n_neighbors=n_neighbors)

        for key in keys:
            key_id = self.key_to_index[key]

            neighbors[key] = [
                (self.index_to_key[ind], 1 - dist)  # similarity = 1 - distance
                for ind, dist in zip(nn_ids[key_id], nn_dist[key_id])
            ]

        return neighbors

    def is_frequency_set(self) -> bool:
        """Verify at least one element frequency is different from default frequency

        Returns:
            bool: True at least one element frequency is different from default
                frequency else otherwise
        """

        return bool(np.any(self.frequencies != self._default_freq))

    def filter_by_frequency(self, threshold: float) -> TEmbedding:
        """Return an Embedding object containing elements which have a greater
        frequency than a given threshold

        Args:
            threshold (float): frequency threshold

        Returns:
            TEmbedding: resulting Embedding object
        """
        vectors = {}
        frequencies = {}

        for key in self.key_to_index:
            freq = self.get_frequency(key)

            if freq >= threshold:
                vectors[key] = self.get_vector(key)
                frequencies[key] = freq

        return self.load_from_dict(vectors, frequencies, remove_null_vectors=False)

    @classmethod
    def load_from_dict(
        cls: Type[TEmbedding],
        embedding_dict: EmbeddingDict,
        frequencies: dict = None,
        remove_null_vectors: bool = True,
    ) -> TEmbedding:
        """Instantiate a Embedding object from an embedding dict

        Args:
            embedding_dict (EmbeddingDict): an embedding dict (key : vector)
            frequencies (dict, optional): a dict of element frequencies (key: frequency).
                If None, frequencies will be initialized as a null vector. Default to None.
            remove_null_vectors (bool, optional): if True, elements with null embedding vectors are
                removed. Default to True.

        Raises:
            ValueError: embedding_dict should not be empty

        Returns:
            Embedding: an Embedding object
        """
        if not embedding_dict:
            raise ValueError(f"Input dictionnary should not be empty")

        vector_size = len(embedding_dict[next(iter(embedding_dict))])

        if frequencies is None:
            frequencies = {}
            embedding_items = embedding_dict.items()
        else:
            # if frequencies are given, we sort elements by their frequency
            embedding_items = sorted(
                embedding_dict.items(),
                key=lambda key_vec: frequencies.get(key_vec[0], 0),
                reverse=True,
            )

        if remove_null_vectors:
            embedding_dict = {
                key: vector
                for key, vector in embedding_items
                if not all(map(lambda x: x == 0, vector))
            }
            embedding_items = embedding_dict.items()

        embedding: Embedding = cls(
            vector_size=vector_size,
            count=len(embedding_dict),
        )

        for key, vector in embedding_items:
            embedding.add_vector(key, vector, frequency=frequencies.get(key, None))

        return embedding

    @classmethod
    def load_from_json(cls: Type[TEmbedding], filepath: Union[str, Path]) -> TEmbedding:
        """Instantiate a Embedding object from a json file

        Args:
            filepath (Union[str, Path]): file path

        Returns:
            TEmbedding: an Embedding object
        """
        with open(filepath, "r") as f:
            embedding_dict = json.load(f)

        return cls.load_from_dict(embedding_dict)

    @classmethod
    def load_from_keyedvectors(
        cls: Type[TEmbedding],
        keyedvectors: KeyedVectors,
        frequencies: Union[dict, list, np.ndarray] = None,
    ) -> TEmbedding:
        """Instantiate a Embedding object from a gensim KeyedVectors object

        Args:
            keyedvectors (KeyedVectors): KeyedVectors object.
            frequencies (Union[dict, list, np.ndarray], optional): Frequencies of elements.
                Can be provided as a dict (key: frequency), or a 1D array-like frequencies ordered
                accordingly to key_to_index. If not provided elements are added in the same order
                than the initial KeyedVectors object. Defaults to None.

        Returns:
            TEmbedding: an Embedding object
        """
        count, vector_size = keyedvectors.vectors.shape

        # Since embedding is initialized with count=keyedvectors.vectors.shape[0]
        # pre-allocated vectors are inherited
        embedding: Embedding = cls(vector_size=vector_size, count=count)

        # If no frequencies has been given we initialize frequencies as an empty dict
        if frequencies is None:
            frequencies = {}
        elif not isinstance(frequencies, dict):
            frequencies = {
                key: freq for key, freq in zip(keyedvectors.index_to_key, frequencies)
            }

        # We sort key by frequency while preserving order for keys that do not have
        # a frequency by sorting according to a tuple (frequency, - rank) in descending
        # order so highest frequencies come first and if frequencies are equals, the lower
        # ranks come firsts
        if frequencies:
            sorted_keys = sorted(
                [(key, ind) for key, ind in keyedvectors.key_to_index.items()],
                key=lambda ki: (frequencies.get(ki[0], 0), -ki[1]),
                reverse=True,
            )
        else:
            sorted_keys = keyedvectors.key_to_index.items()

        for key, _ in sorted_keys:
            embedding.add_vector(
                key, keyedvectors.get_vector(key), frequency=frequencies.get(key, None)
            )

        return embedding
