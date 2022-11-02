import json
from typing import Any

import numpy as np

from .embedding import Embedding
from .export_utils import NumpyArrayEncoder


class EmbeddingReport:
    FEATURES = (
        "vector_size",
        "n_elements",
        "default_frequency",
        "mean_frequency",
        "n_neighbors",
        "mean_distance_neighbors",
        "mean_distance_first_neigbor",
    )

    def __init__(self, embedding: Embedding, n_neighbors: int, include: tuple = None):
        """Initialize a EmbeddingReport object

        All report features are object properties so are computed only on demand

        Args:
            embedding (Embedding): Embedding object
            n_neighbors (int): Number of neighbors for statistics computation.
            include (tuple, optional): Statistics to include in report. By default all
                availables statistics are returned.
        """
        self.embedding = embedding
        self.n_neighbors = n_neighbors
        self.include = include

    @property
    def include(self) -> list:
        """List containing features to include in report"""
        return self._include

    @include.setter
    def include(self, value: list):
        """Set include attribute

        When include is (re)set to None, all features are included in the report

        Args:
            value (list): list of features to include in the report
        """
        if value is None:
            self._include = list(self.FEATURES)
        else:
            self._include = value

    @property
    def vector_size(self) -> int:
        """Embedding vector dimension"""
        return self.embedding.vector_size

    @property
    def n_elements(self) -> int:
        """Number of elements in the embedding"""
        return len(self.embedding.key_to_index)

    @property
    def default_frequency(self) -> float:
        """Default elements frequency"""
        return self.embedding._default_freq

    @property
    def mean_frequency(self) -> float:
        """Mean elements frequency"""
        return np.mean(self.embedding.frequencies)

    @property
    def nearest_neighbors_distances(self) -> np.ndarray:
        """Get nearest neighbors distance matrix from the embedding"""
        nn_dist, _ = self.embedding.compute_neighborhoods(n_neighbors=self.n_neighbors)
        return nn_dist

    @property
    def mean_distance_neighbors(self) -> float:
        """Mean distance to neighbors"""
        return np.mean(self.nearest_neighbors_distances)

    @property
    def mean_distance_first_neigbor(self) -> float:
        """Mean distance to first neighbor"""
        return np.mean(self.nearest_neighbors_distances[:, 0])

    def __repr__(self) -> str:
        """string representation of the object"""
        return (
            f"{self.__class__}(n_neighbors={self.n_neighbors}, "
            f"vector_size={self.vector_size}, n_elements={self.n_elements})"
        )

    def __getitem__(self, feature: str) -> Any:
        """Get a feature dict-like"""
        if not isinstance(feature, str):
            raise ValueError(
                f"feature should be one of {', '.join(self.FEATURES)} "
                f"but is of type : {type(feature)}"
            )
        elif feature in self.FEATURES:
            return getattr(self, feature)
        else:
            raise KeyError(
                f"feature should be one of [{', '.join(self.FEATURES)}] "
                f"(got {feature})"
            )

    def to_dict(self) -> dict:
        """Return report as a python dict"""
        return {stat: self[stat] for stat in self.include}

    def to_json(self, path: str, **kwargs) -> dict:
        """Save report as a JSON file and return it as python dict"""
        stats = self.to_dict()

        kwargs["cls"] = kwargs.get("cls", NumpyArrayEncoder)

        with open(path, "w") as f:
            json.dump(stats, f, **kwargs)

        return stats
