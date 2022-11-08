import json
from functools import cached_property
from typing import Any, Dict, List

import numpy as np

from .embedding import Embedding
from .embeddings_compare import EmbeddingComparison
from .export_utils import NumpyArrayEncoder


class Report:
    FEATURES = ()

    def __init__(self, include_features: tuple = None):
        """Initialization of a report object

        Args:
            include_features (tuple, optional): features to include in the report. Defaults to None.
        """
        self.include_features = include_features

    @property
    def include_features(self) -> list:
        """List containing features to include in report"""
        return self.__include_features

    @include_features.setter
    def include_features(self, features: list):
        """Set include_features attribute

        When include_features is (re)set to None, all features are included
        in the report

        Args:
            features (list): list of features to include in the report
        """
        if features is None:
            self.__include_features = list(self.FEATURES)
        else:
            self.__include_features = features

    def __str__(self) -> str:
        """string representation of the object"""
        return f"{self.__class__.__name__}(include_features=[{', '.join(self.include_features)}])"

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
        dict_report = {}

        for feature in self.include_features:
            if isinstance(feature, str):
                dict_report[feature] = getattr(self, feature)

            elif isinstance(feature, tuple):
                feature, feature_function = feature
                dict_report[feature] = feature_function(self)

            else:
                raise TypeError(
                    "included features shoud be eiter an attribute of the report "
                    "object or a tuple (feature, feature_function)"
                )

        return dict_report

    def to_json(self, path: str, **kwargs) -> dict:
        """Save report as a JSON file and return it as python dict"""
        features = self.to_dict()

        # Add NumpyArrayEncoder as default encoder to encode numpy arrays
        kwargs["cls"] = kwargs.get("cls", NumpyArrayEncoder)

        with open(path, "w") as f:
            json.dump(features, f, **kwargs)

        return features


class EmbeddingReport(Report):
    FEATURES = (
        "vector_size",
        "n_elements",
        "default_frequency",
        "mean_frequency",
        "n_neighbors",
        "mean_distance_neighbors",
        "mean_distance_first_neigbor",
    )

    def __init__(
        self, embedding: Embedding, n_neighbors: int, include_features: tuple = None
    ):
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

        super(EmbeddingReport, self).__init__(include_features=include_features)

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


class EmbeddingComparisonReport(Report):
    FEATURES = (
        "embeddings",
        "neighborhoods_similarities_median",
        "neighborhoods_ordered_similarities_median",
    )

    def __init__(self, comparison: EmbeddingComparison, include_features: tuple = None):
        """Initialize a EmbeddingComparisonReport from a EmbeddingComparison instance

        Args:
            comparison (EmbeddingComparison): EmbeddingComparison instance
            include_features (tuple, optional): features to include. By setting it to None, all features
                containing in FEATURES are included. Defaults to None.
        """
        self.comparison = comparison

        super(EmbeddingComparisonReport, self).__init__(
            include_features=include_features
        )

    @property
    def n_neighbors(self):
        """Number of neighbors on which is based the comparison"""
        return self.comparison.n_neighbors

    @cached_property
    def embeddings(self) -> List[dict]:
        """return a list containing a dict report of both embeddings"""
        first_emb_id, second_emb_id = self.comparison.embeddings_ids
        first_emb, second_emb = self.comparison.embeddings

        first_emb_report = EmbeddingReport(first_emb, self.n_neighbors).to_dict()
        second_emb_report = EmbeddingReport(second_emb, self.n_neighbors).to_dict()

        first_emb_report["name"] = str(first_emb_id)
        second_emb_report["name"] = str(second_emb_id)

        return [first_emb_report, second_emb_report]

    @property
    def neighborhoods_similarities(self) -> Dict[str, float]:
        """neighborhoods similarities of the two embeddings"""
        return self.comparison.neighborhoods_similarities

    @cached_property
    def neighborhoods_similarities_values(self) -> np.ndarray:
        """neighborhoods similarities values in an numpy array"""
        return np.array(list(self.neighborhoods_similarities.values()))

    @property
    def neighborhoods_similarities_median(self) -> float:
        """median neighborhoods similaritiy"""
        return np.median(self.neighborhoods_similarities_values)

    @property
    def neighborhoods_ordered_similarities(self) -> Dict[str, float]:
        """neighborhoods ordered similarities of the two embeddings"""
        return self.comparison.neighborhoods_ordered_similarities

    @cached_property
    def neighborhoods_ordered_similarities_values(self) -> np.ndarray:
        """neighborhoods ordered similarities values in an numpy array"""
        return np.array(list(self.neighborhoods_ordered_similarities.values()))

    @property
    def neighborhoods_ordered_similarities_median(self) -> float:
        """median neighborhoods ordered similaritiy"""
        return np.median(self.neighborhoods_ordered_similarities_values)
