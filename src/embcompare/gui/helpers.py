from pathlib import Path
from typing import Tuple

import streamlit as st
from loguru import logger

from ..embedding import Embedding
from ..embeddings_compare import EmbeddingComparison
from ..load_utils import EMBEDDING_FORMATS, FREQUENCIES_FORMATS


def round_sig(value: float, n_digits: int = 2) -> float:
    """Round float to significant figures

    Args:
        value (float): float number
        n_digits (int, optional): number fo significant digits. Defaults to 2.

    Returns:
        float: float
    """
    return float(f"{value:.{n_digits}g}")


def stop_if_any_embedding_unset(config_embeddings: dict, emb1_id: str, emb2_id: str):
    """Stop streamlit execution if the selected embeddings are not both in config_embeddings

    Args:
        config_embeddings (dict): embeddings configuration dict
        emb1_id (str): first embedding id
        emb2_id (str): second embedding id
    """
    emb1_is_set = emb1_id in config_embeddings
    emb2_is_set = emb2_id in config_embeddings

    if not emb1_is_set and not emb2_is_set:
        st.warning("No embedding set", icon="⚠")

    elif not emb1_is_set:
        st.warning("First embedding not set", icon="⚠")

    elif not emb2_is_set:
        st.warning("Second embedding not set", icon="⚠")

    elif emb1_id == emb2_id:
        st.warning("Selected embeddings are indentical", icon="⚠")

    else:
        return None

    st.stop()


@st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=3)
def load_embedding(
    embedding_path: str,
    embedding_format: str,
    frequencies_path: str = None,
    frequencies_format: str = None,
) -> Embedding:
    """Load and cache an embedding

    Args:
        embedding_path (str): embedding path
        embedding_format (str): embeddding format
        frequencies_path (str, optional): frequencies path. Defaults to None.
        frequencies_format (str, optional): frequencies format. Defaults to None.

    Returns:
        Embedding: Loaded Embedding object
    """
    try:
        loading_function = EMBEDDING_FORMATS[embedding_format.lower()]

        return loading_function(
            embedding_path,
            frequencies_path=frequencies_path,
            frequencies_format=frequencies_format,
        )
    except KeyError:
        st.error(
            f"embedding format shloud be one of `{', '.join(EMBEDDING_FORMATS)}` "
            f"but is `{embedding_format}`\n\n"
            f"Could not load {embedding_path}"
        )


@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def create_comparison(
    config_embeddings: dict,
    emb1_id: str,
    emb2_id: str,
    n_neighbors: int,
    max_emb_size: int,
    min_frequency: float = None,
) -> EmbeddingComparison:
    """Load and cache two embeddings and return them in an EmbeddingComparison object

    Args:
        config_embeddings (dict): embeddings configuration dict
        emb1_id (str): first embedding id
        emb2_id (str): second embedding id
        n_neighbors (int): number of neighbors for comparison
        max_emb_size (int): maximum size of the embeddings
        min_frequency (float, optional): minimal frequency for an element to be taken
            into account. Defaults to None.

    Returns:
        EmbeddingComparison: an EmbeddingComparison object based on the two loaded
            embeddings
    """
    embeddings = {}

    for emb_id, col in zip((emb1_id, emb2_id), st.columns(2)):
        emb_infos = config_embeddings[emb_id]

        logger.info(f"Loading {emb_infos['path']}...")

        emb = load_embedding(
            embedding_path=emb_infos["path"],
            embedding_format=emb_infos["format"],
            frequencies_path=emb_infos.get("frequencies", None),
            frequencies_format=emb_infos.get("frequencies_format", None),
        )

        # If min freqency is set and the embedding contains frequencies
        # we filter elements by their frequency
        if min_frequency and emb.is_frequency_set():
            logger.info(f"Filtering frequencies of {emb_id}...")
            emb = emb.filter_by_frequency(min_frequency)

        elif min_frequency:
            with col:
                st.warning(
                    f"Frequencies are not set in this embedding. Min frequency ignored"
                )

        embeddings[emb_id] = emb

    comparison = EmbeddingComparison(embeddings, n_neighbors=n_neighbors)

    # Sample comparison to reduce memory consuption
    comparison = comparison.sampled_comparison(n_samples=max_emb_size)

    # Load embeddings labels if provided and add them to comparison
    comparison.labels = load_embeddings_labels(config_embeddings, emb1_id, emb2_id)

    return comparison


def load_embeddings_labels(
    config_embeddings: dict, emb1_id: str, emb2_id: str
) -> Tuple[dict, dict]:
    """Load a label file

    Args:
        config_embeddings (dict): embedding configuration
        emb1_id (str): first embedding id
        emb2_id (str): second embdding id

    Returns:
        Tuple[dict, dict]: A tuple containing labels of both embeddings
    """
    labels = []
    for emb_id in (emb1_id, emb2_id):
        emb_infos = config_embeddings[emb_id]

        if "labels" not in emb_infos:
            labels.append({})
            continue
        else:
            path = Path(emb_infos["labels"])

        labels_format = emb_infos.get("labels_format", path.suffix[1:])

        labels.append(FREQUENCIES_FORMATS[labels_format](path))

    return labels
