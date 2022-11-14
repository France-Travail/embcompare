from collections import namedtuple
from typing import Iterable, Tuple

import streamlit as st
from loguru import logger

# Name tuple containing all parameters that the app needs:
#
# - First embedding config id
# - Second embedding config id
# - Number of neighbors
# - Number of elements to use for comparison
# - Minimum frequency of occurrence for elements
#
ComparisonParameters = namedtuple(
    "ComparisonParameters",
    ["emb1_id", "emb2_id", "n_neighbors", "max_emb_size", "min_frequency"],
)


def display_embedding_selection(config_embeddings: Iterable) -> Tuple[str, str]:
    """Display a form for embedding selection

    Args:
        config_embeddings (Iterable): A python iterable containing embeddings ids

    Returns:
        Tuple[str, str]: a tuple containing selected embeddings
    """
    available_embeddings = [None] + list(config_embeddings)

    if not available_embeddings:
        st.warning(
            "No embeddings found in configuration file.\n\n"
            "Add an embedding with `embcompare add` command"
        )
        st.stop()

    selected_embeddings = []

    st.header("Select your embeddings : ")
    with st.form("embedding_selection"):
        for i in range(2):
            selected_embeddings.append(
                st.selectbox(
                    label=f"{'First' if i == 0 else 'Second'} embedding",
                    options=available_embeddings,
                    index=0,
                    key=f"emb{i}_id",
                )
            )
        submitted = st.form_submit_button("Compare embeddings", type="primary")

        if submitted:
            logger.info(
                f"Selected embeddings : {selected_embeddings[0]} and {selected_embeddings[1]}"
            )

    return tuple(selected_embeddings)


def display_global_parameters_selection() -> Tuple[int, int, float]:
    """Display a form for global parameters selection

    Global parameters :
    - Number of neighbors
    - Number of elements to use for comparison
    - Minimum frequency of occurrence for elements

    Returns:
        Tuple[int, int, float]: n_neighbors, max_emb_size, min_frequency
    """
    with st.form("advanced_parameters_selection"):
        st.subheader("Global parameters")

        n_neighbors = st.number_input(
            "Number of neighbors to use in the comparison",
            min_value=1,
            max_value=1000,
            step=10,
            value=25,
            key="n_neighbors",
        )

        max_emb_size = st.number_input(
            "Maximum number of elements in the embeddings "
            "(help to reduce memory footprint) :",
            min_value=100,
            max_value=200000,
            step=10000,
            value=15000,
            key="max_emb_size",
        )

        min_frequency = st.number_input(
            "Minimum freqency for embedding elements :",
            min_value=0.0,
            max_value=1.0,
            step=0.0001,
            value=0.0,
            format="%f",
            key="min_frequency",
        )

        submitted = st.form_submit_button("Change parameters")

        if submitted:
            logger.info(
                f"n_neighbors={n_neighbors}, "
                f"max_emb_size={max_emb_size}, "
                f"min_frequency={min_frequency}"
            )

    return n_neighbors, max_emb_size, min_frequency


def display_parameters_selection(config_embeddings) -> ComparisonParameters:
    """Display forms for parameters selection

    Returns:
        ComparisonParameters: emb1_id, emb2_id, n_neighbors, max_emb_size, min_frequency
    """
    emb1_id, emb2_id = display_embedding_selection(config_embeddings)

    st.markdown("""---""")

    n_neighbors, max_emb_size, min_frequency = display_global_parameters_selection()

    st.markdown("""---""")

    # Indications for use of tab scroll
    st.info("Use shift+wheel or the arrow keys to scroll tabs")

    return ComparisonParameters(
        emb1_id, emb2_id, n_neighbors, max_emb_size, min_frequency
    )
