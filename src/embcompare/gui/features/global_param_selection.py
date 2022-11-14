from collections import namedtuple
from typing import Tuple

import streamlit as st
from loguru import logger

ComparisonParameters = namedtuple(
    "ComparisonParameters",
    ["emb1_id", "emb2_id", "n_neighbors", "max_emb_size", "min_frequency"],
)


def display_embedding_selection(config_embeddings) -> Tuple[str, str]:
    available_embeddings = [None] + list(config_embeddings)

    if not available_embeddings:
        st.warning(
            "No embeddings.\n\n" "Add embedding with `embcompare-compare add` command"
        )
        st.stop()

    selected_embeddings = []

    with st.form("embedding_selection"):
        for i, col in enumerate(st.columns(2)):
            with col:
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


def display_global_parameters_selection():
    with st.form("advanced_parameters_selection"):
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
    emb1_id, emb2_id = display_embedding_selection(config_embeddings)
    n_neighbors, max_emb_size, min_frequency = display_global_parameters_selection()
    st.markdown("""---""")
    st.info("Use shift+wheel or the arrow keys to scroll tabs")

    return ComparisonParameters(
        emb1_id, emb2_id, n_neighbors, max_emb_size, min_frequency
    )
