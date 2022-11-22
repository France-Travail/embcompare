from typing import List, Tuple

import pandas as pd
import streamlit as st
from embcompare import EmbeddingComparison
from loguru import logger


def display_neighborhoods_comparisons(
    comparison: EmbeddingComparison, elements: List[Tuple[str, float]]
):
    """Display comparisons between neighbors of the given list of elements

    Args:
        comparison (EmbeddingComparison): an EmbeddingComparison object
        elements (List[Tuple[str, float]]): The elements to compare
    """
    emb1, emb2 = comparison.embeddings

    if hasattr(comparison, "labels"):
        emb1_labels, emb2_labels = comparison.labels
    else:
        emb1_labels, emb2_labels = ({}, {})

    least_similar_keys, least_similar_sim = list(zip(*elements))

    logger.info("Get first embedding neighbors...")
    neighborhoods_1 = emb1.get_neighbors(
        comparison.n_neighbors, keys=least_similar_keys
    )

    logger.info("Get second embedding neighbors...")
    neighborhoods_2 = emb2.get_neighbors(
        comparison.n_neighbors, keys=least_similar_keys
    )

    logger.info("Display neighbors comparisons...")
    for key, similarity in zip(least_similar_keys, least_similar_sim):

        label = emb1_labels.get(key, key)

        with st.expander(f"{label} (similarity {similarity:.0%})"):
            # Display frequencies
            for emb, col in zip((emb1, emb2), st.columns(2)):
                if emb.is_frequency_set():
                    with col:
                        freq = emb.get_frequency(key)
                        freq_str = f"{freq:.4f}" if freq > 0.0001 else f"{freq:.1e}"

                        st.write(f"term frequency : {freq_str}")

            neighbors1 = {k: s for k, s in neighborhoods_1[key]}
            neighbors2 = {k: s for k, s in neighborhoods_2[key]}

            # Display common neighbors
            common_neighbors = {
                k: i for i, k in enumerate(neighbors1) if k in neighbors2
            }

            if common_neighbors:
                st.subheader("common neighbors")
                st.table(
                    pd.DataFrame(
                        {
                            "neighbor": [
                                emb1_labels.get(k, k) for k in common_neighbors
                            ],
                            "sim1": [f"{neighbors1[k]:.1%}" for k in common_neighbors],
                            "sim2": [f"{neighbors2[k]:.1%}" for k in common_neighbors],
                        }
                    )
                )

            # Display other neighbors
            only1 = {
                k: i for i, k in enumerate(neighbors1) if k not in common_neighbors
            }
            only2 = {
                k: i for i, k in enumerate(neighbors2) if k not in common_neighbors
            }

            if only1:
                st.subheader("distinct neighbors")
                st.table(
                    pd.DataFrame(
                        {
                            "neighbor1": [emb1_labels.get(k, k) for k in only1],
                            "sim1": [f"{neighbors1[k]:.1%}" for k in only1],
                            "sim2": [f"{neighbors2[k]:.1%}" for k in only2],
                            "neighbor2": [emb2_labels.get(k, k) for k in only2],
                        }
                    )
                )


def display_elements_comparison(comparison: EmbeddingComparison):
    """Display a comparison between element neighborhoods

    Args:
        comparison (EmbeddingComparison): an EmbeddingComparison object
    """
    strategies = {
        "least similar": "least_similar",
        "most similar": "most_similar",
        "random": "random",
    }

    is_frequencies_set = comparison.is_frequencies_set()

    st.markdown(
        "Compare neighborhoods of least similar, most similar or random elements."
    )
    if is_frequencies_set:
        st.markdown(
            "You can filter elements that have a too low frequency of occurrence."
        )

    with st.form("neighborhoods_elements_comparison"):
        col1, col2, col3 = st.columns(3)

        with col1:
            strategy_elements_comparison = st.selectbox(
                "Which elements to compare ?", strategies
            )
        with col2:
            n_elements_comparison = st.number_input(
                "Number of elements",
                min_value=1,
                max_value=len(comparison.common_keys),
                value=20,
                step=20,
                key="n_elements_comparison",
            )
        with col3:
            min_frequency_elements = st.number_input(
                "Minimum frequency",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.0001,
                format="%f",
                key="min_frequency_elements",
                disabled=not is_frequencies_set,
            )

        submitted = st.form_submit_button("Change parameters")

        if submitted:
            logger.info(
                f"Selected element comparison parameters : "
                f"strategy_elements_comparison={strategy_elements_comparison}, "
                f"n_elements_comparison={n_elements_comparison}, "
                f"min_frequency_elements={min_frequency_elements}"
            )

    elements = list(
        comparison.neighborhoods_similarities_iterator(
            strategy=strategies[strategy_elements_comparison],
            min_frequency=min_frequency_elements,
            n_elements=n_elements_comparison,
        )
    )

    if not elements:
        st.warning("No elements found")
        return None

    display_neighborhoods_comparisons(comparison, elements)


def display_custom_elements_comparison(comparison: EmbeddingComparison):
    """Display a comparison between element neighborhoods where elements are chosen
    by the user

    Args:
        comparison (EmbeddingComparison): an EmbeddingComparison object
    """
    emb1_labels, _ = comparison.labels

    selected_elements_to_compare = st.multiselect(
        "Select elements to compare",
        comparison.neighborhoods_similarities,
        key="selected_elements_to_compare",
        format_func=emb1_labels.get if emb1_labels else lambda x: x,
    )
    elements = [
        (e, comparison.neighborhoods_similarities[e])
        for e in selected_elements_to_compare
    ]

    if elements:
        display_neighborhoods_comparisons(comparison, elements)
