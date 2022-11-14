import sys

import streamlit as st
from embcompare.config import CONFIG_EMBEDDINGS, load_configs
from embcompare.gui.features import (
    display_custom_elements_comparison,
    display_elements_comparison,
    display_embeddings_config,
    display_frequencies_comparison,
    display_neighborhoods_similarities,
    display_numbers_of_elements,
    display_parameters_selection,
    display_spaces_comparison,
    display_statistics_comparison,
)
from embcompare.gui.helpers import create_comparison, stop_if_any_embedding_unset
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=st.get_option("logger.level").upper())

st.set_page_config(page_title="Embedding comparison", page_icon="📊")

config = load_configs(*sys.argv[1:])


def main():
    config_embeddings = config[CONFIG_EMBEDDINGS]

    (
        tab_infos,
        tab_stats,
        tab_spaces,
        tab_neighbors,
        tab_compare,
        tab_compare_custom,
        tab_frequencies,
    ) = st.tabs(
        [
            "Infos",
            "Statistics",
            "Spaces",
            "Similarities",
            "Elements",
            "Search elements",
            "Frequencies",
        ]
    )

    # Embedding selection (inside the sidebar)
    with st.sidebar:
        parameters = display_parameters_selection(config_embeddings)

    # Display informations about embeddings
    with tab_infos:
        display_embeddings_config(
            config_embeddings, parameters.emb1_id, parameters.emb2_id
        )

    # If same embedding are selected, stop here
    stop_if_any_embedding_unset(
        config_embeddings, parameters.emb1_id, parameters.emb2_id
    )

    # Otherwise we start comparing embeddings
    comparison = create_comparison(
        config_embeddings,
        emb1_id=parameters.emb1_id,
        emb2_id=parameters.emb2_id,
        n_neighbors=parameters.n_neighbors,
        max_emb_size=parameters.max_emb_size,
        min_frequency=parameters.min_frequency,
    )

    # Display number of element in both embedding and common elements
    with tab_infos:
        display_numbers_of_elements(comparison)

    # Display statistics
    with tab_stats:
        display_statistics_comparison(comparison)

    if not comparison.common_keys:
        st.warning("The embeddings have no element in common")
        st.stop()

    # Comparison below are based on common elements comparison
    with tab_spaces:
        display_spaces_comparison(comparison)

    with tab_neighbors:
        display_neighborhoods_similarities(comparison)

    with tab_compare:
        display_elements_comparison(comparison)

    with tab_compare_custom:
        display_custom_elements_comparison(comparison)

    with tab_frequencies:
        display_frequencies_comparison(comparison)

    logger.success("All elements have been displayed")


if __name__ == "__main__":
    main()
