import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embsuivi import EmbeddingComparison
from loguru import logger


def weighted_median(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, c[-1] / 2)]]


def compute_weighted_median_similarity(comparison: EmbeddingComparison):
    emb1, emb2 = comparison.embeddings

    freqs_1 = np.array(
        [emb1.get_frequency(k) for k in comparison.neighborhoods_similarities]
    )
    freqs_2 = np.array(
        [emb2.get_frequency(k) for k in comparison.neighborhoods_similarities]
    )
    freqs_mean = (freqs_1 + freqs_2) / 2

    return weighted_median(comparison.neighborhoods_similarities_values, freqs_mean)


def compute_weighted_median_ordered_similarity(comparison: EmbeddingComparison):
    emb1, emb2 = comparison.embeddings

    freqs_1 = np.array(
        [emb1.get_frequency(k) for k in comparison.neighborhoods_ordered_similarities]
    )
    freqs_2 = np.array(
        [emb2.get_frequency(k) for k in comparison.neighborhoods_ordered_similarities]
    )
    freqs_mean = (freqs_1 + freqs_2) / 2

    return weighted_median(
        comparison.neighborhoods_ordered_similarities_values, freqs_mean
    )


def display_neighborhoods_similarities(comparison: EmbeddingComparison):
    # Neighborhoods similarities
    logger.info(f"Computing neighborhoods_similarities_values...")
    neighborhood_sim_values = comparison.neighborhoods_similarities_values

    df_sim = pd.DataFrame({"similarity": neighborhood_sim_values})

    logger.info(f"Displaying neighborhoods similarities histogram...")

    st.subheader("Neighborhoods similarities")
    st.altair_chart(
        alt.Chart(df_sim)
        .mark_bar()
        .encode(
            x=alt.X("similarity", bin=alt.Bin(extent=[0, 1], maxbins=10), title=None),
            y=alt.Y("count()", axis=None),
            color=alt.Color(
                "similarity", scale=alt.Scale(scheme="redyellowblue", domain=[0, 1])
            ),
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Median similarity", f"{np.median(neighborhood_sim_values):.1%}")

    if comparison.is_frequencies_set():
        with col2:
            weighted_median = compute_weighted_median_similarity(comparison)
            st.metric("Frequency-weighted median similarity", f"{weighted_median:.1%}")

    # Neighborhoods ordered similarity
    logger.info(f"Computing neighborhoods_ordered_similarities_values...")

    neighborhood_o_sim_values = comparison.neighborhoods_ordered_similarities_values
    df_sim = pd.DataFrame({"similarity": neighborhood_o_sim_values})

    logger.info(f"Displaying ordered neighborhoods similarities histogram...")

    st.subheader("Neighborhoods ordered similarities")
    st.altair_chart(
        alt.Chart(df_sim)
        .mark_bar()
        .encode(
            x=alt.X("similarity", bin=alt.Bin(extent=[0, 1], maxbins=10), title=None),
            y=alt.Y("count()", axis=None),
            color=alt.Color(
                "similarity", scale=alt.Scale(scheme="redyellowblue", domain=[0, 1])
            ),
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Median ordered similarity", f"{np.median(neighborhood_o_sim_values):.1%}"
        )

    if comparison.is_frequencies_set():
        with col2:
            weighted_median = compute_weighted_median_ordered_similarity(comparison)
            st.metric("Frequency-weighted median similarity", f"{weighted_median:.1%}")
