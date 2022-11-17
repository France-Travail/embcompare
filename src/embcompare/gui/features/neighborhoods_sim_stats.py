import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embcompare import EmbeddingComparison
from loguru import logger


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median

    See the link below for a definition of the weighted median :
    https://en.wikipedia.org/wiki/Weighted_median

    Args:
        values (np.ndarray): values
        weights (np.ndarray): weights

    Returns:
        float: median value
    """
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, c[-1] / 2)]]


def compute_weighted_median_similarity(comparison: EmbeddingComparison) -> float:
    """Compute frequency weighted similarity median of an embedding comparison

    Args:
        comparison (EmbeddingComparison): embedding comparison

    Returns:
        float: frequency weighted similarity median
    """
    emb1, emb2 = comparison.embeddings

    freqs_1 = np.array(
        [emb1.get_frequency(k) for k in comparison.neighborhoods_similarities]
    )
    freqs_2 = np.array(
        [emb2.get_frequency(k) for k in comparison.neighborhoods_similarities]
    )
    freqs_mean = (freqs_1 + freqs_2) / 2

    return weighted_median(comparison.neighborhoods_similarities_values, freqs_mean)


def compute_weighted_median_ordered_similarity(
    comparison: EmbeddingComparison,
) -> float:
    """Compute frequency weighted ordered similarity median of an embedding comparison

    Args:
        comparison (EmbeddingComparison): embedding comparison

    Returns:
        float: frequency weighted ordered similarity median
    """
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
    """Display neighborhoods similarities between two embeddings

    Args:
        comparison (EmbeddingComparison): embedding comparison
    """
    # Neighborhoods ordered similarity
    logger.info(f"Computing neighborhoods_ordered_similarities_values...")

    neighborhood_o_sim_values = comparison.neighborhoods_ordered_similarities_values
    df_sim = pd.DataFrame({"similarity": neighborhood_o_sim_values})

    logger.info(f"Displaying ordered neighborhoods similarities histogram...")

    st.subheader("Neighborhoods ordered similarities")

    with st.expander("📘 Further explanations"):
        st.markdown(
            f"""The ordered neighborhood similarity is a measure of how similar the nearest 
neighbors of an element are in the two embeddings.

Considering the $n$ nearest neighbors $a_{{1..n}} = (a_1, a_2, ..., a_n)$ and $b_{{1..n}} = (b_1, b_2, ..., b_n)$
of a same element in two different embeddings $A$ and $B$, we define the neighborhood ordered similiraty as follow : 

$$S(a_{{1..n}}, b_{{1..n}}) = \\frac{{1}}{{n}} \sum_{{k=1}}^{{n}} \\left( \\frac{{|a_{{1..k}} \\cap b_{{1..k}}|}}{{k}} \\right)$$

When the similarity is 1, the element has the same nearest neighbors in both embeddings and in the same order.

Thus, the median similarity tells how identical are the neighborhoods of the two embeddings while taking into account 
the order of neighbors.

> _Reminder: we use the {comparison.n_neighbors} nearest neighbors to compute the similarities_
"""
        )

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

    # Neighborhoods similarities
    logger.info(f"Computing neighborhoods_similarities_values...")
    neighborhood_sim_values = comparison.neighborhoods_similarities_values

    df_sim = pd.DataFrame({"similarity": neighborhood_sim_values})

    logger.info(f"Displaying neighborhoods similarities histogram...")

    st.subheader("Neighborhoods similarities")

    with st.expander("📘 Further explanations"):
        st.markdown(
            f"""We define neighborhood similiraty as the [IoU](https://en.wikipedia.org/wiki/Jaccard_index)
between nearest neighbors (i.e. neighborhoods) of a same element in each embedding.

When the similarity is 1, the element has the same neighbors in both embeddings (regardless of how close 
they are to the element).

The median similarity tells how identical are the neighborhoods of the two embeddings.

> _Reminder: we use the {comparison.n_neighbors} nearest neighbors to compute the similarities_
"""
        )
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
