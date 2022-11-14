import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embcompare import EmbeddingComparison
from loguru import logger

from ..helpers import round_sig

# theme from color.adobe.com : #253659 #03A696 #04BF9D #F27457 #BF665E
EMB_COLORS = ("#04BF9D", "#F27457")


def display_statistics_comparison(comparison: EmbeddingComparison):
    emb1, emb2 = comparison.embeddings

    logger.info(f"Computing first embedding neighborhoods...")
    emb1_dist, _ = emb1.compute_neighborhoods(n_neighbors=comparison.n_neighbors)

    logger.info(f"Computing second embedding neighborhoods...")
    emb2_dist, _ = emb2.compute_neighborhoods(n_neighbors=comparison.n_neighbors)

    emb1_df = pd.DataFrame(
        {"mean_dist": np.mean(emb1_dist, axis=1), "mean_first_dist": emb1_dist[:, 0]}
    )
    emb2_df = pd.DataFrame(
        {
            "mean_dist": np.mean(emb2_dist, axis=1),
            "mean_first_dist": emb2_dist[:, 0],
        }
    )

    # Mean distances to neighbors
    st.subheader("Mean distances to neighbors")
    logger.info(f"Displaying mean distances to neighbors...")

    min_mean_dist = min(emb1_df["mean_dist"].min(), emb2_df["mean_dist"].min())
    max_mean_dist = max(emb2_df["mean_dist"].max(), emb2_df["mean_dist"].max())

    for emb_df, col, color in zip((emb1_df, emb2_df), st.columns(2), EMB_COLORS):
        with col:
            st.altair_chart(
                alt.Chart(emb_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "mean_dist",
                        bin=alt.Bin(extent=[min_mean_dist, max_mean_dist], maxbins=20),
                        title=None,
                    ),
                    y=alt.Y("count()", axis=None),
                    color=alt.value(color),
                ),
                use_container_width=True,
            )
            median = round_sig(emb_df["mean_dist"].median(), n_digits=2)
            st.metric("median", median)

    # Mean distances to nearest neighbor
    st.subheader("Mean distances to nearest neighbor")
    logger.info(f"Displaying mean distances to nearest neighbor...")

    min_mean_dist = min(
        emb1_df["mean_first_dist"].min(), emb2_df["mean_first_dist"].min()
    )
    max_mean_dist = max(
        emb2_df["mean_first_dist"].max(), emb2_df["mean_first_dist"].max()
    )

    for emb_df, col, color in zip((emb1_df, emb2_df), st.columns(2), EMB_COLORS):
        with col:
            st.altair_chart(
                alt.Chart(emb_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "mean_first_dist",
                        bin=alt.Bin(extent=[min_mean_dist, max_mean_dist], maxbins=20),
                        title=None,
                    ),
                    y=alt.Y("count()", axis=None),
                    color=alt.value(color),
                ),
                use_container_width=True,
            )
            median = round_sig(emb_df["mean_first_dist"].median(), n_digits=2)
            st.metric("median", median)
