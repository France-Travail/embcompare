import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embcompare import EmbeddingComparison
from loguru import logger
from sklearn.decomposition import PCA


def display_spaces_comparison(comparison: EmbeddingComparison):
    logger.info(f"Computing neighborhoods_similarities_values...")
    neighborhood_sim_values = comparison.neighborhoods_similarities_values

    # Principal Component Analysis visualization
    st.subheader("Principal Component Analysis visualization")
    st.markdown(
        f"""Scatter plots below represent element vectors in each embedding space thanks to
        [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis).

The colors represent element neighborhoods similarities between both embeddings
        """
    )
    for emb, emb_labels, col in zip(
        comparison.embeddings, comparison.labels, st.columns(2)
    ):

        logger.info(f"Computing PCA...")
        emb_pca = PCA(n_components=2).fit_transform(emb.vectors)
        inds = [emb.key_to_index[k] for k in comparison.neighborhoods_similarities]

        df_pca = pd.DataFrame(
            {
                "x": emb_pca[inds, 0],
                "y": emb_pca[inds, 1],
                "sim": neighborhood_sim_values,
                "label": [
                    emb_labels.get(k, k) for k in comparison.neighborhoods_similarities
                ],
            }
        )

        logger.info(f"Displaying vector space...")
        chart = (
            alt.Chart(df_pca)
            .mark_circle(size=60)
            .encode(
                x=alt.X("x", axis=None),
                y=alt.Y("y", axis=None),
                tooltip=["label", "sim"],
                color=alt.Color(
                    "sim",
                    scale=alt.Scale(domain=[0, 1], scheme="redyellowblue"),
                    legend=None,
                ),
            )
            .configure_axis(grid=False)
            .configure_mark(opacity=0.33)
            .configure_view(strokeWidth=0)
            .interactive()
        )

        with col:
            st.altair_chart(chart, use_container_width=True)

    # Add a legend
    chart = (
        alt.Chart(
            pd.DataFrame({"sim": np.round(np.linspace(0, 1, 10), 2), "y": [1] * 10})
        )
        .mark_rect()
        .encode(
            x=alt.X(
                "sim:O", axis=alt.Axis(format="~p", title="similarity", labelAngle=0)
            ),
            y=alt.Y("y:O", axis=None),
            color=alt.Color(
                "sim",
                scale=alt.Scale(domain=[0, 1], scheme="redyellowblue"),
                legend=None,
            ),
        )
    )
    st.altair_chart(chart, use_container_width=True)
