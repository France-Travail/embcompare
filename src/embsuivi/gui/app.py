import sys
from collections import namedtuple
from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embsuivi import EmbeddingComparison, EmbeddingComparisonReport, EmbeddingReport
from embsuivi.gui.cli import CONFIG_EMBEDDINGS, load_configs
from embsuivi.gui.helpers import load_embedding
from sklearn.decomposition import PCA

AdvancedParameters = namedtuple("AdvancedParameters", ["n_neighbors", "max_emb_size"])
EMB_COLORS = ("#04BF9D", "#F27457")

st.set_page_config(page_title="Embedding comparison", page_icon="📊")

config = load_configs(*sys.argv[1:])


def main():
    # Embedding selection (inside the sidebar)
    emb1_id, emb2_id = embedding_selection()
    advanced_parameters = get_advanced_parameters()

    # Tabs
    (tab_infos, tab_stats, tab_spaces, tab_neighbors) = st.tabs(
        ["Infos", "Statistics", "Vector spaces", "Neighborhoods similarities"]
    )

    # Display informations about embeddings
    with tab_infos:
        embedding_infos(emb1_id, emb2_id)

    # If same embedding are selected, stop here
    if emb1_id == emb2_id:
        st.warning("Selected embeddings are indentical", icon="⚠")
        st.stop()

    comparison = create_comparison(
        emb1_id,
        emb2_id,
        advanced_parameters.n_neighbors,
        advanced_parameters.max_emb_size,
    )

    # Display statistics
    with tab_stats:
        statistics_comparison(comparison)

    # Display spaces
    with tab_spaces:
        compare_spaces(comparison)

    with tab_neighbors:
        display_neighborhood_similarities(comparison)


def embedding_selection() -> Tuple[str, str]:
    available_embeddings = list(config[CONFIG_EMBEDDINGS])

    if not available_embeddings:
        st.warning(
            "No embeddings.\n\n" "Add embedding with `embsuivi-compare add` command"
        )
        st.stop()

    with st.sidebar:
        col1, col2 = st.columns(2)

        with col1:
            emb1_id = st.selectbox(
                label="First embedding",
                options=available_embeddings,
                index=0,
                key="emb1_id",
            )

        with col2:
            emb2_id = st.selectbox(
                label="Second embedding",
                options=available_embeddings,
                index=len(available_embeddings) - 1,
                key="emb2_id",
            )

    return emb1_id, emb2_id


def get_advanced_parameters() -> AdvancedParameters:
    with st.sidebar:
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
            value=10000,
            key="max_emb_size",
        )

    return AdvancedParameters(n_neighbors, max_emb_size)


def embedding_infos(emb1_id: str, emb2_id: str):
    emb1_infos = config[CONFIG_EMBEDDINGS][emb1_id]
    emb2_infos = config[CONFIG_EMBEDDINGS][emb2_id]

    for emb_info, col in zip((emb1_infos, emb2_infos), st.columns(2)):
        with col:
            st.header(emb_info["name"])
            st.json(dict(emb_info))


def create_comparison(
    emb1_id: str, emb2_id: str, n_neigbhors: int, max_emb_size: int
) -> EmbeddingComparison:
    emb1_infos = config[CONFIG_EMBEDDINGS][emb1_id]
    emb2_infos = config[CONFIG_EMBEDDINGS][emb2_id]

    emb1 = load_embedding(
        embedding_path=emb1_infos["path"],
        embedding_format=emb1_infos["format"],
        frequencies_path=emb1_infos.get("frequencies", None),
        frequencies_format=emb1_infos.get("frequencies_format", None),
    )

    emb2 = load_embedding(
        embedding_path=emb2_infos["path"],
        embedding_format=emb2_infos["format"],
        frequencies_path=emb2_infos.get("frequencies", None),
        frequencies_format=emb2_infos.get("frequencies_format", None),
    )

    comparison = EmbeddingComparison(
        {emb1_id: emb1, emb2_id: emb2}, n_neighbors=n_neigbhors
    )

    return comparison.sampled_comparison(n_samples=max_emb_size)


def statistics_comparison(comparison: EmbeddingComparison):
    emb1, emb2 = comparison.embeddings

    emb1_report = EmbeddingReport(emb1, comparison.n_neighbors)
    emb2_report = EmbeddingReport(emb2, comparison.n_neighbors)

    emb1_df = pd.DataFrame(
        {
            "mean_dist": np.mean(emb1_report.nearest_neighbors_distances, axis=1),
            "mean_first_dist": emb1_report.nearest_neighbors_distances[:, 0],
        }
    )
    emb2_df = pd.DataFrame(
        {
            "mean_dist": np.mean(emb2_report.nearest_neighbors_distances, axis=1),
            "mean_first_dist": emb2_report.nearest_neighbors_distances[:, 0],
        }
    )

    # Mean distances to neighbors
    st.subheader("Mean distances to neighbors")

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
            st.metric("median", f"{emb_df['mean_dist'].median():.1e}")

    # Mean distances to nearest neighbor
    st.subheader("Mean distances to nearest neighbor")

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
            st.metric("median", f"{emb_df['mean_first_dist'].median():.1e}")


def altair_histogram(
    df: pd.DataFrame,
    col: str,
    extent: list = None,
    color: str = None,
    maxbins: int = 20,
) -> alt.Chart:
    if extent is not None:
        bin = alt.Bin(extent=extent, maxbins=maxbins)
    else:
        bin = True

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(col, bin=bin, title=None),
            y=alt.Y("count()", axis=None),
            color=alt.value(color),
        )
    )


def compare_spaces(comparison: EmbeddingComparison):
    report = EmbeddingComparisonReport(comparison)

    neighborhood_sim_values = list(report.neighborhoods_similarities.values())

    # Principal Component Analysis visualization
    st.subheader("Principal Component Analysis visualization")
    for emb, col in zip(comparison.embeddings, st.columns(2)):

        emb_pca = PCA(n_components=2).fit_transform(emb.vectors)
        inds = [emb.key_to_index[k] for k in report.neighborhoods_similarities]

        df_pca = pd.DataFrame(
            {
                "x": emb_pca[inds, 0],
                "y": emb_pca[inds, 1],
                "sim": neighborhood_sim_values,
                "cle": list(report.neighborhoods_similarities.keys()),
            }
        )

        chart = (
            alt.Chart(df_pca)
            .mark_circle(size=60)
            .encode(
                x=alt.X("x", axis=None),
                y=alt.Y("y", axis=None),
                tooltip=["cle", "sim"],
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


def display_neighborhood_similarities(comparison: EmbeddingComparison):
    report = EmbeddingComparisonReport(comparison)

    # Neighborhoods similarities
    neighborhood_sim_values = list(report.neighborhoods_similarities.values())
    df_sim = pd.DataFrame({"similarity": neighborhood_sim_values})

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

    st.metric("Median similarity", f"{np.median(neighborhood_sim_values):.1%}")

    # Neighborhoods ordered similarity
    neighborhood_o_sim_values = list(report.neighborhoods_ordered_smiliarities.values())
    df_sim = pd.DataFrame({"similarity": neighborhood_o_sim_values})

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

    st.metric(
        "Median ordered similarity", f"{np.median(neighborhood_o_sim_values):.1%}"
    )


if __name__ == "__main__":
    main()
