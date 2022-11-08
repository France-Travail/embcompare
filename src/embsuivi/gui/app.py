import sys
from collections import namedtuple
from random import sample
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embsuivi import EmbeddingComparison
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

    with st.sidebar:
        st.info("Use shift+wheel or the arrow keys to scroll tabs")

    # Tabs
    (
        tab_infos,
        tab_stats,
        tab_spaces,
        tab_neighbors,
        tab_least_similar,
        tab_most_similar,
        tab_random,
        tab_custom,
    ) = st.tabs(
        [
            "Infos",
            "Statistics",
            "Vector spaces",
            "Neighborhoods similarities",
            "Least similar",
            "Most similar",
            "Random elements",
            "Custom elements",
        ]
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

    with tab_least_similar:
        display_least_similar(comparison)

    with tab_most_similar:
        display_most_similar(comparison)

    with tab_random:
        display_random(comparison)

    with tab_custom:
        display_custom(comparison)


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

    emb1_dist, _ = emb1.compute_neighborhoods(n_neighbors=comparison.n_neighbors)
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


def compare_spaces(comparison: EmbeddingComparison):
    neighborhood_sim_values = comparison.neighborhoods_similarities_values

    # Principal Component Analysis visualization
    st.subheader("Principal Component Analysis visualization")
    for emb, col in zip(comparison.embeddings, st.columns(2)):

        emb_pca = PCA(n_components=2).fit_transform(emb.vectors)
        inds = [emb.key_to_index[k] for k in comparison.neighborhoods_similarities]

        df_pca = pd.DataFrame(
            {
                "x": emb_pca[inds, 0],
                "y": emb_pca[inds, 1],
                "sim": neighborhood_sim_values,
                "cle": list(comparison.neighborhoods_similarities.keys()),
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
    # Neighborhoods similarities
    neighborhood_sim_values = comparison.neighborhoods_similarities_values
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
    neighborhood_o_sim_values = comparison.neighborhoods_ordered_similarities_values
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


def display_neighborhoods_elements_comparison(
    comparison: EmbeddingComparison, elements: List[Tuple[str, float]]
):
    emb1, emb2 = comparison.embeddings
    least_similar_keys, least_similar_sim = list(zip(*elements))

    neighborhoods_1 = emb1.get_neighbors(
        comparison.n_neighbors, keys=least_similar_keys
    )
    neighborhoods_2 = emb2.get_neighbors(
        comparison.n_neighbors, keys=least_similar_keys
    )

    for key, similarity in zip(least_similar_keys, least_similar_sim):
        with st.expander(f"{key} (similarity {similarity:.0%})"):
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
                            "neighbor": [k for k in common_neighbors],
                            "sim1": [neighbors1[k] for k in common_neighbors],
                            "sim2": [neighbors2[k] for k in common_neighbors],
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
                            "neighbor1": [k for k in only1],
                            "sim1": [neighbors1[k] for k in only1],
                            "sim2": [neighbors2[k] for k in only2],
                            "neighbor2": [k for k in only2],
                        }
                    )
                )


def display_least_similar(comparison: EmbeddingComparison):
    display_neighborhoods_elements_comparison(
        comparison, comparison.get_least_similar(20)
    )


def display_most_similar(comparison: EmbeddingComparison):
    display_neighborhoods_elements_comparison(
        comparison, comparison.get_most_similar(20)
    )


def display_random(comparison: EmbeddingComparison):
    elements = sample(comparison.neighborhoods_similarities.items(), 20)
    elements.sort(key=lambda x: x[1])
    display_neighborhoods_elements_comparison(comparison, elements)


def display_custom(comparison: EmbeddingComparison):
    selected_elements_to_compare = st.multiselect(
        "Select elements to compare",
        comparison.neighborhoods_similarities,
        key="selected_elements_to_compare",
    )
    elements = [
        (e, comparison.neighborhoods_similarities[e])
        for e in selected_elements_to_compare
    ]

    if elements:
        display_neighborhoods_elements_comparison(comparison, elements)


if __name__ == "__main__":
    main()
