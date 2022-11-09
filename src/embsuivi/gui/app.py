import sys
from random import sample
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embsuivi import EmbeddingComparison
from embsuivi.gui.cli import CONFIG_EMBEDDINGS, load_configs
from embsuivi.gui.helpers import AdvancedParameters, load_embedding, round_sig
from loguru import logger
from sklearn.decomposition import PCA

logger.remove()
logger.add(sys.stderr, level=st.get_option("logger.level").upper())

# theme from color.adobe.com : #253659 #03A696 #04BF9D #F27457 #BF665E
EMB_COLORS = ("#04BF9D", "#F27457")

st.set_page_config(page_title="Embedding comparison", page_icon="📊")

config = load_configs(*sys.argv[1:])


def main():
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

    # Embedding selection (inside the sidebar)
    with st.sidebar:
        emb1_id, emb2_id = embedding_selection()
        st.markdown("""---""")
        advanced_parameters = AdvancedParameters.selection()
        st.markdown("""---""")
        st.info("Use shift+wheel or the arrow keys to scroll tabs")

    # Stop if one or the other embedding is not set
    stop = False
    for emb_id, which in zip((emb1_id, emb2_id), ("first", "second")):
        if emb_id not in config[CONFIG_EMBEDDINGS]:
            st.warning(f"{which} embedding : {emb_id} not in yaml config", icon="⚠")
            stop = True

    if stop:
        st.stop()

    # Display informations about embeddings
    with tab_infos:
        embedding_infos(emb1_id, emb2_id)

    # If same embedding are selected, stop here
    if emb1_id == emb2_id:
        st.warning("Selected embeddings are indentical", icon="⚠")
        st.stop()

    # Otherwise we start comparing embeddings
    else:
        comparison = create_comparison(
            emb1_id,
            emb2_id,
            advanced_parameters.n_neighbors,
            advanced_parameters.max_emb_size,
            min_frequency=advanced_parameters.min_frequency,
        )

    # Display number of element in both embedding and common elements
    with tab_infos:
        for emb, col in zip(comparison.embeddings, st.columns(2)):
            with col:
                st.metric("Number of elements", emb.vectors.shape[0])

        st.markdown("""---""")
        st.metric("Number of common elements", len(comparison.common_keys))

    # Display statistics
    with tab_stats:
        statistics_comparison(comparison)

    if not comparison.common_keys:
        st.warning("The embeddings have no element in common")
        st.stop()

    # Comparison below are based on common elements comparison
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

    logger.success("All elements have been displayed")


def embedding_selection() -> Tuple[str, str]:
    available_embeddings = [None] + list(config[CONFIG_EMBEDDINGS])

    if not available_embeddings:
        st.warning(
            "No embeddings.\n\n" "Add embedding with `embsuivi-compare add` command"
        )
        st.stop()

    selected_embeddings = []

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

    logger.info(
        f"Selected embeddings : {selected_embeddings[0]} and {selected_embeddings[1]}"
    )

    return selected_embeddings


def embedding_infos(emb1_id: str, emb2_id: str):
    for emb_id, col in zip((emb1_id, emb2_id), st.columns(2)):
        emb_infos = config[CONFIG_EMBEDDINGS][emb_id]
        with col:
            st.header(emb_infos["name"])
            st.json(dict(emb_infos))


@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def create_comparison(
    emb1_id: str,
    emb2_id: str,
    n_neigbhors: int,
    max_emb_size: int,
    min_frequency: float = None,
) -> EmbeddingComparison:
    embeddings = {}

    for emb_id, col in zip((emb1_id, emb2_id), st.columns(2)):
        emb_infos = config[CONFIG_EMBEDDINGS][emb_id]

        logger.info(f"Loading {emb_infos['path']}...")

        emb = load_embedding(
            embedding_path=emb_infos["path"],
            embedding_format=emb_infos["format"],
            frequencies_path=emb_infos.get("frequencies", None),
            frequencies_format=emb_infos.get("frequencies_format", None),
        )

        # If min freqency is set and the embedding contains frequencies
        # we filter elements by their frequency
        if min_frequency and emb.is_frequency_set():
            logger.info(f"Filtering frequencies of {emb_id}...")
            emb = emb.filter_by_frequency(min_frequency)

        elif min_frequency:
            with col:
                st.warning(
                    f"Frequencies are not set in this embedding. Min frequency ignored"
                )

        embeddings[emb_id] = emb

    comparison = EmbeddingComparison(embeddings, n_neighbors=n_neigbhors)

    return comparison.sampled_comparison(n_samples=max_emb_size)


def statistics_comparison(comparison: EmbeddingComparison):
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


def compare_spaces(comparison: EmbeddingComparison):
    logger.info(f"Computing neighborhoods_similarities_values...")
    neighborhood_sim_values = comparison.neighborhoods_similarities_values

    # Principal Component Analysis visualization
    st.subheader("Principal Component Analysis visualization")
    for emb, col in zip(comparison.embeddings, st.columns(2)):

        logger.info(f"Computing PCA...")
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

        logger.info(f"Displaying vector space...")
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


def display_neighborhood_similarities(comparison: EmbeddingComparison):
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

    st.metric("Median similarity", f"{np.median(neighborhood_sim_values):.1%}")

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

    st.metric(
        "Median ordered similarity", f"{np.median(neighborhood_o_sim_values):.1%}"
    )


def display_neighborhoods_elements_comparison(
    comparison: EmbeddingComparison, elements: List[Tuple[str, float]]
):
    emb1, emb2 = comparison.embeddings
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
        with st.expander(f"{key} (similarity {similarity:.0%})"):
            # Display frequencies
            for emb, col in zip((emb1, emb2), st.columns(2)):
                if emb.is_frequency_set():
                    with col:
                        freq = emb.get_frequency(key)
                        freq_str = f"{freq:.4f}" if freq > 0.0001 else f"{freq:.1e}"

                        f"term ferquency : {freq_str}"

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
