import sys
from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from embsuivi import EmbeddingComparison
from embsuivi.gui.cli import CONFIG_EMBEDDINGS, load_configs
from embsuivi.gui.helpers import (
    AdvancedParameters,
    compute_weighted_median_ordered_similarity,
    compute_weighted_median_similarity,
    display_neighborhoods_elements_comparison,
    load_embedding,
    load_embeddings_labels,
    round_sig,
)
from loguru import logger
from sklearn.decomposition import PCA

logger.remove()
logger.add(sys.stderr, level=st.get_option("logger.level").upper())

# theme from color.adobe.com : #253659 #03A696 #04BF9D #F27457 #BF665E
EMB_COLORS = ("#04BF9D", "#F27457")

st.set_page_config(page_title="Embedding comparison", page_icon="📊")

config = load_configs(*sys.argv[1:])


def main():
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
        emb1_id, emb2_id = embedding_selection()
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

        for emb in comparison.embeddings:
            assert emb.vectors.shape[0] <= advanced_parameters.max_emb_size

    # Display number of element in both embedding and common elements
    with tab_infos:
        for emb, col in zip(comparison.embeddings, st.columns(2)):
            with col:
                st.metric("Number of loaded elements", emb.vectors.shape[0])

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
        compare_neighborhood_similarities(comparison)

    with tab_compare:
        neighborhoods_elements_comparison(comparison)

    with tab_compare_custom:
        custom_elements_comparison(comparison)

    with tab_frequencies:
        compare_frequencies(comparison)

    logger.success("All elements have been displayed")


def embedding_selection() -> Tuple[str, str]:
    available_embeddings = [None] + list(config[CONFIG_EMBEDDINGS])

    if not available_embeddings:
        st.warning(
            "No embeddings.\n\n" "Add embedding with `embsuivi-compare add` command"
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

    # Sample comparison to reduce memory consuption
    comparison = comparison.sampled_comparison(n_samples=max_emb_size)

    # Load embeddings labels if provided and add them to comparison
    comparison.labels = load_embeddings_labels(
        config[CONFIG_EMBEDDINGS], emb1_id, emb2_id
    )

    return comparison


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


def compare_neighborhood_similarities(comparison: EmbeddingComparison):
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


def neighborhoods_elements_comparison(comparison: EmbeddingComparison):
    strategies = {
        "least similar": "least_similar",
        "most similar": "most_similar",
        "random": "random",
    }
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
                disabled=not comparison.is_frequencies_set(),
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

    display_neighborhoods_elements_comparison(comparison, elements)


def custom_elements_comparison(comparison: EmbeddingComparison):
    emb1_labels, _ = comparison.labels

    selected_elements_to_compare = st.multiselect(
        "Select elements to compare",
        comparison.neighborhoods_similarities,
        key="selected_elements_to_compare",
        format_func=emb1_labels.get if emb1_labels else None,
    )
    elements = [
        (e, comparison.neighborhoods_similarities[e])
        for e in selected_elements_to_compare
    ]

    if elements:
        display_neighborhoods_elements_comparison(comparison, elements)


def compare_frequencies(comparison: EmbeddingComparison):
    emb1, emb2 = comparison.embeddings

    if not emb1.is_frequency_set() or not emb2.is_frequency_set():
        return st.warning("Embeddings should both contain frequencies to compare them")

    n_elements_frequencies = st.number_input(
        "Number of elements to displayed",
        min_value=1,
        max_value=len(comparison.common_keys),
        value=100,
        step=20,
        key="n_elements_frequencies",
    )

    # We get minimum strictly positive frequency to smooth ratio computation
    min_freq = min(
        np.min(emb1.frequencies[emb1.frequencies > 0]),
        np.min(emb2.frequencies[emb2.frequencies > 0]),
    )

    if min_freq <= 0.0:
        return st.warning("Frequencies are all equal to zero")

    # Compute ratio difference bewteen common elements of both embeddings
    # the ratio is smoothed by addind minimum frequency to numerator and
    # denominator
    emb1_freqs = np.array([emb1.get_frequency(k) for k in comparison.common_keys])
    emb2_freqs = np.array([emb2.get_frequency(k) for k in comparison.common_keys])

    diff = np.abs(np.log2((emb1_freqs + min_freq) / (emb2_freqs + min_freq)))

    df_freqs = pd.DataFrame(
        {
            "element": comparison.common_keys,
            "freq1": emb1_freqs,
            "freq2": emb2_freqs,
            "diff": diff,
        }
    ).sort_values("diff", ascending=False)

    df_freqs = df_freqs.iloc[0:n_elements_frequencies]

    # Add a column with variation direction
    df_freqs["variation"] = ""
    df_freqs.loc[
        (df_freqs["diff"] >= np.log2(1.1)) & (df_freqs["freq2"] > df_freqs["freq1"]),
        "variation",
    ] = "↗"
    df_freqs.loc[
        (df_freqs["diff"] >= np.log2(1.1)) & (df_freqs["freq2"] < df_freqs["freq1"]),
        "variation",
    ] = "↘"

    # change freqs to string representation
    df_freqs.loc[:, "freq1"] = df_freqs["freq1"].apply(lambda x: f"{round_sig(x):.2g}")
    df_freqs.loc[:, "freq2"] = df_freqs["freq2"].apply(lambda x: f"{round_sig(x):.2g}")

    # Add style to dataframe
    def styler_dataframe(df, columns):
        top_tier = df.loc[(df["diff"] >= np.log2(1.5))]
        two_tier = df.loc[(df["diff"] >= np.log2(1.1)) & (df["diff"] <= np.log2(1.5))]

        df = df.loc[:, columns]

        styler = df.style.set_properties(
            **{"color": "rgb(242, 87, 97)"},
            subset=pd.IndexSlice[top_tier.index, columns],
        )

        styler.set_properties(
            **{"color": "rgb(242,155,87)"},
            subset=pd.IndexSlice[two_tier.index, columns],
        )

        return styler

    st.table(styler_dataframe(df_freqs, ["element", "freq1", "freq2", "variation"]))


if __name__ == "__main__":
    main()
