import numpy as np
import pandas as pd
import streamlit as st
from embcompare import EmbeddingComparison

from ..helpers import round_sig


def display_frequencies_comparison(comparison: EmbeddingComparison):
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
