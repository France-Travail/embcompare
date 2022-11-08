import streamlit as st

from .load_utils import EMBEDDING_FORMATS


@st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=3)
def load_embedding(
    embedding_path: str,
    embedding_format: str,
    frequencies_path: str = None,
    frequencies_format: str = None,
):
    try:
        loading_function = EMBEDDING_FORMATS[embedding_format.lower()]

        return loading_function(
            embedding_path,
            frequencies_path=frequencies_path,
            frequencies_format=frequencies_format,
        )
    except KeyError:
        st.error(
            f"embedding format shloud be one of `{', '.join(EMBEDDING_FORMATS)}` "
            f"but is `{embedding_format}`\n\n"
            f"Could not load {embedding_path}"
        )


class AdvancedParameters:
    n_neighbors: int = 25
    max_emb_size: int = 10000
    min_frequency: float = 0.0

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def selection(cls):
        with st.sidebar:
            n_neighbors = st.number_input(
                "Number of neighbors to use in the comparison",
                min_value=1,
                max_value=1000,
                step=10,
                value=cls.n_neighbors,
                key="n_neighbors",
            )

            max_emb_size = st.number_input(
                "Maximum number of elements in the embeddings "
                "(help to reduce memory footprint) :",
                min_value=100,
                max_value=200000,
                step=10000,
                value=cls.max_emb_size,
                key="max_emb_size",
            )

            min_frequency = st.number_input(
                "Minimum freqency for embedding elements :",
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                value=cls.min_frequency,
                format="%f",
                key="min_frequency",
            )

        return cls(
            n_neighbors=n_neighbors,
            max_emb_size=max_emb_size,
            min_frequency=min_frequency,
        )
