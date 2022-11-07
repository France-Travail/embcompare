import streamlit as st

from .load_utils import EMBEDDING_FORMATS


@st.cache(suppress_st_warning=True)
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
