import streamlit as st
from embcompare import EmbeddingComparison


def display_embeddings_config(config_embeddings: dict, emb1_id: str, emb2_id: str):
    for emb_id, col in zip((emb1_id, emb2_id), st.columns(2)):
        if emb_id in config_embeddings:
            emb_infos = config_embeddings[emb_id]
            with col:
                st.header(emb_infos.get("name", emb_id))
                st.json(dict(emb_infos))


def display_numbers_of_elements(comparison: EmbeddingComparison):
    for emb, col in zip(comparison.embeddings, st.columns(2)):
        with col:
            st.metric("Number of loaded elements", emb.vectors.shape[0])

    st.markdown("""---""")
    st.metric("Number of common elements", len(comparison.common_keys))
