import streamlit as st
from embsuivi.gui.cli import parse_args
from embsuivi.gui.helpers import load_embeddings

arguments = parse_args()
embeddings = load_embeddings(arguments)

st.title("hello")

embeddings
