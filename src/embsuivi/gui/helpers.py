import glob
import os
from argparse import Namespace
from pathlib import Path
from typing import List

import streamlit as st


def find_files(*glob_patterns) -> List[Path]:
    return [
        Path(file)
        for glob_pattern in glob_patterns
        for file in glob.glob(glob_pattern, recursive=True)
        if os.path.isfile(file)
    ]


def load_embeddings(arguments: Namespace):
    embeddings_path = find_files(*arguments.embeddings)
    embeddings_names = arguments.names
    embeddings_format = arguments.formats

    if embeddings_names is None or len(embeddings_names) != len(embeddings_path):
        embeddings_names = [path.stem for path in embeddings_path]

    return {name: emb for name, emb in zip(embeddings_names, embeddings_path)}
