import logging
import os
import sys
from pathlib import Path
from random import sample

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from embsuivi.embeddings_comparator import EmbeddingsComparator
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Comp2vec - Comparaison embeddings", page_icon="👩‍🔧")

ENV_DATA_FOLDER = os.environ.get("EMBSUIVI_DATA", "embsuivi-data")
DIR_DATA = Path(ENV_DATA_FOLDER)
DIR_EMBEDDINGS = DIR_DATA / "embeddings"

COLORMAP = plt.cm.get_cmap("RdYlBu_r")
COLOR_BLUE = "#193C40"
COLOR_ORANGE = "#D96941"

DEFAULT_MAX_EMB_SIZE = 10000

st.title("Comparaison générale de deux embeddings")


def create_hist(
    array: np.ndarray,
    bins,
    color_bins: bool = False,
    reverse: bool = True,
    xmin: float = None,
    xmax: float = None,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(16, 6))

    if xmin is not None and xmax is not None:
        kwargs["range"] = (xmin, xmax)

    _, bins_lim, patches = ax.hist(array, bins=bins, **kwargs)

    if color_bins:
        bin_centers = 0.5 * (bins_lim[:-1] + bins_lim[1:])
        xmin = min(bin_centers) if xmin is None else xmin
        xmax = max(bin_centers) if xmax is None else xmax

        colors = (bin_centers - xmin) / (xmax - xmin)

        for c, p in zip(colors, patches):
            plt.setp(p, "facecolor", COLORMAP(1 - c if reverse else c))

    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    return fig, ax


embeddings_dispo = {
    "1.0.0": DIR_EMBEDDINGS / "lite" / "1.0.0" / "1.0.0_lite.bin",
    "1.0.1": DIR_EMBEDDINGS / "lite" / "1.0.1" / "1.0.1_lite.bin",
    "frwiki": (
        DIR_EMBEDDINGS
        / "fat"
        / "frwiki"
        / "frWiki_no_phrase_no_postag_1000_skip_cut100.bin"
    ),
}
embeddings_dispo_load_method = {
    "1.0.0": load_facebook_vectors,
    "1.0.1": load_facebook_vectors,
    "frwiki": lambda p: KeyedVectors.load_word2vec_format(p, binary=True),
}
embeddings_dispo_keys = list(embeddings_dispo)

# ---------------------
# Chargement embeddings
# ---------------------
col1, col2 = st.columns(2)

with col1:
    emb1_name = st.selectbox(
        label="Premier embedding",
        options=embeddings_dispo_keys,
        index=0,
        key="emb1_name",
    )

with col2:
    emb2_name = st.selectbox(
        label="Second embedding",
        options=embeddings_dispo_keys,
        index=len(embeddings_dispo_keys) - 1,
        key="emb2_name",
    )

# On initialise le nombre de voisins différement en fonction du type d'embedding
n_neighbors = st.slider(
    "Nombre de voisins :",
    min_value=1,
    max_value=100,
    value=25,
    key="n_neighbors",
)

# On effectue un sampling s'il y a trop de clés dans les embeddings
with st.expander("Paramètres avancés"):
    col1, col2 = st.columns(2)
    with col1:
        max_emb_size = st.number_input(
            "Taille d'embedding à partir de laquelle effectuer un sous-échantillonage :",
            min_value=100,
            max_value=200000,
            step=10000,
            value=DEFAULT_MAX_EMB_SIZE,
            key="max_emb_size",
        )
    with col2:
        st.warning(
            f"Le temps de calcul des voisinages augmente avec le carré du nombre d'élements. "
            f"{DEFAULT_MAX_EMB_SIZE} éléments semble être un bon compromis."
        )


@st.cache
def initiate_comparator(n_neighbors, emb1_name, emb2_name):
    emb1_load_method = embeddings_dispo_load_method[emb1_name]
    emb1_path = embeddings_dispo[emb1_name]
    logger.info(f"Chargement de l'embedding {emb1_name} ({emb1_path})")
    emb1 = emb1_load_method(emb1_path)

    emb2_load_method = embeddings_dispo_load_method[emb2_name]
    emb2_path = embeddings_dispo[emb2_name]
    logger.info(f"Chargement de l'embedding {emb2_name} ({emb2_path})")
    emb2 = emb2_load_method(emb2_path)

    logger.info(f"Sampling des embedding")
    emb1, emb2 = EmbeddingsComparator.sample_embeddings(
        emb1, emb2, n_samples=max_emb_size
    )

    logger.info(f"Initialisation comparator")
    comparator = EmbeddingsComparator(
        {emb1_name: emb1, emb2_name: emb2}, n_neighbors=n_neighbors
    )

    # pré-calcul des similarités
    logger.info(f"Calcul des voisinages")
    comparator.compute_neighborhood_similiraties()

    logger.info(f"Calcul des ordres")
    comparator.compute_neighborhood_order_similiraties()

    return comparator


with st.spinner("Chargement des embeddings..."):
    comparator = initiate_comparator(int(n_neighbors), emb1_name, emb2_name)

# ---------------------
# Affichage des caractéristiques des embeddings
# ---------------------
st.header("Comparaison des caractéristiques des embeddings")

# Calcul des statistiques de chaque embedding
bins_hist = 20
stats_embeddings = {}

for emb_name in (emb1_name, emb2_name):
    with st.spinner("Analyse de l'embedding"):
        mean_dist_distrib = comparator.compute_mean_distance_to_neighbors(
            emb_name, distribution=True
        )
        mean_min_dist_distrib = comparator.compute_mean_min_distance_to_neighbors(
            emb_name, distribution=True
        )
        mean_dist = np.mean(mean_dist_distrib)
        mean_min_dist = np.mean(mean_min_dist_distrib)

        stats_embeddings[emb_name] = {
            "mean_dist": mean_dist,
            "mean_min_dist": mean_min_dist,
            "mean_dist_distrib": mean_dist_distrib,
            "mean_min_dist_distrib": mean_min_dist_distrib,
        }

        # Valeurs extremales
        y, x = np.histogram(mean_dist_distrib, bins=bins_hist)
        x_max, y_max = stats_embeddings.get("mean_dist_lim", (0, 0))
        stats_embeddings["mean_dist_lim"] = (max(x_max, x.max()), max(y_max, y.max()))

        y, x = np.histogram(mean_min_dist_distrib, bins=bins_hist)
        x_max, y_max = stats_embeddings.get("mean_min_dist_lim", (0, 0))
        stats_embeddings["mean_min_dist_lim"] = (
            max(x_max, x.max()),
            max(y_max, y.max()),
        )

# Affichage des métriques et histogrames
st.markdown(f"### Distances moyennes aux voisins")
col1, col2 = st.columns(2)

for col, emb_name in ((col1, emb1_name), (col2, emb2_name)):
    with col:
        stats_emb = stats_embeddings[emb_name]

        mean_dist = stats_emb["mean_dist"]
        mean_dist_distrib = stats_emb["mean_dist_distrib"]

        # Histograme distances moyennes
        x_max, y_max = stats_embeddings["mean_dist_lim"]

        color = COLOR_BLUE if col == col1 else COLOR_ORANGE
        fig, ax = create_hist(
            mean_dist_distrib, bins=bins_hist, color=color, xmin=0, xmax=x_max
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        st.pyplot(fig)

        # Médiane
        st.metric(
            label="Médiane",
            value=f"{mean_dist:.2}",
        )

st.markdown(f"### Distances au premier voisin")
col1, col2 = st.columns(2)

for col, emb_name in ((col1, emb1_name), (col2, emb2_name)):
    with col:
        stats_emb = stats_embeddings[emb_name]

        mean_min_dist = stats_emb["mean_min_dist"]
        mean_min_dist_distrib = stats_emb["mean_min_dist_distrib"]

        # Histograme distances minimales
        x_max, y_max = stats_embeddings["mean_min_dist_lim"]

        color = COLOR_BLUE if col == col1 else COLOR_ORANGE
        fig, ax = create_hist(
            mean_min_dist_distrib, bins=bins_hist, color=color, xmin=0, xmax=x_max
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        st.pyplot(fig)

        # Médiane distances minimales
        st.metric(
            label="Médiane",
            value=f"{mean_min_dist:.2}",
        )


# ---------------------
# Affichage de la similarité des embeddings
# ---------------------

st.header("Similarité entre les deux embeddings")

# PCA
def compute_pca(emb_name: str):
    emb = comparator[emb_name]
    pca = PCA(n_components=2)
    return pca.fit_transform(emb.vectors)


with st.spinner("Analyse des similarités..."):
    neighborhood_sim = comparator.get_neighborhood_similarity(emb1_name, emb2_name)
    neighborhood_sim_array = np.array(list(neighborhood_sim.values()))
    neighborhood_sim_median = np.median(neighborhood_sim_array)

    emb1_pca = compute_pca(emb1_name)
    emb2_pca = compute_pca(emb2_name)

# PCA
st.markdown(f"### Visualisation des embeddings via PCA")

col1, col2 = st.columns(2)
for col, emb_name, pca in ((col1, emb1_name, emb1_pca), (col2, emb2_name, emb2_pca)):

    emb = comparator[emb_name]
    inds = [emb.key_to_index[k] for k in neighborhood_sim]
    colors = [COLORMAP(1 - s) for s in neighborhood_sim_array]

    df_pca = pd.DataFrame(
        {
            "x": pca[inds, 0],
            "y": pca[inds, 1],
            "sim": neighborhood_sim_array,
            "cle": list(neighborhood_sim.keys()),
        }
    )

    with col:
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
        st.altair_chart(chart)


# Similariés
st.markdown(f"### Distribution de la similarité des voisins")
fig, ax = create_hist(neighborhood_sim_array, bins=10, color_bins=True, xmin=0, xmax=1)
ax.set_xlim(0, 1)
st.pyplot(fig)
st.metric(
    label="Similarité médiane des voisins",
    value=f"{neighborhood_sim_median:.2}",
)

# similarité des ordres des voisins
with st.spinner("Analyse des similarités ordonnées..."):
    neighborhood_order_sim = comparator.get_neighborhood_order_similarity(
        emb1_name, emb2_name
    )
    neighborhood_order_sim_array = np.array(list(neighborhood_order_sim.values()))
    neighborhood_order_sim_median = np.median(neighborhood_order_sim_array)

st.markdown(f"### Distribution de la similarité des ordres des voisins")
fig, ax = create_hist(
    neighborhood_order_sim_array, bins=10, color_bins=True, xmin=0, xmax=1
)
ax.set_xlim(0, 1)
st.pyplot(fig)
st.metric(
    label="Similarité médiane de l'ordre des voisins",
    value=f"{neighborhood_order_sim_median:.2}",
)


# ---------------------
# Affichage des similarités entre éléments
# ---------------------


def afficher_similarites(similarites_elements: dict):
    """Affichage des similarités et différences entre des éléments de deux embeddings"""
    for key, comparison in similarites_elements.items():
        similarite = len(comparison["common"]) / sum(map(len, comparison.values()))

        with st.expander(f"{key} (similarité {similarite:.0%})"):

            # Voisins communs
            st.markdown("##### Voisins communs :")
            if comparison["common"]:
                st.markdown("\n".join(["- " + key for key in comparison["common"]]))
            else:
                st.markdown("Aucun voisin commun")

            # Voisins restants
            st.markdown("##### Comparaison des voisins restants : ")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    "\n".join(
                        [f"{i+1}. {key}" for i, key in enumerate(comparison[emb1_name])]
                    )
                )
            with col2:
                st.markdown(
                    "\n".join(
                        [f"{i+1}. {key}" for i, key in enumerate(comparison[emb2_name])]
                    )
                )


NB_DEFAULT = 20
common_keys = comparator.get_common_keys([emb1_name, emb2_name])

# Affichage des compétences très différents
st.subheader("Elements dont les voisinages sont les plus différents")

nb_least_similar = st.slider(
    "Nombre d'éléments à afficher", 5, 100, NB_DEFAULT, key="nb_least_similar"
)

least_similar = comparator.get_least_similar_comparison(
    emb1_name, emb2_name, topn=nb_least_similar
)

afficher_similarites(least_similar)

# Affichage des compétences très similaires
st.subheader("Elements dont les voisinages sont les plus proches")

nb_most_similar = st.slider(
    "Nombre d'éléments à afficher", 5, 100, NB_DEFAULT, key="nb_most_similar"
)

most_similar = comparator.get_most_similar_comparison(
    emb1_name, emb2_name, topn=nb_most_similar
)

afficher_similarites(most_similar)

# Affichage d'éléments au hasard

st.subheader("Elements aléatoires")

nb_random = st.slider(
    "Nombre d'éléments à afficher", 5, 100, NB_DEFAULT, key="nb_random"
)

keys_random = sorted(sample(common_keys, nb_random), key=neighborhood_sim.get)
afficher_similarites(
    comparator.get_neighborhood_comparison(emb1_name, emb2_name, keys_random)
)

# Affichage des similarités pour les éléments les plus courants

st.subheader("Elements les plus courants")

nb_most_frequent = st.slider(
    "Nombre d'éléments à afficher", 5, 100, NB_DEFAULT, key="nb_most_frequent"
)

keys_most_frequent = sorted(common_keys[:nb_most_frequent], key=neighborhood_sim.get)
afficher_similarites(
    comparator.get_neighborhood_comparison(emb1_name, emb2_name, keys_most_frequent)
)
