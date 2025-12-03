#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Matthieu PELINGRE
# Created Date: 18/02/2025
# version ='0.1'
# ----------------------------------------------------------------------------------------------------------------------
"""
Script principal du Streamlit de présentation pour ImageEst

__author__ = "Matthieu PELINGRE"
__copyright__ = ""
__credits__ = ["Matthieu PELINGRE", "Antoine TABBONE"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Matthieu PELINGRE"
__email__ = "matthieu.pelingre@univ-lorraine.fr"
__status__ = "early alpha"
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image

#st.set_page_config(layout="wide")
# logos de la sidebar
#st.logo("pictures/logos/IDMC_LOGO_UL-02.png")

state = st.session_state
dict_lang = state.dict_lang[state.selected_lang]
# ----------------------------------------------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------------------------------------------
# 📂 Chemin vers le dossier d'images
IMAGE_FOLDER = Path(r"data/postcards_dataset_1024")  #TODO : possibilité de modifier le chemin du DataSet

# ⚙️ Paramètres
NUM_COLUMNS = 5   # Nombre de colonnes
NUM_LINES = 3   # Nombre de colonnes
RUN_EVERY = 15


# ----------------------------------------------------------------------------------------------------------------------
# Fonctions
# ----------------------------------------------------------------------------------------------------------------------
def is_landscape(pil_img: Image.Image):
    """Renvoie si une image PIL est au format paysage"""
    width, height = pil_img.size
    return width > height


def load_column(folder: Path, col_num, num_images=NUM_LINES):
    """Charge les images d'une colonne à partir d'un dossier."""
    all_images = list(folder.rglob("*.jpg"))
    completed = False  # détermine si la colonne est complète
    img_counter = 0    # compteur d'image
    col_images = []    # liste des images de la colonne

    while not completed:
        if img_counter == 0 and col_num == 0:  # pour que la première image soit celle avec les "boches".
            col_images.append(Image.open("pictures/intro/manuscrit_historique.png"))
            img_counter += 1

        selected_path = np.random.choice(all_images)
        # on ne chargera pas totalement l'image si elle ne correspond pas aux critères

        with Image.open(selected_path) as selected:
            is_land = is_landscape(selected)

        if is_land:                                          # si elle est au format paysage
            col_images.append(Image.open(selected_path))     # ajout
            img_counter += 1                                 # incrémentation de 1
        #else:                                                # si elle est au format portrait
        #    if img_counter + 2 <= num_images:                # elle compte pour deux
        #        col_images.append(Image.open(selected_path)) # ajout s'il reste de la place
        #        img_counter += 2

        if img_counter >= num_images:                        # on vérifie si la colonne est complète
            completed = True

    return col_images


@st.fragment(run_every=RUN_EVERY)
def display_imgs(places):
    # 📊 Affichage dans Streamlit
    for i, place in enumerate(places):
        place.empty()
        with place.container():
            for img in load_column(IMAGE_FOLDER, col_num=i):
                st.image(img)
    return



def get_imgs_by_tags(df, tags):
    """récupération des images en fonction du tag"""
    if isinstance(tags, str):  # tout similaire peu importe l'input
        tags = [tags]
    elif tags is None:
        tags = []

    return df.loc[df['classes'].map(lambda x: all(tag in x for tag in tags))]


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
# 🖼️ Affichage
st.title(f"🖼️ T2IA - {dict_lang['1-project_title']}")

col_names, col_dates = st.columns([0.85, 0.15])
with col_names:
    st.caption("Matthieu PELINGRE & Antoine TABBONE")
with col_dates:
    st.caption("16/05/2025")

with st.container(height=NUM_LINES * 150, border=True):
    columns = st.columns(NUM_COLUMNS)
    placeholders = []
    for col in columns:
        with col:
            placeholder = st.empty()
            placeholders.append(placeholder)


# Publications
with st.container(border=True):
    st.header("Publications")
    st.markdown("""
        - Pelingre, M. & Tabbone, S. (2025). Historical Postcards Date Stamps Content Understanding, IEEE CBMI 2025.
        - Pelingre, M. & Tabbone, S. (2025). Historical postcards classification combining visual content and text description, 23rd ICIAP 2025.
        - Pelingre, M. & Tabbone, S. (2025). Benchmarking OCR Tools for Historical Postcards: A Dataset and Evaluation, 7th SUMAC @ ACM Multimedia 2025. <a href="https://doi.org/10.1145/3746273.3760201">doi: 10.1145/3746273.3760201</a>
    """, unsafe_allow_html=True)
    st.header("Dataset")
    st.markdown("""
        Pelingre, M. & Tabbone, S. (2025). Historical Postcards Dataset, V1, Recherche Data Gouv. <a href="https://doi.org/10.57745/GELGHH">doi: 10.57745/GELGHH</a>
    """, unsafe_allow_html=True)


# Logos
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.image(Image.open("pictures/logos/Logo_Région_Grand_Est_-_2022.png"), use_container_width=True)

with col2:
    st.image("pictures/logos/imageest_noir_360.png", use_container_width=True)

with col3:
    st.image("pictures/logos/formations-universite-de-lorraine-logo-1671626379.jpg", use_container_width=True)

with col4:
    st.image("pictures/logos/IDMC_LOGO_UL-02.png", use_container_width=True)

with col5:
    st.image("pictures/logos/logo_bpi_360.png", use_container_width=True)


# affichages des images
display_imgs(placeholders)


# détection de fin de chargement de la page
if ("home_init" not in state) or not state.home_init:
    state["home_init"] = True
    state["map_init"] = False
    state["research_init"] = False
    state["stamp_init"] = False
    state["pipeline_init"] = False
    # mise à jour de l'url avec la langue (obligé sinon clic en plus requis pour map notament)
    if "selected_lang" in state:
        st.query_params['lang'] = state["selected_lang"]
    else:
        st.query_params['lang'] = "fr"

    st.rerun()