# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
from PIL import Image, ImageDraw
from pathlib import Path
from streamlit_image_select import image_select
import json

#st.set_page_config(layout="wide")
# logos de la sidebar
#st.logo("pictures/logos/IDMC_LOGO_UL-02.png")

state = st.session_state
dict_lang = state.dict_lang[state.selected_lang]

st.title(dict_lang["4-stamps"])

# CONSTANTES
STAMP_PATH = "data/postmarks/stamps_img"


# ----------------------------------------------------------------------------------------------------------------------
# Fonctions
# ----------------------------------------------------------------------------------------------------------------------
def find_with_keywords(keywords_list, research_kw=[], how='or'):
    """renvoie un filtre, pour une df d'images, à partir d'une série contenant une liste de mots-clés selon une liste de
     mots clés et d'une expression logique"""
    if isinstance(keywords_list, str):
        keywords_list = [keywords_list]
    res = False
    if len(research_kw) == 0:
        res = True
    else:
        if how == 'or':
            res = len(set(keywords_list).intersection(set(research_kw))) > 0
        elif how == 'and':
            res = len(set(keywords_list).intersection(set(research_kw))) == len(research_kw)
    return res


def find_from_dict(df, research_dict):
    """renvoie une df d'images filtrée selon plusieurs listes de mots-clés et d'expressions logiques selons les colonnes"""
    filters = True
    for key in research_dict.keys():
        filters &= df.loc[:, key].apply(lambda x: find_with_keywords(x, **research_dict[key]))

    return df[filters].copy()


def group_years(year):
    """Regroupe les années par décennies"""
    res = "NULL"
    if year != 0:
        res = f"{year:.0f}"[:-1] + '0s'
    return res
# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------


# dictionnaire des tags 
with open('data/images/classes.json', 'r') as f:
    dict_tags = json.load(f)
    dict_tags['no tag'] = {"color": "gray", "emoji": "☹️"}


# TODO à modifier
research_dict = {  # "stamp type", "town_name", "year"
    "stamp type": {
        'how': 'or',
        'research_kw': []
    },
    "town_name": {
        'how': 'or',
        'research_kw': []
    },
    "years": {
        'how': 'or',
        'research_kw': []
    },
}


stamp_cols = ["postal agency", "date", "department", "starred hour", "collection", "stamp type", "quality", "manual"]

# ----------------------------------------------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------------------------------------------
# chargement des df
if 'df_retrieval' not in state:
    state.df_retrieval = pd.read_json('data/keywords/df_retrieval.json')
    state.df_retrieval.set_index('img_name', inplace=True)

if 'df_stamps' not in state:
    # df des tampons
    state.df_stamps = pd.read_csv("data/postmarks/pipeline_YOLO2_gpt4ft150-stamps_reading.csv")
    state.df_stamps.rename(columns={'image': 'img_name'}, inplace=True)
    state.df_stamps = state.df_stamps.merge(state.df_retrieval, how='left', on='img_name')
    # processing des années -> en str
    state.df_stamps.fillna({"year": 0}, inplace=True)
    state.df_stamps["years"] = state.df_stamps.year.apply(group_years)
    # processing des agences
    state.df_stamps.fillna({"town_name": 'NULL'}, inplace=True)
    # processing des types de tampon
    state.df_stamps.fillna({"stamp type": 'NULL'}, inplace=True)
    # noms des tampons
    state.df_stamps.stamp = state.df_stamps.stamp.apply(lambda x: x.replace('_cls1', ''))
    state.df_stamps.set_index('stamp', inplace=True, drop=True)

    state.nb_max_stamps = int(state.df_stamps.img_name.value_counts().max())  # max de tampons par images

    # mise à jour de l'url avec la langue
    if "selected_lang" in state:
        st.query_params['lang'] = state["selected_lang"]
    else:
        st.query_params['lang'] = "fr"

    st.rerun()


# ----------------------------------------------------------------------------------------------------------------------
# Sélection
# ----------------------------------------------------------------------------------------------------------------------

with st.container(border=True):
    for field_name in research_dict.keys():
        if field_name == 'stamp type':
            research_dict[field_name]['research_kw'] = st.multiselect(
                dict_lang[field_name],
                sorted(state.df_stamps[field_name].unique()),
                format_func=lambda x: dict_lang[x],
                key=f'research_kw_{field_name}',
                placeholder=dict_lang["choose_option"]
            )
        else:
            research_dict[field_name]['research_kw'] = st.multiselect(
                dict_lang[field_name],
                sorted(state.df_stamps[field_name].unique()),
                format_func=lambda x: dict_lang[x] if x in dict_lang.keys() else x,
                key=f'research_kw_{field_name}',
                placeholder=dict_lang["choose_option"]
            )

# ----------------------------------------------------------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------------------------------------------------------
selected_images = find_from_dict(state.df_stamps, research_dict).drop_duplicates(subset=["img_name"], keep='first')
selected_images.sort_index(inplace=True)
selected_nb = selected_images.shape[0]
pages_nb = selected_nb // 6 + 1 if selected_nb % 6 != 0 else selected_nb // 6

if selected_nb == 0:
    st.warning(f"⚠️ {dict_lang["3-not_found"]}.")
else:
    with st.container(border=True):
        placeholder = st.empty()
        selected_page = st.number_input(
            "page",
            min_value=1,
            max_value=pages_nb,
            label_visibility='collapsed'
        )
        with placeholder.container():
            image = image_select(
                f"{selected_nb} {dict_lang["3-found"]} ( {selected_page} / {pages_nb} pages )",
                selected_images.img_path.to_list()[6*(selected_page-1):6*selected_page]
            )

    # image sélectionnée
    img_results = state.df_stamps[state.df_stamps.img_name == Path(image).stem]

    col_tags, col_kw = st.columns(2)
    with col_tags:
        # affichage des tags
        img_tags = img_results.iloc[0].tag

        tag_markdown = f"{dict_lang['tags']} "

        if len(img_tags) > 0:
            for tag in img_tags:
                tag_markdown += f":{dict_tags[tag]['color']}-badge[{dict_tags[tag]['emoji']} {dict_lang[tag]}] "
        else:
            tag_markdown += f":gray-badge[☹️ {dict_lang["no tag"]}]"
        st.markdown(tag_markdown.strip())

    with col_kw:
        # affichage des mots-clés
        kws_markdown = f"{dict_lang['keywords']} "
        img_kws = img_results.iloc[0].keywords
        if img_kws[0] != 'no keywords':
            if img_results.iloc[0].predicted:
                dict_kw_pred = img_results.iloc[0].pred_keywords
                if state.selected_lang == 'en':
                    kws_markdown = "Predicted " + kws_markdown
                elif state.selected_lang == 'fr':
                    kws_markdown = kws_markdown[:-2] + "prédits : "
                for kw in img_kws:
                    if dict_kw_pred[kw] < 0.75:
                        kws_markdown += ":red"
                    elif dict_kw_pred[kw] < 0.90:
                        kws_markdown += ":orange"
                    else:
                        kws_markdown += ":green"
                    kws_markdown += f"-badge[{dict_lang[kw]} ({dict_kw_pred[kw]:.0%})] "
            else:
                for kw in img_kws:
                    kws_markdown += f":blue-badge[{dict_lang[kw]}] "
        else:
            kws_markdown += f":gray-badge[{dict_lang["no keywords"]}]"
        st.markdown(kws_markdown.strip())


    # Affichage de l'image sélectionnée
    st.image(image, caption=f"Image : {img_results.iloc[0].img_name}")

# ----------------------------------------------------------------------------------------------------------------------
# Affichage Tampons
# ----------------------------------------------------------------------------------------------------------------------
    nb_stamp_lines = state.nb_max_stamps // 3 + (state.nb_max_stamps % 3 > 0)  # nombre de lignes avec des tampons à afficher

    #liste des colonnes pour les tampons
    st_stamp_cols = []
    for line in range(nb_stamp_lines):
        st_stamp_cols.append(st.columns(3))

    # tampons à afficher
    processed_resp = img_results[stamp_cols].rename(columns=dict_lang)  # renommage des colonnes pour affichage
    # modifications des NULL de stamp type (déjà traduit) pour affichage
    processed_resp[dict_lang["stamp type"]] = processed_resp[dict_lang["stamp type"]].apply(lambda x: None if x == 'NULL' else x)
    # traduction des valeurs à afficher
    processed_resp = processed_resp.map(lambda x: dict_lang[str(x)] if str(x) in dict_lang else x)
    # récupération des tampons sous forme d'une liste de pd.Series
    stamps_list = [row for _, row in processed_resp.iterrows()]

    # affichage
    line_nb = 0
    col_nb = 0
    for i in range(len(stamps_list)):
        stamp_name = stamps_list[i].name.replace('_', '_cls1_')
        st_subcol_img, st_subcol_bin = st_stamp_cols[line_nb][col_nb].columns(2)
        st_subcol_img.image(f"{STAMP_PATH}/images/{stamp_name}.jpg")
        st_subcol_bin.image(f"{STAMP_PATH}/bin/{stamp_name}.jpg")
        st_stamp_cols[line_nb][col_nb].write(stamps_list[i])
        col_nb += 1
        if col_nb == 3:
            line_nb += 1
            col_nb = 0


# ----------------------------------------------------------------------------------------------------------------------
# Fin page
# ----------------------------------------------------------------------------------------------------------------------
# détection de fin de chargement de la page
if ("stamp_init" not in state) or not state.stamp_init:
    state["home_init"] = False
    state["map_init"] = False
    state["research_init"] = False
    state["stamp_init"] = True
    state["pipeline_init"] = False
    # mise à jour de l'url avec la langue (obligé sinon clic en plus requis pour map notament)
    if "selected_lang" in state:
        st.query_params['lang'] = state["selected_lang"]
    else:
        st.query_params['lang'] = "fr"

    st.rerun()