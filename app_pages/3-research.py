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

st.title(dict_lang["3-research"])
# ----------------------------------------------------------------------------------------------------------------------
# Fonctions
# ----------------------------------------------------------------------------------------------------------------------
def find_with_keywords(keywords_list, research_kw=[], how='or'):
    """renvoie un filtre, pour une df d'images, à partir d'une série contenant une liste de mots-clés selon une liste de
     mots clés et d'une expression logique"""
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
    """renvoie une df d'iamges filtrée selon plusieurs listes de mots-clés et d'expressions logiques selons les colonnes"""
    filters = True
    for key in research_dict.keys():
        filters &= df.loc[:, key].apply(lambda x: find_with_keywords(x, **research_dict[key]))

    return df[filters].copy()
# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------


# dictionnaire des tags
with open('data/images/classes.json', 'r') as f:
    dict_tags = json.load(f)
    dict_tags['no tag'] = {"color": "gray", "emoji": "☹️"}


research_dict = {
    'location': {
        'how': None,
        'research_kw': []
    },
    'tag': {
        'how': None,
        'research_kw': []
    },
    'keywords': {
        'how': None,
        'research_kw': []
    }
}

# ----------------------------------------------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------------------------------------------
# chargement des df
if 'df_retrieval' not in state:
    state.df_retrieval = pd.read_json('data/keywords/df_retrieval.json')
    state.df_retrieval.set_index('img_name', inplace=True)

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
        field_cols = st.columns([0.1, 0.9])
        with field_cols[0]:
            research_dict[field_name]['how'] = st.selectbox(
                dict_lang['3-logic'],
                options=['or', 'and'],
                format_func=lambda x: dict_lang[x],
                disabled=field_name == 'location',
                key=f'how_{field_name}'
            )

        with field_cols[1]:
            if field_name == 'tag':
                research_dict[field_name]['research_kw'] = st.multiselect(
                    dict_lang[field_name],
                    set(state.df_retrieval[field_name].sum()),
                    format_func=lambda x: f"{dict_tags[x]['emoji']} {dict_lang[x]}",
                    key=f'research_kw_{field_name}',
                    placeholder=dict_lang["choose_option"]
                )
            elif field_name == 'keywords':
                research_dict[field_name]['research_kw'] = st.multiselect(
                    dict_lang[field_name],
                    set(state.df_retrieval[field_name].sum()),
                    format_func=lambda x: dict_lang[x],
                    key=f'research_kw_{field_name}',
                    placeholder=dict_lang["choose_option"]
                )
            else:
                research_dict[field_name]['research_kw'] = st.multiselect(
                    dict_lang[field_name],
                    set(state.df_retrieval[field_name].sum()),
                    key=f'research_kw_{field_name}',
                    placeholder=dict_lang["choose_option"]
                )

# ----------------------------------------------------------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------------------------------------------------------
selected_images = find_from_dict(state.df_retrieval, research_dict)
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
    img_results = selected_images.loc[Path(image).stem, :]

    col_tags, col_kw = st.columns(2)
    with col_tags:
        # affichage des tags
        img_tags = img_results.tag

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
        img_kws = img_results.keywords
        if img_kws[0] != 'no keywords':
            if img_results.predicted:
                dict_kw_pred = img_results.pred_keywords
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
    st.image(image, caption=f"Image : {img_results.name}")


# détection de fin de chargement de la page
if ("research_init" not in state) or not state.research_init:
    state["home_init"] = False
    state["map_init"] = False
    state["research_init"] = True
    state["pipeline_init"] = False
    # mise à jour de l'url avec la langue (obligé sinon clic en plus requis pour map notament)
    if "selected_lang" in state:
        st.query_params['lang'] = state["selected_lang"]
    else:
        st.query_params['lang'] = "fr"

    st.rerun()