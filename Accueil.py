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
import json


st.set_page_config(layout="wide", page_title="T2IA", page_icon=":material/network_intelligence:")
# logos de la sidebar
st.logo("pictures/logos/IDMC_LOGO_UL-02.png")

state = st.session_state
# ----------------------------------------------------------------------------------------------------------------------
# Languages
# ----------------------------------------------------------------------------------------------------------------------
# Define the available languages
languages = {"fr": {'idx': 0, 'emoji': "🇫🇷"}, "en": {'idx': 1, 'emoji': "🇺🇸"}}

# dictionnaries
if "dict_lang" not in state:
    state.dict_lang = {}
    for file in Path("data/lang").glob("*.json"):
        with open(file.as_posix()) as json_file:
            state.dict_lang[file.stem] = json.load(json_file)

# Get the current query parameters
query_parameters = st.query_params

# Set the default language if not already set
if "lang" not in st.query_params:
    if "selected_lang" in state:
        st.query_params['lang'] = state["selected_lang"]
    else:
        st.query_params['lang'] = "fr"
    st.rerun()


# Define a callback function to set the language
def set_language():
    if "selected_lang" in state:
        st.query_params['lang'] = state["selected_lang"]

# Create a radio button for language selection
with st.sidebar:
    sel_lang = st.radio(
        state.dict_lang[st.query_params['lang']]["language"],
        options=languages,
        index=languages[st.query_params['lang']]['idx'],
        format_func=lambda option: languages[option]['emoji'],
        horizontal=True,
        on_change=set_language,
        key="selected_lang",
    )

# dictionnaire courant
dict_lang = state.dict_lang[state.selected_lang]
# ----------------------------------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------------------------------
page_files = Path('app_pages').glob('*.py')

pages = []
for page_file in sorted(page_files):
    pages.append(st.Page(page_file.as_posix(), title=dict_lang[page_file.stem], default=(page_file.stem=="1-home")))

pg = st.navigation(pages)
pg.run()
