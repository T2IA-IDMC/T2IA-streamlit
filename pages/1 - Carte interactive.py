# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib as mpl
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu


# logos de la sidebar
st.logo("pictures/logos/IDMC_LOGO_UL-02.png")

# ----------------------------------------------------------------------------------------------------------------------
# Fonctions
# ----------------------------------------------------------------------------------------------------------------------
def eval_str_lists(raw_val):
    res = raw_val
    if isinstance(raw_val, str) and (raw_val[0] == '[' and raw_val[-1] == ']'):
        res = eval(raw_val)
    return res


def set_col_color(df, column, cmap='jet'):
    # valeurs uniques de la colonne
    uniques = df[column].unique()
    # nombre de couleurs
    n_colors = df[column].unique().shape[0]
    # définition de la cmap
    color_map = mpl.colormaps[cmap]
    colors = color_map(np.linspace(0, 1, n_colors))
    colors = (colors * 255).round(0).astype(int)
    colors[:, 3] = 80  # canal alpha

    color_dict = {val: color.tolist() for val, color in zip(uniques, colors)}

    return df[column].map(color_dict.get)


def get_index(val, df, column):
    res = None
    if val in df[column].unique():
        res = int(df[df[column] == val].index[0])
    return res


def change_coord(bbox):
    """Modifie l'ordre des coordonnées des bbox"""
    x_min, x_max, y_min, y_max = bbox
    return [x_min, y_min, x_max, y_max]


# Fonction de mise à jour lorsqu'un point est cliqué
def update_selection(index):
    st.session_state['selected_location'] = ocr_locations.loc[index, ...]
    st.session_state['selected_image_index'] = 1  # Réinitialiser à la première image


def adresses_only(list_address):
    res = []
    if len(list_address) > 0:
        for address in list_address:
            res.append(address[0])
    return res


def get_camembert_tagged(camembert_res, word_tag):
    return [ent['word'] for ent in camembert_res if ent['entity_group'] == word_tag]


# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
# dictionnaire des couleurs pour la carte
dict_colors = {
    'Densités': 'cnt_color',
    'Régions': 'region_color',
    'Départements': 'department_color',
}

dict_box_color = {
    0: 'red',
    90: 'yellow',
    180: 'blue',
    270: 'green',
}


# ----------------------------------------------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------------------------------------------
# chargement des df
# TODO : possibilité de modifier le chemin du DataSet
if 'ocr_results' not in st.session_state:
    ocr_results = pd.read_csv('data/ocr/OCR_results_v2.csv')
    ocr_results[
        [col for col in ocr_results.columns if col != 'text']
    ] = ocr_results[[col for col in ocr_results.columns if col != 'text']].map(eval_str_lists)
    ocr_results['bbox'] = ocr_results['bbox'].apply(change_coord)
    ocr_results['bbox_color'] = ocr_results['rotation'].apply(lambda x: dict_box_color[x])
    st.session_state.ocr_results = ocr_results
else:
    ocr_results = st.session_state.ocr_results

if 'ocr_locations' not in st.session_state:
    ocr_locations = pd.read_csv('data/ocr/OCR_locations_v2.csv').map(eval_str_lists)
    ocr_locations['size'] = ocr_locations['size'] * 20000 + 2000
    st.session_state.ocr_locations = ocr_locations
else:
    ocr_locations = st.session_state.ocr_locations


# Initialiser l'état de l'application si non défini
if 'selected_location' not in st.session_state:
    st.session_state['selected_location'] = None
if 'selected_image_index' not in st.session_state:
    st.session_state['selected_image_index'] = 0
if 'display_color' not in st.session_state:
    st.session_state['display_color'] = list(dict_colors.keys())[0]
if 'map_style' not in st.session_state:
    st.session_state['map_style'] = 'Light'
if 'region_list' not in st.session_state:
    st.session_state['region_list'] = ocr_locations[['region_name']].sort_values(by='region_name')\
                                                                    .drop_duplicates(ignore_index=True)
if 'dep_list' not in st.session_state:
    st.session_state['dep_list'] = ocr_locations[['region_name', 'department_name']]\
                                                .sort_values(by=['region_name', 'department_name'])\
                                                .drop_duplicates(ignore_index=True)
if 'city_list' not in st.session_state:
    st.session_state['city_list'] = ocr_locations[['region_name', 'department_name', 'city_code']]\
                                                    .sort_values(by=['region_name', 'department_name', 'city_code'])\
                                                    .drop_duplicates(ignore_index=True)


# ----------------------------------------------------------------------------------------------------------------------
# Carte
# ----------------------------------------------------------------------------------------------------------------------
# Création de la carte Pydeck
layer = pdk.Layer(
    'ScatterplotLayer',
    data=ocr_locations,
    id="cities",
    get_position=['longitude', 'latitude'],
    get_color=dict_colors[st.session_state['display_color']],
    get_radius='size',
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=ocr_locations['latitude'].mean(),
    longitude=ocr_locations['longitude'].mean(),
    controller=True,
    zoom=6)

map_deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style=st.session_state['map_style'].lower(),
    tooltip={'html': "<b>{city_code}</b><br><b>{count}</b> postcards", 'style': {'backgroundColor': 'steelblue', 'color': 'white'}},
)

# Affichage de la carte
title_col, map_st_col, color_col = st.columns([0.6, 0.2, 0.2])
with title_col:
    st.title("Carte interactive")
with map_st_col:
    st.selectbox(
        "Style de carte :",
        ['Light', 'Dark'],
        key='map_style',
    )
with color_col:
    st.selectbox(
        "Couleur des points :",
        dict_colors.keys(),
        key='display_color',
    )

clicked_data = st.pydeck_chart(
    map_deck,
    on_select="rerun",
)  # Récupérer les données du clic

selected_object = clicked_data['selection']['objects']

# Vérification si un point a été cliqué
if len(selected_object) != 0:
    selected_region = selected_object['cities'][0]['region_name']
    selected_department = selected_object['cities'][0]['department_name']
    selected_city = selected_object['cities'][0]['city_code']
    region_index = get_index(selected_region,
                             st.session_state.region_list,
                             'region_name')
else:
    selected_region = None
    selected_department = None
    selected_city = None
    region_index = None
    dep_list = st.session_state.dep_list
    dep_index = None
    city_list = st.session_state.city_list
    city_index = None


# ----------------------------------------------------------------------------------------------------------------------
# Sélection
# ----------------------------------------------------------------------------------------------------------------------
# Sélection d'un lieu
# 3 colonnes : région/département/ville → la couleur devra changer à chaque sélection
st.subheader("Sélectionnez un lieu :")
col_region, col_dep, col_city = st.columns(3)

with col_region:
    selected_reg = st.selectbox(
        "Région :",
        st.session_state.region_list['region_name'],
        index=region_index,
    )
    if selected_reg is not None:
        dep_list = st.session_state.dep_list[st.session_state.dep_list['region_name'] == selected_reg].reset_index(drop=True)
        if selected_department is not None:
            dep_index = get_index(selected_department,
                                  dep_list,
                                  'department_name')
        city_list = st.session_state.city_list[st.session_state.city_list['region_name'] == selected_reg].reset_index(drop=True)
        if selected_city is not None:
            city_index = get_index(selected_city,
                                   city_list,
                                   'city_code')

with col_dep:
    selected_dep = st.selectbox(
        "Département :",
        dep_list['department_name'],
        index=dep_index,
    )
    if selected_dep is not None:
        city_list = st.session_state.city_list[st.session_state.city_list['department_name'] == selected_dep].reset_index(drop=True)
        if selected_city is not None:
            city_index = get_index(selected_city,
                                   city_list,
                                   'city_code')

with col_city:
    selected_point = st.selectbox(
        "Ville :",
        city_list['city_code'],
        index=city_index,
    )

if selected_point is not None:
    update_selection(
            ocr_locations[ocr_locations['city_code'] == selected_point].index[0]
        )


# ----------------------------------------------------------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------------------------------------------------------
# Affichage des images du lieu sélectionné
if st.session_state['selected_location'] is not None:
    loc = st.session_state['selected_location']
    images = loc['img_path']

    st.subheader(loc['city_code'])

    # Sélecteur d'image avec un slider
    if len(images) > 1:
        image_index = st.slider(
            "Choisissez une image",
            min_value=1,
            max_value=len(images),
            value=st.session_state['selected_image_index']
        )
    else:
        st.session_state['selected_image_index'] = 1
        image_index = 1


    # Mise à jour de l'image sélectionnée dans l'état de session
    st.session_state['selected_image_index'] = image_index

    # caractéristiques de l'image
    img_path = images[image_index - 1]
    image = Image.open(img_path)
    img_name = loc['img_name'][image_index - 1]
    img_results = ocr_results[ocr_results.img_name == img_name].copy()

    tab_img, tab_ocr, tab_bert = st.tabs(["Image", "OCR", "CamemBERT"])

    with tab_img:
        # Affichage de l'image sélectionnée
        st.image(image, caption=f"Image : {img_name}")

    with tab_ocr:

        # Dessiner les bounding boxes sur l'image
        def draw_bounding_boxes(img, image_df):
            img_with_boxes = img.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            for _, row in image_df[['bbox_color', 'bbox']].iterrows():
                color, bbox = row
                draw.rectangle(bbox, outline=color, width=2)  # 10 pour img originales
            return img_with_boxes


        # Fonction pour vérifier si un clic est dans une bounding box
        def get_clicked_box(x, y, image_df):
            res = None
            for idx, row in image_df[['bbox']].iterrows():
                x_min, y_min, x_max, y_max = row.iloc[0]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    res = idx
            return res


        # Affichage de l'image avec les bounding boxes
        image_with_boxes = draw_bounding_boxes(image, img_results)

        # Zone interactive avec st_canvas
        value = streamlit_image_coordinates(image_with_boxes, use_column_width=True)

        # Gestion du clic sur l'image
        if value is not None:
            width, height = image.size
            click_x, click_y = value['x'] * width / value['width'], value['y'] * height / value['height']
            box_idx = get_clicked_box(click_x, click_y, img_results)
            if box_idx is not None:
                clicked_box = img_results.loc[box_idx, ...]
                cropped_image = image.crop((clicked_box['bbox'])).rotate(clicked_box['rotation'], expand=True)
                st.image(cropped_image)

                # affichage des résultats d'OCR pour la bbox
                with st.container(border=True):
                    text, conf = clicked_box[['text','confidence']]

                    col_text, col_conf = st.columns([0.80, 0.20])
                    with col_text:
                        st.markdown(f"<p style='font-size:32px;text-align:center;'>{text}</p>", unsafe_allow_html=True)
                    with col_conf:
                        st.caption(f"<p style='text-align:right;'><br>(Confiance : {conf*100:.0f} %)</p>", unsafe_allow_html=True)

    with tab_bert:
        img_results['address'] = img_results['affilgood_address'].apply(adresses_only)

        st.dataframe(
            img_results[['text', 'camembert_loc', 'address']]\
                       .rename(columns={
                'text': 'Résultats d\'OCR',
                'camembert_loc': 'Lieux détectés',
                'address': 'Adresses détectées'
            }),
            hide_index=True,
            use_container_width=True,
            column_config = {
                "Résultats d\'OCR": st.column_config.Column("Résultats d\'OCR", width="medium"),
                "Lieux détectés": st.column_config.Column("Lieux détectés", width="small"),
                "Adresses détectées": st.column_config.Column("Adresses détectées", width="small")
            }
        )

        # autres détections de CamemBERT
        other_tags = ['ORG', 'PER', 'MISC']

        for tag in other_tags:
            img_results[tag] = img_results['camembert'].apply(lambda x: get_camembert_tagged(x, tag))

        with st.expander("Personnes et Organisations", expanded=False):
            st.dataframe(
                img_results[['text', 'ORG', 'PER']] \
                    .rename(columns={
                    'text': 'Résultats d\'OCR',
                    'PER': 'Personnes',
                    'ORG': 'Organisations',
                }),
                hide_index=True,
                use_container_width=True,
                column_config = {
                    "Résultats d\'OCR": st.column_config.Column("Résultats d\'OCR", width="medium"),
                    "Personnes": st.column_config.Column("Personnes", width="small"),
                    "Organisations": st.column_config.Column("Organisations", width="small")
                }
            )

        with st.expander("Dates et autres détections", expanded=False):
            st.dataframe(
                img_results[['text', 'camembert_date', 'MISC']] \
                    .rename(columns={
                    'text': 'Résultats d\'OCR',
                    'camembert_date' : 'Dates détectées',
                    'MISC': 'Autres détections'
                }),
                hide_index=True,
                use_container_width=True,
                column_config = {
                    "Résultats d\'OCR": st.column_config.Column("Résultats d\'OCR", width="medium"),
                    "Dates détectées": st.column_config.Column("Dates détectées", width="small"),
                    "Autres détections": st.column_config.Column("Autres détections", width="small")
                }
            )
