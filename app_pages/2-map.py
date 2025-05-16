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
from sklearn.preprocessing import MinMaxScaler
import json

#st.set_page_config(layout="wide")
# logos de la sidebar
#st.logo("pictures/logos/IDMC_LOGO_UL-02.png")

state = st.session_state
dict_lang = state.dict_lang[state.selected_lang]
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
    state['selected_location'] = state.ocr_locations.loc[index, ...]
    state['selected_image_index'] = 1  # Réinitialiser à la première image


def reset_selection():
    state['selected_location'] = None
    state['selected_image_index'] = 0


def adresses_only(list_address):
    res = []
    if len(list_address) > 0:
        for address in list_address:
            res.append(address[0])
    return res


def get_camembert_tagged(camembert_res, word_tag):
    return [ent['word'] for ent in camembert_res if ent['entity_group'] == word_tag]


def update_map_layer():
    """Mise à jour des couleurs en fonction de la sélection"""
    if "ocr_locations" in state:
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=state.ocr_locations,
            id="cities",
            get_position=['longitude', 'latitude'],
            get_color=dict_colors[state['display_color']],
            get_radius='size',
            pickable=True,
        )
        state.map_deck.layers[0] = layer


def update_map_theme():
    """Mise à jour du thème de la carte"""
    if "map_style" in state:
        state.map_deck.map_style = state.map_style



# ajouts pour les tags
def get_img_classes(df, path_classif):
    """renvoie la colonne des classes des images"""
    with open(path_classif, 'r') as f:
        dict_classif = json.load(f)

    return df.img_name.apply(lambda x: dict_classif[x])


def get_imgs_by_tags(df, tags):
    """récupération des images en fonction du tag"""
    if isinstance(tags, str):  # tout similaire peu importe l'input
        tags = [tags]
    elif tags is None:
        tags = []

    return df.loc[df['classes'].map(lambda x: all(tag in x for tag in tags))]


def locations_from_results(ocr_results, df_cities, tags=None):
    df_res = get_imgs_by_tags(ocr_results, tags)[['img_path', 'img_name', 'city_code']].copy()
    df_res.drop_duplicates(subset='img_name', keep='first', inplace=True)
    df_res.dropna(subset=['city_code'], inplace=True)
    df_res[['img_name', 'img_path']] = df_res[['img_name', 'img_path']].map(lambda x: [x])
    df_res['count'] = 1
    df_res = df_res.groupby('city_code').sum()
    df_res.sort_values(by=['count'], ascending=False, inplace=True)

    return df_res.merge(df_cities, on='city_code')


def set_cnt_color(df, quantiles, column=None, cmap='jet'):
    """pour trouver la couleur des points en fonction du nombre de cartes"""
    if (column is None):
        if isinstance(df, pd.Series):
            color_series = df.copy()
        else:
            color_series = df['count'].copy()
    else:
        color_series = df[column].copy()

    n_colors = len(quantiles) + 3
    color_map = mpl.colormaps[cmap]
    colors = color_map(np.linspace(0, 1, n_colors))
    colors = (colors * 255).round(0).astype(int)
    colors[:, 3] = 100  # canal alpha
    colors = colors.tolist()
    for quant, color in zip(quantiles, colors[2:-1]):
        color_series = color_series.apply(lambda x: color if isinstance(x, int) and (x <= quant) else x)

    return color_series


def get_size_n_color(ocr_locations, normalizer, quantiles, a=20000, b=2000):
    col_count = ocr_locations['count']

    # colonne de la taille actuelle
    col_size = normalizer.transform(col_count.values.reshape(-1, 1))
    col_size = pd.Series(col_size.flatten(), name='size')
    col_size = col_size * a + b  # pour l'affichage

    # colonne pour les couleurs
    col_cnt_color = set_cnt_color(col_count, quantiles).rename('cnt_color')

    return pd.concat([col_size, col_cnt_color], axis=1)


def update_lists():
    state['region_list'] = state.ocr_locations[['region_name']].sort_values(by='region_name') \
                                                                                     .drop_duplicates(ignore_index=True)
    state['dep_list'] = state.ocr_locations[['region_name', 'department_name']] \
                                                   .sort_values(by=['region_name', 'department_name']) \
                                                   .drop_duplicates(ignore_index=True)
    state['city_list'] = state.ocr_locations[['region_name', 'department_name', 'city_code']] \
                                                    .sort_values(by=['region_name', 'department_name', 'city_code']) \
                                                    .drop_duplicates(ignore_index=True)


# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
# dictionnaire des couleurs pour la carte
dict_colors = {
    dict_lang["2-density"]: 'cnt_color',
    dict_lang["2-region"]: 'region_color',
    dict_lang["2-department"]: 'department_color',
}

dict_box_color = {
    0: 'red',
    90: 'yellow',
    180: 'blue',
    270: 'green',
}

dict_map_theme = {
    'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json': ':material/light_mode:',
    'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json': ':material/dark_mode:',
}

# dictionnaire des tags
with open('data/images/classes.json', 'r') as f:
    dict_tags = json.load(f)

# ----------------------------------------------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------------------------------------------
# chargement des df
if 'df_cities' not in state:
    df_cities = pd.read_csv('data/ocr/OCR_locations_v3.csv')
    df_cities = df_cities.drop(columns=['img_name', 'img_path', 'count', 'size', 'cnt_color']).map(eval_str_lists)
    state['df_cities'] = df_cities
else:
    df_cities = state['df_cities']

# TODO : possibilité de modifier le chemin du DataSet ?
if 'ocr_results' not in state:
    ocr_results = pd.read_csv('data/ocr/OCR_results_v3.csv')
    ocr_results[
        [col for col in ocr_results.columns if col != 'text']
    ] = ocr_results[[col for col in ocr_results.columns if col != 'text']].map(eval_str_lists)
    ocr_results['bbox'] = ocr_results['bbox'].apply(change_coord)
    ocr_results['bbox_color'] = ocr_results['rotation'].apply(lambda x: dict_box_color[x])
    # classes des images :
    ocr_results['classes'] = get_img_classes(ocr_results, 'data/images/classification.json')
    # sauvegarde
    state.ocr_results = ocr_results
    state.ocr_results_base = ocr_results
else:
    ocr_results = state.ocr_results

if 'ocr_locations' not in state:
    ocr_locations = locations_from_results(state.ocr_results_base, state.df_cities)

    norm = MinMaxScaler().fit(ocr_locations['count'].values.reshape(-1, 1))
    state.normalizer = norm
    quantiles = np.quantile(ocr_locations['count'].unique(), np.linspace(0.1, 1, 10)).tolist()
    state.quantiles = quantiles

    ocr_locations[['size', 'cnt_color']] = get_size_n_color(ocr_locations,
                                                            state.normalizer,
                                                            state.quantiles)

    state.ocr_locations = ocr_locations
else:
    ocr_locations = state.ocr_locations


# mots-clés
if 'df_kw_by_tag' not in state:
    df_kw_by_tag = pd.read_json('data/keywords/keywords_by_class.json')
    state.df_kw_by_tag = df_kw_by_tag
if 'img_keywords' not in state:
    img_keywords = pd.read_json('data/keywords/img_full_keywords_v1.json')
    state.img_keywords = img_keywords


# Initialiser l'état de l'application si non défini
if 'tag_selection' not in state:
    state['tag_selection'] = None
if 'selected_location' not in state:
    state['selected_location'] = None
if 'selected_image_index' not in state:
    state['selected_image_index'] = 0
if 'display_color' not in state:
    state['display_color'] = list(dict_colors.keys())[0]
if 'region_list' not in state:
    state['region_list'] = ocr_locations[['region_name']].sort_values(by='region_name')\
                                                                    .drop_duplicates(ignore_index=True)
if 'dep_list' not in state:
    state['dep_list'] = ocr_locations[['region_name', 'department_name']]\
                                                .sort_values(by=['region_name', 'department_name'])\
                                                .drop_duplicates(ignore_index=True)
if 'city_list' not in state:
    state['city_list'] = ocr_locations[['region_name', 'department_name', 'city_code']]\
                                                    .sort_values(by=['region_name', 'department_name', 'city_code'])\
                                                    .drop_duplicates(ignore_index=True)


# ----------------------------------------------------------------------------------------------------------------------
# Carte
# ----------------------------------------------------------------------------------------------------------------------
# Création de la carte Pydeck
if 'map_deck' not in state:
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=ocr_locations,
        id="cities",
        get_position=['longitude', 'latitude'],
        get_color=dict_colors[state['display_color']],
        get_radius='size',
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=ocr_locations.head(20)['latitude'].mean(),
        longitude=ocr_locations.head(20)['longitude'].mean(),
        controller=True,
        zoom=6)

    state.map_deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='light',
        tooltip={'html': "<b>{city_code}</b><br><b>{count}</b> postcards", 'style': {'backgroundColor': 'steelblue', 'color': 'white'}},
    )

# Affichage de la carte
title_col, map_st_col, color_col = st.columns([0.7, 0.1, 0.2])
with title_col:
    st.title(dict_lang["2-map"])
with map_st_col:
    st.pills(
        "Style de carte :",
        options=dict_map_theme.keys(),
        format_func=lambda option: dict_map_theme[option],
        default='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        key='map_style',
        on_change=update_map_theme,
        label_visibility='hidden',
    )
with color_col:
    st.selectbox(
        dict_lang["2-points_color"],
        dict_colors.keys(),
        key='display_color',
        on_change=update_map_layer,
    )

clicked_data = st.pydeck_chart(
    state.map_deck,
    on_select="rerun",
)  # Récupérer les données du clic

selected_object = clicked_data['selection']['objects']

# Vérification si un point a été cliqué
if len(selected_object) != 0:
    selected_region = selected_object['cities'][0]['region_name']
    selected_department = selected_object['cities'][0]['department_name']
    selected_city = selected_object['cities'][0]['city_code']
    region_index = get_index(selected_region,
                             state.region_list,
                             'region_name')
else:
    selected_region = None
    selected_department = None
    selected_city = None
    region_index = None
    dep_list = state.dep_list
    dep_index = None
    city_list = state.city_list
    city_index = None



# sélection par tag
option_map = {
    tag: f":{dict_tags[tag]['color']}[{dict_tags[tag]['emoji']} {dict_lang[tag]}]" for tag in dict_tags.keys()
}


def on_change_maj_map():
    if "tag_selection" in state:
        tag_selection = state.tag_selection
        ocr_locations = locations_from_results(state.ocr_results_base, state.df_cities, tags=tag_selection)
        ocr_locations[['size', 'cnt_color']] = get_size_n_color(ocr_locations,
                                                                state.normalizer,
                                                                state.quantiles)
        state.ocr_locations = ocr_locations

        update_map_layer()
        reset_selection()
        update_lists()



tag_selection = st.pills(
    dict_lang["tags"],
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
    on_change=on_change_maj_map,
    key="tag_selection"
)



# ----------------------------------------------------------------------------------------------------------------------
# Sélection
# ----------------------------------------------------------------------------------------------------------------------
# Sélection d'un lieu
# 3 colonnes : région/département/ville → la couleur devra changer à chaque sélection
st.subheader(dict_lang["2-select_location"])
col_region, col_dep, col_city = st.columns(3)

with col_region:
    selected_reg = st.selectbox(
        f"{dict_lang['2-region']}:",
        state.region_list['region_name'],
        index=region_index,
        placeholder=dict_lang["choose_option"]
    )
    if selected_reg is not None:
        dep_list = state.dep_list[state.dep_list['region_name'] == selected_reg].reset_index(drop=True)
        if selected_department is not None:
            dep_index = get_index(selected_department,
                                  dep_list,
                                  'department_name')
        city_list = state.city_list[state.city_list['region_name'] == selected_reg].reset_index(drop=True)
        if selected_city is not None:
            city_index = get_index(selected_city,
                                   city_list,
                                   'city_code')

with col_dep:
    selected_dep = st.selectbox(
        f"{dict_lang['2-department']}:",
        dep_list['department_name'],
        index=dep_index,
        placeholder=dict_lang["choose_option"]
    )
    if selected_dep is not None:
        city_list = state.city_list[state.city_list['department_name'] == selected_dep].reset_index(drop=True)
        if selected_city is not None:
            city_index = get_index(selected_city,
                                   city_list,
                                   'city_code')

with col_city:
    selected_point = st.selectbox(
        f"{dict_lang['2-town']}:",
        city_list['city_code'],
        index=city_index,
        placeholder=dict_lang["choose_option"]
    )

if selected_point is not None:
    ocr_locations = locations_from_results(
        state.ocr_results_base,
        state.df_cities,
        tags=state.tag_selection
    )

    update_selection(
            ocr_locations[
                ocr_locations['city_code'] == selected_point
            ].index[0]
        )



# ----------------------------------------------------------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------------------------------------------------------
# Affichage des images du lieu sélectionné
if state['selected_location'] is not None:
    loc = state['selected_location']
    ocr_results = get_imgs_by_tags(state.ocr_results_base, tag_selection)
    state.ocr_results = ocr_results
    images = loc['img_path']

    st.subheader(loc['city_code'])

    # Sélecteur d'image avec un slider
    if len(images) > 1:
        image_index = st.slider(
            dict_lang['choose_image'],
            min_value=1,
            max_value=len(images),
            value=state['selected_image_index']
        )
    else:
        state['selected_image_index'] = 1
        image_index = 1


    # Mise à jour de l'image sélectionnée dans l'état de session
    state['selected_image_index'] = image_index

    # caractéristiques de l'image
    img_path = images[image_index - 1]
    image = Image.open(img_path)
    img_name = loc['img_name'][image_index - 1]
    img_results = ocr_results[ocr_results.img_name == img_name].copy()

    tab_img, tab_ocr, tab_bert = st.tabs(["Image", "OCR", "CamemBERT"])

    with tab_img:
        col_tags, col_kw = st.columns(2)
        with col_tags:
            # affichage des tags
            img_tags = img_results.classes.values[0]

            tag_markdown = f"{dict_lang['tags']} "

            if len(img_tags) > 0:
                for tag in img_tags:
                    tag_markdown += f":{dict_tags[tag]['color']}-badge[{dict_tags[tag]['emoji']} {dict_lang[tag]}] "
            else:
                tag_markdown += f":gray-badge[☹️ {dict_lang['no tag']}]"
            st.markdown(tag_markdown.strip())

        with col_kw:
            # affichage des mots-clés
            kws_markdown = f"{dict_lang['keywords']} "
            if img_name in state.img_keywords.index:
                img_kws = state.img_keywords.loc[img_name]

                if img_kws.predicted:
                    dict_kw_pred = img_kws.pred_keywords
                    if state.selected_lang == 'en':
                        kws_markdown = "Predicted " + kws_markdown
                    elif state.selected_lang == 'fr':
                        kws_markdown = kws_markdown[:-2] + "prédits : "
                    for kw in img_kws.keywords:
                        if dict_kw_pred[kw] < 0.75:
                            kws_markdown += ":red"
                        elif dict_kw_pred[kw] < 0.90:
                            kws_markdown += ":orange"
                        else:
                            kws_markdown += ":green"
                        kws_markdown += f"-badge[{kw} ({dict_kw_pred[kw]:.0%})] "
                else:
                    for kw in img_kws.keywords:
                        kws_markdown += f":blue-badge[{kw}] "
            else:
                kws_markdown += ":gray-badge[no keyword]"
            st.markdown(kws_markdown.strip())


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
                        st.caption(f"<p style='text-align:right;'><br>({dict_lang["2-confidence"]} {conf*100:.0f} %)</p>", unsafe_allow_html=True)

    with tab_bert:
        img_results['address'] = img_results['affilgood_address'].apply(adresses_only)

        st.dataframe(
            img_results[['text', 'camembert_loc', 'address']]\
                       .rename(columns={
                'text': dict_lang['text'],
                'camembert_loc': dict_lang['camembert_loc'],
                'address': dict_lang['address']
            }),
            hide_index=True,
            use_container_width=True,
            column_config = {
                dict_lang['text']: st.column_config.Column(dict_lang['text'], width="medium"),
                dict_lang['camembert_loc']: st.column_config.Column(dict_lang['camembert_loc'], width="small"),
                dict_lang['address']: st.column_config.Column(dict_lang['address'], width="small")
            }
        )

        # autres détections de CamemBERT
        other_tags = ['ORG', 'PER', 'MISC']

        for tag in other_tags:
            img_results[tag] = img_results['camembert'].apply(lambda x: get_camembert_tagged(x, tag))

        with st.expander(f"{dict_lang['PER']} {dict_lang['and']} {dict_lang['ORG'].lower()}", expanded=False):
            st.dataframe(
                img_results[['text', 'ORG', 'PER']] \
                    .rename(columns={
                    'text': dict_lang['text'],
                    'PER': dict_lang['PER'],
                    'ORG': dict_lang['ORG'],
                }),
                hide_index=True,
                use_container_width=True,
                column_config = {
                    dict_lang['text']: st.column_config.Column(dict_lang['text'], width="medium"),
                    dict_lang['PER']: st.column_config.Column(dict_lang['PER'], width="small"),
                    dict_lang['ORG']: st.column_config.Column(dict_lang['ORG'], width="small")
                }
            )

        with st.expander(f"{dict_lang['camembert_date']} {dict_lang['and']} {dict_lang['MISC'].lower()}", expanded=False):
            st.dataframe(
                img_results[['text', 'camembert_date', 'MISC']] \
                    .rename(columns={
                    'text': dict_lang['text'],
                    'camembert_date' : dict_lang['camembert_date'],
                    'MISC': dict_lang['MISC']
                }),
                hide_index=True,
                use_container_width=True,
                column_config = {
                    dict_lang['text']: st.column_config.Column(dict_lang['text'], width="medium"),
                    dict_lang['camembert_date']: st.column_config.Column(dict_lang['camembert_date'], width="small"),
                    dict_lang['MISC']: st.column_config.Column(dict_lang['MISC'], width="small")
                }
            )
