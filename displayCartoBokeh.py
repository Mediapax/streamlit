# displayCartoBokeh

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from bokeh.models import ColumnDataSource, GeoJSONDataSource, HoverTool
from bokeh.plotting import figure, show
from bokeh.models import Panel, Tabs
from bokeh.layouts import row
from datetime import datetime, timezone, timedelta
from sklearn.metrics import confusion_matrix, classification_report
import geopandas
from sklearn.linear_model import LinearRegression, LogisticRegression



def displayCartoBokeh():
    displayBackground("#000000","#FFFFFF")

    # Lecture des données
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS.csv', parse_dates=[0], infer_datetime_format=True)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})
    df['DayOfYear'] = df.Date.dt.day_of_year
    df['Month'] = df.Date.dt.month
    cities_df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/citiesloc.csv', index_col=0)
    df = df.merge(right=cities_df[['city', 'lat', 'lng']], left_on='Location', 
                  right_on='city', how='left').drop(columns='city').rename(columns={'lat':'Latitude', 'lng':'Longitude'})
    
   
    # Affichage d'une carte interactive pour afficher le climat pour chaque ville du jeu de données

    # Définition des climats pour chaque accronyme
    data_climat = {'abv' : ['ET','Dfc','Cfc','Cfb','Cfa','Cwa','Csb','Csa','BSk','BSh','BWk','BWh','Aw','Am','Af'],
            'detail':['Tundra','Subartic','Subpolar oceanic','Oceanic','Humid subtropical - no dry season',
                        'Humid subtropical - summer monsoon and dry winter','Warm-summer mediterranean','Hot-summer mediterranean','Cold semi-arid','Hot semi-arid',
                        'Cold desert','Hot desert','Savanna','Monsoon','Rainforest']}
    type_climat = pd.DataFrame(data=data_climat)

    def label_climat(abv):
        return type_climat[type_climat['abv']==abv]['detail'].iloc[0]

    cities_df['climate_detail'] = cities_df['climate'].apply(lambda x : label_climat(x))

    bokeh_cities_df = ColumnDataSource(cities_df)


    # contenu de l'info bulle qd on passe la souris sur un point
    contenu_info_bulle = [("Station Météo", "@city"), ("Longitude", "@lng"), ("Latitude", "@lat"), ("Climat","@climate"), ('',"@climate_detail")]

    # chgt carte geopandas Australia + lien avec bokeh (via GeoJSONDataSource)
    cdm = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    cdm = cdm[cdm.name == 'Australia']
    geo_source = GeoJSONDataSource(geojson=cdm.to_json())

    # taille de la carte  + zoom sur l'Australie + couleur terre 'peru' + ligne sep. terre/mer en 'black'
    p1 = figure(plot_width=810, plot_height=486)
    p1.patches(fill_color='peru', line_color='black', source=geo_source, name='bkeh_cdm', alpha=0.4)

    # place les points des villes
    ville = p1.circle(x='lng', y='lat', source=bokeh_cities_df, color='red', size=10)

    # chgt de couleur qd la souris passe sur un pt
    chgt_couleur1 = p1.circle(x='lng', y='lat', source=bokeh_cities_df,size=15,
                            alpha=0, # alpha=0 ==> invisible tant que la souris ne va pas sur le point
                            hover_fill_color='cyan', hover_alpha=0.7)

    # Affichage des étiquettes des villes
    etiquetteville1 = HoverTool(tooltips=contenu_info_bulle,
                            mode='mouse',
                            renderers=[chgt_couleur1])
    p1.add_tools(etiquetteville1)

    # Titre onglet
    ongletcarte = Panel(child=p1, title="Carte de l'Australie avec les lieux d'observations météorologiques et les climats associés")

    # liste des onglets
    groupe_onglets1 = Tabs(tabs=[ongletcarte])

    # layout global, en ligne
    layout = row(groupe_onglets1)
    st.bokeh_chart(layout,use_container_width=True)
        


