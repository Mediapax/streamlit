# displayProjectBokeh

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from bokeh.models import ColumnDataSource, GeoJSONDataSource, HoverTool
from bokeh.plotting import figure, show
# from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import row
from datetime import datetime, timezone, timedelta
from sklearn.metrics import confusion_matrix, classification_report
import geopandas
from sklearn.linear_model import LinearRegression, LogisticRegression


def displayProjectBokeh():
    displayBackground("#000000","#FFFFFF")
    st.title("Nouvelles variables et autres traitements")
    st.write("")
    st.write("Suite aux observations réalisées dans l’exploration des données, nous avons jugé opportun d’augmenter notre jeu de données:")
    st.write("- Avec l'apport de données externes")
    st.write("- En remplissant des NaN pour les variables que l'on souhaite garder")
    st.write("- En créant de nouvelles variables (données calculées à partir des variables existantes)")
    st.write("- En supprimant les variables qui nous semblent les moins pertinentes (devenues inutiles ou redondantes)")
    st.write("")

    # Lecture des données
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS.csv', parse_dates=[0], infer_datetime_format=True)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})
    df['DayOfYear'] = df.Date.dt.day_of_year
    df['Month'] = df.Date.dt.month
    cities_df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/citiesloc.csv', index_col=0)
    df = df.merge(right=cities_df[['city', 'lat', 'lng']], left_on='Location', 
                  right_on='city', how='left').drop(columns='city').rename(columns={'lat':'Latitude', 'lng':'Longitude'})
    
    with st.expander("**Apport de données externes**"):
        st.write("")
        st.write("Pour chaque station météorologique, nous rajoutons une information concernant sa position géographique")
        st.write("- Latitude et Longitude: Données récupérées de “World Cities Database”")
        st.write("A partir de la carte des climats, Nous rajoutons une information concernant le type de climat pour chaque station météorologique")
        st.write("- Données de climat : Données récupérées de Wikipedia")
        st.write("")
        st.image(loadImage(".\explorationDesDonnees\CarteDesClimats.jpg",1200))
        st.write("")
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
        #ongletcarte = Panel(child=p1, title="Carte de l'Australie avec les lieux d'observations météorologiques et les climats associés")

        # liste des onglets
        #groupe_onglets1 = Tabs(tabs=[ongletcarte])

        # layout global, en ligne
        # layout = row(groupe_onglets1)
        st.bokeh_chart(p1)
        


    with st.expander("**Remplissage des NaNs**"):
        st.write("")
        st.write("Pressure:")
        st.write("Etant donné que Pressure3pm et Pressure9am sont assez corrélés, nous remplissons les NaN de Pressure3pm avec Pressure9am.")
        st.write("")
        st.write("Humidity:")
        st.write("Etant donné que Humidity3pm et Humidity9am sont assez corrélés, nous remplissons les NaN de Humidity3pm avec Humidity9am.")
        st.write("")
        st.write("Temp9am:")
        st.write("Etant donné que Temp9am et MinTemp sont assez corrélés, nous remplissons les NaN de Temp9am avec MinTemp.")
        
    with st.expander("**Création de nouvelles variables**"):
        st.write("")
        st.write("En réalisant différentes observations sur nos variables de base, nous sommes capables de sélectionner quelques"\
                 " variables calculées qui ne sont pas corrélées linéairement avec les variables de base et qui semblent"\
                     " avoir un impact sur la variable cible:")
        
        nouvellesVariables = "<style>"\
        "table {"\
            "border-collapse: collapse;"\
            "table-layout: fixed ;"\
            "width: 600px;"\
            "border: 1px solid white;"\
            "color: white;"\
        "}"\
        "tr {"\
            "background-color: black;"\
            "color: white;"\
        "}"\
        "</style>"\
        "<table><tr><td style='width:15%;'>1.</td><td style='width:35%;'>diffTempMinMax</td><td style='width:50%;'>= MaxTemp - MinTemp</td></tr>"\
        "<table><tr><td style='width:15%;'>2.</td><td style='width:35%;'>diffWind3pm9am</td><td style='width:50%;'>= WindSpeed3pm - WindSpeed9am</td></tr>"\
        "<table><tr><td style='width:15%;'>3.</td><td style='width:35%;'>diffPressure9am3pm</td><td style='width:50%;'>= Pressure3pm - Pressure9am</td></tr>"\
        "<table><tr><td style='width:15%;'>4.</td><td style='width:35%;'>diffHimidity9am3pm</td><td style='width:50%;'>= Humidity3pm - Humidity9am</td></tr>"\
        "<table><tr><td style='width:15%;'>5.</td><td style='width:35%;'>DeltaP_1d</td><td style='width:50%;'>= Pressure3pm[today] - Pressure3pm[today- 1 day]</td></tr>"\
        "<table><tr><td style='width:15%;'>6.</td><td style='width:35%;'>DeltaP_2d</td><td style='width:50%;'>= Pressure3pm[today] - Pressure3pm[today- 2 days]</td></tr>"\
        "<table><tr><td style='width:15%;'>7.</td><td style='width:35%;'>DeltaP_3d</td><td style='width:50%;'>= Pressure3pm[today] - Pressure3pm[today- 3 days]</td></tr>"\
        "<table><tr><td style='width:15%;'>8.</td><td style='width:35%;'>DeltaH_1d</td><td style='width:50%;'>= Humidity3pm[today] - Humidity3pm[today- 1 day]</td></tr>"\
        "<table><tr><td style='width:15%;'>9.</td><td style='width:35%;'>DeltaH_2d</td><td style='width:50%;'>= Humidity3pm[today] - Humidity3pm[today- 2 days]</td></tr>"\
        "<table><tr><td style='width:15%;'>10.</td><td style='width:35%;'>DeltaH_3d</td><td style='width:50%;'>= Humidity3pm[today] - Humidity3pm[today- 3 days]</td></tr>"\
        "<table><tr><td style='width:15%;'>11.</td><td style='width:35%;'>WindDirInfluence</td><td style='width:50%;'>= Coefficient représentant l’influence de la direction du vent sur la probabilité d’avoir de la pluie</td></tr>"\
        "<table><tr><td style='width:15%;'>12.</td><td style='width:35%;'>consecutiveRainingDays</td><td style='width:50%;'>= Nombre de jour consécutifs de pluie</td></tr>"\
        "</table>"

        st.markdown(nouvellesVariables,unsafe_allow_html = True)

        

        st.write("")
        st.write("**Analyses de données pour la création de 'diffTempMinMax'**")
        fig1, axs = plt.subplots(nrows=2, ncols=1, figsize=(4,8))

        sns.scatterplot(x='Temp9am', y='Temp3pm', data=df, alpha=0.01, hue='RainTomorrow', ax=axs[0])
        axs[0].plot([df.Temp9am.min(), df.Temp9am.max()], [df.Temp9am.min(), df.Temp9am.max()], 'g--', linewidth='1'); # affichage de x=y

        sns.scatterplot(x='MinTemp', y='MaxTemp', data=df, alpha=0.01, hue='RainTomorrow', ax=axs[1])
        axs[1].plot([df.MinTemp.min(), df.MinTemp.max()], [df.MinTemp.min(), df.MinTemp.max()], 'g--', linewidth='1')
        plt.tight_layout()

        st.pyplot(fig1)

        st.write("Le graphique ci-dessus suggère que :")
        st.write("- Si les écarts de températures entre le matin et l'après-midi sont modérés, le risque de pluie est plus important.")
        st.write("- Si les écarts de températures entre les valeurs Min/Max sont modérés, le risque de pluie est plus important.")
        st.write("On peut donc essayer de créer des variables qui font la différence entre ces données.")
        st.write("")

        df['diffTemp3pm9am'] = df.Temp3pm - df.Temp9am
        df['diffTempMinMax'] = df.MaxTemp - df.MinTemp

        # Affichage de la corrélation entre les variables de températures
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
        sns.heatmap(df[['diffTemp3pm9am', 'diffTempMinMax', 'Temp9am', 'Temp3pm', 'MinTemp', 'MaxTemp',
                                'Rainfall', 'RainTomorrow']].corr(), annot=True, center=0, cmap='seismic_r',ax=ax2)
        st.pyplot(fig2)

        st.write("A la vue de la matrice de corrélation ci-dessus, il semble pertinent de conserver 2 variables très peu corrélées"\
                    " entre elles. Par exemple diffTemp3pm9am et MinTemp parce que ce sont les moins corrélées entre elles.")
        st.write("")

        # Affichage du nuage de point relatif à ces 2 variables
        fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
        sns.scatterplot(x='diffTemp3pm9am', y='MinTemp', data=df, alpha=0.05, hue='RainTomorrow')
        st.pyplot(fig3)

        st.write("Le nuage de point semble confirmer un impact des couples Temp9am/diffTempMinMax sur la variable cible.")

        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        # with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):