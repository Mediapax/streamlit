# fichierDefinitif

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def fichierDefinitif():
    st.write("")
    st.write("A partir du fichier d'origine, nous avons donc rajouté des variables liées à la position géographique de chasqeu station et au climat de chaque station.")
    st.write("La création de nouvelles variables corrélées avec la variable cible 'Raintomorrow' nous permet de supprimer certaines variables existantes qui deviennnet redondantes")
    st.write("")

    st.write("")
    st.write("Nous avons donc choisi de retirer ces diverses variables du dataframe final qui sera utilisé pour nos simulations :")
    st.write("")
    varSuppr = "<style>"\
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
    "<table><tr><td style='width:15%;'>1.</td><td style='width:35%;'>'diffTemp3pm9am', 'Temp3pm', 'MinTemp', 'MaxTemp'</td><td style='width:50%;'>diffTempMinMax et Temp9am sont suffisantes</td></tr>"\
    "<tr><td style='width:15%;'>1.</td><td style='width:35%;'>'diffWindGust9am', 'diffWindGust3pm', 'WindSpeed9am', 'WindSpeed3pm'</td><td style='width:50%;'>diffWind3pm9am et WindGustSpeed sont suffisantes</td></tr>"\
    "<tr><td style='width:15%;'>1.</td><td style='width:35%;'>'WindGustDir', 'WindDir3pm', 'WindDir9am'</td><td style='width:50%;'>diffTempMinMax et Temp9am sont suffisantes</td></tr>"\
    "<tr><td style='width:15%;'>1.</td><td style='width:35%;'>'Pressure9am', 'Pressure3pm'</td><td style='width:50%;'>Remplacé par 'Pressure'</td></tr>"\
    "<tr><td style='width:15%;'>1.</td><td style='width:35%;'>'Humidity9am', 'Humidity3pm'</td><td style='width:50%;'>Remplacé par 'Humidity'</td></tr>"\
    "<tr><td style='width:15%;'>1.</td><td style='width:35%;'>'RainToday'</td><td style='width:50%;'>'RainFall' est suffisant</td></tr>"\
    "<tr><td style='width:15%;'>1.</td><td style='width:35%;'>'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'</td><td style='width:50%;'>Quantité de 'NaN' trop importante</td></tr>"\
    "</table>"

    st.markdown(varSuppr,unsafe_allow_html = True)
    st.write("")

    st.write("Et voici la liste finale des variables utilisées pour nos simulations :")
    st.write("")
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', index_col=0)
    nb_col = df.shape[1]
    nb_li = df.shape[0]
    st.write("Cette liste comporte donc "+ str(nb_col) + " variables et " + str(nb_li) + " lignes de données")
    # dfFormatHTML = df.describe(include='all').transpose().to_html()
    # st.write(dfFormatHTML, unsafe_allow_html=True)
    st.dataframe(df.describe(include='all').transpose())


    st.write("")
    st.write("Vous pouvez télécharger le fichier au format csv ici :")
    st.write("https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv")
    st.write("")
        