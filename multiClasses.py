# multiClasses

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
import pandas as pd
import numpy as np
from joblib import load

from sklearn.preprocessing import StandardScaler

def multiClasses():
    displayBackground("#000000","#FFFFFF")
    st.header("Multi-classes")

    #-----------------------------------------------------#
    # Lecture des données
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', parse_dates=['Date'], index_col=0)

    # Simplification des features
    df = df[["Location", "Date", "Rainfall", "WindGustSpeed", "Temp9am", "diffTempMinMax", "diffWind3pm9am", "diffPressure9am3pm", "Pressure", "DeltaP_1d", "diffHimidity9am3pm", "Humidity", "DeltaH_1d"]]

    # Création de la classe cible
    df = df.set_index(['Location', 'Date'])
    df = df.sort_values(by=['Location', 'Date'])
    df['Rainfall_T'] = df.groupby(by=['Location'])['Rainfall'].shift(-1) # création de Rainfall_T (Tomorrow)
    df.dropna(subset=['Rainfall_T'], inplace=True) # on retire les Rainfall_T avec des NaN
    df['RainClassTomorrow'] = df['Rainfall_T'].apply(lambda x: 0 if x==0 else 1 if x<=1.0 else 2)

    # reset_index
    df = df.reset_index()
    
    # on peut supprimer les colonnes inutiles
    df.drop(['Location', 'Date', 'Rainfall_T'], axis=1, inplace=True)

     # on retire les NaN
    df.dropna(inplace=True)

    # création des jeux de données et de données cibles
    data = df.drop(['RainClassTomorrow'], axis=1)
    target = df['RainClassTomorrow']
