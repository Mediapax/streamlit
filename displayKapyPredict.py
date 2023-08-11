# displayKapyPredict

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

def displayKapyPredict():
    st.title("Kapy Predict : Va-t-il pleuvoir demain ?")
    
    st.write('Dans cette onglet, nous allons pouvoir prédire le risque de pluie du lendemain à partir des données météos des 2 jours précédents.')
    st.divider()

    #--------Lecture des données-------#
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', index_col=0)
        
    #--------Sélection d'un jeu de colonne réduit-------#
    columns_selection=['Rainfall', 'WindGustSpeed', 'Temp9am', 'diffTempMinMax', 'diffWind3pm9am',
           'diffPressure9am3pm', 'Pressure', 'DeltaP_1d',
           'diffHimidity9am3pm', 'Humidity', 'DeltaH_1d']
    
    # Reporter les dernières données météos
    st.subheader('Reporter vos dernières données météos:')
    cols = st.columns(4)
    i=0
    values = {}
    
    for c in columns_selection:
        with cols[i//3]:
            values[c] = st.number_input(c, value=df[c].median(), min_value=df[c].min(), max_value=df[c].max(), step=(df[c].std()/10))
            i += 1
    st.divider()
    
    # vecteur à prédire
    X_new = pd.DataFrame.from_dict(values, orient='index').transpose()
    
    #-----------------Prédiction avec la regression-----------------#
    st.subheader('Prédiction avec une régression linéaire')
    
    # récupération du modèle
    pipe = load('models/linreg.joblib')
    st.write('Le type de modèle utilisé:')
    st.write(pipe.named_steps)
    
    # fonction de prédiction de la précision
    def kapy_acc_score(x):
        if x < 1:
            y = 0.24902482120090305 + x * 0.16254928791072928
        else:
            y = 1.0217989542055308
            y += 0.0 * 1/ (x ** 0)
            y += -1.4554248529478369 * 1/ (x ** 1)
            y += 1.1348933361812372 * 1/ (x ** 2)
            y += -0.32348398974721754 * 1/ (x ** 3)
            y += 0.02237893370342592 * 1/ (x ** 4)
        y = max(0, y)
        return y
    
    # prédiction
    y_new = pipe.predict(X_new)[0]
    y_new_positive = np.max([0.0, y_new])
    st.write('Prédiction (en mm): ', np.round(y_new_positive,2))
    
    st.write('% de chance de pluie le lendemain: ', np.round(kapy_acc_score(y_new)*100,2)) 
    #-----------------Fin de la prédiction avec la regression-----------------#
