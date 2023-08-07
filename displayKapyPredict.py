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
    
    # Lecture des données
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', index_col=0)
    
    # Création de Rainfall_T qui correspond à un équivalent de Rainfall for tomorrow
    df = df.set_index(['Location', 'Date'])
    df = df.sort_values(by=['Location', 'Date']) # on s'assure que les données soient triées par Lieu et par Date
    df['Rainfall_T'] = df.groupby(by=['Location'])['Rainfall'].shift(-1) # création de Rainfall_T (Tomorrow)
    df.dropna(subset=['Rainfall_T'], inplace=True) # on retirer les Rainfall_T avec des NaN
    
    # reset_index
    df = df.reset_index()
    
    # on peut supprimer les colonnes inutiles
    df.drop(['Location', 'Date'], axis=1, inplace=True)
    
    # Par soucis de gain de mémoire, je drop qq colonnes peu utiles (à priori)
    df.drop(['DeltaP_2d', 'DeltaH_2d', 'Climate_Am',
           'Climate_Aw', 'Climate_BSh', 'Climate_BSk', 'Climate_BWh',
           'Climate_Cfa', 'Climate_Cfb', 'Climate_Csa', 'Climate_Csb'], axis=1, inplace=True)
    
    # on retire les NaN
    df.dropna(inplace=True)
    
    # création des jeux de données et de données cibles
    data = df.drop(['Rainfall_T'], axis=1)
    target = df['Rainfall_T']
    
    X_train, X_test, y_train, y_test = train_test_split(data, target.values, test_size=0.2, random_state=1234)
    
    # On isole la variable RainTomorrow pour ne pas biaisé les résultats
    y_test_RainTomorrow = X_test['RainTomorrow']
    y_train_RainTomorrow = X_train['RainTomorrow']
    X_test = X_test.drop(['RainTomorrow'], axis=1)
    X_train = X_train.drop(['RainTomorrow'], axis=1)
    
    # Création d'un jeu de donnée réduit
    columns_selection=['Rainfall', 'WindGustSpeed', 'Temp9am', 'diffTempMinMax', 'diffWind3pm9am',
           'diffPressure9am3pm', 'Pressure', 'DeltaP_1d',
           'diffHimidity9am3pm', 'Humidity', 'DeltaH_1d']
    X_train_reduced = X_train[columns_selection]
    X_test_reduced = X_test[columns_selection]
    
    # récupération du modèle
    pipe = load('models/linreg.joblib')
    y_test_pred = pipe.predict(X_test_reduced)
    st.write('Le type de modèle utilisé:')
    st.write(pipe.named_steps)
    
    score_test = r2_score(y_test, y_test_pred) 
    st.write('r2_score de ce modèle: ', r2_score(y_test, y_test_pred))
    
    
    st.divider()
    st.subheader('Enregistrer les dernières données météos:')
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
