# RealTimeModelingRegressions

import streamlit as st

import pandas as pd
import numpy as np
from joblib import load

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def RealTimeModelingRegressions():
    st.header("Régressions")
    
    st.write("Nos meilleurs simulations nous ont permis d'obtenir un F1-score de 0.662 et de maintenir une Accuracy de 0.87")
    st.write("Pouvez-vous faire mieux ?")

    st.divider()

    #-----------------------------------------------------#
    # Lecture des données
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', index_col=0)
    #df = pd.read_csv('./weatherAUS_Kapy_v2.csv', index_col=0)
    
    # Création de Rainfall_T qui correspond à un équivalent de Rainfall for tomorrow
    df = df.set_index(['Location', 'Date'])
    df = df.sort_values(by=['Location', 'Date']) # on s'assure que les données soient triées par Lieu et par Date
    df['Rainfall_T'] = df.groupby(by=['Location'])['Rainfall'].shift(-1) # création de Rainfall_T (Tomorrow)
    df.dropna(subset=['Rainfall_T'], inplace=True) # on retirer les Rainfall_T avec des NaN
    
    # reset_index
    df = df.reset_index()
    
    # on peut supprimer les colonnes inutiles
    df.drop(['Location', 'Date'], axis=1, inplace=True)
    

    #--------Sélection des variables avec affichage sur 4 colonnes-------#

    st.subheader("1. Choix des variables à utiliser:")
    features = df.drop(["RainTomorrow", "Rainfall_T"],axis=1)
    labels = features.columns
    cols = st.columns(4)
    i=0
    col = {}
    
    for c in labels:
        with cols[i%4]:
            col[c] = st.checkbox(label = c, value = False)
            i += 1

    columns_selection = []
    for k in col:
        if col[k]:
            columns_selection.append(k)

    st.markdown("Visualisation des premières lignes:")
    st.dataframe(df[columns_selection].head())

    # on retire les NaN
    st.write("A l'issue de votre choix, seules les lignes sans 'NaN' seront retenues pour la modélisation")
    
    df.dropna(inplace=True)
    
    # création des jeux de données et de données cibles
    data = df.drop(['RainTomorrow'], axis=1)
    target = df['RainTomorrow']
    
    X_train, X_test, y_train_RainTomorrow, y_test_RainTomorrow = train_test_split(data, target.values, test_size=0.2, random_state=1234)

    st.subheader("2. Ré-échantillonnage:")
    #--------- Choix d'un rescaler --------#
    st.markdown("imblearn ne semble pas fonctionner avec streamlit cloud")
    #resample_on = st.selectbox('Souhaitez-vous activer un sous-échantillonnage avec RandomUnderSampler ?', ('Non', 'Oui'))

    
    # initialisation des valeurs par défaut de X_train2 et y_train2_RainTomorrow
    X_train2 = X_train
    y_train2_RainTomorrow = y_train_RainTomorrow
   

    #--------- Séparation des variables pour la regression --------#
    y_test2_RainTomorrow = y_test_RainTomorrow
    y_train2 = X_train2["Rainfall_T"]
    y_test2 = X_test["Rainfall_T"]
    X_train2 = X_train2.drop('Rainfall_T', axis=1)
    X_test2 = X_test.drop('Rainfall_T', axis=1)

    # --- Affichage du nombre de valeurs et de leurs répartitions
    len_0 = pd.Series(y_train2_RainTomorrow).value_counts()[0]
    len_1 = pd.Series(y_train2_RainTomorrow).value_counts()[1]
    st.markdown(f"Nombre d'observations sur le jeu d'entrainement: {len(X_train2)} (Ratio de '1': {np.round(len_1/(len_0+len_1),2)})")
    len_0_test = pd.Series(y_test2_RainTomorrow).value_counts()[0]
    len_1_test = pd.Series(y_test2_RainTomorrow).value_counts()[1]
    st.markdown(f"Nombre d'observations sur le jeu de test: {len(X_test2)} (Ratio de '1': {np.round(len_1_test/(len_0_test+len_1_test),2)})")
    
    st.subheader("3. Normalisation:")
    #--------- Choix d'une solution de normalisation  --------#
    #scaling_on = st.toggle('Activer la normalisation')
    scaling_on = st.selectbox('Activer la normalisation', ('Non', 'StandardScaler', 'MinMaxScaler'))

    # initialisation des valeurs par défaut de X_train3 et X_train3
    X_train3 = X_train2
    X_test3 = X_test2
    
    if scaling_on=="StandardScaler":
        scaler = StandardScaler()
        X_train3 = scaler.fit_transform(X_train2)
        X_test3 = scaler.transform(X_test2)
        st.markdown("StandardScaler activé.")
    elif scaling_on=="MinMaxScaler":
        st.markdown("Sélectionner l'intervale [ min : max ]:")
        col1, col2 = st.columns(2)
        with col1:
            scal_min = st.slider('Min:', -5, 4, 0)
        with col2:
            scal_max = st.slider('Max:', scal_min+1, 5, max(scal_min+1,0))
        scaler = MinMaxScaler()
        X_train3 = scaler.fit_transform(X_train2)
        X_test3 = scaler.transform(X_test2)
        st.markdown(f"MinMaxScaler activé sur l'intervale [{scal_min}:{scal_max}]")

    st.subheader("4. Entrainement du modèle:")

    if st.button('Lancer la modélisation'):
        lr = LinearRegression()
        lr.fit(X_train3, y_train2)
    
        y_test_pred = lr.predict(X_test3)

        mse = np.round(mean_squared_error(y_test2, y_test_pred),4)
        mae = np.round(mean_absolute_error(y_test2, y_test_pred),4)
        r2 = np.round(r2_score(y_test2, y_test_pred),4)
        st.markdown(f"MSE = {mse}")
        st.markdown(f"MAE = {mae}")
        st.markdown(f"R2 = {r2}")
        
        y_max = np.max(y_test2)
    
        fig, ax = plt.subplots(1,1)
        ax.scatter(y_test2, y_test_pred, s=1, alpha=0.4, zorder=10)
        ax.plot([0,y_max], [0,y_max], c='grey', zorder=3)
        ax.set_xlim(0,y_max)
        ax.set_ylim(0,y_max)
        ax.set_xlabel('Précipitations observées')
        ax.set_ylabel('Précipitations prédites')
        ax.set_title('Prédiction vs Observation (sur le jeu de test)')
        st.pyplot(fig)
        
