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
            values[c] = st.slider(c, value=df[c].median(), min_value=df[c].min(), max_value=df[c].max())
            i += 1
    st.divider()
    
    # vecteur à prédire
    X_new = pd.DataFrame.from_dict(values, orient='index').transpose()

    col1, col2, col3 = st.columns(3)
    with col1:
        #-----------------Prédiction avec la regression-----------------#
        st.subheader('Prédiction 1')
        
        # récupération du modèle
        pipe = load('models/linreg.joblib')
        st.markdown('**Modèle utilisé:**')
        st.markdown("* `PolynomialFeatures(3)`")
        st.markdown("* `LinearRegression`")
        
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
            y = min(y, 1)
            return y
        
        # prédiction
        y_new = pipe.predict(X_new)[0]
        y_new_positive = np.max([0.0, y_new])
        st.write('Prédiction (en mm): ', np.round(y_new_positive,2))
        
        st.write('% de chance de pluie le lendemain: ', np.round(kapy_acc_score(y_new)*100,2)) 
        #-----------------Fin de la prédiction avec la regression-----------------#

    with col2:
        #-----------------Prédiction avec KNeighborsClassifier-----------------#
        st.subheader('Prédiction 2')
        chemin = "./models/" # sous unix ==> chemin : "models/""

        # récupération du modèle
        st.markdown('**Modèle utilisé:**')
        st.markdown("* `KNeighborsClassifier`")
        st.markdown("- `n_neighbors = 10`")
        st.markdown("- `weights = distance`")
        st.markdown("- `metric = 'manhattan'`")

        # Charger le modèle depuis le fichier
        loaded_knn_model = load(chemin + 'knnmodel1.joblib')

        # chargement normalisation
        loaded_minmax = load(chemin + 'knnminmax1.joblib')
        
        # Affichage de la prédiction
        new_imput_perso_normalized = loaded_minmax.transform(X_new)
        probaPluie = loaded_knn_model.predict_proba(new_imput_perso_normalized)[0,1]
        probaSec = loaded_knn_model.predict_proba(new_imput_perso_normalized)[0,0]

        # pas de pluie
        if  probaSec >=  probaPluie :
            st.write("Prédiction : 0 ou moins de 1mm")
            st.write("% de chance de temps sec le lendemain: ", np.round(probaSec*100,2)) 

        # pluie
        else :
            st.write("Prédiction : 1mm ou plus")
            st.write("% de chance de pluie le lendemain: ", np.round(probaPluie*100,2))
        
        st.write("")
        st.write("")
        
                
        #-----------------Fin de la Prédiction n°2-----------------#
    
    with col3:
        #-----------------Prédiction n°3-----------------#
        st.subheader('Prédiction 3')

        # Charger le modèle depuis le fichier
        chemin = "./models/arima/"
        df_params = load(chemin+'Locations_ArimaParameters.joblib')
        df_params.sort_index(inplace=True)

        loc = st.selectbox("Ville :", df_params.index)

         # récupération du modèle
        st.markdown('**Modèle utilisé:**')
        st.markdown("* `AR(1)`")
        st.markdown("- `pas de saisonnalité`")

        # Affichage de la prédiction
        model_params = df_params.loc[loc]
        cst = model_params['const']
        ar = model_params['ar.L1']
        lin_idx = X_new.columns[1:]
        X_offset_idx = [i + '_offset' for i in lin_idx]
        X_scale_idx = [i + '_scale' for i in lin_idx]
        lin = model_params[lin_idx]
        st.write(lin)
        log_rainfall = np.log1p(X_new.iloc[0, 0])
        X_scaled = (X_new.iloc[1:, 0] + lin[X_offset_idx]) * lin[X_scale_idx]
        y_pred = (lin * X_scaled).sum()
        y_pred += cst
        y_pred += ar * log_rainfall
        rainfall_pred = np.expm1(y_pred)
        rainfall_pred_pos = max(0.0, rainfall_pred)

        st.write('Prédiction (en mm): ', np.round(rainfall_pred_pos,2))
        
        st.write('% de chance de pluie le lendemain: ', np.round(kapy_acc_score(rainfall_pred)*100,2)) 

        #-----------------Fin de la Prédiction n°3-----------------#
