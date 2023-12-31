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

def print_weather(rain_ratio=0.5, width=50):
    # affiche le logo du temps prédit en fonction du % de chance de pluie le lendemain
    logo_path = "./pictures/"
    if rain_ratio>0.75: # temp pluvieux presque sûr
        return st.image(logo_path+"rain.png", width=width)
    elif rain_ratio>0.5: # temp pluvieux presque sûr
        return st.image(logo_path+"mixed_rain.png", width=width)
    elif rain_ratio>0.25: # temp pluvieux presque sûr
        return st.image(logo_path+"mixed_sun.png", width=width)
    else: # temp pluvieux presque sûr
        return st.image(logo_path+"sun.png", width=width)

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
    #st.divider()
    
    # vecteur à prédire
    
    X_new = pd.DataFrame.from_dict(values, orient='index').transpose()

    #-----------------Prédiction avec la regression-----------------#
    pipe = load('models/linreg.joblib')
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
    reg_y_new = pipe.predict(X_new)[0]
    reg_y_new_positive = np.max([0.0, reg_y_new])
    reg_rain_ratio = kapy_acc_score(reg_y_new)
    #-----------------Fin de la Prédiction avec la regression-----------------#
    #-----------------Prédiction avec KNeighborsClassifier-----------------#
    chemin = "./models/" # sous unix ==> chemin : "models/""

    # Charger le modèle depuis le fichier
    loaded_knn_model = load(chemin + 'knnmodel1.joblib')

    # chargement normalisation
    loaded_minmax = load(chemin + 'knnminmax1.joblib')
    
    # Affichage de la prédiction
    new_imput_perso_normalized = loaded_minmax.transform(X_new)
    probaPluie = loaded_knn_model.predict_proba(new_imput_perso_normalized)[0,1]
    probaSec = loaded_knn_model.predict_proba(new_imput_perso_normalized)[0,0]

    knn_rain_ratio = probaPluie
    #-----------------Fin de la Prédiction avec KNeighborsClassifier-----------------#
    #-----------------Prédiction n°3-----------------#
    
    # Charger le modèle depuis le fichier
    chemin = "./models/arima/"
    df_params = load(chemin+'Locations_ArimaParameters.joblib')
    df_params.sort_index(inplace=True)

    with cols[i//3]:
        loc = st.selectbox("Ville :", df_params.index)
        i += 1
    
    st.divider()

    # Affichage de la prédiction
    model_params = df_params.loc[loc]
    cst = model_params['const']
    ar = model_params['ar.L1']
    lin_idx = X_new.columns[1:]
    X_offset_idx = [i + '_offset' for i in lin_idx]
    X_scale_idx = [i + '_scale' for i in lin_idx]
    lin = model_params[lin_idx].to_numpy()
    lin_offset = model_params[X_offset_idx].to_numpy()
    lin_scale = model_params[X_scale_idx].to_numpy()
    log_rainfall = np.log1p(X_new.iloc[0, 0])
    X_scaled = (X_new.iloc[0, 1:].to_numpy() + lin_offset) / lin_scale
    y_pred = np.sum(lin * X_scaled)
    y_pred += cst
    y_pred += ar * log_rainfall
    rainfall_pred = np.expm1(y_pred)
    rainfall_pred_pos = min(max(0.0, rainfall_pred), 371.0)

    arima_rain_ratio = kapy_acc_score(rainfall_pred)

    #-----------------Fin de la Prédiction n°3-----------------#

    # affichage de la prédiction générale
    st.subheader("Prédiction:")
    mean_rain_ratio = np.mean([reg_rain_ratio, knn_rain_ratio, arima_rain_ratio])
    st.write('% de chance de pluie le lendemain: ', np.round(mean_rain_ratio*100,2))
    print_weather(mean_rain_ratio, width=100)
    st.write("Note : cette prédiction résulte de la moyenne des 3 prédictions détaillées ci-dessous.")
    
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        #-----------------Affichage de la Prédiction avec la regression-----------------#
        st.subheader('Prédiction 1')
        
        # récupération du modèle
        st.markdown('**Modèle utilisé:**')
        st.markdown("* `PolynomialFeatures(3)`")
        st.markdown("* `LinearRegression`")
        
        st.write('Prédiction (en mm): ', np.round(reg_y_new_positive,2))

        st.write('% de chance de pluie le lendemain: ', np.round(reg_rain_ratio*100,2)) 

        # affichage du logo
        print_weather(reg_rain_ratio)
        #-----------------Fin de la prédiction avec la regression-----------------#

    with col2:
        #-----------------Prédiction avec KNeighborsClassifier-----------------#
        st.subheader('Prédiction 2')

        # récupération du modèle
        st.markdown('**Modèle utilisé:**')
        st.markdown("* `KNeighborsClassifier`")
        st.markdown("- `n_neighbors = 10`")
        st.markdown("- `weights = distance`")
        st.markdown("- `metric = 'manhattan'`")

        # pas de pluie
        if  probaSec >=  probaPluie :
            st.write("Prédiction : 0 ou moins de 1mm")
            st.write("% de chance de temps sec le lendemain: ", np.round(probaSec*100,2)) 

        # pluie
        else :
            st.write("Prédiction : 1mm ou plus")
            st.write("% de chance de pluie le lendemain: ", np.round(probaPluie*100,2))
        
        st.write("")
        print_weather(knn_rain_ratio)
        #-----------------Fin de la Prédiction n°2-----------------#
    
    with col3:
        #-----------------Prédiction n°3-----------------#
        st.subheader('Prédiction 3')

         # récupération du modèle
        st.markdown('**Modèle utilisé:**')
        st.markdown("* `AR(1)`")
        st.markdown("- `pas de saisonnalité`")

        st.write('Prédiction (en mm): ', np.round(rainfall_pred_pos,2))
        
        st.write('% de chance de pluie le lendemain: ', np.round(kapy_acc_score(rainfall_pred)*100,2)) 
        print_weather(arima_rain_ratio)

        #-----------------Fin de la Prédiction n°3-----------------#
