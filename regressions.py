import streamlit as st
import re

import pandas as pd
import numpy as np
from joblib import load

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, recall_score, precision_score, f1_score
from sklearn.pipeline import Pipeline

def regression():
    st.title('Regressions') 
    
    st.markdown("Une regression peut être envisagée en transformant la variable `Rainfall` en une variable `Rainfall_Tomorrow`. Il s'agira donc de prédire une quantité de pluie à venir.")
    
    
    st.subheader("Jeu de données d'entrée")
    columns_selection=['Rainfall', 'WindGustSpeed', 'Temp9am', 'diffTempMinMax', 'diffWind3pm9am',
           'diffPressure9am3pm', 'Pressure', 'DeltaP_1d',
           'diffHimidity9am3pm', 'Humidity', 'DeltaH_1d']
    st.markdown('Par soucis de simplicité et pour préserver des temps de calculs acceptable, seules les variables suivantes seront conservées en raison de leur qualité de grandeur physique:')
    st.write(columns_selection)
    st.markdown("La sélection de ces variables est aussi la résultantes de plusieurs tests qui ont montré qu'elles étaient les plus importantes dans la prédiction.")
    
    st.divider()
    st.subheader("Les modèles et métriques principalement utilisés:")
    col1, col2 = st.columns(2, gap='medium')
    
    # modèles de regression
    with col1:
        st.markdown("**Modèles de régressions**")
        st.markdown("1. `LinearRegression`")
        st.markdown('2. `HistGradientBoostingRegressor`')
        st.markdown('3. `KNeighborsRegressor`')
    
    # metrics
    with col2:
        st.markdown("**Métriques**")
        st.markdown("1. `r2_score`")
        st.markdown('2. `mean_squared_error`')
        st.markdown('3. `mean_absolute_error`')
    
    st.divider()
    st.subheader("Transformation des données:")
    st.markdown("En plus des différents types de régression, une transformation des données sources ainsi que des données cibles a été envisagée. Ceci afin de corriger des ordres de grandeurs très différents (ex.: Pressure vs Humidity) et/ou des distributions très déséquilibrées (ex.: Rainfall).")
    col1, col2 = st.columns(2, gap='medium')
    
    with col1:
        st.markdown("**Données sources:**")
        st.markdown("1. `StandardScaler`")
        st.markdown('2. `PolynomialFeatures`')
        
    with col2:
        st.markdown("**Données cibles:**")
        st.markdown("1. Log")
        st.markdown('2. 1/x')
    
    # chemin d'accès aux modèles enregistrés
    filename_path = './models/regressions/'
    
    # Résultats
    st.header("Résultats")
    df_models = load(filename_path+'joblib_regression_results.dataframe')
    st.dataframe(df_models.round(2), hide_index=True)
    
    
    # afficher les résultats pour les 3 modèles principaux
    st.subheader("Combinaison de modèle")
    st.markdown("L'idée est ici d'essayer de combiner 3 modèles qui ont donnés une relative satisfaction. La combinaison sera effectuée en moyennant les résultats de chaque modèle pour obtenir un résultat final.")
    st.markdown("Les 3 modèles suivants sont choisis:")
    
    
    df_pred_train, df_pred_test = load(filename_path+'joblib_pred_mean.dataframe')
    
    st.markdown("Moyenne des résultats sur le jeu d'entrainement:")
    st.dataframe(df_pred_train.head().round(2))
    st.markdown("Moyenne des résultats sur le jeu de test:")
    st.dataframe(df_pred_test.head().round(2))
    
    st.markdown("**Résultats:**")
    st.markdown("Rappel des résultats pour les 3 modèles séparés")
    short_list = ['joblib_linreg_xpart_pf3', 'joblib_HistGradBoostingReg_xpart_ss', 'joblib_KNNreg_xpart_ss']
    st.dataframe(df_models.loc[short_list][['r2_train', 'r2_test', 'mse_train', 'mse_test', 'mae_train', 'mae_test']].round(2), hide_index=True)
    
    st.markdown("Résultats pour les 3 modèles séparés")
    
    df_combine = load(filename_path+'joblib_results_combined.dataframe')
    st.dataframe(df_combine.round(2), hide_index=True)
    st.markdown("Nous observons une légère amélioration du R2 score.")
    
    # Classification binaire
    st.subheader("Convertion en classification binaire")
    st.markdown("A partir des résultats précédemment obtenus, nous pouvons tenter de faire une convertion en une classification binaire. Nous pouvons jouer sur les seuils afin d'optimiser un résultat comme exposé dans le graphique ci-après.")
    
    # on calcule les précisions/r2 score/recalls pour différentes valeurs de seuil pour considéré s'il pleut ou non
    data_visu = load(filename_path+'joblib_data_visu.dataframe')
    st.line_chart(data=data_visu, x='x', y=['recall', 'precision', 'f1_score'], width=540, height=400, use_container_width=False)
    
    max_arg = np.argmax(data_visu['f1_score'])
    
    x_opti = np.round(data_visu.iloc[max_arg]['x'],2)
    st.markdown('La plus grande valeur du f1_score est obtenue pour x='+ str(x_opti))
    
    st.markdown('Pour cette valeur, les métriques sont les suivantes:')
    
    st.write(data_visu.round(2).set_index('x').iloc[max_arg].rename('x='+str(x_opti)))
