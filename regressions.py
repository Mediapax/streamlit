# regressions

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

# chemin d'accès aux modèles enregistrés
filename_path = './models/'

def regressions():
    displayBackground("#000000","#FFFFFF")
    st.header("Régressions")

    #-----------------------------------------------------#
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
    
    #-----------------------------------------------------#
    
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
        st.markdown('3. etc')
        
    with col2:
        st.markdown("**Données cibles:**")
        st.markdown("1. Log")
        st.markdown('2. 1/x')
        st.markdown('3. etc')
    
    
    # liste des modèles et noms correspondants
    model_list = {'linreg': 'LinearRegression',
                  'HistGradBoostingReg': 'HistGradientBoostingRegressor', 
                  'KNNreg': 'KNeighborsRegressor'}
    
    ytransform_list = {'ylog1p': 'log1p',
                       'yrecip': '1/(y+1)'}
    
    # Résultats
    st.header("Résultats")
    models = {'joblib_linreg_xpart': 'LinearRegression',# OK
              'joblib_linreg_xpart_ss': 'LinearRegression + StandardScaler', # OK
              'joblib_linreg_xpart_pf3': 'LinearRegression + PolynomialFeatures(3)', # OK
              'joblib_linreg_xpart_ss_pf3': 'LinearRegression + StandardScaler + PolynomialFeatures(3)', # OK
              'joblib_linreg_xpart_ylog1p': 'LinearRegression + log1p(y)',# OK
              'joblib_linreg_xpart_ss_ylog1p': 'LinearRegression + StandardScaler + log1p(y)', # OK
              'joblib_linreg_xpart_pf3_ylog1p': 'LinearRegression + PolynomialFeatures(3) + log1p(y)', # OK
              'joblib_linreg_xpart_ss_pf3_ylog1p': 'LinearRegression + StandardScaler + PolynomialFeatures(3) + log1p(y)', # OK
              'joblib_linreg_xpart_yrecip': 'LinearRegression + 1/y',# OK
              'joblib_linreg_xpart_ss_yrecip': 'LinearRegression + StandardScaler + 1/y', # OK
              'joblib_linreg_xpart_pf3_yrecip': 'LinearRegression + PolynomialFeatures(3) + 1/y', # OK
              'joblib_linreg_xpart_ss_pf3_yrecip': 'LinearRegression + StandardScaler + PolynomialFeatures(3) + 1/y', # OK
              'joblib_HistGradBoostingReg_xpart': 'HistGradientBoostingRegressor', # OK
              'joblib_HistGradBoostingReg_xpart_ss': 'HistGradientBoostingRegressor + StandardScaler', # OK
              'joblib_HistGradBoostingReg_xpart_pf3': 'HistGradientBoostingRegressor + PolynomialFeatures(3)', # OK
              'joblib_HistGradBoostingReg_xpart_ss_pf3': 'HistGradientBoostingRegressor + StandardScaler + PolynomialFeatures(3)', # OK
              
              'joblib_HistGradBoostingReg_xpart_ylog1p': 'HistGradientBoostingRegressor + log1p(y)', # OK
              'joblib_HistGradBoostingReg_xpart_ss_ylog1p': 'HistGradientBoostingRegressor + StandardScaler + log1p(y)', # OK
              'joblib_HistGradBoostingReg_xpart_pf3_ylog1p': 'HistGradientBoostingRegressor + PolynomialFeatures(3) + log1p(y)', # OK
              'joblib_HistGradBoostingReg_xpart_ss_pf3_ylog1p': 'HistGradientBoostingRegressor + StandardScaler + PolynomialFeatures(3) + log1p(y)', # OK
              'joblib_KNNreg_xpart_ss': 'KNeighborsRegressor + StandardScaler', # OK
              'joblib_KNNreg_xpart_ss_pf3': 'KNeighborsRegressor + StandardScaler + PolynomialFeatures(3)'} # OK
    df_models = pd.DataFrame(list(models.keys()), columns=['id'])
    df_models['Model'] = df_models['id'].apply(lambda x: model_list[re.findall('joblib_([a-zA-Z]+)_', x)[0]])
    df_models['StdScaler(x)'] = df_models['id'].apply(lambda x: 'X' if len(re.findall('ss',x))>0 else '')
    df_models['PolyFeat(x)'] = df_models['id'].apply(lambda x: 'X' if len(re.findall('pf3',x))>0 else '')
    df_models['f(y)'] = df_models['id'].apply(lambda x: ytransform_list[re.findall('(y[a-zA-Z0-9]+)', x)[0]] if len(re.findall('(y[a-zA-Z0-9]+)', x))>0 else '')
    
    for i in range(len(df_models)):
        id = filename_path+df_models.iloc[i, 0]
        pred_train = load(id+'.predtrain')
        pred_test = load(id+'.predtest')
    
        # calcul du r2 score
        df_models.loc[i, 'r2_train'] = r2_score(y_train, pred_train)
        df_models.loc[i, 'r2_test'] = r2_score(y_test, pred_test)
        # calcul MSE
        df_models.loc[i, 'mse_train'] = mean_squared_error(y_train, pred_train)
        df_models.loc[i, 'mse_test'] = mean_squared_error(y_test, pred_test)
        # calcul MAE
        df_models.loc[i, 'mae_train'] = mean_absolute_error(y_train, pred_train)
        df_models.loc[i, 'mae_test'] = mean_absolute_error(y_test, pred_test)
    
    df_models.set_index('id', inplace=True)
    df_models.sort_values(by='r2_test', ascending=False, inplace=True)
    st.dataframe(df_models.round(2), hide_index=True)
    
    
    # afficher les résultats pour les 3 modèles principaux
    st.subheader("Combinaison de modèle")
    st.markdown("L'idée est ici d'essayer de combiner 3 modèles qui ont donnés une relative satisfaction. La combinaison sera effectuée en moyennant les résultats de chaque modèle pour obtenir un résultat final.")
    st.markdown("Les 3 modèles suivants sont choisis:")
    
    short_list = ['joblib_linreg_xpart_pf3', 'joblib_HistGradBoostingReg_xpart_ss', 'joblib_KNNreg_xpart_ss']
    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    
    i=1
    for id in short_list:
        filepath = filename_path+id
        st.markdown(str(i)+". "+models[id])
        df_pred_train[id] = load(filepath+'.predtrain')
        df_pred_test[id] = load(filepath+'.predtest')
        i += 1
    
    df_pred_train['pred_mean'] = df_pred_train.mean(axis=1)
    df_pred_test['pred_mean'] = df_pred_test.mean(axis=1)
    st.markdown("Moyenne des résultats sur le jeu d'entrainement:")
    st.dataframe(df_pred_train.head().round(2))
    st.markdown("Moyenne des résultats sur le jeu de test:")
    st.dataframe(df_pred_test.head().round(2))
    
    st.markdown("**Résultats:**")
    st.markdown("Rappel des résultats pour les 3 modèles séparés")
    st.dataframe(df_models.loc[short_list][['r2_train', 'r2_test', 'mse_train', 'mse_test', 'mae_train', 'mae_test']].round(2), hide_index=True)
    
    st.markdown("Résultats pour les 3 modèles séparés")
    df_combine = pd.DataFrame()
    # calcul R2
    df_combine.loc[0, 'r2_train'] = r2_score(y_train, df_pred_train['pred_mean'])
    df_combine.loc[0, 'r2_test'] = r2_score(y_test, df_pred_test['pred_mean'])
    # calcul MSE
    df_combine.loc[0, 'mse_train'] = mean_squared_error(y_train, df_pred_train['pred_mean'])
    df_combine.loc[0, 'mse_test'] = mean_squared_error(y_test, df_pred_test['pred_mean'])
    # calcul MAE
    df_combine.loc[0, 'mae_train'] = mean_absolute_error(y_train, df_pred_train['pred_mean'])
    df_combine.loc[0, 'mae_test'] = mean_absolute_error(y_test, df_pred_test['pred_mean'])
    st.dataframe(df_combine.round(2), hide_index=True)
    st.markdown("Nous observons une légère amélioration du R2 score.")
    
    # Classification binaire
    st.subheader("Convertion en classification binaire")
    st.markdown("A partir des résultats précédemment obtenus, nous pouvons tenter de faire une convertion en une classification binaire. Nous pouvons jouer sur les seuils afin d'optimiser un résultat comme exposé dans le graphique ci-après.")
    
    # on calcule les précisions/r2 score/recalls pour différentes valeurs de seuil pour considéré s'il pleut ou non
    threshold = np.linspace(0.5, 5, 20)
    
    data_visu = pd.DataFrame(columns=['recall', 'precision', 'f1_score'])
    
    i=0
    for t in threshold:
        df_pred_test['pred_bin'] = df_pred_test.apply(lambda x: 1 if x.pred_mean > t else 0, axis=1)
        data_visu.loc[i, 'x'] = t
        data_visu.loc[i, 'recall'] = recall_score(y_test_RainTomorrow, df_pred_test.pred_bin)
        data_visu.loc[i, 'precision'] = precision_score(y_test_RainTomorrow, df_pred_test.pred_bin)
        data_visu.loc[i, 'f1_score'] = f1_score(y_test_RainTomorrow, df_pred_test.pred_bin)
        i += 1
    
    data_visu = data_visu.astype('float')
    st.line_chart(data=data_visu, x='x', y=['recall', 'precision', 'f1_score'], width=540, height=400, use_container_width=False)
    
    max_arg = np.argmax(data_visu['f1_score'])
    
    x_opti = np.round(data_visu.iloc[max_arg]['x'],2)
    st.markdown('La plus grande valeur du f1_score est obtenue pour x='+ str(x_opti))
    
    st.markdown('Pour cette valeur, les métriques sont les suivantes:')
    
    st.write(data_visu.round(2).set_index('x').iloc[max_arg].rename('x='+str(x_opti)))
