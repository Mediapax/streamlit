# multiClasses

import streamlit as st
import re
#from resizeImage import resizeImage, loadImage
#from PIL import Image
from displayBackground import displayBackground
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# chemin d'accès aux modèles enregistrés
filename_path = './models/multiclass/'

def multiClasses():
    displayBackground("#000000","#FFFFFF")
    st.header("Multi-classes")

    #-----------------------------------------------------#
    """
    # Lecture des données
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', parse_dates=['Date'], index_col=0)

    # Réduction des features
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
    """
    id = filename_path+'joblib_truth'
    y_train = load(id+'.predtrain')
    y_test = load(id+'.predtest')
    
    #-----------------------------------------------------#
    
    st.markdown("Blablabla.")    
    
    st.subheader("Jeu de données d'entrée")
    columns_selection=['Rainfall', 'WindGustSpeed', 'Temp9am', 'diffTempMinMax', 'diffWind3pm9am',
           'diffPressure9am3pm', 'Pressure', 'DeltaP_1d',
           'diffHimidity9am3pm', 'Humidity', 'DeltaH_1d']
    st.markdown('Par souci de simplicité et pour préserver des temps de calculs acceptable, seules les variables suivantes seront conservées en raison de leur qualité de grandeur physique:')
    st.write(columns_selection)
    st.markdown("La sélection de ces variables est aussi la résultantes de plusieurs tests qui ont montré qu'elles étaient les plus importantes dans la prédiction.")
    
    st.divider()
    st.subheader("Les modèles et métriques principalement utilisés:")
    col1, col2 = st.columns(2, gap='medium')
    
    # modèles de regression
    with col1:
        st.markdown("**Modèles de régressions**")
        st.markdown("1. `LogisticRegression`")
        st.markdown('3. `HistGradientBoostingRegressor`')
    
    # metrics
    with col2:
        st.markdown("**Métriques**")
        st.markdown("1. `balanced_accuracy`")
        st.markdown('2. `f1_score`')

    # liste des modèles et noms correspondants
    model_list = {'logreg': 'LogisticRegression',
                  'HistGradBoostingReg': 'HistGradientBoostingRegressor'}

    # Résultats
    st.header("Résultats")
    models = {'joblib_logreg': 'LogisticRegression',
              'joblib_HistGradBoostingReg': 'HistGradientBoostingRegressor'}
    df_models = pd.DataFrame(list(models.keys()), columns=['id'])
    df_models['Model'] = df_models['id'].apply(lambda x: model_list[re.findall('joblib_([a-zA-Z]+)', x)[0]])

    y_pred_train = {}
    y_pred_test = {}
    
    for i in range(len(df_models)):
        id = filename_path+df_models.iloc[i, 0]
        pred_train = load(id+'.predtrain')
        pred_test = load(id+'.predtest')
        y_pred_train[df_models['Model'].iloc[i]] = pred_train
        y_pred_test[df_models['Model'].iloc[i]] = pred_test

        # calcul du balanced accuracy score
        df_models.loc[i, 'bal_acc_train'] = balanced_accuracy_score(y_train, pred_train)
        df_models.loc[i, 'bal_acc_test'] = balanced_accuracy_score(y_test, pred_test)
        # calcul du score F1
        df_models.loc[i, 'f1_train'] = f1_score(y_train, pred_train, average='macro')
        df_models.loc[i, 'f1_test'] = f1_score(y_test, pred_test, average='macro')

    df_models.set_index('id', inplace=True)
    df_models.sort_values(by='bal_acc_test', ascending=False, inplace=True)
    st.dataframe(df_models.round(2), hide_index=True)

    # Visualisation des matrices de confusion
    st.header("Matrices de confusion à 3 classes")
    col1, col2 = st.columns(2, gap='medium')

    # modèles de regression
    with col1:
        st.markdown("**LogisticRegression**")
        fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test['LogisticRegression'], cmap=plt.cm.Blues)
        st.pyplot(fig.figure_)
    
    # metrics
    with col2:
        st.markdown("**HistGradientBoostingRegressor**")
        fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test['HistGradientBoostingRegressor'], cmap=plt.cm.Blues)
        st.pyplot(fig.figure_)

    # Réduction à 2 classes

    # Réduction des classes: {0, 1} -> 0, 2->1
    truth_train = y_train // 2
    truth_test = y_test // 2
    
    for i in range(len(df_models)):
        pred_train = y_pred_train[df_models['Model'].iloc[i]] // 2
        pred_test = y_pred_test[df_models['Model'].iloc[i]] // 2

        df_models.loc[i, 'bal_acc_train'] = balanced_accuracy_score(truth_train, pred_train)
        df_models.loc[i, 'bal_acc_test'] = balanced_accuracy_score(truth_test, pred_test)
        # calcul du score F1
        df_models.loc[i, 'f1_train'] = f1_score(truth_train, pred_train, average='macro')
        df_models.loc[i, 'f1_test'] = f1_score(truth_test, pred_test, average='macro')

    df_models.set_index('id', inplace=True)
    df_models.sort_values(by='bal_acc_test', ascending=False, inplace=True)
    st.dataframe(df_models.round(2), hide_index=True)

    st.header("Matrices de confusion réduite à 2 classes")
    col1, col2 = st.columns(2, gap='medium')

    # modèles de regression
    with col1:
        st.markdown("**LogisticRegression**")
        fig = ConfusionMatrixDisplay.from_predictions(truth_test, y_pred_test['LogisticRegression'] // 2, cmap=plt.cm.Blues)
        st.pyplot(fig.figure_)
    
    # metrics
    with col2:
        st.markdown("**HistGradientBoostingRegressor**")
        fig = ConfusionMatrixDisplay.from_predictions(truth_test, y_pred_test['HistGradientBoostingRegressor'] // 2, cmap=plt.cm.Blues)
        st.pyplot(fig.figure_)
