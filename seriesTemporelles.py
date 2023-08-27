# seriesTemporelles

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# chemin d'accès aux modèles enregistrés
filename_path = './models/arima/'

def seriesTemporelles():
    displayBackground("#000000","#FFFFFF")
    st.header("Séries Temporelles")
    st.write("L'approche par série temporelle consiste à considérer qu'il existe une corrélation entre la quantité de \
    pluie tombée un jour donné et la quantité de pluie tombée la veille, et à exploiter cette corrélation pour réaliser des \
    prédictions de quantité de pluie.")
    st.markdown("Des modèles de type **ARIMA** sont entraînés pour prévoir la pluie. Par contre, ces modèles apportent plusieurs contraintes :")
    st.markdown("*   Il ne peut y avoir qu'une seule variable prédite, ce qui nécessite d'entraîner un modèle par ville.")
    st.markdown("*   La variable prédite doit être continue, ce qui fait que l'on va prédire la quantité de pluie et non la probabilité de pluie.")
    st.markdown("Compte tenu de la distribution extrême de la variable quantité de pluie, il est préférable d'appliquer une transformation à cette variable afin de tasser cette distribution.")
    st.markdown("Après essai de différentes fonctions, la transformation de la variable par la fonction `log(1+x)` donne de bons résultats.")
    st.markdown("La saisonnalité de la série temporelle est de manière naturelle d'un an. Or cette période est très élevée (365,25 jours) \
    et les algorithmes d'estimation de série saisonnière de type SARIMAX ne sont pas adaptés à des périodes élevées. \
    Il est conseillé dans ce cas d'utiliser des modèles ARIMA auxquels on adjoint la saisonnalité au moyen de variables de Fourier.")
    st.markdown("Par ailleurs, la quantité de pluie reste proche de zéro tout au long de la période analysée, aussi on considère qu'il n'y \
    a pas de tendance particulière, ni de besoin de différencier la série.")
    st.markdown("L'analyse des diagrammes d'auto-corrélation et d'autocorrélation partielle de la série temporelle amène à envisager un modèle auto-régressif ARIMA(1, 0, 0) :")
    col1, col2 = st.columns(2, gap='medium')
    
    with col1:
        st.image(loadImage(filename_path+"Arima_acf.png",400))
    
    with col2:
        st.image(loadImage(filename_path+"Arima_pacf.png",400))

    st.divider()

    # Résultats
    location = 'Albury'
    st.subheader("Résultats pour la ville d'Albury")

    y_test_rainfall = load(filename_path+'joblib_Albury_rainfall_truth.predtest')
    y_pred_rainfall = load(filename_path+'joblib_Albury_rainfall_Arima.predtest')
    y_test_raintomorrow = load(filename_path+'joblib_Albury_raintomorrow_truth.predtest')
    y_pred_raintomorrow = load(filename_path+'joblib_Albury_raintomorrow_Arima.predtest')

    fig = plt.figure(figsize=(15,6))
    plt.plot(y_test_rainfall, color='blue', label='Réel')
    plt.plot(y_pred_rainfall, color='red', label='Prédit')
    plt.ylim(0,10)
    plt.axhline(y=1, color='green', ls='--', label='Seuil de 1mm')
    plt.title('Prévision de précipitations à ' + location)
    plt.legend()    
    st.pyplot(fig)

    df_result = pd.DataFrame(data=[balanced_accuracy_score(y_test_raintomorrow, y_pred_raintomorrow), f1_score(y_test_raintomorrow, y_pred_raintomorrow)],
                             columns=['balanced_accuracy', 'f1-score'])
    st.dataframe(df_result.round(2), hide_index=True)

    st.subheader("Matrice de confusion")
    fig = ConfusionMatrixDisplay.from_predictions(y_test_raintomorrow, y_pred_raintomorrow, cmap=plt.cm.Blues)
    st.pyplot(fig.figure_)
