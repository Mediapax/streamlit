# reseauxDeNeurones

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
import pandas as pd

# chemin d'accès aux modèles enregistrés
filename_path = './techniquesML/neural/'

def reseauxDeNeurones():
    displayBackground("#000000","#FFFFFF")
    st.header("Réseaux de Neurones")
    st.markdown("La richesse des réseaux de neurones fait qu'on peut aborder la modélisation du problème de nombreuses façons :")
    st.markdown("""
    - Entrée :
        - Variables du jour
        - Variables d'une fenêtre de *n* jours antérieurs
    - Traitement :
        - Couches denses (réseau de neurones à propagation avant)
        - Couches de convolution (détection de pattern)        
        - Réseau de neurones récurrents (approche temporelle)        
    - Sortie :
        - Probabilité de pluie
        - Quantité de pluie
    """)
    st.markdown("Compte tenu du succès de la prédiction de pluie par le modèle ARIMA, on envisage une approche temporelle avec \
    un réseau de neurones récurrents alimenté par une fenêtre de quelques jours de variables. En sortie, le réseau retourne \
    la probabilité de pluie au moyen d'une fonction logistique.")
    st.markdown("L'architecture du réseau se compose des éléments suivants :")
    st.markdown("- En entrée : une fenêtre de 4 jours des différentes variables standardisées")
    st.markdown("- Deux neurones récurrents de type LSTM en cascade")
    st.markdown("- Deux couches denses")
    st.markdown("- En sortie : un unique neurone avec activation sigmoïde retournant la probabilité de pluie")
    st.markdown("- Des couches de *dropout* sont intercalées au niveau des couches denses")
    st.subheader("Architecture du réseau")
    st.image(loadImage(filename_path+"lstm_model_graph.png", 300))
    st.subheader("Entraînement du réseau")
    st.markdown("Le modèle est entraîné sur 5 années de données et validé sur les données de la dernière année.")
    st.markdown("L'entraînement est assez délicat car la convergence du modèle dépend fortement de \
    l'initialisation des poids. Il est nécessaire d'entraîner une vingtaine de fois le modèle avec des poids \
    initiaux différents pour espérer obtenir un modèle donnant des résultats pas trop mauvais.")
    st.image(loadImage(filename_path+"lstm_model_training.png", 400))
    st.subheader("Métriques pour la ville d'Albury")
    df_result = pd.DataFrame(data={'balanced_accuracy':[0.602], 'f1-score':[0.605]})
    st.dataframe(df_result, hide_index=True)
