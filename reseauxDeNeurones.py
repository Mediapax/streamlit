# reseauxDeNeurones

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

# chemin d'accès aux modèles enregistrés
filename_path = './techniquesML/neural/'

def reseauxDeNeurones():
    displayBackground("#000000","#FFFFFF")
    st.header("Réseaux de Neurones")
    st.markdown("La richesse des réseaux de neurones fait qu'on peut aborder la modélisation du problème de nombreuses façons :")
    """
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
    """
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
