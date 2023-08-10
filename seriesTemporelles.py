# seriesTemporelles

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def seriesTemporelles():
    displayBackground("#000000","#FFFFFF")
    st.header("Séries Temporelles")
    st.write("L'approche par série temporelle consiste à considérer qu'il existe une corrélation entre la quantité de \
    pluie tombée un jour donné et la quantité de pluie tombée la veille, et à exploiter cette corrélation pour réaliser des \
    prédictions de quantité de pluie.")
    st.markdown("Des modèles de type **ARIMA** sont entraînés pour prévoir la pluie. Par contre, ces modèles apportent plusieurs contraintes :")
    st.markdown("*   Il ne peut y avoir qu'une seule variable prédite, ce qui nécessite d'entraîner un modèle par ville.")
    st.markdown("*   La variable prédite doit être continue, ce qui fait que l'on va prédire la quantité de pluie et non la probabilité de pluie.")
    
