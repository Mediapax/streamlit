# displayRealTimeModeling

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

from modelesNaifs import modelesNaifs
from KNN import KNN
from RealTimeModelingKNN import RealTimeModelingKNN
from RealTimeModelingArima import RealTimeModelingArima
from RealTimeModelingReseauDeNeurones import RealTimeModelingReseauDeNeurones
from RealTimeModelingRegressions import RealTimeModelingRegressions


def displayRealTimeModeling():
    st.title("Construisez votre modèle")
    displayBackground("#000000","#FFFFFF")

    tab1, tab2, tab3, tab4 = st.tabs(["KNeighborsClassifier", "Arima", "Réseau de Neurones", "Régressions"])
    
    with tab1 :
        RealTimeModelingKNN()

    with tab2 :
        RealTimeModelingArima()

    with tab3:
        RealTimeModelingReseauDeNeurones()

    with tab4:
        RealTimeModelingRegressions()
