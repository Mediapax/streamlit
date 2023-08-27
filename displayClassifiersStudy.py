# displayClassifiersStudy

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

from modelesNaifs import modelesNaifs
from KNN import KNN
from reechantillonage import reechantillonage
from multiClasses import multiClasses
from regressions import regressions
from seriesTemporelles import seriesTemporelles
from reseauxDeNeurones import reseauxDeNeurones

def displayClassifiersStudy():
    displayBackground("#000000","#FFFFFF")
    st.title("Techniques de 'Machine Learning'")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Modèles Naïfs", "KNN", "Rééchantillonages", "Multi-classes",
                                                         "Régressions", "Séries Temporelles", "Réseaux de Neurones"])

    
    with tab1 :
        modelesNaifs()

    with tab2 :
        #KNN()

    with tab3:
        reechantillonage()

    with tab4:
        multiClasses()

    with tab5:
        #regressions()

    with tab6:
        seriesTemporelles()

    with tab7:
        reseauxDeNeurones()

    
