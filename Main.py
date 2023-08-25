import streamlit as st
import numpy as np

from displayDataPreprocesssing import displayDataPreprocesssing
from displayClassifiersStudy_avec_cache_data import displayClassifiersStudy
from displayRealTimeModeling import displayRealTimeModeling
from displayKapyPredict import displayKapyPredict
from displayBranchingOut import displayBranchingOut
from displayProjectBranchingIn import displayProjectBranchingIn
from displayBackground import displayBackground

from resizeImage import resizeImage, loadImage

def main():

    # fonction "weights_sqr"
    def weights_sqr(d):
        inverse = 1.0 / d.astype(float)
        sqr = np.square(inverse)
        return sqr

    # fonction "weights_sqr4"
    def weights_sqr4(d):
        sqr2 = weights_sqr(d)
        sqr4 = np.square(sqr2)
        return sqr4

    # fonction "weights_sqr8"
    def weights_sqr8(d):
        sqr4 = weights_sqr4(d)
        sqr8 = np.square(sqr4)
        return sqr8


    st.sidebar.image(loadImage("logo_datascientest.png",200))
    st.sidebar.title("Kapy : Prévision météo en Australie")

    menuMain = st.sidebar.radio(
        "Menu ", 
        ("Présentation du projet", 
         "Etude du jeu de Données", 
         "Techniques de Machine Learning", 
         "Construisez votre modèle", 
         "Kapy Predict : Va-t-il pleuvoir demain ?", 
         "Bilan et suite du projet" )
        )

    if menuMain == "Présentation du projet":
        displayProjectBranchingIn()
    elif menuMain == "Etude du jeu de Données":
        displayDataPreprocesssing()
    elif menuMain == "Techniques de Machine Learning":
        displayClassifiersStudy()
    elif menuMain == "Construisez votre modèle":
        displayRealTimeModeling()
    elif menuMain == "Kapy Predict : Va-t-il pleuvoir demain ?":
        displayKapyPredict()
    elif menuMain == "Bilan et suite du projet":
        displayBranchingOut()

    #displayBackground("#000000","#FFFFFF")

if __name__ == "__main__":
    main()


