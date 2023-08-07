import streamlit as st

from displayDataPreprocesssing import displayDataPreprocesssing
from displayClassifiersStudy import displayClassifiersStudy
from displayRealTimeModeling import displayRealTimeModeling
from displayKapyPredict import displayKapyPredict
from displayBranchingOut import displayBranchingOut
import displayProjectBranchingIn as displayProjectBranchingIn

from resizeImage import resizeImage, loadImage

def main():
    st.sidebar.image(loadImage("logo_datascientest.png",200))
    st.sidebar.title("Kapy : Prévision météo en Australie")

    menuMain = st.sidebar.radio(
        "Menu ", 
        ("Présentation du projet", 
         "Etude du jeu de Données", 
         "Etude de divers 'Classifiers'", 
         "Choisissez votre modèle", 
         "Kapy Predict : Va-t-il pleuvoir demain ?", 
         "Bilan et suite du projet" )
        )

    if menuMain == "Présentation du projet":
        menuPresentation = st.sidebar.selectbox(
        "", 
        ("Introduction", 
         "Contextualisation")
        )
        if menuPresentation == "Introduction":
            displayProjectBranchingIn.displayIntroduction()
        elif menuPresentation == "Contextualisation":
            displayProjectBranchingIn.displayContextualisation()
    if menuMain == "Etude du jeu de Données":
        displayDataPreprocesssing()
    elif menuMain == "Etude de divers 'Classifiers'":
        displayClassifiersStudy()
    elif menuMain == "Choisissez votre modèle":
        displayRealTimeModeling()
    elif menuMain == "Kapy Predict : Va-t-il pleuvoir demain ?":
        displayKapyPredict()
    elif menuMain == "Bilan et suite du projet":
        displayBranchingOut()

if __name__ == "__main__":
    main()


