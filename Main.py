import streamlit as st

from displayDataPreprocesssing import displayDataPreprocesssing
from displayClassifiersStudy import displayClassifiersStudy
from displayRealTimeModeling import displayRealTimeModeling
from displayKapyPredict import displayKapyPredict
from displayBranchingOut import displayBranchingOut
from displayProjectBranchingIn import displayProjectBranchingIn
from displayBackground import displayBackground

from resizeImage import resizeImage, loadImage

def main():
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
    if menuMain == "Etude du jeu de Données":
        displayDataPreprocesssing()
    elif menuMain == "Techniques de Machine Learning":
        displayClassifiersStudy()
    elif menuMain == "Construisez votre modèle":
        displayRealTimeModeling()
    elif menuMain == "Kapy Predict : Va-t-il pleuvoir demain ?":
        displayKapyPredict()
    elif menuMain == "Bilan et suite du projet":
        displayBranchingOut()

    displayBackground("#000000","#FFFFFF")

if __name__ == "__main__":
    main()


