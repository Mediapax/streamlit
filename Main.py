import streamlit as st

from displayDataPreprocesssing import displayDataPreprocesssing
from displayClassifiersStudy import displayClassifiersStudy
from displayRealTimeModeling import displayRealTimeModeling
from displayKapyPredict import displayKapyPredict
from displayBranchingOut import displayBranchingOut
from displayProjectBranchingIn import displayProjectBranchingIn

def main():
    st.sidebar.title("Kapy : Prévisions Météo en Australie")

    menu_selection = st.sidebar.radio(
        "Menu ", 
        ("Présentation du projet", "Etude du jeu de Données", "Etude de divers 'Classifiers'", "Choisissez votre modèle", "Kapy Predict : Va-t-il pleuvoir demain ?", "Bilan et suite du projet" )
        )

    if menu_selection == "Présentation du projet":
        displayProjectBranchingIn()
    if menu_selection == "Etude du jeu de Données":
        displayDataPreprocesssing()
    elif menu_selection == "Etude de divers 'Classifiers'":
        displayClassifiersStudy()
    elif menu_selection == "Choisissez votre modèle":
        displayRealTimeModeling()
    elif menu_selection == "Kapy Predict : Va-t-il pleuvoir demain ?":
        displayKapyPredict()
    elif menu_selection == "Bilan et suite du projet":
        displayBranchingOut()

if __name__ == "__main__":
    main()


