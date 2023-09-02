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
    #Parame de la page
    st.set_page_config(page_title="Projet Kapy", page_icon=":cloud:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

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

    st.sidebar.divider()
    
    st.sidebar.markdown(
    """
    <span style='font-style: italic; color: gray; font-weight: bold; font-size: 14px; line-height: 0.8;'>
        L'Equipe Kapy :<br>
        <a href='https://www.linkedin.com/in/arnaudmachefel/'>Arnaud Machefel</a>&nbsp;&nbsp;&nbsp;
        <a href='https://www.linkedin.com/in/olivier-lauffenburger-ba2792276'>Olivier Lauffenburger</a><br>
        <a href='https://www.linkedin.com/in/pierre-le-bert-2270122'>Pierre Le Bert</a>&nbsp;&nbsp;&nbsp;
        <a href='https://www.linkedin.com/in/marionkaisserlian/'>Marion Kaisserlian</a>
    </span>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()


