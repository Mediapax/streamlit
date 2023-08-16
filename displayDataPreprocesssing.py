# displayDataPreprocesssing

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from explorationDesDonnees import explorationDesDonnees
from climatEtGeographie import climatEtGeographie
from nans import nans
from nouvellesVariables import nouvellesVariables
from fichierDefinitif import fichierDefinitif


def displayDataPreprocesssing():
    displayBackground("#000000","#FFFFFF")
    st.title("Etude du jeu de Données")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Sources", "Explo. data", "Climat & Géographie", "NaNs", "Nlles Variables", "Fichier définitif"])

    def sources():
        st.header("Sources des Données")
        st.write("")
        st.write("Nos données sources sont issues du  ’’Australian Government - Bureau of Meteorology” et ont été téléchargées sur Kaggle:")
        st.write("https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package")
        st.write("")
        st.write("Par défaut, il faudra noter que ces données sont sous Copyright Commonwealth of Australia 2010 et ne sont donc pas disponibles"\
                 " librement pour une utilisation commerciale")
        st.write("")
        st.write("Nous avons utilisé quelques sources de données utilisables librement (Licence Creative Commons) afin d’augmenter les données:")
        st.write("")
        st.write("Voir l'onglet 'Climat & Géographie'")
    
    with tab1 :
        sources()

    with tab2 :
        explorationDesDonnees()

    with tab3 :
        climatEtGeographie()

    with tab4 :
        nans()

    with tab5:
        nouvellesVariables()

    with tab6 :
        fichierDefinitif()


    

