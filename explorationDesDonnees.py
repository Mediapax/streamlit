# explorationDesDonnees

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def explorationDesDonnees():
    displayBackground("#000000","#FFFFFF")
    st.title("Exploration des données")
    st.write("")

    with st.expander("Vue d'ensemble"):
        st.write("Nous disposons d’un jeu de données conséquent avec un total de 145460 entrées; correspondant chacune à une observation quotidienne"\
                 " dans une des 59 stations d’observation météorologique australienne entre 2008 et 2017.")
        st.write("")
        st.write("Pour chaque observation, nous disposons de 23 colonnes en incluant la variable cible:")

    with st.expander("**Liste des variables catégorielles**"):
        listeVariablesCategorielles = "<style>"\
        "table {"\
            "border-collapse: collapse;"\
            "width: 100%;"\
            "border: 2px solid white;"\
            "color: white;"\
        "}"\
        "tr, td {"\
            "border: 1px solid white;"\
            "padding: 8px;"\
            "text-align: center;"\
        "}"\
        "tr {"\
            "background-color: black;"\
            "color: white;"\
        "}"\
        "</style>"\
        "<table><tr><td>1.</td><td>bla bla</td></tr>"\
        "</table>"
        st.markdown(listeVariablesCategorielles,unsafe_allow_html = True)

    st.write("")

    with st.expander("**Liste des variables quantitatives**"):
        listeVariablesQuantitatives = "<style>"\
        "table {"\
            "border-collapse: collapse;"\
            "width: 100%;"\
            "border: 2px solid white;"\
            "color: white;"\
        "}"\
        "tr, td {"\
            "border: 1px solid white;"\
            "padding: 8px;"\
            "text-align: center;"\
        "}"\
        "tr {"\
            "background-color: black;"\
            "color: white;"\
        "}"\
        "</style>"\
        "<table><tr><td>1.</td><td>bla bla</td></tr>"\
        "</table>"
        st.markdown(listeVariablesQuantitatives,unsafe_allow_html = True)

        st.write("")

    with st.expander("**Variable cible**"):
        variableCible = "<style>"\
        "table {"\
            "border-collapse: collapse;"\
            "width: 100%;"\
            "border: 2px solid white;"\
            "color: white;"\
        "}"\
        "tr, td {"\
            "border: 1px solid white;"\
            "padding: 8px;"\
            "text-align: center;"\
        "}"\
        "tr {"\
            "background-color: black;"\
            "color: white;"\
        "}"\
        "</style>"\
        "<table><tr><td>1.</td><td>bla bla</td></tr>"\
        "</table>"
        st.markdown(variableCible,unsafe_allow_html = True)
        st.write("")
        st.write("**Type de problème:**")
        st.write("A première vue, il s’agit d’un problème de classification avec la variable binaire RainTomorrow."\
                 " On pourra cependant utiliser la variable RainToday afin de transformer le problème en un problème de régression.")
        st.write("")
        st.write("**Répartition de la variable cible:**")
        st.write("-	Le jeu est déséquilibré avec seulement environ 20% de jours de pluie.")
        st.write("-	La plupart des précipitation sont assez faible (<10mm) mais nous observons aussi des valeurs extrêmes très élevées (jusqu’à >300mm)")
        st.write("-	Il y a une saisonnalité dans la quantité de précipitation tout au long de l’année.")

        st.image(loadImage(".\explorationDesDonnees\DistributionDesPrecipitations.jpg",1200))
        st.write("")
        st.image(loadImage(".\explorationDesDonnees\DistributionTemporelleDesPrecipitations1.jpg",1200))
        st.write("")
        st.image(loadImage(".\explorationDesDonnees\DistributionTemporelleDesPrecipitations2.jpg",1200))

        st.write("")
    with st.expander("**Lieux d’observations:**"):
        st.write("Les différents lieux d’observation couvrent toutes la surface de l’Australie; qui compte plusieurs climat très différents")
        st.write("La majorité des lieux d’observation est concentrée sur la région sud-est. Certaines stations sont très isolées.")
        st.write("")
        st.image(loadImage(".\explorationDesDonnees\ProportionDesJoursAvecPluie.jpg",1200))

    with st.expander("**Les lieux d’observations n’ont pas tous le même historique de relevés:**"):
        st.image(loadImage(".\explorationDesDonnees\JoursObsDispoParVilleEtParDate.jpg",1200))


        st.write("**Valeurs manquantes**")
        st.write("**Valeurs extrêmes::**")
        st.write("**Corrélation des données à la variable cible:**")





    
