# explorationDesDonnees

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def explorationDesDonnees():
    chemin = "./explorationDesDonnees/"
    displayBackground("#000000","#FFFFFF")
    st.title("Exploration des données")
    st.write("")

    st.write("Nous disposons d’un jeu de données conséquent avec un total de 145460 entrées - correspondant chacune à une observation quotidienne"\
                " dans une des 59 stations d’observation météorologique australienne entre 2008 et 2017.")
    st.write("")
    st.write("Pour chaque observation, nous disposons de 23 colonnes en incluant la variable cible.")

    with st.expander("**Liste des variables catégorielles**"):
        listeVariablesCategorielles = "<style>"\
        "table {"\
            "border-collapse: collapse;"\
            "table-layout: fixed ;"\
            "width: 600px;"\
            "border: 1px solid white;"\
            "color: white;"\
        "}"\
        "tr {"\
            "background-color: black;"\
            "color: white;"\
        "}"\
        "</style>"\
        "<table><tr><td style='width:15%;'>1.</td><td style='width:35%;'>Date [YYYY-MM-DD]</td><td style='width:50%;'>Date de l'observation au format YYYY-MM-DD</td></tr>"\
        "<table><tr><td style='width:15%;'>2.</td><td style='width:35%;'>Location [string]</td><td style='width:50%;'>Lieu de l'observation</td></tr>"\
        "<table><tr><td style='width:15%;'>3.</td><td style='width:35%;'>RainToday [Yes/No]</td><td style='width:50%;'>Les précipitations sont-elles > 1mm aujourd'hui ?</td></tr>"\
        "<table><tr><td style='width:15%;'>4.</td><td style='width:35%;'>WindGustDir [string]</td><td style='width:50%;'>Direction de la plus forte rafale de vent dans les 24h jusqu'à 0:00 (en 16ème de direction N,S,E,O)</td></tr>"\
        "<table><tr><td style='width:15%;'>5.</td><td style='width:35%;'>WindDir9am [string]</td><td style='width:50%;'>Direction du vent à 9:00 (en 16ème de direction N,S,E,O)</td></tr>"\
        "<table><tr><td style='width:15%;'>6.</td><td style='width:35%;'>WindDir3pm [string]</td><td style='width:50%;'>Direction du vent à 15:00 (en 16ème de direction N,S,E,O)</td></tr>"\
        "</table>"
        st.markdown(listeVariablesCategorielles,unsafe_allow_html = True)

    st.write("")

    with st.expander("**Liste des variables quantitatives**"):
        listeVariablesQuantitatives = "<style>"\
        "table {"\
            "border-collapse: collapse;"\
            "table-layout: fixed ;"\
            "width: 600px;"\
            "border: 1px solid white;"\
            "color: white;"\
        "}"\
        "tr {"\
            "background-color: black;"\
            "color: white;"\
        "}"\
        "</style>"\
        "<table><tr><td style='width:15%;'>7.</td><td style='width:35%;'>MinTemp [°C]</td><td style='width:50%;'>Température minimale dans les 24h en °C</td></tr>"\
        "<table><tr><td style='width:15%;'>8.</td><td style='width:35%;'>MaxTemp [°C]</td><td style='width:50%;'>Température maximale dans les 24h en °C</td></tr>"\
        "<table><tr><td style='width:15%;'>9.</td><td style='width:35%;'>Rainfall [mm]</td><td style='width:50%;'>Précipitations dans les 24h en mm</td></tr>"\
        "<table><tr><td style='width:15%;'>10.</td><td style='width:35%;'>Evaporation [mm]</td><td style='width:50%;'>Mesure standardisée d'évaporation dans la journée en mm</td></tr>"\
        "<table><tr><td style='width:15%;'>11.</td><td style='width:35%;'>Sunshine [h]</td><td style='width:50%;'>Durée d'ensoleillement dans les 24h jusqu'à 9:00 en heure</td></tr>"\
        "<table><tr><td style='width:15%;'>12.</td><td style='width:35%;'>WindGustSpeed [km/h]</td><td style='width:50%;'>Vitesse de la plus forte rafale de vent dans les 24h jusqu'à 0:00 en km/h</td></tr>"\
        "<table><tr><td style='width:15%;'>13.</td><td style='width:35%;'>WindSpeed9am [km/h]</td><td style='width:50%;'>Vitesse du vent à 9:00 en km/h</td></tr>"\
        "<table><tr><td style='width:15%;'>14.</td><td style='width:35%;'>WindSpeed3pm [km/h]</td><td style='width:50%;'>Vitesse du vent à 15:00 en km/h</td></tr>"\
        "<table><tr><td style='width:15%;'>15.</td><td style='width:35%;'>Humidity9am [%]</td><td style='width:50%;'>Taux d'hygrométrie à 9:00</td></tr>"\
        "<table><tr><td style='width:15%;'>16.</td><td style='width:35%;'>Humidity3pm [%]</td><td style='width:50%;'>Taux d'hygrométrie à 15:00</td></tr>"\
        "<table><tr><td style='width:15%;'>17.</td><td style='width:35%;'>Pressure9am [hPa]</td><td style='width:50%;'>Pression atmosphérique à 9:00 en hPa</td></tr>"\
        "<table><tr><td style='width:15%;'>18.</td><td style='width:35%;'>Pressure3pm [hPa]</td><td style='width:50%;'>Pression atmosphérique à 15:00 en hPa</td></tr>"\
        "<table><tr><td style='width:15%;'>19.</td><td style='width:35%;'>Cloud9am [octa]</td><td style='width:50%;'>Fraction du ciel ennuagé à 9:00 (entier de 0 à 8 + la valeur spéciale 9 si le ciel ne peut être observé)</td></tr>"\
        "<table><tr><td style='width:15%;'>20.</td><td style='width:35%;'>Cloud3pm [octa]</td><td style='width:50%;'>Fraction du ciel ennuagé à 15:00 (entier de 0 à 8 + la valeur spéciale 9 si le ciel ne peut être observé)</td></tr>"\
        "<table><tr><td style='width:15%;'>21.</td><td style='width:35%;'>Temp9am [°C]</td><td style='width:50%;'>Température à 9:00 en °C</td></tr>"\
        "<table><tr><td style='width:15%;'>22.</td><td style='width:35%;'>Temp3pm [°C]</td><td style='width:50%;'>Température à 15:00 en °C</td></tr>"\
        "</table>"
        st.markdown(listeVariablesQuantitatives,unsafe_allow_html = True)

        st.write("")

    with st.expander("**Variable cible**"):
        variableCible = "<style>"\
        "table {"\
            "border-collapse: collapse;"\
            "table-layout: fixed ;"\
            "width: 600px;"\
            "border: 1px solid white;"\
            "color: white;"\
        "}"\
        "tr {"\
            "background-color: black;"\
            "color: white;"\
        "}"\
        "</style>"\
        "<table><tr><td style='width:15%;'>23.</td><td style='width:35%;'>RainTomorrow [Yes/No]</td><td style='width:50%;'>Les précipitations seront-elles >> 1mm demain ?</td></tr>"\
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

        st.image(loadImage(chemin+"DistributionDesPrecipitations.jpg",1200))
        st.write("")
        st.image(loadImage(chemin+"DistributionTemporelleDesPrecipitations1.jpg",1200))
        st.write("")
        st.image(loadImage(chemin+"DistributionTemporelleDesPrecipitations2.jpg",1200))

        st.write("")
    with st.expander("**Lieux d’observations:**"):
        st.write("Les différents lieux d’observation couvrent toutes la surface de l’Australie; qui compte plusieurs climat très différents")
        st.write("La majorité des lieux d’observation est concentrée sur la région sud-est. Certaines stations sont très isolées.")
        st.write("")
        st.image(loadImage(chemin+"ProportionDesJoursAvecPluie.jpg",1200))

    with st.expander("**Les lieux d’observations n’ont pas tous le même historique de relevés:**"):
        st.image(loadImage(chemin+"JoursObsDispoParVilleEtParDate.jpg",1200))

    with st.expander("**Valeurs manquantes**"):
        st.write("Nous observons beaucoup de valeurs valeurs manquantes (NaN):")
        st.write("")
        st.image(loadImage(chemin+"DonneesManquantes.jpg",1200))

    with st.expander("**Valeurs extrêmes**"):
        st.write("Nous n’observons pas de valeurs aberrantes. Cependant, nous avons des valeurs extrêmes sur quelques variables:"\
                 " vitesse du vent, évaporation et précipitation.")
        st.write("")
        st.image(loadImage(chemin+"ValeursExtremes.jpg",1200))

    with st.expander("Corrélation des données à la variable cible"):
        st.write("")
        st.write("Cette matrice de corrélation nous permet tout d’abord de noter qu’aucune de nos variables n’a de corrélation"\
                 " linéaire importante (>0.5) avec la variable cible.")
        st.write("Certaines valeurs (9am, 3pm) semblent parfois assez corrélées entre elles.")
        st.image(loadImage(chemin+"MatriceCorrelation.jpg",1200))
        st.write("")
        st.write("Une analyse graphique semble indiquer que quelques variables sont bel et bien corrélées à la variable cible:")
        st.image(loadImage(chemin+"ViolonCorrelCible.jpg",1200))
        st.write("")
        st.write("On observe que les variations de certaines valeurs physiques (pression, température, humidité)"\
                 " sur quelques jours semblent indiquer l’apparition de précipitations:")
        st.image(loadImage(chemin+"VariationVarCorrelCible.jpg",1200))
        st.write("")
        st.write("La direction du vent semble favoriser (ou non) l’apparition de pluie:")
        st.image(loadImage(chemin+"ProbabilitePluieEnFctDuVent.jpg",1200))







    
