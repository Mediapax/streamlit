# nouvellesVariables

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def nouvellesVariables() :
    chemin = "./explorationDesDonnees/"
    displayBackground("#000000","#FFFFFF")
    st.title("Création de nouvelles variables")
    st.write("")
    st.write("En réalisant différentes observations sur nos variables de base, nous sommes capables de sélectionner quelques"\
                 " variables calculées qui ne sont pas corrélées linéairement avec les variables de base et qui semblent"\
                     " avoir un impact sur la variable cible:")
    
    st.write("**Liste des nouvelles variables :**")
    st.write("")
    nouvellesVariables = "<style>"\
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
    "<table><tr><td style='width:15%;'>1.</td><td style='width:35%;'>diffTempMinMax</td><td style='width:50%;'>= MaxTemp - MinTemp</td></tr>"\
    "<table><tr><td style='width:15%;'>2.</td><td style='width:35%;'>diffWind3pm9am</td><td style='width:50%;'>= WindSpeed3pm - WindSpeed9am</td></tr>"\
    "<table><tr><td style='width:15%;'>3.</td><td style='width:35%;'>diffPressure9am3pm</td><td style='width:50%;'>= Pressure3pm - Pressure9am</td></tr>"\
    "<table><tr><td style='width:15%;'>4.</td><td style='width:35%;'>diffHimidity9am3pm</td><td style='width:50%;'>= Humidity3pm - Humidity9am</td></tr>"\
    "<table><tr><td style='width:15%;'>5.</td><td style='width:35%;'>DeltaP_1d</td><td style='width:50%;'>= Pressure3pm[today] - Pressure3pm[today- 1 day]</td></tr>"\
    "<table><tr><td style='width:15%;'>6.</td><td style='width:35%;'>DeltaP_2d</td><td style='width:50%;'>= Pressure3pm[today] - Pressure3pm[today- 2 days]</td></tr>"\
    "<table><tr><td style='width:15%;'>7.</td><td style='width:35%;'>DeltaP_3d</td><td style='width:50%;'>= Pressure3pm[today] - Pressure3pm[today- 3 days]</td></tr>"\
    "<table><tr><td style='width:15%;'>8.</td><td style='width:35%;'>DeltaH_1d</td><td style='width:50%;'>= Humidity3pm[today] - Humidity3pm[today- 1 day]</td></tr>"\
    "<table><tr><td style='width:15%;'>9.</td><td style='width:35%;'>DeltaH_2d</td><td style='width:50%;'>= Humidity3pm[today] - Humidity3pm[today- 2 days]</td></tr>"\
    "<table><tr><td style='width:15%;'>10.</td><td style='width:35%;'>DeltaH_3d</td><td style='width:50%;'>= Humidity3pm[today] - Humidity3pm[today- 3 days]</td></tr>"\
    "<table><tr><td style='width:15%;'>11.</td><td style='width:35%;'>WindDirInfluence</td><td style='width:50%;'>= Coefficient représentant l’influence de la direction du vent sur la probabilité d’avoir de la pluie</td></tr>"\
    "<table><tr><td style='width:15%;'>12.</td><td style='width:35%;'>consecutiveRainingDays</td><td style='width:50%;'>= Nombre de jour consécutifs de pluie</td></tr>"\
    "</table>"

    st.markdown(nouvellesVariables,unsafe_allow_html = True)
    st.write("")


    with st.expander("**Analyses de données pour la création de 'diffTempMinMax'**"):
        st.write("")
        st.image(loadImage(chemin+"diffTemp1.jpg",1200))
        st.write("")

        st.write("")
        st.image(loadImage(chemin+"diffTemp2.jpg",1200))
        st.write("")

        st.write("Le graphique ci-dessus suggère que :")
        st.write("- Si les écarts de températures entre le matin et l'après-midi sont modérés, le risque de pluie est plus important.")
        st.write("- Si les écarts de températures entre les valeurs Min/Max sont modérés, le risque de pluie est plus important.")
        st.write("On peut donc essayer de créer des variables qui font la différence entre ces données.")
        st.write("")

        st.write("")
        st.image(loadImage(chemin+"diffTempMatCorr.png",1200))
        st.write("")

        st.write("A la vue de la matrice de corrélation ci-dessus, il semble pertinent de conserver 2 variables très peu corrélées"\
                    " entre elles. Par exemple diffTemp3pm9am et MinTemp parce que ce sont les moins corrélées entre elles.")
        st.write("")

        st.write("")
        st.image(loadImage(chemin+"diffTempNuagePoints.png",1200))
        st.write("")

        st.write("Le nuage de point semble confirmer un impact des couples Temp9am/diffTempMinMax sur la variable cible.")
        st.write("")


    with st.expander("**Analyses de données pour la création de 'diffWind3pm9am'**"):
        st.write("Vitesses du vent")
        st.write("Le jeu de données contient 3 variables de températures très corrélées: WindGustSpeed, WindSpeed9am, WindSpeed3pm.")

        st.write("")
        st.image(loadImage(chemin+"wind1.jpg",1200))
        st.write("")
        st.write("")
        st.image(loadImage(chemin+"wind2.jpg",1200))
        st.write("")
        st.write("")
        st.image(loadImage(chemin+"wind3.jpg",1200))
        st.write("")

        st.write("- WindSpeed9am vs 3pm : les vents à 9am et 3pm ne semblent pas tellement corrélés à la variable cible")
        st.write("- WindSpeed9am/3pm vs Gust : on observe un impact du couple de données sur la variable cible.")
        st.write("")
        
        # Affichage de la corrélation entre les variables de vitesses de vents
        st.write("")
        st.image(loadImage(chemin+"wind4MatCorr.png",1200))
        st.write("")

        st.write("Ceci tend à dire qu'il faut conserver WindGustSpeed associée à WindSpeed 9am ou 3pm. Pour aller plus loin,"\
                " on peut regarder (comme pour les températures) si le delta entre ces variables peut avoir un intérêt.")
        st.write("")

    with st.expander("**Analyses de données pour la création des variables basées sur les 'différences de pression'**"):
        st.write("")
        st.write("Pression atmosphérique")
        st.write("Le jeu de données contient 2 variables de pression atmosphérique très corrélées: Pressure9am, Pressure3pm.")

        st.write("")
        st.image(loadImage(chemin+"Pression1.png",1200))
        st.write("")

        st.write("On ne voit pas de tendance intra-day claire comme avec les températures ou la vitesse du vent.")

        # Affichage de la corrélation entre les variables de pressions
        st.write("")
        st.image(loadImage(chemin+"Pression2MatCorr.png",1200))
        st.write("")

        st.write("")
        st.write("La différence de pression intra-day est très peu corrélée aux pression 9am et 3pm ce qui tend à enrichir le jeu de données. Cette variable est donc à conserver.")
        st.write("")
        st.write("Nous conserverons la variable Pressure3pm que nous rempliront avec les valeurs de Pressure9am lorsque pour remplir les NaN."\
                 " L'ensemble sera enregistré dans une nouvelle colonne 'Pressure'.")

        st.write("")
        st.image(loadImage(chemin+"PressionDiagViolin.png",1200))
        st.write("")

        st.write("A la vue des violinplot et des valeurs numériques de moyennes, on tend à observer que la différence de pression sur les 3 derniers"\
                 " jours est optimale pour permettre d'observer un impact sur la variable cible. Il n'est cependant pas évident de dire s'il est plus"\
                     " pertinent de conserver DeltaP-1d, 2d ou 3d.")
        st.write("")
        st.write("On peut ainsi conserver Pressure et DeltaP_1d, DeltaP_2d DeltaP_3d. Nous étudierons l'importance des colonnes par la suite.")

    with st.expander("**Analyses de données pour la création des variables basées sur les 'différences d'humidité'**"):
        st.write("")
        st.image(loadImage(chemin+"DiffHumiditeMatCorr.png",1200))
        st.write("")
        
        st.write("Une variable intra-day sur Humidity semble intéressante car relativement peu corrélé aux autres variables. De la même manière"\
                " que pour les pressions, on peut regarder l'impact de l'évolution de l'humidité sur plusieurs jours.")
        st.write("")
        st.write("Nous conserverons la variable Humidity3pm que nous rempliront avec les valeurs de Humidity9am "\
                 "lorsque pour remplir les NaN. L'ensemble sera enregistré dans une nouvelle colonne Humidity.")

        st.write("")
        st.image(loadImage(chemin+"HumiditeDiagViolin.png",1200))
        st.write("")

        st.write("")
        st.write("A la vue des violinplot et des valeurs numériques de moyennes, on tend à observer que la différence"\
                 " d'humidité sur les 3 derniers jours est optimale pour permettre d'observer un impact sur la variable cible.")
        
        st.write("")
        st.write("Les DeltaH sont corrélés de manière similaire avec Humidity. La corrélation avec RainTomorrow s'améliore"\
                 " sensiblement à partir d'un delta de 2j.")
        st.write("")
        st.write("Il n'est pas évident de conclure, on peut conserver Humidity et DeltaH_1d, DeltaH_2d, DeltaH_3d afin d'étudier"\
                 " ultérieurement les colonnes les plus importantes.")


    with st.expander("**Analyses de données pour la création des variables 'WindDirInfluence' et ''consecutiveRainingDays'**"):
        st.write("")
        st.write("Influence de la direction du vent")
        st.write("Un vent venant de la mer apporte plus d'humidité qu'un vent venant des terres. La direction"\
                 " du vent devrait donc a priori apporter une information supplémentaire.")
        
        st.write("")
        st.image(loadImage(chemin+"FreqJoursPluieDarwin.png",1200))
        st.write("")
        
        st.write("On peut noter une forte influence de la direction du vent sur la probabilité de pluie le lendemain."\
                 " Par contre, il est clair que la distribution de probabilité en fonction de la direction du vent va être"\
                     " différente selon l'emplacement des villes puisque la côte se situe dans des directions différentes et que"\
                         " la ville peut être plus ou moins dans les terres avec une influence plus ou moins forte.")
        st.write("On va donc calculer cette distribution de probabilité pour chaque ville et l'utiliser comme feature supplémentaire.")
    
        st.write("")
        st.write("**Analyses de données pour le calcul de la variable 'consecutiveRainingDays'**")
        st.write("Jours de pluie consécutifs")
        st.write("Ajout du nombre de jours de pluie consécutif pour chaque ligne.")

        st.write("")
        st.image(loadImage(chemin+"DistribJoursPluieConsecutifs.png",1200))
        st.write("")
        
        st.write("On observe une décroissance exponentielle. Il y a peu de chance que le nombre de jours consécutif ait une"\
                 " influence sur le modèle. Dans le doute, on gardera la variable 'consecutiveRainingDays' pour le travail de machine learning.")
        
