# displayProjectBranchingIn

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def displayProjectBranchingIn() :

    displayBackground("#000000","#FFFFFF")

    st.title("Prévision météo en Australie")
    image_de_titre = loadImage("Uluru at sunset with four silhouettes with KAPY.png", 600)
    st.image(image_de_titre)
    st.caption("@Credits DALL-E 'Four silhouettes looking at Uluru rock at sunset with rain'")
    st.write("")
    st.markdown("Le nom de notre projet est :red[**‘KaPy’**] en référence à :red[**‘Kapi’**] qui signifie :red[**‘eau’**] pour les Aborigènes et :red[**Python**]"\
                " qui est le langage de programmation utilisé pour nos modélisations.")

    def displayIntroduction():
        st.header("Notre objectif")
        st.write("")
        st.write("Dans le cadre de la formation DataScientest - Data Scientist, nous formons une équipe de 4 apprentis venant d’horizons différents"\
                "mais avec la même envie de découvrir la data science.")
        st.write(":red[Notre objectif commun dans le cadre du projet Kapy sera de prédire les risques de précipitations en Australie en utilisant"\
                 " des techniques de machine learning.]")
        st.write("")
        st.write("Même si nous n’avons pas d’expertise particulière en météorologie, nous avons choisi ce sujet chacun pour des raisons autant liées à nos métiers"\
                " qu’à notre curiosité: ")
        
        table = "<style>"\
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
        "<table><tr><td>Olivier</td><td>Intérêt métier en particuliers sur les méthodes d’analyse et de prédiction des séries temporelles</td></tr>"\
        "<tr><td>Arnaud</td><td>Pas d’intérêt métier mais curieux de m’essayer au machine learning sur des données météos</td><tr>"\
        "<tr><td>Pierre</td><td>Pas d’utilisation possible directe des modélisations du projet KAPY. Par contre, les méthodes d’analyse de données"\
        " étudiées et les 'classifiers' utilisés pourront être repris dans un cadre professionnel (quelques applications au CRM dans le cadre"\
        " assurantiel : 'next best action', rebond commercial, actuariat, marketing et ciblage pour la prospection, prévenir les résiliations"\
        " par des actions de rétention ciblées)</td></tr>"\
        "<tr><td>Marion</td><td>Pas d’intérêt métier mais curiosité sur le sujet météo</td></tr></table>"

        st.markdown(table, unsafe_allow_html=True)

        st.write("")
        st.write("Afin de faire avancer le projet au mieux, nous nous sommes organisés différemment selon les phases du projet:")

        phasesProjet = "<ul><li>Exploration des données et feature engineering: travail en parallèle puis mise en commun nos résultats</li>"\
        "<li>Modélisation + interprétabilité: répartition des différents types de modélisations entre chacun d’entre nous:</li>"\
        "<ul style='padding-left:40px'><li>Classification binaire (KNeighborsClassifier, PCA, KMeans) : Pierre</li>"\
        "<li>Ré-échantillonnage (RandomUndersampling/RandomOverSampling sur LogisticRegression, KNN, DecisionTree, RandomForest): Marion</li>"\
        "<li>Classification multi-classe (Regression): Olivier</li>"\
        "<li>Régression (Echelle quotidienne, mensuelle, combinaison de modèles) : Arnaud</li>"\
        "<li>Série temporelle (ARIMA, SARIMA): Olivier</li>"\
        "<li>Réseaux de neurones récurents RNN : Olivier</li></ul>"\
        "<li>Conclusion: travail collectif</li></ul>"
        st.markdown(phasesProjet, unsafe_allow_html=True)



    def displayContextualisation():
        st.header("Exemple de contextualisation")
        st.write("")
        st.write("GrouseLager est un brasseur qui fabrique et distribue sur le territoire Australien. Les départements de production et d’achats"\
        " souhaitent intégrer les prédictions météorologiques afin d’optimiser les niveaux de production, besoins en matières premières et stockages")
        st.write("")
        st.write("L’entreprise a besoin de réduire ses coûts. Pour celà, elle identifie en particulier le besoin de mieux prévoir les périodes"\
        " de pluies/sécheresse afin d’éviter des coûts de production inutiles (irrigation, arrosage...) et de pallier efficacement à des périodes"\
        " non-pluvieuses qui pourraient impacter les récoltes négativement.")
        st.write("")
        st.write("")
        st.write("Cependant, la prédiction météorologique est particulièrement compliquée:")
        st.write(" “Les pluies en Australie posent un problème encore plus délicat que leur faible valeur moyenne : elles sont imprévisibles."\
        " Dans de nombreuses régions agricoles du monde, la saison à laquelle la pluie tombe est prévisible d’année en année [...] Sur la plus grande"\
        " partie du territoire australien, au contraire, les pluies dépendent de ce qu’on appelle l’ENSO (El Niño Southern Oscillation), à savoir le fait"
        " que la pluie n’est pas prévisible d’année en année sur une décennie et est encore plus imprévisible de décennie en décennie.” (Effondrement - Jared Diamond)")
        st.write("")
        st.write("“L'analyse à partir de données provenant d'une seule station météorologique est sans aucun doute la méthode d'analyse la plus ancienne"\
        " et la plus répandue. Elle est utilisée par le marin, le berger, l'agriculteur et l'homme de la rue, avec des résultats qui font souvent l'envie"\
        " du professionnel qui, dans certaines conditions, est susceptible d'être aveuglé par une masse de données météorologiques.”"\
        " (Principles of Meteorological Analysis, Chapitre 12 “Local Analysis” - Walter J. Saucier)")
        st.write("")
        st.write("Nous sommes donc missionnés par GrouseLager pour trouver un moyen de les aider dans cette tâche. Nous nous appuierons sur"\
        " un jeu de données contenant une dizaine d'années de relevé météorologique en Australie. ")
        st.write("")

    tab1, tab2 = st.tabs(["Introduction", "Exemple de contextualisation"])
    
    with tab1 :
        displayIntroduction()

    with tab2 :
        displayContextualisation()

        

    
