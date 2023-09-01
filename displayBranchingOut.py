# displayBranchingOut

import streamlit as st
from resizeImage import resizeImage, loadImage

def displayBranchingOut():
    chemin = ".\\photosequipe\\"
    st.title("Bilan et suite du projet")
    st.write("")
    st.write("Le tableau suivant résume l’ordre de grandeur des meilleurs résultats que nous avons obtenus:")
    meilleursResultats = "<style>"\
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
        "<table><tr><td>Modèle</td><td>Metric (meilleurs scores)</td></tr>"\
        "<tr><td>Classification KNeighborsClassifier sans rééchantillonage</td><td>f1-score=0.66</td></tr>"\
        "<tr><td>Classification binaire avec rééchantillonage</td><td>f1-score=0.68</td></tr>"\
        "<tr><td>Régression</td><td>r2-score=0.35 // (f1-score=0.65)</td></tr>"\
        "<tr><td>Série temporelle</td><td>f1-score=0.61 (jusqu’à 0.8 selon la ville)</td></tr>"\
        "<tr><td>Réseau de neurones</td><td>f1-score=0.56</td></tr>"\
        "</table>"
    st.markdown(meilleursResultats,unsafe_allow_html = True)
    st.write("")

    col1, col2 = st.columns([0.2, 0.7])
    with col1 :
        imgarnaud = loadImage(chemin + "arnaud150.jpg", 150)
        st.image(imgarnaud)
        st.caption("Arnaud")
    
    with col2:
        st.write("")
        st.write("")
        st.write("*Il est intéressant de noter qu’avec quelques optimisations, nous avons chacun réussi à obtenir des scores similaires malgré"\
             " des méthodes de simulation très différentes.*")

    col3, col4 = st.columns([0.7, 0.2])
    with col3 :
        st.write("")
        st.write("")
        st.write("*Nos résultats montrent qu’avec du machine learning, nous réussissons à améliorer la prédiction par rapport à des modèles naïfs."\
             " Cependant, il semble malgré tout extrêmement compliqué de prédire avec précision la pluie à J+1 avec le jeu de données dont on dispose.*")
    with col4:
        imgpierre = loadImage(chemin + "pierre150.jpg", 150)
        st.image(imgpierre)
        st.caption("Pierre")

    col5, col6 = st.columns([0.2, 0.7])
    with col5 :
        imgolivier = loadImage(chemin + "olivier150.jpg", 150)
        st.image(imgolivier)
        st.caption("Olivier")
        
    with col6:
        st.write("")
        st.write("*Nos résultats sont malgré tout encourageants pour 2 raisons principales:*")
        st.write("*-	Nous avons réalisé ces études avec très peu de ressources*")
        st.write("*-	Les modélisations ont déjà permis -pour certaines villes- d’apporter des prédictions que nous jugeons utilisables.*")
        

    st.divider()
    st.write()
    
    st.write("Si **GrouseLager** *(notre hypothétique brasseur Australien)* souhaite utiliser nos modèles de prédiction, l’entreprise devra le faire en gardant à l’esprit que l’indice de confiance"\
             " est faible. Il doit s’agir là d’un outil utilisé en complément d’autres processus de décision. Cet outil ne se suffit pas à lui-même "\
                "pour prédire la pluie.")
    imgbeer = loadImage("AustralianBeer.jpg",1045)
    st.image(imgbeer)
    st.write("")
    st.divider()
    st.write("")
    st.write("**Pour aller plus loin, nous recommandons de financer:**")
    st.write("-	l’acquisition de bases de données météorologiques plus fournies en données exploitables, car cette base est notoirement lacunaire sur"\
             " certains types de données et les données présentes y sont extrêmement basiques.")
    st.write("-	des ressources matérielles conséquentes afin de pouvoir entrainer des modèles très complexes.")
    st.write("")
    st.divider()
    st.write("**Exemples de besoins permettant d’améliorer nos prédictions :**")
    st.write("-	Disposer d’un maillage bien plus conséquent de stations météorologiques. En effet, pour couvrir toutes l’Australie, nos données"\
             " ne proposent que 49 sites. De plus, ces sites sont très mal répartis sur le territoire.")
    st.write("-	Disposer d’un relevé toutes les 3 heures, au lieu d’un relevé par 24h avec quelques informations portant sur des phénomènes météorologiques"\
             "relevés dans la matinée (9h) et dans l’après-midi (15h). Ces informations augmenteraient le nombre de prévisions disponibles avec un pas de"\
                 " quelques heures au lieu d’une absence d’informations pendant 18h (de 15h à 9h le lendemain matin).")
    st.write("-	Nos données n’exploitent pas la troisième dimension. Il serait intéressant d’y ajouter des données topographiques, relevé/observation"\
             " multi-couche des colonnes d’air (utilisées par les modèles traditionnels de prévision).")
    st.write("-	Nous avons volontairement supprimé quelques variables en raison du trop grand nombre de valeurs manquantes. Pourtant, ces variables"\
             " semblaient assez bien corrélées à la variable cible; en particuliers la  variable “nébulosité totale”. De plus, la connaissance du type"\
                 " de nuages et de leur altitude estimée pourrait nous apporter de précieuses informations manquantes.")
    st.write("")
    st.divider()
    st.write("")
    
    col7, col8 = st.columns([0.7, 0.3])
    with col7 :
        st.write("**Nous pourrions acquérir différents types de relevés sur la couverture nuageuse, vus du sol mais avec un découpage à minima en 3 couches,"\
                " avec par exemple :**")
        
    with col8:
        imgdata = loadImage("datagouvfr.jpg",300)
        st.image(imgdata)
        
    st.write("- nébulosité des nuages de l’étage inférieur,")
    st.write("- hauteur de la base des nuages de l’étage inférieur,")
    st.write("- nébulosité de la couche inférieure,")
    st.write("- type de nuages de l’étage inférieur,")
    st.write("- type de nuages de l’étage moyen,")
    st.write("- type de nuages de l’étage supérieur,")
    st.write("- nébulosité totale")
    st.write("(référence : https://www.data.gouv.fr/fr/datasets/observation-meteorologique-historiques-france-synop/)")