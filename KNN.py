# KNN

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from resultatsKNNBokeh import resultatsKNNBokeh

def KNN():

    chemin = "./techniquesML/"
    displayBackground("#000000","#FFFFFF")
    st.header("KNeighborsClassifier")
    st.write("")
    st.write("Nous allons étudier ici le “classifier“ : KNeighborsClassifier et son utilisation dans le cadre de Kapy.")
    st.write("")

    def presentationParam():
        st.write("**Voici la liste des paramètres utilisés avec KNeighborsClassifier et leur valeur poiur cette première simulation:**")
        st.write("")
        parametres = "<style>"\
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
        "<table>"\
        "<tr><td style='width:10%;font-weight: bold;'>No</td><td style='width:25%;font-weight: bold;'>Paramètre</td>"\
            "<td style='width:65%;font-weight: bold;'>Remarques</td></tr>"\
        "<tr><td style='width:10%;'>1.</td><td style='width:25%;'>n_neighbors = 2</td><td style='width:65%;'>Indique le nombre "\
            "de voisins utilisés pour déterminer la classe de l'échantillon. Plus le nombre de voisins est grand et moins chaque"\
                " voisin impacte le résultat. Par contre, ceci peut diminuer la faciliter à classer les échantillons</td></tr>"\
        "<tr><td style='width:10%;'>2.</td><td style='width:25%;'>weights = 'uniform'</td><td style='width:65%;'>Indique le poids relatif "\
            "de chaque voisin. <p>Deux calculs sont proposés en standard :<p>'uniform' : chaque voisin a le même impact, <p>et 'distance' : l'"\
                "impact de chaque voisin est inversement proportionnel à sa distance</td></tr>"\
        "<tr><td style='width:10%;'>3.</td><td style='width:25%;'>algorithm = 'auto'</td><td style='width:65%;'>Indique l'algorithme utilisé"\
            " pour déterminer qui sont les k voisins. <p>Quatre options sont possibles : <p>'brute' : utilise la 'force brute' [temps de calcul ~DxN]"\
            "<p>'ball_tree': utilise l'algorithme BallTree [temps de calcul ~Dlog(N)],<p>'kd_tree': utilise l'algorithme KDTree"\
                " [D<15 : temps de calcul ~Dlog(N); D>15 temps de calcul ~DxN] <p>et 'Auto' : voir plus bas.</td></tr>"\
        "<tr><td style='width:10%;'>4.</td><td style='width:25%;'>leaf_size = 30</td><td style='width:65%;'>indique la taille des feuilles passées en paramètre"\
            " à l'algorithme BallTree ou KDTree. Impacte la durée des calculs. En effet, les algorithmes KD Tree et Ball Tree prévoient de passer"\
                " à 'force brute' sur les n derniers échantillons composant les feuilles au bout de l'arbre. Ici on repassera"\
                    " à brute force pour n= 30 (valeur conseillée). "\
                    "<p><a href=""https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/neighbors/_classification.py#L1"">"\
                    "Le code BallTree ou KDTree avec KNeighborsClassifier</a></td></tr>"\
        "<tr><td style='width:10%;'>5.</td><td style='width:25%;'>p = 2</td><td style='width:65%;'>Ce paramètre de 'puissance'"\
            " n’est utilisé que pour la distance de 'Minkowski'</td></tr>"\
        "<tr><td style='width:10%;'>6.</td><td style='width:25%;'>metric = 'minkowski'</td><td style='width:65%;'>Indique la métrique utilisée"\
            " pour calculer les distances entre chaque échantillon et n’importe lequel de ses voisins. Il existe diverses distances proposées"\
                " par ce classifier en standard. <p>(voir plus bas).</td></tr>"\
        "<tr><td style='width:10%;'>7.</td><td style='width:25%;'>metric_params = None</td><td style='width:65%;'>Paramètres fournis"\
            " lors de l'utilisation d'une fonction à la place des métriques disponibles.</td></tr>"\
        "<tr><td style='width:10%;'>8.</td><td style='width:25%;'>n_jobs = None</td><td style='width:65%;'>Ressources machines allouées"\
            " au calcul (-1 pour allouer toutes les ressources machines au calcul).</td></tr>"\
        "</table>"

        st.markdown(parametres,unsafe_allow_html = True)
        st.write("")
        st.write("")

    def paramAuto():
        st.write("**Remarque concernant le paramètre algorithm = 'auto':**")
        st.write("")
        st.write("ref : https://scikit-learn.org/stable/modules/neighbors.html#choice-of-nearest-neighbors-algorithm")
        st.write("")
        st.write("'Brute Force' est utilisé si n'importe laquelle de ces 5 conditions est vérifiée :")
        st.write("●	Faible nombre d'échantillons (ici : N>>10000 lignes)")
        st.write("●	La 'metric' utilisée n'est pas une metric 'precomputed' ! Cette condition est vérifiée si l'on fait appel à une fonction 'metric' que nous pourrions créer."\
                " Dans le cas présent, nous utilisons toutes les 'metrics' disponibles en 'standard', donc cette condition n'est pas vérifiée."\
                    " Lors d'un usage d'une metric créée par nos soins, cette condition sera vérifiée et donc Brute Force sera utilisée."\
                        " Ceci peut impacter négativement les temps de calcul si KD Tree est utilisé.")
        st.write("●	D>15, Une réduction de dimensions en dessous de 15 (Via une PCA par exemple) pourra donc considérablement diminuer la durée des calculs.")
        st.write("●	k>=N/2,")
        st.write("●	La 'metric' utilisée n'est pas dans la liste de 'valid_metrics' de Ball Tree ni de KD Tree."\
                "-1) KDTree.valid_metrics :'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity'"\
                "-2) BallTree.valid_metrics :'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity',"\
                    " 'seuclidean', 'mahalanobis', 'wminkowski', 'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard',"\
                        " 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'haversine', 'pyfunc'.")
        st.write("")
        st.write("Si 'Brute Force' n'est pas utilisé, alors c'est l'un des deux algorithmes 'kd_tree' ou 'ball-tree' qui sera utilisé.")

    def metricUtilisees():
        st.write("**Remarque concernant les metric utilisées :**")
        st.write("")
        st.write("Certaines ‘metric’ portent des noms différents mais font appel en réalité aux mêmes méthodes de calcul. Nos simulations permettent de bien le vérifier.")
        st.write("●	'manhattan' : metrics.pairwise.manhattan_distances")
        st.write("●	'cityblock' : metrics.pairwise.manhattan_distances")
        st.write("●	'l1' : metrics.pairwise.manhattan_distances")
        st.write("●	'euclidean' : metrics.pairwise.euclidean_distances")
        st.write("●	'l2' : metrics.pairwise.euclidean_distances")
        st.write("●	'haversine' : metrics.pairwise.haversine_distances")
        st.write("●	'cosine' : metrics.pairwise.cosine_distances")
        st.write("●	'nan_euclidean' : metrics.pairwise.nan_euclidean_distances")
        st.write("●	'minkowski' : utilise la distance de Minkowski")
        st.write("●	Pour la distance de Minkowsky, on peut résumer le calcul comme suit : ")
        st.write("")
        st.image(loadImage(chemin+"Minkowsky.jpg",1200))
        st.write("")
        st.write("●	Avec n, le nombre de dimensions, et p le facteur de puissance.")
        st.write("●	p=2 ⇒ norme standard (distance euclidienne), valeur par défaut")
        st.write("●	p=1 ⇒ distance de Manhattan")
        st.write("●	p=infini => distance de “chebyshev”")
        st.write("●	'chebyshev' : utilise la distance de Chebyshev")
        st.write("")
        st.write("")

    def bilanSimul1():
        st.write("**Bilan de cette première simulation**")
        st.write("")
        st.write("L'accuracy est de 84.52% > 77% du modèle 'Désertique'.")
        st.write("")
        st.write("Par contre, le F1-Score (53%) et le Recall (40.9%) sont 'assez bas'.")
        st.write("")
        st.write("")

    def explicationKNN():
        st.write("**Qu'est-ce que le 'KNeighborsClassifier' ?**")
        st.write("")
        st.write("Le KNeighborsClassifier est une méthode de classification des échantillons basée sur un calcul de la 'distance'"\
                 " entre un échantillon et ses plus proches voisins et puis par l'attribution de la classe sur la base"\
                     " du plus grand nombre de voisins ayant la même classe.")
        st.write("")
        st.write("ref : https://scikit-learn.org/stable/modules/neighbors.html#classification")
        st.write("")
        st.write("**KNeighborsClassifier sur RainTomorrow binaire**")
        st.write("Nous commençons par tester un modèle simple sur KNeighborsClassifier afin de prédire la variable binaire RainTomorrow.")
        st.write("")
        st.write("**Détail des paramètres pour KNeighborsClassifier**")
        st.write("Le classifier KNeighborsClassifier dispose de nombreux paramètres permettant d’optimiser les modélisations.")
        st.write("Nous allons passer en revue ces paramètres et la valeur attribuée pour cette première simulation.")
        st.write("Par la suite nous tenterons une optimisation puis une étude détaillée de certains éléments du paramétrage.")
        st.write("ref : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn-neighbors-kneighborsclassifier")
        st.write("")


    with st.expander("Qu'est-ce que le 'KNeighborsClassifier' et première simulation (Accuracy=0.84, F1-Score=0.53)"):    
        menuKNN = st.radio(
        " ", 
        ("Qu'est-ce que le 'KNeighborsClassifier' ?",
         "Présentation des paramètres de KNeighborsClassifier", 
         "Remarque concernant le paramètre algorithm = 'auto'", 
         "Remarque concernant les 'metric' utilisées", 
         "Commentaires sur cette première simulation KNN",)
        )

        if menuKNN == "Qu'est-ce que le 'KNeighborsClassifier' ?":
            explicationKNN()
        if menuKNN == "Présentation des paramètres de KNeighborsClassifier":
            presentationParam()
        elif menuKNN == "Remarque concernant le paramètre algorithm = 'auto'":
            paramAuto()
        elif menuKNN == "Remarque concernant les 'metric' utilisées":
            metricUtilisees()
        elif menuKNN == "Commentaires sur cette première simulation KNN":
            bilanSimul1()

    with st.expander("Simulations en faisant varier les principaux paramètres (Accuracy=0.86, F1-Score=0.65)"):
        st.write("")
        # fichier : KNN Simulations en faisant varier les principaux parametres.ipynb
        # indicateurs : _v1.0.csv ==> 'accuracy_v1.0.csv', 'precision_v1.0.csv', 'recall_v1.0.csv', 'f1score_v1.0.csv', 'specificity_v1.0.csv'
        metricsUtilisees = ['l1','l2','manhattan','nan_euclidean','minkowski','chebyshev','cityblock','cosine','euclidean']
        nbVoisinsMax = 51
        poids = ['uniform','distance']
        couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red']
        resultatsKNNBokeh("accuracy_v1.0.csv", "precision_v1.0.csv", "recall_v1.0.csv", "f1score_v1.0.csv", "specificity_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)









    
