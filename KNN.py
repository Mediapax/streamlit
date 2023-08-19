# KNN

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
# from resultatsKNNBokeh import resultatsKNNBokeh

def KNN():

    chemin = ".\\techniquesML\\"
    displayBackground("#000000","#FFFFFF")
    st.header("KNeighborsClassifier")
    st.write("")
    st.write("Nous allons étudier ici le “classifier“ : KNeighborsClassifier et son utilisation dans le cadre de Kapy.")
    st.write("")

    def presentationParam():
        st.write("")
        st.write("**Présentation des paramètres de KNeighborsClassifier**")        
        st.write("Voici la liste des paramètres utilisés avec KNeighborsClassifier et leur valeur pour cette première simulation:")
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
        st.write("")
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
        st.write("")
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
        st.write("")
        st.write("**Bilan de cette première simulation**")
        st.write("")
        st.write("L'accuracy est de 84.52% > 77% du modèle 'Désertique'.")
        st.write("")
        st.write("Par contre, le F1-Score (53%) et le Recall (40.9%) sont 'assez bas'.")
        st.write("")
        st.write("")

    def explicationKNN():
        st.write("")
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
        st.write("")


    with st.expander("Qu'est-ce que le 'KNeighborsClassifier' et première simulation (Accuracy=0.84, F1-Score=0.53)"): 
        st.write("")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Intro.", "Les Paramètres", "algorithm = 'auto'", "les 'metrics'", "Bilan 1ère simulation"])
    
        with tab1 :
            explicationKNN()

        with tab2 :
            presentationParam()

        with tab3 :
            paramAuto()

        with tab4 :
            metricUtilisees()

        with tab5 :
            bilanSimul1()

    with st.expander("Simulations en faisant varier les principaux paramètres (Accuracy=0.86, F1-Score=0.65)"):
        st.write("")
        st.write("Avec toujours ce même jeu de données, nous allons faire varier dans nos simulations"\
                 " le nombre de voisins ainsi que le type de 'distance' utilisée et la pondération sur le calcul de la distance.")
        
        st.write("")
        tab1, tab2, tab3, tab4 = st.tabs(["Paramètres", "Simulations et Résultats", "Analyses", "Bilan"])

        with tab1 :
            st.write("")
            parametres1 = "<style>"\
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
                "<td style='width:65%;font-weight: bold;'>Plage de variation</td></tr>"\
            "<tr><td style='width:10%;'>1.</td><td style='width:25%;'>n_neighbors</td><td style='width:65%;'>2 - 50</td></tr>"\
            "<tr><td style='width:10%;'>2.</td><td style='width:25%;'>metric</td><td style='width:65%;'><p>'l1','l2','manhattan','nan_euclidean'"\
                ",'minkowski',<p>'chebyshev','cityblock','cosine','euclidean'</td></tr>"\
            "<tr><td style='width:10%;'>3.</td><td style='width:25%;'>weights</td><td style='width:65%;'>'uniform','distance'</td></tr>"\
            "</table>"

            st.markdown(parametres1,unsafe_allow_html = True)
            st.write("algorithm = 'auto', leaf_size = 30, p = 2, metric_params = None, n_jobs = None")
            st.write("")

        with tab2: 
            st.write("")
            st.write("**Durée de le simulation**")
            st.image(loadImage(chemin+"DureeSimulation1.jpg",300))
            st.write("")
            st.write("")
            st.write("**Sous Bokeh, nous obtenons une représentation graphique de nos résultats :**")
            st.write("")
            
            # fichier : KNN Simulations en faisant varier les principaux parametres.ipynb
            # indicateurs : _v1.0.csv ==> 'accuracy_v1.0.csv', 'precision_v1.0.csv', 'recall_v1.0.csv', 'f1score_v1.0.csv', 'specificity_v1.0.csv'
            metricsUtilisees = ['l1','l2','manhattan','nan_euclidean','minkowski','chebyshev','cityblock','cosine','euclidean']
            nbVoisinsMax = 51
            poids = ['uniform','distance']
            couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                        'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                        'goldenrod','navy','grey','darksalmon','red']
            #resultatsKNNBokeh("accuracy_v1.0.csv", "precision_v1.0.csv", "recall_v1.0.csv", "f1score_v1.0.csv", "specificity_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
            st.write("")

        with tab3 :
            st.write("")
            st.write("**Métrique 'Cosine'**")
            st.write("")
            st.write("Les meilleurs résultats obtenus avec cette métrique sont plus un peu plus faibles qu’avec d’autres métriques.")
            st.write("")
            st.image(loadImage(chemin+"CosineExplication.jpg",1200))
            st.write("")
            st.write("Plus les vecteurs pointent vers la même direction et plus les deux échantillons A et B sont proches. Comme cosinus(0) = 1,"\
                    " on prend la distance comme étant 1-cosinus(Théta). La 'distance de cosiné' n'est pas une véritable distance, car l'inégalité"\
                        " triangulaire n'est pas vérifiée.")
            st.write("")
            st.write("")
            st.write("**Métrique 'Manhattan'**")
            st.write("")
            st.write("On fait la différence entre les 'coordonnées' de chaque échantillon.")
            st.write("")
            st.image(loadImage(chemin+"ManhattanExplication.jpg",1200))
            st.write("")
            st.write("Dans Scikit Learn, Les distances L1, Cityblock et Manhattan sont toutes trois associées à la “metric” définie par metrics.pairwise.manhattan_distances."\
                    " Ceci explique les résultats identiques. Il est donc sans intérêt de lancer plusieurs fois la même simulation"\
                        " avec ces différentes distances qui fonty donc appel à la même 'metric'")
            st.write("")
            st.write("**La distance Manhattan obtient de meilleurs résultats**")
            st.write("Prenons un exemple avec les deux variables 'Rainfall' et 'WindGustSpeed'. La normalisation MinMax ramène l'étendue des "\
                  "valeurs possibles sur l'intervalle [0,1]. Une même variation sur Rainfall normée et sur WindGustSpeed normée correspond en"\
                     " réalité à des variations très différentes sur les valeurs physiques non normées.")
            st.write("Lors du calcul des plus proches voisins, on utilise la différence entre deux valeurs de 'Rainfall' et deux valeurs de 'WindGustSpeed'."\
                     " Ces deux différences ont la même importance pour calculer la distance entre voisins.")
            st.write("D'un point de vue physique, ces deux paramètres n'ont cependant pas du tout la même importance pour déterminer la valeur de 'RainTomorrow'."\
                     " Et les différences sur ces deux paramètres n'ont pas non plus la même importance pour déterminer la valeur de RainTomorrow.")
            st.write("KNN n’est pas pensé pour prendre en compte la nature physique des différentes variables. C’est à nous de préparer les diverses"\
                     " variables pour permettre une comparaison “agnostique” sur la base de “metric”.")
            st.write("●	La distance euclidienne (ou  minkowski (car p=2) ) pose problème, car elle présuppose une relation “géographique”"\
                     " entre les relevés météo. Ce n’est globalement pas le cas.")
            st.write("●	La distance cosiné présuppose également une relation “spatiale” entre les variables. Des relevés météo proches sont vus sous un angle"\
                     " solide proche. Ce n’est pas non plus le cas. De plus, le nombre important d'échantillons dans l’ensemble d'entraînement empêche"\
                         " l’apparition de directions privilégiées. Quelque soit la direction, on est proche d’un échantillon avec le label 1"\
                             " et d’un échantillon avec le label 0.")
            st.write("●	Enfin, la distance de Chebychev pose également problème. Cette distance suppose également une relation spatiale entre deux relevés"\
                     " météorologiques. On construit une “bulle” autour du relevé et les variables sont vues de manière équivalente. Ce n’est pas le cas.")
            st.write("**Il est donc logique que la “distance” qui donne le meilleur résultat soit une distance qui n'implique pas à priori une relation"\
                     " physique ou une “équi-pondération” entre les variables.**")
            st.write("Au final, la distance “Manhattan” reste la plus appropriée, **car ce n’est justement PAS une distance**, mais simplement un moyen pour "\
                     "indiquer la similarité entre deux relevés météo. Aucune hypothèse n’est faite sur l'équivalence entre telle ou telle autre variable.")
            st.write("Au final, la distance “Manhattan” reste la plus appropriée, car ce n’est justement PAS une distance, mais simplement un moyen"\
                     " pour indiquer la similarité entre deux relevés météo. Aucune hypothèse n’est faite sur l'équivalence entre telle ou telle autre variable.")
            st.write("")
            st.write("Cinq des 6 meilleurs résultats ont pour paramètre weight = Distance")
            st.write("Ceci signifie que les relevés météorologiques impliquant de la pluie ou du temps sec le lendemain sont proches les uns des autres et que"\
                     " le seul moyen pour obtenir une bonne labellisation de l’échantillon est d’appliquer une pondération inversement proportionnelle à la distance.")
            st.write("A partir de 3 voisins ou plus, les voisins supplémentaires apportent une perturbation sur le vote fourni par les trois voisins les plus proches."\
                    "Cette perturbation est minimisée en ajoutant une pondération inversement proportionnelle à la distance trouvée.")
            st.write("")
            st.write("Les meilleurs résultats sont obtenus pour le nombre de voisin k (2<k<9)")
            st.write("L'essentiel du vote est apporté par les premiers voisins et au-delà de 8 voisins, les votes supplémentaires des voisins plus éloignés apportent"\
                     " plutôt une perturbation plutôt qu’un renforcement positif des résultats.")
            st.write("")
            st.write("Les courbes représentant les métriques en fonction de k présentent toutes une forme d’accordéon quand le paramètre weights vaut “uniform”")
            st.write("Prenons l’exemple du F1-Score pour:")
            st.write("●	la metric Chebyshev / distance : uniform ")
            st.write("●	ou encore, la metric Manhattan / distance : distance")
            st.write("")
            st.image(loadImage(chemin+"accordeon.jpg",1200))
            st.write("")
            st.write("Si tous les voisins pris en compte ont le même poids, alors le fait de rajouter un voisin va déséquilibrer le vote."\
                     " De plus, avec un nombre impair de voisins, le vote est facile à effectuer. Avec un nombre pair de voisins et en cas"\
                         " d’égalité des votes, le label de classe est choisi en fonction de l'ordre des classes dans les données d'entraînement."\
                             " Par exemple, si la classe 0 apparaît en premier dans les données d'entraînement, le classifieur attribuera le label 0."\
                                 " Ceci explique la baisse de performance sur les valeurs paires pour Chebychev - uniform.")
            st.write("")
            
        
        with tab4:
            st.write("")
            st.write("Les meilleurs résultats sont ceux présentant la plus forte valeur de F1-score tout en gardant une Accuracy"\
                    " la plus importante. Il faut donc faire un compromis entre ces deux indicateurs. Nous choisissons 6 compromis parmi les meilleurs résultats.")
            st.write("")

            simu1MeilleursResultats = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Résultats</td><td style='width:50%;font-weight: bold;'>Paramètres de simulation</td></tr>"\
            "<tr><td style='width:50%;'><p>F1-Score : 0.641<p>Accuracy : 0.866<p>Recall : 0.545<p>Precision : 0.780<p>Specificity : 0.957</td><td style='width:50%;'><p>k = 8"\
            "<p>Metric = Manhattan ou L1 ou Cityblock<p>Weights = Distance</td></tr>"\
            "<tr><td style='width:50%;'><p>F1-Score : 0.648<p>Accuracy : 0.866<p>Recall : 0.561<p>Precision : 0.766<p>Specificity : 0.952</td><td style='width:50%;'><p>k = 6"\
            "<p>Metric = Manhattan ou L1 ou Cityblock<p>Weights = Distance</td></tr>"\
            "<tr><td style='width:50%;'><p>F1-Score : 0.642<p>Accuracy : 0.864<p>Recall : 0.556<p>Precision : 0.761<p>Specificity : 0.951</td><td style='width:50%;'><p>k = 5"\
            "<p>Metric = Manhattan ou L1 ou Cityblock<p>Weights = Distance</td></tr>"\
            "<tr><td style='width:50%;'><font color=""red"">"\
                "<p>F1-Score : 0.650<p>Accuracy : 0.862<p>Recall : 0.586<p>Precision : 0.731<p>Specificity : 0.939</font></td>"\
                    "<td style='width:50%;'><font color=""red""><p>k = 4<p>Metric = Manhattan ou L1 ou Cityblock<p>Weights = Distance</font></td></tr>"\
            "<tr><td style='width:50%;'><p>F1-Score : 0.645<p>Accuracy : 0.860<p>Recall : 0.579<p>Precision : 0.728<p>Specificity : 0.939</td><td style='width:50%;'><p>k = 3"\
            "<p>Metric = Manhattan ou L1 ou Cityblock<p>Weights = Distance</td></tr>"\
            "<tr><td style='width:50%;'><p>F1-Score : 0.642<p>Accuracy : 0.859<p>Recall : 0.575<p>Precision : 0.726<p>Specificity : 0.939</td><td style='width:50%;'><p>k = 3"\
            "<p>Metric = Manhattan ou L1 ou Cityblock<p>Weights = Uniform</td></tr>"\
            "</table>"

            st.markdown(simu1MeilleursResultats,unsafe_allow_html = True)
            st.write("algorithm = 'auto', leaf_size = 30, p = 2, metric_params = None, n_jobs = None")
            st.write("")
            st.write("L'Accuracy optimale est autour de 86% (Pour rappel, le Modèle 'Désertique' qui prédit qu'il ne pleut jamais présente une accuracy de 77%...).")
            st.write("Ici, c'est donc le F1-Score qui sera notre principal indicateur.")
            st.write("")
            st.write("**Le F1-Score maximum ne dépasse pas 0.65 et 0.86 pour l’accuracy maximale**")
            st.write("Nous proposons ici quelques explications :")
            st.write("●	Les variables utilisées pour KNN sont mal adaptées à une comparaison via une “distance”. Des variables parfaites pour KNN seraient"\
                     " pondérées entre elles en fonction directe de leur importance dans la détermination de la target et seraient comparables entre elles"\
                         " via le calcul de la distance. Ce n’est clairement pas facile d’imaginer de telles variables.")
            st.write("●	Les variables actuellement utilisées ne contiennent pas “toute l'information permettant de déterminer la valeur de Raintomorrow”.")
            st.write("●	Nous manquons de données sur les évolutions nuageuses. Connaître les types de nuages apportant la pluie est salutaire. C’est le réflexe"\
                     " habituel quand une personne sort de chez elle : si le ciel est nuageux et présente des nuages menaçants, il vaut mieux prendre un parapluie.")
            st.write("●	Nous manquons de données géographiques : un maillage du territoire avec un réseau plus dense de stations permettrait de connaître"\
                     " les endroits proches où il pleut et de suivre le déplacement des masses nuageuses apportant la pluie.")
            st.write("●	La fréquence des relevés météorologiques est trop faible : Nous manquons de relevés météorologiques fréquents. Disposer d'une information"\
                     " mise à jour régulièrement permettrait de suivre les changements de manière plus fine.")
            st.write("")



    with st.expander("Création d'une 'metric' personnalisée : 'distance_mi' (Accuracy=0.85, F1-Score=0.61)"):
        st.write("")
        st.write("A partir des remarques précédentes nous allons tenter de créer notre propre 'metric' afin d'utilisr la possibilité offerte par KNeighborsClassifier"\
                 " de passer par l'appel à une fonction personnalisée pour le calcul des distances entre un point est ses voisins.")
        st.write("")

        tab1, tab2, tab3, tab4 = st.tabs(["Définition de notre 'metric' personnalisée.", "Simulation", "Distance perso. et 'Brute Force'", "Bilan"])

        with tab1 :
            st.write("")
            st.write("**Propositions d'améliorations :**")
            st.write("")
            st.write("**Création d'une 'metric' personnalisée**")
            st.write("Nous nous proposons de créer une distance basée sur l'information mutuelle : la 'distance_mi'")
            st.write("**Contrainte technique : **KNeighborsClassifier permet de faire appel à une fonction 'metric' développée par nos soins. Cependant,"\
                     " cette fonction sera appelée à chaque calcul de distance. Il y aura très probablement des dizaines de milliers de calculs de distances."\
                         " Comme ce code externe s'exécute en code interprété Python, il est potentiellement très lent. Nous tentons ici d’utiliser une"\
                             " ‘metric’ facile à calculer. ")
            st.write("**-1) Nous commençons par prendre la 'metric' Manhattan**, car c’est la **'metrics'** ayant donné les meilleurs résultats sur la simulation"\
                     " précédente. De plus, le code de cette distance est simple. On fait la différence entre les coordonnées des deux échantillons,"\
                         " puis on prend la valeur absolue et enfin on additionne ces valeurs absolues.")
            st.write("La distance de Manhattan est basée sur la norme 'L1' (Taxicab norm) : somme des valeurs absolues des coordonnées.")
            st.write("L1 est bien un norme, car elle vérifie les trois propriétés :")
            st.write("●	Séparation : N(x) = 0 <=> x = 0")
            st.write("●	Homogénéité : : N(lambda * x) = abs(lambda) * N(x)")
            st.write("●	Inégalité triangulaire : N(x + y) <= N(x) + N(y)")
            st.write("La distance de Manhattan est créée à partir de cette norme et sera donc :")
            st.write("d(x,y) = N(x-y)")
            st.write("")
            st.write("**-2) Prise en compte d'une pondération :** les résultats de l'info mutuelle calculée sur l'ensemble d'entraînement utilisé pour entraîner"\
                     " le modèle (Par exemple 'X_train_minmax') permettent d'apporter un critère discriminant entre les variables. Plus l'information mutuelle"\
                         " sur une variable est grande et plus cette variable doit avoir de 'poids' sur le calcul de la distance.")
            st.write("")
            st.write("")
            st.write("**Qu'est-ce que l'information mutuelle ?**")
            st.write("Intuitivement, l'information mutuelle entre l'une des variables de notre jeu de données et la cible 'RainTomorrow' s'explique de la manière suivante :"\
                     " Plus l'information mutuelle est grande et plus la connaissance sur la valeur de la variable nous donne une connaissance sur la cible. Une IM de"\
                         " 0 indique que la variable et la cible sont totalement décorrélées. Une IM de 1 indique que la connaissance de la valeur de la variable"\
                             " permet de connaître parfaitement la valeur de la cible.")
            st.write("Il existe par ailleurs également un F1-test qui permet de mesurer le degré de linéarité entre une variable et la cible ")
            st.write("Pour plus d'information sur la gestion par Sckit-learn de l'information mutuelle et du F1-test, voir :"\
                     "https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py")
            st.write("")
            st.write("")
            st.image(loadImage(chemin+"IMdesVariables.jpg",400))
            st.write("")
            st.write("Nous nous proposons donc simplement de multiplier cette pondération sur chaque différence de variable. Ceci nous permettra de calculer"\
                     " la distance “metric” personnalisée.")
            st.write("**Remarque :** Ceci revient donc exactement au même que d’appliquer une pondération en amont sur les variables du jeu de données."\
                     " Mais dans notre cas, le calcul pondéré de la distance est effectué sur chaque couple (“échantillon”, voisin).")
            st.write("")
            st.write("La formule pour un calcul personnalisé  'distance_mi' sera donc : ")
            st.write("**Pour les échantillons A et B, sur la dimension i : abs( A(i) - B(i) ) * info mutuelle (i)**")
            st.write("")
            st.write("")



        with tab2 :
            st.write("")
            st.write("**Simulation avec la distance 'distance_mi':**")
            st.write("Comme vu plus haut, nous utilisons les paramètres :")
            st.write("●	leaf_size = 30 : propose un bon compromis")
            st.write("●	weights = 'distance'")
            st.write("●	metric = distance_mi")
            st.write("●	algorithm = 'auto'.")
            st.write("●	k = 4 : Afin de limiter les calculs et en utilisant la distance personnalisée  'distance_mi', nous choisissons k = 4 qui a donné"\
                     " de bons résultats avec la distance de Manhattan.")
            

        with tab3 :
            st.write("")
            st.write("Est-ce que 'brute force' sera utilisé ?")
            st.write("")
            st.write("Nous avons vu précédemment que si l'un de ces 5 critères est vérifié, alors 'Brute Force' est utilisé :")
            st.write("●	'peu de données' ? ==> faux,")
            st.write("●	metric = 'precomputed' ? ==> faux,")
            st.write("●	'D>15' ? ==> faux ")
            st.write("●	(D=15), 'k>N/2' ? ==> faux, ")
            st.write("●	le metric utilisée n'est pas dans la liste des métriques valides pour kd_tree ou ball_tree ? ==> VRAI.")
            st.write("**Donc 'brute force' sera utilisé.**")
            st.write("La 'metric' utilisée est une fonction que nous avons créée. Donc l'algorithme retenu par KNN sera automatiquement 'brute force'.")
            st.write("D'un point de vue technique, la fonction accepte en entrée deux vecteurs (Series) dont les coordonnées sont les variables"\
                     " (colonnes) de X_train ou de X_test. La fonction renvoie un scalaire (float).")
            
        with tab4 :
            st.write("")
            st.write("Analyse des résultats : 'distance_mi'")
            st.write("Les résultats obtenus sont les suivants :")
            st.write("")
            st.write("Matrice de confusion :")
            st.write("")
            st.image(loadImage(chemin+"MatConfusionDistance_mi.jpg",200))
            st.write("")
            simu3DistPerso = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Metric</td><td style='width:50%;font-weight: bold;'>Valeur trouvée</td></tr>"\
            "<tr><td style='width:50%;'><font color=""red"">accuracy</font></td><td style='width:50%;'><font color=""red"">0.8469965999244428</font></td></tr>"\
            "<tr><td style='width:50%;'><font color=""red"">f1-score</font></td><td style='width:50%;'><font color=""red"">0.6097848196124611</font</td></tr>"\
            "<tr><td style='width:50%;'>precision </td><td style='width:50%;'>0.6932814021421616</td></tr>"\
            "<tr><td style='width:50%;'>recall</td><td style='width:50%;'>0.5442384865278043</td></tr>"\
            "<tr><td style='width:50%;'>specificity </td><td style='width:50%;'>0.9322216245293168</td></tr>"\
            "</table>"
            st.markdown(simu3DistPerso,unsafe_allow_html = True)
            st.write("")
            st.write("Le f1-score obtenu n’est pas noté parmi les meilleurs. Les résultats sont décevants.")
            st.write("")
            st.write("**Durée du traitement importante :**")
            st.write("")
            st.image(loadImage(chemin+"DureeSimulationDistPerso.jpg",300))
            st.write("")
            st.write("Comme pressenti , la durée du traitement pour k= 4 est de 1h21 minutes ce qui est tout à fait prohibitif.")
            st.write("Nous ne recommandons donc pas d’utiliser une fonction personnalisée pour le calcul des distances pour Kapy.")
            st.write("")
            st.write("**Peut-on faire un usage “détourné” d’une fonction personnalisée ?**")
            st.write("Le classifieur KNeighborsClassifier utilisera l’algorithme “Brute Force” à chaque fois que la distance sera"\
                     " une distance personnalisée. Aussi nous recommandons de réserver l’usage d'une fonction personnalisée uniquement"\
                         " pour les petits jeux de données ou pour des ensembles de paramètres déjà optimisés.")
            st.write("En effet, utiliser une fonction personnalisée pour rechercher les meilleurs jeux de paramètres va provoquer une"\
                     " véritable explosion des temps de calcul pour la simulation.")
            st.write("On peut penser également à l’usage d'une fonction personnalisée en “détournant “ l’utilisation du KNeighborsClassifier."\
                     " En effet, la “Brute force” va amener le classifier à effectuer un ensemble de calculs par échantillon de l’ensemble de test."\
                         " Cet ensemble de calculs contiendra l’ensemble des échantillons de l’ensemble d'entraînement.")
            st.write("On aura donc au total N * M appels à la fonction personnalisée (N échantillons dans l'ensemble de test , M échantillons dans l’ensemble d'entraînement).")
            st.write("Enfin, cette fonction personnalisée n’a pas besoin d’être un calcul de distance. Par contre elle doit prendre en entrée"\
                     " un échantillon du jeu de test et un échantillon du jeu d'entraînement et renvoie un scalaire")
            st.write("")
            st.write("**Autres usages d’une fonction personnalisée :**")
            st.write("Dans le cadre de Kapy, l’usage de ce classifier avec une fonction personnalisée n’est pas adapté. Par contre on peut imaginer d’autres cas où"\
                     " cette fonction pourrait être utilisée. En recherchant dans la littérature, nous avons trouvé ces deux autres exemples :")
            st.write("")
            usageMetricPerso = "<style>"\
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
            "<tr><td style='width:30%;font-weight: bold;'>Autre fonction</td><td style='width:30%;font-weight: bold;'>Définition</td><td style='width:40%;font-weight: bold;'>Usage</td></tr>"\
            "<tr><td style='width:30%;'>Dynamic Time Warping (Distance dynamique temporelle)</td><td style='width:30%;'><p>from fastdtw import fastdtw"\
                "<p>def dtw_distance(x, y):<p>    distance, _ = fastdtw(x, y)<p>    return distance</td><td style='width:40%;'><p>Travail sur des séries"\
                     " temporelles<p><a href=""https://hal.science/hal-00647522/document"">Lien vers l'article</a></td></tr>"\
            "<tr><td style='width:30%;'>Distance de Jaccard</td><td style='width:30%;'><p>from scipy.spatial import distance<p>def jaccard_distance"\
                "(x, y):<p>    return distance.jaccard(x, y)</td><td style='width:40%;'><p>La distance de Jaccard peut être utilisée dans le traitement"\
                     " des données textuelles et la recommandation d’articles.<p><a href=""https://fr.wikipedia.org/wiki/Indice_et_distance_de_Jaccard"">Lien wikipedia</a><p>"\
                        "<a href=""https://www.arboretum.link/notes/similarit%C3%A9"">Lien vers l'article de arboretum</a></td></tr>"\
            "<tr><td style='width:30%;'>Distance pondérée par les variables (comme la tentative précédente)</td><td style='width:30%;'><p>def weighted_distance"\
                "(x, y):<p>    weights = [a, b, c, …]<p># Poids respectifs des attributs<p>    squared_diff = (x - y) ** 2<p>    weighted_squared_diff = squared_diff"\
                 " * weights<p>    return np.sqrt(np.sum(weighted_squared_diff))<p><p>ou encore dans le cas de Kapy : <p>def distance_mi_inv(x,y):<p>    ma_distance"\
                     " =  np.abs(x-y).dot(mi_inv)<p>    # mi_inv :  inv. de l’info mutuelle.<p>    return ma_distance</td><td style='width:40%;'><p>Si"\
                      " certaines variables sont plus importantes que d'autres dans la classification, nous pouvons définir une distance qui va attribuer"\
                         " des poids (weights = [a, b, c, …]) différents à ces variables lors du calcul de la distance.<p>⇒ notre projet Kapy et"\
                             " le KNeighborsClassifier</td></tr>"\
            "</table>"
            st.markdown(usageMetricPerso,unsafe_allow_html = True)
            st.write("")


    with st.expander("Création d'une fonction de pondération personnalisée : 'weights_sqr' **(Accuracy=0.87, F1-Score=0.66)**"):
        st.write("")
        st.write("Le classifier KNeighborsClassifier permet également de définir une fonction de pondération personnalisée à la place de la pondération 'distance'.")

        tab1, tab2 = st.tabs(["Définition de la fonction 'weights_sqr'", "Simulation, résultats, bilan"])

        with tab1 :
            st.write("")
            st.write("Création d'une fonction de pondérations 'weights_sqr' pour le paramètre weights")
            st.write("Nous avons vu un premier usage d'une fonction personnalisée pour calculer la distance entre deux points. "\
                     "Nous nous proposons maintenant de concevoir une fonction afin d'augmenter la capacité à différencier les voisins entre eux."\
                         " La fonction 'distance' propose d'appliquer un ratio inversement proportionnel à la distance calculée.")
            st.write("Nous allons tenter ici de définir une pondération qui ne varie pas en 1/x mais en 1/x²")
            st.write("Puis nous ferons une seconde simulation avec une pondération en 1/x² * 1/x².")
            st.write("Et enfin une troisième simulation avec une pondération en 1/x² * 1/x² * 1/x² * 1/x².")
            st.write("D'un point de vue technique, la fonction reçoit en entrée un tableau à une dimension contenant des distances et retourne un tableau")
            st.write("ayant les mêmes dimensions et contenant les distances pondérées.")
            st.write("")
            st.write("**Considérations techniques :**")
            st.write("La taille du tableau passé en paramètres varie en fonction de k. L’utilisation des divers algorithmes permettant d’éviter"\
                    "l’utilisation de “brute force” entraîne une variation de la taille du tableau.")
            st.write("Par ailleurs, l’appel à la fonction de pondération ne modifie que de façon très marginale la durée du traitement.")

        with tab2 :
            st.write("")
            st.write("Simulation, résultats et bilan")
            st.write("")

            # fichier : KNN Simulation avec la fonction de pondérations weights_sqr.ipynb
            # indicateurs : _weights_sqr_v1.0.csv ==> 'accuracy_weights_sqr_v1.0.csv, 'precision_weights_sqr_v1.0.csv', 'recall_weights_sqr_v1.0.csv', 'f1score_weights_sqr_v1.0.csv', 'specificity_weights_sqr_v1.0.csv'
            metricsUtilisees = ['manhattan']
            nbVoisinsMax = 21
            poids = ['weights_sqr','weights_sqr4','weights_sqr8']
            couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey','darksalmon','red']
            #resultatsKNNBokeh("accuracy_weights_sqr_v1.0.csv", "precision_weights_sqr_v1.0.csv", "recall_weights_sqr_v1.0.csv", "f1score_weights_sqr_v1.0.csv", "specificity_weights_sqr_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
            st.write("")
            st.write("Le meilleur compromis (f1-score, accuracy) est obtenu avec le couple (f1-score de 0.662, accuracy de 0.870) pour k= 8 et pour la"\
                     " fonction de pondération 'weights_sqr4'.")
            st.write("L’utilisation de la fonction de pondération **permet d’améliorer légèrement les résultats** : f1-score augmenté de 1.2% et l’accuracy augmentée de 0.4%.")
            usagePondPerso = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Meilleur Résultat</td><td style='width:50%;font-weight: bold;'>Paramètres de simulation</td></tr>"\
            "<tr><td style='width:50%;'><font color=""red""><p>F1-Score : 0.662<p>Accuracy : 0.870</font><p>Recall : 0.580<p>Precision : 0.770<p>Specificity : 0.951</td><td style='width:50%;'>"\
                "<p>k = 8<p>Metric = Manhattan<p>Weights =  'weights_sqr4'</td></tr>"\
            "</table>"
            st.markdown(usagePondPerso,unsafe_allow_html = True)
            st.write("")
            st.write("L’utilisation d’une distance plus discriminante permet justement de différencier des voisins proches de l’échantillon.")
            st.write("")
            st.write("")

    with st.expander("Modification du jeu de données via une pondération : Information Mutuelle (Accuracy=0.86, F1-Score=0.63)"):
        st.write("")

        tab1, tab2 = st.tabs(["Pondération par l'Information Mutuelle", "Résultats et bilan"])

        with tab1 :
            st.write("")
            st.write("L'utilisation de la fonction personnalisée distance_mi avait déjà fait appel à l'Information Mutuelle."\
                    " Ici, en simulant avec la distance 'Manhattan' sur un jeu de données pondéré par l'Information Mutuelle, on reproduit exactement le comportement de cette distance")
            st.write("Mais cette fois-ci, la 'Brute force' ne sera pas utilsée et du coup la simulations sera beaucpup plus rapide.")
            st.write("D'où l'idée : Pourquoi ne pas simuler tout un ensemble de cas sur un jeu de données pondéré par l'Information Mutuelle ?")
            st.write("")
            st.write("**Remarque :**")
            st.write("La somme de l'info. mutuelle de toutes les variables du jeu de données est : 0.6681604911298061")
            st.write("On obtient 95% de cette information mutuelle à partir de : 0.6347524665733157")
            st.write("En regardant le cumul de l'information mutuelle, on obtient 95% de toute l'information mutuelle disponible si l'on garde les variables:")
            st.write("")
            st.image(loadImage(chemin+"InfoMutuelle95pourcents.jpg",300))
            st.write("")
            st.write("Il pourrait être tentant de tronquer notre jeu de données et de nous limiter à ces 15 vbariables. Cependant, toute l'information Mutuelle disponible ne"\
                     " représente que 66% de l'information permettant de prédire la pluie. Tronquer encore notre jeu de données fera baisser nos résultats qui ne sont déjà pas très"\
                     " bons. Ceci nous donne cependant l'idée d'utiliser des méthodes optimisées pour réduire le nombre de dimensions, comme la PCA. Nous verrons celà plus loin.")
            
        with tab2 :
            st.write("")
            st.write("Analyse des résultats de la pondération des variables via l’information mutuelle")
            st.write("")
            # fichier : KNN Mutual Information - fonction distance_mi - puis simulation avec pondération du dataset.ipynb
            # indicateurs : _pond_var_inf_mut_v1.0.csv
            metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
            nbVoisinsMax = 21
            poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
            couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey']
            #resultatsKNNBokeh("accuracy_pond_var_inf_mut_v1.0.csv", "precision_pond_var_inf_mut_v1.0.csv", "recall_pond_var_inf_mut_v1.0.csv", "f1score_pond_var_inf_mut_v1.0.csv", "specificity_pond_var_inf_mut_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
            st.write("")
            st.write("")
            resultPonIM = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Meilleurs Résultats</td><td style='width:50%;font-weight: bold;'>Paramètres de simulation</td></tr>"\
            "<tr><td style='width:50%;'><font color=""red""><p>F1-Score : 0.627<p>Accuracy : 0.861</font><p>Recall : 0.532<p>Precision : 0.763<p>Specificity : 0.953</td><td style='width:50%;'>"\
                "<p>k = 11<p>Metric = manhattan<p>Weights = weights_sqr4</td></tr>"\
            "<tr><td style='width:50%;'><font color=""red""><p>F1-Score : 0.628<p>Accuracy : 0.860</font><p>Recall : 0.538<p>Precision : 0.753<p>Specificity : 0.950</td><td style='width:50%;'>"\
                "<p>k = 9<p>Metric = manhattan<p>Weights = weights_sqr4</td></tr>"\
            "</table>"
            st.markdown(resultPonIM,unsafe_allow_html = True)
            st.write("")
            st.write("**La pondération sur la base de l'Information Mutuelle ne permet pas d’améliorer les résultats.**")

    with st.expander("Utilisation d'une PCA sur KNN (Accuracy=0.87, F1-Score=0.65)"):
        st.write("")

        tab1, tab2, tab3, tab4 = st.tabs(["Principe", "PCA, KNN et Kapy", "Simulations", "Bilan : Meilleurs Résultats"])

        with tab1 :
            st.write("")
            st.write("Principe de la PCA")
            st.write("L'idée derrière l'analyse en Composantes Principales (PCA) est de considérer que chaque variable est une dimension"\
                     " d'analyse et que cette dimension contient un partie de l'information permettant de prédire la valeur de la cible.")
            st.write("La PCA est une technique qui permet d'extraire l'essentiel de l'information contenue dans ces n dimensions du jeu de"\
                     " données initial. Cette information est alors répartie sur un ensemble de ‘m’ composantes principales ('m<n') qui forment un nouveau jeu de données.")
            st.write("Grâce à cette transformation, en perdant un minimum d'information, on peut gagner un temps considérable pour la modélisation.")
            st.write("La technique mathématique sous-jacente consiste à trouver des transformations pour maximiser l’orthogonalité"\
                     " entre les composantes (des composantes qui ne sont pas corrélées linéairement). On utilise pour cela le concept"\
                         " de “variance expliquée”. La variance expliquée par une composante principale est équivalente à la"\
                             " “variance des données projetées sur cette composante”.")
            st.write("")
            st.write("")

        with tab2 :
            st.write("")
            st.write("PCA avec KNeighborsClassifier et Kapy")
            st.write("**Pourquoi utiliser une PCA avec KNeighborsClassifier ?** La durée des traitements pour KNN peut être assez longue.En utilisant un ePCA sur"\
                     " KNN on pourra donc raccourcir d'autant la durée de traitement et tenter plus de scénarios de simulation.")
            st.write("Ensuite, la PCA modifie les variables physiques initiales en créant des composantes sans valeur physique. Ceci s’adapte bien à"\
                     " KNeighborsClassifier, car cet algorithme, en particulier avec la distance de Manhattan, compare entre elles"\
                         " des variables de nature physique différentes. Ceci n’a aucun sens physique.")
            st.write("Alors autant abandonner tout espoir de relier l’algorithme KNeighborsClassifier à une explication physique"\
                     " sur les variables et concentrons-nous sur des algorithmes permettant de répartir la variance de"\
                         " l'information d'origine sur des axes indépendants.")
            st.write("")
            st.write("")
            st.write("**Recherche du nombre optimal de composantes à utiliser**")
            st.write("Nous fixons à 95% de variance expliquée le seuil pour déterminer le nombre de composantes PCA à garder.")
            st.write("")
            st.image(loadImage(chemin+"PCAnbCompPrincipales.jpg",1200))
            st.write("")
            st.write("On prendra donc les 12 premières composantes de la PCA.")
            st.write("")            

        with tab3 :
            st.write("")
            st.write("**Nos simulations**")
            st.write("")
            st.write("")
            st.write("**1.	Simulation PCA sur 12 composantes principales**")
            st.write("**Nombre de composantes retenues :** Nous allons garder les 12 composantes"\
                     " principales qui représentent 95% de la variance expliquée (à comparer aux 15 variables présentant 95% de l’information mutuelle).")
            st.write("Paramètres de la simulation :")
            st.write("Nous effectuons une série de simulation sur ce jeu de données avec les paramètres suivants : ")
            PCAs1 = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Paramètres</td><td style='width:50%;font-weight: bold;'>Plages de valeurs</td></tr>"\
            "<tr><td style='width:50%;'>metric</td><td style='width:50%;'>Manhattan, Pond_PCA*</td></tr>"\
            "<tr><td style='width:50%;'>weights</td><td style='width:50%;'><p>uniform, distance, weights_sqr, <p>weights_sqr4, weights_sqr8</td></tr>"\
            "<tr><td style='width:50%;'>k</td><td style='width:50%;'>1 à 20</td></tr>"\
            "</table>"
            st.markdown(PCAs1,unsafe_allow_html = True)
            st.write("")
            st.write("Pour les autres paramètres, les valeurs sont :")
            st.write("algorithm = 'auto', leaf_size = 30, p = 2, metric_params = None, n_jobs = None.")
            st.write("")
            st.write("***: Remarque concernant la métrique : Pond_PCA**")
            st.write("On a défini une fonction de pondération qui applique simplement une pondération égale à la variance expliquée de chaque variable."\
                     " Puis nous effectuons simplement un calcul de distance de Manhattan. Ceci revient donc à effectuer un travail préalable sur le"\
                         " jeu de données que l’on traite en appliquant une pondération sur chacune de ses variables. On gagne ainsi énormément"\
                             " de temps plutôt que d’appeler la fonction pour chaque calcul de distance.")
            st.write("")
            st.write("**Durée de la simulation :**")
            st.write("")
            st.image(loadImage(chemin+"DureePCAs1.jpg",300))
            st.write("")
            st.write("Quatre minutes pour un ensemble de simulations. La PCA permet vraiment d’abaisser "\
                     "la durée des simulations en diminuant le nombre de variables.")
            st.write("")
            st.write("PCA sur les 12 composantes principales : ")
            # fichier : KNN Simulations PCA avec reduction de dimensions.ipynb
            # indicateurs : _PCA_12c_v1.0.csv
            metricsUtilisees = ['manhattan','pond_PCA']
            nbVoisinsMax = 21
            poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
            couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey','darksalmon','red']
            #resultatsKNNBokeh("accuracy_PCA_12c_v1.0.csv", "precision_PCA_12c_v1.0.csv", "recall_PCA_12c_v1.0.csv", "f1score_PCA_12c_v1.0.csv", "specificity_PCA_12c_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
            st.write("")
            st.write("")
            st.write("**2.	Simulation PCA sur les 28 composantes principales (sans pondération)**")
            st.write("Nous effectuons deux nouvelles simulations basées sur la PCA, mais sans faire de réduction de dimension."\
                     " En effet, avec 95% de la variance expliquée ceci entraîne 5% de perte d’information.")
            st.write("Par ailleurs, nous faisons l’hypothèse que l’utilisation des composantes principales permet de ne plus faire"\
                     " référence à des variables physiques. Leur comparaison dans le calcul d’une distance fait alors sens. Nous allons voir ce qu’il en est. ")
            PCAs2 = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Paramètres</td><td style='width:50%;font-weight: bold;'>Plages de valeurs</td></tr>"\
            "<tr><td style='width:50%;'>metric</td><td style='width:50%;'>Manhattan, Pond_PCA</td></tr>"\
            "<tr><td style='width:50%;'>weights</td><td style='width:50%;'><p>uniform, distance, weights_sqr, <p>weights_sqr4, weights_sqr8</td></tr>"\
            "<tr><td style='width:50%;'>k</td><td style='width:50%;'>1 à 20</td></tr>"\
            "</table>"
            st.markdown(PCAs2,unsafe_allow_html = True)
            st.write("")
            st.write("Pour les autres paramètres, les valeurs sont :")
            st.write("algorithm = 'auto', leaf_size = 30, p = 2, metric_params = None, n_jobs = None.")
            st.write("")
            st.write("**Durée de la simulation :**")
            st.write("")
            st.image(loadImage(chemin+"DureePCAs2.jpg",300))
            st.write("")
            st.write("Environ 30 minutes pour 200 simulations. La limitation du nombre de voisins permet de gagner au final un précieux temps de calcul.")
            st.write("")
            st.write("PCA sans pondération sur les 28 composantes :")
            # fichier : KNN Simulations PCA sans reduc de dim.ipynb
            # indicateurs : _PCA_28c_v1.0.csv
            metricsUtilisees = ['manhattan','pond_PCA']
            nbVoisinsMax = 21
            poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
            couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey','darksalmon','red']
            #resultatsKNNBokeh("accuracy_PCA_28c_v1.0.csv", "precision_PCA_28c_v1.0.csv", "recall_PCA_28c_v1.0.csv", "f1score_PCA_28c_v1.0.csv", "specificity_PCA_28c_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
            st.write("")
            st.write("")
            st.write("**3.	Simulation PCA avec pondération sur les 28 composantes principales**")
            PCAs3 = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Paramètres</td><td style='width:50%;font-weight: bold;'>Plages de valeurs</td></tr>"\
            "<tr><td style='width:50%;'>metric</td><td style='width:50%;'><p>manhattan', nan_euclidean, minkowski,<p>chebyshev, cosine, euclidean</td></tr>"\
            "<tr><td style='width:50%;'>weights</td><td style='width:50%;'><p>uniform, distance, weights_sqr, <p>weights_sqr4, weights_sqr8</td></tr>"\
            "<tr><td style='width:50%;'>k</td><td style='width:50%;'>1 à 20</td></tr>"\
            "</table>"
            st.markdown(PCAs3,unsafe_allow_html = True)
            st.write("")
            st.write("Pour les autres paramètres, les valeurs sont :")
            st.write("algorithm = 'auto', leaf_size = 30, p = 2, metric_params = None, n_jobs = None.")
            st.write("")
            st.write("")
            st.write("**Durée de la simulation :**")
            st.write("")
            st.image(loadImage(chemin+"DureePCAs3.jpg",300))
            st.write("")
            st.write("Environ 6h30 minutes pour 600 simulations.")
            st.write("")
            st.write("PCA avec pondération sur les 28 composantes :")
            # fichier : KNN Simulations PCA sans reduc de dim.ipynb
            # indicateurs : _pond_PCA_28c_v1.0.csv
            metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
            nbVoisinsMax = 21
            poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
            couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey']
            #resultatsKNNBokeh("accuracy_pond_PCA_28c_v1.0.csv", "precision_pond_PCA_28c_v1.0.csv", "recall_pond_PCA_28c_v1.0.csv", "f1score_pond_PCA_28c_v1.0.csv", "specificity_pond_PCA_28c_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
            st.write("")
            st.write("")

        with tab4 :
            st.write("")
            st.write("Voici les meilleurs résultats pour chaque simulation")
            st.write("")
            st.write("**1.	Simulation PCA sur 12 composantes principales**")
            PCAresultats1 = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Meilleurs Résultats</td><td style='width:50%;font-weight: bold;'>Paramètres de Simulation</td></tr>"\
            "<tr><td style='width:50%;'><p>F1-Score : 0.611<p>Accuracy : 0.854<p>Recall : 0.524<p>Precision : 0.733<p>Specificity : 0.946</td><td style='width:50%;'>"\
                "<p>k = 9<p>Metric = pond_PCA<p>Weights =  distance</td></tr>"\
            "<tr><td style='width:50%;'><p>F1-Score : 0.610<p>Accuracy : 0.855<p>Recall : 0.517<p>Precision : 0.742<p>Specificity : 0.949<p></td>"\
                "<td style='width:50%;'><p>k = 16<p>Metric = pond_PCA<p>Weights =  weights_sqr</td></tr>"\
            "</table>"
            st.markdown(PCAresultats1,unsafe_allow_html = True)
            st.write("")
            st.write("Les meilleures performances obtenues avec la PCA sont légèrement inférieures aux meilleures performances obtenues avec"\
                     " les autres simulations. Ceci est normal, car on a perdu 5% de la variance expliquée est réduisant à 12 composantes.")
            st.write("Par contre, on peut observer que la pondération 'pond_PCA' permet d’obtenir les meilleurs simulations.")
            st.write("")
            st.write("")


            st.write("**2.	Simulation PCA sur les 28 composantes principales (sans pondération)**")
            PCAresultats2 = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Meilleurs Résultats</td><td style='width:50%;font-weight: bold;'>Paramètres de Simulation</td></tr>"\
            "<tr><td style='width:50%;'>"\
                "<font color=""red""><p><b>f1-score : 0.648</b>"\
                "<p><b>accuracy : 0.866</b></font>"\
                "<p>recall : 0.563<p>precision : 0.765"\
                "<p>specificity : 0.951</td>"\
                "<td style='width:50%;'><p>k : 10<p>metric : manhattan<p>weights : weights-sqr4</td></tr>"\
            "</table>"
            st.markdown(PCAresultats2,unsafe_allow_html = True)
            st.write("")
            st.write("Comme le montre ces résultats, l'utilisation de la PCA sans diminution de dimensions renforce légèrement la capacité"\
                     " à discriminer les deux classes. Il s'agit de notre meilleur résultat avec PCA.")
            st.write("")
            st.write("")


            st.write("**3.	Simulation PCA avec pondération sur les 28 composantes principales**")
            PCAresultats3 = "<style>"\
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
            "<tr><td style='width:50%;font-weight: bold;'>Meilleurs Résultats</td><td style='width:50%;font-weight: bold;'>Paramètres de Simulation</td></tr>"\
            "<tr><td style='width:50%;'><p>f1-score : 0.623<p>accuracy : 0.856<p>recall : 0.543<p>precision : 0.731<p>specificity : 0.944</td>"\
                "<td style='width:50%;'><p>k : 15<p>metric : manhattan<p>weights : weights_sqr4</td></tr>"\
            "</table>"
            st.markdown(PCAresultats3,unsafe_allow_html = True)
            st.write("")
            st.write("Ce résultat est moins bon qu’avec la PCA non pondérée.")
            st.write("")
            st.write("")
            st.write("**Bilan des PCA avec KNeighborsClassifier :**")
            st.write("L’utilisation de la PCA n’apporte pas de plus-value pour KNeighborsClassifier dans le cadre de Kapy."\
                     " Nous devons donc continuer à utiliser la PCA dans son cadre habituel, par exemple le traitement d'images. "\
                        "Ceci permet de diminuer de manière très notoire la durée des simulations,"\
                             " au prix toutefois d’une performance amoindrie. Dans notre cas, le F1-Score est déjà très bas, aussi"\
                                 " une dégradation additionnelle des résultats au profit d'un gain de temps ne semble pas pertinente.")



    with st.expander("Utilisation de KNeighborsClassifier sur un KMeans (Accuracy=0.83, F1-Score=0.66)"):
        st.write("")
        tab1, tab2, tab3  = st.tabs(["Principe du KMeans", "Simulations", "Bilan"])

        with tab1 :
            st.write("")
            st.write("**Principe du KMeans :**")   
            st.write("L’algorithme KMeans est une manière élégante de déterminer k centroïdes à partir d’un ensemble d’échantillons."\
                     " La valeur k est fournie en paramètre.")
            st.write("Voici les étapes suivies par l’algorithme :")
            st.write("●	1) L’algorithme va déterminer de manière aléatoire k centroïdes pour initialiser la construction des k clusters")
            st.write("●	2) Un calcul de distance est effectué entre chaque échantillon et chacun des k centroïdes. Chaque échantillon est alors affecté au cluster le plus proche.")
            st.write("●	3) Un nouveau centroïde est calculé comme étant la résultante de la moyenne de tous les échantillons appartenant au même cluster.")
            st.write("●	4) Répéter les étapes 2 et 3 jusqu’à ce que le calcul de cluster devienne stable ou bien jusqu’à"\
                     " ce que le nombre d’itération (fourni en paramètre, par défaut : 300).")
            st.write("●	5) Calculer la variance observée au sein de chaque cluster (somme des variances)")
            st.write("●	6) Répéter les étapes 1 à 5 jusqu’à atteindre un minimum de somme de variances")
            st.write("")
            st.write("L’algorithme va donc se stabiliser sur des minima de variance, mais ces minima ne sont pas automatiquement des minima absolus de variance.") 


        with tab2 :
            st.write("")
            st.write("**Simulations effectuées :**")
            st.write("Nous allons utiliser KMeans pour diminuer le nombre d’échantillons de l’ensemble d'entraînement et rééquilibrer le jeu de données.")
            st.write("")
            st.write("A chaque fois, nous créons un jeu de centroïdes composés de deux ensembles de centroïdes : ")
            st.write("●	A partir des échantillons d'entraînement avec le label 0, nous créons un ensemble de centroïdes  avec le label 0 (pas de pluie demain).")
            st.write("●	A partir des échantillons d'entraînement avec le label 1, nous créons un ensemble de centroïdes  avec le label 1 (pluie demain).")
            st.write("En rassemblant ces deux groupes de centroïdes, on forme le nouvel ensemble d'entraînement.Nous pouvons donc jouer "\
                    "à la fois sur le nombre total de centroïdes mais aussi sur le ratio entre chaque label. Nous pouvons ainsi rééquilibrer le jeu de données.")
            st.write("")
            st.write("Pour chaque simulation, nous faisons varier les trois principaux paramètres : ")
            st.write("●	k, le nombre de voisins varie entre 1 et 20,")
            st.write("●	metric : toutes les métriques proposées en standard")
            st.write("●	weights : uniform et distance plus les 3 fonctions de pondération personnalisées : 'weights_sqr', 'weights_sqr4' et 'weights_sqr8'.")
            st.write("")
            st.write("Nous pouvons nous permettre de jouer toutes ces simulations, car l’un des principaux intérêts de KMeans est de réduire la "\
                    "taille des échantillons d'entraînement, tout en conservant dans chaque centroïde (nous l’espérons) le"\
                        " pattern des échantillons composant son cluster.")
            st.write("")

            soustab2, soustab3, soustab4,soustab5,soustab6,soustab7,soustab1,  = st.tabs(["400s/600p","200s/800p","600s/400p","50s/50p","15s/15p","100s/100p","500s/500p"])

            with soustab1:
                st.write("")
                st.write("**KNeighborsClassifier sur 500 centroïdes sec et 500 centroïdes pluie**")
                st.write("")
                st.write("**Durée de la simulation :**")
                st.write("")
                st.image(loadImage(chemin+"duree500s500p.jpg",300))
                st.write("")
                st.write("**Résultats :**")
                st.write("")
                # fichier : KaPy - Reequilibrage des donnees - ClusterCentroids - KNN local.ipynb
                # indicateurs : _kmeans_500s_500p_v1.0.csv
                metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
                nbVoisinsMax = 21
                poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
                couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey']
                #resultatsKNNBokeh("accuracy_kmeans_500s_500p_v1.0.csv", "precision_kmeans_500s_500p_v1.0.csv", "recall_kmeans_500s_500p_v1.0.csv", "f1score_kmeans_500s_500p_v1.0.csv", "specificity_kmeans_500s_500p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
                st.write("")
                KmeansResults1 = "<style>"\
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
                "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                    "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                    "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                    "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
                "<tr><td style='width:25%;'><font color=""red""><p>500 centroïdes sec<p>500 centroïdes pluie</font></td>"\
                    "<td style='width:25%;'><font color=""red""><p><b>F1-Score : 0.662</b><p><b>accuracy : 0.833</b><p>Recall : 0.742<p>Precision : 0.597<p>Specificity : 0.859</font></td>"\
                    "<td style='width:25%;'><font color=""red""><p>k = 15<p>Metric = euclidean<p>Weights =  weights_sqr4<p></font></td>"\
                    "<td style='width:25%;'><font color=""red""><p><b>Meilleur f1-score de toutes nos simulations</b><p>La metric manhattan sous-performe</td></tr></font>"\
                "</table>"
                st.markdown(KmeansResults1,unsafe_allow_html = True)
                st.write("")
                st.write("")

            with soustab2:
                st.write("")
                st.write("**KNeighborsClassifier sur 400 centroïdes sec et 600 centroïdes pluie**")
                st.write("")
                st.write("**Durée de la simulation :**")
                st.write("")
                st.image(loadImage(chemin+"duree400s600p.jpg",300))
                st.write("")
                st.write("**Résultats :**")
                st.write("")
                # fichier : KaPy - Reequilibrage des donnees - ClusterCentroids - KNN local.ipynb
                # indicateurs : _kmeans_400s_600p_v1.0.csv
                metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
                nbVoisinsMax = 21
                poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
                couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey']
                #resultatsKNNBokeh("accuracy_kmeans_400s_600p_v1.0.csv", "precision_kmeans_400s_600p_v1.0.csv", "recall_kmeans_400s_600p_v1.0.csv", "f1score_kmeans_400s_600p_v1.0.csv", "specificity_kmeans_400s_600p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
                st.write("")
                KmeansResults2 = "<style>"\
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
                "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                    "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                    "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                    "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
                "<tr><td style='width:25%;'><p>400 centroïdes sec<p>600 centroïdes pluie</td>"\
                    "<td style='width:25%;'><p>F1-Score : 0.634<p>Accuracy : 0.816<p>Recall : 0.726<p>Precision : 0.563<p>Specificity : 0.842</td>"\
                    "<td style='width:25%;'><p>k = 6<p>Metric = euclidean<p>Weights = uniform</td>"\
                    "<td style='width:25%;'></td></tr>"\
                "</table>"
                st.markdown(KmeansResults2,unsafe_allow_html = True)
                st.write("")
                st.write("")

            with soustab3:
                st.write("")
                st.write("**KNeighborsClassifier sur 200 centroïdes sec et 800 centroïdes pluie**")
                st.write("")
                st.write("**Durée de la simulation :**")
                st.write("")
                st.image(loadImage(chemin+"duree200s800p.jpg",300))
                st.write("")
                st.write("**Résultats :**")
                st.write("")
                # fichier : KaPy - Reequilibrage des donnees - ClusterCentroids - KNN local.ipynb
                # indicateurs : _kmeans_200s_800p_v1.0.csv
                metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
                nbVoisinsMax = 21
                poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
                couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                'goldenrod','navy','grey']
                #resultatsKNNBokeh("accuracy_kmeans_200s_800p_v1.0.csv", "precision_kmeans_200s_800p_v1.0.csv", "recall_kmeans_200s_800p_v1.0.csv", "f1score_kmeans_200s_800p_v1.0.csv", "specificity_kmeans_200s_800p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
                st.write("")
                KmeansResults3 = "<style>"\
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
                "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                    "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                    "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                    "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
                "<tr><td style='width:25%;'><p>200 centroïdes sec<p>800 centroïdes pluie</td>"\
                    "<td style='width:25%;'><p>F1-Score : 0.606<p>Accuracy : 0.800<p>Recall : 0.702<p>Precision : 0.534<p>Specificity : 0.827</td>"\
                    "<td style='width:25%;'><p>k = 2<p>Metric = cosine<p>Weights = uniform</td>"\
                    "<td style='width:25%;'></td></tr>"\
                "</table>"
                st.markdown(KmeansResults3,unsafe_allow_html = True)
                st.write("")
                st.write("")

            with soustab4:
                st.write("")
                st.write("**KNeighborsClassifier sur 600 centroïdes sec et 400 centroïdes pluie**")
                st.write("")
                st.write("**Durée de la simulation :**")
                st.write("")
                st.image(loadImage(chemin+"duree600s400p.jpg",300))
                st.write("")
                st.write("**Résultats :**")
                st.write("")
                # fichier : KaPy - Reequilibrage des donnees - ClusterCentroids - KNN local.ipynb
                # indicateurs : _kmeans_600s_400p_v1.0.csv
                metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
                nbVoisinsMax = 21
                poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
                couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey']
                #resultatsKNNBokeh("accuracy_kmeans_600s_400p_v1.0.csv", "precision_kmeans_600s_400p_v1.0.csv", "recall_kmeans_600s_400p_v1.0.csv", "f1score_kmeans_600s_400p_v1.0.csv", "specificity_kmeans_600s_400p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
                st.write("")
                KmeansResults4 = "<style>"\
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
                "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                    "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                    "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                    "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
                "<tr><td style='width:25%;'><p>600 centroïdes sec<p>400 centroïdes pluie</td>"\
                    "<td style='width:25%;'><p>F1-Score : 0.660<p>Accuracy : 0.855<p>Recall : 0.642<p>Precision : 0.679<p>Specificity : 0.914</td>"\
                    "<td style='width:25%;'><p>k = 14<p>Metric = cosine<p>Weights = weights_sqr</td>"\
                    "<td style='width:25%;'><p><b>cosine :</b> il existe des directions privilégiées pour les clusters"\
                        "<p>Equilibrage du jeu de données : un léger rééquilibrage permet d’améliorer les résultats</td></tr>"\
                "</table>"
                st.markdown(KmeansResults4,unsafe_allow_html = True)
                st.write("")
                st.write("")

            with soustab5:
                st.write("")
                st.write("**KNeighborsClassifier sur 50 centroïdes sec et 50 centroïdes pluie**")
                st.write("")
                st.write("**Durée de la simulation :**")
                st.write("")
                st.image(loadImage(chemin+"duree50s50p.jpg",300))
                st.write("")
                st.write("**Résultats :**")
                st.write("")
                # fichier : KaPy - Reequilibrage des donnees - ClusterCentroids - KNN local.ipynb
                # indicateurs : _kmeans_50s_50p_v1.0.csv
                metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
                nbVoisinsMax = 21
                poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
                couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey']
                #resultatsKNNBokeh("accuracy_kmeans_50s_50p_v1.0.csv", "precision_kmeans_50s_50p_v1.0.csv", "recall_kmeans_50s_50p_v1.0.csv", "f1score_kmeans_50s_50p_v1.0.csv", "specificity_kmeans_50s_50p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
                st.write("")
                KmeansResults5 = "<style>"\
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
                "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                    "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                    "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                    "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
                "<tr><td style='width:25%;'><p>50 centroïdes sec<p>50 centroïdes pluie</td>"\
                    "<td style='width:25%;'><p>F1-Score : 0.631<p>Accuracy : 0.807<p>Recall : 0.750<p>Precision : 0.545<p>Specificity : 0.824</td>"\
                    "<td style='width:25%;'><p>k = 8<p>Metric = euclidean<p>Weights = weights_sqr4</td>"\
                    "<td style='width:25%;'></td></tr>"\
                "</table>"
                st.markdown(KmeansResults5,unsafe_allow_html = True)
                st.write("")
                st.write("")

            with soustab6:
                st.write("")
                st.write("**KNeighborsClassifier sur 15 centroïdes sec et 15 centroïdes pluie**")
                st.write("")
                st.write("**Durée de la simulation :**")
                st.write("")
                st.image(loadImage(chemin+"duree15s15p.jpg",300))
                st.write("")
                st.write("**Résultats :**")
                st.write("")
                # fichier : KaPy - Reequilibrage des donnees - ClusterCentroids - KNN local.ipynb
                # indicateurs : _kmeans_15s_15p_v1.0.csv
                metricsUtilisees = ['manhattan','nan_euclidean','minkowski','cosine','euclidean'] # 'chebyshev',
                nbVoisinsMax = 21
                poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
                couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey']
                #resultatsKNNBokeh("accuracy_kmeans_15s_15p_v1.0.csv", "precision_kmeans_15s_15p_v1.0.csv", "recall_kmeans_15s_15p_v1.0.csv", "f1score_kmeans_15s_15p_v1.0.csv", "specificity_kmeans_15s_15p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
                st.write("")
                KmeansResults6 = "<style>"\
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
                "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                    "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                    "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                    "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
                "<tr><td style='width:25%;'><p>15 centroïdes sec<p>15 centroïdes pluie</td>"\
                    "<td style='width:25%;'><p>F1-Score : 0.620<p>Accuracy : 0.789<p>Recall : 0.727<p>Precision : 0.513<p>Specificity : 0.806</td>"\
                    "<td style='width:25%;'><p>k = 7<p>Metric = Manhattan<p>Weights = weights_sqr</td>"\
                    "<td style='width:25%;'><p>* : Voir ci-dessous</td></tr>"\
                "</table>"
                st.markdown(KmeansResults6,unsafe_allow_html = True)
                st.write("")
                st.write("Pour le calcul “k=13--metric=chebyshev--weights=uniform'”, on se retrouve avec une matrice de confusion totalement déformée:")
                st.write("")
                st.image(loadImage(chemin+"chebyshevDeforme.jpg",150))
                st.write("")

            with soustab7:
                st.write("")
                st.write("**KNeighborsClassifier sur 100 centroïdes sec et 100 centroïdes pluie**")
                st.write("")
                st.write("**Durée de la simulation :**")
                st.write("")
                st.image(loadImage(chemin+"duree100s100p.jpg",300))
                st.write("")
                st.write("**Résultats :**")
                st.write("")
                # fichier : KaPy - Reequilibrage des donnees - ClusterCentroids - KNN local.ipynb
                # indicateurs : _kmeans_100s_100p_v1.0.csv
                metricsUtilisees = ['manhattan','nan_euclidean','minkowski','chebyshev','cosine','euclidean']
                nbVoisinsMax = 21
                poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
                couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red','magenta','maroon','mediumaquamarine','mediumblue',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey']
                #resultatsKNNBokeh("accuracy_kmeans_100s_100p_v1.0.csv", "precision_kmeans_100s_100p_v1.0.csv", "recall_kmeans_100s_100p_v1.0.csv", "f1score_kmeans_100s_100p_v1.0.csv", "specificity_kmeans_100s_100p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
                st.write("")
                KmeansResults7 = "<style>"\
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
                "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                    "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                    "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                    "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
                "<tr><td style='width:25%;'><p>100 centroïdes sec<p>100 centroïdes pluie</td>"\
                    "<td style='width:25%;'><p>F1-Score : 0.649<p>Accuracy : 0.830<p>Recall : 0.714<p>Precision : 0.594<p>Specificity : 0.863</td>"\
                    "<td style='width:25%;'><p>k = 6<p>Metric = euclidean<p>Weights = weights_sqr</td>"\
                    "<td style='width:25%;'></td></tr>"\
                "</table>"
                st.markdown(KmeansResults7,unsafe_allow_html = True)
                st.write("")
                st.write("")
            
        
        with tab3 :
            st.write("**Meilleurs Résultats :**")
            st.write("Nous résumons les principaux résultats obtenus dans le tableau ci-dessous. A chaque fois,"\
                     " nous choisissons “visuellement” la simulation proposant le meilleur compromis entre Accuracy et F1-Score.")
            KmeansResults = "<style>"\
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
            "<tr><td style='width:25%;font-weight: bold;'>Simulation</td>"\
                "<td style='width:25%;font-weight: bold;'>Meilleur Résultat</td>"\
                "<td style='width:25%;font-weight: bold;'>Paramètres</td>"\
                "<td style='width:25%;font-weight: bold;'>Remarque</td></tr>"\
            "<tr><td style='width:25%;'><font color=""red""><p>500 centroïdes sec<p>500 centroïdes pluie</font></td>"\
                "<td style='width:25%;'><font color=""red""><p><b>F1-Score : 0.662</b><p><b>accuracy : 0.833</b><p>Recall : 0.742<p>Precision : 0.597<p>Specificity : 0.859</font></td>"\
                "<td style='width:25%;'><font color=""red""><p>k = 15<p>Metric = euclidean<p>Weights =  weights_sqr4<p></font></td>"\
                "<td style='width:25%;'><font color=""red""><p><b>Meilleur f1-score de toutes nos simulations</b><p>La metric manhattan sous-performe</td></tr></font>"\
            "<tr><td style='width:25%;'><p>400 centroïdes sec<p>600 centroïdes pluie</td>"\
                "<td style='width:25%;'><p>F1-Score : 0.634<p>Accuracy : 0.816<p>Recall : 0.726<p>Precision : 0.563<p>Specificity : 0.842</td>"\
                "<td style='width:25%;'><p>k = 6<p>Metric = euclidean<p>Weights = uniform</td>"\
                "<td style='width:25%;'></td></tr>"\
            "<tr><td style='width:25%;'><p>200 centroïdes sec<p>800 centroïdes pluie</td>"\
                "<td style='width:25%;'><p>F1-Score : 0.606<p>Accuracy : 0.800<p>Recall : 0.702<p>Precision : 0.534<p>Specificity : 0.827</td>"\
                "<td style='width:25%;'><p>k = 2<p>Metric = cosine<p>Weights = uniform</td>"\
                "<td style='width:25%;'></td></tr>"\
            "<tr><td style='width:25%;'><p>600 centroïdes sec<p>400 centroïdes pluie</td>"\
                "<td style='width:25%;'><p>F1-Score : 0.660<p>Accuracy : 0.855<p>Recall : 0.642<p>Precision : 0.679<p>Specificity : 0.914</td>"\
                "<td style='width:25%;'><p>k = 14<p>Metric = cosine<p>Weights = weights_sqr</td>"\
                "<td style='width:25%;'><p><b>cosine :</b> il existe des directions privilégiées pour les clusters"\
                    "<p>Equilibrage du jeu de données : un léger rééquilibrage permet d’améliorer les résultats</td></tr>"\
            "<tr><td style='width:25%;'><p>50 centroïdes sec<p>50 centroïdes pluie</td>"\
                "<td style='width:25%;'><p>F1-Score : 0.631<p>Accuracy : 0.807<p>Recall : 0.750<p>Precision : 0.545<p>Specificity : 0.824</td>"\
                "<td style='width:25%;'><p>k = 8<p>Metric = euclidean<p>Weights = weights_sqr4</td>"\
                "<td style='width:25%;'></td></tr>"\
            "<tr><td style='width:25%;'><p>15 centroïdes sec<p>15 centroïdes pluie</td>"\
                "<td style='width:25%;'><p>F1-Score : 0.620<p>Accuracy : 0.789<p>Recall : 0.727<p>Precision : 0.513<p>Specificity : 0.806</td>"\
                "<td style='width:25%;'><p>k = 7<p>Metric = Manhattan<p>Weights = weights_sqr</td>"\
                "<td style='width:25%;'><p>* : Voir ci-dessous</td></tr>"\
            "<tr><td style='width:25%;'><p>100 centroïdes sec<p>100 centroïdes pluie</td>"\
                "<td style='width:25%;'><p>F1-Score : 0.649<p>Accuracy : 0.830<p>Recall : 0.714<p>Precision : 0.594<p>Specificity : 0.863</td>"\
                "<td style='width:25%;'><p>k = 6<p>Metric = euclidean<p>Weights = weights_sqr</td>"\
                "<td style='width:25%;'></td></tr>"\
            "</table>"
            st.markdown(KmeansResults,unsafe_allow_html = True)
            st.write("")
            st.write("Pour le calcul “k=13--metric=chebyshev--weights=uniform'”, on se retrouve avec une matrice de confusion totalement déformée:")
            st.write("")
            st.image(loadImage(chemin+"chebyshevDeforme.jpg",150))
            st.write("")


    with st.expander("Conclusions sur l’utilisation du KNeighborsClassifier"):
        st.write("")
        st.write("")
        st.write("Nos diverses tentatives pour améliorer les résultats de KNeighborsClassifier ne nous ont pas permis de nous détacher"\
                 " significativement des résultats obtenus avec les métriques proposées en standard et avec la pondération proposée"\
                     " en standard (paramètre “weights”). Des gains en performance ont toutefois pu être réalisés sans pour autant"\
                         " dépasser les quelques % d’amélioration sur le f1-score.")
        st.write("")
        st.write("Une transformation des données via PCA ou encore la pondération entre les variables n’a pas permis d’améliorer"\
                 " significativement les performances. Sur ces jeux de données, la distance Manhattan est la plus performante.")
        st.write("")
        st.write("L’utilisation d’une fonction de pondération personnalisée (weight) a permis de gagner un peu sur le f1-score.")
        st.write("")
        st.write("La métrique personnalisée (distance) doit hélas être exclue en raison de l’augmentation des temps de calcul sur le dataset initial.")
        st.write("")
        st.write("Enfin, l’utilisation de KMeans a permis de **très fortement diminuer les durées de traitement**, tout en maintenant"\
                 " sur certains jeux de paramètres de bons scores (meilleur f1-score). La distance de Manhattan sous-performe lorsque l’on a affaire"\
                     " à des centroïdes en nombre limité. On retrouve alors les metric “cosine” et “euclidean”.")
        st.write("")
        st.write("KNeighborsClassifier permet d’obtenir des résultats intéressants mais son amélioration se heurte"\
                 " à son absence de lien avec la réalité physique des variables manipulées. La seule voie d’amélioration"\
                     " restante sera comme à chaque fois de construire de nouvelles variables portant un supplément d’information sur la cible.")
        st.write("")
        st.write("Avec KNeighborsClassifier, il apparaît **obligatoire** de passer en revue et de **simuler de manière agnostique"\
                 " de nombreuses combinaisons de paramètres.**")
        st.write("")
        st.write("**:red[... C’est ce que nous avons fait...]**")

            






    
        













    