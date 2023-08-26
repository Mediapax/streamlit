# RealTimeModelingKNN

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
import pandas as pd
import numpy as np

def RealTimeModelingKNN():
    st.header("KNeighborsClassifier")
    st.write("Nos meilleurs simulations nous ont permis d'obtenir un F1-score de 0.662 et de maintenir une Accuracy de 0.87")
    st.write("Pouvez-vous faire mieux ?")

    st.divider()

    #--------Lecture des données-------#
    df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', index_col=0)
    df = df.drop(['Date','Location','Latitude','Longitude'],axis=1)
        
    #--------Sélection des variables avec affichage sur 4 colonnes-------#

    st.subheader("Choix des variables à utiliser:")
    features = df.drop("RainTomorrow",axis=1)
    labels = features.columns
    cols = st.columns(4)
    i=0
    col = {}
    
    for c in labels:
        with cols[i%4]:
            col[c] = st.checkbox(label = c, value = False)
            i += 1


    st.write("A l'issue de votre choix, seules les lignes sans 'Nans' seront retenues pour la modélisation")
    #st.write("Vous avez donc la possibilité de choisir des variables avec beaucoup de Nans")

    st.divider()
    st.subheader("Choix d'une normalisation des données :")
    norm_selection = ["Pas de normalisation", "StandardScaler", "MinMaxScaler"]
    st.write("Choix de la normalisation des données")
    norm = st.radio(label = "", options = norm_selection, horizontal = True)

    st.divider()
    st.subheader("Choix d'une pondération à appliquer")
    st.write("Attention ! Le calcul de l'Information Mutuelle prend le plus de temps.")
    pond_selection = ["Pas de pondération", "Pondération : Information Mutuelle", "Pondération : variance expliquée"]
    st.write("Choix d'une pondération des données")
    pond = st.radio(label = "", options = pond_selection, horizontal = True)

    st.divider()
    st.subheader("Choix d'une simplification des données")

    simplification= ["Pas de simplification", "PCA:95%", "PCA:90%", "PCA:85%", "PCA avec toutes les composantes", "Kmeans: 100s,100p",
                      "Kmeans: 80s,120p", "Kmeans: 60s, 140p", "Kmeans: 120s,80p",
                      "Kmeans: 140s, 60p"]
    simpl = st.radio(label = "", options = simplification, horizontal = False)

    st.divider()
    st.write("Choix des paramètres de la simulation:")
    st.write("")

    # n_neighbors = 2, weights = "uniform", algorithm = "auto", leaf_size = 30, p = 2, 
    #                        metric = "minkowski", metric_params = None, n_jobs = None
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nb_voisins = st.slider(label = "n_neighbors", value=3, min_value=1, max_value=30)

    with col2:
        weights = ["uniform","distance","weights_sqr", "weights_sqr4", "weights_sqr8"]
        weights_choisi = st.selectbox(label="weights",options=weights)

    with col3:
        metric = ["manhattan","minkowski","chebyshev","cosine","euclidean"]
        metric_choisie = st.selectbox(label="metric",options=metric)

    with col4:
        pui = st.slider(label = "puissance (minkowsky)", value=2, min_value=1, max_value=7)


    st.divider()
    if st.button("**Lancer la simulation et afficher les résultats**"):
        from datetime import datetime

        st.write("**Dataframe d'origine:**")
        st.write(df.describe())
        st.write("")

        col_choisies = []
        for c in labels:
            if col[c] == True :
                col_choisies.append(c)
        st.write("**Liste des colonnes sélectionnées:**")
        st.write(col_choisies)
        df_final = features[col_choisies]
        df_final["RainTomorrow"] = df["RainTomorrow"]
        st.write("**Dataframe final utilisé pour la simulation**")
        df_final.dropna(inplace=True)
        st.write("Une fois retirées les lignes avec des Nans, il reste alors :",len(df_final), " lignes.")
        st.write(df_final.describe())
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df_final.drop("RainTomorrow",axis=1), df_final["RainTomorrow"], test_size = .2, random_state = 42, shuffle = True)
        st.write("")


        st.write("**Paramètres de la simulation :**")
        # Application de la normalisation sélectionnée ["Pas de normalisation", "StandardScaler", "MinMaxScaler"]
        if norm == "Pas de normalisation":
            st.write("-1) Pas de normalisation des données avant simulation")
            X_train_norm = X_train
            X_test_norm = X_test

        if norm == "StandardScaler":
            st.write("-1) Normalisation : StandardScaler()")
            from sklearn.preprocessing import StandardScaler
            stscaler = StandardScaler()
            X_train_norm = stscaler.fit_transform(X_train)
            X_test_norm = stscaler.transform(X_test)

        if norm == "MinMaxScaler":
            st.write("-1) Normalisation : MinMaxScaler()")
            from sklearn.preprocessing import MinMaxScaler
            minmax = MinMaxScaler()
            X_train_norm = minmax.fit_transform(X_train,y_train)
            X_test_norm = minmax.transform(X_test)

        # Application de la pondération sélectionnée ["Pas de pondération", "Pondération : Information Mutuelle", "Pondération : variance expliquée"]
        if pond == "Pas de pondération":
            st.write("-2) Pas de pondération des données avant simulation")
            X_train_pond = X_train_norm
            X_test_pond = X_test_norm
        if pond == "Pondération : Information Mutuelle":
            hdebut_pond = datetime.now()
            st.write("-2) Pondération : Information Mutuelle")
            with st.spinner("Pondération en cours..."):
                from sklearn.feature_selection import mutual_info_regression
                # Permet de déterminer sur une dizaine de tirages au sort
                # la moyenne de l'information mutuelle
                info_mutuelle = []
                for i in range(10):
                    info_mutuelle.append(mutual_info_regression(X_train_norm, y_train))
                info_mutuelle_moy = np.mean(info_mutuelle,axis=0)
                X_train_pond = X_train_norm*info_mutuelle_moy
                X_test_pond = X_test_norm*info_mutuelle_moy
            hfin_pond = datetime.now()
            st.write("Durée du calcul de pondération = ",hfin_pond - hdebut_pond)

        if pond == "Pondération : variance expliquée":
            hdebut_pond = datetime.now()
            st.write("-2) Pondération : variance expliquée")
            with st.spinner("Pondération en cours..."):
                from sklearn.decomposition import PCA
                nb_dim = len(df_final.columns)-1
                st.write("nb composantes = ", str(nb_dim))
                pca_pond = PCA(n_components = nb_dim)
                dn = pca_pond.fit(X_train_norm)
                X_train_pond = X_train_norm*dn.explained_variance_
                X_test_pond = X_test_norm*dn.explained_variance_
            hfin_pond = datetime.now()
            st.write("Durée du calcul de pondération = ",hfin_pond - hdebut_pond)
        


        # application de la 'simplification' sélectionnée
        ''' ["Pas de simplification", "PCA:95%", "PCA:90%", "PCA:85%", "PCA avec toutes les composantes", "Kmeans: 100s,100p",
                      "Kmeans: 80s,120p", "Kmeans: 60s, 140p", "Kmeans: 120s,80p",
                      "Kmeans: 140s, 60p"]
        '''

        def calcul_Kmeans(sec,pluie,X_train_pond,y_train):
            with st.spinner("Pondération Kmeans en cours..."):
                # jeux de données d'entrainement
                df_Kmeans = pd.DataFrame(X_train_pond,y_train.index)
                df_Kmeans["RainTomorrow"] = y_train
                df_pluie = df_Kmeans[df_Kmeans["RainTomorrow"]==1]
                df_sec = df_Kmeans[df_Kmeans["RainTomorrow"]==0]
                X_pluie = df_pluie.drop("RainTomorrow",axis=1)
                X_sec = df_sec.drop("RainTomorrow",axis=1)

                # utilisation de Kmeans pour trouver des centroïdes
                from sklearn.cluster import KMeans
                # création de n centroïdes à partir des échantillons secs
                kmeans_sec = KMeans(n_clusters = sec)
                kmeans_sec.fit(X_sec)
                # création de n centroïdes à partir des échantillons de pluie
                kmeans_pluie = KMeans(n_clusters = pluie)
                kmeans_pluie.fit(X_pluie)
                # Création du jeux d'entrainement à partir des centroïdes
                centroids_sec = kmeans_sec.cluster_centers_
                centroids_pluie = kmeans_pluie.cluster_centers_
                X_train_simpl = np.append(centroids_sec,centroids_pluie,axis=0)
                X_test_simpl = X_test_pond
                y_train = []
                for i in np.arange(0,sec):
                    y_train.append(0)
                for i in np.arange(0,pluie):
                    y_train.append(1)
                return X_train_simpl, X_test_simpl, y_train
            
        def calcul_PCA(taux):
            with st.spinner("Pondération PCA en cours..."):
                from sklearn.decomposition import PCA
                pca95 = PCA(n_components  = taux)
                d_optimal = pca95.fit(X_train_pond)
                X_train_simpl = pca95.transform(X_train_pond)
                X_test_simpl = pca95.transform(X_test_pond)
                return X_train_simpl, X_test_simpl

        if simpl == "Pas de simplification":
            st.write("-3) Pas de simplification")
            X_train_simpl = X_train_pond
            X_test_simpl = X_test_pond
        else :
            hdebut_pond = datetime.now()
            if simpl == "PCA:95%":
                st.write("-3) PCA:95%")
                X_train_simpl, X_test_simpl = calcul_PCA(0.95)

            if simpl == "PCA:90%":
                st.write("-3) PCA:90%")
                X_train_simpl, X_test_simpl = calcul_PCA(0.90)
            
            if simpl == "PCA:85%":
                st.write("-3) PCA:85%")
                X_train_simpl, X_test_simpl = calcul_PCA(0.85)
            
            if simpl == "PCA avec toutes les composantes":
                st.write("-3) PCA avec toutes les composantes")
                X_train_simpl, X_test_simpl = calcul_PCA(0.99999)

            if simpl == "Kmeans: 100s,100p":
                st.write("-3) Kmeans: 100s,100p")
                X_train_simpl, X_test_simpl, y_train = calcul_Kmeans(100,100,X_train_pond,y_train)
            
            if simpl == "Kmeans: 80s,120p":
                st.write("-3) Kmeans: 80s,120p")
                X_train_simpl, X_test_simpl, y_train = calcul_Kmeans(80,120,X_train_pond,y_train)
            
            if simpl == "Kmeans: 60s, 140p":
                st.write("-3) Kmeans: 60s, 140p")
                X_train_simpl, X_test_simpl, y_train = calcul_Kmeans(60,140,X_train_pond,y_train)
            
            if simpl == "Kmeans: 120s,80p":
                st.write("-3) Kmeans: 120s,80p")
                X_train_simpl, X_test_simpl, y_train = calcul_Kmeans(120,80,X_train_pond,y_train)
            
            if simpl == "Kmeans: 140s, 60p":
                st.write("-3) Kmeans: 140s, 60p")
                X_train_simpl, X_test_simpl, y_train = calcul_Kmeans(140,60,X_train_pond,y_train)
            hfin_pond = datetime.now()
            st.write("Durée du traitement = ",hfin_pond - hdebut_pond)

        st.write("**Paramètres du classifier KNeighborsClassifier:**")
        st.write("●	k= ",str(nb_voisins))
        st.write("●	weights = ",weights_choisi)
        st.write("●	metric= ",metric_choisie)
        if metric_choisie == "minkowski":
            st.write("●	p = ",str(pui))
        st.write("●	algorithm = 'auto', leaf_size = 30, n_jobs = None")

        def Accuracy(TP,TN,FP,FN):
            return (TP+TN)/(TP+FP+FN+TN)

        def Precision(TP,FP):
            return TP/(TP+FP)

        def Recall(TP,FN):
            return TP/(TP+FN)

        def F1score(Recall,Precision):
            return 2*(Recall * Precision) / (Recall + Precision)

        def Specificity(TN,FP):
            return TN/(TN+FP)
        
        def Accuracy(TP,TN,FP,FN):
            return (TP+TN)/(TP+FP+FN+TN)

        def Precision(TP,FP):
            return TP/(TP+FP)

        def Recall(TP,FN):
            return TP/(TP+FN)

        def F1score(Recall,Precision):
            return 2*(Recall * Precision) / (Recall + Precision)

        def Specificity(TN,FP):
            return TN/(TN+FP)
        
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
        
        # simulation :
        from sklearn.neighbors import KNeighborsClassifier
        hdebut = datetime.now()

        #initialisation du modèle modèle :            
        if weights_choisi == 'weights_sqr':
            knn = KNeighborsClassifier(n_neighbors = nb_voisins, weights = weights_sqr, algorithm = "auto", leaf_size = 30, p = pui,
                                    metric = metric_choisie, metric_params = None, n_jobs = None)
        elif weights_choisi == 'weights_sqr4':
            knn = KNeighborsClassifier(n_neighbors = nb_voisins, weights = weights_sqr4, algorithm = "auto",
                                            leaf_size = 30, p = pui, metric = metric_choisie, metric_params = None, n_jobs = None)
        elif weights_choisi == 'weights_sqr8':
            knn = KNeighborsClassifier(n_neighbors = nb_voisins, weights = weights_sqr8, algorithm = "auto",
                                            leaf_size = 30, p = pui, metric = metric_choisie, metric_params = None, n_jobs = None)
        else :
            knn = KNeighborsClassifier(n_neighbors = nb_voisins, weights = weights_choisi, algorithm = "auto",
                                            leaf_size = 30, p = pui, metric = metric_choisie, metric_params = None, n_jobs = None)

        with st.spinner("Simulation en cours..."):
            knn.fit(X_train_simpl,y_train)
            y_predict = knn.predict(X_test_simpl)

        hfin = datetime.now()
        st.write("Durée de la simulation :",hfin - hdebut)
        st.write("")

        # Matrice de confusion et les 5 principaux indicateurs
        st.write("Affichage des résultats :")
        st.write("Matrice de confusion : ")
        st.write(pd.crosstab(y_test,y_predict,rownames=['Realité\\Prédiction'],colnames=['Prédiction']))

        TP = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[1,1]
        TN = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[0,0]
        FP = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[0,1]
        FN = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[1,0]
        st.write("")
        st.write("TP=", TP)
        st.write("TN=", TN)
        st.write("FP=", FP)
        st.write("FN=", FN)
        st.write("")
        st.write("Accuracy = (TP+TN)/(TP+FP+FN+TN) ==>",Accuracy(TP,TN,FP,FN))
        st.write("Precision = TP/(TP+FP) ==>",Precision(TP,FP))
        st.write("Recall = TP/(TP+FN) ==>",Recall(TP,FN))
        st.write("F1 Score = 2*(Recall * Precision) / (Recall + Precision) ==>",F1score(Recall(TP,FN),Precision(TP,FP)))
        st.write("Specificity = TN/(TN+FP) ==>",Specificity(TN,FP))
        st.write("")
        st.write("La meilleure simulation obtenue lors du projet avec KNeighborsClassifier :")
        st.write("F1-score = 0.662, Accuracy = 0.87")