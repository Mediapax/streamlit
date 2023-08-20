import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from joblib import dump
from displayKapyPredict import displayKapyPredict


def main():

    def genModel() :
        chemin = ".\\models\\" # sous unix ==> chemin : "models/"
        st.header("Génération du modèle")
        # df_final = pd.read_csv(path, index_col=0)
        df = pd.read_csv('https://raw.githubusercontent.com/ArnoMac/weatherAUS/main/weatherAUS_Kapy_v2.csv', index_col=0)

        # Affichage du nombre de valeurs restantes avec un dropna
        df.dropna(inplace=True)
        df = df.drop(['Date','Location','Latitude','Longitude'],axis=1)

        target = df["RainTomorrow"]
        df = df[['Rainfall', 'WindGustSpeed', 'Temp9am', 'diffTempMinMax', 'diffWind3pm9am',
           'diffPressure9am3pm', 'Pressure', 'DeltaP_1d',
           'diffHimidity9am3pm', 'Humidity', 'DeltaH_1d']]
        features = df

        # séparation en jeux d'entrainement et de test
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = .2, random_state = 42, shuffle = True)

        # Normalisation des données
        # pour plus d'info, voir : https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        knnminmax1 = MinMaxScaler()
        X_train_minmax = knnminmax1.fit_transform(X_train,y_train)
        X_test_minmax = knnminmax1.transform(X_test)

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

        # utilisation des fonctions prédéfinies sur la pondération des données

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

        hdebut = datetime.now()
        st.write("heure de début KNN sur X_train_minmax: " + str(hdebut))
        # initialisation du modèle modèle :
        knn = KNeighborsClassifier(n_neighbors = 10, weights = 'distance', algorithm = "auto", leaf_size = 30, p = 2, 
                                metric = "manhattan", metric_params = None, n_jobs = None)

        knn.fit(X_train_minmax,y_train)

        y_predict = knn.predict(X_test_minmax)

        st.write(pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']))

        TP = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[1,1]
        TN = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[0,0]
        FP = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[0,1]
        FN = pd.crosstab(y_test,y_predict,rownames=['Realité'],colnames=['Prédiction']).iloc[1,0]

        st.write("")
        st.write("TP : ",TP)
        st.write("TN : ",TN)
        st.write("FP : ",FP)
        st.write("FN : ",FN)
        st.write("")
        st.write("Accuracy = (TP+TN)/(TP+FP+FN+TN) ==> ",Accuracy(TP,TN,FP,FN))
        st.write("F1 Score = 2*(Recall * Precision) / (Recall + Precision) ==> ",F1score(Recall(TP,FN),Precision(TP,FP)))
        st.write("Precision = TP/(TP+FP) ==> ",Precision(TP,FP))
        st.write("Recall = TP/(TP+FN) ==> ", Recall(TP,FN))
        st.write("Specificity = TN/(TN+FP) ==> ",Specificity(TN,FP))

        # Enregistrer le modèle et la fonction personnalisée dans un fichier
        dump(knn, chemin + 'knnmodel1.joblib')
        dump(knnminmax1, chemin + 'knnminmax1.joblib')

        hfin = datetime.now()
        st.write("heure de fin KNN sur X_train_minmax + calcul de y_predict: " + str(hfin))
        delta = hfin-hdebut
        print("durée du traitement : " + str(delta))


    st.sidebar.title("TESTS")

    menuMain = st.sidebar.radio(
        "Menu ", 
        ("génération modèle","KapyPredict")
        )

    if menuMain == "génération modèle":
        genModel()

    if menuMain == "KapyPredict":
        displayKapyPredict()

if __name__ == "__main__":
    main()


