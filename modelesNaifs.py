# modelesNaifs

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def modelesNaifs():

    chemin = "./techniquesML/"
    displayBackground("#000000","#FFFFFF")
    st.header("Modèles Naïfs")

    with st.expander("**Modèle 'Parfait' qui prédit la pluie**"):
        st.write("")
        st.write("Ce modèle 'parfait' aurait cette matrice de confusion :")
        st.write("")
        st.write("")
        st.image(loadImage(chemin+"ModeleParfait.jpg",1200))
        st.write("")
        st.write("Et donc les éléments suivants :")
        st.write("●	Accuracy = (TP+TN)/(TP+FP+FN+TN) = 1.0")
        st.write("●	Precision = TP/(TP+FP) = 1.0")
        st.write("●	Recall = TP/(TP+FN) = 1.0")
        st.write("●	F1-Score = 2*(Recall * Precision) / (Recall + Precision) = 1.0")
        st.write("●	Specificity = TN/(TN+FP) = 1.0")
        st.write("")
        st.write("**De manière générale, l'objectif de la modélisation est donc de faire tendre vers 1 ces quatre 'metric' :"\
                 " l'Accuracy, la Precision, le Recall( rappel), le F1-Score et la Specificity.**")
        st.write("")

    with st.expander("**Modèle 'Désertique' qui prédit qu'il ne pleut jamais**"):
        st.write("")
        st.write("Un modèle qui prédirait qu'il ne pleut jamais aurait cette matrice de confusion :")
        st.write("")
        st.image(loadImage(chemin+"ModeleDesertique.jpg",1200))
        st.write("")
        st.write("Et donc les éléments suivants :")
        st.write("●	Accuracy = (TP+TN)/(TP+FP+FN+TN) ==> 0.777")
        st.write("●	Recall = TP/(TP+FN) ==> 0.0")
        st.write("●	Specificity = TN/(TN+FP) ==> 1.0")
        st.write("●	Le F1-Score et la Précision ne peuvent être calculés ici (division par 0).")
        st.write("**Ce modèle 'désertique' a déjà une accuracy de plus de 77%**")
        st.write("")
        st.write("Dans la suite, nous chercherons de manière générale à optimiser le F1-score tout en évitant de trop diminuer l'Acuracy.")

    st.header("Normalisation des données")
    st.write("")
    st.write("Pourquoi effectuer une normalisation ?")
    st.write("")
    st.write("Prenons un exemple. Les plages de valeurs sont très grandes sur la pression et faibles sur la température."\
             " Aussi, sans renormalisation, la valeur de la pression prendra une part bien plus importante que la température"\
                 " dans les résultats de notre modélisation. Cette pondération est sans rapport direct avec"\
                     " l’importance réelle de la pression ou de la température sur la capacité à prédire la valeur de la “target”.")
    st.write("")
    st.write("**Il est donc très important de normaliser nos variables.**")
    st.write("")
    st.write("Nous choisissons d'utiliser un preprocessing de normalisation MinMax. La normalisation MinMax effectue le calcul suivant sur chaque variable :")
    st.write("X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))")
    st.write("X_scaled = X_std * (max - min) + min")
    st.write("**Les valeurs des variables sont donc ramenées entre 0 et 1.**")
    st.write("")

    

