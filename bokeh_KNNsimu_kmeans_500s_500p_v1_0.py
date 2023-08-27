# bokeh_KNNsimu_kmeans_500s_500p_v1_0

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from resultatsKNNBokeh import resultatsKNNBokeh


from resizeImage import resizeImage, loadImage

def main():
    
    displayBackground("#000000","#FFFFFF")

    def bokeh_KNNsimu_kmeans_500s_500p_v1_0():
        st.write("KNeighborsClassifier : Résultats des simulations KMeans 500 centroïdes sec et 500 centroïdes pluie")
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
        resultatsKNNBokeh("accuracy_kmeans_500s_500p_v1.0.csv", "precision_kmeans_500s_500p_v1.0.csv", "recall_kmeans_500s_500p_v1.0.csv", "f1score_kmeans_500s_500p_v1.0.csv", "specificity_kmeans_500s_500p_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
        st.write("")

    bokeh_KNNsimu_kmeans_500s_500p_v1_0()
    

if __name__ == "__main__":
    main()


