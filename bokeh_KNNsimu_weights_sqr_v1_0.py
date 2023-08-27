# bokeh_KNNsimu_weights_sqr_v1_0

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from resultatsKNNBokeh import resultatsKNNBokeh


from resizeImage import resizeImage, loadImage

def main():
    
    displayBackground("#000000","#FFFFFF")

    def bokeh_KNNsimu_weights_sqr_v1_0():
        st.write("KNeighborsClassifier : Résultats des simulations avec une fonction sur le paramètre 'weights' (fonction de pondération des voisins)")
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

    bokeh_KNNsimu_weights_sqr_v1_0()
    

if __name__ == "__main__":
    main()


