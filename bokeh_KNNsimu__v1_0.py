# bokeh_KNNsimu__v1_0

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from resultatsKNNBokeh import resultatsKNNBokeh


from resizeImage import resizeImage, loadImage

def main():
    
    displayBackground("#000000","#FFFFFF")

    def bokeh_KNNsimu__v1_0():
        st.write("KNeighborsClassifier : Résultats des simulations en faisant varier les principaux paramètres")
        # fichier : KNN Simulations en faisant varier les principaux parametres.ipynb
        # indicateurs : _v1.0.csv ==> 'accuracy_v1.0.csv', 'precision_v1.0.csv', 'recall_v1.0.csv', 'f1score_v1.0.csv', 'specificity_v1.0.csv'
        metricsUtilisees = ['l1','l2','manhattan','nan_euclidean','minkowski','chebyshev','cityblock','cosine','euclidean']
        nbVoisinsMax = 51
        poids = ['uniform','distance']
        couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red']
        resultatsKNNBokeh("accuracy_v1.0.csv", "precision_v1.0.csv", "recall_v1.0.csv", "f1score_v1.0.csv", "specificity_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
        st.write("")

    bokeh_KNNsimu__v1_0()
    

if __name__ == "__main__":
    main()


