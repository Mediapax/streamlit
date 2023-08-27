# bokeh_KNNsimu_PCA_28c_v1_0

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
from resultatsKNNBokeh import resultatsKNNBokeh


from resizeImage import resizeImage, loadImage

def main():
    
    displayBackground("#000000","#FFFFFF")

    def bokeh_KNNsimu_PCA_28c_v1_0():
        st.write("KNeighborsClassifier : Résultats des simulations PCA sur les 28 composantes principales (sans pondération)")
        # fichier : KNN Simulations PCA sans reduc de dim.ipynb
        # indicateurs : _PCA_28c_v1.0.csv
        metricsUtilisees = ['manhattan','pond_PCA']
        nbVoisinsMax = 21
        poids = ['uniform','distance','weights_sqr','weights_sqr4','weights_sqr8']
        couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
            'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
            'goldenrod','navy','grey','darksalmon','red']
        resultatsKNNBokeh("accuracy_PCA_28c_v1.0.csv", "precision_PCA_28c_v1.0.csv", "recall_PCA_28c_v1.0.csv", "f1score_PCA_28c_v1.0.csv", "specificity_PCA_28c_v1.0.csv", metricsUtilisees, nbVoisinsMax, poids, couleurs)
        st.write("")

    bokeh_KNNsimu_PCA_28c_v1_0()
    

if __name__ == "__main__":
    main()


