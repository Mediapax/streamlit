# climatEtGeographie

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground
# from displayCartoBokeh import displayCartoBokeh

def climatEtGeographie():
    displayBackground("#000000","#FFFFFF")
    st.title("Climat & Géographie")
    st.write("")
    st.write("Suite aux observations réalisées dans l’exploration des données, nous avons jugé opportun d’augmenter notre jeu de données:")
    st.write("- Avec l'apport de données externes")

    st.write("")
    st.write("Pour chaque station météorologique, nous rajoutons une information concernant sa position géographique")
    st.write("-	World Cities Database pour les latitudes et longitudes des villes:")
    st.write("https://simplemaps.com/data/world-cities")
    st.write("")
    st.write("A partir de la carte des climats, Nous rajoutons une information concernant le type de climat pour chaque station météorologique")
    st.write("-	Wikipedia pour les types de climat de chaque ville en Australie:")
    st.write("https://en.wikipedia.org/wiki/Climate_of_Australia")
    st.write("")
    st.image(loadImage(".\explorationDesDonnees\CarteDesClimats.jpg",1200))
    st.write("")
    st.write("Ceci nous permet alors de disposer des informations de position géographique et de climat des différentes stations météorologiques")
    # displayCartoBokeh()
