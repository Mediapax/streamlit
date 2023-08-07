# reechantillonage

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def reechantillonage():
    displayBackground("#000000","#FFFFFF")
    st.header("Rééchantillonages")