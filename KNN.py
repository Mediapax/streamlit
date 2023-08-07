# KNN

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def KNN():
    displayBackground("#000000","#FFFFFF")
    st.title("KNeighborsClassifier")