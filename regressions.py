# regressions

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def regressions():
    displayBackground("#000000","#FFFFFF")
    st.header("Régressions")