# multiClasses

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def multiClasses():
    displayBackground("#000000","#FFFFFF")
    st.header("Multi-classes")