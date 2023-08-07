# seriesTemporelles

import streamlit as st
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def seriesTemporelles():
    displayBackground("#000000","#FFFFFF")
    st.title("Séries Temporelles")