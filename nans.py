# nans

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from resizeImage import resizeImage, loadImage
from PIL import Image
from displayBackground import displayBackground

def nans() :
    displayBackground("#000000","#FFFFFF")
    st.title("Remplissage des NaNs")
    st.write("")
    st.write("**Pressure:**")
    st.write("Etant donné que Pressure3pm et Pressure9am sont assez corrélés, nous remplissons les NaN de Pressure3pm avec Pressure9am.")
    st.write("")
    st.write("**Humidity:**")
    st.write("Etant donné que Humidity3pm et Humidity9am sont assez corrélés, nous remplissons les NaN de Humidity3pm avec Humidity9am.")
    st.write("")
    st.write("**Temp9am:**")
    st.write("Etant donné que Temp9am et MinTemp sont assez corrélés, nous remplissons les NaN de Temp9am avec MinTemp.")