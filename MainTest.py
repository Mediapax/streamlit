import streamlit as st

from displayProjectBokeh import displayProjectBokeh
from resizeImage import resizeImage, loadImage
from displayBackground import displayBackground

def main():
    displayBackground("#000000","#FFFFFF")
    st.sidebar.image(loadImage("logo_datascientest.png",200))
    st.sidebar.title("TESTS")

    menuMain = st.sidebar.radio(
        "Menu ", 
        ("Bokeh",)
        )

    if menuMain == "Bokeh":
        displayProjectBokeh()

if __name__ == "__main__":
    main()


