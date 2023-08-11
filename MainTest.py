import streamlit as st

from displayProjectBokeh import displayProjectBokeh
from resizeImage import resizeImage, loadImage

def main():
    st.sidebar.image(loadImage("logo_datascientest.png",200))
    st.sidebar.title("TESTS")

    menuMain = st.sidebar.radio(
        "Menu ", 
        ("Bokeh")
        )

    if menuMain == "Bokeh":
        displayProjectBokeh()

    displayBackground("#000000","#FFFFFF")

if __name__ == "__main__":
    main()


