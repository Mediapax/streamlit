# displayBackground

import streamlit as st

def displayBackground(bgHex,textHex):
    style = "<style>body {{background-color: colorHex;color: textHex;}}</style"
    st.markdown(style, unsafe_allow_html=True)
