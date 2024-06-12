import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import os
import upscale


st.title("Send the image")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
bytes_data = None
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # Save file as image
    with open(f'/Volumes/XS1000/GlowUpYourImage/data/input/{uploaded_file.name}', 'wb') as f:
        f.write(uploaded_file.getvalue())
    st.success("File saved as image!")


# Display uploaded image
if bytes_data is not None:
    st.image(bytes_data)