import streamlit as st
import pandas as pd
from io import StringIO

st.title("Send the image")
uploaded_file = st.file_uploader("Choose a file")
bytes_data = None
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Save file as image
    with open(f'/Users/admin/Desktop/studia/num_project/proj/GlowUpYourImage/data/input/{uploaded_file.name}', 'wb') as f:
        f.write(uploaded_file.getvalue())
    st.success("File saved as image!")


# Display uploaded image
if bytes_data is not None:
    st.image(bytes_data)