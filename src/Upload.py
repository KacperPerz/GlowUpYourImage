import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import os
import cv2
import upscale

st.title("Send the image")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Save file as image
    input_path = f'/Volumes/XS1000/GlowUpYourImage/data/input/{uploaded_file.name}'
    with open(input_path, 'wb') as f:
        f.write(bytes_data)
    st.success("File saved as image!")

    # Process the image
    output_path = f'/Volumes/XS1000/GlowUpYourImage/data/output/{uploaded_file.name}'
    combined_image = upscale.process_image(input_path, output_path)
    combined_image =cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    
    # Display original, downscaled, and upscaled images
    st.image(combined_image, caption='Original Resized | Downscaled | Upscaled', use_column_width=True)