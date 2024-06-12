import os
import streamlit as st
import os
import streamlit as st

# Check if there are any photos in the data/input directory
if len(os.listdir('/Volumes/XS1000/GlowUpYourImage/data/output')) == 0:
    st.write('You have no photos, upload some')
else:
    # Display the photos in a grid
    for filename in os.listdir('/Volumes/XS1000/GlowUpYourImage/data/output'):
        image_path = os.path.join('/Volumes/XS1000/GlowUpYourImage/data/output', filename)
        st.image(image_path)
        
        # Create a download button for each image
        download_button = st.download_button(label="Download", data=open(image_path, 'rb'), file_name=filename, mime='image/png')
