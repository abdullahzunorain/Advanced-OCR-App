import streamlit as st
import easyocr
from PIL import Image
import numpy as np

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add other languages if needed

# Streamlit app layout
st.title('Real-time OCR Text Extraction')

# File uploader to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to numpy array
    image_np = np.array(image)

    # Extract text from the image using EasyOCR
    with st.spinner('Extracting text...'):
        results = reader.readtext(image_np)

    # Display the extracted text
    st.subheader('Extracted Text:')
    for result in results:
        st.write(result[1])
