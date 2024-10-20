import streamlit as st
import easyocr
from PIL import Image
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'])  # You can specify other languages if needed

# Streamlit App Interface
st.title("Real-Time Image Text Extraction")

# File uploader for images
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to numpy array for EasyOCR
    image_np = np.array(image)

    # Perform OCR and display results
    with st.spinner("Extracting text..."):
        results = reader.readtext(image_np)

    # Display the detected text
    for bbox, text, confidence in results:
        st.write(f"Detected Text: {text} (Confidence: {confidence:.2f})")
