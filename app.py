import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr

# Advanced Preprocessing: Grayscale, Adaptive Thresholding, Noise Removal, Denoising
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding to improve contrast
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Noise removal using a median filter
    denoised_image = cv2.medianBlur(adaptive_thresh, 3)

    # Optionally, apply further denoising if the text is not sharp
    denoised_image = cv2.fastNlMeansDenoising(denoised_image, None, 30, 7, 21)

    return denoised_image

# OCR with EasyOCR
def extract_text_easyocr(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)

# Streamlit App
st.title("Advanced OCR Text Extraction App")

# Image uploader
uploaded_file = st.file_uploader("Upload an image (jpg, png, bmp)", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)
    processed_pil_image = Image.fromarray(processed_image)

    st.image(processed_pil_image, caption="Processed Image", use_column_width=True)

    # Extract text using EasyOCR
    extracted_text = extract_text_easyocr(processed_pil_image)

    # Display extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)
