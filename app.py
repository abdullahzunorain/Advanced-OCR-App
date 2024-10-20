import streamlit as st
import requests
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

st.title("Image to Text App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Optionally, process the image using EasyOCR
    img = uploaded_file.read()
    results = reader.readtext(img)

    extracted_text = "\n".join([result[1] for result in results])
    st.text_area("Extracted Text", value=extracted_text, height=300)

    # Or send the image to Groq API (pseudo-code)
    # response = requests.post("https://api.groq.com/ocr", files={"file": uploaded_file})
    # extracted_text = response.json().get("text", "")
