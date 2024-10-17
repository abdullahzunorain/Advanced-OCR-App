import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
# from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Preprocess the Image: Grayscale, Thresholding, and Noise Removal
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    processed_image = cv2.medianBlur(thresh, 3)
    return processed_image

# OCR with EasyOCR
def extract_text_easyocr(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)



# OCR with Hugging Face LayoutLMv3 (Free Pretrained Model)
def extract_text_huggingface(image):
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    # Convert the image to RGB format
    image = image.convert("RGB")
    
    # Prepare the image for LayoutLMv3
    encoding = processor(images=image, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Decode the predicted tokens to get text
    tokens = torch.argmax(outputs.logits, dim=-1).tolist()[0]
    
    # Convert token IDs back to words
    extracted_text = processor.tokenizer.decode(tokens)
    
    return extracted_text
    

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

    # Choose OCR engine
    ocr_choice = st.selectbox("Choose OCR engine", ["EasyOCR", "Hugging Face Donut Model"])

    # Extract text based on OCR choice
    if ocr_choice == "EasyOCR":
        extracted_text = extract_text_easyocr(processed_pil_image)
    elif ocr_choice == "Hugging Face Donut Model":
        extracted_text = extract_text_huggingface(image)

    # Display extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)
