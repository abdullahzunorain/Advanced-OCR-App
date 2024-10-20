import streamlit as st
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

st.title("Image to Text Extraction Using Hugging Face Donut Model")

# Load the model and processor
processor = DonutProcessor.from_pretrained("naver-clova-io/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-io/donut-base")

# Upload the image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(pixel_values)

    # Decode the generated output
    extracted_text = processor.decode(outputs[0], skip_special_tokens=True)

    st.text_area("Extracted Text", value=extracted_text, height=300)
