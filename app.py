import streamlit as st
from transformers import ViTImageProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

st.title("Image to Text Extraction Using Hugging Face Vision Model")

# Load the model and processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = VisionEncoderDecoderModel.from_pretrained("google/vit-gpt2")

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

