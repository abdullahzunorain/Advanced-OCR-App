import streamlit as st
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# Streamlit app title
st.title("Image to Text Extraction Using LayoutLMv3")

# Load the processor and model
@st.cache_resource  # Cache the model for efficiency
def load_model():
    try:
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

processor, model = load_model()

# Upload the image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        encoding = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**encoding)

        # Extract text (implement your logic here)
        # Example: Assuming outputs[0] contains the token predictions
        extracted_text = "Extracted text based on model outputs"  # Customize this as needed
        
        # Display extracted text
        st.text_area("Extracted Text", value=extracted_text, height=300)
    
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Footer
st.write("Made with ❤️ by Abdullah Zunorain")
