import streamlit as st
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
import torch

# Function to load the model and processor
@st.cache_resource
def load_model():
    try:
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize the model and processor
processor, model = load_model()

# Streamlit app title
st.title("Image to Text Extraction Using LayoutLMv3")

# Check if PyTesseract is available
if not pytesseract.pytesseract.tesseract_cmd:
    st.warning("Tesseract is not installed. Please ensure it is installed in your environment.")

# Upload the image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image using LayoutLMv3
        encoding = processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**encoding)

        # Extract tokens and their corresponding scores
        logits = outputs.logits
        predicted_class_ids = logits.argmax(-1).squeeze().tolist()
        words = processor.tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().tolist())

        # Extract text and filter out special tokens
        extracted_text = ""
        for word, class_id in zip(words, predicted_class_ids):
            if class_id > 0:  # Assuming class_id 0 is for the padding token
                extracted_text += word + " "

        # Display extracted text in a text area
        st.text_area("Extracted Text", value=extracted_text.strip(), height=300)

    except Exception as e:
        st.error(f"Error processing image: {e}")

# Footer
st.write("Made with ❤️ by Abdullah Zunorain")
