import streamlit as st
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

st.title("Image to Text Extraction Using Hugging Face Model")

# Load the model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Upload the image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    encoding = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    # Get predicted classes
    predicted_class_ids = logits.argmax(-1).squeeze().tolist()
    tokens = encoding['input_ids'][0].tolist()
    words = processor.tokenizer.convert_ids_to_tokens(tokens)

    # Extract text from predicted classes
    extracted_text = []
    for word, predicted_class_id in zip(words, predicted_class_ids):
        if predicted_class_id != 0:  # Filter out the padding class
            extracted_text.append(word)

    st.text_area("Extracted Text", value=" ".join(extracted_text), height=300)
