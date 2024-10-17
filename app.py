import streamlit as st
import requests
import os

# Function to call the DistilGPT-2 API
def call_gpt2_api(text):
    api_url = "https://api-inference.huggingface.co/models/distilgpt2"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    # Send a POST request to the Hugging Face API
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    return response.json()

# Streamlit app
st.title("Hugging Face Model Integration")

# Add some CSS for styling
st.markdown(
    """
    <style>
    .stTextArea {
        font-size: 20px;
        padding: 10px;
    }
    .stButton {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True
)

# Input box for text
input_text = st.text_area("Enter text to generate next tokens:", height=150)

if st.button("Generate"):
    with st.spinner("Generating..."):
        result = call_gpt2_api(input_text)

        # Display results
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            generated_text = result[0]['generated_text']  # Extract the generated text
            st.write("### Generated Text:")
            st.write(generated_text)  # Display the extracted text
        else:
            st.error("Error: " + str(result))
