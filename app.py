import streamlit as st
import requests
import os

# Function to call the GPT-2 API
def call_gpt2_api(text):
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    # Send a POST request to the Hugging Face API
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    return response.json()

# Streamlit app
st.title("Hugging Face Model Integration")

# Input box for text
input_text = st.text_area("Enter text to generate next tokens:")

if st.button("Generate"):
    with st.spinner("Generating..."):
        result = call_gpt2_api(input_text)  # Call the correct function

        # Display results
        if isinstance(result, dict) and 'generated_text' in result:
            st.write("Generated Text:")
            st.write(result['generated_text'])
        else:
            st.error("Error: " + str(result))
