import streamlit as st
import requests
import os

def call_gpt2_api(text):
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    response = requests.post(api_url, headers=headers, json={"inputs": text})
    return response.json()

# Use this function in the same way as the previous call_hugging_face_api


# Streamlit app
st.title("Hugging Face Model Integration")

# Input box for text
input_text = st.text_area("Enter text to generate next tokens:")

if st.button("Generate"):
    with st.spinner("Generating..."):
        result = call_hugging_face_api(input_text)

        # Display results
        if 'generated_text' in result:
            st.write("Generated Text:")
            st.write(result['generated_text'])
        else:
            st.error("Error: " + str(result))
